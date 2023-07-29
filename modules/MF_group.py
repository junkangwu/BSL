'''
Created on October 1, 2020
你更新了没呀
@author: Tinglin Huang (huangtinglin@outlook.com)
'''
from functools import reduce
from tarfile import POSIX_MAGIC
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
from utils import losses
from scipy.special import lambertw
from torch_scatter import scatter

class MF_group(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx, logger=None):
        super(MF_group, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.logger = logger
        if self.mess_dropout:
            self.dropout = nn.Dropout(args_config.mess_dropout_rate)
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.tau_plus = args_config.tau_plus
        # self.temperature = args_config.temperature
        self.temperature_1 = args_config.temperature
        self.temperature_2 = args_config.temperature_2
        self.temperature_3 = args_config.temperature_3
        self.item_group_idx = item_group_idx
        self.m = args_config.m
        self.beta = args_config.beta
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.pos_num = args_config.pos_num
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.mix_replace = args_config.mix_replace
        if args_config.trans_mode == "gd":
            self.temperature = nn.Parameter(torch.FloatTensor([args_config.temperature]))
        elif args_config.trans_mode == "newton":
            # self.temperature = torch.full((args_config.batch_size,), device=self.device)
            # self.temperature = torch.tensor([args_config.temperature], device=self.device)
            # self.temperature = self.temperature.float()
            print("NEWTON!!!")
        else:
            self.temperature = args_config.temperature
        self.eta = args_config.temperature_2
        self.cnt_lr = args_config.cnt_lr
        self.generate_mode= args_config.generate_mode
        self.tau_min = args_config.tau_min
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        
        # define loss function
        self.loss_name = args_config.loss_fn
        if self.loss_name == "PairwiseLogisticDROLoss":
            self.loss_fn = losses.PairwiseLogisticDROLoss()
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            self.loss_fn = losses.PairwiseLogisticEasyLoss()
        elif self.loss_name == "PairwiseLogisticTWOLoss":
            self.loss_fn = losses.PairwiseLogisticTWOLoss()
        elif self.loss_name == "PairwiseLogisticDRO2Loss":
            self.loss_fn = losses.PairwiseLogisticDRO2Loss()
        elif self.loss_name == "PairwiseLogisticDRO3Loss":
            self.loss_fn = losses.PairwiseLogisticDRO3Loss()
        elif self.loss_name == "BCEDROLoss":
            self.loss_fn = losses.BCEDROLoss()
        elif self.loss_name == "BCEDRO2Loss":
            self.loss_fn = losses.BCEDRO2Loss()
        elif self.loss_name == "InforNCEPosDROLoss":
            # print(self.temperature_1, self.temperature_2)
            self.loss_fn = losses.InforNCEPosDROLoss(self.temperature_1, self.temperature_2, self.temperature_3)
        else:
            raise NotImplementedError
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        # neg_item = batch['neg_items']
        # pos_item_ = batch["pos_items_"]
        item_emb = F.normalize(self.item_embed, dim=-1)
        neg_item_emb = scatter(item_emb, self.item_group_idx, dim=0, reduce='mean')
        # torch.save(neg_item_emb, "neg_item_emb.pkl")
        # print(neg_item_emb.size())
        return self.Uniform_loss(self.user_embed[user], self.item_embed[pos_item], neg_item_emb, user)

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=-2)
        elif self.pool == 'sum':
            return embeddings.sum(dim=-2)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, mode='test', split=True):
        user_gcn_emb = self.user_embed
        item_gcn_emb = self.item_embed
        if self.generate_mode == "cosine":
            if self.u_norm:
                user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            if self.i_norm:
                item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user):
        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb
        # neg_e = neg_gcn_emb.unsqueeze(0).repeat(pos_e.size(0), 1, 1) # [B, M, F]
        # pos_e_ = pos_gcn_emb_
        # print(pos_e_)
        # item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        # print(item_e.size())
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            pos_e = F.normalize(pos_e, dim=-1)
            neg_e = F.normalize(neg_e, dim=-1)
            # item_e = F.normalize(item_e, dim=-1)
        y_pred_pos = torch.mul(u_e, pos_e).sum(dim=-1)
        y_pred_neg = torch.mm(u_e, neg_e.t())
        # y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        if self.loss_name == "PairwiseLogisticDROLoss" or self.loss_name == "PairwiseLogisticDRO2Loss" or self.loss_name == "PairwiseLogisticDRO3Loss" \
            or self.loss_name == "BCEDROLoss" or self.loss_name == "BCEDRO2Loss":
            loss  = self.loss_fn(y_pred, self.temperature.detach(), self.eta)
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            loss = self.loss_fn(y_pred, self.temperature)
        elif self.loss_name == "PairwiseLogisticTWOLoss":
            loss = self.loss_fn(y_pred_pos, y_pred_neg, self.temperature)
        elif self.loss_name == "InforNCEPosDROLoss":
            loss = self.loss_fn(y_pred, self.pos_num)

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        
        return loss.mean() + emb_loss, emb_loss

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
# lalalla