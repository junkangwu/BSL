'''
Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
from ast import arg
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
from random import sample
class mf_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(mf_frame, self).__init__()

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
        self.m = args_config.m
        self.beta = args_config.beta
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.group_num = args_config.group_num
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.mix_replace = args_config.mix_replace
        self.n_negs = args_config.n_negs
        if args_config.trans_mode == "gd":
            self.temperature = nn.Parameter(torch.FloatTensor([args_config.temperature]))
        elif args_config.trans_mode == "newton":
            print("NEWTON!!!")
        else:
            self.temperature = args_config.temperature
        self.eta = args_config.w1
        self.cnt_lr = args_config.cnt_lr
        self.generate_mode= args_config.generate_mode
        self.tau_min = args_config.tau_min
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.mode = args_config.pos_mode
        self.item_group_idx = item_group_idx
        self.group_mix_mode = args_config.group_mix_mode
        # define loss function
        self.loss_name = args_config.loss_fn
        if self.loss_name == "PairwiseLogisticDROLoss":
            self.loss_fn = losses.PairwiseLogisticDROLoss()
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            self.loss_fn = losses.PairwiseLogisticEasyLoss()
        elif self.loss_name == "PairwiseLogisticDRO2Loss":
            self.loss_fn = losses.PairwiseLogisticDRO2Loss()
        elif self.loss_name == "PairwiseLogisticDRO3Loss":
            self.loss_fn = losses.PairwiseLogisticDRO3Loss()
        elif self.loss_name == "BCEDROLoss":
            self.loss_fn = losses.BCEDROLoss()
        elif self.loss_name == "BCEDRO2Loss":
            self.loss_fn = losses.BCEDRO2Loss()
        elif self.loss_name == "InforNCEPosDROLoss":
            self.loss_fn = losses.InforNCEPosDROLoss(self.temperature_1, self.temperature_2, self.temperature_3)
        elif self.loss_name == "Pos_DROLoss":
            self.loss_fn = losses.Pos_DROLoss(self.temperature_1, self.temperature_2, self.temperature_3, args_config.loss_re, self.mode, self.device)
        elif self.loss_name == "SSMLoss":
            self.loss_fn = losses.SSMLoss(self.temperature_1, self.temperature_2, self.temperature_3, args_config.loss_re, self.mode, self.device)
        elif self.loss_name == "Pos_Reweight":
            self.loss_fn = losses.Pos_Reweight(self.temperature_1, self.temperature_2, self.temperature_3, args_config.loss_re, self.mode, self.device)
        elif self.loss_name == "DRO_reweightLoss":
            self.loss_fn = losses.DRO_reweightLoss(self.temperature_1, self.temperature_2, self.temperature_3)
        elif self.loss_name == "DCL_Loss":
            self.loss_fn = losses.DCL_Loss(self.temperature, self.temperature_2, self.temperature_3)
        elif self.loss_name == "DORO_Loss":
            self.loss_fn = losses.DORO_Loss(self.temperature, self.temperature_2)
        elif self.loss_name == "BCELoss":
            self.loss_fn = losses.BCELoss()
        elif self.loss_name == "BPRLoss":
            self.loss_fn = losses.BPRLoss()
        elif self.loss_name == "MSELoss":
            self.loss_fn = losses.MSELoss()
        elif self.loss_name == "R_CELoss":
            self.loss_fn = losses.R_CELoss()
        elif self.loss_name == "T_CELoss":
            self.loss_fn = losses.T_CELoss()
        elif self.loss_name == "CosineContrastiveLoss":
            self.loss_fn = losses.CosineContrastiveLoss(self.temperature, self.temperature_2)
        else:
            raise NotImplementedError
        if train_cf_len is not None:
            pos_label_1 = torch.ones(args_config.batch_size).to(self.device) # [B]
            neg_label_1 = torch.zeros(args_config.batch_size, args_config.n_negs).to(self.device) # [B M]
            self.label_1 = torch.cat([pos_label_1.unsqueeze(1), neg_label_1], dim=1)
            pos_label_2 = torch.ones(train_cf_len % args_config.batch_size).to(self.device) # [B]
            neg_label_2 = torch.zeros(train_cf_len % args_config.batch_size, args_config.n_negs).to(self.device) # [B M]
            self.label_2 = torch.cat([pos_label_2.unsqueeze(1), neg_label_2], dim=1)
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

    def forward(self, batch=None, cluster_result=None, step=0):
        user = batch['users']
        pos_item = batch['pos_items']
        if cluster_result is None:
            neg_item = batch['neg_items']
            return self.Uniform_loss(self.user_embed[user], self.item_embed[pos_item], self.item_embed[neg_item], user, step)
        else:
            item2cluster = cluster_result['item2cluster']
            # prototypes = cluster_result['centroids']
            item_emb = self.item_embed
            item_emb_scatter = scatter(item_emb, item2cluster, dim=0, reduce='mean')

            pos_proto_id = item2cluster[pos_item]
            pos_prototypes = item_emb_scatter[pos_proto_id]
            # negative prototypes
            if self.n_negs >= self.group_num:
                neg_prototypes = item_emb_scatter
                return self.Uniform_Group_loss(self.user_embed[user], pos_prototypes, neg_prototypes, user)
            else:
                neg_item = batch['neg_items']
                neg_prototypes = item_emb_scatter[neg_item]
                return self.Uniform_loss(self.user_embed[user], pos_prototypes, neg_prototypes, user)

    def forward_group(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        if self.group_mix_mode == "v1":
            item_emb = self.item_embed
            item_emb = F.normalize(item_emb, p=2, dim=-1)
            neg_item = scatter(item_emb, self.item_group_idx, dim=0, reduce='mean')
        elif self.group_mix_mode == "v2":
            item_emb = self.item_embed
            neg_item = scatter(item_emb, self.item_group_idx, dim=0, reduce='mean')
        return self.Uniform_Group_loss(self.user_embed[user], self.item_embed[pos_item], neg_item, user)

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
    
    def rating2(self, u_g_embeddings=None, i_g_embddings=None):
        norm_user = torch.sum(u_g_embeddings * u_g_embeddings, dim=-1)
        norm_item = torch.sum(i_g_embddings * i_g_embddings, dim=-1)
        inner_user_item = torch.einsum("ik,jk->ij", u_g_embeddings, i_g_embddings)
        scores = - (norm_user.unsqueeze(1) + norm_item.unsqueeze(0) - 2 * inner_user_item)
        return scores
    
    def Uniform_Group_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user):
        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb # [B, M, F]

        # item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            pos_e = F.normalize(pos_e, dim=-1)
            neg_e = F.normalize(neg_e, dim=-1)
        pos_pred = torch.mul(u_e, pos_e).sum(dim=-1, keepdim=True)
        neg_pred = torch.mm(u_e, neg_e.T)
        y_pred = torch.cat([pos_pred, neg_pred], dim=1)
        # y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        
        if self.loss_name == "PairwiseLogisticDROLoss" or self.loss_name == "PairwiseLogisticDRO2Loss" or self.loss_name == "PairwiseLogisticDRO3Loss" \
            or self.loss_name == "BCEDROLoss" or self.loss_name == "BCEDRO2Loss":
            loss  = self.loss_fn(y_pred, self.temperature.detach(), self.eta)
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            loss = self.loss_fn(y_pred, self.temperature)
        elif self.loss_name == "Pos_DROLoss":
            loss = self.loss_fn(y_pred, user)
        else:
            loss = self.loss_fn(y_pred)
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        
        return loss + emb_loss, emb_loss
    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, step):
        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb # [B, M, F]
        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        # uniform
        # uniform_item = item_e[0, 1:] # [M F]
        # uniform_score = torch.pdist(uniform_item, p=2).pow(2)
        # uniform_score = uniform_score.mul(-2).exp().mean().log()

        # ligh_score = (u_e - item_e[:, 0]).norm(dim=1).pow(2).mean()

        if self.loss_name == "PairwiseLogisticDROLoss" or self.loss_name == "PairwiseLogisticDRO2Loss" or self.loss_name == "PairwiseLogisticDRO3Loss" \
            or self.loss_name == "BCEDROLoss" or self.loss_name == "BCEDRO2Loss":
            loss  = self.loss_fn(y_pred, self.temperature.detach(), self.eta)
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            loss = self.loss_fn(y_pred, self.temperature)
        elif self.loss_name == "Pos_DROLoss" or self.loss_name == "SSMLoss":
            loss, neg_rate = self.loss_fn(y_pred, user)
        elif self.loss_name == "Pos_Reweight":
            loss, neg_rate = self.loss_fn(y_pred, user, self.temperature_1, self.temperature_2, self.temperature_3, self.mode)
        elif self.loss_name == "MSELoss" or self.loss_name == "BPRLoss":
            loss = self.loss_fn(y_pred)
        elif self.loss_name == "BCELoss":
            if step == 1:
                loss = self.loss_fn(y_pred, self.label_2)
            else:
                loss = self.loss_fn(y_pred, self.label_1)
        elif self.loss_name == "R_CELoss":
            if step == 1:
                loss = self.loss_fn(y_pred, self.label_2, self.temperature)
            else:
                loss = self.loss_fn(y_pred, self.label_1, self.temperature)
        elif self.loss_name == "T_CELoss":
            if step == 1:
                loss = self.loss_fn(y_pred, self.label_2, self.temperature)
            else:
                loss = self.loss_fn(y_pred, self.label_1, self.temperature)
        elif self.loss_name == "CosineContrastiveLoss":
            loss = self.loss_fn(y_pred)
        else:
            loss, neg_rate = self.loss_fn(y_pred)
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        # pos_neg_score
        # pos_score = y_pred[:, 0].view(-1, 1).half().cpu().detach().numpy()
        # neg_score = y_pred[:, 1:].mean(dim=-1, keepdim=True).half().cpu().detach().numpy()
        # gap_score = torch.mean(torch.exp(y_pred[:, 1:] / self.temperature_1 - y_pred[:, 0:1] / self.temperature_1), dim=-1, keepdim=True).half().cpu().detach().numpy()
        # scores = np.concatenate((pos_score, gap_score, neg_score), axis=1)
        return loss + emb_loss, emb_loss, y_pred

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
