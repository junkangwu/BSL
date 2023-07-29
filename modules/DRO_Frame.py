'''
Created on October 1, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''
from ast import arg
from functools import reduce
from tarfile import POSIX_MAGIC
from tkinter.tix import Y_REGION
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
class dro_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(dro_frame, self).__init__()

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
        self.n_negs = args_config.n_negs
        # self.temperature = args_config.temperature
        self.temperature_1 = args_config.temperature
        self.temperature_2 = args_config.temperature_2
        self.temperature_3 = args_config.temperature_3
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm

        self.generate_mode= args_config.generate_mode
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.warm_up = False
        logger.info("DRO ETA FRAME: t1 {} t2 {} t3 {}".format(self.temperature_1, self.temperature_2, self.temperature_3))
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

    def Obtain_tau(self, y_pred, eta):
        neg_logits = y_pred[:, 1:]
        tau = torch.sqrt(torch.var(neg_logits, dim=-1) / (2 * eta))
        return tau.detach()

    def forward(self, batch=None, cluster_result=None, step=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_gcn_emb, pos_gcn_emb, neg_gcn_emb = self.user_embed[user], self.item_embed[pos_item], self.item_embed[neg_item]

        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb # [B, M, F]
        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        u_e = F.normalize(u_e, dim=-1)
        item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        
        if self.warm_up:
            tau = self.Obtain_tau(y_pred, self.temperature_1)
            pos_logits = torch.exp(y_pred[:, 0] / tau)
            neg_logits = torch.exp(y_pred[:, 1:] / tau.unsqueeze(1))
        else:
            tau = None
            pos_logits = torch.exp(y_pred[:, 0] / 0.10)
            neg_logits = torch.exp(y_pred[:, 1:] / 0.10)
            
        neg_logits = neg_logits.sum(dim=-1)
        neg_logits = torch.pow(neg_logits, self.temperature_2)

        loss = - torch.log(pos_logits / neg_logits).mean()

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return loss + emb_loss, emb_loss, tau

    def generate(self, mode='test', split=True):
        user_gcn_emb = self.user_embed
        item_gcn_emb = self.item_embed
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
