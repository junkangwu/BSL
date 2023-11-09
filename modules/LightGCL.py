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
import random as rd
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.autograd import Variable
import scipy.sparse as sp

class lightgcl_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(lightgcl_frame, self).__init__()
        # self._init_weight()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_size = args_config.dim
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.emb_size)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.emb_size)))
        u_mul_s, v_mul_s, ut, vt, adj_norm = item_group_idx
        # self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = args_config.context_hops
        l = self.l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = args_config.temperature
        self.lambda_1 = args_config.w1
        self.lambda_2 = args_config.l2
        self.dropout = args_config.mess_dropout_rate
        self.act = nn.LeakyReLU(0.5)

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.logger = logger
        logger.info("l2: {} w1: {} temp: {} dropout: {}".format(self.lambda_2, self.lambda_1, self.temp, self.dropout))

    def forward(self, batch, step=0):
        uids, pos, neg = batch['users'], batch['pos_items'], batch['neg_items']
        uids = uids.long().to(self.device)
        pos = pos.long().to(self.device)
        neg = neg.long().to(self.device).squeeze(-1)
        iids = torch.concat([pos, neg], dim=0)

        for layer in range(1,self.l+1):
            # GNN propagation
            self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
            self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)
        # cl loss
        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i
        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
        loss_s = -pos_score + neg_score

        # bpr loss
        u_emb = self.E_u[uids]
        pos_emb = self.E_i[pos]
        neg_emb = self.E_i[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= self.lambda_2

        # total loss
        loss = loss_r + self.lambda_1 * loss_s + loss_reg
        #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
        return loss, loss_r, self.lambda_1 * loss_s
    
    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.E_u, self.E_i
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)