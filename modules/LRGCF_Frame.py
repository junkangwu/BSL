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
import random

class lrgcf_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(lrgcf_frame, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.emb_size = args_config.dim
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.d_i_train = torch.FloatTensor(adj_mat[2]).to(self.device).squeeze(1)
        self.d_j_train = torch.FloatTensor(adj_mat[3]).to(self.device).squeeze(0)
        # print(self.d_i_train.size())
        assert self.d_i_train.size(0) == self.n_users
        assert self.d_j_train.size(0) == self.n_items
        self.user_item_matrix, self.item_user_matrix = self._convert_sp_mat_to_sp_tensor(adj_mat)
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.hops = args_config.context_hops
        self.decay = args_config.l2

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X[0].tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        sparse_1 = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        coo = X[1].tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        sparse_2 = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return sparse_1, sparse_2

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

    def forward(self, batch,step=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].squeeze(1)
        
        users_embedding=self.user_embed
        items_embedding=self.item_embed
        
        finnal_emb_user = []
        finnal_emb_item = []

        finnal_emb_user.append(users_embedding)
        finnal_emb_item.append(items_embedding)
        for _ in range(self.hops):
            users_embedding_ = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train.unsqueeze(1)))
            items_embedding_ = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train.unsqueeze(1)))
            users_embedding = users_embedding_
            items_embedding = items_embedding_
            finnal_emb_user.append(users_embedding)
            finnal_emb_item.append(items_embedding)
        
        gcn_users_embedding = torch.cat(finnal_emb_user, dim=-1)
        gcn_items_embedding = torch.cat(finnal_emb_item, dim=-1)
        
        user_cnt_emb = gcn_users_embedding[user]
        pos_cnt_emb = gcn_items_embedding[pos_item]
        neg_cnt_emb = gcn_items_embedding[neg_item]

        prediction_i = (user_cnt_emb * pos_cnt_emb).sum(dim=-1)
        prediction_j = (user_cnt_emb * neg_cnt_emb).sum(dim=-1) 

        l2_regulization = self.decay * (user_cnt_emb**2+pos_cnt_emb**2+neg_cnt_emb**2).sum(dim=-1).mean()
        loss = -((prediction_i - prediction_j).sigmoid().log().mean())

        return loss + l2_regulization, l2_regulization, 0.

    def generate(self, mode='test', split=True):
        users_embedding=self.user_embed
        items_embedding=self.item_embed
        
        finnal_emb_user = []
        finnal_emb_item = []

        finnal_emb_user.append(users_embedding)
        finnal_emb_item.append(items_embedding)
        for _ in range(self.hops):
            users_embedding_ = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train.unsqueeze(1)))
            items_embedding_ = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train.unsqueeze(1)))
            users_embedding = users_embedding_
            items_embedding = items_embedding_
            finnal_emb_user.append(users_embedding)
            finnal_emb_item.append(items_embedding)
        
        user_gcn_emb = torch.cat(finnal_emb_user, dim=-1)
        item_gcn_emb = torch.cat(finnal_emb_item, dim=-1)

        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

