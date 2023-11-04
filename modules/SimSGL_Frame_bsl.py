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

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

class SimGCL_Encoder(nn.Module):
    def __init__(self, sparse_norm_adj, emb_size, eps, n_layers, n_users, n_items):
        super(SimGCL_Encoder, self).__init__()
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.sparse_norm_adj = sparse_norm_adj
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, ego_embeddings, perturbed=False):
        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops

    def forward(self, user_embed, item_embed):
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self.interact_mat
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]

class simsgl_frame_bsl(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(simsgl_frame_bsl, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.emb_size = args_config.dim
        self.n_layers = args_config.context_hops
        self.cl_rate = args_config.w1
        self.eps = args_config.w2
        self.reg = args_config.l2
        self.generate_mode= args_config.generate_mode
        self.sampling_method = args_config.sampling_method

        self.logger = logger
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        # self.gcn = self._init_model()
        self.model_enc = SimGCL_Encoder(self.sparse_norm_adj, self.emb_size, self.eps, self.n_layers, self.n_users, self.n_items)
        self.temperature_1 = args_config.temperature
        self.temperature_2 = args_config.temperature_2
        self.temperature_3 = args_config.temperature_3
        self.mode = args_config.pos_mode
        self.loss_name = args_config.loss_fn
        if self.loss_name == "Pos_DROLoss":
            self.loss_fn = losses.Pos_DROLoss(self.temperature_2, self.temperature_3, 1.0, False, self.mode, self.device)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.n_layers,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch, step=0):
        user_list = batch['users']
        pos_item_list = batch['pos_items']

        ego_embeddings = torch.cat([self.user_embed, self.item_embed], 0)
        rec_user_emb, rec_item_emb = self.model_enc(ego_embeddings)

        # user_emb, pos_item_emb = rec_user_emb[user_list], rec_item_emb[pos_item_list]
        if self.sampling_method.startswith("uniform") or self.sampling_method == "neg":
            neg_item_list = batch['neg_items']
            rec_loss, _ = self.calc_bpr_loss_(rec_user_emb, rec_item_emb, user_list, pos_item_list, neg_item_list)
        else:    
            rec_loss, _ = self.calc_bpr_loss(rec_user_emb, rec_item_emb, user_list, pos_item_list)
        cl_loss = self.cl_rate * self.cal_cl_loss([user_list, pos_item_list])
        batch_loss = rec_loss + cl_loss
        return batch_loss, cl_loss, 0.

    def calc_bpr_loss_(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        batch_size = user_list.size(0)
        u_e = user_emd[user_list]
        pos_e = item_emd[pos_item_list]
        neg_e = item_emd[neg_item_list]
   
        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        u_e = F.normalize(u_e, dim=-1)
        item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]

        if self.loss_name == "Pos_DROLoss":
            loss, neg_rate = self.loss_fn(y_pred, user_list)
        # cul regularizer
        regularize = (torch.norm(user_emd[user_list]) ** 2
                       + torch.norm(item_emd[pos_item_list]) ** 2
                       + torch.norm(item_emd[neg_item_list]) ** 2) / 2  # take hop=0
        emb_loss = self.reg * regularize / batch_size

        return loss + emb_loss, emb_loss
    
    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list):
        batch_size = user_list.size(0)
        user_emd_ = user_emd[user_list]
        item_emd_ = item_emd[pos_item_list]
   
        u_e = F.normalize(user_emd_, dim=-1)
        pos_e = F.normalize(item_emd_, dim=-1)
        # ipdb.set_trace()
        # contrust y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(u_e, pos_e.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        
        if self.loss_name == "Pos_DROLoss":
            loss, neg_rate = self.loss_fn(y_pred, user_list)
        # cul regularizer
        regularize = (torch.norm(user_emd_[:, :]) ** 2
                       + torch.norm(item_emd_[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.reg * regularize / batch_size

        return loss + emb_loss, emb_loss

    def pooling(self, embeddings):
        return embeddings.mean(dim=-2)
    def cal_cl_loss(self, idx):
        u_idx = torch.unique(idx[0])
        i_idx = torch.unique(idx[1])
        ego_embeddings = torch.cat([self.user_embed, self.item_embed], 0)
        user_view_1, item_view_1 = self.model_enc(ego_embeddings, perturbed=True)
        user_view_2, item_view_2 = self.model_enc(ego_embeddings, perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temperature_1)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temperature_1)
        return user_cl_loss + item_cl_loss


    def generate(self, mode='test', split=True):
        # user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
        #                                       self.item_embed)
        # user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        ego_embeddings = torch.cat([self.user_embed, self.item_embed], 0)
        user_gcn_emb, item_gcn_emb = self.model_enc(ego_embeddings)
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())