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

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]

class lgn_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(lgn_frame, self).__init__()

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
        self.temperature = args_config.temperature
        self.temperature_1 = args_config.temperature
        self.temperature_2 = args_config.temperature_2
        self.temperature_3 = args_config.temperature_3
        self.beta = args_config.beta
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.generate_mode= args_config.generate_mode
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.mode = args_config.pos_mode
        # define loss function
        self.loss_fn = losses.Pos_DROLoss(self.temperature_1, self.temperature_2, self.temperature_3, args_config.loss_re, self.mode, self.device)
        self.loss_name = args_config.loss_fn
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        print("sample method is {}".format(self.sampling_method))
            
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, cluster_result=None, step=0):
        user = batch['users']
        pos_item = batch['pos_items']
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        if self.sampling_method.startswith("uniform") or self.sampling_method == "neg":
            neg_item = batch['neg_items']
            return self.Uniform_Sample_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item], user, step)
        return self.Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], user)

    def Uniform_Sample_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, step):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb)
        neg_e = self.pooling(neg_gcn_emb)

        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        loss, neg_rate = self.loss_fn(y_pred, user)
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return loss + emb_loss, emb_loss, neg_rate

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
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
    
    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, user):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb)
        # if self.mess_dropout:
        #     u_e = self.dropout(u_e)
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            pos_e = F.normalize(pos_e, dim=-1)
        # contrust y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(u_e, pos_e.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        neg_rate = None
        loss, neg_rate = self.loss_fn(y_pred, user)
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size 
        return loss + emb_loss, emb_loss, neg_rate