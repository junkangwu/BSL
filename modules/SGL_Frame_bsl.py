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
import ipdb
def xavier_uniform_initialization(module):
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class sgl_frame_bsl(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(sgl_frame_bsl, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
    
        self._user = item_group_idx[:, 0]
        self._item = item_group_idx[:, 1]
        self.embed_dim = args_config.dim
        self.n_layers = args_config.context_hops
        self.type = "ED"

        self.drop_ratio = args_config.w1
        self.ssl_tau = args_config.temperature
        self.temperature_2 = args_config.temperature_2
        self.temperature_3 = args_config.temperature_3
        self.mode = args_config.pos_mode
        self.reg_weight = args_config.l2
        self.ssl_weight = args_config.w2
        self.generate_mode= args_config.generate_mode
        self.sampling_method = args_config.sampling_method

        self.user_embedding = torch.nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embed_dim)
        self.reg_loss = EmbLoss()
        self.train_graph = self.csr2tensor(self.create_adjust_matrix(is_sub=False))
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        self.logger = logger
        self.loss_name = args_config.loss_fn
        if self.loss_name == "Pos_DROLoss":
            self.loss_fn = losses.Pos_DROLoss(self.temperature_2, self.temperature_3, 1.0, False, self.mode, self.device)

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.

        """
        self.sub_graph1 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph1 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph1.append(_g)

        self.sub_graph2 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph2 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph2.append(_g)

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def create_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user, dtype=np.float32)
            matrix = sp.csr_matrix((ratings, (self._user, self._item + self.n_users)),
                                   shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        else:
            if self.type == "ND":
                drop_user = self.rand_sample(self.n_users, size=int(self.n_users * self.drop_ratio), replace=False)
                drop_item = self.rand_sample(self.n_items, size=int(self.n_items * self.drop_ratio), replace=False)
                R_user = np.ones(self.n_users, dtype=np.float32)
                R_user[drop_user] = 0.
                R_item = np.ones(self.n_items, dtype=np.float32)
                R_item[drop_item] = 0.
                R_user = sp.diags(R_user)
                R_item = sp.diags(R_item)
                R_G = sp.csr_matrix((np.ones_like(self._user, dtype=np.float32), (self._user, self._item)),
                                    shape=(self.n_users, self.n_items))
                res = R_user.dot(R_G)
                res = res.dot(R_item)

                user, item = res.nonzero()
                ratings = res.data
                matrix = sp.csr_matrix((ratings, (user, item + self.n_users)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))

            elif self.type == "ED" or self.type == "RW":
                keep_item = self.rand_sample(
                    len(self._user), size=int(len(self._user) * (1 - self.drop_ratio)), replace=False
                )
                user = self._user[keep_item]
                item = self._item[keep_item]

                matrix = sp.csr_matrix((np.ones_like(user), (user, item + self.n_users)),
                                       shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)

    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)), matrix.shape
        ).to(self.device)
        return x

    def forward_embed(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            for i in range(self.n_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                all_ego.append(main_ego)
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)

        return user_emd, item_emd

    def forward(self, batch,step=0):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user_list = batch['users']
        pos_item_list = batch['pos_items']
        user_emd, item_emd = self.forward_embed(self.train_graph)
        user_sub1, item_sub1 = self.forward_embed(self.sub_graph1)
        user_sub2, item_sub2 = self.forward_embed(self.sub_graph2)
        if self.sampling_method.startswith("uniform") or self.sampling_method == "neg":
            neg_item_list = batch['neg_items']
            bpr_loss, l2_loss = self.calc_bpr_loss_(user_emd,item_emd,user_list,pos_item_list, neg_item_list)
        else:
            bpr_loss, l2_loss = self.calc_bpr_loss(user_emd,item_emd,user_list,pos_item_list)
        ssl_loss = self.calc_ssl_loss(user_list,pos_item_list,user_sub1,user_sub2,item_sub1,item_sub2)

        return bpr_loss + ssl_loss, l2_loss, 0.

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
        emb_loss = self.reg_weight * regularize / batch_size

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
        emb_loss = self.reg_weight * regularize / batch_size

        return loss + emb_loss, emb_loss

    def calc_ssl_loss(self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2,dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2,dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.forward_embed(self.train_graph)
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)

        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called.

        """
        T = super().train(mode=mode)
        if mode:
            self.logger.info("Start graph construction")
            self.graph_construction()
            self.logger.info("Finish graph construction")
        return T
