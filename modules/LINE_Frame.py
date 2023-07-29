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
def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

class NegSamplingLoss(nn.Module):

    def __init__(self):
        super(NegSamplingLoss, self).__init__()

    def forward(self, score, sign):
        return -torch.mean(torch.sigmoid(sign * score))

class line_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(line_frame, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.embedding_size = args_config.dim
        self.order = int(args_config.pos_mode)
        self.second_order_loss_weight = args_config.lambda_

        self.interaction_feat = item_group_idx

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        if self.order == 2:
            self.user_context_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_context_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss_fct = NegSamplingLoss()

        self.used_ids = self.get_used_ids()
        self.random_list = self.get_user_id_list()
        np.random.shuffle(self.random_list)
        self.random_pr = 0
        self.random_list_length = len(self.random_list)

        self.apply(xavier_normal_initialization)

    def get_used_ids(self):
        cur = np.array([set() for _ in range(self.n_items)])
        for uid, iid in self.interaction_feat:
            cur[iid].add(uid)
        return cur

    def sampler(self, key_ids):

        key_ids = np.array(key_ids.cpu())
        key_num = len(key_ids)
        total_num = key_num
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        key_ids = np.tile(key_ids, 1)
        while len(check_list) > 0:
            value_ids[check_list] = self.random_num(len(check_list))
            check_list = np.array([
                i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                if v in used
            ])

        return torch.tensor(value_ids, device=self.device)

    def random_num(self, num):
        value_id = []
        self.random_pr %= self.random_list_length
        while True:
            if self.random_pr + num <= self.random_list_length:
                value_id.append(self.random_list[self.random_pr:self.random_pr + num])
                self.random_pr += num
                break
            else:
                value_id.append(self.random_list[self.random_pr:])
                num -= self.random_list_length - self.random_pr
                self.random_pr = 0
                np.random.shuffle(self.random_list)
        return np.concatenate(value_id)

    def get_user_id_list(self):
        return np.arange(1, self.n_users)

    def context_forward(self, h, t, field):

        if field == "uu":
            h_embedding = self.user_embedding(h)
            t_embedding = self.item_context_embedding(t)
        else:
            h_embedding = self.item_embedding(h)
            t_embedding = self.user_context_embedding(t)

        return torch.sum(h_embedding.mul(t_embedding), dim=1)

    def forward_embed(self, h, t):
        h_embedding = self.user_embedding(h)
        t_embedding = self.item_embedding(t)

        return torch.sum(h_embedding.mul(t_embedding), dim=1) 

    def forward(self, batch,step=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].squeeze(1)
        score_pos = self.forward_embed(user, pos_item)

        ones = torch.ones(len(score_pos), device=self.device)

        if self.order == 1:
            if random.random() < 0.5:
                score_neg = self.forward_embed(user, neg_item)
            else:
                neg_user = self.sampler(pos_item)
                score_neg = self.forward_embed(neg_user, pos_item)
            return self.loss_fct(ones, score_pos) + self.loss_fct(-1 * ones, score_neg), self.loss_fct(ones, score_pos) + self.loss_fct(-1 * ones, score_neg), 0.

        else:
            # randomly train i-i relation and u-u relation with u-i relation
            if random.random() < 0.5:
                score_neg = self.forward_embed(user, neg_item)
                score_pos_con = self.context_forward(user, pos_item, 'uu')
                score_neg_con = self.context_forward(user, neg_item, 'uu')
            else:
                # sample negative user for item
                neg_user = self.sampler(pos_item)
                score_neg = self.forward_embed(neg_user, pos_item)
                score_pos_con = self.context_forward(pos_item, user, 'ii')
                score_neg_con = self.context_forward(pos_item, neg_user, 'ii')
            return self.loss_fct(ones, score_pos) \
                   + self.loss_fct(-1 * ones, score_neg) \
                   + self.loss_fct(ones, score_pos_con) * self.second_order_loss_weight \
                   + self.loss_fct(-1 * ones, score_neg_con) * self.second_order_loss_weight, self.loss_fct(ones, score_pos) + self.loss_fct(-1 * ones, score_neg), 0.


    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.user_embedding.weight, self.item_embedding.weight

        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

