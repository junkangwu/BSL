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


class enmf_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(enmf_frame, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.embedding_size = args_config.dim
        self.dropout_prob = args_config.mess_dropout_rate
        self.reg_weight = args_config.l2
        self.negative_weight = args_config.lambda_

        # get all users' history interaction information.
        # matrix is padding by the maximum number of a user's interactions
        self.history_item_matrix = item_group_idx
        self.history_item_matrix = self.history_item_matrix.to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.H_i = nn.Linear(self.embedding_size, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.apply(xavier_normal_initialization)

    def reg_loss(self):
        """calculate the reg loss for embedding layers and mlp layers

        Returns:
            torch.Tensor: reg loss

        """
        l2_reg = self.user_embedding.weight.norm(2) + self.item_embedding.weight.norm(2)
        loss_l2 = self.reg_weight * l2_reg

        return loss_l2

    def forward_embed(self, user):
        user_embedding = self.user_embedding(user)  # shape:[B, embedding_size]
        user_embedding = self.dropout(user_embedding)  # shape:[B, embedding_size]

        user_inter = self.history_item_matrix[user]  # shape :[B, max_len]
        item_embedding = self.item_embedding(user_inter)  # shape: [B, max_len, embedding_size]
        score = torch.mul(user_embedding.unsqueeze(1), item_embedding)  # shape: [B, max_len, embedding_size]
        score = self.H_i(score)  # shape: [B,max_len,1]
        score = score.squeeze(-1)  # shape:[B,max_len]

        return score    

    def forward(self, batch,step=0):
        user = batch['users']
        pos_score = self.forward_embed(user)

        # shape: [embedding_size, embedding_size]
        item_sum = torch.bmm(self.item_embedding.weight.unsqueeze(2),
                             self.item_embedding.weight.unsqueeze(1)).sum(dim=0)

        # shape: [embedding_size, embedding_size]
        batch_user = self.user_embedding(user)
        user_sum = torch.bmm(batch_user.unsqueeze(2),
                             batch_user.unsqueeze(1)).sum(dim=0)

        # shape: [embedding_size, embedding_size]
        H_sum = torch.matmul(self.H_i.weight.t(), self.H_i.weight)

        t = torch.sum(item_sum * user_sum * H_sum)

        loss = self.negative_weight * t

        loss = loss + torch.sum((1 - self.negative_weight) * torch.square(pos_score) - 2 * pos_score)
        reg_loss = self.reg_loss()
        loss = loss + reg_loss

        return loss, reg_loss, reg_loss


    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.user_embedding.weight, self.item_embedding.weight

        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        score = torch.mul(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0))  # shape: [B, n_item, embedding_dim]

        score = self.H_i(score).squeeze(2)  # shape: [B, n_item]

        return score
    
   
