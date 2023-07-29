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

from zmq import device
sys.path.append("..")
from utils import losses
from scipy.special import lambertw
from torch_scatter import scatter
from random import sample

class Ultra_gcn(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, Ultra_matrix, logger=None):
        super(Ultra_gcn, self).__init__()

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
        # self.user_embed = nn.Parameter(self.user_embed)
        # self.item_embed = nn.Parameter(self.item_embed)
        self.mode = args_config.pos_mode
        # self.item_group_idx = item_group_idx
        self.group_mix_mode = args_config.group_mix_mode
        # define loss function
        self.loss_name = args_config.loss_fn
        self.w1 = args_config.w1
        self.w2 = args_config.w2
        self.w3 = args_config.w3
        self.w4 = args_config.w4
        self.negative_weight = args_config.negative_weight2
        self.gamma = args_config.gamma
        self.lambda_ = args_config.lambda_

        self.user_embeds = nn.Embedding(self.n_users, self.emb_size)
        self.item_embeds = nn.Embedding(self.n_items, self.emb_size)
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        # print(self.constraint_mat['beta_uD'].size())
        # print(self.constraint_mat['beta_iD'].size())
        # if self.lambda_ > 0:
        self.ii_constraint_mat = Ultra_matrix[1].to(self.device)
        self.ii_neighbor_mat = Ultra_matrix[2].to(self.device)

        self.initial_weight = args_config.initial_weight
        self.initial_weights()
        self.bool_sigmoid = args_config.bool_sigmoid
        self.bool_normalized = args_config.bool_normalized
        self.sampling_method = args_config.sampling_method
        self.bool_omega = args_config.bool_omega
        self.lambda_mode = args_config.lambda_mode
        self.constraint_mat = Ultra_matrix[0]
        self.beta_uD = Ultra_matrix[0]['beta_uD'].to(self.device)
        self.beta_iD = Ultra_matrix[0]['beta_iD'].to(self.device)
        print("Here is 2!!! GPU version!! sampling_method is {} normalized\tsigmoid\tomega is {}\t{}\t{}".format(self.sampling_method, self.bool_normalized, self.bool_sigmoid, self.bool_omega))
        print("mode is {} generate_mode is {}".format(self.mode, self.generate_mode))

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2


    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_CL_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        if self.generate_mode == "cosine":
            user_embeds = F.normalize(user_embeds, dim=-1)
            pos_embeds = F.normalize(pos_embeds, dim=-1)
            neg_embeds = F.normalize(neg_embeds, dim=-1)
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num
        pos_scores = torch.exp( pos_scores * omega_weight[:len(pos_scores)] / self.temperature_1)
        neg_scores = torch.exp( neg_scores * omega_weight[len(pos_scores):].view(len(pos_scores), -1) / self.temperature_1).sum(dim=-1)

        loss =  - torch.log(pos_scores / neg_scores)
      
        return loss.sum()

    def cal_CL_sigmoid_in_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num
        pos_scores = torch.exp(pos_scores.sigmoid() * omega_weight[:len(pos_scores)] / self.temperature_1)
        neg_scores = torch.exp(neg_scores.sigmoid() * omega_weight[len(pos_scores):].view(len(pos_scores), -1) / self.temperature_1).sum(dim=-1)
        neg_scores = torch.pow(neg_scores, self.temperature_2)
        loss =  - torch.log(pos_scores / neg_scores)
      
        return loss.sum()

    def cal_CL_sigmoid_out_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num
        pos_scores = torch.exp(pos_scores.sigmoid() / self.temperature_1) * omega_weight[:len(pos_scores)]
        neg_scores = torch.sum(torch.exp(neg_scores.sigmoid() / self.temperature_1) * omega_weight[len(pos_scores):].view(len(pos_scores), -1), dim=-1)
        loss =  - torch.log(pos_scores / neg_scores)
      
        return loss.sum()

    def cal_CL_out_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        if self.u_norm:
            user_embeds = F.normalize(user_embeds, dim=-1)
        if self.i_norm:
            pos_embeds = F.normalize(pos_embeds, dim=-1)
            neg_embeds = F.normalize(neg_embeds, dim=-1)
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        pos_scores = torch.exp( pos_scores  / self.temperature_1) * omega_weight[:len(pos_scores)]
        neg_scores = torch.sum(torch.exp( neg_scores / self.temperature_1) * omega_weight[len(pos_scores):].view(len(pos_scores), -1), dim=-1)

        loss =  - torch.log(pos_scores / neg_scores)
      
        return loss.sum()

    def forward2(self, batch=None, cluster_result=None):
        users = batch['users']
        pos_items = batch['pos_items']
        # print("omega_weight min {} max {}".format(omega_weight.min().item(), omega_weight.max().item()))
        if self.loss_name == "bce":
            neg_items = batch['neg_items']
            omega_weight = self.get_omegas(users, pos_items, neg_items)
            loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        elif self.loss_name == "cl_in":
            neg_items = batch['neg_items']
            omega_weight = self.get_omegas(users, pos_items, neg_items)
            loss = self.cal_CL_loss_L(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        elif self.loss_name == "cl_sig_in":
            neg_items = batch['neg_items']
            omega_weight = self.get_omegas(users, pos_items, neg_items)
            loss = self.cal_CL_sigmoid_in_loss_L(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        elif self.loss_name == "cl_sig_out":
            neg_items = batch['neg_items']
            omega_weight = self.get_omegas(users, pos_items, neg_items)
            loss = self.cal_CL_sigmoid_out_loss_L(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        elif self.loss_name == "cl_out":
            neg_items = batch['neg_items']
            omega_weight = self.get_omegas(users, pos_items, neg_items)
            loss = self.cal_CL_out_loss_L(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        elif self.loss_name == "infoNCE":
            loss = self.infoNCE(users, pos_items)
            loss += self.gamma * self.norm_loss()
            loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def Sample_loss(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        u_e = user_embeds
        batch_size = user_embeds.size(0)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        item_e = torch.cat([pos_embeds.unsqueeze(1), neg_embeds], dim=1) # [B, M+1, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        if self.bool_normalized:
            u_e = F.normalize(u_e, dim=-1)
            item_e = F.normalize(item_e, dim=-1)
        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        if self.bool_sigmoid:
            y_pred = y_pred.sigmoid()
        # calculate the loss
        if self.bool_omega:
            if self.mode == "reweight_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature) * omega_weight[:batch_size]
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature) * omega_weight[batch_size:].view(batch_size, -1)
                Ng = neg_logits.sum(dim=-1)
                Ng = torch.pow(Ng, self.temperature_2)
                loss = - torch.log(pos_logits / Ng).sum()
                return loss
            elif self.mode == "multi_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature) * omega_weight[:batch_size]
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature) * omega_weight[batch_size:].view(batch_size, -1)
                user = users.contiguous().view(-1, 1)
                mask = torch.eq(user, user.T).float().to(self.device)
                pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
                neg_logits = torch.pow(torch.sum(neg_logits, dim=-1), self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                return loss
            elif self.mode == "once_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature) * omega_weight[:batch_size]
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature) * omega_weight[batch_size:].view(batch_size, -1)
                user = users
                unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
                pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
                neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
                neg_logits = torch.pow(neg_logits, self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                return loss
            pos_logits = torch.exp(y_pred[:, 0] * omega_weight[:batch_size] / self.temperature)
            neg_logits = torch.exp(y_pred[:, 1:] * omega_weight[batch_size:].view(batch_size, -1) / self.temperature)
            if self.mode == "multi":
                user = users.contiguous().view(-1, 1)
                mask = torch.eq(user, user.T).float().to(self.device)
                pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
                neg_logits = torch.pow(torch.sum(neg_logits, dim=-1), self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                return loss
            elif self.mode == "reweight":
                Ng = neg_logits.sum(dim=-1)
                Ng = torch.pow(Ng, self.temperature_2)
                loss = - torch.log(pos_logits / Ng).sum()
                return loss
            elif self.mode == "once":
                user = users
                unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
                pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
                neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
                neg_logits = torch.pow(neg_logits, self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                return loss
            else:
                raise NotImplementedError
        else:
            if self.mode == "reweight_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
                Ng = neg_logits.sum(dim=-1)
                Ng = torch.pow(Ng, self.temperature_2)
                loss = - torch.log(pos_logits / Ng).sum()
                return loss
            elif self.mode == "multi_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
                user = users.contiguous().view(-1, 1)
                mask = torch.eq(user, user.T).float().to(self.device)
                pos_logits = (pos_logits.unsqueeze(0) * mask).sum(1) / mask.sum(1)
                neg_logits = torch.pow(torch.sum(neg_logits, dim=-1), self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                 # cul regularizer
                # regularize = (torch.norm(user_embeds) ** 2
                #             + torch.norm(pos_embeds) ** 2
                #             + torch.norm(neg_embeds) ** 2) / 2  # take hop=0
                # emb_loss = self.decay * regularize / batch_size
                return loss 
            elif self.mode == "once_out":
                pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
                neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
                user = users
                unique_user, index = torch.unique(user, return_inverse=True, sorted=True)
                pos_logits = scatter(pos_logits, index, dim=0, reduce='mean')
                neg_logits = scatter(neg_logits, index, dim=0, reduce='mean').mean(dim=-1) # n_unique_user
                neg_logits = torch.pow(neg_logits, self.temperature_2)
                loss = - torch.log(pos_logits / neg_logits).sum()
                return loss
            pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
            neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        loss = - torch.log(pos_logits / neg_logits.sum(dim=-1)).sum()
      
        return loss.sum()

    def forward(self, batch=None, cluster_result=None):
        users = batch['users']
        pos_items = batch['pos_items']
        omega_weight = None
        batch_size = users.size(0)
        # print("omega_weight min {} max {}".format(omega_weight.min().item(), omega_weight.max().item()))
        if self.sampling_method.startswith("uniform"):
            neg_items = batch['neg_items']
            if self.bool_omega:
                omega_weight = self.get_omegas(users, pos_items, neg_items)
            # omega_weight = self.get_omegas(users, pos_items, neg_items)
            # print("omega_weight pos {} {} neg {} {}".format(omega_weight[:batch_size].min(), omega_weight[:batch_size].max(), omega_weight[batch_size:].min(), omega_weight[batch_size:].max()))
            loss = self.Sample_loss(users, pos_items, neg_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            if self.lambda_mode == 0:
                loss += self.lambda_ * self.cal_loss_I(users, pos_items)
            elif self.lambda_mode == 1:
                loss += self.lambda_ * self.cal_loss_I_info(users, pos_items)
            else:
                loss += self.lambda_ * self.cal_loss_I_wo_log(users, pos_items)
        else:
            if self.bool_omega:
                omega_weight = self.get_No_sample_omegas(users, pos_items)
            loss = self.No_Sample_loss(users, pos_items, omega_weight)
            loss += self.gamma * self.norm_loss()
            if self.lambda_mode == 0:
                loss += self.lambda_ * self.cal_loss_I(users, pos_items)
            elif self.lambda_mode == 1:
                loss += self.lambda_ * self.cal_loss_I_info(users, pos_items)
            else:
                loss += self.lambda_ * self.cal_loss_I_wo_log(users, pos_items)
        return loss

    def No_Sample_loss(self, users, pos_items, omega_weight):
        device = self.device
        batch_size = users.size(0)
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        if self.bool_normalized:
            user_embeds = F.normalize(user_embeds, dim=-1)
            pos_embeds = F.normalize(pos_embeds, dim=-1)
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(user_embeds, pos_embeds.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        if self.bool_sigmoid:
            y_pred = y_pred.sigmoid()
        # calculate the loss
        if self.bool_omega:
            omega_weight_neg = omega_weight[batch_size:].view(batch_size, -1)
            assert omega_weight_neg.size(0) == omega_weight_neg.size(1)
            omega_weight_neg[row_swap, col_before] = omega_weight_neg[row_swap, col_after]
            pos_logits = torch.exp(y_pred[:, 0] * omega_weight[:batch_size] / self.temperature)
            neg_logits = torch.exp(y_pred[:, 1:] * omega_weight_neg[:, 1:] / self.temperature)
        else:
            pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
            neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        loss = - torch.log(pos_logits / neg_logits.sum(dim=-1)).sum()
      
        return loss.sum()

    def cal_loss_I_info(self, users, pos_items):
        device = self.device
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items])    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items]  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        if self.bool_normalized:
            user_embeds = F.normalize(user_embeds, dim=-1)
            neighbor_embeds = F.normalize(neighbor_embeds, dim=-1)
        loss = - sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1)
    
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.device
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items])    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items]  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def cal_loss_I_wo_log(self, users, pos_items):
        device = self.device
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items])    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items]  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def get_omegas(self, users, pos_items, neg_items):
        device = self.device
        if self.w2 > 0:
            pos_weight = torch.mul(self.beta_uD[users], self.beta_iD[pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.beta_uD[users], neg_items.size(1)), self.beta_iD[neg_items.flatten()])
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def get_No_sample_omegas(self, users, pos_items):
        device = self.device
        if self.w2 > 0:
            pos_weight = torch.mul(self.beta_uD[users], self.beta_iD[pos_items])
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        neg_items = pos_items.repeat(pos_items.size(0)) # B * B
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.beta_uD[users], pos_items.size(0)), self.beta_iD[pos_items].repeat(pos_items.size(0)))
            # neg_weight = torch.mul(torch.repeat_interleave(self.beta_uD[users], pos_items.size(0)), self.beta_iD[neg_items.flatten()])
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def infoNCE(self, users, pos_items):
        device = self.device
        batch_size = users.size(0)
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        if self.generate_mode == "cosine":
            user_embeds = F.normalize(user_embeds, dim=-1)
            pos_embeds = F.normalize(pos_embeds, dim=-1)
        # contrust y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(user_embeds, pos_embeds.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        pos_logits = torch.exp(y_pred[:, 0] / self.temperature)
        neg_logits = torch.exp(y_pred[:, 1:] / self.temperature)
        loss = - torch.log(pos_logits / neg_logits.sum(dim=-1)).sum()

        return loss

    def generate(self, mode='test', split=True):
        user_gcn_emb = self.user_embeds.weight
        item_gcn_emb = self.item_embeds.weight
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
        if self.mess_dropout:
            u_e = self.dropout(u_e)
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
        if self.loss_name == "PairwiseLogisticDROLoss" or self.loss_name == "PairwiseLogisticDRO2Loss" or self.loss_name == "PairwiseLogisticDRO3Loss" \
            or self.loss_name == "BCEDROLoss" or self.loss_name == "BCEDRO2Loss":
            loss  = self.loss_fn(y_pred, self.temperature.detach(), self.eta)
        elif self.loss_name == "PairwiseLogisticEasyLoss":
            loss = self.loss_fn(y_pred, self.temperature)
        elif self.loss_name == "Pos_DROLoss":
            loss, neg_rate = self.loss_fn(y_pred, user)
        else:
            loss, neg_rate = self.loss_fn(y_pred)
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        # pos_neg_score
        # pos_score = y_pred[:, 0].view(-1, 1).half().cpu().detach().numpy()
        # neg_score = y_pred[:, 1:].mean(dim=-1, keepdim=True).half().cpu().detach().numpy()
        # gap_score = torch.mean(torch.exp(y_pred[:, 1:] / self.temperature_1 - y_pred[:, 0:1] / self.temperature_1), dim=-1, keepdim=True).half().cpu().detach().numpy()
        # scores = np.concatenate((pos_score, gap_score, neg_score), axis=1)
        return loss + emb_loss, emb_loss, neg_rate

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
