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
import faiss
# import ipdb 

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

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
        
class ncl_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, item_group_idx=None, logger=None, train_cf_len=None):
        super(ncl_frame, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        # load dataset info
        self.interaction_matrix = item_group_idx

        # load parameters info
        self.latent_dim = args_config.dim  # int type: the embedding size of the base model
        self.n_layers = args_config.context_hops         # int type: the layer num of the base model
        self.reg_weight = args_config.l2     # float32 type: the weight decay for l2 normalization

        self.ssl_temp = args_config.temperature
        self.ssl_reg = args_config.w2
        self.hyper_layers = int(args_config.w1)

        self.alpha = args_config.w3

        self.proto_reg = args_config.w4
        self.k = args_config.K

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        self.warm_up = False
        self.logger = logger

    def e_step(self):
        self.logger.info("I am coming!")
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.logger.info('start user !')
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.logger.info('start item !')
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):

        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster
        
        # print('performing kmeans clustering')
        # results = {'im2cluster':[],'centroids':[],'density':[]}
    
        # num_cluster = self.k
        # # intialize faiss clustering parameters
        # d = x.shape[1]
        # k = int(num_cluster)
        # clus = faiss.Clustering(d, k)
        # clus.verbose = True
        # clus.niter = 20
        # clus.nredo = 5
        # clus.seed = 2022
        # clus.max_points_per_centroid = 1000
        # clus.min_points_per_centroid = 10

        # res = faiss.StandardGpuResources()
        # cfg = faiss.GpuIndexFlatConfig()
        # cfg.useFloat16 = False
        # cfg.device = 0
        # index = faiss.GpuIndexFlatL2(res, d, cfg)  
        # print(111)
        # clus.train(x, index)   
        # print(222)
        # D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        # im2cluster = [int(n[0]) for n in I]
        # print(333)
        
        # # get cluster centroids
        # centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # # convert to cuda Tensors for broadcast
        # centroids = torch.Tensor(centroids).to(self.device)
        # centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        # im2cluster = torch.LongTensor(im2cluster).to(self.device)
            
        # return centroids, im2cluster

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward_embed(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]     # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]     # [B,]
        user2centroids = self.user_centroids[user2cluster]   # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def forward(self, batch, step=0):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].squeeze(1)

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward_embed()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        if self.warm_up:
            return mf_loss + self.reg_weight * reg_loss + ssl_loss + proto_loss, ssl_loss, proto_loss
        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb, embeddings_list = self.forward_embed()

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
            self.logger.info("Running E-step ! ")
            # ipdb.set_trace()
            self.e_step()
            self.logger.info("Endup E-step ! ")
        return T
