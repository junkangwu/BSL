from .metrics import *
from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time
# from scipy import sparse
cores = multiprocessing.cpu_count() // 2
import bottleneck as bn
args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag
import pickle
# import ipdb

def test_sp(model, user_dict, sp_mat, n_params, valid_pre, test_pre, item_group_idx=None, mode='test'):
    global n_users, n_items
    n_users = n_params['n_users']
    n_items = n_params['n_items']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']
    train_sp_mat = sp_mat['train_sp_mat']
    if mode == 'test':
        test_sp_mat = sp_mat['test_sp_mat']
        uid2swap_idx, uid2rev_swap_idx, pos_len_list = test_pre
    else:
        test_sp_mat = sp_mat['valid_sp_mat']
        uid2swap_idx, uid2rev_swap_idx, pos_len_list = valid_pre

    pos_len_list = np.array(pos_len_list)
    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    if args.gnn == 'lightgcn_byol':
        u_online, u_target, i_online, i_target = model.generate()

    else:
        user_gcn_emb, item_gcn_emb = model.generate(mode)
    Ks = eval(args.Ks)
    batch_matrix_list = []
    batch_group = []
    for u_batch_id in range(n_user_batchs):
        user_list = test_users[u_batch_id * u_batch_size: (u_batch_id + 1) * u_batch_size]
        user_batch = torch.LongTensor(np.array(user_list)).to(device)
        if args.gnn == 'lightgcn_byol':
            u_g_embeddings_online = u_online[user_batch]
            u_g_embeddings_target = u_target[user_batch]
            i_g_embddings_online = i_online
            i_g_embddings_target = i_target
            scores = model.rating(u_g_embeddings_online, i_g_embddings_target,
                                      u_g_embeddings_target, i_g_embddings_online)
        elif args.pos_mode == "cml":
            u_g_embeddings = user_gcn_emb[user_batch]
            i_g_embddings = item_gcn_emb
            scores = model.rating2(u_g_embeddings, i_g_embddings)
        else:
            if args.gnn == "Ultra_gcn":
                u_g_embeddings = user_gcn_emb[user_batch]
            else:
                u_g_embeddings = user_gcn_emb[user_batch]
            i_g_embddings = item_gcn_emb
            scores = model.rating(u_g_embeddings, i_g_embddings)
        # original test
        # rate_batch = scores.detach().cpu().numpy()
        # rate_batch[train_sp_mat[user_list].nonzero()] = -np.inf
        # ground_truth_batch = test_sp_mat[user_list]
        # for m in metrics:
        #     m['score'].append(m['metric'](rate_batch, ground_truth_batch, k=m['k']))
        # new valid
        # swap 
        swap_idx = uid2swap_idx[user_list]
        rev_swap_idx = uid2rev_swap_idx[user_list]

        swap_row = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx)])
        swap_col_after = torch.cat(list(swap_idx))
        swap_col_before = torch.cat(list(rev_swap_idx))
        # filling
        scores[train_sp_mat[user_list].nonzero()] = -np.inf
        # start to calculate the long tail evaluation
        if args.restore and args.group_valid:
            # batch_group = []
            # print(pos_len_list)
            # print(pos_len_list.shape)
            # num_test_per_user = pos_len_list[user_list]
            num_test_per_user = pos_len_list[u_batch_id * u_batch_size: (u_batch_id + 1) * u_batch_size]
            num_test_per_user_inv = np.power(num_test_per_user, -1.)
            num_test_per_user_inv[np.isinf(num_test_per_user_inv)] = 1.
            diag_num_test_per_user_inv = np.diag(num_test_per_user_inv) # B B

            batch_count_per_group = np.zeros([len(user_list), 10], dtype=np.int32)
            _, topk_idx_tail = torch.topk(scores, 20, dim=-1)
            topk_idx_tail = topk_idx_tail.cpu().numpy()
            for user in range(len(user_list)):
                for i in topk_idx_tail[user]:
                    if i in test_user_set[user_list[user]]:
                        idx_cnt_item = item_group_idx[i]
                        batch_count_per_group[user, idx_cnt_item] += 1
            batch_count_per_group = np.matmul(diag_num_test_per_user_inv, batch_count_per_group)
            batch_group.append(batch_count_per_group)

        # re-organizing
        swap_row = swap_row.to(device).long()
        swap_col_after = swap_col_after.to(device).long()
        swap_col_before = swap_col_before.to(device).long()
        scores[swap_row, swap_col_after] = scores[swap_row, swap_col_before]
        # topk-finding
        _, topk_idx = torch.topk(scores, max(Ks), dim=-1)  # B * K
        batch_matrix_list.append(topk_idx)

    topk_idx_batch = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
    # pos_len_list = np.array([test_sp_mat[i].nnz for i in range(n_test_users)])
    
    pos_idx_matrix = topk_idx_batch < pos_len_list.reshape(-1, 1)
    # matrix cal
    if args.restore and args.group_valid:
        mean_ndcg = ndcg_(pos_idx_matrix, pos_len_list)
        mean_recall = recall_(pos_idx_matrix, pos_len_list)
        mean_precision = precision_(pos_idx_matrix, pos_len_list).mean(axis=0)
        # np.save("./scores/{}_recall_per_user.npy".format(args.name), mean_recall[:, 20 - 1])
        # np.save("./scores/{}_ndcg_per_user.npy".format(args.name), mean_ndcg[:, 20 - 1])
        batch_group = np.concatenate(batch_group, axis=0)
        batch_group = batch_group.mean(axis=0)
        for i in batch_group:
            print(i)
        print("mean is {:.3}".format(batch_group.mean()))
        dict_file = {"recall_per_user": mean_recall[:, 20 - 1], 
                    "ndcg_per_user": mean_ndcg[:, 20 - 1], "batch_group":batch_group}
        mean_ndcg = mean_ndcg.mean(axis=0)
        mean_recall = mean_recall.mean(axis=0)
        with open("/data/wujk/codes/LDS7_0621/outputs/scores_fairness_/{}_dict.pkl".format(args.name), "wb") as f:
            pickle.dump(dict_file, f)
    else:
        mean_ndcg = ndcg_(pos_idx_matrix, pos_len_list).mean(axis=0)
        mean_recall = recall_(pos_idx_matrix, pos_len_list).mean(axis=0)
        mean_precision = precision_(pos_idx_matrix, pos_len_list).mean(axis=0)  
    # print("CUDA\t ndcg20:{}\trecall:{}".format(ndcg20, recall20))


    # for m in metrics:
    #     m['score'] = np.concatenate(m['score']).mean()

    return (mean_ndcg, mean_recall, mean_precision)

def test_sp2(model, user_dict, sp_mat, n_params, valid_pre, test_pre, item_group_idx=None, mode='test'):
    global n_users, n_items
    n_users = n_params['n_users']
    n_items = n_params['n_items']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']
    train_sp_mat = sp_mat['train_sp_mat']
    if mode == 'test':
        test_sp_mat = sp_mat['test_sp_mat']
        uid2swap_idx, uid2rev_swap_idx, pos_len_list = test_pre
    else:
        test_sp_mat = sp_mat['valid_sp_mat']
        uid2swap_idx, uid2rev_swap_idx, pos_len_list = valid_pre

    pos_len_list = np.array(pos_len_list)
    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    if args.gnn == 'lightgcn_byol':
        u_online, u_target, i_online, i_target = model.generate()

    else:
        user_gcn_emb, item_gcn_emb = model.generate(mode)
    Ks = eval(args.Ks)
    batch_matrix_list = []
    batch_group = []
    for u_batch_id in range(n_user_batchs):
        user_list = test_users[u_batch_id * u_batch_size: (u_batch_id + 1) * u_batch_size]
        user_batch = torch.LongTensor(np.array(user_list)).to(device)
        test_sp_index = torch.LongTensor(test_sp_mat[user_list].todense()).to(device)

        if args.gnn == 'lightgcn_byol':
            u_g_embeddings_online = u_online[user_batch]
            u_g_embeddings_target = u_target[user_batch]
            i_g_embddings_online = i_online
            i_g_embddings_target = i_target
            scores = model.rating(u_g_embeddings_online, i_g_embddings_target,
                                      u_g_embeddings_target, i_g_embddings_online)
        elif args.pos_mode == "cml":
            u_g_embeddings = user_gcn_emb[user_batch]
            i_g_embddings = item_gcn_emb
            scores = model.rating2(u_g_embeddings, i_g_embddings)
        else:
            if args.gnn == "Ultra_gcn":
                u_g_embeddings = user_gcn_emb[user_batch]
            else:
                u_g_embeddings = user_gcn_emb[user_batch]
            i_g_embddings = item_gcn_emb
            scores = model.rating(u_g_embeddings, i_g_embddings)
        # original test
        # rate_batch = scores.detach().cpu().numpy()
        # rate_batch[train_sp_mat[user_list].nonzero()] = -np.inf
        # ground_truth_batch = test_sp_mat[user_list]
        # for m in metrics:
        #     m['score'].append(m['metric'](rate_batch, ground_truth_batch, k=m['k']))
        # new valid
        # swap 
        # fill
        scores[train_sp_mat[user_list].nonzero()] = -np.inf
        _, topk_idx = torch.topk(scores, max(Ks), dim=-1)  # B * K
        index_after = torch.gather(test_sp_index, dim=1, index=topk_idx)
        batch_matrix_list.append(index_after)
        mat = test_sp_mat[user_list]
        # ipdb.set_trace()
    topk_idx_batch = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
    pos_idx_matrix = topk_idx_batch
    # matrix cal
    if args.restore and args.group_valid:
        mean_ndcg = ndcg_(pos_idx_matrix, pos_len_list)
        mean_recall = recall_(pos_idx_matrix, pos_len_list)
        mean_precision = precision_(pos_idx_matrix, pos_len_list).mean(axis=0)
        # np.save("./scores/{}_recall_per_user.npy".format(args.name), mean_recall[:, 20 - 1])
        # np.save("./scores/{}_ndcg_per_user.npy".format(args.name), mean_ndcg[:, 20 - 1])
        batch_group = np.concatenate(batch_group, axis=0)
        batch_group = batch_group.mean(axis=0)
        for i in batch_group:
            print(i)
        print("mean is {:.3}".format(batch_group.mean()))
        dict_file = {"recall_per_user": mean_recall[:, 20 - 1], 
                    "ndcg_per_user": mean_ndcg[:, 20 - 1], "batch_group":batch_group}
        mean_ndcg = mean_ndcg.mean(axis=0)
        mean_recall = mean_recall.mean(axis=0)
        with open("/data/wujk/codes/LDS7_0621/outputs/scores/{}_dict.pkl".format(args.name), "wb") as f:
            pickle.dump(dict_file, f)
    else:
        mean_ndcg = ndcg_(pos_idx_matrix, pos_len_list).mean(axis=0)
        mean_recall = recall_(pos_idx_matrix, pos_len_list).mean(axis=0)
        mean_precision = precision_(pos_idx_matrix, pos_len_list).mean(axis=0)  
    # print("CUDA\t ndcg20:{}\trecall:{}".format(ndcg20, recall20))

    # for m in metrics:
    #     m['score'] = np.concatenate(m['score']).mean()

    return (mean_ndcg, mean_recall, mean_precision)

def recall_(pos_index, pos_len):
    r"""Recall_ (also known as sensitivity) is the fraction of the total amount of relevant instances
    that were actually retrieved

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
        \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Recall@K` of each user.

    """
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
    :math:`U^{te}` stands for all users in the test set.

    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result

def precision_(pos_index, pos_len):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.

    """
    return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
