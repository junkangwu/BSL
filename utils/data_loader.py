import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]

def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    # print(len(inter_mat))
    return np.array(inter_mat)

def swap_train_sp_mat(valid_user_set, test_user_set):
    # valid
    valid_uid2swap_idx = np.array([None] * n_users)
    valid_uid2rev_swap_idx = np.array([None] * n_users)
    # valid_pos_len_list = np.zeros(n_users)
    # test_pos_len_list = np.zeros(n_users)
    valid_pos_len_list = []
    for uid in valid_user_set:
        positive_item = valid_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        valid_uid2swap_idx[uid] = swap_idx
        valid_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # valid_pos_len_list[uid] = postive_item_num
        valid_pos_len_list.append(postive_item_num)
    # test
    test_uid2swap_idx = np.array([None] * n_users)
    test_uid2rev_swap_idx = np.array([None] * n_users)
    test_pos_len_list = []
    for uid in test_user_set:
        positive_item = test_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        test_uid2swap_idx[uid] = swap_idx
        test_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # test_pos_len_list[uid] = postive_item_num
        test_pos_len_list.append(postive_item_num)

    return (valid_uid2swap_idx, valid_uid2rev_swap_idx, valid_pos_len_list), \
            (test_uid2swap_idx, test_uid2rev_swap_idx, test_pos_len_list)


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if dataset == 'ali' or dataset == 'amazon':
        if args.neg_rate != "" and args.neg_rate != "_0.":
            print("DIRTY!!!! COMPLETE AMAZON NEED TO MINUS N_USERS!!!")
            valid_data[:, 1] -= n_users
            test_data[:, 1] -= n_users
        elif args.neg_rate == "":
            print("COMPLETE AMAZON NEED TO MINUS N_USERS!!!")
            n_items -= n_users
            # remap [n_users, n_users+n_items] to [0, n_items]
            train_data[:, 1] -= n_users
            valid_data[:, 1] -= n_users
            test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

    train_sp_mat = sp.csr_matrix((np.ones_like(train_data[:, 0]),
                                    (train_data[:, 0], train_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    valid_sp_mat = sp.csr_matrix((np.ones_like(valid_data[:, 0]),
                                    (valid_data[:, 0], valid_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    test_sp_mat =sp.csr_matrix((np.ones_like(test_data[:, 0]), 
                                    (test_data[:, 0], test_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    # prepare for top k accerate
    valid_pre, test_pre = swap_train_sp_mat(valid_user_set, test_user_set)
    # print(len(train_user_set))
    # print(train_sp_mat.sum())
#     print()
    return train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre

def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)

def lr_mat(train_cf):
    user_num = n_users
    item_num = n_items
    vals = [1.] * len(train_cf)
    adj = sp.coo_matrix((vals, (train_cf[:, 0], train_cf[:, 1])), shape=(user_num, item_num))

    rowsum = np.array(adj.sum(1)) + 1
    colsum = np.array(adj.sum(0)) + 1

    d_inv_sqrt_row = np.sqrt(np.power(rowsum, -1)).flatten()
    d_inv_sqrt_col = np.sqrt(np.power(colsum, -1)).flatten()

    d_inv_sqrt_row[np.isinf(d_inv_sqrt_row)] = 0.
    d_inv_sqrt_col[np.isinf(d_inv_sqrt_col)] = 0.

    d_inv_sqrt_row = sp.diags(d_inv_sqrt_row)
    d_inv_sqrt_col = sp.diags(d_inv_sqrt_col)
    
    bi_lap = d_inv_sqrt_row.dot(adj).dot(d_inv_sqrt_col)
    bi_lap2 = d_inv_sqrt_col.dot(adj.getH()).dot(d_inv_sqrt_row)

    
    return (bi_lap, bi_lap2, 1./rowsum, 1./ colsum)

def load_data(model_args, logger):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset == 'yelp2018' or dataset == "amazon-book" or dataset == "gowalla" or dataset == 'ml' or dataset == "citeulike-new" or dataset == "ali-new":
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    # read_cf = read_cf_yelp2018
    print(args.neg_rate)
    if args.neg_rate == "_0.":
        print("neg_rate is 0!!!")
        train_cf = read_cf(directory + 'train.txt')
    else:
        train_cf = read_cf(directory + 'train{}.txt'.format(args.neg_rate))
    logger.info("load train{}.txt".format(args.neg_rate))
    test_cf = read_cf(directory + 'test.txt')
    if dataset == 'yelp2018' or dataset == "amazon-book" or dataset == "gowalla" or dataset == 'ml' or dataset == "citeulike-new":
        valid_cf = test_cf
    else:
        valid_cf = read_cf(directory + 'valid.txt')
    train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre = statistics(train_cf, valid_cf, test_cf)

    logger.info('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)
    logger.info("train set len is {}".format(len(train_cf)))
    logger.info("test set len is {}".format(len(valid_cf)))

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset != 'yelp2018' else None,
        'test_user_set': test_user_set,
    }
    sp_matrix = {
        'train_sp_mat': train_sp_mat,
        'valid_sp_mat': valid_sp_mat,
        'test_sp_mat': test_sp_mat
    }
    logger.info('loading over ...')
    logger.info("users are {} items are {}".format(n_users, n_items))
   
    return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, None
# START Ultra

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

def dict_to_list(train_mat):
    inter_mat = list()
    for uid in train_mat:
        items = list(train_mat[uid].keys())
        for item in items:
            inter_mat.append([uid, item])
    return np.array(inter_mat)

def dict_set(base_dict, user_id, item_id, val):
    if user_id in base_dict:
        base_dict[user_id][item_id] = val
    else:
        base_dict[user_id] = {item_id: val}

def list_to_dict(base_list):
    result = {}
    for user_id, item_id, value in base_list:
        dict_set(result, user_id, item_id, value)
    return result

def read_interaction_file(f):
    total_interactions = []
    for user_id, line in enumerate(f.readlines()):
        items = line.replace('\n', '').split(' ')[1:]
        for item in items:
            item_id = item
            total_interactions.append((user_id, item_id, 1))
    return total_interactions

def get_count_dict(total_interactions):
    user_count_dict, item_count_dict = {}, {}

    for interaction in total_interactions:
        user, item, rating = interaction

        if user not in user_count_dict:
            user_count_dict[user] = 0
        if item not in item_count_dict:
            item_count_dict[item] = 0
    
        user_count_dict[user] += 1
        item_count_dict[item] += 1

    return user_count_dict, item_count_dict

def get_count_user(total_interactions):
    user_count_dict = {}

    for interaction in total_interactions:
        user, _, _ = interaction

        if user not in user_count_dict:
            user_count_dict[user] = 0
        user_count_dict[user] += 1

    return user_count_dict

def get_count_item(total_interactions):
    item_count_dict = {}

    for interaction in total_interactions:
        _, item, _ = interaction

        if item not in item_count_dict:
            item_count_dict[item] = 0
        item_count_dict[item] += 1

    return item_count_dict

def filter_interactions_order(total_interaction_tmp, min_count=[5, 0]):
    total_interactions_1 = []
    user_to_id, item_to_id = {}, {}
    user_count, item_count = 0, 0
    # filter by users
    user_count_dict = get_count_user(total_interaction_tmp)
    for line in total_interaction_tmp:
        user, item, rating = line

        if user_count_dict[user] < min_count[0]:
            continue

        if user not in user_to_id:
            user_to_id[user] = user_count
            user_count += 1

        if item not in item_to_id:
            item_to_id[item] = item_count
            item_count += 1

        user_id = user_to_id[user]
        item_id = item_to_id[item]
        rating = 1.

        total_interactions_1.append((user_id, item_id, rating))
    #filter by item
    total_interactions_complete = []
    user_to_id, item_to_id = {}, {}
    user_count, item_count = 0, 0
    item_count_dict = get_count_item(total_interactions_1)
    for line in total_interactions_1:
        user, item, rating = line

        if item_count_dict[item] < min_count[1]:
            continue

        if user not in user_to_id:
            user_to_id[user] = user_count
            user_count += 1

        if item not in item_to_id:
            item_to_id[item] = item_count
            item_count += 1

        user_id = user_to_id[user]
        item_id = item_to_id[item]
        rating = 1.

        total_interactions_complete.append((user_id, item_id, rating))

    return user_count, item_count, user_to_id, item_to_id, total_interactions_complete

def filter_interactions(total_interaction_tmp, user_count_dict, item_count_dict, min_count=[5, 0]):
    total_interactions = []
    user_to_id, item_to_id = {}, {}
    user_count, item_count = 0, 0
    user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
    for line in total_interaction_tmp:
        user, item, rating = line

        # count filtering
        if user_count_dict[user] < min_count[0]:
            continue

        if item_count_dict[item] < min_count[1]:
            continue

        if user not in user_to_id:
            user_to_id[user] = user_count
            user_count += 1

        if item not in item_to_id:
            item_to_id[item] = item_count
            item_count += 1

        user_id = user_to_id[user]
        item_id = item_to_id[item]
        rating = 1.

        total_interactions.append((user_id, item_id, rating))

    return user_count, item_count, user_to_id, item_to_id, total_interactions


