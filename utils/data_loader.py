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
        if args.neg_rate != "":
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
    # with open("train_sp_mat", 'wb') as f:
    #     pickle.dump(train_sp_mat, f)
    # if args.group_mode == "equal_pop":
    #     i_degree = np.array(train_sp_mat.sum(0))[0].astype(np.int32)
    #     i_degree_sort = np.argsort(i_degree)
    #     i_degree_cumsum = i_degree.copy()
    #     cum_sum = 0
    #     for x in i_degree_sort:
    #         cum_sum += i_degree_cumsum[x]
    #         i_degree_cumsum[x] = cum_sum
    #     split_idx = np.linspace(0, train_sp_mat.sum(), args.group_num + 1)
    #     item_group_idx = np.searchsorted(split_idx[1: -1], i_degree_cumsum)
    # elif args.group_mode == "equal_num":
    #     i_degree = np.array(train_sp_mat.sum(0))[0].astype(np.int32)
    #     i_degree_sort = np.argsort(i_degree)
    #     num_per_group = int(n_items / args.group_num)
    #     print("The whole items are {} Group_capacity is {}, Num_per_Group {}".format(n_items, args.group_num, num_per_group))
    #     item_group_idx = np.zeros(len(i_degree_sort))
    #     cnt_item = 0
    #     for i in i_degree_sort:
    #         item_group_idx[i] = cnt_item // num_per_group
    #         cnt_item += 1
    #     print("Group ID max is {}".format(np.max(item_group_idx)))
    if args.group_valid:
        i_degree = np.array(train_sp_mat.sum(0))[0].astype(np.int32)
        i_degree_sort = np.argsort(i_degree)
        i_degree_cumsum = i_degree.copy()
        cum_sum = 0
        for x in i_degree_sort:
            cum_sum += i_degree_cumsum[x]
            i_degree_cumsum[x] = cum_sum
        split_idx = np.linspace(0, train_sp_mat.sum(), args.group_num + 1)
        item_group_idx = np.searchsorted(split_idx[1: -1], i_degree_cumsum)
        cnt_sum_ = 0
        group_sum_ = 0
        # group_idx_item = np.zeros(n_items)

        for i in range(10):
            # item_index = item_group_idx[i]
            # group_idx_item[item_index] = i

            cnt_sum = i_degree[item_group_idx == i]
            cnt_sum_ += len(cnt_sum)
            group_sum_ += cnt_sum.sum()
            print("size of the group is {}".format(len(cnt_sum)))
            print("sum degree of the group is {}".format(cnt_sum.sum()))
            print("Min degree of the group is {}".format(cnt_sum.min()))
            print("Max degree of the group is {}".format(cnt_sum.max()))
        print("The whole items is {}".format(cnt_sum_))
        print("The whole interactions is {}".format(group_sum_))
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, item_group_idx
    if args.gnn == "Ultra_gcn" or args.gnn == "Ultra_gcn_vanilla":
        train_mat = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        for x in train_cf:
            train_mat[x[0], x[1]] = 1.0
        items_D = np.sum(train_mat, axis = 0).reshape(-1)
        users_D = np.sum(train_mat, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                        "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
        # Compute \Omega to extend UltraGCN to the item-item occurrence graph
        ii_cons_mat_path = directory + '_ii_constraint_mat'
        ii_neigh_mat_path = directory + '_ii_neighbor_mat'
        
        if os.path.exists(ii_cons_mat_path):
            ii_constraint_mat = pload(ii_cons_mat_path)
            ii_neighbor_mat = pload(ii_neigh_mat_path)
        else:
            ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, args.ii_neighbor_num)
            pstore(ii_neighbor_mat, ii_neigh_mat_path)
            pstore(ii_constraint_mat, ii_cons_mat_path)
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, (constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    elif args.gnn == "dgcf_frame":
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, train_sp_mat
    elif args.gnn == "enmf_frame":
        row_ids, col_ids = train_cf[:,0], train_cf[:, 1]
        values = np.ones(len(train_cf))

        row_num = n_params['n_users']
        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1
        col_num = np.max(history_len)

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_len[row_id] += 1
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, torch.LongTensor(history_matrix)
    elif args.gnn == "sgl_frame":
        print("SGL comming!!")
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, train_cf
    elif args.gnn == "sgl_frame_bsl":
        print("SGL BSL comming!!")
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, train_cf
    elif args.gnn == "line_frame":
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, train_cf
    elif args.gnn == "lrgcf_frame":
        norm_mat = lr_mat(train_cf)
        train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, None
    elif args.gnn == "ncl_frame":
        return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, train_sp_mat.tocoo()
    return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, None
# START Ultra

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

def statistics_ciao(train_data, valid_data, test_data):
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

    return train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre

def load_data_ciao(model_args, train_ratio=0.5, min_count=[5, 5], logger=None):
    test_ratio = (1. - train_ratio) / 2
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset 
    if dataset == 'ciao':
        filename = directory + '/ratings.txt'
        file_ciao = np.loadtxt(filename, dtype=np.int32)
        total_interaction_tmp = []
        for i in file_ciao:
            total_interaction_tmp.append((i[0], i[1], 1))
    elif dataset == 'citeulike':
        with open(os.path.join(directory, 'users.dat'), 'r') as f:
            total_interaction_tmp = read_interaction_file(f)
    elif dataset == "ali":
        total_interaction_tmp = []
        for file_type in ["train", "valid", "test"]:
            logger.info("start to load {}.txt".format(file_type))
            filename = directory + '/{}.txt'.format(file_type)
            file_ciao = np.loadtxt(filename, dtype=np.int32)
            for i in file_ciao:
                total_interaction_tmp.append((i[0], i[1], 1))
    else:
        raise NotImplementedError
    logger.info("origin len is {}".format(len(total_interaction_tmp)))
    # user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
    global n_users, n_items
    # n_users, n_items, user_to_id, item_to_id, total_interactions = filter_interactions(total_interaction_tmp, user_count_dict, item_count_dict, min_count=min_count)
    n_users, n_items, user_to_id, item_to_id, total_interactions = filter_interactions_order(total_interaction_tmp, min_count=min_count)
    # ciao filter, split train / val /test
    total_mat = list_to_dict(total_interactions)
    train_mat, valid_mat, test_mat = {}, {}, {}

    for user in total_mat:
        items = list(total_mat[user].keys())
        np.random.shuffle(items)

        num_test_items = int(len(items) * test_ratio)
        test_items = items[:num_test_items]
        valid_items = items[num_test_items: num_test_items*2]
        train_items = items[num_test_items*2:]

        for item in test_items:
            dict_set(test_mat, user, item, 1)

        for item in valid_items:
            dict_set(valid_mat, user, item, 1)

        for item in train_items:
            dict_set(train_mat, user, item, 1)
           
    train_mat_t = {}

    for user in train_mat:
        for item in train_mat[user]:
            dict_set(train_mat_t, item, user, 1)
    
    for user in list(valid_mat.keys()):
        for item in list(valid_mat[user].keys()):
            if item not in train_mat_t:
                del valid_mat[user][item]
        if len(valid_mat[user]) == 0:
            del valid_mat[user]
            del test_mat[user]
            
    for user in list(test_mat.keys()):
        for item in list(test_mat[user].keys()):
            if item not in train_mat_t:
                del test_mat[user][item]
        if len(test_mat[user]) == 0:
            del test_mat[user]
            del valid_mat[user]
    logger.info('Split completed!')
    # statistics
    train_cf = dict_to_list(train_mat)
    valid_cf = dict_to_list(valid_mat)
    test_cf = dict_to_list(test_mat)
    # print(train_cf)
    train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre = statistics_ciao(train_cf, valid_cf, test_cf) 
    np.savetxt(directory + '-new/train.txt', train_cf.astype(int), fmt='%i')
    np.savetxt(directory + '-new/valid.txt', valid_cf.astype(int), fmt='%i')
    np.savetxt(directory + '-new/test.txt', test_cf.astype(int), fmt='%i')
    logger.info('building the adj mat ...')
    logger.info("train set len is {}".format(len(train_cf)))
    logger.info("valid set len is {}".format(len(valid_cf)))
    logger.info("test set len is {}".format(len(test_cf)))
    norm_mat = build_sparse_graph(train_cf)

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

    return train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre

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


