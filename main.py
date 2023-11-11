# from ast import arg
import argparse
import os
import random
from modules.MF_group import MF_group

import torch
import numpy as np
from utils.parser import parse_args

import time, json, sys, os
import logging, logging.config
from tqdm import tqdm
from copy import deepcopy
import logging
# from prettytable import PrettyTable
# from torch_scatter import scatter
from utils.data_loader import load_data, load_data_ciao
from utils.evaluate import test_sp
from utils.helper import early_stopping
import torch.nn.functional as F
import os.path as osp
import torch.autograd as autograd
n_users = 0
n_items = 0
def get_logger(name, log_dir, config_dir):
    config_dict = json.load(open( config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

class Sample(object):
    def __init__(self, user_dict, n_users, n_items, sampling_method="uniform", train_cf = None, train_mat=None, test_mat=None):
        self.n_users = n_users
        self.n_items = n_items
        self.random_list = []
        self.random_pr = 0
        self.random_list_length = 0
        self.sampling_method = sampling_method
        if self.sampling_method == "neg" :
            self.set_distribution(train_cf)
            self.used_ids = np.array([set() for _ in range(n_users)])
            for user in user_dict['train_user_set']:
                self.used_ids[user] = set(user_dict['train_user_set'][user])
        elif self.sampling_method == "pop":
            self.set_distribution(train_cf)
        elif self.sampling_method == "pop_p" or self.sampling_method == "pop_share":
            self.random_list = np.arange(self.n_items)
            self.p_item_candidate = np.zeros(n_items)
            for i in train_cf[:, 1]:
                self.p_item_candidate[i] += 1
            self.p_item_candidate = np.power(self.p_item_candidate, args.pop_pow)
            self.p_item_candidate = self.p_item_candidate / np.sum(self.p_item_candidate)
            print("sampling probablity has loaded! The whole is {}".format(np.sum(self.p_item_candidate)))
        elif self.sampling_method == "pos_sample":
            self.used_ids = []
            for user in range(n_users):
                self.used_ids.append(user_dict['train_user_set'][user])
            self.used_ids = np.array(self.used_ids)
            try:
                from cppimport import imp_from_filepath
                from os.path import join, dirname
                path = "./sampling.cpp"
                self.sampling = imp_from_filepath(path)
                self.sampling.seed(2020)
                sample_ext = True
            except:
                print("Cpp extension not loaded")
                sample_ext = False 
        elif self.sampling_method == "uniform_gpu":
            self.p_sample_1 = torch.ones((args.batch_size, self.n_items), device=device)
            self.p_sample_2 = torch.ones((len(train_cf) % args.batch_size, self.n_items), device=device)
        elif self.sampling_method == "uniform_gpu_ratio":
            users = train_mat.nonzero()[0]
            items = train_mat.nonzero()[1]
            self.p_sample = torch.ones(self.n_users, self.n_items)
            self.p_sample[users, items] = args.pos_prob
            self.p_sample = self.p_sample.to(device)
        elif self.sampling_method == "uniform_gpu_test_ratio":
            users = test_mat.nonzero()[0]
            items = test_mat.nonzero()[1]
            self.p_sample = torch.ones(self.n_users, self.n_items)
            self.p_sample[users, items] = args.pos_prob
            self.p_sample = self.p_sample.to(device)

    def set_distribution(self, train_cf=None):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        if self.sampling_method == "neg":
            self.random_list = np.arange(self.n_items)
            np.random.shuffle(self.random_list)
            self.random_pr = 0
            self.random_list_length = len(self.random_list)

        elif self.sampling_method == "pop":
            self.random_list = train_cf[:, 1]
            np.random.shuffle(self.random_list)

    def random_num(self, num):
        value_id = []
        self.random_pr %= self.random_list_length
        cnt = 0
        while True:
            if self.random_pr + num <= self.random_list_length:
                value_id.append(self.random_list[self.random_pr: self.random_pr + num])
                self.random_pr += num
                break
            else:
                value_id.append(self.random_list[self.random_pr:])
                num -= self.random_list_length - self.random_pr
                self.random_pr = 0
                cnt += 1
        return np.concatenate(value_id)

    def get_Pos_sample_by_key_ids(self, key_ids, num):
        key_ids = np.array(key_ids.cpu().numpy())

        value_ids = self.sampling.sample_postive_ByUser(key_ids, 
                                n_items, self.used_ids[key_ids], num)
        return torch.LongTensor(value_ids).to(device)

    def get_sample_by_key_ids(self, key_ids, num):
        key_ids = np.array(key_ids.cpu().numpy())
        key_num = len(key_ids)
        total_num = key_num * num
        # start
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        key_ids = np.tile(key_ids, num)
        # cnt = 0
        while len(check_list) > 0:
            value_ids[check_list] = self.random_num(len(check_list))
            check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        value_ids = torch.LongTensor(value_ids).to(device).view(-1, key_num) # [M, B]
        value_ids = value_ids.t().contiguous() # [B, M]

        return value_ids
        
    def get_feed_dict(self, train_entity_pairs, train_pos_set, start, end, sampling_method, n_negs=1):
        feed_dict = {}
        entity_pairs = train_entity_pairs[start: end]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        if sampling_method == "uniform":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif sampling_method == "uniform_gpu":
            neg_items = torch.multinomial(self.p_sample_1, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif sampling_method == "uniform_gpu_ratio" or sampling_method == "uniform_gpu_test_ratio":
            p_prop = torch.index_select(self.p_sample, 0, feed_dict['users'])
            neg_items = torch.multinomial(p_prop, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif sampling_method == "uniform_once":
            neg_items = np.random.choice(self.n_items, size=(n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif sampling_method == "neg":
            feed_dict['neg_items'] = self.get_sample_by_key_ids(entity_pairs[:, 0], n_negs)
        elif self.sampling_method == "pos_sample":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
            # feed_dict['neg_items'] = self.get_Neg_sample_by_key_ids(entity_pairs.shape[0], n_negs)
            feed_dict['pos_items_'] = self.get_Pos_sample_by_key_ids(entity_pairs[:, 0], args.pos_num)
        elif sampling_method == "group":
            return feed_dict
        elif sampling_method == "no_sample":
            return feed_dict
        return feed_dict

    def get_feed_dict_reset(self, train_entity_pairs, train_pos_set, start, sampling_method, n_negs=1):
        feed_dict = {}
        entity_pairs = train_entity_pairs[start:]
        feed_dict['users'] = entity_pairs[:, 0]
        feed_dict['pos_items'] = entity_pairs[:, 1]
        if sampling_method == "uniform":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif sampling_method == "uniform_gpu":
            neg_items = torch.multinomial(self.p_sample_2, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif sampling_method == "uniform_gpu_ratio" or sampling_method == "uniform_gpu_test_ratio":
            p_prop = torch.index_select(self.p_sample, 0, feed_dict['users'])
            neg_items = torch.multinomial(p_prop, num_samples=n_negs, replacement=True)
            feed_dict['neg_items'] = neg_items
        elif sampling_method == "uniform_once":
            neg_items = np.random.choice(self.n_items, size=(n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
        elif sampling_method == "neg":
            feed_dict['neg_items'] = self.get_sample_by_key_ids(entity_pairs[:, 0], n_negs)
        elif sampling_method == "pos_sample":
            neg_items = np.random.choice(self.n_items, size=(entity_pairs.shape[0], n_negs),
                                        replace=True)
            feed_dict['neg_items'] = torch.LongTensor(neg_items).to(device)
            # feed_dict['neg_items'] = self.get_Neg_sample_by_key_ids(entity_pairs.shape[0], n_negs)
            feed_dict['pos_items_'] = self.get_Pos_sample_by_key_ids(entity_pairs[:, 0], args.pos_num)
        elif sampling_method == "group":
            return feed_dict
        elif sampling_method == "no_sample":
            return feed_dict
        return feed_dict
  
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device, K
    args = parse_args()
    # print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    if not args.restore: 
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
    if args.bash_train:
        new_log_dir = args.log_dir + args.name[:args.name.find('D')-1] + '/'
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
        args.log_dir = new_log_dir
    logger = get_logger(args.name, args.log_dir, args.config_dir)
    logger.info(vars(args))
    train_cf, user_dict, sp_matrix, n_params, norm_mat, valid_pre, test_pre, item_group_idx = load_data(args, logger=logger)
        
    train_cf_size = len(train_cf)
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K
    args.Ks = eval(args.Ks)
    sample = Sample(user_dict, n_users, n_items, sampling_method=args.sampling_method, train_cf=train_cf, train_mat=sp_matrix['train_sp_mat'], test_mat=sp_matrix['test_sp_mat'])
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    """define model"""
    from modules.MF_simplex_alpha import MF
    from modules.MF_positive2 import MF_pos2
    from modules.MF_Frame import mf_frame
    from modules.LGN_Frame import lgn_frame
    from modules.MF_Uniform import mf_uniform
    from modules.NGCF_Frame import ngcf_frame
    from modules.DGCF_Frame import dgcf_frame
    from modules.ENMF_Frame import enmf_frame
    from modules.LINE_Frame import line_frame
    from modules.LRGCF_Frame import lrgcf_frame
    from modules.SGL_Frame import sgl_frame
    from modules.NCL_Frame import ncl_frame
    from modules.SimSGL_Frame import simsgl_frame
    from modules.DRO_Frame import dro_frame
    from modules.MF_Frame_align import mf_frame_align
    from modules.SGL_Frame_bsl import sgl_frame_bsl
    from modules.SimSGL_Frame_bsl import simsgl_frame_bsl
    from modules.LightGCL import lightgcl_frame
    from modules.LightGCL_bsl import lightgcl_frame_bsl
    
    if args.gnn == "mf_frame":
        model = mf_frame(n_params, args, norm_mat, item_group_idx, logger, len(train_cf)).to(device)
    elif args.gnn == "lgn_frame":
        model = lgn_frame(n_params, args, norm_mat, item_group_idx, logger, len(train_cf)).to(device)
    else:
        raise NotImplementedError
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    kill_cnt = 0
    best_ndcg = -np.inf
    eval_earlystop = args.eval_earlystop.split('@')
    eval_to_int = {'ndcg':0, 'recall':1, 'precision':2}
    eval_str = [eval_to_int[eval_earlystop[0]], eval(eval_earlystop[1])]
    logger.info('Evaluation Protocols is {} @ {}'.format(eval_str[0], eval_str[1]))
    """ makdir weights dir"""
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not args.restore:
        logger.info("start training ...")
        for epoch in range(args.epoch):
            train_cf_ = train_cf
            index = np.arange(len(train_cf_))
            np.random.shuffle(index)
            train_cf_ = train_cf_[index].to(device)
            """training"""
            model.train()
            loss, s = 0, 0
            losses_all = []
            losses_embed = []
            neg_rate_np = []
            train_s_t = time.time()
            tau_tmp = []
            uniform_scores = []
            aligh_scores = []
            while s + args.batch_size <= len(train_cf):
                # print('Step: {}'.format(s))
                batch = sample.get_feed_dict(train_cf_,
                                    user_dict['train_user_set'],
                                    s, s + args.batch_size,
                                    args.sampling_method,
                                    n_negs)
                loss, emb_loss, neg_rate = model(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_all.append(loss.item())
                losses_embed.append(emb_loss.item())
                loss += loss.item()
                s += args.batch_size
            
            # reset pairs training
            if len(train_cf) - s < args.batch_size:
                batch = sample.get_feed_dict_reset(train_cf_,
                                    user_dict['train_user_set'],
                                    s, args.sampling_method,
                                    n_negs)
                loss, emb_loss, neg_rate = model(batch, step=1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_all.append(loss.item())
                losses_embed.append(emb_loss.item())
                loss += loss.item()
                s += args.batch_size

            train_e_t = time.time()
            model.eval()
            with torch.no_grad():
                valid_st = time.time()
                valid_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='valid')
                test_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='test')
                valid_ed = time.time()
            print_result = 'E:{}|train_time: {:.4}, VALID_time: {:.4}, losses:{:.4}, losses_emb:{:.4}, best_valid({}): {:.4}\n'.format(epoch, 
                        train_e_t - train_s_t, valid_ed - valid_st, np.mean(losses_all), np.mean(losses_embed), args.eval_earlystop, best_ndcg)
            for k in args.Ks:
                print_result += 'valid \t N@{}: {:.4}, R@{}: {:.4}, P@{}: {:.4}\n'.format(
                    k, valid_ret[0][k-1], k, valid_ret[1][k-1], k, valid_ret[2][k-1])
            logger.info(print_result)

            if valid_ret[eval_str[0]][eval_str[1] - 1] > best_ndcg:
                best_ndcg = valid_ret[eval_str[0]][eval_str[1] - 1]
                kill_cnt = 0
                save_path = os.path.join(args.out_dir, args.name + '.ckpt')
                torch.save(model.state_dict(), save_path)
            else:
                kill_cnt += 1
                if kill_cnt > args.t_patience:
                    break
    # test
    if args.restore:
        logger.info('start to test!!\n')
        load_path = os.path.join(args.out_dir, args.name + '.ckpt')
        model.load_state_dict(torch.load(load_path), False)
        model.eval()
        with torch.no_grad():
            test_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, item_group_idx, mode='test')

        # logger.info('Test result: NDCG@20: {:.4} Recall@20: {:.4}'.format(test_ret[0], test_ret[1]))
        print_result = '\n'
        for k in args.Ks:
            print_result += 'TEST \t N@{}: {:.4}, R@{}: {:.4}, P@{}: {:.4}\n'.format(
                k, test_ret[0][k-1], k, test_ret[1][k-1], k, test_ret[2][k-1])
        logger.info(print_result)
    else:
        logger.info('start to test!!\n')
        load_path = os.path.join(args.out_dir, args.name + '.ckpt')
        model.load_state_dict(torch.load(load_path), False)
        model.eval()
        with torch.no_grad():
            test_ret = test_sp(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, item_group_idx, mode='test')

        print_result = '\n'
        for k in args.Ks:
            print_result += 'TEST \t N@{}: {:.4}, R@{}: {:.4}, P@{}: {:.4}\n'.format(
                k, test_ret[0][k-1], k, test_ret[1][k-1], k, test_ret[2][k-1])
        logger.info(print_result)



