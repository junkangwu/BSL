import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon",
                        help="Choose a dataset:[amazon,yelp2018,ali]")
    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")

    parser.add_argument('--name',        default='testrun',                  help='Set run name for saving/restoring models')

    # ===== train ===== #
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train_ratio')
    parser.add_argument("--gnn", nargs="?", default="mf_frame",
                        help="Choose a recommender:[lightgcn, ngcf]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--ns", type=str, default='mixgcf', help="rns,mixgcf")
    parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 15, 20]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument("--eval_earlystop", type=str, default='recall@20', help="rns,mixgcf")
    # ===== neg_debias ===== #
    parser.add_argument('--tau_plus', type=float, default=0.1, help='tau_plus')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--beta', type=float, default=1.0, help='temperature')
    parser.add_argument('--sample_num', type=int, default=2048, help='temperature')
    parser.add_argument("--s_neg", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--s_pos", type=int, default=64, help="number of candidate negative")
    # ==== norm change ===== #
    parser.add_argument('--norm_change', dest='norm_change', action='store_true', help='whether to change norm')
    parser.add_argument('--norm_path', type=str, help='path of other model')
    parser.add_argument('--u_norm', dest='u_norm', action='store_true', help='whether to change norm')
    parser.add_argument('--i_norm', dest='i_norm', action='store_true', help='whether to change norm')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument('--save_analyses', dest='save_analyses', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('--log_hard_rate', dest='log_hard_rate', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('--norm_before', dest='norm_before', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('--norm_after', dest='norm_after', action='store_true', help='Restore from the previously saved model')

    parser.add_argument('--batch_var', dest='batch_var', action='store_true', help='Restore from the previously saved model')
    # parser.add_argument("--save_analyses", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    parser.add_argument('--logdir',          dest='log_dir',         default='./log_pos/',               help='Log directory')
    parser.add_argument('--config',          dest='config_dir',      default='./config/',            help='Config directory')
    parser.add_argument('--restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('--emb_l2',         dest='emb_l2',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('--lam', type=float, default=0.1, help='alpha')
    parser.add_argument('--loss_fn', type=str, default="Pos_DROLoss", help="loss for training")
    parser.add_argument('--sampling_method', type=str, default="uniform", help="loss for training")
    parser.add_argument('--load_name', type=str, default="uniform", help="loss for training")
    parser.add_argument('--generate_mode', type=str, default="normal", help="loss for training")
    parser.add_argument('--temperature_2', type=float, default=1.0, help='temperature_2')
    parser.add_argument('--temperature_3', type=float, default=1.0, help='temperature_2')
    parser.add_argument('--neg_rate', type=str, default="", help='t_patience')
    parser.add_argument('--pos_mode', type=str, default="multi", help='t_patience')
    return parser.parse_args()
