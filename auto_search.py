import os 
import argparse
import torch
import numpy as np
from corrupter import BernCorrupter
from read_data import DataLoaderAuto
from utils import logger_init, plot_config_auto
import warnings
from base_auto import BaseModel

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter
from smac.scenario.scenario  import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

# TODO

parser = argparse.ArgumentParser(description="Parser for NSCaching (auto)")
parser.add_argument('--task_dir', type=str, default='../../KG_Data/FB15K', help='the directory to dataset')
parser.add_argument('--model', type=str, default='SimplE',  help='scoring function, support [TransE, TransD, TransH, DistMult, ComplEx, SimplE, RotatE]')
parser.add_argument('--sample', type=str, default='auto', help='sampling method from the cache')
parser.add_argument('--remove', type=bool, default=False, help='whether to remove positive sample in cache periodically')
parser.add_argument('--loss', type=str, default='point', help='loss function, pair_loss or  point_loss')
parser.add_argument('--save', type=bool, default=False, help='whether save model')
parser.add_argument('--s_epoch', type=int, default=1000, help='which epoch should be saved, only work when save=True')
parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--margin', type=float, default=4.0, help='set margin value for pair loss')
parser.add_argument('--lamb', type=float, default=0.01, help='set weight decay value')
parser.add_argument('--hidden_dim', type=int, default=100, help='set embedding dimension')
parser.add_argument('--gpu', type=str, default='0', help='set gpu #')
parser.add_argument('--p', type=int, default=1, help='set distance norm')
parser.add_argument('--lr', type=float, default=0.0001, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument('--lazy', type=int, default=1, help='period of iterations for lazy update')
parser.add_argument('--n_batch', type=int, default=4096, help='number of batch size')
parser.add_argument('--n_sample', type=int, default=1, help='number of negative samples')
parser.add_argument('--epoch_per_test', type=int, default=50, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')
parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')

parser.add_argument('--log_prefix', type=str, default='', help='log prefix')

args = parser.parse_args()

dataset = args.task_dir.split('/')
if len(dataset[-1]) > 0:
    dataset = dataset[-1]
else:
    dataset = dataset[-2]

directory = os.path.join('results', args.model)
if not os.path.exists(directory):
    os.makedirs(directory)
   
args.out_dir = directory
args.perf_file = os.path.join(directory, '_'.join([dataset, args.sample]) + args.out_file_info + '.txt')
print('output file name:', args.perf_file)

logger_init(args)

task_dir = args.task_dir
loader = DataLoaderAuto(task_dir)

n_ent, n_rel = loader.graph_size()

train_data = loader.load_data('train')
valid_data = loader.load_data('valid')
test_data  = loader.load_data('test')
args.n_train = len(train_data[0])
print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))


heads, tails = loader.heads_tails()

train_data = [torch.LongTensor(vec) for vec in train_data]
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data  = [torch.LongTensor(vec) for vec in test_data]


def run_kge(params):
    if params["alpha_1"] < -4:
        args.alpha_1 = 0
    else:
        args.alpha_1 = 10**params["alpha_1"]

    if params["alpha_2"] < -1:
        args.alpha_2 = 0
    else:
        args.alpha_2 = 10**params["alpha_2"]

    if params["alpha_3"] < -1:
        args.alpha_3 = 0
    else:
        args.alpha_3 = 10**params["alpha_3"]

    args.N_1 = params["N_1"]
    args.N_2 = params["N_2"]
    head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos = loader.get_cache_list(args.N_1)
    caches = [head_idx, tail_idx, head_cache, tail_cache, head_pos, tail_pos]
    plot_config_auto(args)
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    model = BaseModel(n_ent, n_rel, args)
    tester_val = lambda: model.test_link(valid_data, n_ent, heads, tails, args.filter)
    tester_tst = lambda: model.test_link(test_data, n_ent, heads, tails, args.filter)
    best_mrr, best_str = model.train(train_data, caches, corrupter, tester_val, tester_tst)
    with open(args.perf_file, 'a') as f:
        print('Training finished and best performance:', best_str)
        f.write('best_performance: '+best_str)

    return -best_mrr 


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(5)
    warnings.filterwarnings("ignore", category=UserWarning)

    # AutoML based on SMAC
    cs = ConfigurationSpace()
    alpha_1 = UniformFloatHyperparameter("alpha_1", -5, 0, default_value=-5)
    cs.add_hyperparameter(alpha_1)
    alpha_2 = UniformFloatHyperparameter("alpha_2", -2, 2, default_value=-2)
    cs.add_hyperparameter(alpha_2)
    alpha_3 = UniformFloatHyperparameter("alpha_3", -2, 2, default_value=-2)
    cs.add_hyperparameter(alpha_3)
    N_1 = CategoricalHyperparameter("N_1", [10, 30, 50, 70, 90], default_value=50) 
    cs.add_hyperparameter(N_1)
    N_2 = CategoricalHyperparameter("N_2", [10, 30, 50, 70, 90], default_value=50) 
    cs.add_hyperparameter(N_2)
    
    scenario = Scenario(
            {"run_obj": "quality",  
             "runcount-limit": 500,
             "cs": cs,
             "deterministic": "true",
             })
    print('start optimizing')
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42), tae_runner=run_kge)
    incubent = smac.optimize()
    print("Optimized hyper:", incubent)
