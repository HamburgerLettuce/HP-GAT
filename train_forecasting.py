import os
import sys
sys.path.append("..")
import time
import datetime
import argparse
import numpy as np
import math

from random import SystemRandom

parser = argparse.ArgumentParser('ITS Forecasting')
parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=100, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=72, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--pred_window', type=int, default=168, help="number of hours (months for ushcn) as pred window")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--lr',  type=float, default=1e-4, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=1e-5, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='iowa', help="Dataset to load. Available: physionet, mimic, ushcn, activity")
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='Hi-Patch', help="Model name")
parser.add_argument('--nhead', type=int, default=8, help="heads in Transformer")
parser.add_argument('--nlayer', type=int, default=3, help="# of layer in TSmodel")
parser.add_argument('-ps', '--patch_size', type=float, default=12, help="window size for a patch")
parser.add_argument('--stride', type=float, default=6, help="period stride for patch sliding")
parser.add_argument('-hd', '--hid_dim', type=int, default=128, help="Hidden dim of node embeddings")
parser.add_argument('--alpha', type=float, default=0.85, help="Proportion of Time decay")
parser.add_argument('--res', type=float, default=1, help="Res")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
torch.cuda.empty_cache()
import torch.optim as optim
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from lib.evaluation import *
from model.hpgat import *
import warnings
warnings.filterwarnings("ignore")

file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()

print("PID, device:", args.PID, args.device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_of_patches(n_patch):
    n_patch = int(n_patch)
    
    if n_patch <= 1:
        return 1
    
    if n_patch % 2 != 0:
        n_patch = n_patch + 1
    
    return 1 + layer_of_patches(n_patch // 2)

def layer_of_patches_iterative(n_patch):
    n_patch = int(n_patch)
    layers = 0
    
    while n_patch > 1:
        layers += 1
        if n_patch % 2 != 0:
            n_patch = n_patch + 1
        n_patch = n_patch // 2
    
    return layers + 1

if __name__ == '__main__':
    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)
    
    data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]
    num_stations = data_obj["num_stations"]
    
    args.ndim = input_dim
    args.num_stations = num_stations
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1
    args.patch_layer = layer_of_patches_iterative(args.npatch)
    args.scale_patch_size = args.patch_size / (args.history + args.pred_window)
    args.task = 'forecasting'
    
    model = HP_GAT(args, num_sites=num_stations, use_spatial=False).to(args.device)
    
    if 'norm_dict' in data_obj and 'topology_info' in data_obj["norm_dict"]:
        topology_info = data_obj["norm_dict"]['topology_info']
        if topology_info['has_topology'] and topology_info['edge_index'] is not None:
            model.topology_info = {
                'edge_index': topology_info['edge_index'].long().to(args.device),
                'edge_weights': topology_info['edge_weights'].to(args.device)
                    if topology_info['edge_weights'] is not None else None,
                'num_nodes': topology_info['num_nodes'],
                'has_topology': True
            }
    
    if(args.n < 12000):
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr_{}seed.log". \
            format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr, args.seed)
    
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_batches = data_obj["n_train_batches"]
    print(f"\nNumber of training batches: {num_batches}")
    
    best_val_mse = np.inf
    test_res = None
    best_val_kge = -np.inf
    best_val_nse = -np.inf
    
    print("\nStarting training...")
    for itr in range(args.epoch):
        st = time.time()
    
        model.train()
        train_loss_sum = 0
        train_batch_count = 0
        
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            
            train_res = compute_all_losses(model, batch_dict, norm_dict=None, phase="train")
            train_loss = train_res["loss"]
            train_loss.backward()
            optimizer.step()
            
            train_loss_sum += train_loss.item()
            train_batch_count += 1
        
        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else 0
    
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"],
                                norm_dict=data_obj["norm_dict"])
            
            if (val_res["kge"] > best_val_kge) or (val_res["kge"] == best_val_kge and val_res["mse"] < best_val_mse):
                best_val_kge = val_res["kge"]
                best_val_mse = val_res["mse"]
                best_val_nse = val_res["nse"]
                best_iter = itr
                
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"],
                                    norm_dict=data_obj["norm_dict"])
    
            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
            logger.info("Train - Avg Loss: {:.5f}".format(avg_train_loss))
            
            logger.info("Val - Loss: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, MAE: {:.5f}, KGE: {:.5f}, NSE: {:.5f}" \
                .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["kge"], val_res["nse"]))
    
            if test_res is not None:
                logger.info("Test (Best) - Epoch: {}, Loss: {:.5f}, MSE: {:.5f}, RMSE: {:.5f}, MAE: {:.5f}, KGE: {:.5f}, NSE: {:.5f}" \
                    .format(best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["kge"], test_res["nse"]))
                
                logger.info("Best Val Metrics - KGE: {:.5f}, MSE: {:.5f}, NSE: {:.5f}".format(
                    best_val_kge, best_val_mse, best_val_nse))
    
        if(itr - best_iter >= args.patience):
            print(f"\nEarly stopping! Best epoch: {best_iter}")
            logger.info(f"Early stopping at epoch {itr}, best epoch was {best_iter}")
            logger.info(f"Final Test Results - KGE: {test_res['kge']:.5f}, NSE: {test_res['nse']:.5f}, RMSE: {test_res['rmse']:.5f}")
            break
    
    if test_res is not None:
        print("\n" + "="*60)
        print("Final Test Results:")
        print(f"  Best Epoch: {best_iter}")
        print(f"  KGE:  {test_res['kge']:.5f}")
        print(f"  NSE:  {test_res['nse']:.5f}")
        print(f"  RMSE: {test_res['rmse']:.5f}")
        print(f"  MAE:  {test_res['mae']:.5f}")
        print(f"  MSE:  {test_res['mse']:.5f}")
        print("="*60)
        
        logger.info("\n" + "="*60)
        logger.info("Final Test Results:")
        logger.info(f"  Best Epoch: {best_iter}")
        logger.info(f"  KGE:  {test_res['kge']:.5f}")
        logger.info(f"  NSE:  {test_res['nse']:.5f}")
        logger.info(f"  RMSE: {test_res['rmse']:.5f}")
        logger.info(f"  MAE:  {test_res['mae']:.5f}")
        logger.info(f"  MSE:  {test_res['mse']:.5f}")
        logger.info("="*60)