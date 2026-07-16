from torch.utils.data import DataLoader
from lib.iowa import *
import lib.utils as utils
import torch
import math
import pandas as pd
import os

def parse_datasets(args, patch_ts=False, length_stat=False):
    device = args.device
    dataset_name = args.dataset
    
    if dataset_name == "iowa":
        total_dataset = Iowa('data/iowa/', n_samples=args.n, device=torch.device("cpu"))
        
        all_station_data = {}
        for station_id, tt, vals, mask in total_dataset.data:
            all_station_data[station_id] = {
                'times': tt,
                'values': vals,
                'mask': mask
            }
        
        all_times_set = set()
        for data in all_station_data.values():
            for t in data['times']:
                all_times_set.add(t.item())
        
        all_times = sorted(list(all_times_set))
        total_timesteps = len(all_times)
        
        train_end = int(total_timesteps * 0.6)
        val_end = int(total_timesteps * 0.8)
        
        train_times = all_times[:train_end]
        val_times = all_times[train_end:val_end]
        test_times = all_times[val_end:]
        
        all_station_ids = sorted(list(all_station_data.keys()))
        num_stations = len(all_station_ids)
        
        topology_file = os.path.join('data/iowa/raw', 'topology.csv')
        topology_data = None
        if os.path.exists(topology_file):
            print(f"Loading topology file: {topology_file}")
            topology_data = pd.read_csv(topology_file)
        
        train_windows = create_multi_site_time_windows(
            train_times, all_station_data, all_station_ids, args, torch.device("cpu"), "train"
        )
        val_windows = create_multi_site_time_windows(
            val_times, all_station_data, all_station_ids, args, torch.device("cpu"), "val"
        )
        test_windows = create_multi_site_time_windows(
            test_times, all_station_data, all_station_ids, args, torch.device("cpu"), "test"
        )
        
        norm_dict = compute_normalization_params(train_windows, device)
        norm_dict["num_stations"] = num_stations
        norm_dict["station_ids"] = all_station_ids
        
        if topology_data is not None:
            station_id_to_idx = {station_id: idx for idx, station_id in enumerate(all_station_ids)}
            valid_edges = []
            for _, row in topology_data.iterrows():
                from_id, to_id = int(row['from']), int(row['to'])
                if from_id in station_id_to_idx and to_id in station_id_to_idx:
                    valid_edges.append((station_id_to_idx[from_id], station_id_to_idx[to_id], float(row['cost'])))
            
            if valid_edges:
                from_nodes = torch.tensor([e[0] for e in valid_edges], dtype=torch.long, device=device)
                to_nodes = torch.tensor([e[1] for e in valid_edges], dtype=torch.long, device=device)
                costs = torch.tensor([e[2] for e in valid_edges], dtype=torch.float32, device=device)
                edge_index = torch.stack([from_nodes, to_nodes], dim=0)
                edge_weights = 1.0 / (costs + 1e-8)
                norm_dict['topology_info'] = {
                    'edge_index': edge_index, 'edge_weights': edge_weights,
                    'num_nodes': num_stations, 'has_topology': True
                }
        
        if patch_ts:
            print(f"Using multi-site patched collation, number of patches: {args.npatch}")
            collate_fn = create_patch_collate_fn(args, device, norm_dict)
        else:
            print("Using multi-site non-patched collation")
            collate_fn = create_nonpatch_collate_fn(args, device, norm_dict)
        
        train_dataloader = DataLoader(train_windows, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_windows, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_windows, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        return {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": 3,
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "n_test_batches": len(test_dataloader),
            "norm_dict": norm_dict,
            "num_stations": num_stations,
            "batch_size": args.batch_size
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_multi_site_time_windows(time_points, all_station_data, station_ids, args, device, split_name):
    seq_len, pred_len = args.history, args.pred_window
    window_len = seq_len + pred_len
    step_size = pred_len
    num_stations, num_features = len(station_ids), 3
    
    time_to_idx = {t: i for i, t in enumerate(time_points)}
    total_timesteps = len(time_points)
    
    data_matrix = torch.full((num_stations, total_timesteps, num_features), float('nan'), device=device)
    mask_matrix = torch.zeros((num_stations, total_timesteps, num_features), device=device)
    
    for station_idx, station_id in enumerate(station_ids):
        sd = all_station_data[station_id]
        for t_idx, t_val in enumerate(sd['times']):
            t = t_val.item()
            if t in time_to_idx:
                idx = time_to_idx[t]
                data_matrix[station_idx, idx] = sd['values'][t_idx]
                mask_matrix[station_idx, idx] = sd['mask'][t_idx]
    
    time_tensor = torch.tensor(time_points, device=device).float()
    windows = []
    for start_idx in range(0, total_timesteps - window_len + 1, step_size):
        end_idx = start_idx + window_len
        t_bias = time_tensor[start_idx].clone()
        windows.append({
            "times": time_tensor[start_idx:end_idx] - t_bias,
            "data": data_matrix[:, start_idx:end_idx, :].clone(),
            "mask": mask_matrix[:, start_idx:end_idx, :].clone(),
            "t_bias": t_bias,
            "station_ids": station_ids
        })
    print(f"{split_name} set created {len(windows)} windows")
    return windows

def compute_normalization_params(train_windows, target_device):
    all_data = torch.cat([w['data'] for w in train_windows], dim=1)
    all_mask = torch.cat([w['mask'] for w in train_windows], dim=1)
    
    data_mean = torch.zeros(3, device=target_device)
    data_std = torch.ones(3, device=target_device)
    
    for i in range(3):
        valid_data = all_data[:, :, i][all_mask[:, :, i] == 1]
        if len(valid_data) > 0:
            data_mean[i] = valid_data.mean().to(target_device)
            data_std[i] = (valid_data.std() + 1e-8).to(target_device)
    
    print(f"Normalization parameters - Mean: {data_mean.cpu().numpy()}, Std: {data_std.cpu().numpy()}")
    return {"data_mean": data_mean, "data_std": data_std}

def create_nonpatch_collate_fn(args, device, norm_dict):
    def collate_fn(batch_windows):
        obs_data, obs_mask, obs_tp = [], [], []
        pre_data, pre_mask, pre_tp = [], [], []
        biases = []
        
        for w in batch_windows:
            biases.append(w['t_bias'])
            obs_data.append(w['data'][:, :args.history, :])
            obs_mask.append(w['mask'][:, :args.history, :])
            obs_tp.append(w['times'][:args.history])
            pre_data.append(w['data'][:, args.history:, :])
            pre_mask.append(w['mask'][:, args.history:, :])
            pre_tp.append(w['times'][args.history:])

        obs_data = torch.stack(obs_data).to(device)
        obs_mask = torch.stack(obs_mask).to(device)
        pre_data = torch.stack(pre_data).to(device)
        pre_mask = torch.stack(pre_mask).to(device)
        obs_tp = torch.stack(obs_tp).to(device) / args.history
        pre_tp = torch.stack(pre_tp).to(device) / args.history
        
        if norm_dict:
            mean, std = norm_dict["data_mean"], norm_dict["data_std"]
            obs_data = (obs_data - mean) / std
            obs_data[obs_mask == 0] = 0.0
            pre_data = (pre_data - mean) / std
            pre_data[pre_mask == 0] = 0.0

        return {
            "observed_data": obs_data, "observed_tp": obs_tp, "observed_mask": obs_mask,
            "data_to_predict": pre_data, "tp_to_predict": pre_tp, "mask_predicted_data": pre_mask,
            "batch_t_bias": torch.stack(biases).to(device) / args.history
        }
    return collate_fn

def create_patch_collate_fn(args, device, norm_dict):
    def collate_fn(batch_windows):
        obs_data, obs_mask, obs_tp = [], [], []
        pre_data, pre_mask, pre_tp = [], [], []
        for w in batch_windows:
            obs_data.append(w['data'][:, :args.history, :])
            obs_mask.append(w['mask'][:, :args.history, :])
            obs_tp.append(w['times'][:args.history])
            pre_data.append(w['data'][:, args.history:, :])
            pre_mask.append(w['mask'][:, args.history:, :])
            pre_tp.append(w['times'][args.history:])
        
        obs_data, obs_mask = torch.stack(obs_data).to(device), torch.stack(obs_mask).to(device)
        pre_data, pre_mask = torch.stack(pre_data).to(device), torch.stack(pre_mask).to(device)
        obs_tp_batch = torch.stack(obs_tp).to(device) / args.history
        pre_tp_batch = torch.stack(pre_tp).to(device) / args.history
        
        if norm_dict:
            mean, std = norm_dict["data_mean"], norm_dict["data_std"]
            obs_data = (obs_data - mean) / std
            obs_data[obs_mask == 0] = 0.0
            pre_data = (pre_data - mean) / std
            pre_data[pre_mask == 0] = 0.0
        
        B, S, T_obs, N = obs_data.shape
        p_size_int = int(args.patch_size)
        actual_npatch = int(args.npatch)
        
        p_size_norm = p_size_int / args.history
        stride_norm = float(args.stride) / args.history
        
        patch_data_batch = torch.zeros((B, S, actual_npatch, p_size_int, N), device=device)
        patch_mask_batch = torch.zeros((B, S, actual_npatch, p_size_int, N), device=device)
        patch_tp_batch = torch.zeros((B, S, actual_npatch, p_size_int), device=device)
        
        for b in range(B):
            t_points = obs_tp_batch[b]
            for i in range(actual_npatch):
                st, ed = i * stride_norm, i * stride_norm + p_size_norm
                inds = torch.where((t_points >= st) & (t_points < (ed if i < actual_npatch-1 else 1.01)))[0]
                if len(inds) > 0:
                    valid_inds = inds[:p_size_int] 
                    L = len(valid_inds)
                    patch_data_batch[b, :, i, :L, :] = obs_data[b, :, valid_inds, :]
                    patch_mask_batch[b, :, i, :L, :] = obs_mask[b, :, valid_inds, :]
                    patch_tp_batch[b, :, i, :L] = t_points[valid_inds]
        
        return {
            "observed_data": patch_data_batch, "observed_tp": patch_tp_batch, "observed_mask": patch_mask_batch,
            "data_to_predict": pre_data, "tp_to_predict": pre_tp_batch, "mask_predicted_data": pre_mask,
            "patch_size": p_size_int, "num_patches": actual_npatch
        }
    return collate_fn