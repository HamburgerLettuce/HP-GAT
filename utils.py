import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
import sklearn as sk
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True   

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False, mode='a'):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode=mode)
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def flatten(x, dim):
    return x.reshape(x.size()[:dim] + (-1, ))

def subsample_timepoints(data, time_steps, mask, n_tp_to_sample = None):
    if n_tp_to_sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_tp_to_sample > 1:
        assert(n_tp_to_sample <= n_tp_in_batch)
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.size(0)):
            missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

            data[i, missing_idx] = 0.
            if mask is not None:
                mask[i, missing_idx] = 0.
    
    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.size(0)):
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n_to_sample = int(n_tp_current * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
            tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

            data[i, tp_to_set_to_zero] = 0.
            if mask is not None:
                mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask

def cut_out_timepoints(data, time_steps, mask, n_points_to_cut = None):
    if n_points_to_cut is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_points_to_cut < 1:
        raise Exception("Number of time points to cut out must be > 1")

    assert(n_points_to_cut <= n_tp_in_batch)
    n_points_to_cut = int(n_points_to_cut)

    for i in range(data.size(0)):
        start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)

        data[i, start : (start + n_points_to_cut)] = 0.
        if mask is not None:
            mask[i, start : (start + n_points_to_cut)] = 0.

    return data, time_steps, mask

def get_device(tensor):
    device = torch.device("cuda:0")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def split_train_test(data, train_fraq = 0.8):
    n_samples = data.size(0)
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]
    return data_train, data_test

def split_train_test_data_and_time(data, time_steps, train_fraq = 0.8):
    n_samples = data.size(0)
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]

    assert(len(time_steps.size()) == 2)
    train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
    test_time_steps = time_steps[:, int(n_samples * train_fraq):]

    return data_train, data_test, train_time_steps, test_time_steps

def get_next_batch(dataloader):
    data_dict = dataloader.__next__()
    return data_dict

def get_ckpt_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    checkpt = torch.load(ckpt_path)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict) 
    model.load_state_dict(state_dict)
    model.to(device)

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def linspace_vector(start, end, n_points):
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        res = torch.linspace(start, end, n_points)
    else:
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, 
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def reverse(tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1, 
    n_units = 100, nonlinear = nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def get_item_from_pickle(pickle_file, item_name):
    from_pickle = load_pickle(pickle_file)
    if item_name in from_pickle:
        return from_pickle[item_name]
    return None

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            }

def normalize_masked_data(data, mask, att_mean, att_std):
    std = att_std + (att_std == 0) * 1e-8
    data_norm = (data - att_mean) / std
    data_norm[mask == 0] = 0
    return data_norm

def normalize_masked_tp(data, att_mean, att_std):
    scale = att_std - att_mean
    scale = scale + (scale == 0) * 1e-8
    if (scale != 0.).all():
        data_norm = (data - att_mean) / scale
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm

def shift_outputs(outputs, first_datapoint = None):
    outputs = outputs[:,:,:-1,:]

    if first_datapoint is not None:
        n_traj, n_dims = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
        outputs = torch.cat((first_datapoint, outputs), 2)
    return outputs

def split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices):
    B = data_dict["data"].shape[0]
    S = data_dict["data"].shape[1]
    T = n_observed_tp
    N = data_dict["data"].shape[-1]
    
    M = len(patch_indices)
    
    patched_data = torch.zeros((B, S, M, args.patch_size, N), device=data_dict["data"].device)
    patched_mask = torch.zeros((B, S, M, args.patch_size, 1), device=data_dict["data"].device)
    patched_time = torch.zeros((B, S, M, args.patch_size, 1), device=data_dict["data"].device)
    
    for s in range(S):
        for i, inds in enumerate(patch_indices):
            if len(inds) > 0:
                patch_data = data_dict["data"][:, s, inds, :]
                patch_mask = data_dict["mask"][:, s, inds, :].sum(dim=-1, keepdim=True)
                patch_time = data_dict["time_steps"][inds].unsqueeze(0).unsqueeze(-1)
                
                if patch_data.shape[1] < args.patch_size:
                    pad_size = args.patch_size - patch_data.shape[1]
                    patch_data = F.pad(patch_data, (0, 0, 0, pad_size), mode='constant', value=0)
                    patch_mask = F.pad(patch_mask, (0, 0, 0, pad_size), mode='constant', value=0)
                    patch_time = F.pad(patch_time, (0, 0, 0, pad_size), mode='constant', value=0)
                
                patched_data[:, s, i, :, :] = patch_data[:, :args.patch_size, :]
                patched_mask[:, s, i, :, :] = patch_mask[:, :args.patch_size, :]
                patched_time[:, s, i, :, :] = patch_time[:, :args.patch_size, :]
    
    data_dict["observed_data"] = patched_data
    data_dict["observed_mask"] = patched_mask
    data_dict["observed_tp"] = patched_time
    
    return data_dict

def split_data_forecast(data_dict, dataset, n_observed_tp):
    device = get_device(data_dict["data"])

    split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
                "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
                "data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
                "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone()}

    split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 

    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()

    split_dict["mode"] = "forecast"

    return split_dict

def split_data_interp(data_dict):
    device = get_device(data_dict["data"])

    split_dict = {"observed_data": data_dict["data"].clone(),
                "observed_tp": data_dict["time_steps"].clone(),
                "data_to_predict": data_dict["data"].clone(),
                "tp_to_predict": data_dict["time_steps"].clone()}

    split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "interp"
    return split_dict

def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict

def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
    if n_tp_to_sample is not None:
        data, time_steps, mask = subsample_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_tp_to_sample = n_tp_to_sample)

    if n_points_to_cut is not None:
        data, time_steps, mask = cut_out_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_points_to_cut = n_points_to_cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["observed_data"] = data.clone()
    new_data_dict["observed_tp"] = time_steps.clone()
    new_data_dict["observed_mask"] = mask.clone()

    if n_points_to_cut is not None:
        new_data_dict["data_to_predict"] = data.clone()
        new_data_dict["tp_to_predict"] = time_steps.clone()
        new_data_dict["mask_predicted_data"] = mask.clone()

    return new_data_dict

def split_and_subsample_batch(data_dict, args, n_observed_tp):
    processed_dict = split_data_forecast(data_dict, args.dataset, n_observed_tp)

    processed_dict = add_mask(processed_dict)

    return processed_dict

def split_and_patch_station_batch(data_dict, args, n_observed_tp, patch_indices):
    device = get_device(data_dict["data"])
    
    B, S, L_obs, N = data_dict["data"].shape
    
    observed_tp = data_dict["time_steps"].clone()
    observed_data = data_dict["data"].clone()
    observed_mask = data_dict["mask"].clone()
    
    M = args.npatch
    
    observed_tp_patches = []
    observed_data_patches = []
    observed_mask_patches = []
    
    for s in range(S):
        station_data = observed_data[:, s, :, :]
        station_mask = observed_mask[:, s, :, :]
        
        station_tp_patches = []
        station_data_patches = []
        station_mask_patches = []
        
        for i in range(M):
            indices = patch_indices[i]
            if len(indices) == 0:
                continue
            
            patch_tp = observed_tp[:, indices]
            patch_data = station_data[:, indices, :]
            patch_mask = station_mask[:, indices, :]
            
            station_tp_patches.append(patch_tp)
            station_data_patches.append(patch_data)
            station_mask_patches.append(patch_mask)
        
        if station_tp_patches:
            station_tp_patches = torch.stack(station_tp_patches, dim=1)
            station_data_patches = torch.stack(station_data_patches, dim=1)
            station_mask_patches = torch.stack(station_mask_patches, dim=1)
            
            observed_tp_patches.append(station_tp_patches)
            observed_data_patches.append(station_data_patches)
            observed_mask_patches.append(station_mask_patches)
    
    if observed_tp_patches:
        observed_tp_patches = torch.stack(observed_tp_patches, dim=1)
        observed_tp_patches = observed_tp_patches.unsqueeze(-1)
        
        observed_data_patches = torch.stack(observed_data_patches, dim=1)
        observed_mask_patches = torch.stack(observed_mask_patches, dim=1)
    else:
        observed_tp_patches = torch.zeros(B, S, M, 1, 1).to(device)
        observed_data_patches = torch.zeros(B, S, M, 1, N).to(device)
        observed_mask_patches = torch.zeros(B, S, M, 1, N).to(device)
    
    data_dict["observed_tp"] = observed_tp_patches
    data_dict["observed_data"] = observed_data_patches
    data_dict["observed_mask"] = observed_mask_patches
    
    if "tp_to_predict" in data_dict:
        data_dict["tp_to_predict"] = data_dict["tp_to_predict"].unsqueeze(1)
    
    return data_dict