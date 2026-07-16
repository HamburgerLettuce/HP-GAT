    import os
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    import lib.utils as utils
    
    class Iowa(object):
    
        def __init__(self, root, n_samples=None, device=torch.device("cpu")):
            self.root = root
            self.device = device
            
            self.process()
            
            if device == torch.device("cpu"):
                self.data = torch.load(os.path.join(self.processed_folder, 'iowa.pt'), map_location='cpu')
            else:
                self.data = torch.load(os.path.join(self.processed_folder, 'iowa.pt'))
                
            if n_samples is not None:
                print('Total records:', len(self.data))
                self.data = self.data[:n_samples]
        
        def process(self):
            filename = os.path.join(self.raw_folder, 'hydrological_data.csv')
            os.makedirs(self.processed_folder, exist_ok=True)
            
            print('Processing Iowa hydrological data: {}'.format(filename))
            
            full_data = pd.read_csv(filename)
            
            full_data = self._validate_and_clean_data(full_data)
            
            entities = []
            
            data_gp = full_data.groupby('ID')
            
            for station_id, data in data_gp:
                data = data.sort_values('Time')
                
                tt = torch.tensor(data['Time'].values).to(self.device).float()
                vals = torch.tensor(data[['Value_0', 'Value_1', 'Value_2']].values).to(self.device).float()
                mask = torch.tensor(data[['Mask_0', 'Mask_1', 'Mask_2']].values).to(self.device).float()
                
                entities.append((station_id, tt, vals, mask))
            
            torch.save(entities, os.path.join(self.processed_folder, 'iowa.pt'))
            print('Total hydrological stations:', len(entities))
            print('Iowa hydrological data processing completed!')
        
        def _validate_and_clean_data(self, data):
            original_count = len(data)
            
            completely_missing = (data['Mask_0'] == 0) & (data['Mask_1'] == 0) & (data['Mask_2'] == 0)
            data = data[~completely_missing]
            
            if completely_missing.sum() > 0:
                print(f"Removed {completely_missing.sum()} completely missing time points")
            
            return data
        
        @property
        def raw_folder(self):
            return os.path.join(self.root, 'raw')
        
        @property
        def processed_folder(self):
            return os.path.join(self.root, 'processed')
            
        def __getitem__(self, index):
            return self.data[index]
        
        def __len__(self):
            return len(self.data)
    
    def Iowa_variable_time_collate_fn(batch, args, device=torch.device("cpu"), 
                                     data_type="train", station_stats=None, time_max=None):
        
        observed_tp = []
        observed_data = []
        observed_mask = []
        predicted_tp = []
        predicted_data = []
        predicted_mask = []
        station_ids = []
        
        for b, (record_id, tt, vals, mask, t_bias, station_id) in enumerate(batch):
            tt = tt.to(device)
            vals = vals.to(device)
            mask = mask.to(device)
            t_bias = t_bias.to(device)
            
            tt = tt + t_bias
            n_observed_tp = torch.lt(tt, args.history).sum()
            
            observed_tp.append(tt[:n_observed_tp])
            observed_data.append(vals[:n_observed_tp])
            observed_mask.append(mask[:n_observed_tp])
            predicted_tp.append(tt[n_observed_tp:])
            predicted_data.append(vals[n_observed_tp:])
            predicted_mask.append(mask[n_observed_tp:])
            station_ids.append(station_id)
        
        observed_tp = pad_sequence(observed_tp, batch_first=True).to(device)
        observed_data = pad_sequence(observed_data, batch_first=True).to(device)
        observed_mask = pad_sequence(observed_mask, batch_first=True).to(device)
        predicted_tp = pad_sequence(predicted_tp, batch_first=True).to(device)
        predicted_data = pad_sequence(predicted_data, batch_first=True).to(device)
        predicted_mask = pad_sequence(predicted_mask, batch_first=True).to(device)
        
        if station_stats is not None:
            normalized_observed_data = []
            normalized_predicted_data = []
            
            for i, station_id in enumerate(station_ids):
                if station_id in station_stats:
                    station_min, station_max = station_stats[station_id]
                    station_min = station_min.to(device)
                    station_max = station_max.to(device)
                    
                    obs_data_i = observed_data[i].unsqueeze(0)
                    obs_mask_i = observed_mask[i].unsqueeze(0)
                    norm_obs_i = utils.normalize_masked_data(obs_data_i, obs_mask_i, 
                                                            att_min=station_min, att_max=station_max)
                    normalized_observed_data.append(norm_obs_i.squeeze(0))
                    
                    pred_data_i = predicted_data[i].unsqueeze(0)
                    pred_mask_i = predicted_mask[i].unsqueeze(0)
                    norm_pred_i = utils.normalize_masked_data(pred_data_i, pred_mask_i,
                                                             att_min=station_min, att_max=station_max)
                    normalized_predicted_data.append(norm_pred_i.squeeze(0))
                else:
                    normalized_observed_data.append(observed_data[i])
                    normalized_predicted_data.append(predicted_data[i])
            
            observed_data = torch.stack(normalized_observed_data, dim=0)
            predicted_data = torch.stack(normalized_predicted_data, dim=0)
        
        time_max = time_max.to(device) if time_max is not None else torch.tensor(1.0).to(device)
        observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
        predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)
        
        return {
            "observed_data": observed_data,
            "observed_tp": observed_tp,
            "observed_mask": observed_mask,
            "data_to_predict": predicted_data,
            "tp_to_predict": predicted_tp,
            "mask_predicted_data": predicted_mask,
            "station_ids": station_ids
        }
    
    
    def Iowa_patch_variable_time_collate_fn(batch, args, device=torch.device("cpu"), 
                                           data_type="train", station_stats=None, time_max=None):
        
        D = batch[0][2].shape[1]
    
        all_times = []
        for b, (record_id, tt, vals, mask, t_bias, station_id) in enumerate(batch):
            tt = tt.to(device)
            all_times.append(tt)
        
        combined_tt = torch.unique(torch.cat(all_times), sorted=True)
        
        combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
        combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
        
        predicted_tp = []
        predicted_data = []
        predicted_mask = []
        batch_t_bias = []
        station_ids = []
        
        for b, (record_id, tt, vals, mask, t_bias, station_id) in enumerate(batch):
            batch_t_bias.append(t_bias)
            station_ids.append(station_id)
            
            indices = torch.searchsorted(combined_tt, tt)
            
            valid_indices = indices < len(combined_tt)
            if valid_indices.any():
                combined_vals[b, indices[valid_indices]] = vals[valid_indices]
                combined_mask[b, indices[valid_indices]] = mask[valid_indices]
        
            tmp_n_observed_tp = torch.lt(tt, args.history).sum()
            predicted_tp.append(tt[tmp_n_observed_tp:])
            predicted_data.append(vals[tmp_n_observed_tp:])
            predicted_mask.append(mask[tmp_n_observed_tp:])
        
        n_observed_tp = torch.lt(combined_tt, args.history).sum()
        observed_tp = combined_tt[:n_observed_tp]
        observed_vals = combined_vals[:, :n_observed_tp]
        observed_mask = combined_mask[:, :n_observed_tp]
        
        predicted_tp = pad_sequence(predicted_tp, batch_first=True)
        predicted_data = pad_sequence(predicted_data, batch_first=True)
        predicted_mask = pad_sequence(predicted_mask, batch_first=True)
        
        time_max = time_max.to(device) if time_max is not None else torch.tensor(1.0).to(device)
        observed_tp = utils.normalize_masked_tp(observed_tp, att_min=0, att_max=time_max)
        predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)
        
        if station_stats is not None:
            normalized_observed_vals = []
            normalized_predicted_data_list = []
            
            for i, station_id in enumerate(station_ids):
                if station_id in station_stats:
                    station_min, station_max = station_stats[station_id]
                    station_min = station_min.to(device)
                    station_max = station_max.to(device)
                    
                    obs_data_i = observed_vals[i].unsqueeze(0)
                    obs_mask_i = observed_mask[i].unsqueeze(0)
                    norm_obs_i = utils.normalize_masked_data(obs_data_i, obs_mask_i,
                                                            att_min=station_min, att_max=station_max)
                    normalized_observed_vals.append(norm_obs_i.squeeze(0))
                    
                    pred_data_i = predicted_data[i].unsqueeze(0)
                    pred_mask_i = predicted_mask[i].unsqueeze(0)
                    norm_pred_i = utils.normalize_masked_data(pred_data_i, pred_mask_i,
                                                             att_min=station_min, att_max=station_max)
                    normalized_predicted_data_list.append(norm_pred_i.squeeze(0))
                else:
                    normalized_observed_vals.append(observed_vals[i])
                    normalized_predicted_data_list.append(predicted_data[i])
            
            observed_vals = torch.stack(normalized_observed_vals, dim=0)
            predicted_data = torch.stack(normalized_predicted_data_list, dim=0)
        
        patch_indices = []
        st, ed = 0, args.patch_size
        for i in range(args.npatch):
            if i == args.npatch - 1:
                inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
            else:
                inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
            patch_indices.append(inds)
            st += args.stride
            ed += args.stride
        
        data_dict = {
            "data": observed_vals,
            "time_steps": observed_tp,
            "mask": observed_mask,
            "data_to_predict": predicted_data,
            "tp_to_predict": predicted_tp,
            "mask_predicted_data": predicted_mask,
            "station_ids": station_ids
        }
        
        data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
        
        batch_t_bias = torch.stack(batch_t_bias)
        batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min=0, att_max=time_max)
        
        data_dict["observed_tp"] = data_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1, 1)
        data_dict["tp_to_predict"] = data_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
        data_dict["tp_to_predict"][data_dict["mask_predicted_data"].sum(dim=-1) < 1e-8] = 0
        
        return data_dict
    
    
    def Iowa_get_seq_length(args, records):
        max_input_len = 0
        max_pred_len = 0
        lens = []
        
        for b, (record_id, tt, vals, mask, t_bias, station_id) in enumerate(records):
            n_observed_tp = torch.lt(tt, args.history).sum()
            max_input_len = max(max_input_len, n_observed_tp)
            max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
            lens.append(n_observed_tp)
        
        lens = torch.stack(lens, dim=0)
        median_len = lens.median()
        
        return max_input_len, max_pred_len, median_len