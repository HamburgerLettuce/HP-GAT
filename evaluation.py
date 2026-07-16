import torch
import lib.utils as utils

def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):

​    if not isinstance(truth, torch.Tensor):
​        truth = torch.tensor(truth)
​    if not isinstance(pred_y, torch.Tensor):
​        pred_y = torch.tensor(pred_y)
​    if not isinstance(mask, torch.Tensor):
​        mask = torch.tensor(mask)
​    


    if len(pred_y.shape) == 4 and pred_y.shape[0] == 1:
        pred_y_adjusted = pred_y.squeeze(0)  # (B, L_pred, 1)
    else:
        pred_y_adjusted = pred_y
    
    last_station_idx = 30  
    flow_var_idx = 2       
    
    truth_flow = truth[:, last_station_idx, :, flow_var_idx].unsqueeze(-1)  # (B, L_pred, 1)
    mask_flow = mask[:, last_station_idx, :, flow_var_idx].unsqueeze(-1)    # (B, L_pred, 1)
    
    if len(truth_flow.shape) == 3:
        truth_flow = truth_flow.unsqueeze(0)      # (1, B, L_pred, 1)
        pred_y_adjusted = pred_y_adjusted.unsqueeze(0)  # (1, B, L_pred, 1)
        mask_flow = mask_flow.unsqueeze(0)        # (1, B, L_pred, 1)


​    
    if norm_dict is not None and "data_mean" in norm_dict and "data_std" in norm_dict:
        data_mean = norm_dict["data_mean"]
        data_std = norm_dict["data_std"]
        
        data_std_last = data_std[flow_var_idx]  
        data_mean_last = data_mean[flow_var_idx] 
        
        truth_flow = truth_flow * data_std_last + data_mean_last
        pred_y_adjusted = pred_y_adjusted * data_std_last + data_mean_last


​    
    if func == "MSE":
        error = ((truth_flow - pred_y_adjusted) ** 2) * mask_flow
    elif func == "MAE":
        error = torch.abs(truth_flow - pred_y_adjusted) * mask_flow
    elif func == "MAPE":
        mask_flow = (truth_flow != 0) * mask_flow
        truth_div = truth_flow + (truth_flow == 0) * 1e-8
        error = torch.abs(truth_flow - pred_y_adjusted) / truth_div * mask_flow
    elif func == "RMSE":
     
        mse = ((truth_flow - pred_y_adjusted) ** 2) * mask_flow
        error_sum = mse.sum()
        mask_count = mask_flow.sum()
        if mask_count > 0:
            mse_value = error_sum / mask_count
            error = torch.sqrt(mse_value)
        else:
            error = torch.tensor(0.0).to(truth.device)
    
        if reduce == "mean":
            return error
        elif reduce == "sum":
            return error * mask_count, mask_count
    else:
        raise Exception("Error function not specified")
    
    error_sum = error.sum()
    mask_count = mask_flow.sum()
    
    if reduce == "mean":
        result = error_sum / (mask_count + 1e-8)
        return result
    elif reduce == "sum":
        return error_sum, mask_count
    else:
        raise Exception("Reduce argument not specified!")

def compute_kge(truth, pred_y, mask, norm_dict=None):

​    if len(pred_y.shape) == 4 and pred_y.shape[0] == 1:
​        pred_y = pred_y.squeeze(0)  # (B, T, 1)
​    


    last_station_idx = 30  
    flow_var_idx = 2       
    
    truth_flow = truth[:, last_station_idx, :, flow_var_idx]  # (B, T) 
    pred_flow = pred_y[..., 0]   # (B, T) 
    mask_flow = mask[:, last_station_idx, :, flow_var_idx]    # (B, T)


​    
    if norm_dict is not None and "data_mean" in norm_dict and "data_std" in norm_dict:
        data_mean = norm_dict["data_mean"]
        data_std = norm_dict["data_std"]
        
        data_std_last = data_std[flow_var_idx] 
        data_mean_last = data_mean[flow_var_idx] 


​    
        truth_flow = truth_flow * data_std_last + data_mean_last
        pred_flow = pred_flow * data_std_last + data_mean_last
    
    mask_indices = mask_flow > 0
    truth_masked = truth_flow[mask_indices]
    pred_masked = pred_flow[mask_indices]


​    
    if len(truth_masked) < 2:
        return torch.tensor(-1.0).to(truth.device)


​    
    if len(truth_masked) > 1:
        corr_matrix = torch.corrcoef(torch.stack([truth_masked, pred_masked]))
        r = corr_matrix[0, 1]
    else:
        r = torch.tensor(0.0).to(truth.device)
    
    alpha = torch.std(pred_masked) / (torch.std(truth_masked) + 1e-8)
    beta = torch.mean(pred_masked) / (torch.mean(truth_masked) + 1e-8)


​    
    kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge

def compute_nse(truth, pred_y, mask, norm_dict=None):


​    
    last_station_idx = 30  
    flow_var_idx = 2       
    
    truth_flow = truth[:, last_station_idx, :, flow_var_idx]  # (B, T) 
    pred_flow = pred_y[..., 0]   # (B, T)
    mask_flow = mask[:, last_station_idx, :, flow_var_idx]    # (B, T) 
    
    if norm_dict is not None and "data_mean" in norm_dict and "data_std" in norm_dict:
        data_mean = norm_dict["data_mean"]
        data_std = norm_dict["data_std"]


​       
        data_std_last = data_std[flow_var_idx]  
        data_mean_last = data_mean[flow_var_idx]  
    
        truth_flow = truth_flow * data_std_last + data_mean_last
        pred_flow = pred_flow * data_std_last + data_mean_last
    
    mask_indices = mask_flow > 0
    truth_masked = truth_flow[mask_indices]
    pred_masked = pred_flow[mask_indices]
    
    if len(truth_masked) < 2:
        return torch.tensor(-1.0).to(truth.device)
    
    numerator = torch.sum((truth_masked - pred_masked)**2)
    denominator = torch.sum((truth_masked - torch.mean(truth_masked))**2)


​    
    nse = 1 - numerator / (denominator + 1e-8)
    
    return nse

def compute_all_losses(model, batch_dict, norm_dict=None, phase="train"):

​    pred_y = model.forecasting(
​        batch_dict["tp_to_predict"],
​        batch_dict["observed_data"],
​        batch_dict["observed_tp"],
​        batch_dict["observed_mask"]
​    )


    use_denorm = (phase in ["val", "test"]) and (norm_dict is not None)
    
    mse = compute_error(
        batch_dict["data_to_predict"], pred_y, 
        mask=batch_dict["mask_predicted_data"], 
        func="MAE", reduce="mean",
        norm_dict=norm_dict if use_denorm else None
    )


​    
    rmse = compute_error(
        batch_dict["data_to_predict"], pred_y,
        mask=batch_dict["mask_predicted_data"], 
        func="RMSE", reduce="mean",
        norm_dict=norm_dict if use_denorm else None
    )
    
    mae = compute_error(
        batch_dict["data_to_predict"], pred_y,
        mask=batch_dict["mask_predicted_data"], 
        func="MAE", reduce="mean",
        norm_dict=norm_dict if use_denorm else None
    )


​    
    kge = compute_kge(
        batch_dict["data_to_predict"], pred_y, 
        batch_dict["mask_predicted_data"],
        norm_dict=norm_dict if use_denorm else None
    )
    nse = compute_nse(
        batch_dict["data_to_predict"], pred_y, 
        batch_dict["mask_predicted_data"],
        norm_dict=norm_dict if use_denorm else None
    )


​    
    loss = compute_error(
        batch_dict["data_to_predict"], pred_y, 
        mask=batch_dict["mask_predicted_data"], 
        func="MSE", reduce="mean",
        norm_dict=None
    )


​    
    results = {
        "loss": loss,
        "mse": mse.item() if isinstance(mse, torch.Tensor) else mse,
        "rmse": rmse.item() if isinstance(rmse, torch.Tensor) else rmse,
        "mae": mae.item() if isinstance(mae, torch.Tensor) else mae,
        "kge": kge.item() if isinstance(kge, torch.Tensor) else kge,
        "nse": nse.item() if isinstance(nse, torch.Tensor) else nse
    }
    
    return results

def evaluation(model, dataloader, n_batches, norm_dict=None):

​    n_eval_samples = 0
​    n_eval_samples_mape = 0
​    total_results = {
​        "loss": 0, "mse": 0, "mae": 0, "rmse": 0, 
​        "mape": 0, "kge": 0, "nse": 0
​    }
​    


    all_truth_flat = []
    all_pred_flat = []
    all_mask_flat = []


​    
    for i in range(n_batches):
    
        batch_dict = utils.get_next_batch(dataloader)


​    
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"], 
            batch_dict["observed_tp"],
            batch_dict["observed_mask"]
        )


​    
        if len(pred_y.shape) == 4 and pred_y.shape[0] == 1:
            pred_y = pred_y.squeeze(0)  # (B, T, 1)


​    
        last_station_idx = 18  
        flow_var_idx = 2       
    
        truth_flow = batch_dict["data_to_predict"][:, last_station_idx, :, flow_var_idx]  # (B, T) 
        pred_flow = pred_y[..., 0]  # (B, T) 
        mask_flow = batch_dict["mask_predicted_data"][:, last_station_idx, :, flow_var_idx]  # (B, T) 
    
        truth_flat = truth_flow.flatten()
        pred_flat = pred_flow.flatten()
        mask_flat = mask_flow.flatten()
        
        all_truth_flat.append(truth_flat)
        all_pred_flat.append(pred_flat)
        all_mask_flat.append(mask_flat)


​    
        se_sum, mask_count = compute_error(
            batch_dict["data_to_predict"], pred_y,
            mask=batch_dict["mask_predicted_data"], 
            func="MSE", reduce="sum",
            norm_dict=norm_dict
        )
    
        ae_sum, _ = compute_error(
            batch_dict["data_to_predict"], pred_y, 
            mask=batch_dict["mask_predicted_data"],
            func="MAE", reduce="sum",
            norm_dict=norm_dict
        )


​    
        rmse_sum, rmse_mask_count = compute_error(
            batch_dict["data_to_predict"], pred_y,
            mask=batch_dict["mask_predicted_data"], 
            func="RMSE", reduce="sum",
            norm_dict=norm_dict
        )
    
        ape_sum, mask_count_mape = compute_error(
            batch_dict["data_to_predict"], pred_y,
            mask=batch_dict["mask_predicted_data"], 
            func="MAPE", reduce="sum",
            norm_dict=norm_dict
        )


​    
        total_results["loss"] += ae_sum
        total_results["mse"] += se_sum
        total_results["mae"] += ae_sum
        total_results["rmse"] += rmse_sum
        total_results["mape"] += ape_sum
        n_eval_samples += mask_count
        n_eval_samples_mape += mask_count_mape


​    
    if n_eval_samples > 0:
        total_results["loss"] = total_results["loss"] / n_eval_samples
        total_results["mse"] = total_results["mse"] / n_eval_samples
        total_results["mae"] = total_results["mae"] / n_eval_samples
        total_results["rmse"] = total_results["rmse"] / n_eval_samples
        total_results["mape"] = total_results["mape"] / (n_eval_samples_mape + 1e-8)


​    
    if all_truth_flat:
    
        truth_tensor = torch.cat(all_truth_flat, dim=0)  # (total_samples,)
        pred_tensor = torch.cat(all_pred_flat, dim=0)    # (total_samples,)
        mask_tensor = torch.cat(all_mask_flat, dim=0)    # (total_samples,)


​    
        mask_indices = mask_tensor > 0
        truth_masked = truth_tensor[mask_indices]
        pred_masked = pred_tensor[mask_indices]


​    
        if len(truth_masked) >= 2:
    
            if norm_dict is not None and "data_mean" in norm_dict and "data_std" in norm_dict:
                data_mean = norm_dict["data_mean"]
                data_std = norm_dict["data_std"]
                
                flow_var_idx = 2  
                data_std_last = data_std[flow_var_idx]  
                data_mean_last = data_mean[flow_var_idx]  
                
                truth_masked = truth_masked * data_std_last + data_mean_last
                pred_masked = pred_masked * data_std_last + data_mean_last


​         
            if len(truth_masked) > 1:
                corr_matrix = torch.corrcoef(torch.stack([truth_masked, pred_masked]))
                r = corr_matrix[0, 1]
            else:
                r = torch.tensor(0.0).to(truth_masked.device)
            
            alpha = torch.std(pred_masked) / (torch.std(truth_masked) + 1e-8)
            beta = torch.mean(pred_masked) / (torch.mean(truth_masked) + 1e-8)
            kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            # 计算NSE
            numerator = torch.sum((truth_masked - pred_masked)**2)
            denominator = torch.sum((truth_masked - torch.mean(truth_masked))**2)
            nse = 1 - numerator / (denominator + 1e-8)
            
            total_results["kge"] = kge.item() if isinstance(kge, torch.Tensor) else kge
            total_results["nse"] = nse.item() if isinstance(nse, torch.Tensor) else nse
        else:
            total_results["kge"] = -1.0
            total_results["nse"] = -1.0


​    
    for key, var in total_results.items():
        if isinstance(var, torch.Tensor):
            var = var.item()
        total_results[key] = var
    
    return total_results