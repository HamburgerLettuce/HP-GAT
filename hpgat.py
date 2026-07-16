from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def softmax(src, index):
    N = maybe_num_nodes(index)
    
    max_per_group = scatter(src, index, dim=0, dim_size=N, reduce='max')
    max_per_node = max_per_group[index]
    
    global_out = src - max_per_node
    global_out = global_out.exp()
    
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    
    c = global_out / (global_out_sum + 1e-16)
    return c

class Intra_Inter_Patch_Graph_Layer(MessagePassing):
    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(Intra_Inter_Patch_Graph_Layer, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha

        self.w_k_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) 
            for i in range(self.n_heads)
        ])
        self.bias_k_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) 
            for i in range(self.n_heads)
        ])
        
        self.w_q_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) 
            for i in range(self.n_heads)
        ])
        self.bias_q_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) 
            for i in range(self.n_heads)
        ])
        
        self.w_v_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) 
            for i in range(self.n_heads)
        ])
        self.bias_v_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) 
            for i in range(self.n_heads)
        ])
        
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.uniform_(param)
    
        self.layer_norm = nn.LayerNorm(d_input)
    
    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)
        return self.propagate(
            edge_index, 
            x=x, 
            edges_temporal=edge_value,
            edge_same_time_diff_var=edge_same_time_diff_var,
            edge_diff_time_same_var=edge_diff_time_same_var,
            edge_diff_time_diff_var=edge_diff_time_diff_var,
            n_layer=n_layer, 
            residual=residual
        )
    
    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]
            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]
    
            attention = self.each_head_attention(
                x_j, w_k, bias_k, w_q, bias_q, x_i,
                edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var
            )
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            attention_norm = softmax(attention, edge_index_i)
    
            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
    
            message = attention_norm * sender
            messages.append(message)
    
        message_all_head = torch.cat(messages, 1)
        return message_all_head
    
    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0])
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1])
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2])
        x_i = x_i_0 + x_i_1 + x_i_2
    
        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2
    
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))
        return torch.squeeze(attention, 1)
    
    def update(self, aggr_out, residual):
        return self.res * residual + F.gelu(aggr_out)
    
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class HP_GAT(nn.Module):
    def __init__(self, args, num_sites=None, use_spatial=True):
        super(Hi_Patch, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.S = num_sites if num_sites is not None else 1
        self.batch_size = None
        self.n_layer = args.nlayer
        self.alpha = args.alpha
        self.res = args.res
        self.patch_layer = args.patch_layer
        
        self.use_spatial = use_spatial
        
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.relu = nn.ReLU()
        self.nodevec = nn.Embedding(self.N, d_model)
        
        self.temporal_gcs = nn.ModuleList()
        for l in range(self.n_layer):
            self.temporal_gcs.append(
                Intra_Inter_Patch_Graph_Layer(
                    args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res
                )
            )
        
        if self.use_spatial and self.S > 1:
            self.site_embedding = nn.Embedding(self.S, d_model)
            
            self.spatial_gcs = nn.ModuleList()
            for l in range(min(2, self.n_layer)):
                self.spatial_gcs.append(
                    TopologyAwareSpatialLayer(
                        in_channels=d_model,
                        out_channels=d_model // 2,
                        num_sites=self.S,
                        heads=2,
                        dropout=0.1
                    )
                )
            
            self.upstream_aggregator = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model // 2)
            )
            
            self.var_spatial_fusion_weights = nn.Parameter(torch.ones(self.N, 1))
    
        self.variable_fusion_weights = nn.Parameter(torch.ones(self.N, 1))
        
        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        
        self.topology_info = None
        
        if args.task == 'forecasting':
            self.decoder = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1)
            )
        else:
            print("Task type error! Only 'forecasting' task is supported.")
    
    def set_topology_info(self, topology_info):
        self.topology_info = topology_info
    
    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
    
    def IMTS_Model(self, x, mask_X, x_time):
        B, N, M, L, D = x.shape
        variable_indices = torch.arange(N).to(x.device)
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)
    
        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')
    
        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
        
        int_max = torch.iinfo(torch.int32).max
        element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]
        if element_count > int_max:
            once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
            sd, ed = 0, once_num
            total_num = math.ceil(B / once_num)
            for k in range(total_num):
                if k == 0:
                    edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3 = edge_ind
                elif k == total_num - 1:
                    cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                else:
                    cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                sd += once_num
                ed += once_num
        else:
            edge_ind = torch.where(cur_adj == 1)
    
        source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
        target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])
    
        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])
        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0
    
        for gc in self.temporal_gcs:
            cur_x = gc(
                cur_x, edge_index, edge_time, cur_x_time,
                edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var,
                n_layer=0
            )
    
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)
    
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1
    
        obs_num_per_patch = torch.sum(mask_X, dim=3)
        x_time_per_patch = torch.sum(x_time, dim=3)
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype), obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
        time_te = self.LearnableTE(x_time)
    
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
    
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)
        x_time = avg_x_time
    
        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape
    
            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)
    
            cur_variable_indices = variable_indices.view(1, N, 1, 1)
            cur_variable_indices = cur_variable_indices.expand(B, N, T, 1).reshape(-1, 1)
    
            patch_indices = torch.arange(T).float().to(x.device)
            cur_patch_indices = patch_indices.view(1, 1, T).expand(B, N, T).reshape(B, -1)
    
            missing_indices = torch.where(mask_X.reshape(B, -1) == 0)
    
            patch_indices_matrix_1 = cur_patch_indices.unsqueeze(1).expand(B, N*T, N*T)
            patch_indices_matrix_2 = cur_patch_indices.unsqueeze(-1).expand(B, N*T, N*T)
            patch_interval = patch_indices_matrix_1 - patch_indices_matrix_2
            patch_interval[missing_indices[0], missing_indices[1]] = torch.zeros(len(missing_indices[0]), N*T).to(x.device)
            patch_interval[missing_indices[0], :, missing_indices[1]] = torch.zeros(len(missing_indices[0]), N*T).to(x.device)
    
            edge_ind = torch.where(torch.abs(patch_interval) == 1)
            source_nodes = (N * T * edge_ind[0] + edge_ind[1])
            target_nodes = (N * T * edge_ind[0] + edge_ind[2])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])
    
            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])
            edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0
    
            if edge_index.shape[1] > 0:
                for gc in self.temporal_gcs:
                    cur_x = gc(
                        cur_x, edge_index, edge_time, cur_x_time,
                        edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var,
                        n_layer=n_layer
                    )
                x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)
    
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1
    
            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)
    
            obs_num_per_patch = torch.sum(mask_X, dim=3)
            x_time_per_patch = torch.sum(x_time, dim=3)
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype), obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
            time_te = self.LearnableTE(x_time)
    
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)
    
            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            x_time = avg_x_time
    
        return torch.squeeze(x)
    
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        B, S, M, L_in, N = X.shape
        L_pred = time_steps_to_predict.shape[-1]
        self.batch_size = B
        
        all_site_variable_features = []
        
        for s in range(S):
            site_X = X[:, s, :, :, :]
            site_X_reshaped = site_X.permute(0, 3, 1, 2).unsqueeze(-1)
            site_X_encoded = self.obs_enc(site_X_reshaped)
            
            if len(truth_time_steps.shape) == 5 and truth_time_steps.shape[1] == S:
                site_truth_time = truth_time_steps[:, s, :, :, :]
                site_truth_time = site_truth_time.unsqueeze(-1)
            else:
                site_truth_time = torch.linspace(0, 1, M * L_in, device=X.device).view(1, 1, M, L_in, 1)
                site_truth_time = site_truth_time.expand(B, N, M, L_in, 1)
            
            if mask is None:
                site_mask = torch.ones(B, N, M, L_in, 1).to(X.device)
            else:
                if mask.shape[1] == S:
                    site_mask = mask[:, s, :, :, :]
                    site_mask = site_mask.permute(0, 3, 1, 2)
                    site_mask = site_mask.unsqueeze(-1)
                else:
                    site_mask = torch.ones(B, N, M, L_in, 1).to(X.device)
            
            te_his = self.LearnableTE(site_truth_time)
            var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
            
            site_input = self.relu(site_X_encoded + var_emb + te_his)
            
            try:
                site_variable_features = self.IMTS_Model(site_input, site_mask, site_truth_time)
                
                if len(site_variable_features.shape) == 3:
                    pass
                elif len(site_variable_features.shape) == 4:
                    site_variable_features = torch.mean(site_variable_features, dim=2)
                else:
                    site_variable_features = site_variable_features.view(B, N, -1)
                    site_variable_features = site_variable_features[:, :, :self.hid_dim]
                
            except Exception as e:
                print(f"Site {s} IMTS_Model error: {e}")
                site_variable_features = torch.zeros(B, N, self.hid_dim, device=X.device, requires_grad=True)
            
            all_site_variable_features.append(site_variable_features)
        
        temporal_features = torch.stack(all_site_variable_features, dim=1)
    
        if self.use_spatial and S > 1 and hasattr(self, 'spatial_gcs'):
            fusion_weights = torch.softmax(self.variable_fusion_weights, dim=0)
            weighted_temporal = temporal_features * fusion_weights.view(1, 1, N, 1)
            fused_temporal = torch.sum(weighted_temporal, dim=2)
            
            station_ids = torch.arange(S, device=X.device).unsqueeze(0).expand(B, -1)
            station_feats = self.site_embedding(station_ids)
            node_feats = station_feats + fused_temporal
            
            enhanced_features = node_feats
    
            if self.topology_info and self.topology_info.get('has_topology', False):
                edge_index = self.topology_info['edge_index'].to(X.device)
                edge_weights = self.topology_info['edge_weights'].to(X.device)
            else:
                edge_index = torch.tensor([[0], [0]], device=X.device)
                edge_weights = torch.tensor([1.0], device=X.device)
            
            for spatial_gc in self.spatial_gcs:
                enhanced_features = spatial_gc(
                    enhanced_features, 
                    edge_index, 
                    edge_weights
                )
            
            target_idx = S - 1
            enhanced_target_features = enhanced_features[:, target_idx:target_idx+1, :]
            
            fusion_weights_var = torch.sigmoid(self.var_spatial_fusion_weights)
            
            temporal_target = temporal_features[:, target_idx:target_idx+1, :, :]
            temporal_expanded_for_fusion = temporal_target.permute(0, 2, 1, 3)
            
            if enhanced_target_features.shape[-1] != temporal_expanded_for_fusion.shape[-1]:
                if not hasattr(self, 'spatial_dim_adjust'):
                    self.spatial_dim_adjust = nn.Linear(
                        enhanced_target_features.shape[-1],
                        temporal_expanded_for_fusion.shape[-1]
                    ).to(X.device)
                
                enhanced_target_features_adjusted = self.spatial_dim_adjust(enhanced_target_features)
            else:
                enhanced_target_features_adjusted = enhanced_target_features
            
            spatial_expanded = enhanced_target_features_adjusted.unsqueeze(1).expand(-1, N, 1, -1)
            
            fusion_weights_var_expanded = fusion_weights_var.view(1, N, 1, 1).expand(
                B, N, 1, temporal_expanded_for_fusion.shape[-1]
            )
            
            h_fused = fusion_weights_var_expanded * temporal_expanded_for_fusion + \
                    (1 - fusion_weights_var_expanded) * spatial_expanded
            h_last_var = h_fused[:, -1:, 0, :]
        
        else:
            target_idx = S - 1
            target_temporal = temporal_features[:, target_idx, :, :]
            
            fusion_weights = torch.softmax(self.variable_fusion_weights, dim=0)
            
            weighted_features = target_temporal * fusion_weights.view(1, N, 1)
            fused_features = torch.sum(weighted_features, dim=1)
            
            h_last_var = fused_features.unsqueeze(1)
        
        if len(time_steps_to_predict.shape) == 2:
            pred_times = time_steps_to_predict
        else:
            pred_times = torch.linspace(0, 1, L_pred, device=X.device).unsqueeze(0).repeat(B, 1)
        
        pred_times_emb = pred_times.unsqueeze(-1)
        te_pred = self.LearnableTE(pred_times_emb)
        
        h_last_var_expanded = h_last_var.unsqueeze(2)
        h_last_var_expanded = h_last_var_expanded.expand(B, 1, L_pred, self.hid_dim)
        
        te_pred_expanded = te_pred.unsqueeze(1)
        
        combined_features = torch.cat([h_last_var_expanded, te_pred_expanded], dim=-1)
        
        outputs = self.decoder(combined_features)
        
        outputs = outputs.permute(1, 0, 2, 3)
        
        return outputs
    
    def forward(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        return self.forecasting(time_steps_to_predict, X, truth_time_steps, mask)