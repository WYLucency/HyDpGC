import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
# from gcn import GCN

class PoincareEdgeGenerator(nn.Module):
    def __init__(self, manifold, n, d, c=1.0, batch_size=1024):
        super(PoincareEdgeGenerator, self).__init__()
        self.n = n
        self.d = d
        self.c = c
        # self.manifold = geoopt.PoincareBall(c=c)
        self.manifold = manifold
        self.batch_size = batch_size
        # Initialize node embeddings in hyperbolic space
        self.embeddings = geoopt.ManifoldParameter(
            torch.randn(n, d) * 0.01,
            manifold=self.manifold
        )
        
    def forward(self):
        # Check if we need to use batch processing (for large datasets)
        # Use 10000 as threshold (ogb-arxiv has many nodes)
        if self.n > 10000 or self.n > 1000 and self.d > 100:
            return self.forward_batch()
        else:
            return self.forward_full()
    
    def forward_full(self):
        """Original implementation: compute all distances at once"""
        # Compute hyperbolic distances between all pairs of nodes
        dists = self.manifold.dist(self.embeddings.unsqueeze(1), 
                                 self.embeddings.unsqueeze(0))
        
        # Convert distances to probabilities using a sigmoid
        probs = torch.sigmoid(-dists)
        
        # Zero out diagonal
        probs = probs * (1 - torch.eye(self.n, device=probs.device))
        
        return probs, dists
        
    def forward_batch(self):
        """Batch implementation: compute distances in smaller batches"""
        # Prepare tensors to store results
        dists = torch.zeros(self.n, self.n, device=self.embeddings.device)
        
        # 检测可用的GPU数量
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            # 使用两个GPU进行分布式计算
            device_map = {0: 'cuda:0', 1: 'cuda:1'}
            # print(f"PoincareEdgeGenerator: 使用双GPU计算距离矩阵")
            
            # 将embeddings复制到两个GPU
            embeddings_0 = self.embeddings.to(device_map[0])
            embeddings_1 = self.embeddings.to(device_map[1])
            
            # 计算操作索引
            operation_counter = 0
            
            # Process in batches to save memory
            for i in range(0, self.n, self.batch_size):
                end_i = min(i + self.batch_size, self.n)
                
                for j in range(0, self.n, self.batch_size):
                    end_j = min(j + self.batch_size, self.n)
                    
                    # 决定使用哪个GPU (根据矩阵位置或操作索引)
                    gpu_id = operation_counter % 2  # 或者使用: 0 if i <= j else 1
                    
                    # 在选定的GPU上进行计算
                    if gpu_id == 0:
                        batch_i = embeddings_0[i:end_i].unsqueeze(1)
                        batch_j = embeddings_0[j:end_j].unsqueeze(0)
                    else:
                        batch_i = embeddings_1[i:end_i].unsqueeze(1)
                        batch_j = embeddings_1[j:end_j].unsqueeze(0)
                    
                    # 计算当前批次的距离
                    with torch.cuda.device(gpu_id):
                        batch_dists = self.manifold.dist(batch_i, batch_j)
                    
                    # 将结果传回原始设备
                    dists[i:end_i, j:end_j] = batch_dists.to(self.embeddings.device)
                    
                    # 清理当前GPU缓存
                    torch.cuda.empty_cache()
                    operation_counter += 1
        else:
            # 原始的单GPU实现
            # Process in batches to save memory
            for i in range(0, self.n, self.batch_size):
                end_i = min(i + self.batch_size, self.n)
                batch_i = self.embeddings[i:end_i]
                
                for j in range(0, self.n, self.batch_size):
                    end_j = min(j + self.batch_size, self.n)
                    batch_j = self.embeddings[j:end_j]
                    
                    # Compute distances for current batch
                    batch_dists = self.manifold.dist(
                        batch_i.unsqueeze(1),
                        batch_j.unsqueeze(0)
                    )
                    
                    # Store result in the corresponding part of the output tensor
                    dists[i:end_i, j:end_j] = batch_dists
                torch.cuda.empty_cache()
                
        # Convert distances to probabilities using a sigmoid
        probs = torch.sigmoid(-dists)
        
        # Zero out diagonal
        probs = probs * (1 - torch.eye(self.n, device=probs.device))
        
        return probs, dists
    
    @torch.no_grad()
    def inference(self):
        # self.eval()
        adj_syn, dists = self.forward()
        return adj_syn
    
# class HyperbolicGCond(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, 
#                  n_cond, c=1.0, device=None, args=None):
#         super(HyperbolicGCond, self).__init__()
#         self.c = c
#         self.manifold = geoopt.PoincareBall(c=c)
#         self.n_cond = n_cond
        
#         # Edge generator using Poincaré ball model
#         self.edge_generator = PoincareEdgeGenerator(n_cond, hidden_channels, c)
        
#         # Feature generator
#         self.feature_generator = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, in_channels)
#         )
        
#         # Target GNN
#         # self.target_gnn = GCN(in_channels, hidden_channels, out_channels)
#         self.target_gnn = GCN(in_channels, hidden_channels, out_channels,
#                               dropout=args.dropout, with_bn=False,
#                               weight_decay=0e-4, nlayers=args.nlayers,
#                               nclass=args.nclass,
#                               device=device).to(device)
        
#         self.multi_label = None
#     # def forward(self, x, edge_index):
#     #     # Project original features to hyperbolic space
#     #     x_hyp = self.manifold.expmap0(x)
        
#     #     # Generate synthetic features in hyperbolic space
#     #     x_cond_hyp = self.embeddings
        
#     #     # Project back to Euclidean space for GNN
#     #     x_cond = self.manifold.logmap0(x_cond_hyp)
#     #     x_cond = self.feature_generator(x_cond)
        
#     #     # Compute geodesic distances
#     #     dists_cond = self.manifold.dist(x_cond_hyp.unsqueeze(1), 
#     #                                   x_cond_hyp.unsqueeze(0))
#     #     dists_orig = self.manifold.dist(x_hyp.unsqueeze(1), 
#     #                                   x_hyp.unsqueeze(0))
        
#     #     # Get predictions from both graphs
#     #     out_cond = self.target_gnn(x_cond, edge_index)
#     #     out_orig = self.target_gnn(x, edge_index)
        
#     #     return out_cond, out_orig, dists_cond, dists_orig
#     def forward_sampler(self, x, adjs):
#         x_hyp = self.manifold.expmap0(x)
#         out = self.target_gnn.forward_hyper_sampler(x_hyp, adjs)
#         out = self.manifold.logmap0(out)
#         if self.multi_label:
#             return torch.sigmoid(out)
#         else:
#             return F.log_softmax(out, dim=1)

#     def forward(self, x, adj):
#         out = self.target_gnn.forward_hyper(x, adj)
#         x_cond = self.manifold.logmap0(out)
#         x_cond = self.feature_generator(x_cond)

#         if self.multi_label:
#             return torch.sigmoid(x_cond)
#         else:
#             return F.log_softmax(x_cond, dim=1)