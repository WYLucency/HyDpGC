import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_sparse import matmul

from torch_geometric.utils import dense_to_sparse
from dataset import *
from evaluation import *
from evaluation.utils import *
from models import *
from torch_sparse import SparseTensor
from dataset.convertor import ei2csr
from utils import accuracy, seed_everything, normalize_adj_tensor, to_tensor, is_sparse_tensor, is_identity, \
    f1_macro


class Evaluator:
    """
    A class to evaluate different models and their hyperparameters on graph data.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments and configuration parameters.
    **kwargs : keyword arguments
        Additional parameters.
    """

    def __init__(self, args, **kwargs):
        """
        Initializes the Evaluator with given arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments and configuration parameters.
        **kwargs : keyword arguments
            Additional parameters.
        """
        self.args = args
        self.device = args.device
        self.reset_parameters()
        self.metric = args.metric

    def reset_parameters(self):
        """
        Initializes or resets model parameters.
        """
        pass

    def grid_search(self, data, model_type, param_grid, reduced=True):
        """
        Performs a grid search over hyperparameters.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model used for evaluation.
        param_grid : dict
            A dictionary containing parameter grids for grid search.
        reduced : bool, optional, default=True
            Whether to use synthetic data.

        Returns
        -------
        best_test_result : tuple
            Best test result as (mean_accuracy, std_accuracy).
        best_params : dict
            Best parameters found during grid search.
        """
        args = self.args
        best_val_result = None
        best_test_result = None
        best_params = None

        for params in tqdm(ParameterGrid(param_grid)):
            for key, value in params.items():
                setattr(args, key, value)

            res = []
            for i in range(args.run_eval):
                seed_everything(i)
                res.append([self.test(data, model_type=model_type, verbose=False, reduced=reduced, mode='cross')])
                torch.cuda.empty_cache()

            res = np.array(res).reshape(args.run_eval, -1)
            res_mean, res_std = res.mean(axis=0), res.std(axis=0)

            if args.verbose:
                print(
                    f'{model_type} Test results with params {params}: {100 * res_mean[1]:.2f} +/- {100 * res_std[1]:.2f}')

            if best_val_result is None or res_mean[0] > best_val_result[0]:
                best_val_result = (res_mean[0], res_std[0])
                best_test_result = (res_mean[1], res_std[1])
                best_params = params

        return best_test_result, best_params

    def train_cross(self, data, grid_search=True, reduced=True):
        """
        Trains models and performs grid search if required.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        grid_search : bool, optional, default=True
            Whether to perform grid search over hyperparameters.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        """
        args = self.args

        if grid_search:
            gs_params = {
                'GCN': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5]},
                'SGC': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5], 'ntrans': [1, 2]},
                'APPNP': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5], 'ntrans': [1, 2], 'alpha': [0.1, 0.2]},
                'Cheby': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                          'dropout': [0.0, 0.5]},
                'GraphSage': {'hidden': [64, 256], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                              'dropout': [0.0, 0.5]},
                'GAT': {'hidden': [16, 64], 'lr': [0.01, 0.001], 'weight_decay': [0, 5e-4],
                        'dropout': [0.0, 0.5, 0.7]},
                'SGFormer': {'trans_num_layers': [1, 2, 3], 'lr': [0.01, 0.001], 'trans_weight_decay': [0.001, 0.0001],
                             'trans_dropout': [0.0, 0.5, 0.7]}
            }

            if args.dataset in ['reddit']:
                gs_params['GAT']['hidden'] = [8, 16]

            for model_type in gs_params:
                if reduced:
                    data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                                verbose=args.verbose)
                print(f'Starting Grid Search for {model_type}')
                best_result, best_params = self.grid_search(data, model_type, gs_params[model_type], reduced=reduced)
                args.logger.info(
                    f'Best {model_type} Test Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f} with params {best_params}')
        else:
            eval_model_list = ['GCN', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
            for model_type in eval_model_list:
                data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                            verbose=args.verbose)
                best_result = self.evaluate(data, model_type=args.eval_model)
                args.logger.info(
                    f'{model_type} Result: {100 * best_result[0]:.2f} +/- {100 * best_result[1]:.2f}')

    def test(self, data, model_type, verbose=True, reduced=True, mode='eval', MIA=False):
        """
        Tests a model and returns accuracy and loss.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to test.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        best_acc_val : float
            Best accuracy on validation set.
        acc_test : float
            Accuracy on test set.
        """
        args = self.args

        if verbose:
            print(f'======= testing {model_type}')

        model = eval(model_type)(data.feat_full.shape[1], args.hidden, data.nclass, args, mode=mode).to(self.device)
        best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                          setting=args.setting, reduced=reduced)

        model.eval()
        labels_test = data.labels_test.long().to(args.device)
        labels_train = data.labels_train.long().to(args.device)

        if args.attack is not None:
            data = attack(data, args)

        if args.setting == 'ind':
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = self.metric(output, labels_test).item()
            if MIA:
                output_train = model.predict(data.feat_train, data.adj_train)
                conf_train = F.softmax(output_train, dim=1)
                conf_test = F.softmax(output, dim=1)

            if verbose:
                print("Test set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test:.4f}")
        else:
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = self.metric(output[data.idx_test], labels_test).item()
            if MIA:
                conf_train = F.softmax(output[data.idx_train], dim=1)
                conf_test = F.softmax(output[data.idx_test], dim=1)
            if verbose:
                print("Test full set results:",
                      f"loss= {loss_test.item():.4f}",
                      f"accuracy= {acc_test:.4f}")
        if MIA:
            mia_acc = inference_via_confidence(conf_train.cpu().numpy(), conf_test.cpu().numpy(), labels_train.cpu(),
                                               labels_test.cpu())
            # print(f"MIA accuracy: {mia_acc}")
            return best_acc_val, acc_test, mia_acc
        return best_acc_val, acc_test

    def evaluate(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        """
        Evaluates a model over multiple runs and returns mean and standard deviation of accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        mean_acc : float
            Mean accuracy over multiple runs.
        std_acc : float
            Standard deviation of accuracy over multiple runs.
        """

        args = self.args

        # Prepare synthetic data if required
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            print(f'Evaluating reduced data using {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        # Collect accuracy results from multiple runs
        res = []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            _, best_acc = self.test(data, model_type=model_type, verbose=args.verbose, reduced=reduced,
                                    mode=mode)
            res.append(best_acc)
            if verbose:
                run_evaluation.set_postfix(test_acc=best_acc)

        res = np.array(res)

        # Log and return mean and standard deviation of accuracy
        args.logger.info(f'Seed:{args.seed}, Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')
        return res.mean(), res.std()

    def MIA_evaluate(self, data, model_type, verbose=True, reduced=True, mode='eval'):
        """
        Evaluates a model over multiple runs and returns mean and standard deviation of accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=True
            Whether to print detailed logs.
        reduced : bool, optional, default=True
            Whether to use synthetic data.
        mode : str, optional, default='eval'
            The mode for the model (e.g., 'eval' or 'cross').

        Returns
        -------
        mean_acc : float
            Mean accuracy over multiple runs.
        std_acc : float
            Standard deviation of accuracy over multiple runs.
        """

        args = self.args

        # Prepare synthetic data if required
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            print(f'Evaluating reduced data using {model_type}')
            run_evaluation = trange(args.run_eval)
        else:
            run_evaluation = range(args.run_eval)

        # Collect accuracy results from multiple runs
        res = []
        mia_res = []
        for i in run_evaluation:
            seed_everything(args.seed + i)
            _, best_acc, mia_acc = self.test(data, model_type=model_type, verbose=args.verbose, reduced=reduced,
                                             mode=mode, MIA=True)
            res.append(best_acc)
            mia_res.append(mia_acc)
            if verbose:
                run_evaluation.set_postfix(test_acc=best_acc,MIA_acc=mia_acc)

        res = np.array(res)
        mia_res = np.array(mia_res)

        # Log and return mean and standard deviation of accuracy
        args.logger.info(f'Seed:{args.seed}, Test Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}, '
                         f'MIA Accuracy: {100 * mia_res.mean():.2f} +/- {100 * mia_res.std():.2f}')
        return res.mean(), res.std()

    def GradReconAttack_evaluate(self, data, reduced=True, model_type='GCN', n_runs=5, verbose=True, mode='eval'):
        """评估模型对梯度重构攻击的防御效果
        
        Args:
            data: 数据对象
            reduced: 是否使用压缩后的图
            model_type: 模型类型
            n_runs: 运行次数
        """
        print('\nEvaluating Gradient Reconstruction Attack...')
        args = self.args
        # 准备数据
        if reduced:
            data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, args, model_type=model_type,
                                                                        verbose=verbose)
            # 在压缩图上进行攻击，但目标是恢复原始图信息
            target_features = data.feat_syn.to(self.device)
            target_adj = data.adj_syn.to(self.device)
            target_labels = data.labels_syn.to(self.device)
        else:
            features = data.feat_full
            adj = data.adj_full
            labels = data.labels_full


        # 保存原始图的信息
        original_features = data.feat_full.to(self.device)
        original_adj = data.adj_full
        original_labels = data.labels_full.to(self.device)
        

        # 初始化结果存储
        feature_recovery = []
        structure_recovery = []
        label_recovery = []
        
        for run in range(n_runs):
            # 1. 训练目标模型
            model = eval(model_type)(target_features.shape[1], self.args.hidden, data.nclass, self.args).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # 2. 收集梯度信息
            model.train()
            optimizer.zero_grad()
            output = model(target_features, target_adj)
            loss = F.nll_loss(output, target_labels)
            loss.backward()
            
            # 3. 收集攻击所需的梯度信息
            gradients = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] = param.grad.data.clone()
        
            # 4. 进行梯度重构攻击
            # 4.1 特征恢复攻击
            feature_subspace = None
            for name, grad in gradients.items():
                if 'weight' in name and len(grad.shape) == 2 and grad.shape[0] == target_features.shape[1]:
                    # 找到特征维度匹配的第一层权重梯度
                    try:
                        # 提取特征子空间
                        U, S, V = torch.svd(grad)
                        k = min(20, U.shape[1])  # 取前k个主成分
                        feature_subspace = U[:, :k]
                        break
                    except Exception as e:
                        print(f"SVD计算错误: {e}")
            
            if feature_subspace is not None:
                # 4.2 计算压缩图特征子空间与原始图特征的关系
                
                # 计算原始特征在估计子空间上的投影
                original_proj = torch.mm(original_features, feature_subspace)
                reconstructed_features = torch.mm(original_proj, feature_subspace.t())
                
                # 计算重构误差
                recon_error = F.mse_loss(reconstructed_features, original_features).item()
                
                # 计算特征恢复得分 (与原始特征的相似度)
                # 归一化为0-1，值越高表示恢复程度越高
                max_error = torch.var(original_features) * original_features.shape[1]
                feature_score = max(0, 1.0 - recon_error / max_error)
                feature_recovery.append(feature_score)
                
                # 4.3 结构恢复攻击
                # 使用权重和梯度尝试恢复图的邻接信息
                if 'weight' in list(gradients.keys())[0]:
                    # 获取最后一层的梯度
                    last_layer_grads = None
                    for name, grad in gradients.items():
                        if 'weight' in name and grad.shape[1] == data.nclass:
                            last_layer_grads = grad
                            break
                    
                    if last_layer_grads is not None and target_adj.shape[0] < 1000:  # 对小图执行
                        # 使用梯度相似性估计节点间的连接
                        grad_sim = torch.mm(last_layer_grads, last_layer_grads.t())
                        
                        # 转换为稀疏表示
                        if isinstance(original_adj, torch.sparse.Tensor):
                            true_adj = original_adj.to_dense()
                        else:
                            true_adj = original_adj
                        
                        # 对大于阈值的相似度标记为边
                        thresh = torch.quantile(grad_sim.flatten(), 0.8)  # 取top 20%作为边
                        est_adj = (grad_sim > thresh).float()
                        
                        # 计算与原始邻接矩阵的重叠率
                        if true_adj.shape[0] == est_adj.shape[0]:
                            overlap = (est_adj * true_adj).sum() / true_adj.sum()
                            structure_score = overlap.item()
                            structure_recovery.append(structure_score)
                        else:
                            # 形状不匹配时，计算随机采样的重叠率
                            n_sample = min(true_adj.shape[0], est_adj.shape[0])
                            idx = torch.randperm(true_adj.shape[0])[:n_sample]
                            sampled_true = true_adj[idx][:, idx].toarray()
                            sampled_est = est_adj[:n_sample, :n_sample]
                            overlap = (sampled_est * sampled_true).sum() / sampled_true.sum()
                            structure_score = overlap.item()
                            structure_recovery.append(structure_score)
                    else:
                        structure_recovery.append(0.0)
                else:
                    structure_recovery.append(0.0)
                    
                # 4.4 标签恢复攻击
                # 使用最后一层梯度推断标签分布
                last_layer_grads = None
                for name, grad in gradients.items():
                    if 'weight' in name and grad.shape[1] == data.nclass:
                        last_layer_grads = grad
                        break
                
                if last_layer_grads is not None:
                    # 通过梯度范数估计各类别的分布
                    grad_norm_per_class = torch.norm(last_layer_grads, dim=0)
                    pred_label_dist = F.softmax(-grad_norm_per_class, dim=0)
                    
                    # 计算原始标签分布
                    true_label_dist = torch.zeros(data.nclass, device=self.device)
                    for l in original_labels:
                        if l < data.nclass:
                            true_label_dist[l] += 1
                    true_label_dist = true_label_dist / true_label_dist.sum()
                    
                    # 计算分布相似度 (1 - JS散度)
                    m = 0.5 * (pred_label_dist + true_label_dist)
                    js_div = 0.5 * (
                        F.kl_div(pred_label_dist.log(), m, reduction='sum') +
                        F.kl_div(true_label_dist.log(), m, reduction='sum')
                    )
                    
                    # 归一化为0-1，值越高表示恢复程度越高
                    label_score = max(0, 1.0 - min(1.0, js_div.item()))
                    label_recovery.append(label_score)
                else:
                    label_recovery.append(0.0)
            else:
                feature_recovery.append(0.0)
                structure_recovery.append(0.0)
                label_recovery.append(0.0)
        
        # 计算平均得分
        avg_feature = np.mean(feature_recovery) if feature_recovery else 0.0
        std_feature = np.std(feature_recovery) if feature_recovery else 0.0
        
        avg_structure = np.mean(structure_recovery) if structure_recovery else 0.0
        std_structure = np.std(structure_recovery) if structure_recovery else 0.0
        
        avg_label = np.mean(label_recovery) if label_recovery else 0.0
        std_label = np.std(label_recovery) if label_recovery else 0.0
        
        # 计算总体隐私泄露分数 (平均恢复率)
        avg_recovery = (avg_feature + avg_structure + avg_label) / 3.0
        privacy_score = 1.0 - avg_recovery
        
        print(f'特征恢复率: {avg_feature:.4f} ± {std_feature:.4f} (越低越好)')
        print(f'结构恢复率: {avg_structure:.4f} ± {std_structure:.4f} (越低越好)') 
        print(f'标签恢复率: {avg_label:.4f} ± {std_label:.4f} (越低越好)')
        print(f'总体隐私泄露率: {avg_recovery:.4f} (越低越好)')
        print(f'隐私保护得分: {privacy_score:.4f} (越高越好)')
        
        recovery_metrics = {
            'feature_recovery': avg_feature,
            'structure_recovery': avg_structure,
            'label_recovery': avg_label,
            'privacy_leakage': avg_recovery,
            'privacy_score': privacy_score
        }
        
        # return recovery_metrics 

    def nas_evaluate(self, data, model_type, verbose=False, reduced=None):
        """
        Evaluates a model for neural architecture search (NAS) and returns mean and standard deviation of validation accuracy.

        Parameters
        ----------
        data : Dataset
            The dataset containing the graph data.
        model_type : str
            The type of model to evaluate.
        verbose : bool, optional, default=False
            Whether to print detailed logs.
        reduced : bool, optional, default=None
            Whether to use synthetic data.

        Returns
        -------
        mean_acc_val : float
            Mean validation accuracy over multiple runs.
        std_acc_val : float
            Standard deviation of validation accuracy over multiple runs.
        """
        args = self.args
        res = []

        # Prepare synthetic data if required
        data.feat_syn, data.adj_syn, data.labels_syn = get_syn_data(data, model_type=model_type, verbose=verbose)

        # Initialize progress bar based on verbosity
        if verbose:
            run_evaluation = trange(args.run_evaluation)
        else:
            run_evaluation = range(args.run_evaluation)

        # Collect validation accuracy results from multiple runs
        for i in run_evaluation:
            model = eval(model_type)(data.feat_syn.shape[1], args.hidden, data.nclass, args, mode='eval').to(
                self.device)
            best_acc_val = model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                              setting=args.setting, reduced=reduced)
            res.append(best_acc_val)
            if verbose:
                run_evaluation.set_postfix(best_acc_val=best_acc_val)

        res = np.array(res)

        # Print and return mean and standard deviation of validation accuracy
        if verbose:
            print(f'Validation Mean Accuracy: {100 * res.mean():.2f} +/- {100 * res.std():.2f}')

        return res.mean(), res.std()

    def tsne_vis(self, feat_train, labels_train, feat_syn, labels_syn):
        """
        Visualize t-SNE for original and synthetic data.

        Parameters:
            feat_train (torch.tensor): Original features.
            labels_train (torch.tensor): Labels for original features.
            feat_syn (torch.tensor): Synthetic features.
            labels_syn (torch.tensor): Labels for synthetic features.
        """
        labels_train_np = labels_train.cpu().numpy()
        feat_train_np = feat_train.cpu().numpy()
        labels_syn_np = labels_syn.cpu().numpy()
        feat_syn_np = feat_syn.cpu().numpy()

        # Separate features based on labels for original and synthetic data
        data_feat_ori_0 = feat_train_np[labels_train_np == 0]
        data_feat_ori_1 = feat_train_np[labels_train_np == 1]
        data_feat_syn_0 = feat_syn_np[labels_syn_np == 0]
        data_feat_syn_1 = feat_syn_np[labels_syn_np == 1]

        # Concatenate all features for t-SNE visualization
        all_data = np.concatenate((data_feat_ori_0, data_feat_ori_1, data_feat_syn_0, data_feat_syn_1), axis=0)
        perplexity_value = min(30, len(all_data) - 1)

        # Apply t-SNE to reduce dimensionality
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
        all_data_2d = tsne.fit_transform(all_data)

        # Separate 2D features based on their original and synthetic labels
        data_ori_0_2d = all_data_2d[:len(data_feat_ori_0)]
        data_ori_1_2d = all_data_2d[len(data_feat_ori_0):len(data_feat_ori_0) + len(data_feat_ori_1)]
        data_syn_0_2d = all_data_2d[
                        len(data_feat_ori_0) + len(data_feat_ori_1):len(data_feat_ori_0) + len(data_feat_ori_1) + len(
                            data_feat_syn_0)]
        data_syn_1_2d = all_data_2d[len(data_feat_ori_0) + len(data_feat_ori_1) + len(data_feat_syn_0):]

        # Plot t-SNE results
        plt.figure(figsize=(6, 4))
        plt.scatter(data_ori_0_2d[:, 0], data_ori_0_2d[:, 1], c='blue', marker='o', alpha=0.1, label='Original Class 0')
        plt.scatter(data_syn_0_2d[:, 0], data_syn_0_2d[:, 1], c='blue', marker='*', label='Synthetic Class 0')
        plt.scatter(data_ori_1_2d[:, 0], data_ori_1_2d[:, 1], c='red', marker='o', alpha=0.1, label='Original Class 1')
        plt.scatter(data_syn_1_2d[:, 0], data_syn_1_2d[:, 1], c='red', marker='*', label='Synthetic Class 1')

        plt.legend()
        plt.title('t-SNE Visualization of Original and Synthetic Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Save and display the figure
        plt.savefig(f'tsne_visualization_{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}.pdf',
                    format='pdf')
        print(
            f"Saved figure to tsne_visualization_{self.args.method}_{self.args.dataset}_{self.args.reduction_rate}.pdf")
        plt.show()

    def visualize(args, data):
        """
        Visualizes synthetic and original data using t-SNE and saves the plot as a PDF file.

        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments and configuration parameters.
        data : Dataset
            The dataset containing the graph data.
        """
        save_path = f'{args.save_path}/reduced_graph/{args.method}'
        feat_syn = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        labels_syn = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cpu')
        try:
            adj_syn = torch.load(
                f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
        except:
            adj_syn = torch.eye(feat_syn.size(0), device=args.device)

        # Obtain embeddings
        data.adj_train = to_tensor(data.adj_train)
        data.pre_conv = normalize_adj_tensor(data.adj_train, sparse=True)
        data.pre_conv = matmul(data.pre_conv, data.pre_conv)
        feat_train_agg = matmul(data.pre_conv, data.feat_train).float()

        adj_syn = to_tensor(data.adj_syn)
        pre_conv_syn = normalize_adj_tensor(adj_syn, sparse=True)
        pre_conv_syn = matmul(pre_conv_syn, pre_conv_syn)
        feat_syn_agg = matmul(pre_conv_syn, labels_syn).float()

        self.tsne_vis(data.feat_train, data.labels_train, feat_syn, labels_syn)  # Visualizes feature
        self.tsne_vis(feat_train_agg, data.labels_train, feat_syn_agg, labels_syn)  # Visualizes embedding
