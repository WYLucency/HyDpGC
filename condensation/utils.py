from utils import *
import torch_geometric
import math
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import convert
import networkx as nx
from dp_utils import *

def match_loss(gw_syn, gw_real, args, manifold=None, device=None):
    """
    Computes the loss between synthetic and real gradients based on the specified distance metric.

    Parameters
    ----------
    gw_syn : list of torch.Tensor
        List of synthetic gradients for different model parameters.
    gw_real : list of torch.Tensor
        List of real gradients for different model parameters.
    args : Namespace
        Arguments object containing hyperparameters for training and model.
    device : torch.device
        Device (CPU or GPU) on which computations are performed.

    Returns
    -------
    torch.Tensor
        The computed distance (loss) between synthetic and real gradients.
    """
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            if args.apply_dp:
                noise = get_noise(gw_real[ig], gw_syn[ig], args, manifold, device)
                dis += distance_wb(gwr+noise, gws)
            else:
                dis += distance_wb(gwr, gws)
            if args.apply_dro:
                return dis, dis
            
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        if args.apply_dp:
            for ig in range(len(gw_real)):
                noise = get_noise(gw_real[ig], gw_syn[ig], args, manifold, device)
                gw_real_vec.append((gw_real[ig]+noise).reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)
        else:
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)
        if args.apply_dro:
            dis_dl = torch.tensor(0.0).to(device)
            dis_dl = args.dl_fn(gw_syn_vec, gw_real_vec)
            return dis, dis_dl
        
    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        if args.apply_dp:
            for ig in range(len(gw_real)):
                noise = get_noise(gw_real[ig], gw_syn[ig], args, device)
                gw_real_vec.append((gw_real[ig]+noise).reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        else:
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    """
    Computes the distance between two tensors representing gradients using cosine similarity.

    Parameters
    ----------
    gwr : torch.Tensor
        The real gradient tensor.
    gws : torch.Tensor
        The synthetic gradient tensor.

    Returns
    -------
    torch.Tensor
        The computed distance between the real and synthetic gradients.
    """
    shape = gwr.shape

    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


# GCSNTK utils
def sub_E(idx, A):
    """
    Generates a sparse adjacency matrix of the subgraph defined by the given indices.

    Parameters
    ----------
    idx : torch.Tensor
        A tensor containing the indices of the nodes that define the subgraph.
    A : torch.Tensor
        The original adjacency matrix of the graph.

    Returns
    -------
    torch.sparse_coo_tensor
        The sparse adjacency matrix of the subgraph.
    """
    n = A.shape[0]
    n_neig = len(idx)
    operator = torch.zeros([n, n_neig])
    operator[idx, range(n_neig)] = 1
    sub_A = torch.matmul(torch.matmul(operator.t(), A), operator)

    ind = torch.where(sub_A != 0)
    inds = torch.cat([ind[0], ind[1]]).reshape(2, len(ind[0]))
    values = torch.ones(len(ind[0]))
    sub_E = torch.sparse_coo_tensor(inds, values, torch.Size([n_neig, n_neig])).to(A.device)

    return sub_E


def update_E(x_s, neig):
    """
    Update the adjacency matrix based on the features of the nodes and the average number of neighbors.

    Parameters
    ----------
    x_s : torch.Tensor
        A tensor containing the feature vectors of the nodes.
    neig : float
        The average number of neighbors each node should have.

    Returns
    -------
    torch.sparse_coo_tensor
        The sparse adjacency matrix based on the updated similarities.
    """
    n = x_s.shape[0]
    K = torch.empty(n, n)
    A = torch.zeros(n * n)

    for i in range(n):
        for j in range(i, n):
            K[i, j] = torch.norm(x_s[i] - x_s[j])
            K[j, i] = K[i, j]

    edge = int(n + torch.round(torch.tensor(neig * n / 2)))
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)
    _, indices = torch.sort(Simil)
    A[indices[0:edge]] = 1
    A = A.reshape(n, n)
    ind = torch.where(A == 1)

    ind = torch.cat([ind[0], ind[1]]).reshape(2, edge)
    values = torch.ones(edge)
    E = torch.sparse_coo_tensor(ind, values, torch.Size([n, n])).to(x_s.device)

    return E


# utils for GCSNTK
def normalize_data(data):
    """
    Normalize the input data using mean and standard deviation.

    Parameters
    ----------
    data : torch.Tensor
        The data to be normalized. Each column represents a feature, and normalization is applied to each feature independently.

    Returns
    -------
    torch.Tensor
        The normalized data where each feature has zero mean and unit variance.
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    std[std == 0] = 1
    normalized_data = (data - mean) / std
    return normalized_data


def GCF(adj, x, k=1):
    """
    Apply Graph Convolution Filter (GCF) to features using the adjacency matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix of the graph. It must include self-loops. Shape: (N, N), where N is the number of nodes.

    x : torch.Tensor
        Node features. Shape: (N, F), where F is the number of features for each node.

    k : int, optional
        Number of hops (or layers) to apply the filter. Default is 1.

    Returns
    -------
    torch.Tensor
        Filtered features after applying the graph convolution. Shape: (N, F).
    """
    D = torch.sum(adj, dim=1)
    D = torch.pow(D, -0.5)
    D = torch.diag(D)

    filter = torch.matmul(torch.matmul(D, adj), D)
    for i in range(k):
        x = torch.matmul(filter, x)
    return x


# geom
def neighborhood_difficulty_measurer(data, adj, label):
    """
    Measure the difficulty of neighborhoods in the graph based on the label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph. The shape is (N, N) where N is the number of nodes.

    label : torch.Tensor
        Tensor containing the label of each node. Shape: (N,)

    Returns
    -------
    torch.Tensor
        Difficulty scores for each node. Higher scores indicate more difficult neighborhoods.
    """
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = torch_geometric.utils.add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = neighbor_class[data.idx_train]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    return local_difficulty


def difficulty_measurer(data, adj, label):
    """
    Measure the difficulty of nodes in the graph based on their neighborhood label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph. The shape is (N, N) where N is the number of nodes.

    label : torch.Tensor
        Tensor containing the label of each node. Shape: (N,)

    Returns
    -------
    torch.Tensor
        Difficulty scores for each node. Higher scores indicate more difficult nodes.
    """
    local_difficulty = neighborhood_difficulty_measurer(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes(data, adj, label):
    """
    Sort training nodes based on their difficulty measured by neighborhood label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    numpy.ndarray
        Indices of the training nodes sorted by their difficulty, from easiest to hardest.
    """
    node_difficulty = difficulty_measurer(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()
    sorted_trainset = data.idx_train[indices]
    return sorted_trainset


def neighborhood_difficulty_measurer_in(data, adj, label):
    """
    Measure the difficulty of each node in a graph based on the entropy of neighbor labels.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    torch.Tensor
        Tensor of local difficulty scores for each node.
    """
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = torch_geometric.utils.add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    return local_difficulty


def difficulty_measurer_in(data, adj, label):
    """
    Measure the difficulty of each node in a graph based on local entropy of neighbor labels.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    torch.Tensor
        Tensor of local difficulty scores for each node.
    """
    local_difficulty = neighborhood_difficulty_measurer_in(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes_in(data, adj, label):
    """
    Sort training nodes based on their difficulty scores in ascending order.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    numpy.ndarray
        Indices of training nodes sorted by difficulty scores.
    """
    node_difficulty = difficulty_measurer_in(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()
    return indices


def training_scheduler(lam, t, T, scheduler='geom'):
    """
    Adjust the value of a parameter based on the chosen scheduling strategy.

    Parameters
    ----------
    lam : float
        The initial value or a baseline value for the parameter (0 <= lam <= 1).

    t : int
        The current training iteration or epoch.

    T : int
        The total number of training iterations or epochs.

    scheduler : str, optional
        The type of scheduling strategy to use. Options are 'linear', 'root', or 'geom'.
        Default is 'geom'.

    Returns
    -------
    float
        The adjusted value of the parameter at iteration `t` based on the scheduling strategy.
    """
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


def get_largest_cc(adj, num_nodes, data_name):
    mx = (adj).tocoo()
    edge_index = torch.LongTensor(np.vstack([mx.row, mx.col]))
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    nx_graph = convert.to_networkx(data, to_undirected=True)

    largest_cc = max(nx.connected_components(nx_graph), key=len)
    idx_lcc = list(largest_cc)
    largest_cc_graph = nx_graph.subgraph(largest_cc)

    if len(idx_lcc) == num_nodes:
        adj_lcc = adj
        adj_norm_lcc = gcn_normalize_adj(adj_lcc)
    else:
        adj_lcc = nx.to_scipy_sparse_array(largest_cc_graph)
        adj_norm_lcc = gcn_normalize_adj(adj_lcc)

    return idx_lcc, adj_norm_lcc, adj_lcc

def get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat(
        [real_eigenvals[k1_list], real_eigenvals[k2_list]]
    )
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]], dim=1,
    )

    return eigenvals, eigenvecs


def get_subspace_embed(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    u_unsqueeze = (eigenvecs.T).unsqueeze(2)  # kn1
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    sub_embed = torch.bmm(u_unsqueeze, x_trans_unsqueeze)  # kn1 @ k1d = knd
    return x_trans, sub_embed


def get_subspace_covariance_matrix(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = F.normalize(input=x_trans, p=2, dim=1)
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    co_matrix = torch.bmm(x_trans_unsqueeze.permute(0, 2, 1), x_trans_unsqueeze)  # kd1 @ k1d = kdd
    return co_matrix


def get_embed_sum(eigenvals, eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = torch.diag(1 - eigenvals) @ x_trans  # kd
    embed_sum = eigenvecs @ x_trans  # nk @ kd = nd
    return embed_sum


def get_embed_mean(embed_sum, label):
    class_matrix = F.one_hot(label).float()  # nc
    class_matrix = class_matrix.T  # cn
    embed_sum = class_matrix @ embed_sum  # cd
    mean_weight = (1 / class_matrix.sum(1)).unsqueeze(-1)  # c1
    embed_mean = mean_weight * embed_sum
    embed_mean = F.normalize(input=embed_mean, p=2, dim=1)
    return embed_mean


def get_eigh(laplacian_matrix, data_name, save=True):
    dir = "../../data/" + data_name + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"

    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)

    if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):
        eigenvalues = np.load(eigvals_path)
        eigenvectors = np.load(eigvecs_path)
    else:
        if data_name in ['ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
            eigenvalues, eigenvectors = eigsh(A=laplacian_matrix, k=1000, which="SA", tol=1e-5)
        else:
            if sp.issparse(laplacian_matrix):
                laplacian_matrix = laplacian_matrix.todense()
            eigenvalues, eigenvectors = eigh(laplacian_matrix)

        if save:
            np.save(eigvals_path, eigenvalues)
            np.save(eigvecs_path, eigenvectors)

    if data_name in ['ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)

        if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):
            eigenvalues_la = np.load(eigvals_path)
            eigenvectors_la = np.load(eigvecs_path)
        else:
            eigenvalues_la, eigenvectors_la = eigsh(A=laplacian_matrix, k=1000, which="LA", tol=1e-5)
            if save:
                np.save(eigvals_path, eigenvalues_la)
                np.save(eigvecs_path, eigenvectors_la)

        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]

    return eigenvalues, eigenvectors


def load_eigen(dataset,load_path):
    dir = os.path.join(load_path, dataset) 
    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"

    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)
    eigenvalues = np.load(eigvals_path)
    eigenvectors = np.load(eigvecs_path)

    if dataset in ['ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)
        eigenvalues_la = np.load(eigvals_path)
        eigenvectors_la = np.load(eigvecs_path)

        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]

    return eigenvalues, eigenvectors


def get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    idx_train_lcc = list(set(idx_train).intersection(set(idx_lcc)))
    y_full = y_full.cpu().numpy()
    if len(idx_lcc) == num_nodes:
        idx_map = idx_train
    else:
        y_train = y_full[idx_train]
        y_train_lcc = y_full[idx_train_lcc]

        y_lcc_idx = list((set(range(num_nodes)) - set(idx_train)).intersection(set(idx_lcc)))
        y_lcc_ = y_full[y_lcc_idx]
        counter_train = Counter(y_train)
        counter_train_lcc = Counter(y_train_lcc)
        idx = np.arange(len(y_lcc_))
        for c in range(num_classes):
            num_c = counter_train[c] - counter_train_lcc[c]
            if num_c > 0:
                idx_c = list(idx[y_lcc_ == c])
                idx_c = np.array(y_lcc_idx)[idx_c]
                idx_train_lcc += list(np.random.permutation(idx_c)[:num_c])
        idx_map = [idx_lcc.index(i) for i in idx_train_lcc]

    return idx_train_lcc, idx_map
