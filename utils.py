import random

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.sparse as ts
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch_sparse import SparseTensor,fill_diag,matmul,mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import degree
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import scipy.sparse as sp
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# from deeprobust.graph.utils import *
def is_identity(mx, device):
    identity = torch.eye(mx.size(0), device=device)
    if isinstance(mx, torch.Tensor):
        if is_sparse_tensor(mx):
            dense_tensor = mx.to_dense().float()
        else:
            dense_tensor = mx.float()
    elif isinstance(mx, SparseTensor):
        dense_tensor = mx.to_dense().float()
    else:
        raise ValueError("Input must be a torch.Tensor or torch.sparse.FloatTensor or torch_sparse.SparseTensor")
    return torch.allclose(dense_tensor, identity)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss


def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1)) / n - 0.5)


def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro


def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum() - thresh)
    # return F.relu(adj.sum()-thresh) / n**2


def feature_smoothing(adj, X):
    adj = (adj.t() + adj) / 2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv + 1e-8
    r_inv = r_inv.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat


def row_normalize_tensor(mx):
    mx -= mx.min()
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx


# sfgc utils#

def one_hot_sfgc(x,
                 num_classes,
                 center=True,
                 dtype=np.float32):
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


def calc(gntk, feat1, feat2, diag1, diag2, A1, A2):
    return gntk.gntk(feat1, feat2, diag1, diag2, A1, A2)


# def loss_acc_fn_train(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
#     # print(k_ss.device, torch.abs(torch.tensor(reg)).to(k_ss.device),torch.trace(k_ss).device, torch.eye(k_ss.shape[0]).device)
#     k_ss_reg = (k_ss + torch.abs(torch.tensor(reg)).to(k_ss.device) * torch.trace(k_ss).to(k_ss.device) * torch.eye(
#         k_ss.shape[0]).to(k_ss.device) / k_ss.shape[0])
#     pred = torch.matmul(k_ts[data.idx_train, :].cuda(), torch.matmul(torch.linalg.inv(k_ss_reg).cuda(),
#                                                                      torch.from_numpy(y_support).to(
#                                                                          torch.float64).cuda()))
#     mse_loss = torch.nn.functional.mse_loss(pred.to(torch.float64).cuda(),
#                                             torch.from_numpy(y_target).to(torch.float64).cuda(), reduction="mean")
#     acc = 0
#     return mse_loss, acc


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    k_ss_reg = (k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc


# =================scaling up========#


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def to_camel_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


# def cal_storage(data, setting):
#     if setting == 'trans':
#         origin_storage = getsize_mb([data.x, data.edge_index, data.y])
#     else:
#         origin_storage = getsize_mb([data.feat_train, data.adj_train, data.labels_train])
#     condensed_storage = getsize_mb([data.feat_syn, data.adj_syn, data.labels_syn])
#     print(f'Origin graph:{origin_storage:.2f}Mb  Condensed graph:{condensed_storage:.2f}Mb')

def to_tensor(*vars, **kwargs):
    device = kwargs.get('device', 'cpu')
    tensor_list = []
    for var in vars:
        var = check_type(var, device)
        tensor_list.append(var)
    for key, value in kwargs.items():
        if key != 'device':
            value = check_type(value, device)
            if 'label' in key and len(value.shape) == 1:
                tensor = value.long()
            else:
                tensor = value
            tensor_list.append(tensor)
    if len(tensor_list) == 1:
        return tensor_list[0]
    else:
        return tensor_list


def check_type(var, device):
    if sp.issparse(var):
        var = sparse_mx_to_torch_sparse_tensor(var).coalesce()
    elif isinstance(var, np.ndarray):
        var = torch.from_numpy(var)
    else:
        pass
    return var.float().to(device)


# ============the following is copy from deeprobust/graph/utils.py=================
import scipy.sparse as sp


def encode_onehot(labels):
    """Convert label to onehot format.

    Parameters
    ----------
    labels : numpy.array
        node labels

    Returns
    -------
    numpy.array
        onehot labels
    """
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx


def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.

    Parameters
    ----------
    labels : torch.LongTensor
        node labels

    Returns
    -------
    torch.LongTensor
        onehot labels tensor

    """

    eye = torch.eye(labels.max() + 1).to(labels.device)
    onehot_mx = eye[labels]
    return onehot_mx


def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_adj_sgformer(adj):
    """
    Normalize the adjacency matrix. Works for both dense and sparse matrices.

    Args:
    adj (torch.Tensor): The adjacency matrix (either dense or sparse COO).
    device (torch.device): The device to run the normalization on.

    Returns:
    torch.Tensor or SparseTensor: The normalized adjacency matrix.
    """
    # if is_sparse_tensor(adj):
    #     # Sparse matrix normalization
    #     row = adj.indices()[0]
    #     col = adj.indices()[1]
    #     values = adj.values()
    if isinstance(adj, SparseTensor):
        row, col, values = adj.coo()
        # Number of nodes
        N = adj.size(0)

        # Compute degree for normalization
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()

        # Normalize the values directly
        normalized_values = values * d_norm_in * d_norm_out
        normalized_values = torch.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Create a new SparseTensor with normalized values
        adj_normalized = SparseTensor(row=row, col=col, value=normalized_values, sparse_sizes=(N, N))

    else:
        # Dense matrix normalization
        N = adj.size(0)

        # Compute degree for normalization
        d = adj.sum(dim=1).float()
        d_norm = torch.diag((1. / d).sqrt())

        # Normalize the dense adjacency matrix
        adj_normalized = d_norm @ adj @ d_norm
        adj_normalized = torch.nan_to_num(adj_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return adj_normalized


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix, return sparse or not
    """
    device = adj.device
    if sparse:
        adj = to_scipy(adj)
        mx = gcn_normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(mx).to(device).coalesce()
        adj = SparseTensor(row=adj.indices()[0], col=adj.indices()[1],
                           value=adj.values(), sparse_sizes=adj.size()).t()
        return adj

    else:
        if len(adj.shape) == 3:
            adjs = []
            for i in range(adj.shape[0]):
                ad = adj[i]
                ad = dense_gcn_norm(ad, device)
                adjs.append(ad)
            return torch.stack(adjs)
        else:
            adj = dense_gcn_norm(adj, device)

            return adj


def dense_gcn_norm(adj, device):
    if type(adj) is not torch.Tensor:
        adj = torch.from_numpy(adj)
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def gcn_normalize_adj(adj, device='cpu'):
    if sp.issparse(adj):
        return sparse_gcn_norm(adj)
    elif type(adj) is torch.Tensor:
        return dense_gcn_norm(adj, device)
    else:
        return dense_gcn_norm(adj, device).numpy()


def sparse_gcn_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj).dot(r_mat_inv)
    return adj


def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # mx = mx.dot(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def degree_normalize_sparse_tensor(adj, fill_value=1):
    """degree_normalize_sparse_tensor.
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight
    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def degree_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """

    device = adj.device
    if sparse:
        # return  degree_normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = degree_normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def roc_auc(output, labels, is_sigmoid=False):
    """Return ROC-AUC score of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        true labels (0 or 1)
    is_sigmoid : bool, optional
        If True, apply sigmoid thresholding on the output, by default False.

    Returns
    -------
    float
        ROC-AUC score
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    labels = labels.cpu().numpy()
    output = output.cpu().numpy()

    if not is_sigmoid:
        # For multi-class classification (softmax output), get probabilities for the positive class (class 1).
        if output.shape[1] > 1:
            output = output[:, 1]  # Use the probabilities of the positive class
        else:
            output = np.argmax(output, axis=1)
    else:
        # For binary classification (sigmoid output)
        output = output[:, 1]

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(labels, output)

    return roc_auc


def f1_macro(output, labels, is_sigmoid=False):
    """Return F1-macro score of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        true labels (0 or 1)

    Returns
    -------
    float
        F1-macro score
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    labels = labels.cpu().numpy()
    output = output.cpu().numpy()

    if not is_sigmoid:
        output = np.argmax(output, axis=1)
    else:
        output = output[:, 1]
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

    f1 = f1_score(labels, output, average="macro")

    return f1

# def loss_acc(output, labels, targets, avg_loss=True):
#     if type(labels) is not torch.Tensor:
#         labels = torch.LongTensor(labels)
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()[targets]
#     loss = F.nll_loss(output[targets], labels[targets], reduction='mean' if avg_loss else 'none')

#     if avg_loss:
#         return loss, correct.sum() / len(targets)
#     return loss, correct
    # correct = correct.sum()
    # return loss, correct / len(labels)


# def get_perf(output, labels, mask, verbose=True):
#     """evalute performance for test masked data"""
#     loss = F.nll_loss(output[mask], labels[mask])
#     acc = accuracy(output[mask], labels[mask])
#     if verbose:
#         print("loss= {:.4f}".format(loss.item()),
#               "accuracy= {:.4f}".format(acc.item()))
#     return loss.item(), acc.item()


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse_coo_tensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


# slower version....
# sparse_mx = sparse_mx.tocoo().astype(np.float32)
# indices = torch.from_numpy(
#     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
# values = torch.from_numpy(sparse_mx.data)
# shape = torch.Size(sparse_mx.shape)
# return torch.sparse.FloatTensor(indices, values, shape)


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def get_train_test(nnodes, test_size=0.8, stratify=None, seed=None):
    """This function returns training and test set without validation.
    It can be used for settings of different label rates.

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_test :
        node test indices
    """
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - test_size
    idx_train, idx_test = train_test_split(idx, random_state=None,
                                           train_size=train_size,
                                           test_size=test_size,
                                           stratify=stratify)

    return idx_train, idx_test


def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20:])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test


def get_train_test_labelrate(labels, label_rate):
    """Get train test according to given label rate.
    """
    nclass = labels.max() + 1
    train_size = int(round(len(labels) * label_rate / nclass))
    print("=== train_size = %s ===" % train_size)
    idx_train, idx_val, idx_test = get_splits_each_class(labels, train_size=train_size)
    return idx_train, idx_test


def get_splits_each_class(labels, train_size):
    """We randomly sample n instances for class, where n = train_size.
    """
    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_size])).astype(np.int)
        idx_val = np.hstack((idx_val, labels_i[train_size: 2 * train_size])).astype(np.int)
        idx_test = np.hstack((idx_test, labels_i[2 * train_size:])).astype(np.int)

    return np.random.permutation(idx_train), np.random.permutation(idx_val), \
        np.random.permutation(idx_test)


def unravel_index(index, array_shape):
    rows = torch.div(index, array_shape[1], rounding_mode='trunc')
    cols = index % array_shape[1]
    return rows, cols


def get_degree_squence(adj):
    try:
        return adj.sum(0)
    except:
        return ts.sum(adj, dim=1).to_dense()


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                            + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll


def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.

    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.

    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]


# def visualize(your_var):
#     """visualize computation graph"""
#     from torchviz import make_dot
#     make_dot(your_var).view()
#

def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)


def add_mask(data, dataset):
    """data: ogb-arxiv pyg data format"""
    # for arxiv
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    n = data.x.shape[0]
    data.train_mask = index_to_mask(train_idx, n)
    data.val_mask = index_to_mask(valid_idx, n)
    data.test_mask = index_to_mask(test_idx, n)
    data.y = data.y.squeeze()
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def add_feature_noise(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    # noise = torch.normal(mean=torch.zeros(int(noise_ratio*n), d), std=1)
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio * n), d))).to(data.x.device)
    indices = np.arange(n)
    indices = np.random.permutation(indices)[: int(noise_ratio * n)]
    delta_feat = torch.zeros_like(data.x)
    delta_feat[indices] = noise - data.x[indices]
    data.x[indices] = noise
    mask = np.zeros(n)
    mask[indices] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask


def add_feature_noise_test(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    indices = np.arange(n)
    test_nodes = indices[data.test_mask.cpu()]
    selected = np.random.permutation(test_nodes)[: int(noise_ratio * len(test_nodes))]
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio * len(test_nodes)), d)))
    noise = noise.to(data.x.device)

    delta_feat = torch.zeros_like(data.x)
    delta_feat[selected] = noise - data.x[selected]
    data.x[selected] = noise
    # mask = np.zeros(len(test_nodes))
    mask = np.zeros(n)
    mask[selected] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask

