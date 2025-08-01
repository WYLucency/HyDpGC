from collections import Counter

import numpy as np
import torch
from sklearn.cluster import BisectingKMeans
from torch_scatter import scatter_mean

from coarsening.coarsening_base import Coarsen
from dataset.utils import save_reduced
from evaluation.utils import verbose_time_memory

from torch_sparse import matmul
from utils import normalize_adj_tensor, to_tensor


class ClusterAgg(Coarsen):
    """
    ClusteringAgg is a variant of Clustering, which clustering aggregated features instead of original features.
    A structure-free coarsening method that also serves as initialization for condensation methods.
    Outputs synthesized features (`feat_syn`) and labels (`label_syn`).

    Parameters
    ----------
    setting : str
        Configuration setting.
    data : object
        Data object containing the graph and feature information.
    args : object
        Arguments containing various settings for the coarsening process.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, setting, data, args):
        super(ClusterAgg, self).__init__(setting, data, args)
        args.method = "clustering_agg"

    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by aggregating features using a pre-convolution step and clustering.

        Parameters
        ----------
        data : object
            The data to be reduced.
        verbose : bool, optional
            If True, prints verbose output. Defaults to True.
        save : bool, optional
            If True, saves the reduced data. Defaults to True.

        Returns
        -------
        object
            The reduced data with synthesized features and labels.
        """
        args = self.args
        n_classes = data.nclass

        if args.setting == 'trans':
            data.adj_fully = to_tensor(data.adj_full)
            data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
            data.pre_conv = matmul(data.pre_conv, data.pre_conv)
            feat_agg = matmul(data.pre_conv, data.feat_full).float()
        else:
            data.adj_fully = to_tensor(data.adj_train)
            data.pre_conv = normalize_adj_tensor(data.adj_fully, sparse=True)
            data.pre_conv = matmul(data.pre_conv, data.pre_conv)
            feat_agg = matmul(data.pre_conv, data.feat_train).float()

        if hasattr(data, 'labels_syn') and data.labels_syn is not None:
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
        else:
            y_syn, y_train = self.prepare_select(data, args)

        x_train = feat_agg[data.train_mask] if args.setting == 'trans' else feat_agg
        x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1])

        for c in range(n_classes):
            x_c = x_train[y_train == c].cpu()
            n_c = (y_syn == c).sum().item()
            k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
            k_means.fit(x_c)
            clusters = torch.from_numpy(k_means.predict(x_c)).long()
            x_syn[y_syn == c] = scatter_mean(x_c, clusters, dim=0)

        data.feat_syn, data.labels_syn = x_syn.to(x_train.device), y_syn
        data.adj_syn = torch.eye(data.feat_syn.shape[0])

        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        return data

    def prepare_select(self, data, args):
        """
        Prepares and selects synthetic labels and features for coarsening.

        Parameters
        ----------
        data : object
            The data to be processed.
        args : object
            Arguments containing various settings for the coarsening process.

        Returns
        -------
        tuple
            A tuple containing:
            - labels_syn : ndarray
                Synthesized labels.
            - labels_train : tensor
                Training labels.
        """
        num_class_dict = {}
        syn_class_indices = {}
        labels_train = data.labels_train

        counter = Counter(data.labels_train.tolist())
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []

        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        labels_syn = np.array(labels_syn)

        return labels_syn, labels_train
