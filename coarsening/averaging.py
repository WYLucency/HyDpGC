from collections import Counter

import numpy as np
import torch

from coarsening.coarsening_base import Coarsen
from dataset.utils import save_reduced
from evaluation.utils import verbose_time_memory


class Average(Coarsen):
    """
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
        super(Average, self).__init__(setting, data, args)
        args.method = "averaging"


    @verbose_time_memory
    def reduce(self, data, verbose=True, save=True):
        """
        Reduces the data by averaging features for each class.

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
        if hasattr(data, 'labels_syn'):
            y_syn = data.labels_syn
            self.labels_train = data.labels_train
            y_train = data.labels_train
            x_train = data.feat_train
        else:
            y_syn, y_train, x_train = self.prepare_select(data, args)

        x_syn = torch.zeros(y_syn.shape[0], x_train.shape[1])
        for c in range(n_classes):
            x_c = x_train[y_train == c]
            x_syn[y_syn == c] = x_c.mean(0)

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
            - feat_train : tensor
                Training features.
        """
        num_class_dict = {}
        syn_class_indices = {}
        feat_train = data.feat_train
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

        return labels_syn, labels_train, feat_train

