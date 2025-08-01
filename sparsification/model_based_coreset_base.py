import numpy as np

from dataset.utils import save_reduced
from evaluation import *
from models import *
from sparsification.coreset_base import CoreSet
from utils import to_tensor


class MBCoreSet(CoreSet):
    def __init__(self, setting, data, args, **kwarg):
        super(MBCoreSet, self).__init__(setting, data, args, **kwarg)

    @verbose_time_memory
    def reduce(self, data, verbose=False, save=True):

        args = self.args
        model = eval(self.condense_model)(data.feat_full.shape[1], args.hidden, data.nclass, args).to(
            self.device)
        if self.setting == 'trans':
            if args.method in ['sfgc']:
                # model.fit_with_val(data, train_iters=1200, normadj=True, verbose=verbose,
                #                    setting=args.setting, reduced=False, final_output=True)
                # embeds = model.predict(data.feat_full, data.adj_full, output_layer_features=True)[0].detach()
                idx_selected = np.load(f'sparsification/fixed_idx/idx_{args.dataset}_{args.reduction_rate}_kcenter_15.npy')
            else:
                model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                   setting=args.setting, reduced=False)
                embeds = model.predict(data.feat_full, data.adj_full).detach()
                idx_selected = self.select(embeds)

            data.adj_syn = data.adj_full[np.ix_(idx_selected, idx_selected)]
            data.feat_syn = data.feat_full[idx_selected]
            data.labels_syn = data.labels_full[idx_selected]

        if self.setting == 'ind':
            if args.method in ['sfgc']:
                idx_selected = np.load(f'sparsification/fixed_idx/idx_{args.dataset}_{args.reduction_rate}_kcenter_15.npy')
            else:
                model.fit_with_val(data, train_iters=args.eval_epochs, normadj=True, verbose=verbose,
                                   setting=args.setting, reduced=False)
                embeds = model.predict(data.feat_full, data.adj_full).detach()

                idx_selected = self.select(embeds)
            data.feat_syn = data.feat_train[idx_selected]
            data.adj_syn = data.adj_train[np.ix_(idx_selected, idx_selected)]
            data.labels_syn = data.labels_train[idx_selected]

        if verbose:
            print('selected nodes:', idx_selected.shape[0])
            print('induced edges:', data.adj_syn.sum())
        data.adj_syn, data.feat_syn, data.labels_syn = to_tensor(data.adj_syn, data.feat_syn, data.labels_syn,
                                                                 device='cpu')
        if save:
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)

        # if args.method in ['sfgc', 'geom']:
        #     # recover args
        #     args.eval_epochs = epoch
        #     args.weight_decay = wd
        #     args.lr = lr

        return data
