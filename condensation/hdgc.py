from tqdm import trange

from condensation.base import GCondBase
from dataset.utils import save_reduced
from evaluation.utils import verbose_time_memory
from utils import *
from models import *


class HDGC(GCondBase):
    """
    
    """
    def __init__(self, setting, data, args, **kwargs):
        super(HDGC, self).__init__(setting, data, args, **kwargs)

        self.scheduler_feat = torch.optim.lr_scheduler.StepLR(self.optimizer_feat, step_size=200, gamma=0.5)
        self.scheduler_heg = torch.optim.lr_scheduler.StepLR(self.optimizer_heg, step_size=200, gamma=0.5)
       

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        heg = self.heg
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        if self.manifold:
            self.manifold = self.manifold.to(self.device)
        # initialization the features
        feat_init = self.init()
       

        self.feat_syn.data.copy_(feat_init)
        adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0
        epsilon_start=args.target_epsilon
        epsilon_end=args.down_epsilon

        # seed_everything(args.seed + it)
        model = eval(args.condense_model)(feat_syn.shape[1], args.hidden, data.nclass, args).to(self.device)
        for it in trange(args.epochs):

            model.initialize()
            model.train()
            if args.apply_dp:
                args.current_epsilon = epsilon_start - (epsilon_start - epsilon_end) * it / args.epochs

            for ol in range(outer_loop):
                adj_syn, _ = heg()
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                model = self.check_bn(model)

                loss = self.train_class(model, adj, features, labels, labels_syn, args)

                loss_avg += loss.item()
                # print('loss_avg:',loss_avg)
                self.optimizer_feat.zero_grad()
                self.optimizer_heg.zero_grad()
                loss.backward()
                self.optimizer_heg.step()
                self.optimizer_feat.step()

            loss_avg /= (data.nclass * outer_loop)
            # Step the learning rate schedulers
            self.scheduler_feat.step()
            self.scheduler_heg.step()

            
            if it in args.checkpoints:
                self.adj_syn = heg.inference()

                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
