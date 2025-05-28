import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
import numpy as np

from config import cli
from dataset import *
from evaluation import *
from condensation import *

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    graph = get_dataset(args.dataset, args)
    if args.attack is not None:
        if args.setting == 'ind':
            data = attack(graph, args)
    if args.method == 'hdgc':
        agent = HDGC(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph, verbose=args.verbose)
    if args.method in ['variation_edges', 'variation_neighborhoods', 'vng', 'heavy_edge', 'algebraic_JC', 'affinity_GS',
                       'kron']:
        if args.setting == 'trans':
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / graph.x.shape[0] * 100, "%")
        else:
            print("real reduction rate", reduced_graph.feat_syn.shape[0] / sum(graph.train_mask).item() * 100, "%")
    evaluator = Evaluator(args)
    res_mean, res_std = evaluator.evaluate(reduced_graph, model_type='GCN')