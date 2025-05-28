import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from config import cli
from dataset import *
from evaluation import Evaluator, PropertyEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args, load_path=args.load_path)
    if args.eval_whole:
        evaluator = Evaluator(args)
        evaluator.MIA_evaluate(data, reduced=False, model_type='GCN')
    else:
        if args.attack is not None:
            args.reduced = False
            data = attack(data, args)
            args.save_path = f'checkpoints'
        evaluator = Evaluator(args)
        evaluator.MIA_evaluate(data, reduced=True, model_type='GCN')
