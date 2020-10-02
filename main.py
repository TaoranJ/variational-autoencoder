import random
import argparse

import torch
import numpy as np

from init import init
from learn import training

if __name__ == '__main__':
    pparser = argparse.ArgumentParser()
    # Global configurations
    pparser.add_argument('--seed', type=int, default=582838,
                         help='Random seed.')
    pparser.add_argument('--use-cuda', type=int, default=0,
                         help='Which GPU to use.')
    pparser.add_argument('--data-dir', type=str, required=True,
                         help='Data directory.')
    # Model configurations
    pparser.add_argument('--type', type=str, choices=['flow', 'mean-field'],
                         default='mean-field',
                         help='Variational family to use.')
    pparser.add_argument('--flow-depth', type=int, default=2,
                         help='Flow depth.')
    pparser.add_argument('--data-dim', type=int, default=784,
                         help='Dimension of the input data.')
    pparser.add_argument('--latent-dim', type=int, default=128,
                         help='Dimension of latent space.')
    pparser.add_argument('--n-samples', type=int, default=128,
                         help='# of samples to use for variational inference.')
    # Training configurations
    pparser.add_argument('--lr', type=float, default=0.001,
                         help='Learning rate.')
    pparser.add_argument('--batch-size', type=int, default=128,
                         help='Minibatch size.')
    pparser.add_argument('--epochs', type=int, default=100000,
                         help='# of epochs of training.')
    # Evaluation configurations
    pparser.add_argument('--test-batch-size', type=int, default=1024,
                         help='Minibatch size for evaluation.')
    args = pparser.parse_args()

    # config = """
    # early_stopping_interval: 5
    # """

    args.device = 'cuda:' + str(args.use_cuda) if torch.cuda.is_available()\
        else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train, valid, test, model, variational, optim = init(args)
    training(train, valid, variational, model, optim, args)
