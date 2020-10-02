import torch

from data import binary_mnist_dataloader
from models import Model, VariationalFlow, VariationalMeanField


def init(args):
    model = Model(args.latent_dim, args.data_dim)
    if args.type == 'flow':
        variational = VariationalFlow(latent_size=args.latent_size,
                                      data_dim=args.data_dim,
                                      flow_depth=args.flow_depth)
    elif args.type == 'mean-field':
        variational = VariationalMeanField(args.data_dim, args.latent_dim)
    model.to(args.device)
    variational.to(args.device)
    optim = torch.optim.RMSprop(list(model.parameters()) +
                                list(variational.parameters()),
                                lr=args.lr, centered=True)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    train, valid, test = binary_mnist_dataloader(args, **kwargs)
    return train, valid, test, model, variational, optim
