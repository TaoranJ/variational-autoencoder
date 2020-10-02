"""Get the binarized MNIST dataset and convert to hdf5.
From https://github.com/yburda/iwae/blob/master/datasets.py
"""

import os
import urllib.request

import h5py
import torch
import numpy as np


binary_mnist_url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/' + \
        'binarized_mnist/'


def binary_mnist_dataloader(args, **kwcfg):
    """Dataloader of the binary mnist dataset.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Configurations.
    kwcfg : dict
        Other configurations for training set dataloader.

    Returns
    -------
    train : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader of the training set.
    valid : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader of the validation set.
    test : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader of the test set.
    """

    data_file = os.path.join(args.data_dir, 'binary_mnist.h5')
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
        download_binary_mnist(args.data_dir)
    elif not os.path.isfile(data_file):
        download_binary_mnist(args.data_dir)
    f = h5py.File(data_file, 'r')
    train, valid, test = f['train'][::], f['valid'][::], f['test'][::]
    train = torch.utils.data.TensorDataset(torch.from_numpy(train))
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                        shuffle=True, **kwcfg)
    valid = torch.utils.data.TensorDataset(torch.from_numpy(valid))
    valid = torch.utils.data.DataLoader(valid, batch_size=args.test_batch_size,
                                        shuffle=False)
    test = torch.utils.data.TensorDataset(torch.from_numpy(test))
    test = torch.utils.data.DataLoader(test, batch_size=args.test_batch_size,
                                       shuffle=False)
    return train, valid, test


def download_binary_mnist(data_dir):
    """Download binary minist dataset to data_dir, parse and save them as
    data_dir/binary_mnist.h5.

    Parameters
    ----------
    data_dir : str
        Directory to save the binary minist dataset.

    """

    # Download
    print('Downloading binary MNIST data ...')
    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = binary_mnist_url + 'binarized_mnist_{}.amat'.format(subdataset)
        urllib.request.urlretrieve(url, os.path.join(data_dir, filename))
    # Parse dataset
    data = parse_binary_mnist(data_dir)
    # Save as h5
    f = h5py.File(os.path.join(data_dir, 'binary_mnist.h5'), 'w')
    f.create_dataset('train', data=data['train'])
    f.create_dataset('valid', data=data['valid'])
    f.create_dataset('test', data=data['test'])
    f.close()


def parse_binary_mnist(data_dir):
    """Parse files in data_dir and save them in h5 format.

    Parameters
    ----------
    data_dir : str
        Data directory.

    Returns
    -------
    data : dict
        Split dataset in ndarray format.
        {'train': training set, 'valid': validation set, 'test': test set}.
    """

    def lines_to_np_array(lines):  # str line to array of binary int
        return np.array([[int(i) for i in line.split()] for line in lines])

    data = {'train': None, 'valid': None, 'test': None}
    for key in data:
        fpath = os.path.join(data_dir, 'binarized_mnist_{}.amat'.format(key))
        lines = open(fpath, 'r').readlines()
        data[key] = lines_to_np_array(lines).astype('float32')
    return data
