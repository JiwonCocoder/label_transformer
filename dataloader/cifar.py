import pdb
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.append('.')
from util import data
from dataloader import SSLDataset


class CifarSSL(SSLDataset):
    def read_x(self, idx):
        return Image.fromarray(self.x[idx].copy())


class Cifar10SSL(CifarSSL):
    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        root_dir = Path(root_dir)

        # test (xt: data, yt: labels)
        file = root_dir/'test_batch'
        batch = pickle.load(open(file, 'rb'), encoding='latin1')
        xt = np.transpose(batch['data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
        yt = np.array(batch['labels'], dtype=np.int)

        # val, lab, unlab
        files = [root_dir/f'data_batch_{i}' for i in range(1, 6)]
        batches = [pickle.load(open(file, 'rb'), encoding='latin1') for file in files]
        x = [batch['data'].reshape((-1, 3, 32, 32)) for batch in batches] #(10000, 3, 32, 32) x 5
        x = np.concatenate([np.transpose(xi, (0, 2, 3, 1)) for xi in x]) # All_x in files
        y = np.concatenate([np.array(batch['labels'], dtype=np.int) for batch in batches]) #All_y in files
        if r_val is not None: #if we have to separate val from train
            (xv, yv), (x, y) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)
        else:
            xv, yv = xt, yt # else val = test
        (xl, yl), (xu, yu) = data.split_data(x.copy(), y.copy(), rand_seed, r_lab)
        # reduce data
        if r_data is not None: #(reduce_data)
            xu, yu = data.split_data(xu.copy(), yu.copy(), rand_seed, r_data)[0]

        return xl, yl, xu, xv, yv, xt, yt


class Cifar100SSL(CifarSSL):
    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        root_dir = Path(root_dir)

        # test
        file = root_dir / 'test'
        batch = pickle.load(open(file, 'rb'), encoding='latin1')
        xt = np.transpose(batch['data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
        yt = np.array(batch['fine_labels'], dtype=np.int)

        # val, lab, unlab
        file = root_dir / 'train'
        batch = pickle.load(open(file, 'rb'), encoding='latin1')
        x = np.transpose(batch['data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
        y = np.array(batch['fine_labels'], dtype=np.int)
        if r_val is not None:
            (xv, yv), (x, y) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)
        else:
            xv, yv = xt, yt
        (xl, yl), (xu, yu) = data.split_data(x.copy(), y.copy(), rand_seed, r_lab)

        # reduce data
        if r_data is not None:
            xu, yu = data.split_data(xu.copy(), yu.copy(), rand_seed, r_data)[0]

        return xl, yl, xu, xv, yv, xt, yt
