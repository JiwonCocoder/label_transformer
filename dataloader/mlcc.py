import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import pdb
import sys
sys.path.append('.')
from util import data
from dataloader import SSLDataset
from torchvision.datasets import ImageFolder
import os
from collections import Counter

class MLCCSSL(SSLDataset):
    def read_x(self, idx):
        return Image.fromarray(self.x[idx].copy())

    @staticmethod
    def split_data(root_dir, tgt_domains, src_domains, r_val, r_lab, r_unlab, w_unlab, rand_seed, r_data=None):
        mlcc_root_dir = '/data/samsung/'
        mlcc_test_dir = os.path.join(mlcc_root_dir, 'labeled/Test')
        mlcc_train_dir = os.path.join(mlcc_root_dir, 'labeled/Train')
        mlcc_unlabled_dir = os.path.join(mlcc_root_dir, 'unlabeled')
        root_dir = Path(root_dir)
        print("Debug : 000")
        # test
        # file = root_dir/'test_batch'
        # batch = pickle.load(open(file, 'rb'), encoding='latin1')
        # pdb.set_trace()
        xt = np.array([np.array(tmp_x[0]) for tmp_x in
                       ImageFolder(root=mlcc_test_dir)])
        yt = np.array([tmp[1] for tmp in ImageFolder(root=mlcc_test_dir)],
                      dtype=np.int)
        from sklearn.utils import shuffle
        xt, yt = shuffle(xt, yt, random_state=0)
        print("Debug : 111")
        # pdb.set_trace()
        # xt = np.transpose(batch['data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
        # yt = np.array(batch['labels'], dtype=np.int)

        # val, lab, unlab
        # files = [root_dir/f'data_batch_{i}' for i in range(1, 6)]
        # batches = [pickle.load(open(file, 'rb'), encoding='latin1') for file in files]

        # x = [batch['data'].reshape((-1, 3, 32, 32)) for batch in batches]
        # x = np.concatenate([np.transpose(xi, (0, 2, 3, 1)) for xi in x])
        # y = np.concatenate([np.array(batch['labels'], dtype=np.int) for batch in batches])
        # pdb.set_trace()
        x = np.array([np.array(tmp_x[0]) for tmp_x in
                      ImageFolder(root=mlcc_train_dir)])
        y = np.array([tmp[1] for tmp in ImageFolder(root=mlcc_train_dir)],
                     dtype=np.int)
        print("Debug : aaa_")
        xu = np.array([np.array(tmp_x[0]) for tmp_x in
                       ImageFolder(root=mlcc_unlabled_dir)])

        if r_val is not None:
            # JY Validation set 제외 안시킴
            (xv, yv), (x, y) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)
            # (xv, yv) = data.split_data(x.copy(), y.copy(), rand_seed, r_val)[0]
        else:
            xv, yv = xt, yt
        # (xl, yl), (xu, yu) = data.split_data(x.copy(), y.copy(), rand_seed, r_la, b)
        xl, yl = x, y;

        # reduce data
        if r_data is not None:
            xu, yu = data.split_data(xu.copy(), np.zeros(len(xu)), rand_seed, r_data)[0]

        test_class_count = Counter(yv)
        train_class_count = Counter(yv)

        return xl, yl, xu, xv, yv, xt, yt