import os
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from .preprocessing import preprocess_bvp

def load_bvp(path, T=20, lowpass=False, cutoff=0.4):
    x = sio.loadmat(path)['velocity_spectrum_ro'].astype(np.float32)
    return preprocess_bvp(x, T, lowpass, cutoff)

class BVPDataset(Dataset):
    def __init__(self, df, T=20, lowpass=False, domain_label=0):
        self.X = np.stack([load_bvp(p, T, lowpass) for p in df['path']])[:, None]
        self.y = df['label'].values.astype(np.int64)
        self.d = np.full(len(df), domain_label, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), self.y[i], self.d[i]

def make_datasets(root, split_dir='splits', lowpass=False, T=20):
    out = {}
    for name, dom in [('src_tr', 0), ('src_va', 0), ('tgt_tr', 1), ('tgt_te', 1)]:
        df = pd.read_csv(os.path.join(split_dir, f'{name}.csv'))
        df['path'] = df['rel'].apply(lambda r: os.path.join(root, r))
        out[name] = BVPDataset(df, T, lowpass, dom)
    return out
