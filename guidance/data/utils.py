import h5py
import torch
from os.path import exists
import numpy as np

def movie2feats(feat_file, movies):
    assert exists(feat_file), '{} not found'.format(feat_file)
    with h5py.File(feat_file, 'r') as f:
        vid_feats = {m:f[m][:].astype(np.float16) for m in movies}
    return vid_feats

def query2feats(query_file, sentences):
    assert exists(query_file), '{} not found'.format(query_file)
    with h5py.File(query_file, 'r') as f:
        q_feats = {m:f[m][:].astype(np.float16) for m in sentences}
    return q_feats
