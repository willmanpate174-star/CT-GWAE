"""Graph-only spectral inputs; no ground truth or external partitions."""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from data_io import DT
import model_core

def spectral_features(layers, k):
    n = layers[0].shape[0]
    if not 1 < k < n:
        raise ValueError('Require 1 < K < N')
    parts=[]
    for a in layers:
        degree=np.asarray(a.sum(1)).ravel()
        inv=np.divide(1.,np.sqrt(degree),out=np.zeros_like(degree,dtype=float),where=degree>0)
        lap=sparse.eye(n)-sparse.diags(inv)@a@sparse.diags(inv)
        values,v=eigsh(lap,k=k,which='SA',v0=np.random.default_rng(20261004).normal(size=n),tol=1e-10,maxiter=10000)
        v=v[:,np.argsort(values)]
        # Fix eigenvector signs; degenerate eigenspaces remain solver-dependent.
        for column in range(k):
            if v[np.argmax(np.abs(v[:,column])),column]<0:v[:,column]*=-1
        parts.append(v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12))
    joined=np.concatenate(parts,axis=1)
    return torch.tensor(joined,dtype=DT)

def build_features(layers,seed,k,mode,spectral=None):
    original=model_core.node_features(layers,seed)
    if mode=='degree':return original
    if mode!='spectral':raise ValueError(mode)
    if spectral is None:spectral=spectral_features(layers,k)
    if spectral.shape!=(layers[0].shape[0],len(layers)*k):raise ValueError('Spectral shape mismatch')
    return torch.cat((original,spectral),dim=1)
