"""Symmetric fusion of genuine edge-bearing layers for input diffusion."""
import numpy as np,torch

def support_fusion(operators,layers):
    active=torch.tensor(np.stack([np.asarray(A.sum(1)).ravel()>0 for A in layers]),dtype=operators[0].dtype)
    count=active.sum(0);inv=count.clamp_min(1).rsqrt();pieces=[]
    for op,mask in zip(operators,active):
        op=op.coalesce();ij=op.indices();values=op.values()*mask[ij[0]]*mask[ij[1]]*inv[ij[0]]*inv[ij[1]]
        keep=values!=0;pieces.append(torch.sparse_coo_tensor(ij[:,keep],values[keep],op.shape,check_invariants=True).coalesce())
    aggregate=pieces[0]
    for part in pieces[1:]:aggregate=aggregate+part
    isolated=torch.nonzero(count==0).flatten()
    if len(isolated):aggregate=aggregate+torch.sparse_coo_tensor(torch.stack([isolated,isolated]),torch.ones(len(isolated),dtype=operators[0].dtype),operators[0].shape,check_invariants=True)
    return aggregate.coalesce()
