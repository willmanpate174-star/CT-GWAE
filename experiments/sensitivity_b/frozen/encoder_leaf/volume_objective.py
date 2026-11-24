from data_io import ProfiledGW, _torch_sparse
import torch
import numpy as np
class VolumeRegularizedGW(ProfiledGW):
    """Original profiled GW plus uniform layer normalized-association penalty.
    A separate structural objective, not an algebraic equivalent of frozen GW.
    W=T/p is membership. Sparse adjacency and degree vectors only.
    """
    def __init__(self,layers,K):
        super().__init__(layers,K)
        self.A=[_torch_sparse(a) for a in layers if float(a.sum())>0]
        self.degree=[torch.tensor(np.asarray(a.sum(1)).ravel(),dtype=torch.float64) for a in layers if float(a.sum())>0]
        self.weight=.25
    def association_penalty(self,T):
        W=T/self.p[:,None]
        scores=[]
        for A,d in zip(self.A,self.degree):
            volume=(d[:,None]*W).sum(0)
            within=(W*torch.sparse.mm(A,W)).sum(0)
            scores.append((within/volume.clamp_min(1e-12)).sum()/self.K)
        return 1-torch.stack(scores).mean()
    def loss(self,T):
        return super().loss(T)+self.weight*self.association_penalty(T)
