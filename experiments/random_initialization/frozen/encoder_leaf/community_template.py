"""Independent experimental profiled template; never imported by formal V8.

Each layer has K within-community coefficients and one between-community
coefficient, constrained theta_k >= theta_out >= 0. Nonnegative graph C and
strictly positive T with uniform fixed row masses are required.
"""
from pathlib import Path
import sys
import torch

from data_io import ProfiledGW

def fit_coefficients(weights, moments):
    """Weighted star-order regression, sorted pooling of violated means.

    Index 0 is the between-community group. Sorting is detached; all coefficient
    values remain differentiable inside each fixed active-set region.
    """
    means=moments/weights
    order=torch.argsort(means[1:].detach())+1
    pooled=[0]
    for index in order.tolist():
        current=moments[pooled].sum()/weights[pooled].sum()
        if float(means[index].detach()) < float(current.detach()):
            pooled.append(index)
        else:break
    common=moments[pooled].sum()/weights[pooled].sum()
    return torch.stack([common if i in pooled else means[i] for i in range(len(weights))])

class CommunitySpecificGW(ProfiledGW):
    def loss(self,T):
        q=T.sum(0);within=q.square();outside=1-within.sum()
        weights=torch.cat((outside[None],within))
        residuals=[]
        for c,mu,null in zip(self.C,self.mu,self.null):
            x=(T*torch.sparse.mm(c,T)).sum(0)
            moments=torch.cat(((mu-x.sum())[None],x))
            theta=fit_coefficients(weights,moments)
            residual=(null+mu.square()-2*(theta*moments).sum()+(theta.square()*weights).sum())/null
            residuals.append(residual)
        mass=.02*(torch.relu(self.q_floor-q)/self.q_floor).square().sum()
        return torch.stack(residuals).mean()+mass
