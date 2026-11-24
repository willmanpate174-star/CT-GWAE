"""Fixed equal-weight node/degree GW, same common neural membership W."""
import torch
from edge_retention_objective import EdgeRetentionGW
from data_io import ProfiledGW
from profiled_template import template_profile

class DualMeasureGW(EdgeRetentionGW):
    def __init__(self,layers,K,coefficient=.5):
        super().__init__(layers,K,coefficient)
        if coefficient!=.5:raise ValueError('The registered trial uses equal weights only')
        self.node=ProfiledGW(layers,K)
    def layer_parts(self,T):
        q,b,a,x=self.coefficients(T)
        _,_,gain,_,_=template_profile(q,self.mu,x)
        degree=(self.nu-gain)/self.null
        nq=T.sum(0);h=nq.square().sum()
        mask=1-torch.eye(self.K,dtype=T.dtype)
        cross=(nq[:,None]*nq[None,:]*mask).sum()
        nx=self.node._x(T)
        node=1-torch.relu(nx-h*self.node.mu).square()/(h*cross).clamp_min(torch.finfo(T.dtype).tiny)/self.node.null
        return node,degree
    def parts(self,T):
        node,degree=self.layer_parts(T)
        nodeq=T.sum(0);mass=.02*(torch.relu(self.q_floor-nodeq)/self.q_floor).square().sum()
        return (node.mean()+degree.mean())/2,mass,T.new_zeros(())
