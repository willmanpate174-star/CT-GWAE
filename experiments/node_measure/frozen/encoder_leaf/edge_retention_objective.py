"""Degree-measure profiled GW plus globally scaled original-edge retention."""
from pathlib import Path
import sys,torch
from degree_original_template import DegreeOriginalGW
from profiled_template import template_profile

class EdgeRetentionGW(DegreeOriginalGW):
    def __init__(self,layers,K,coefficient=.25):
        super().__init__(layers,K);self.coefficient=coefficient
        self.audit=dict(coefficient_calls=0,q_guard_calls=0,nonfinite_inputs=0,nonfinite_losses=0)
    def coefficients(self,T):
        W=T/self.p[:,None];q=self.degree_measure@W
        self.audit['coefficient_calls']+=1
        self.audit['q_guard_calls']+=int(bool((q.detach()<1e-12).any()))
        self.audit['nonfinite_inputs']+=int(not bool(torch.isfinite(T.detach()).all()))
        diag=torch.stack([(W*torch.sparse.mm(A,W)).sum(0)/v for A,v in zip(self.raw,self.volumes)])
        x=(diag/q.clamp_min(1e-12)).sum(1)
        b,a,_,_,_=template_profile(q,self.mu,x)
        return q,b,a,x
    def parts(self,T):
        W=T/self.p[:,None];q,b,a,x=self.coefficients(T)
        _,_,gain,_,_=template_profile(q,self.mu,x)
        structure=((self.nu-gain)/self.null).mean()
        nodeq=T.sum(0);mass=.02*(torch.relu(self.q_floor-nodeq)/self.q_floor).square().sum()
        edge=torch.stack([1-(W*torch.sparse.mm(A,W)).sum()/v for A,v in zip(self.raw,self.volumes)]).mean()
        return structure,mass,edge
    def loss(self,T):
        structure,mass,edge=self.parts(T)
        value=structure+mass+self.coefficient*edge
        self.audit['nonfinite_losses']+=int(not bool(torch.isfinite(value.detach())))
        return value
