"""Degree source measures with original shared two-level GW prototype.
Uniform T only parameterizes common W; all actual couplings are layer-specific.
"""
import numpy as np,torch
from data_io import DT,_torch_sparse
class DegreeOriginalGW:
    def __init__(self,layers,K):
        if K<2:raise ValueError('K>=2 required')
        self.n,self.K=layers[0].shape[0],K
        self.p=torch.full((self.n,),1/self.n,dtype=DT);self.delta=.01;self.q_floor=1/(4*K)
        raw=[];measure=[];volumes=[];nu=[];ids=[]
        for i,A0 in enumerate(layers):
            A=A0.copy().astype(float);A.setdiag(0);A.eliminate_zeros();d=np.asarray(A.sum(1)).ravel();v=float(d.sum())
            if v<=0:continue
            c=A.tocoo();second=float(np.sum(c.data**2/(d[c.row]*d[c.col])))
            if second-1<=1e-10:continue
            raw.append(_torch_sparse(A));measure.append(d/v);volumes.append(v);nu.append(second);ids.append(i)
        if not raw:raise ValueError('No positive degree-weighted variance layer')
        self.raw=raw;self.degree_measure=torch.tensor(np.stack(measure),dtype=DT);self.volumes=torch.tensor(volumes,dtype=DT)
        self.mu=torch.ones(len(raw),dtype=DT);self.nu=torch.tensor(nu,dtype=DT);self.null=self.nu-1;self.active_layers=ids
    def statistics(self,T):
        W=T/self.p[:,None];q=self.degree_measure@W
        x=torch.stack([(W*torch.sparse.mm(A,W)).sum()/v for A,v in zip(self.raw,self.volumes)])
        h=q.square().sum(1)
        # Positive off-diagonal mass evaluated without subtracting h from 1.
        pair=q[:,:,None]*q[:,None,:];mask=1-torch.eye(self.K,dtype=DT)
        cross=(pair*mask).sum((1,2))
        return q,x,h,cross
    def loss(self,T):
        q,x,h,cross=self.statistics(T)
        gain=torch.relu(x-h).square()/(h*cross).clamp_min(torch.finfo(DT).tiny)
        E=1-gain/self.null
        nodeq=T.sum(0);mass=.02*(torch.relu(self.q_floor-nodeq)/self.q_floor).square().sum()
        return E.mean()+mass
    def decode_templates(self,T):
        q,x,h,cross=self.statistics(T);separated=x>=h
        within=torch.where(separated,x/h,self.mu)
        outside=torch.where(separated,(self.mu-x)/cross.clamp_min(torch.finfo(DT).tiny),self.mu)
        return dict(active_layers=self.active_layers,degree_target_masses=q.detach().tolist(),theta_in=within.detach().tolist(),theta_out=outside.detach().tolist(),cross_mass=cross.detach().tolist())
