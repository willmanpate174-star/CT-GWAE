"""Independent dual-measure GW with community-specific assortative levels."""
from pathlib import Path
import sys,numpy as np,torch
from dual_objective import DualMeasureGW
def fit_star(weights,moments):
    means=moments/weights
    order=torch.argsort(means[1:].detach())+1;pooled=[0]
    for index in order.tolist():
        current=moments[pooled].sum()/weights[pooled].sum()
        if float(means[index].detach())<float(current.detach()):pooled.append(index)
        else:break
    common=moments[pooled].sum()/weights[pooled].sum()
    return torch.stack([common if i in pooled else means[i] for i in range(len(weights))])


class DualCommunityGW(DualMeasureGW):
 def __init__(self,layers,K,coefficient=.5,cost='square'):
  super().__init__(layers,K,coefficient)
  if cost not in ['square','kl']:raise ValueError(cost)
  if self.active_layers!=self.node.active_layers:raise ValueError('Matched active layer set required')
  self.cost=cost
  self.node_scale=torch.stack([C.values().sum() for C in self.node.C])
  self.node_nu=1+self.node.null/self.node.mu.square()
  nc=[];dc=[]
  for index,C,scale,pl,v in zip(self.active_layers,self.node.C,self.node_scale,self.degree_measure,self.volumes):
   nc.append((C.values()/scale*(C.values()/self.node.mu[len(nc)]).log()).sum())
   A=self.raw[len(dc)];ij=A.indices();dc.append((A.values()/v*(A.values()/(v*pl[ij[0]]*pl[ij[1]])).log()).sum())
  self.node_kl=torch.stack(nc);self.degree_kl=torch.stack(dc)
  if bool((self.node_kl<=1e-10).any()) or bool((self.degree_kl<=1e-10).any()):raise ValueError('Degenerate KL constant')
 def measure_profile(self,T,measure):
  W=T/self.p[:,None];mask=1-torch.eye(self.K,dtype=T.dtype)
  if measure=='node':
   q=(self.p@W)[None,:].expand(len(self.node.C),-1);graphs=self.node.C;scales=self.node_scale;nu=self.node_nu;constant=self.node_kl
  elif measure=='degree':
   q=self.degree_measure@W;graphs=self.raw;scales=self.volumes;nu=self.nu;constant=self.degree_kl
  else:raise ValueError(measure)
  blocks=torch.stack([W.T@torch.sparse.mm(A,W)/v for A,v in zip(graphs,scales)])
  weights=torch.cat(((q[:,:,None]*q[:,None,:]*mask).sum((1,2))[:,None],q.square()),1)
  moments=torch.cat(((blocks*mask).sum((1,2))[:,None],blocks.diagonal(dim1=1,dim2=2)),1)
  if bool(((weights.detach()<=0)|(moments.detach()<=0)).any()):raise ValueError('Positive group moments required')
  theta=torch.stack([fit_star(w,m) for w,m in zip(weights,moments)])
  if self.cost=='square':per=(nu-2*(moments*theta).sum(1)+(weights*theta.square()).sum(1))/(nu-1)
  else:per=(constant-(moments*theta.log()).sum(1)-1+(weights*theta).sum(1))/constant
  return per,dict(q=q,weights=weights,moments=moments,theta=theta)
 def layer_parts(self,T):return self.measure_profile(T,'node')[0],self.measure_profile(T,'degree')[0]
 def parts(self,T):
  node,degree=self.layer_parts(T);q=T.sum(0);mass=.02*(torch.relu(self.q_floor-q)/self.q_floor).square().sum()
  return (node.mean()+degree.mean())/2,mass,T.new_zeros(())
