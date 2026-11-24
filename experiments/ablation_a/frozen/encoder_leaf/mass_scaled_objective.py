"""Independent profiled GW prototype B_ab=b+alpha*1[a=b]/q_a.
Positive free target masses; nonnegative b and alpha, per-layer coefficients.
This changes the template family, not the graph input or neural encoder.
"""
import torch
from data_io import ProfiledGW
class MassScaledGW(ProfiledGW):
    def coefficients(self,T):
        q=T.sum(0)
        # Valid positive transports have q>0. Clamp is only a numerical guard.
        qsafe=q.clamp_min(1e-12)
        xg=torch.stack([((T*torch.sparse.mm(c,T)).sum(0)/qsafe).sum() for c in self.C])
        alpha=torch.relu(xg-self.mu)/(self.K-1)
        boundary=xg>self.K*self.mu
        alpha=torch.where(boundary,xg/self.K,alpha)
        base=torch.where(boundary,torch.zeros_like(alpha),self.mu-alpha)
        return q,base,alpha,xg
    def loss(self,T):
        q,base,alpha,xg=self.coefficients(T)
        gain=2*base*self.mu+2*alpha*xg-base.square()-2*base*alpha-self.K*alpha.square()
        E=(self.null+self.mu.square()-gain)/self.null
        mass=.02*(torch.relu(self.q_floor-q)/self.q_floor).square().sum()
        return E.mean()+mass
    def decode_templates(self,T):
        q,base,alpha,xg=self.coefficients(T)
        return dict(active_layers=self.active_layers,theta_out=base.detach().tolist(),theta_in=(base[:,None]+alpha[:,None]/q.clamp_min(1e-12)).detach().tolist(),alpha=alpha.detach().tolist(),target_mass=q.detach().tolist())
