import torch

def template_profile(q,mu,x,threshold=1e-12):
    safe=q.clamp_min(threshold)
    first=(q.square()/safe).sum(-1)
    second=(q.square()/safe.square()).sum(-1)
    determinant=(second-first.square()).clamp_min(torch.finfo(q.dtype).tiny)
    alpha=torch.relu(x-first*mu)/determinant
    base=mu-first*alpha
    boundary=base<0
    alpha=torch.where(boundary,x/second.clamp_min(torch.finfo(q.dtype).tiny),alpha)
    base=torch.where(boundary,torch.zeros_like(base),base)
    gain=2*base*mu+2*alpha*x-base.square()-2*base*alpha*first-alpha.square()*second
    return base,alpha,gain,first,second
