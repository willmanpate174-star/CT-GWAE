"""Source, budget, tensor, direct-GW, gradient and HVP checks. No training."""
import json
from pathlib import Path
import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
from run import ROOT, runtime, verify_package, graph, adapter, write, sha, read


def assert_close(a,b,tol=1e-9):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not np.isfinite(err) or err>tol:raise AssertionError((err,tol))
    return err


def main():
    if sys.version_info < (3,11):
        raise RuntimeError('The frozen training engine requires Python >=3.11; use a compatible environment.')
    verify_package();values=runtime()
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=values
    from objectives import AblationObjective,HardAblation
    tasks=read(ROOT/'config/tasks_A.json')
    assignment=read(ROOT/'config/assignment.json')
    assert assignment['this_package']=='A' and assignment['A']=='同伴' and assignment['B']=='用户本人'
    assert len(tasks)==200 and len({t['id'] for t in tasks})==200
    assert sum(t['kind']=='ablation' for t in tasks)==120
    assert sum(t['kind']=='sensitivity' for t in tasks)==80
    assert cfg['proposals_per_candidate']==600 and cfg['learned_eta_count']==12
    assert cfg['train_unroll_steps']==24
    assert len({(t['dataset'],t['variant'],t['S_train'],t['round']) for t in tasks})==200
    assert len(read(ROOT/'full_reference/winners.json'))==40
    for t in tasks:
        assert t['seed_base']==41000000+10000*t['canonical_index']
        assert t['S_train'] in (0,6,12,24,48)
    errors={}
    a=np.array([[0,3,2,0,0,0],[3,0,2,.05,0,0],[2,2,0,0,0,0],
                [0,.05,0,0,2,3],[0,0,0,2,0,2],[0,0,0,3,2,0]],float)
    layers=[sp.csr_matrix(a),sp.csr_matrix(a*np.array([1,2,1,1,2,1])[:,None]*np.array([1,2,1,1,2,1])[None,:])]
    k=3;torch.manual_seed(731)
    logits=torch.randn(6,k,dtype=torch.float64)*.3
    logits[:3,0]+=3;logits[3:,1]+=3
    T=(torch.softmax(logits,1)*.98+.02/k)/6
    z=torch.randn_like(T);z=z-z.mean(1,keepdim=True);z=z/z.norm()
    for variant in ('Full','shared_coeff','no_dist_norm','no_mass_correction'):
        obj=AblationObjective(layers,k,variant)
        t=T.clone().requires_grad_(True);r=obj.profiles(t);direct=[];grams=[];rhs=[]
        for l,layer in enumerate(obj.raw):
            mu=obj.source_measures[l];q=r['q'][l];U=mu[:,None]*(t/obj.p[:,None])
            raw=layer.to_dense();den=mu[:,None]*mu[None,:]
            R=torch.where(den>0,raw/den,torch.zeros_like(raw))
            Bt=r['template'][l]
            four=((R[:,:,None,None]-Bt[None,None,:,:])**2*
                  U[:,None,:,None]*U[None,:,None,:]).sum()
            direct.append(four if variant=='no_dist_norm' else four/obj.null[l])
            basis=(torch.eye(k,dtype=t.dtype) if variant=='no_mass_correction' else torch.diag(1/q))
            qw=q[:,None]*q[None,:]
            g=np.array([[1.,float((qw*basis).sum().detach())],
                        [float((qw*basis).sum().detach()),float((qw*basis.square()).sum().detach())]])
            cross=torch.einsum('ia,ij,jb->ab',U,R,U)
            rr=np.array([1.,float((cross*basis).sum().detach())])
            grams.append(g);rhs.append(rr)
            if variant!='shared_coeff':
                chol=np.linalg.cholesky(g);beta=nnls(chol.T,np.linalg.solve(chol,rr))[0]
                errors[variant+f'_nnls_{l}']=assert_close(beta,[float(r['b'][l].detach()),float(r['alpha'][l].detach())])
        if variant=='shared_coeff':
            w=1/obj.null.numpy();g=sum(wl*gl for wl,gl in zip(w,grams));rr=sum(wl*rl for wl,rl in zip(w,rhs))
            chol=np.linalg.cholesky(g);beta=nnls(chol.T,np.linalg.solve(chol,rr))[0]
            errors['shared_joint_nnls']=assert_close(beta,[float(r['b'][0].detach()),float(r['alpha'][0].detach())])
        penalty=.02*(torch.relu(obj.q_floor-t.sum(0))/obj.q_floor).square().sum()
        reference=torch.stack(direct).mean()+penalty
        actual=obj.loss(t)
        errors[variant+'_direct_GW']=assert_close(float(actual.detach()),float(reference.detach()))
        g=torch.autograd.grad(actual,t,create_graph=True)[0]
        gd=torch.autograd.grad(reference,t,create_graph=True)[0]
        # Both formulas are identical on fixed-row transports; compare tangent derivatives.
        errors[variant+'_gradient']=assert_close((g-g.mean(1,keepdim=True)).detach().numpy(),
                                               (gd-gd.mean(1,keepdim=True)).detach().numpy(),1e-8)
        hv=torch.autograd.grad((g*z).sum(),t)[0]
        eps=1e-6
        def grad_at(v):
            v=v.detach().requires_grad_(True)
            return torch.autograd.grad(obj.loss(v),v)[0]
        fd=(grad_at(t+eps*z)-grad_at(t-eps*z))/(2*eps)
        errors[variant+'_HVP']=assert_close(hv.detach().numpy(),fd.numpy(),3e-5)
        scorer=HardAblation(B.Hard(layers,k),variant)
        labels=np.array([0,0,0,1,1,2]);hard=scorer.score(labels)
        onehot=torch.tensor(np.eye(k)[labels]/6,dtype=torch.float64)
        errors[variant+'_hard_soft']=assert_close(hard['objective'],float(obj.loss(onehot)))
        collapsed=scorer.score(np.zeros(6,dtype=int));assert np.isfinite(collapsed['objective'])
        # A layer can have zero source mass for a globally nonempty community.
        sparse_case=layers[0].copy().tolil();sparse_case[5,:]=0;sparse_case[:,5]=0
        hard_layers=[sparse_case.tocsr(),layers[1]]
        hard=HardAblation(B.Hard(hard_layers,k),variant).score(np.array([0,0,1,1,1,2]))
        losses=[]
        for layer,pf in zip(hard_layers,hard['profiles']):
            mu=np.array(pf['mu']);vol=float(layer.sum());prod=mu[:,None]*mu[None,:]
            rel=np.divide(layer.toarray()/vol,prod,out=np.zeros_like(prod),where=prod>0)
            matrix=np.full((k,k),pf['b']);np.fill_diagonal(matrix,pf['template_diagonal'])
            U=mu[:,None]*np.eye(k)[[0,0,1,1,1,2]]
            loss=((rel[:,:,None,None]-matrix[None,None,:,:])**2*
                  U[:,None,:,None]*U[None,:,None,:]).sum()
            losses.append(loss if variant=='no_dist_norm' else loss/pf['null'])
        errors[variant+'_zero_layer_mass']=assert_close(hard['objective'],np.mean(losses)+hard['node_mass'])
    # NNLS b=0 branch is genuine and must not be omitted.
    from objectives import inverse_coeff
    b,alpha=inverse_coeff(torch.tensor(5.,dtype=torch.float64),3)
    assert float(b)==0 and abs(float(alpha)-5/3)<1e-12
    # Every layer source must still be sqrt-degree; copy/entry equivalence for Full.
    layers,m=graph('aucs',np)
    net,obj,_=B.construct(layers,m['K'],41000000,'restored','original-control')
    for s in (0,6,12,24,48):
        t=dict(variant='Full',S_train=s)
        proxy=adapter(values,t)
        T1,h1,loss1,_=proxy.forward(net,obj,24,False)
        eta=net.log_eta[:s] if s<=12 else torch.cat([net.log_eta,net.log_eta[-1].repeat(s-12)])
        T2,h2=B.E.infer(obj,net().detach(),eta.detach(),create_graph=False)
        errors[f'S{s}_forward']=assert_close(T1.numpy(),T2.numpy(),0)
        assert len(h1)==len(h2)==s and net.log_eta.numel()==12
    t1,h1,l1,_=B.forward(net,obj,100,False)
    t2,h2,l2,_=adapter(values,dict(variant='Full',S_train=24)).forward(net,obj,100,False)
    errors['frozen_full_S100']=assert_close(t1.numpy(),t2.numpy(),0)
    info=dict(status='passed',scope='No training; direct GW/NNLS/tangent gradients/HVP/hard scoring and frozen-forward parity',
              torch=torch.__version__,python=sys.version,errors=errors,formal_candidates=2000,
              formal_proposals=1200000,source_manifest=sha(ROOT/'config/source_manifest.json'))
    write(ROOT/'checks/preflight_pass.json',info)
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
