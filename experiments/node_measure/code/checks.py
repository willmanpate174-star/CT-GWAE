"""Non-training numerical preflight; no ground truth access."""
from common import *
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
def main():
    if sys.version_info<(3,11):raise RuntimeError('Python3.11 or newer required by frozen training engine')
    stamp=verify();v=runtime();B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=v
    from measures import MeasureHard,MeasureObjective
    tasks=read(ROOT/'config/tasks.json');plan=read(ROOT/'config/plan.json')
    assert len(tasks)==800 and len({t['output'] for t in tasks})==800
    for t in tasks:
        assert t['seed']==plan['seed_bases'][t['dataset']]+100*t['round']+t['start']
        assert t['proposals']==600 and t['S_train']==24 and t['S_infer']==100
    assert cfg['proposals_per_candidate']==600 and cfg['learned_eta_count']==12
    assert len(read(ROOT/'reference/winners.json'))==40
    errors={}
    def near(name,a,b,tol=1e-9):
        a=np.asarray(a);b=np.asarray(b);e=float(np.max(np.abs(a-b)))
        if not np.isfinite(e) or e>tol:raise AssertionError((name,e,tol))
        errors[name]=e
    a=np.array([[0,3,2,0,0,0,0],[3,0,2,.05,0,0,0],[2,2,0,0,0,0,0],
       [0,.05,0,0,2,3,0],[0,0,0,2,0,2,0],[0,0,0,3,2,0,0],[0,0,0,0,0,0,0]],float)
    f=np.array([1,2,1,1,2,1,1.]);layers=[sp.csr_matrix(a),sp.csr_matrix(a*f[:,None]*f[None,:])]
    k=3;torch.manual_seed(731)
    logits=torch.randn(7,k,dtype=torch.float64)*.3;logits[:3,0]+=3;logits[3:6,1]+=3
    T=(torch.softmax(logits,1)*.98+.02/k)/7
    z=torch.randn_like(T);z-=z.mean(1,keepdim=True);z/=z.norm()
    support={}
    for mode in ('sqrt','uniform','degree'):
        obj=MeasureObjective(layers,k,mode);t=T.clone().requires_grad_(True)
        r=obj.profiles(t);direct=[]
        for l,layer in enumerate(obj.raw):
            mu=obj.source_measures[l];q=r['q'][l];U=mu[:,None]*(t/obj.p[:,None])
            raw=layer.to_dense();den=mu[:,None]*mu[None,:]
            # Safe division, no 0/0 gradients on absent mass support.
            safe=torch.where(den>0,den,torch.ones_like(den));R=torch.where(den>0,raw/safe,torch.zeros_like(raw))
            Bt=r['template'][l]
            four=((R[:,:,None,None]-Bt[None,None,:,:]).square()*U[:,None,:,None]*U[None,:,None,:]).sum()
            direct.append(four/obj.null[l])
            basis=torch.diag(1/q);qw=q[:,None]*q[None,:]
            gram=np.array([[1.,float((qw*basis).sum().detach())],
                 [float((qw*basis).sum().detach()),float((qw*basis.square()).sum().detach())]])
            cross=torch.einsum('ia,ij,jb->ab',U,R,U)
            rhs=np.array([1.,float((cross*basis).sum().detach())]);ch=np.linalg.cholesky(gram)
            beta=nnls(ch.T,np.linalg.solve(ch,rhs))[0]
            near(mode+f'_NNLS_l{l}',beta,[float(r['b'][l].detach()),float(r['alpha'][l].detach())])
        reference=torch.stack(direct).mean()+.02*(torch.relu(obj.q_floor-t.sum(0))/obj.q_floor).square().sum()
        loss=obj.loss(t);near(mode+'_direct_four_index_GW',float(loss.detach()),float(reference.detach()))
        g=torch.autograd.grad(loss,t,create_graph=True)[0]
        gr=torch.autograd.grad(reference,t,create_graph=True)[0]
        near(mode+'_direct_tangent_gradient',(g-g.mean(1,keepdim=True)).detach().numpy(),
            (gr-gr.mean(1,keepdim=True)).detach().numpy(),1e-8)
        near(mode+'_analytic_gradient',g.detach().numpy(),obj.analytic_gradient(t).detach().numpy(),1e-8)
        hv=torch.autograd.grad((g*z).sum(),t)[0]
        near(mode+'_analytic_HVP',hv.numpy(),obj.analytic_hvp(t,z).detach().numpy(),1e-7)
        eps=1e-6
        def grad(at):
            at=at.detach().requires_grad_(True);return torch.autograd.grad(obj.loss(at),at)[0]
        near(mode+'_finite_difference_HVP',hv.numpy(),((grad(t+eps*z)-grad(t-eps*z))/(2*eps)).numpy(),3e-5)
        scorer=MeasureHard(layers,k,mode)
        for labels in (np.array([0,0,0,1,1,2,2]),np.array([0,0,1,1,1,1,2]),np.zeros(7,dtype=int)):
            hard=scorer.score(labels);losses=[]
            for pf in hard['profiles']:
                mu=np.array(pf['mu']);la=layers[pf['layer_id']];raw=la.toarray()/float(la.sum());den=mu[:,None]*mu[None,:]
                rel=np.divide(raw,den,out=np.zeros_like(raw),where=den>0)
                M=np.eye(k)[labels];U=mu[:,None]*M;Bt=np.full((k,k),pf['b']);np.fill_diagonal(Bt,pf['template_diagonal'])
                losses.append(((rel[:,:,None,None]-Bt[None,None,:,:])**2*U[:,None,:,None]*U[None,:,None,:]).sum()/pf['null'])
            near(mode+'_hard_support_'+str(labels.tolist()),hard['objective'],np.mean(losses)+hard['node_mass'])
        support[mode]=obj.source_measures[:,6].tolist()
    assert support['uniform']==[1/7,1/7] and support['sqrt']==support['degree']==[0.,0.]
    per_graph={}
    for ds in plan['datasets']:
        layers,m=graph(ds,np);seed=plan['seed_bases'][ds]
        net,original,base=B.construct(layers,m['K'],seed,'restored','original-control')
        generalized=MeasureObjective(layers,m['K'],'sqrt')
        near(ds+'_Full_mu_exact',original.source_measures.numpy(),generalized.source_measures.numpy(),0)
        near(ds+'_Full_nu_exact',original.nu.numpy(),generalized.nu.numpy(),0)
        t1,h1,l1,c1=B.forward(net,original,24,False);t2,h2,l2,c2=B.forward(net,generalized,24,False)
        near(ds+'_Full_S24_exact',t1.numpy(),t2.numpy(),0)
        modes={}
        for mode in ('uniform','degree'):
            net2,obj,base2=adapt(v,mode).construct(layers,m['K'],seed,'restored','original-control')
            assert B.state_digest(base2)==B.state_digest(base)
            near(ds+'_'+mode+'_features_exact',net.X.numpy(),net2.X.numpy(),0)
            for j,layerid in enumerate(obj.active_layers):
                d=np.asarray(layers[layerid].sum(1)).ravel()
                expected=np.full(m['N'],1/m['N']) if mode=='uniform' else d/d.sum()
                near(ds+'_'+mode+f'_formula_l{layerid}',obj.source_measures[j].numpy(),expected,0)
            ti,hi,li,ci=B.forward(net2,obj,24,False)
            assert np.isfinite(float(li)) and len(hi)==24
            modes[mode]=dict(active_layers=obj.active_layers,zero_mass_by_layer=(obj.source_measures==0).sum(1).tolist(),
                             initial_S24=float(li),cold_state_digest=B.state_digest(base2))
        per_graph[ds]=dict(N=m['N'],L=m['L'],K=m['K'],modes=modes)
    import scipy,sklearn,psutil
    info=dict(status='passed',source_manifest=stamp,no_training=True,truth_read=False,
        scope='direct4-index GW, NNLS, tangent gradient, analytic gradient/HVP and finite difference, hard zero-support, original Full parity,4real cold states',
        errors=errors,per_graph=per_graph,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,physical_cpus=psutil.cpu_count(logical=False),
        formal_candidates=800,proposals_per_candidate=600)
    write(ROOT/'checks/preflight_pass.json',info)
    print(json.dumps(dict(status='passed',tests=len(errors),graphs=per_graph,truth_read=False,training=False),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
