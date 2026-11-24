"""Registered local paired pilot. Full and shared template, four datasets, j=0."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types
sys.dont_write_bytecode = True
import run as original
ROOT = original.ROOT


def init():
    original.verify_package()
    values = original.runtime()
    from shared_template import SharedTemplateObjective, SharedTemplateHard
    B = values[0]
    def adapter(values, task):
        proxy = types.SimpleNamespace(**B.__dict__)
        def construct(layers, k, seed, mode, variant):
            net, obj, base = B.construct(layers, k, seed, mode, variant)
            if task['variant'] == 'shared_template':
                new = SharedTemplateObjective(layers, k)
                new.hard_diagnostic = SharedTemplateHard(obj.hard_diagnostic)
                obj = new
            return net, obj, base
        proxy.construct = construct
        return proxy
    original.adapter = adapter
    return values


def preflight():
    values = init()
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support = values
    import scipy.sparse as sp
    from scipy.optimize import nnls
    from shared_template import SharedTemplateObjective,SharedTemplateHard,torch_fit
    a=np.array([[0,3,2,0,0,0],[3,0,2,.05,0,0],[2,2,0,0,0,0],
                [0,.05,0,0,2,3],[0,0,0,2,0,2],[0,0,0,3,2,0]],float)
    v=np.array([1,2,1,1,2,1]); layers=[sp.csr_matrix(a),sp.csr_matrix(a*v[:,None]*v[None,:])]
    torch.manual_seed(731)
    logits=torch.randn(6,3,dtype=torch.float64)*.3
    logits[range(6),[0,0,0,1,1,2]]+=3
    t=(torch.softmax(logits,1)*.98+.02/3)/6
    t.requires_grad_(True)
    obj=SharedTemplateObjective(layers,3); r=obj.profiles(t)
    errors={}
    def check(name,a,b,tol):
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        assert np.isfinite(err) and err<=tol,(name,err,tol)
        errors[name]=err
    check('identical_templates',r['template'][0].detach(),r['template'][1].detach(),0)
    # Explicit four-index GW, including the original layer source measures.
    W=t/obj.p[:,None];direct=[];xs=[];ys=[]
    for i,raw in enumerate(obj.raw):
        mu=obj.source_measures[i]; prod=mu[:,None]*mu[None,:]
        c=raw.to_dense()/prod
        U=mu[:,None]*W; template=r['template'][i]
        direct.append(((c[:,:,None,None]-template[None,None,:,:]).square()*
                       U[:,None,:,None]*U[None,:,None,:]).sum()/obj.null[i])
        # NNLS design for fixed current T.
        weight=(prod[:,:,None,None]*(obj.null[i]**-1)).expand(6,6,3,3)
        # Equivalent template-space weights q_k q_l, direct first moment.
        qq=r['q'][i][:,None]*r['q'][i][None,:]
        root=(qq/obj.null[i]).sqrt()
        D=torch.diag(r['inverse_common_mass'])
        block=W.T@raw.to_dense()@W
        target=block/qq
        xs.append(torch.stack((root.flatten(),(root*D).flatten()),1).detach().numpy())
        ys.append((root*target).detach().numpy().flatten())
    expected=nnls(np.concatenate(xs),np.concatenate(ys))[0]
    check('NNLS',expected,[float(r['b'][0].detach()),float(r['alpha'][0].detach())],1e-10)
    node=obj.parts(t)[1]; loss=obj.loss(t); definition=torch.stack(direct).mean()+node
    check('direct_GW',float(loss.detach()),float(definition.detach()),1e-10)
    g=torch.autograd.grad(loss,t,create_graph=True)[0]
    gd=torch.autograd.grad(definition,t,retain_graph=True)[0]
    check('direct_tangent_gradient',(g-g.mean(1,keepdim=True)).detach(),
          (gd-gd.mean(1,keepdim=True)).detach(),1e-8)
    z=torch.randn_like(t);z-=z.mean(1,keepdim=True);z/=z.norm()
    hv=torch.autograd.grad((g*z).sum(),t)[0]
    def grad(v):
        v=v.detach().requires_grad_(True)
        return torch.autograd.grad(obj.loss(v),v)[0]
    fd=(grad(t+1e-6*z)-grad(t-1e-6*z))/(2e-6)
    check('HVP',hv.detach(),fd,3e-5)
    for label in ([0,0,0,1,1,2],):
        hard=SharedTemplateHard(B.Hard(layers,3)).score(np.array(label))
        soft=float(obj.loss(torch.tensor(np.eye(3)[label]/6,dtype=torch.float64)))
        check('hard_soft_'+str(len(set(label))),hard['objective'],soft,1e-10)
    collapsed=SharedTemplateHard(B.Hard(layers,3)).score(np.zeros(6,dtype=int))
    check('collapsed_canonical_baseline',collapsed['structure'],1.,1e-10)
    # One source layer lacks a community; the pooled mass remains positive.
    sparse=layers[0].tolil();sparse[5,:]=0;sparse[:,5]=0
    zero_layers=[sparse.tocsr(),layers[1]]
    zero_obj=SharedTemplateObjective(zero_layers,3)
    labels=np.array([0,0,1,1,1,2]); hard=SharedTemplateHard(B.Hard(zero_layers,3)).score(labels)
    losses=[]
    for layer,pf in zip(zero_layers,hard['profiles']):
        mu=np.array(pf['mu']);prod=mu[:,None]*mu[None,:]
        rel=np.divide(layer.toarray()/layer.sum(),prod,out=np.zeros_like(prod),where=prod>0)
        template=np.full((3,3),pf['b']);np.fill_diagonal(template,pf['template_diagonal'])
        U=mu[:,None]*np.eye(3)[labels]
        losses.append(((rel[:,:,None,None]-template[None,None,:,:])**2*
                       U[:,None,:,None]*U[None,:,None,:]).sum()/pf['null'])
    check('zero_layer_mass',hard['objective'],np.mean(losses)+hard['node_mass'],1e-10)
    # One layer: common mass equals layer mass, recovering the original loss.
    one=SharedTemplateObjective(layers[:1],3);full=values[0].Model(layers[:1],3)
    check('single_layer_Full_parity',float(one.loss(t).detach()),float(full.loss(t).detach()),1e-10)
    original.write(ROOT/'checks/shared_template_preflight.json',dict(status='passed',errors=errors,
        scope='No training; joint NNLS/direct GW/tangent gradient/HVP/hard support/single-layer parity',
        python=sys.version,torch=torch.__version__))
    print(json.dumps(errors,indent=2),flush=True)


def worker(index):
    plan=original.read(ROOT/'protocol.json');task=plan['tasks'][index]
    values=init(); old_save=original.merged_save
    def save(path,value):
        old_save(path,value)
        p=Path(path)
        if p.parent.name=='proposal_logs':
            i=int(p.stem)
            if i%25==0 or i==599:
                original.write(ROOT/'progress'/f'{index:02d}.json',dict(task=task,
                    completed_proposals=i+1,after_loss=value['after_loss'],updated=time.time()))
                print(f"{task['id']} proposals={i+1}/600 loss={value['after_loss']:.12g}",flush=True)
    original.merged_save=save
    row=original.run_candidate(values,task,0,ROOT/'results'/task['dataset']/task['variant']/'j00')
    original.verify_package()
    print(json.dumps(row),flush=True)


def controller():
    if not (ROOT/'checks/shared_template_preflight.json').exists():
        raise RuntimeError('Run preflight first')
    plan=original.read(ROOT/'protocol.json')
    if (ROOT/'launch.json').exists():raise RuntimeError('Existing run; do not duplicate')
    (ROOT/'logs').mkdir(exist_ok=True)
    processes=[]
    original.write(ROOT/'launch.json',dict(started=time.time(),python=sys.executable,
                                         plan_sha256=original.sha(ROOT/'protocol.json')))
    for i,task in enumerate(plan['tasks']):
        log=(ROOT/'logs'/f'{i:02d}_{task["dataset"]}_{task["variant"]}.log').open('w',encoding='utf-8')
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--index',str(i)],
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT)
        processes.append((p,log,task))
    original.write(ROOT/'processes.json',[dict(pid=p.pid,task=t) for p,_,t in processes])
    codes=[]
    for p,log,t in processes:
        codes.append(dict(task=t,exit_code=p.wait()));log.close()
    original.write(ROOT/'process_exits.json',codes)
    if any(c['exit_code'] for c in codes):
        original.write(ROOT/'failure.json',dict(scope='pilot',exits=codes));raise RuntimeError('Pilot failed; inspect retained logs')
    # Commit all eight predictions before reading any ground truth.
    locked=[]
    for task in plan['tasks']:
        folder=ROOT/'results'/task['dataset']/task['variant']/'j00'
        success=original.read(folder/'success.json')
        for rel,h in success['artifacts'].items():
            assert original.sha(folder/rel)==h,folder/rel
        locked.append(dict(task=task,labels_sha256=original.sha(folder/'labels.npy')))
    original.write(ROOT/'all_predictions_lock.json',dict(predictions=locked,truth_read=False))
    import summarize
    import numpy as np
    rows=[]
    for item in locked:
        task=item['task'];folder=ROOT/'results'/task['dataset']/task['variant']/'j00'
        labels=np.load(folder/'labels.npy',allow_pickle=False)
        r=original.read(folder/'success.json')['record']
        templates=np.load(folder/'templates.npz',allow_pickle=False)['B']
        diff=float(np.max(np.abs(templates-templates[0])))
        if task['variant']=='shared_template':assert diff==0
        rows.append(dict(dataset=task['dataset'],variant=task['variant'],seed=r['seed'],
            proposals=r['proposals'],seconds=r['seconds'],template_max_layer_difference=diff,
            **summarize.evaluate(task['dataset'],labels)))
    summarize.csv_write(ROOT/'paired_metrics.csv',rows)
    original.write(ROOT/'summary.json',dict(status='passed_local_paired_pilot',rows=rows,
        formal_repeats=0,cold_starts_per_condition_per_dataset=1,
        proposals_per_start=600,selection_by_labels=False,
        description='Eight fixed cold-start diagnostics; not ten-run best-of-ten formal results'))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','worker','controller'])
    parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    {'preflight':preflight,'worker':lambda:worker(args.index),'controller':controller}[args.action]()
