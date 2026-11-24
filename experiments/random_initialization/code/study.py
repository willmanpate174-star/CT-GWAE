"""Paired initialization intervention: frozen Full step sizes, random T0, S=100.

No GCN is constructed in formal random-initialization runs. Full GCN is loaded
only for the four preflight replays. Truth is opened only after selection locks.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse, csv, hashlib, json, time, traceback
from functools import cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
DATASETS=('aucs','ckm','dkpol','workplace')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,p)
def mapped(p):return Path(str(p).replace('C:\\Users\\Dingjunhao\\Documents','D:\\用户文件\\文档'))
def table(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@cache
def runtime():
    sys.path.insert(0,str(ROOT/'frozen'))
    from bootstrap import activate
    v=activate();torch=v[6]
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    return v

def graph(ds,np):
    import scipy.sparse as sp
    m=read(ROOT/'data'/ds/'metadata.json')
    with np.load(ROOT/'data'/ds/'graphs.npz',allow_pickle=False) as z:
        layers=[sp.csr_matrix((z[f'data_{i}'],z[f'indices_{i}'],z[f'indptr_{i}']),shape=(m['N'],m['N'])) for i in range(m['L'])]
    for a in layers:
        a.check_format(full_check=True)
        assert (a-a.T).nnz==0 and not a.diagonal().any()
    return layers,m

def verify():
    plan=read(ROOT/'config/plan.json')
    for rel,digest in plan['input_hashes'].items():assert sha(ROOT/rel)==digest,rel
    assert plan['steps']==100 and plan['rounds']==10 and plan['starts']==10
    assert plan['initial_logits_distribution']=='iid Normal(0,1)'
    assert len(plan['candidates'])==400
    return plan


def eta100(torch,np,c):
    x=torch.from_numpy(np.load(ROOT/c['relative_reference']/'log_eta.npy',allow_pickle=False)).clone()
    return torch.cat([x,x[-1].repeat(88)])


def run_pool(args):
    ds,r=args;plan=read(ROOT/'config/plan.json');v=runtime()
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=v
    from geometric_train_model import LayerGeometricSingleGW
    from geometric_hard_model import LayerGeometricFixedPartition
    layers,m=graph(ds,np);obj=LayerGeometricSingleGW(layers,m['K']);scorer=LayerGeometricFixedPartition(layers,m['K'])
    folder=ROOT/'results'/ds/f'r{r:02d}'
    folder.mkdir(parents=True,exist_ok=True);records=[]
    for c in [x for x in plan['candidates'] if x['dataset']==ds and x['round']==r]:
        j=c['start'];dest=folder/f'j{j:02d}'
        if (dest/'success.json').exists():
            receipt=read(dest/'success.json')
            for name,digest in receipt['artifacts'].items():assert sha(dest/name)==digest
            old=receipt['record']
            assert (old['dataset'],old['round'],old['start'],old['seed'])==(ds,r,j,c['seed'])
            assert old['steps']==100 and old['GCN_evaluations']==0 and old['optimizer_updates']==0
            records.append(old)
            continue
        if dest.exists():raise RuntimeError('Incomplete candidate retained; inspect explicitly: '+str(dest))
        dest.mkdir()
        started=time.perf_counter()
        gen=torch.Generator().manual_seed(c['seed'])
        logits=torch.randn((m['N'],m['K']),dtype=torch.float64,generator=gen)
        initial=B.E.R.initial_transport(obj,logits).detach()
        eta=eta100(torch,np,c)
        T,hist=B.E.infer(obj,logits,eta,create_graph=False)
        checks=B.coupling_checks(obj,T)
        loss0=float(obj.loss(initial));loss=float(obj.loss(T))
        assert len(hist)==100 and hist[-1]==loss and loss<=loss0+1e-12
        assert all(y<=x+1e-12 for x,y in zip(hist,hist[1:]))
        raw_labels=T.argmax(1).numpy()
        labels,receipt,raw,hard=wrapper.decode_and_score(T.numpy(),raw_labels,m['K'],scorer,decoder,np)
        profile=obj.profiles(T)
        np.savez_compressed(dest/'transports.npz',initial=initial.numpy(),final=T.numpy(),raw_logits=logits.numpy(),raw_step_parameters=eta.numpy())
        np.save(dest/'labels.npy',labels);np.save(dest/'raw_argmax.npy',raw_labels)
        np.savez_compressed(dest/'templates.npz',B=profile['template'].detach().numpy(),q=profile['q'].detach().numpy())
        write(dest/'hard_score.json',hard);write(dest/'decoder.json',receipt)
        record=dict(dataset=ds,round=r,start=j,seed=c['seed'],steps=100,initial_loss=loss0,soft_objective=loss,
            hard_objective=hard['objective'],labels_sha256=sha(dest/'labels.npy'),seconds=time.perf_counter()-started,
            GCN_evaluations=0,optimizer_updates=0,step_source=c['relative_reference'],truth_read=False,
            raw_argmax_classes=int(len(np.unique(raw_labels))),decoder_changed_nodes=int(np.count_nonzero(labels!=raw_labels)))
        write(dest/'trace.json',dict(record=record,objective_history=hist,checks=checks))
        write(dest/'success.json',dict(record=record,artifacts={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
        records.append(record)
    assert len(records)==10
    winner=min(records,key=lambda x:(x['hard_objective'],x['start']))
    lock=dict(dataset=ds,round=r,records=records,winner=winner,truth_read=False,planned_starts=10,valid_starts=10)
    write(folder/'selection_before_truth.json',lock)
    return dict(dataset=ds,round=r,starts=10,seconds=sum(x['seconds'] for x in records))

def evaluate(ds,labels,np):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    meta=read(ROOT/'data'/ds/'metadata.json')
    with np.load(ROOT/'data'/ds/'evaluation.npz',allow_pickle=False) as z:
        mask=z['mask'].astype(bool)&(z['truth']>=0)
        t=np.tile(z['truth'][mask],meta['L']);y=np.tile(labels[mask],meta['L'])
    _,ti=np.unique(t,return_inverse=True);_,yi=np.unique(y,return_inverse=True)
    a=np.zeros((ti.max()+1,yi.max()+1),dtype=np.int64);np.add.at(a,(ti,yi),1)
    i,j=linear_sum_assignment(-a)
    return dict(ari=float(adjusted_rand_score(t,y)),nmi=float(normalized_mutual_info_score(t,y,average_method='arithmetic')),acc=float(a[i,j].sum()/len(t)))

def summarize():
    import numpy as np
    plan=verify();locks=[]
    for ds in DATASETS:
        for r in range(10):
            folder=ROOT/'results'/ds/f'r{r:02d}';lock=read(folder/'selection_before_truth.json')
            assert lock['valid_starts']==10 and lock['truth_read'] is False
            assert lock['winner']==min(lock['records'],key=lambda x:(x['hard_objective'],x['start']))
            for x in lock['records']:
                d=folder/f"j{x['start']:02d}";receipt=read(d/'success.json')
                for name,digest in receipt['artifacts'].items():assert sha(d/name)==digest
            locks.append(lock)
    write(ROOT/'checks/selection_locks_before_truth.json',dict(complete=True,pools=40,starts=400,locks={f"{x['dataset']}/r{x['round']:02d}":sha(ROOT/'results'/x['dataset']/f"r{x['round']:02d}"/'selection_before_truth.json') for x in locks}))
    rows=[]
    refs=read(ROOT/'reference/full_winners.json')
    for lock in locks:
        w=lock['winner'];ds=w['dataset'];r=w['round'];j=w['start']
        y=np.load(ROOT/'results'/ds/f'r{r:02d}'/f'j{j:02d}'/'labels.npy',allow_pickle=False)
        rows.append(dict(dataset=ds,variant='Random initialization, matched steps',round=r,start=j,hard_objective=w['hard_objective'],**evaluate(ds,y,np)))
        f=next(x for x in refs if x['dataset']==ds and x['round']==r)
        y=np.load(ROOT/'reference'/ds/f"r{r:02d}_j{f['start']:02d}"/'labels.npy',allow_pickle=False)
        assert sha(ROOT/'reference'/ds/f"r{r:02d}_j{f['start']:02d}"/'labels.npy')==f['labels_sha256']
        metrics=evaluate(ds,y,np)
        assert all(abs(metrics[k]-f[k])<1e-12 for k in ('ari','nmi','acc'))
        rows.append(dict(dataset=ds,variant='Full (frozen reference)',round=r,start=f['start'],hard_objective=f['hard_objective'],**metrics))
    table(ROOT/'summary/round_winners.csv',rows)
    summary=[]
    for ds in DATASETS:
        for variant in ('Full (frozen reference)','Random initialization, matched steps'):
            rr=[x for x in rows if x['dataset']==ds and x['variant']==variant]
            d=dict(dataset=ds,variant=variant,rounds=len(rr))
            for k in ('ari','nmi','acc','hard_objective'):
                d[k+'_mean']=float(np.mean([x[k] for x in rr]));d[k+'_sd']=float(np.std([x[k] for x in rr],ddof=1))
            summary.append(d)
    table(ROOT/'summary/summary.csv',summary)
    changed=sum(x['decoder_changed_nodes'] for lock in locks for x in lock['records'])
    write(ROOT/'summary/complete.json',dict(status='passed',new_candidates=400,rounds=40,steps_per_candidate=100,GCN_training_runs=0,Full_retrained=False,all_Full_metrics_reproduced=True,decoder_changed_nodes_total=changed,finished=datetime.now().isoformat(),plan_sha256=sha(ROOT/'config/plan.json')))
    print('COMPLETE: 400/400 starts, 40/40 locked rounds; Full metrics verified',flush=True)

def run(workers):
    plan=verify()
    started=time.perf_counter();done=0
    write(ROOT/'progress.json',dict(status='running',completed_starts=0,planned_starts=400,workers=workers,pid=os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run_pool,(ds,r)):(ds,r) for ds in DATASETS for r in range(10)}
            for f in as_completed(futures):
                result=f.result();done+=10
                write(ROOT/'progress.json',dict(status='running',completed_starts=done,planned_starts=400,workers=workers,pid=os.getpid(),last_pool=result,elapsed_seconds=time.perf_counter()-started,updated=datetime.now().isoformat()))
                print(f"PROGRESS {done}/400 {result['dataset']} r{result['round']:02d}",flush=True)
        summarize()
        write(ROOT/'progress.json',dict(status='complete',completed_starts=400,planned_starts=400,elapsed_seconds=time.perf_counter()-started,updated=datetime.now().isoformat()))
    except BaseException:
        write(ROOT/'progress.json',dict(status='failed',completed_starts=done,planned_starts=400,error=traceback.format_exc(),updated=datetime.now().isoformat()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['check','run','summarize'])
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.action=='check':
        verify()
        print('Input hashes and all 400 paired step-parameter references verified; no training.')
    elif args.action=='run':run(args.workers)
    else:summarize()
