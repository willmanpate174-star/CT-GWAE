"""Frozen-budget common-template ablation. Truth is read only by finish()."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'runtime/code'))
import run as original


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=original.write


def verify():
    for rel,digest in read(ROOT/'source_manifest.json').items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Registered source/input changed: '+rel)
    original.verify_package()
    protocol=read(ROOT/'protocol.json')
    assert protocol['Full_training_runs']==0 and len(protocol['tasks'])==40
    assert all(t['variant']=='shared_template' for t in protocol['tasks'])
    assert sorted(i for pair in protocol['pairs'] for i in pair)==list(range(40))


def runtime():
    verify()
    import local_test
    return local_test.init()


def environment():
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','scikit-learn','threadpoolctl')}
    expected={'numpy':'2.5.1','scipy':'1.18.0','torch':'2.14.0+cpu','scikit-learn':'1.9.0','threadpoolctl':'3.6.0'}
    if versions!=expected:raise RuntimeError('Use the existing scalability environment; version mismatch: '+str(versions))
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Python 3.12 required')
    return dict(python=sys.version,executable=sys.executable,packages=versions,
                user=os.environ.get('USER'),host=__import__('socket').gethostname(),
                slurm_job=os.environ.get('SLURM_JOB_ID'),allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                source_manifest=sha(ROOT/'source_manifest.json'))


def check():
    verify();env=environment();print(json.dumps(env,indent=2),flush=True)
    import local_test
    local_test.preflight()
    # Two layers with disjoint positive support: the joint Gram may be rank one
    # even when the pooled community support has more than one element.
    import numpy as np
    import scipy.sparse as sp
    import torch
    from shared_template import SharedTemplateHard
    import geometric_hard_model
    a=np.zeros((6,6));a[:3,:3]=1-np.eye(3)
    b=np.zeros((6,6));b[3:,3:]=1-np.eye(3)
    result=SharedTemplateHard(geometric_hard_model.LayerGeometricFixedPartition([sp.csr_matrix(a),sp.csr_matrix(b)],3)).score(np.array([0,0,0,1,1,1]))
    assert abs(result['structure']-1)<1e-12
    write(ROOT/'checks/preflight_pass.json',dict(status='passed',environment=env,
        source_manifest=sha(ROOT/'source_manifest.json'),no_training=True,
        disjoint_layer_support_degeneracy_checked=True,
        objective_checks=read(ROOT/'runtime/checks/shared_template_preflight.json')))
    print('CHECK PASSED. No training started.',flush=True)


def run_candidate(values,task,j,folder):
    old_save=original.merged_save
    def save(path,value):
        old_save(path,value)
        p=Path(path)
        if p.parent.name=='proposal_logs':
            count=int(p.stem)+1
            if count==1 or count%50==0:
                write(folder/'progress.json',dict(task_id=task['id'],start=j,completed_proposals=count,
                    total_proposals=600,after_loss=value['after_loss'],time=time.time()))
                print(f"{task['id']} j={j:02d} {count}/600",flush=True)
    original.merged_save=save
    try:return original.run_candidate(values,task,j,folder)
    finally:original.merged_save=old_save


def trial():
    verify();environment()
    checked=read(ROOT/'checks/preflight_pass.json')
    assert checked['source_manifest']==sha(ROOT/'source_manifest.json')
    if (ROOT/'checks/trial_pass.json').exists():raise RuntimeError('Trial already complete; no duplicate training')
    values=runtime()
    # The first registered shared-template candidate is reused by the array.
    task=read(ROOT/'protocol.json')['tasks'][0]
    folder=ROOT/'results'/task['id']/'j00'
    row=run_candidate(values,task,0,folder)
    replay=readback(values,task,folder)
    verify()
    write(ROOT/'checks/trial_pass.json',dict(status='passed',Full_training_runs=0,
        shared_template_readback=replay,source_manifest=sha(ROOT/'source_manifest.json'),
        formal_start_reused=task['id']+'/j00',truth_read=False))
    print('TRIAL PASSED. The registered array can now start.',flush=True)


def readback(values,task,folder):
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=values
    saved=read(folder/'success.json');row=saved['record']
    for rel,digest in saved['artifacts'].items():assert sha(folder/rel)==digest,folder/rel
    layers,m=original.graph(task['dataset'],np);proxy=original.adapter(values,task)
    net,obj,_=proxy.construct(layers,m['K'],row['seed'],'restored','original-control')
    net.load_state_dict(torch.load(folder/'checkpoint.pt',weights_only=False)['network'])
    T,history,loss,_=proxy.forward(net,obj,100,False)
    assert np.max(np.abs(T.numpy()-np.load(folder/'transport.npy',allow_pickle=False)))<=1e-12
    labels,receipt,raw,hard=wrapper.decode_and_score(T.numpy(),T.argmax(1).numpy(),m['K'],obj.hard_diagnostic,decoder,np)
    assert np.array_equal(labels,np.load(folder/'labels.npy',allow_pickle=False))
    assert abs(hard['objective']-row['hard_objective'])<1e-10
    templates=obj.profiles(T)['template'].detach().numpy()
    assert np.max(np.abs(templates-templates[0]))==0
    assert len(history)==100
    return dict(proposals=row['proposals'],labels_identical=True,soft_templates_identical=True,
                checkpoint_replay=True,truth_read=False)


def pool(index):
    verify();env=environment()
    passed=read(ROOT/'checks/trial_pass.json')
    assert passed['status']=='passed' and passed['source_manifest']==sha(ROOT/'source_manifest.json')
    task=read(ROOT/'protocol.json')['tasks'][index]
    values=runtime();folder=ROOT/'results'/task['id'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'environment.json',env)
    records=[];failures=[]
    for j in range(10):
        try:records.append(run_candidate(values,task,j,folder/f'j{j:02d}'))
        except Exception as exc:
            failures.append(dict(start=j,error=str(exc)));print('FAILED',task['id'],j,str(exc),flush=True)
    winner=min(records,key=lambda x:(x['hard_objective'],x['start'])) if records else None
    write(folder/'selection_before_truth.json',dict(task=task,records=records,failures=failures,
        planned_starts=10,valid_starts=len(records),winner=winner,truth_read=False,
        source_manifest=sha(ROOT/'source_manifest.json')))
    verify()
    if failures or len(records)!=10:raise RuntimeError('Incomplete pool retained; no automatic replacement')
    print('POOL COMPLETE',task['id'],'winner',winner['start'],flush=True)


def pair(index):
    verify();environment()
    if int(os.environ.get('SLURM_CPUS_PER_TASK','2'))<2:raise RuntimeError('Pair requires two allocated CPUs')
    tasks=read(ROOT/'protocol.json')['pairs'][index]
    paths=ROOT/'logs';paths.mkdir(exist_ok=True);children=[]
    for task_index in tasks:
        f=(paths/f'pool_{task_index:02d}.log').open('a',encoding='utf-8')
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'pool','--index',str(task_index)],
            cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children.append((p,f,task_index))
    codes=[]
    for p,f,i in children:codes.append(dict(task_index=i,exit_code=p.wait()));f.close()
    write(ROOT/'checks'/f'pair_{index:02d}.json',dict(exits=codes,source_manifest=sha(ROOT/'source_manifest.json')))
    if any(x['exit_code'] for x in codes):raise RuntimeError('One or more pools failed; logs retained')


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finish():
    verify();env=environment();import numpy as np
    import summarize
    tasks=read(ROOT/'protocol.json')['tasks'];predictions=[];winner_rows=[]
    for task in tasks:
        folder=ROOT/'results'/task['id'];lock=read(folder/'selection_before_truth.json')
        assert lock['task']==task and lock['valid_starts']==10 and not lock['failures']
        assert lock['source_manifest']==sha(ROOT/'source_manifest.json') and lock['truth_read'] is False
        assert lock['winner']==min(lock['records'],key=lambda x:(x['hard_objective'],x['start']))
        assert sorted(r['start'] for r in lock['records'])==list(range(10))
        for row in lock['records']:
            candidate=folder/f"j{row['start']:02d}";success=read(candidate/'success.json')
            identity=success['identity']
            assert identity['seed']==task['seed_base']+100*task['round']+row['start']
            assert identity['task_id']==task['id'] and identity['variant']=='shared_template'
            assert row['proposals']==600 and row['truth_read'] is False
            for rel,digest in success['artifacts'].items():assert sha(candidate/rel)==digest,candidate/rel
            for name in ('proposal_logs.jsonl','proposal_ledgers.jsonl'):
                with (candidate/name).open(encoding='utf-8') as f:
                    ids=[json.loads(line)['proposal'] for line in f]
                assert ids==list(range(600))
            matrices=np.load(candidate/'templates.npz',allow_pickle=False)['B']
            assert np.max(np.abs(matrices-matrices[0]))==0
            hard=read(candidate/'hard_score.json')['profiles']
            assert all(p['template_diagonal']==hard[0]['template_diagonal'] and
                       p['template_off_diagonal']==hard[0]['template_off_diagonal'] for p in hard)
            assert sha(candidate/'labels.npy')==row['labels_sha256']
            predictions.append(dict(task_id=task['id'],start=row['start'],seed=row['seed'],
                                    labels_sha256=row['labels_sha256']))
        winner_rows.append((task,lock['winner']))
    assert len(predictions)==400 and len(winner_rows)==40
    dest=ROOT/'summary';dest.mkdir(exist_ok=True)
    # Commit all 400 predictions and all 40 unsupervised winners before labels.
    write(dest/'all_predictions_lock.json',dict(predictions=predictions,
        winners=[dict(task=t,winner=r) for t,r in winner_rows],truth_read=False,
        source_manifest=sha(ROOT/'source_manifest.json')))
    rows=[]
    for task,row in winner_rows:
        candidate=ROOT/'results'/task['id']/f"j{row['start']:02d}"
        labels=np.load(candidate/'labels.npy',allow_pickle=False)
        rows.append(dict(dataset=task['dataset'],variant='shared_template',round=task['round'],
            start=row['start'],seed=row['seed'],hard_objective=row['hard_objective'],
            source='new_shared_template',**summarize.evaluate(task['dataset'],labels)))
    ref=read(ROOT/'runtime/full_reference/winners.json')
    assert len(ref)==40
    for r in ref:
        rows.append(dict(dataset=r['dataset'],variant='Full',round=r['round'],start=r['start'],seed=r['seed'],
            hard_objective=r['hard_objective'],source='original_frozen_Full',
            ari=r['ari'],nmi=r['nmi'],acc=r['acc']))
    aggregate=[];differences=[]
    for dataset in ('aucs','ckm','dkpol','workplace'):
        for variant in ('Full','shared_template'):
            group=sorted([r for r in rows if r['dataset']==dataset and r['variant']==variant],key=lambda r:r['round'])
            assert [r['round'] for r in group]==list(range(10))
            aggregate.append(dict(dataset=dataset,variant=variant,n=10,
                **{f'{stat}_{metric}':float(getattr(np,stat)([r[metric] for r in group],**({'ddof':1} if stat=='std' else {})))
                   for metric in ('ari','nmi','acc') for stat in ('mean','std')}))
        for round_id in range(10):
            f=next(r for r in rows if r['dataset']==dataset and r['variant']=='Full' and r['round']==round_id)
            s=next(r for r in rows if r['dataset']==dataset and r['variant']=='shared_template' and r['round']==round_id)
            differences.append(dict(dataset=dataset,round=round_id,**{m:f[m]-s[m] for m in ('ari','nmi','acc')}))
    csv_write(dest/'round_winners.csv',rows);csv_write(dest/'metrics.csv',aggregate)
    csv_write(dest/'paired_differences.csv',differences)
    write(dest/'success.json',dict(status='passed_shared_template_R10_J10',new_candidates=400,
        new_proposals=240000,new_winners=40,reused_Full_winners=40,sample_std_ddof=1,
        all_shared_templates_identical=True,environment=env,source_manifest=sha(ROOT/'source_manifest.json')))
    print('COMPLETE: 400 starts, 40 winners; original Full reference retained.',flush=True)


def status():
    tasks=read(ROOT/'protocol.json')['tasks'];done=0;failed=0;pools=0
    for task in tasks:
        folder=ROOT/'results'/task['id']
        done+=len(list(folder.glob('j*/success.json')));failed+=len(list(folder.glob('j*/failure.json')))
        if (folder/'selection_before_truth.json').exists():
            p=read(folder/'selection_before_truth.json');pools+=int(p['valid_starts']==10 and not p['failures'])
    print(f'Successful starts: {done}/400; failed: {failed}; complete pools: {pools}/40')
    print('Full references: 40 original winners, not retrained.')
    if (ROOT/'summary/success.json').exists():print('FORMAL RESULTS COMPLETE: summary/metrics.csv')
    if failed:print('Inspect logs and failure.json. Failed starts are retained; no automatic restart.')


def collect():
    verify()
    if not (ROOT/'summary/success.json').exists():raise RuntimeError('Formal results are not complete. Run 03_status.sh; retain partial records.')
    import tarfile
    folder=ROOT/'returns';folder.mkdir(exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S');path=folder/f'shared_template_results_{stamp}.tar.gz'
    if path.exists():raise RuntimeError('Existing return archive')
    with tarfile.open(path,'w:gz') as archive:
        for item in ROOT.iterdir():
            if item.name not in ('returns','__pycache__'):
                archive.add(item,arcname='shared_template_school/'+item.name,filter=lambda m: None if '__pycache__' in m.name.split('/') else m)
    digest=sha(path);Path(str(path)+'.sha256').write_text(digest+'  '+path.name+'\n',encoding='ascii')
    print('Download:',path);print('SHA256:',digest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','trial','pool','pair','finish','status','collect'])
    p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action in ('pool','pair'):globals()[a.action](a.index)
    else:globals()[a.action]()
