"""Slurm-only submission:10allocations x2CPUs,2one-thread candidates/allocation."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
from common import *
import argparse,concurrent.futures,subprocess,datetime,time,zipfile,shutil,traceback,importlib.metadata

def environment():
    stamp=verify()
    if sys.version_info<(3,11):raise RuntimeError('Frozen600 engine requires Python3.11+')
    packages={k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','scikit-learn','threadpoolctl','psutil')}
    tasks=read(ROOT/'config/tasks.json')
    if len(tasks)!=800:raise RuntimeError('Wrong fixed task count')
    return dict(status='verified',source_manifest=stamp,python=sys.version,packages=packages,
                candidates=800,proposals=600,cpus_per_allocation=2,max_allocations=10,max_cpus=20,
                training=False,truth_read=False)

def submit():
    print(json.dumps(environment(),indent=2),flush=True)
    if not shutil.which('sbatch'):raise RuntimeError('sbatch not found; run submit.sh in school WebShell')
    concurrency=int(os.environ.get('SLURM_CONCURRENCY','10'))
    if not 1<=concurrency<=10:raise ValueError('SLURM_CONCURRENCY must be1..10; max20CPUs')
    if Path(os.environ.get('PYTHON_BIN',sys.executable)).resolve()!=Path(sys.executable).resolve():raise RuntimeError('PYTHON_BIN differs from submission interpreter')
    for t in read(ROOT/'config/tasks.json'):
        if (ROOT/'results'/t['output']).exists():raise RuntimeError('Existing formal outputs retained; do not submit a new batch into them')
    cfg=dict(account=os.environ['SLURM_ACCOUNT'],partition=os.environ['SLURM_PARTITION'],
             qos=os.environ['SLURM_QOS'])
    (ROOT/'logs').mkdir(exist_ok=True)
    claim=ROOT/'submission.json'
    # Exclusive persisted claim before any sbatch; repeated submissions cannot race.
    record=dict(status='submitting',source_manifest=verify(),configuration=cfg,concurrency=concurrency,
        max_allocations=10,cpus_per_allocation=2,max_cpus=2*concurrency,
        candidates=800,proposals=600,submitted_jobs={},submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with claim.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    def enqueue(stage,cpus,dependency=None,array=None):
        command=['sbatch','--parsable','--no-requeue','--ntasks=1',f'--cpus-per-task={cpus}',
            '--mem=4G','--time=24:00:00',f'--account={cfg["account"]}',f'--partition={cfg["partition"]}',
            f'--qos={cfg["qos"]}',f'--job-name=NM_{stage}',f'--chdir={ROOT}',
            f'--output={ROOT}/logs/{stage}_%A_%a.out',f'--error={ROOT}/logs/{stage}_%A_%a.err']
        if dependency:command.append('--dependency=afterok:'+dependency)
        if array:command.append('--array='+array)
        command.extend([str(ROOT/'job.sh'),stage])
        result=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Unexpected sbatch job ID: '+result.stdout)
        record['submitted_jobs'][stage]=job;write(claim,record)
        return job
    try:
        smoke=enqueue('smoke',2)
        formal=enqueue('pair',2,smoke,f'0-399%{concurrency}')
        evaluation=enqueue('evaluate',1,formal)
        record['status']='submitted';write(claim,record)
    except BaseException as e:
        record['status']='partial_submission_preserved';record['error']=repr(e);write(claim,record)
        print('Submission interrupted; submitted jobs are recorded in submission.json. Do not repeat blindly.',flush=True)
        raise
    print(json.dumps(record,indent=2),flush=True)
    print('Queued full600 smoke ->400pairs(800candidates),max10jobs/20CPUs ->evaluation. No GPU.',flush=True)

def require_allocation():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Training is allowed only in a Slurm compute allocation')
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<2:raise RuntimeError('Two allocated CPUs required')

def run_pair(index,smoke=False):
    require_allocation();stamp=verify()
    (ROOT/'logs').mkdir(exist_ok=True)
    if not 0<=index<400:raise ValueError('Pair index must be0..399')
    indices=[2*index,2*index+1];tasks=read(ROOT/'config/tasks.json')
    kind='smoke' if smoke else 'pair'
    claims=ROOT/'job_claims';claims.mkdir(exist_ok=True)
    claim=claims/(f'smoke_{os.environ["SLURM_JOB_ID"]}.json' if smoke else f'pair_{index:03d}.json')
    identity=dict(stage=kind,index=index,indices=indices,job_id=os.environ['SLURM_JOB_ID'],
                  source_manifest=stamp,truth_read=False,status='running')
    with claim.open('x',encoding='utf-8') as f:json.dump(identity,f,indent=2)
    children={};mutex=__import__('threading').Lock();stop=__import__('threading').Event()
    def one(i):
        t=tasks[i]
        folder=(ROOT/'smoke'/os.environ['SLURM_JOB_ID']/t['measure']) if smoke else ROOT/'results'/t['output']
        if folder.exists():raise RuntimeError('Existing candidate retained, not retrained: '+str(folder))
        log=ROOT/'logs'/f'{kind}_{os.environ["SLURM_JOB_ID"]}_candidate_{i:03d}.log'
        with log.open('x',encoding='utf-8') as out:
            for action in ('train','replay'):
                if stop.is_set():return dict(index=i,status='interrupted')
                cmd=[sys.executable,'-B',str(ROOT/'code/worker.py'),action,'--index',str(i),'--folder',str(folder)]
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                with mutex:children[i]=proc
                code=proc.wait()
                with mutex:children.pop(i,None)
                if code:return dict(index=i,status='failed',action=action,exitcode=code,log=str(log.relative_to(ROOT)))
        s=audit_candidate(folder,t)
        return dict(index=i,status='passed',seed=t['seed'],measure=t['measure'],
            seconds=s['record']['seconds'],receipt_sha256=sha(folder/'replay_receipt.json'))
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures=[pool.submit(one,i) for i in indices];records=[]
    try:
        for f in concurrent.futures.as_completed(futures):
            records.append(f.result());print(json.dumps(records[-1]),flush=True)
    except BaseException as e:
        stop.set()
        with mutex:
            for p in children.values():p.terminate()
        identity.update(status='failed_parts_preserved',error=repr(e));write(claim,identity)
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    identity.update(records=sorted(records,key=lambda r:r['index']),status='passed' if all(r['status']=='passed' for r in records) else 'failed_parts_preserved')
    write(claim,identity);verify()
    if identity['status']!='passed':raise RuntimeError('Pair failed. Preserve all parts; no retry or replacement.')

def smoke():
    require_allocation();verify()
    subprocess.run([sys.executable,'-B',str(ROOT/'code/checks.py')],check=True,cwd=ROOT)
    # Original seeds, two complete600 fixtures outside formal results; never use GT.
    run_pair(0,smoke=True)

def status():
    groups={};active=[]
    for t in read(ROOT/'config/tasks.json'):
        f=ROOT/'results'/t['output'];g=groups.setdefault((t['dataset'],t['measure']),dict(passed=0,trained600=0,failed=0,partial=0,not_started=0))
        if (f/'success.json').exists():g['passed']+=1;g['trained600']+=1
        elif (f/'failure.json').exists() or (f/'replay_failure.json').exists():g['failed']+=1
        elif (f/'progress.json').exists():
            p=read(f/'progress.json');g['partial']+=1
            if p['completed']==600:g['trained600']+=1
            active.append(dict(seed=t['seed'],measure=t['measure'],**p))
        else:g['not_started']+=1
    for (ds,m),g in groups.items():print(ds,m,json.dumps(g),flush=True)
    for a in active:print(json.dumps(a),flush=True)
    print('Independent replay passed:',sum(g['passed'] for g in groups.values()),'/800',flush=True)
    if (ROOT/'submission.json').exists():
        record=read(ROOT/'submission.json');ids=','.join(record['submitted_jobs'].values());print('Jobs:',ids)
        if ids and shutil.which('squeue'):
            subprocess.run(['squeue','-j',ids,'-o','%.24i %.10T %.6C %.40R'],check=False)
    if (ROOT/'summary/metrics.json').exists():print('Complete evaluated results: summary/summary.csv')

def collect():
    verify();stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/f'NodeMeasure_server_small_{stamp}.zip';files=[]
    for name in ('config','reference','checks','summary','job_claims'):
        if (ROOT/name).exists():files.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    for name in ('submission.json','VALIDATION.json','README.md'):
        if (ROOT/name).exists():files.append(ROOT/name)
    allowed=('task.json','progress.json','success.json','failure.json','replay_failure.json','replay_receipt.json',
             'training_receipt.json','labels.npy','hard_score.json','initialization.json','decoder.json')
    for name in ('results','smoke'):
        if (ROOT/name).exists():files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.name in allowed)
    if (ROOT/'logs').exists():files.extend(p for p in (ROOT/'logs').iterdir() if p.is_file())
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
    print(str(out));print(f'{out.stat().st_size/1024**2:.2f} MB. Download this ZIP; full weights and ledgers remain on server.')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','submit','smoke','pair','status','collect'])
    p.add_argument('--index',type=int,default=-1);a=p.parse_args()
    if a.action=='verify':print(json.dumps(environment(),indent=2))
    elif a.action=='submit':submit()
    elif a.action=='smoke':smoke()
    elif a.action=='pair':run_pair(a.index)
    elif a.action=='status':status()
    elif a.action=='collect':collect()
if __name__=='__main__':main()
