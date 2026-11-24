"""Portable IO and frozen runtime; no truth is loaded by this module."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    # A concurrent Windows progress reader may briefly deny replacement.
    # Retry only the atomic storage commit, never a training proposal.
    import time
    for attempt in range(10):
        try:
            os.replace(tmp,p)
            break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05*(attempt+1))
def verify():
    seal=read(ROOT/'config/source_manifest.json')
    for rel,h in seal.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=h: raise RuntimeError('Source/input hash mismatch: '+rel)
    return sha(ROOT/'config/source_manifest.json')
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
    meta=read(ROOT/'data'/ds/'metadata.json')
    with np.load(ROOT/'data'/ds/'graphs.npz',allow_pickle=False) as z:
        layers=[sp.csr_matrix((z[f'data_{i}'],z[f'indices_{i}'],z[f'indptr_{i}']),shape=(meta['N'],meta['N'])) for i in range(meta['L'])]
    return layers,meta
def adapt(v,measure):
    import types
    from measures import MeasureObjective,MeasureHard
    B=v[0];proxy=types.SimpleNamespace(**B.__dict__)
    def construct(layers,k,seed,mode,variant):
        net,original,base=B.construct(layers,k,seed,mode,variant)
        if measure=='sqrt': return net,original,base
        obj=MeasureObjective(layers,k,measure);obj.hard_diagnostic=MeasureHard(layers,k,measure)
        return net,obj,base
    proxy.construct=construct
    return proxy
def artifacts(folder,exclude=('success.json','failure.json','replay_failure.json')):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in exclude}
def audit_training(folder):
    t=read(folder/'training.json')
    if t['proposals']!=600 or len(t['trace'])!=600 or not t['save_load_verified']:raise RuntimeError('Incomplete 600 training')
    for name in ('proposal_logs.jsonl','proposal_ledgers.jsonl'):
        rows=[json.loads(x) for x in (folder/name).read_text(encoding='utf-8').splitlines()]
        if [x['proposal'] for x in rows]!=list(range(600)): raise RuntimeError('Incomplete '+name)
    best=t['initial_loss'];epoch=0
    for i,r in enumerate(t['trace'],1):
        if r['after_loss']<best:best=r['after_loss'];epoch=i
    if best!=t['selected_loss'] or epoch!=t['selected_epoch']:raise RuntimeError('Wrong own-S24 checkpoint')
    return t
def audit_candidate(folder,task):
    s=read(folder/'success.json');r=read(folder/'replay_receipt.json')
    if s['task']!=task or s['status']!='passed' or r['status']!='passed':raise RuntimeError('Wrong task/replay')
    if s['source_manifest']!=sha(ROOT/'config/source_manifest.json'):raise RuntimeError('Source identity changed')
    for rel,h in s['artifacts'].items():
        p=(folder/rel).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p)!=h:raise RuntimeError('Candidate artifact changed: '+rel)
    if r['checkpoint_sha256']!=sha(folder/'checkpoint.pt') or r['labels_sha256']!=sha(folder/'labels.npy'):raise RuntimeError('Replay binding mismatch')
    audit_training(folder)
    return s
