"""Seal all complete800 predictions and80 winners before GT loading. No training."""
from common import *
import numpy as np
import csv,io,zipfile

def scores(ds,pred):
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment
    m=read(ROOT/'data'/ds/'metadata.json')
    with np.load(ROOT/'data'/ds/'evaluation.npz',allow_pickle=False) as z:
        mask=z['mask'].astype(bool)&(z['truth']>=0)
        t=np.tile(z['truth'][mask],m['L']);y=np.tile(pred[mask],m['L'])
    _,a=np.unique(t,return_inverse=True);_,b=np.unique(y,return_inverse=True)
    tab=np.zeros((a.max()+1,b.max()+1),dtype=np.int64);np.add.at(tab,(a,b),1)
    i,j=linear_sum_assignment(-tab)
    return dict(ari=float(adjusted_rand_score(t,y)),nmi=float(normalized_mutual_info_score(t,y,average_method='arithmetic')),
                acc=float(tab[i,j].sum()/len(t)))
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    stamp=verify();tasks=read(ROOT/'config/tasks.json');candidate=[];payload={};groups={}
    for t in tasks:
        f=ROOT/'results'/t['output'];s=audit_candidate(f,t);rec=s['record'];data=(f/'labels.npy').read_bytes()
        if hashlib.sha256(data).hexdigest()!=s['artifacts']['labels.npy']:raise RuntimeError('Prediction bytes changed')
        payload[t['index']]=data
        row=dict(**t,hard_objective=rec['hard_objective'],selected_epoch=rec['selected_epoch'],
            seconds=rec['seconds'],labels_sha256=sha(f/'labels.npy'),receipt_sha256=sha(f/'replay_receipt.json'))
        candidate.append(row);groups.setdefault((t['dataset'],t['measure'],t['round']),[]).append(row)
    selected=[]
    for key,rr in sorted(groups.items()):
        if sorted(r['start'] for r in rr)!=list(range(10)):raise RuntimeError('Incomplete pool')
        selected.append(min(rr,key=lambda x:(x['hard_objective'],x['start'])))
    if len(selected)!=80 or len(candidate)!=800:raise RuntimeError('Incomplete formal suite')
    dest=ROOT/'summary';dest.mkdir(exist_ok=True)
    seal=dict(source_manifest=stamp,plan_sha256=sha(ROOT/'config/plan.json'),
              candidates=candidate,winners=selected,truth_read=False,selection='ownhard_first_argmin',
              reference_sha256=sha(ROOT/'reference/winners.json'))
    p=dest/'selection_before_truth.json'
    if p.exists():
        if read(p)!=seal:raise RuntimeError('Existing selection seal differs; no overwrite')
    else:write(p,seal)
    sealsha=sha(p)
    if (dest/'metrics.json').exists():
        old=read(dest/'metrics.json')
        if old['selection_sha256']!=sealsha:raise RuntimeError('Metrics binding changed')
        for rel,h in old['files'].items():
            if sha(dest/rel)!=h:raise RuntimeError('Existing metric output changed')
        print('Existing complete metrics verified; not re-evaluated.');return
    # All selected and candidate prediction bytes are already cached before this point.
    rows=[];allrows=[]
    for r in candidate:
        a=np.load(io.BytesIO(payload[r['index']]),allow_pickle=False)
        allrows.append(dict(dataset=r['dataset'],measure=r['measure'],round=r['round'],start=r['start'],seed=r['seed'],
            selected_epoch=r['selected_epoch'],hard_objective=r['hard_objective'],**scores(r['dataset'],a)))
    byindex={r['index']:x for r,x in zip(candidate,allrows)}
    for r in selected:rows.append(dict(**byindex[r['index']],source='new_complete600_R10J10'))
    for r in read(ROOT/'reference/winners.json'):
        if sha(ROOT/r['labels'])!=r['labels_sha256']:raise RuntimeError('Reference predictions changed')
        sc=scores(r['dataset'],np.load(ROOT/r['labels'],allow_pickle=False))
        if max(abs(sc[k]-r[k]) for k in sc)>1e-12:raise RuntimeError('Original Full score mismatch')
        rows.append(dict(dataset=r['dataset'],measure='sqrt',round=r['round'],start=r['start'],seed=r['seed'],
            selected_epoch='',hard_objective=r['hard_objective'],**sc,source='frozen_Full_reference'))
    summary=[];paired=[]
    for ds in read(ROOT/'config/plan.json')['datasets']:
        ref={r['round']:r for r in rows if r['dataset']==ds and r['measure']=='sqrt'}
        for mode in ('sqrt','uniform','degree'):
            rr=sorted([r for r in rows if r['dataset']==ds and r['measure']==mode],key=lambda x:x['round'])
            if len(rr)!=10:raise RuntimeError('Not ten rounds')
            out=dict(dataset=ds,measure=mode,rounds=10)
            for k in ('ari','nmi','acc'):
                v=[r[k] for r in rr];out[k+'_mean']=float(np.mean(v));out[k+'_sd']=float(np.std(v,ddof=1))
                out[k+'_delta_vs_sqrt']=float(np.mean([r[k]-ref[r['round']][k] for r in rr]))
            summary.append(out)
            if mode!='sqrt':
                for r in rr:paired.append(dict(dataset=ds,measure=mode,round=r['round'],
                    **{k+'_delta_vs_sqrt':r[k]-ref[r['round']][k] for k in ('ari','nmi','acc')}))
    csvout(dest/'all800_candidates.csv',allrows);csvout(dest/'round_winners.csv',rows)
    csvout(dest/'summary.csv',summary);csvout(dest/'paired_differences.csv',paired)
    lines=['# 节点测度消融结果','',
        '四张原图；每种测度10轮，每轮10起点，每起点完整600次训练提案。平方根方案复用原Full。',
        '选解仅使用各自内部目标，ARI/NMI/ACC在全部选解封存后统一计算。标准差为10轮样本标准差。','',
        '|图|测度|ARI|NMI|ACC|','|---|---|---:|---:|---:|']
    names={'sqrt':'平方根度（原Full）','uniform':'全节点均匀','degree':'度比例'}
    for r in summary:lines.append('|'+r['dataset']+'|'+names[r['measure']]+'|'+'|'.join(f"{r[k+'_mean']:.4f} ± {r[k+'_sd']:.4f}" for k in ('ari','nmi','acc'))+'|')
    lines+=['','各方案的原始loss对应不同测度，不能跨方案按loss大小判断优劣。',
        '均匀方案给孤立节点正质量；本三方案实验同时反映参与支持和度加权差异，未单独剥离这两个因素。',
        '同一真实图的10轮用于评估初始化与选解波动，并非10个独立网络。']
    (dest/'结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Small loss exports for all80 selected candidates only; checkpoints/ledgers remain local.
    for r in selected:
        t=read(ROOT/'results'/r['output']/'training.json')
        trace=[dict(proposal=i,loss=x['after_loss'],accepted_scale=x['accepted_scale']) for i,x in enumerate(t['trace'],1)]
        f=dest/'loss'/f"{r['dataset']}_{r['measure']}_r{r['round']:02d}.csv";f.parent.mkdir(exist_ok=True)
        csvout(f,trace)
    verify()
    files={x.relative_to(dest).as_posix():sha(x) for x in dest.rglob('*') if x.is_file() and x.name not in ('selection_before_truth.json','metrics.json')}
    write(dest/'metrics.json',dict(status='complete',selection_sha256=sealsha,source_manifest=stamp,new_candidates=800,
        new_winners=80,reused_Full_winners=40,files=files))
    print('\n'.join(lines))
if __name__=='__main__':main()
