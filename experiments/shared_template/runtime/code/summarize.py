"""Summarize locked winners; only this stage loads ground-truth labels."""
import argparse
import csv
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from run import ROOT,read,write,sha,verify_package


def evaluate(dataset,labels):
    m=read(ROOT/'data'/dataset/'metadata.json')
    with np.load(ROOT/'data'/dataset/'evaluation.npz',allow_pickle=False) as z:
        valid=z['mask'].astype(bool)&(z['truth']>=0)
        t=np.tile(z['truth'][valid],m['L']);y=np.tile(labels[valid],m['L'])
    _,ti=np.unique(t,return_inverse=True);_,yi=np.unique(y,return_inverse=True)
    table=np.zeros((ti.max()+1,yi.max()+1),dtype=np.int64);np.add.at(table,(ti,yi),1)
    i,j=linear_sum_assignment(-table)
    return dict(ari=float(adjusted_rand_score(t,y)),
                nmi=float(normalized_mutual_info_score(t,y,average_method='arithmetic')),
                acc=float(table[i,j].sum()/len(t)))


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/A');a=p.parse_args()
    verify_package();tasks=read(ROOT/'config/tasks_A.json');out=ROOT/a.output
    rows=[];missing=[];partial=[]
    for task in tasks:
        folder=out/task['id'];path=folder/'selection_before_truth.json'
        if not path.exists():missing.append(task['id']);continue
        lock=read(path)
        if lock['task']!=task or lock['planned_starts']!=10:raise RuntimeError('Wrong pool')
        if lock['valid_starts']!=10:partial.append(dict(task_id=task['id'],valid_starts=lock['valid_starts']))
        winner=lock['winner']
        if winner is None:continue
        assert winner==min(lock['records'],key=lambda r:(r['hard_objective'],r['start']))
        candidate=folder/f"j{winner['start']:02d}"
        if sha(candidate/'labels.npy')!=winner['labels_sha256']:raise RuntimeError('Locked labels changed')
        labels=np.load(candidate/'labels.npy',allow_pickle=False)
        rows.append(dict(dataset=task['dataset'],variant=task['variant'],S_train=task['S_train'],
                    round=task['round'],kind=task['kind'],start=winner['start'],seed=winner['seed'],
                    hard_objective=winner['hard_objective'],valid_starts=lock['valid_starts'],
                    seconds_all_starts=sum(r['seconds'] for r in lock['records']),source='new_A',
                    **evaluate(task['dataset'],labels)))
    for r in read(ROOT/'full_reference/winners.json'):
        rows.append(dict(dataset=r['dataset'],variant='Full',S_train=24,round=r['round'],kind='reference',
                    start=r['start'],seed=r['seed'],hard_objective=r['hard_objective'],valid_starts=10,
                    seconds_all_starts='',source='frozen_reference',ari=r['ari'],nmi=r['nmi'],acc=r['acc']))
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['variant'],r['S_train'],r['kind'])].append(r)
    summary=[]
    for (ds,v,s,kind),rr in sorted(groups.items()):
        summary.append(dict(dataset=ds,variant=v,S_train=s,kind=kind,valid_rounds=len(rr),planned_rounds=10,
                    mean_ari=float(np.mean([r['ari'] for r in rr])),
                    mean_nmi=float(np.mean([r['nmi'] for r in rr])),
                    mean_acc=float(np.mean([r['acc'] for r in rr])),
                    valid_starts=sum(r['valid_starts'] for r in rr)))
    dest=ROOT/'summary/A';dest.mkdir(parents=True,exist_ok=True)
    csv_write(dest/'round_winners.csv',rows);csv_write(dest/'summary.csv',summary)
    write(dest/'status.json',dict(planned_pools=200,missing_pools=missing,partial_pools=partial,
          new_winners=sum(r['source']=='new_A' for r in rows),full_reference_winners=40,
          complete=not missing and not partial))
    print(f"New winners={sum(r['source']=='new_A' for r in rows)}/200; missing={len(missing)}; partial={len(partial)}")


if __name__=='__main__':main()
