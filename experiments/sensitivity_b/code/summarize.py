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
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/B');a=p.parse_args()
    verify_package();tasks=read(ROOT/'config/tasks_B.json');out=ROOT/a.output
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
        rows.append(dict(dataset=task['dataset'],condition=task['condition'],graph_index=task['graph_index'],instance=task['instance'],
                    variant=task['variant'],lambda_rho=task['lambda_rho'],round=task['round'],kind=task['kind'],
                    start=winner['start'],seed=winner['seed'],hard_objective=winner['hard_objective'],
                    valid_starts=lock['valid_starts'],source='new_B',**evaluate(task['dataset'],labels)))
    for r in read(ROOT/'full_reference/winners.json'):
        rows.append(dict(dataset=r['dataset'],condition=r['condition'],graph_index=r['graph_index'],instance=r['instance'],
                    variant='Full',lambda_rho=.02,round=r['round'],kind=r['kind'],start=r['start'],seed=r['seed'],
                    hard_objective=r['hard_objective'],valid_starts=10,source='frozen_reference',
                    ari=r['ari'],nmi=r['nmi'],acc=r['acc']))
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['variant'],r['lambda_rho'])].append(r)
    graph_rows=[]
    for (ds,v,lam),rr in sorted(groups.items()):
        sim=ds.startswith('sim/')
        graph_rows.append(dict(dataset=ds,condition=rr[0]['condition'],graph_index=rr[0]['graph_index'],instance=rr[0]['instance'],
                    variant=v,lambda_rho=lam,valid_rounds=len(rr),planned_rounds=5 if sim else 10,
                    mean_ari=float(np.mean([r['ari'] for r in rr])),mean_nmi=float(np.mean([r['nmi'] for r in rr])),
                    mean_acc=float(np.mean([r['acc'] for r in rr])),valid_starts=sum(r['valid_starts'] for r in rr)))
    sim_groups=defaultdict(list)
    for r in graph_rows:
        if r['condition']:sim_groups[(r['condition'],r['variant'])].append(r)
    conditions=[]
    for (c,v),rr in sorted(sim_groups.items()):
        conditions.append(dict(condition=c,variant=v,n_graphs=len(rr),planned_graphs=10,
                    mean_ari=float(np.mean([r['mean_ari'] for r in rr])),
                    mean_nmi=float(np.mean([r['mean_nmi'] for r in rr])),
                    mean_acc=float(np.mean([r['mean_acc'] for r in rr])),
                    complete_graphs=sum(r['valid_rounds']==5 and r['valid_starts']==50 for r in rr)))
    paired=[]
    lookup={(r['dataset'],r['variant']):r for r in graph_rows if r['condition']}
    for r in graph_rows:
        if not r['condition'] or r['variant']=='Full':continue
        full=lookup[(r['dataset'],'Full')]
        paired.append(dict(dataset=r['dataset'],condition=r['condition'],graph_index=r['graph_index'],variant=r['variant'],
                    full_ari=full['mean_ari'],variant_ari=r['mean_ari'],delta_ari=r['mean_ari']-full['mean_ari']))
    dest=ROOT/'summary/B';dest.mkdir(parents=True,exist_ok=True)
    csv_write(dest/'round_winners.csv',rows)
    csv_write(dest/'graph_level.csv',graph_rows)
    csv_write(dest/'real_lambda.csv',[r for r in graph_rows if not r['condition']])
    if conditions:csv_write(dest/'simulation_conditions.csv',conditions)
    if paired:csv_write(dest/'paired_graph_differences.csv',paired)
    write(dest/'status.json',dict(planned_pools=200,missing_pools=missing,partial_pools=partial,
          new_winners=sum(r['source']=='new_B' for r in rows),full_reference_winners=120,
          statistical_unit='real: outer repeat; simulation: independent graph after five-repeat averaging',
          complete=not missing and not partial))
    print(f"New winners={sum(r['source']=='new_B' for r in rows)}/200; missing={len(missing)}; partial={len(partial)}")


if __name__=='__main__':main()
