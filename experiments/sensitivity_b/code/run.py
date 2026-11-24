"""B group: candidate-isolated execution; truth is only read by summarization."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, p)


def verify_package():
    for rel, digest in read(ROOT/'config/source_manifest.json').items():
        if sha(ROOT/rel) != digest:
            raise RuntimeError('Package input/source changed: '+rel)


def runtime():
    sys.path.insert(0, str(ROOT/'frozen'))
    from bootstrap import activate
    values = activate()
    torch = values[6]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    return values


def graph(name, np):
    import scipy.sparse as sp
    folder = ROOT/'data'/name
    meta = read(folder/'metadata.json')
    with np.load(folder/'graphs.npz', allow_pickle=False) as z:
        layers = [sp.csr_matrix((z[f'data_{i}'],z[f'indices_{i}'],z[f'indptr_{i}']),
                   shape=(meta['N'],meta['N'])) for i in range(meta['L'])]
    for a in layers:
        a.check_format(full_check=True)
        if (a-a.T).nnz or np.any(a.diagonal()) or not np.isfinite(a.data).all() or (a.data<0).any():
            raise ValueError('Invalid original relation')
    return layers, meta


def adapter(values, task):
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support = values
    from objectives import BObjective, BHard
    proxy = types.SimpleNamespace(**B.__dict__)
    def construct(layers,k,seed,mode,variant):
        net,original,base = B.construct(layers,k,seed,mode,variant)
        if task['variant']=='Full' and task['lambda_rho']==.02:
            obj=original
        else:
            obj=BObjective(layers,k,task['variant'],task['lambda_rho'])
            obj.hard_diagnostic=BHard(original.hard_diagnostic,task['variant'],task['lambda_rho'])
        return net,obj,base
    def forward(net,obj,steps,gradient):
        horizon = task['S_train'] if steps==24 else steps
        if horizon==24:
            return B.forward(net,obj,24,gradient)
        eta = (net.log_eta[:horizon] if horizon<=12 else
               torch.cat([net.log_eta,net.log_eta[-1].repeat(horizon-12)]))
        T,history=B.E.infer(obj,net() if gradient else net().detach(),
                          eta if gradient else eta.detach(),create_graph=gradient)
        checks=B.coupling_checks(obj,T)
        loss=obj.loss(T)
        if len(history)!=horizon or not bool(torch.isfinite(loss)):
            raise RuntimeError('Incomplete/nonfinite forward')
        return T,history,loss,checks
    proxy.construct,proxy.forward=construct,forward
    return proxy


def merged_save(path,value):
    """Only storage layout changes: retain frozen training logs in two JSONL files."""
    p=Path(path)
    if p.parent.name in ('proposal_logs','proposal_ledgers'):
        target=p.parent.parent/(p.parent.name+'.jsonl')
        with target.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'proposal':int(p.stem),'record':value},allow_nan=False)+'\n')
    else:
        write(p,value)


def run_candidate(values,task,j,folder):
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=values
    seed=task['seed_base']+100*task['round']+j
    identity=dict(task_id=task['id'],dataset=task['dataset'],variant=task['variant'],
                  S_train=task['S_train'],lambda_rho=task['lambda_rho'],round=task['round'],start=j,seed=seed,
                  source_manifest=sha(ROOT/'config/source_manifest.json'))
    if folder.exists():
        if (folder/'success.json').exists():
            old=read(folder/'success.json')
            if old['identity']!=identity:
                raise RuntimeError('Completed output identity mismatch')
            for rel,h in old['artifacts'].items():
                if sha(folder/rel)!=h:raise RuntimeError('Completed artifact changed')
            return old['record']
        raise RuntimeError('Interrupted/failed candidate retained; use an explicit new attempt: '+str(folder))
    folder.mkdir(parents=True)
    started=time.perf_counter();cpu=time.process_time();context={}
    write(folder/'identity.json',identity)
    try:
        layers,m=graph(task['dataset'],np)
        proxy=adapter(values,task)
        H.save=merged_save
        slot=dict(candidate_id=f"{task['id']}/j{j:02d}",seed=seed,mode='restored',
                  variant='original-control',proposals=600)
        def protected(*args,**kwargs):
            opt,log,raw_gradient=C.protected_proposal(*args,**kwargs)
            log['inner_unroll_steps_per_S24']=task['S_train']
            log['callback_slot_name']='S24 is the frozen callback name; actual steps recorded separately'
            log['actual_training_unroll_steps']=task['S_train']
            return opt,log,raw_gradient
        guard=types.SimpleNamespace(protected_proposal=protected)
        net,obj,training=H.train_candidate(slot,layers,m,proxy,torch,Schedule,guard,folder,cfg,context)
        training['actual_training_unroll_steps']=task['S_train']
        training['learned_eta_count']=net.log_eta.numel()
        write(folder/'training.json',training)
        T,history,soft,checks=proxy.forward(net,obj,100,False)
        raw=T.argmax(1).numpy()
        labels,receipt,rawscore,hard=wrapper.decode_and_score(T.numpy(),raw,m['K'],obj.hard_diagnostic,decoder,np)
        np.save(folder/'transport.npy',T.numpy());np.save(folder/'labels.npy',labels)
        profile=obj.profiles(T)
        np.savez_compressed(folder/'templates.npz',B=profile['template'].detach().numpy(),
                           q=profile['q'].detach().numpy(),b=profile['b'].detach().numpy(),
                           alpha=profile['alpha'].detach().numpy(),source_mu=obj.source_measures.numpy())
        write(folder/'hard_score.json',hard);write(folder/'decoder.json',receipt)
        record=dict(**identity,soft_objective=float(soft.detach()),hard_objective=hard['objective'],
                    labels_sha256=sha(folder/'labels.npy'),selected_epoch=training['selected_epoch'],
                    selected_loss=training['selected_loss'],proposals=training['proposals'],
                    train_steps=task['S_train'],inference_steps=100,learned_eta_count=net.log_eta.numel(),
                    seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                    output_classes=int(len(np.unique(labels))),truth_read=False)
        write(folder/'endpoint.json',dict(record=record,history=history,checks=checks))
        write(folder/'success.json',dict(identity=identity,record=record,artifacts={
            p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
        return record
    except BaseException as exc:
        write(folder/'failure.json',dict(identity=identity,error=str(exc),traceback=traceback.format_exc(),
              seconds=time.perf_counter()-started,context=context,truth_read=False))
        raise


def run_pool(values,task,out,starts=10):
    folder=out/task['id'];folder.mkdir(parents=True,exist_ok=True)
    records=[];failures=[]
    for j in range(starts):
        try:
            row=run_candidate(values,task,j,folder/f'j{j:02d}');records.append(row)
            print(f"{task['id']} j={j} hard={row['hard_objective']:.12g}",flush=True)
        except Exception as exc:
            failures.append(dict(start=j,error=str(exc)))
            print(f"FAILED {task['id']} j={j}: {exc}",flush=True)
    winner=min(records,key=lambda r:(r['hard_objective'],r['start'])) if records else None
    lock=dict(task=task,records=records,failures=failures,winner=winner,
              planned_starts=starts,valid_starts=len(records),truth_read=False)
    write(folder/'selection_before_truth.json',lock)
    if failures:raise RuntimeError(f'{len(failures)} failed candidates; retained without replacement')
    return lock


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['task','pilot'])
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',default='results/B')
    args=p.parse_args();verify_package();source_stamp=sha(ROOT/'config/source_manifest.json');values=runtime()
    if args.action=='task':
        tasks=read(ROOT/'config/tasks_B.json')
        if not 0<=args.index<len(tasks):raise ValueError('B task index must be 0..199')
        task=tasks[args.index]
        run_pool(values,task,ROOT/args.output)
    else:
        tasks=read(ROOT/'config/pilot_B.json')
        for t in tasks:
            lock=run_pool(values,t,ROOT/args.output,starts=1)
            if t['variant']=='Full' and t['lambda_rho']==.02:
                expected=read(ROOT/'full_reference'/('simulation_probe.json' if t['dataset'].startswith('sim/') else 'aucs_probe.json'));actual=lock['winner']
                if (actual['labels_sha256']!=expected['labels_sha256'] or
                        abs(actual['hard_objective']-expected['hard_objective'])>1e-10):
                    raise RuntimeError('Full differs from frozen AUCS probe. Stop and align environment; do not retune with truth.')
                write(ROOT/('checks/full_parity_'+('sim' if t['dataset'].startswith('sim/') else 'real')+'.json'),dict(status='passed',actual=actual,expected=expected))
        from validate_trial import validate
        validate(values,tasks,ROOT/args.output)
        verify_package()
        if source_stamp!=sha(ROOT/'config/source_manifest.json'):raise RuntimeError('Source seal changed during run')
        write(ROOT/'checks/pilot_pass.json',dict(status='passed',source_manifest=sha(ROOT/'config/source_manifest.json'),
              cases=[t['id'] for t in tasks],formal_results=False))
    verify_package()
    if source_stamp!=sha(ROOT/'config/source_manifest.json'):raise RuntimeError('Source seal changed during run')


if __name__=='__main__':
    main()
