"""One cold full600 candidate, or its fresh-process independent replay. No GT."""
from common import *
import argparse,time,traceback,types

def decode(v,obj,T):
    labels,receipt,raw,hard=v[4].decode_and_score(T.numpy(),T.argmax(1).numpy(),obj.K,obj.hard_diagnostic,v[5],v[7])
    return labels,receipt,hard

def train(task,folder):
    stamp=verify();v=runtime();B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=v
    layers,m=graph(task['dataset'],np);proxy=adapt(v,task['measure'])
    folder.mkdir(parents=True,exist_ok=False);write(folder/'task.json',task)
    started=time.perf_counter();write(folder/'progress.json',dict(completed=0,total=600,stage='initializing'))
    net,obj,_=proxy.construct(layers,m['K'],task['seed'],'restored','original-control')
    initial={}
    for steps in (24,100):
        t,h,l,c=proxy.forward(net,obj,steps,False);initial[f'S{steps}']=t.numpy()
    np.savez_compressed(folder/'initial_endpoints.npz',**initial)
    del net,obj
    def save(p,value):
        p=Path(p)
        if p.parent.name in ('proposal_logs','proposal_ledgers'):
            target=folder/(p.parent.name+'.jsonl')
            with target.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(proposal=int(p.stem),record=value),allow_nan=False)+'\n')
            if p.parent.name=='proposal_logs':
                write(folder/'progress.json',dict(completed=int(p.stem)+1,total=600,
                    after_loss=value['after_loss'],seconds=time.perf_counter()-started,stage='training'))
        else:write(p,value)
    H.save=save;context={}
    slot=dict(candidate_id=task['output'],seed=task['seed'],mode='restored',variant='original-control',proposals=600)
    net,obj,t=H.train_candidate(slot,layers,m,proxy,torch,Schedule,C,folder,cfg,context)
    T,h,l,checks=proxy.forward(net,obj,100,False)
    labels,receipt,hard=decode(v,obj,T)
    np.save(folder/'transport.npy',T.numpy());np.save(folder/'labels.npy',labels)
    write(folder/'hard_score.json',hard);write(folder/'decoder.json',receipt)
    pf=obj.profiles(T)
    np.savez_compressed(folder/'layer_statistics.npz',mu=obj.source_measures.numpy(),nu=obj.nu.numpy(),
        null=obj.null.numpy(),q=pf['q'].detach().numpy(),b=pf['b'].detach().numpy(),alpha=pf['alpha'].detach().numpy())
    rec=dict(task=task,source_manifest=stamp,proposals=600,selected_epoch=t['selected_epoch'],
        initial_loss=t['initial_loss'],selected_loss=t['selected_loss'],terminal_loss=t['terminal_loss'],
        accepted=t['accepted'],fallback_proposals=t['fallback_proposals'],hard_objective=hard['objective'],
        soft_S100=float(l),seconds=time.perf_counter()-started,truth_read=False,
        active_layers=obj.active_layers,zero_volume_layers=obj.zero_volume_layers,
        inactive_variance_layers=obj.inactive_variance_layers,threads=1)
    write(folder/'training_receipt.json',rec);audit_training(folder);verify()
    write(folder/'progress.json',dict(completed=600,total=600,stage='training_complete_replay_pending',seconds=rec['seconds']))

def replay(task,folder):
    stamp=verify();v=runtime();B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=v
    if read(folder/'task.json')!=task:raise RuntimeError('Task mismatch')
    t=audit_training(folder);r=read(folder/'training_receipt.json')
    layers,m=graph(task['dataset'],np);proxy=adapt(v,task['measure'])
    net,obj,_=proxy.construct(layers,m['K'],task['seed'],'restored','original-control')
    if B.state_digest(B.state(net))!=read(folder/'initialization.json')['base_digest']:raise RuntimeError('Cold initialization changed')
    errors={}
    initial=torch.load(folder/'initial.pt',weights_only=False);net.load_state_dict(initial)
    with np.load(folder/'initial_endpoints.npz',allow_pickle=False) as z:
        for steps in (24,100):
            a,h,l,c=proxy.forward(net,obj,steps,False);expected=z[f'S{steps}']
            errors[f'initial_S{steps}']=float(np.max(np.abs(a.numpy()-expected)))
            np.testing.assert_allclose(a.numpy(),expected,rtol=1e-9,atol=1e-12)
            if steps==24 and abs(float(l)-t['initial_loss'])>1e-10:raise RuntimeError('Initial loss mismatch')
    ck=torch.load(folder/'checkpoint.pt',weights_only=False);net.load_state_dict(ck['network'])
    terminal=torch.load(folder/'terminal.pt',weights_only=False)
    if terminal['schedule']['completed']!=600:raise RuntimeError('Terminal budget mismatch')
    for steps in (24,100):
        a,h,l,c=proxy.forward(net,obj,steps,False)
        if steps==24:
            with np.load(folder/'training_transport.npz',allow_pickle=False) as z:expected=z['selected'].copy()
            if abs(float(l)-t['selected_loss'])>1e-10:raise RuntimeError('Selected S24 mismatch')
        else:
            expected=np.load(folder/'transport.npy',allow_pickle=False)
            labels,receipt,hard=decode(v,obj,a)
            if not np.array_equal(labels,np.load(folder/'labels.npy',allow_pickle=False)):raise RuntimeError('MAP changed on replay')
            if abs(hard['objective']-r['hard_objective'])>1e-10:raise RuntimeError('Hard objective mismatch')
        errors[f'selected_S{steps}']=float(np.max(np.abs(a.numpy()-expected)))
        np.testing.assert_allclose(a.numpy(),expected,rtol=1e-9,atol=1e-12)
    verify()
    write(folder/'replay_receipt.json',dict(status='passed',task=task,source_manifest=stamp,
        process_id=os.getpid(),fresh_process=True,threads=1,memo=False,proposals=600,errors=errors,
        checkpoint_sha256=sha(folder/'checkpoint.pt'),labels_sha256=sha(folder/'labels.npy'),truth_read=False))
    write(folder/'progress.json',dict(completed=600,total=600,stage='replay_passed',seconds=r['seconds']))
    write(folder/'success.json',dict(status='passed',task=task,record=r,source_manifest=stamp,artifacts=artifacts(folder)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','replay'])
    p.add_argument('--index',type=int,required=True);p.add_argument('--folder',required=True)
    a=p.parse_args();tasks=read(ROOT/'config/tasks.json')
    if not 0<=a.index<len(tasks):raise ValueError('Candidate index outside fixed pool')
    task=tasks[a.index];folder=Path(a.folder).resolve()
    if a.action=='train' and folder.exists():raise RuntimeError('Existing candidate preserved; not retrained')
    if a.action=='replay' and (folder/'success.json').exists():raise RuntimeError('Passed replay preserved; not overwritten')
    try:(train if a.action=='train' else replay)(task,folder)
    except BaseException as e:
        # Never overwrite another active candidate's folder on an accidental duplicate.
        if folder.exists() and read(folder/'task.json')==task:
            write(folder/('failure.json' if a.action=='train' else 'replay_failure.json'),dict(action=a.action,
                task=task,error=repr(e),traceback=traceback.format_exc(),truth_read=False))
        raise
if __name__=='__main__':main()
