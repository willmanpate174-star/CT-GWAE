"""Reuse audited v6 infrastructure; this import does not load torch."""
from pathlib import Path
import importlib.util,sys
BASE=Path('C:/Users/Dingjunhao/Desktop/李慧嘉课题组/V8神经网络/实验/准备_20261006_下降恢复J10完整开发')
ROOT=BASE.parent
V6=ROOT/'准备_20261006_下降恢复分支实图预检'
spec=importlib.util.spec_from_file_location('audited_recovery_v6',V6/'preflight_entry_v6.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
sha,read,save,state_hash=E.sha,E.read,E.save,E.state_hash
REG=BASE/'registration.json'
OUTPUT=BASE/'output'

def verify_manifest(manifest):
    for p,h in manifest.items():assert sha(p)==h,p

def verify(reg,references=True):
    verify_manifest(reg['source_sha256']);verify_manifest(reg['input_sha256'])
    verify_manifest(reg['evidence_sha256'])
    if references:verify_manifest(reg['reference_sha256'])
    assert sha(reg['runtime']['executable'])==reg['runtime']['sha256']

def artifacts(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'success.json'}

def verify_candidate(folder,slot,reg):
    folder=Path(folder);r=read(folder/'success.json')
    assert r['status']=='passed' and r['identity']==slot and r['registration_sha256']==sha(REG)
    assert r['training']['proposals']==70 and r['new_cold_training'] and not r['preflight_weights_loaded']
    for name,h in r['artifact_sha256'].items():
        p=(folder/name).resolve();assert p.is_relative_to(folder.resolve()) and sha(p)==h
    assert read(folder/'cost.json')['status']=='passed'
    return r

def train_candidate(slot,layers,m,B,torch,Schedule,C,folder,cfg,context):
    """v6 proposal/state/checkpoint path, generalized identities; no old-trace gate."""
    net,obj,base=B.construct(layers,m['K'],slot['seed'],slot['mode'],slot['variant'])
    initial=dict(seed=slot['seed'],base_digest=B.state_digest(base),full_digest=B.state_digest(B.state(net)),
                 feature_sha256=__import__('hashlib').sha256(net.X.numpy().tobytes()).hexdigest(),
                 mode=slot['mode'],variant=slot['variant'],historical_state_loading=False,preflight_state_loading=False)
    save(folder/'initialization.json',initial);torch.save(B.state(net),folder/'initial.pt')
    opt=torch.optim.Adam(net.parameters(),lr=.012);schedule=Schedule();trace=[]
    _,_,initial_loss,_=B.forward(net,obj,24,False)
    best=float(initial_loss);chosen=B.state(net);chosen_opt=__import__('copy').deepcopy(opt.state_dict())
    chosen_schedule=dict(vars(schedule));epoch=0
    budget=cfg['proposals_per_candidate']
    assert budget==600 and slot['proposals']==600
    for i in range(budget):
        before_parameters=B.state_digest(B.state(net));before_optimizer=state_hash(opt.state_dict());old_optimizer=opt
        ledger=dict(parameter_digest_before=before_parameters,optimizer_digest_before=before_optimizer)
        context['active_proposal_ledger']=ledger
        try:opt,log,raw=C.protected_proposal(net,obj,opt,schedule,B.forward,cfg,ledger=ledger)
        finally:
            primary=sys.exception()
            try:save(folder/'proposal_ledgers'/f'{i:02d}.json',ledger)
            except BaseException as exc:
                if primary is None:raise
                primary.add_note(f'Proposal persistence failed: {exc}')
        after_parameters=B.state_digest(B.state(net));after_optimizer=state_hash(opt.state_dict())
        evidence=dict(proposal_id=i,direction=log['direction_type'],accepted_scale=log['accepted_scale'],
          parameter_digest_before=before_parameters,parameter_digest_after=after_parameters,
          optimizer_digest_before=before_optimizer,optimizer_digest_after=after_optimizer,
          returned_optimizer_is_new=opt is not old_optimizer,optimizer_state_entries=len(opt.state_dict()['state']))
        if log['accepted_scale']==0:
            assert before_parameters==after_parameters and before_optimizer==after_optimizer and opt is old_optimizer
        elif log['direction_type']=='negative_gradient':
            assert opt is not old_optimizer and opt.state_dict()['state']=={} and log['reset_event'] is not None
        else:assert opt is old_optimizer
        assert all(any(p is q for q in net.parameters()) for g in opt.param_groups for p in g['params'])
        if i>0 and trace[-1]['direction_type']=='adam' and trace[-1]['accepted_scale']==0:
            assert log['direction_type']=='negative_gradient'
        elif i>0 and trace[-1]['direction_type']=='negative_gradient':assert log['direction_type']=='adam'
        log['state_evidence']=evidence;save(folder/'proposal_logs'/f'{i:02d}.json',log);trace.append(log)
        if log['after_loss']<best:
            best=log['after_loss'];chosen=B.state(net);chosen_opt=__import__('copy').deepcopy(opt.state_dict())
            chosen_schedule=dict(vars(schedule));epoch=i+1
        if i==69:
            # Instrument only: preserve this cold run's own original-budget checkpoint.
            torch.save(dict(network=chosen,optimizer=chosen_opt,schedule=chosen_schedule),folder/'checkpoint70.pt')
            torch.save(dict(network=B.state(net),optimizer=opt.state_dict(),schedule=dict(vars(schedule))),folder/'terminal70.pt')
            save(folder/'training70.json',dict(proposals=70,initial_loss=float(initial_loss),selected_loss=best,selected_epoch=epoch,terminal_loss=trace[-1]['after_loss'],trace=trace.copy(),initial_base_digest=initial['base_digest'],save_load_verified=False))
    assert schedule.completed==budget
    torch.save(dict(network=B.state(net),optimizer=opt.state_dict(),schedule=dict(vars(schedule))),folder/'terminal.pt')
    torch.save(dict(network=chosen,optimizer=chosen_opt,schedule=chosen_schedule),folder/'checkpoint.pt')
    # Actual save/load for every new candidate, only its own new checkpoint.
    checkpoint=torch.load(folder/'checkpoint.pt',weights_only=False)
    net.load_state_dict(checkpoint['network']);opt.load_state_dict(checkpoint['optimizer'])
    assert B.state_digest(B.state(net))==B.state_digest(chosen)
    assert state_hash(opt.state_dict())==state_hash(chosen_opt)
    assert checkpoint['schedule']==chosen_schedule
    t,h,replay,checks=B.forward(net,obj,24,False);assert abs(float(replay)-best)<1e-10
    B.np.savez_compressed(folder/'training_transport.npz',selected=t.numpy())
    record=dict(proposals=budget,initial_loss=float(initial_loss),selected_loss=best,selected_epoch=epoch,
      terminal_loss=trace[-1]['after_loss'],trace=trace,replay_history=h,checks=checks,
      accepted=sum(x['accepted_scale']>0 for x in trace),fallback_proposals=sum(x['direction_type']=='negative_gradient' for x in trace),
      save_load_verified=True,initialization=initial,initial_base_digest=initial['base_digest'])
    save(folder/'training.json',record)
    return net,obj,record


