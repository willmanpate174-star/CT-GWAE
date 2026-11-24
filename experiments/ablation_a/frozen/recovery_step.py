"""Prepared optimizer variant. No torch import or model computation at import."""
from dataclasses import dataclass
import copy
import math
import time

SCALES=tuple(.5**j for j in range(11))

@dataclass
class DirectionSchedule:
    """A rejected Adam changes the NEXT budget proposal, never appends one."""
    budget:int=70
    completed:int=0
    next_direction:str='adam'

    def current(self):
        if self.completed>=self.budget:raise RuntimeError('Proposal budget exhausted')
        return self.next_direction

    def finish(self,direction,accepted):
        assert direction==self.current()
        reset=direction=='negative_gradient'
        self.next_direction='negative_gradient' if direction=='adam' and not accepted else 'adam'
        self.completed+=1
        return dict(reset_adam=reset,next_direction=self.next_direction,completed_proposals=self.completed)

def strictly_accepted(before,after):
    return math.isfinite(before) and math.isfinite(after) and after<before-1e-12

def rebuild_adam(optimizer):
    """Preserve groups/options; fresh state removes step, all moments, AMSGrad too."""
    if type(optimizer).__name__!='Adam':raise TypeError('This protocol supports Adam only')
    groups=[{key:(list(value) if key=='params' else copy.deepcopy(value)) for key,value in group.items()}
            for group in optimizer.param_groups]
    fresh=type(optimizer)(groups,**dict(optimizer.defaults))
    assert not fresh.state and fresh.state_dict()['state']=={}
    return fresh

def protected_proposal(net,obj,optimizer,schedule,forward,cfg):
    """Future runner API: (new_optimizer, log, raw_gradient_map).

    forward is the unchanged v2 forward(net,obj,24,gradient)->T/history/loss/checks.
    That callback enforces positive coupling/q and original loss finiteness.
    Exceptions are fatal to the candidate; the caller persists error/cost evidence.
    """
    import torch  # Only future explicitly authorized numerical integration imports it.
    assert cfg['learning_rate']==.012 and cfg['gradient_clip_norm']==5
    assert tuple(cfg['outer_line_search_scales'])==SCALES
    assert cfg['acceptance_improvement']==1e-12 and cfg['proposals_per_candidate']==70
    assert schedule.budget==70
    assert all(group['lr']==.012 and group.get('weight_decay',0)==0 for group in optimizer.param_groups)
    direction=schedule.current();proposal_id=schedule.completed
    started=time.perf_counter();cpu=time.process_time();calls={'S24_gradient_forward':0,'S24_primal_forward':0,'backward':0}

    def evaluate(gradient):
        calls['S24_gradient_forward' if gradient else 'S24_primal_forward']+=1
        return forward(net,obj,24,gradient)

    def snapshot():return {name:value.detach().clone() for name,value in net.state_dict().items()}

    optimizer.zero_grad();_,_,loss,checks=evaluate(True)
    before_loss=float(loss.detach());assert math.isfinite(before_loss)
    loss.backward();calls['backward']+=1
    parameters=dict(net.named_parameters())
    raw={name:(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for name,p in parameters.items()}
    assert all(bool(torch.isfinite(g).all()) for g in raw.values())
    raw_norm=float(torch.cat([g.reshape(-1) for g in raw.values()]).norm())
    torch.nn.utils.clip_grad_norm_(net.parameters(),5)
    clipped={name:(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for name,p in parameters.items()}
    clipped_norm=float(torch.cat([g.reshape(-1) for g in clipped.values()]).norm())
    before=snapshot();moments_before=copy.deepcopy(optimizer.state_dict())
    if direction=='adam':
        optimizer.step();proposal=snapshot()
    else:
        # Recomputed current gradient, not the previous rejected iteration's gradient.
        proposal={name:(value-.012*clipped[name] if name in parameters else value.clone())
                  for name,value in before.items()}
    vectors={name:proposal[name]-before[name] for name in parameters}
    assert all(bool(torch.isfinite(v).all()) for v in vectors.values())
    direction_norm=float(torch.cat([v.reshape(-1) for v in vectors.values()]).norm())
    gtd=float(sum((clipped[name]*vectors[name]).sum() for name in parameters))
    attempts=[];accepted_scale=0.
    for scale in SCALES:
        net.load_state_dict({name:before[name]+scale*(proposal[name]-before[name]) for name in before})
        _,_,candidate_loss,candidate_checks=evaluate(False)
        value=float(candidate_loss.detach());assert math.isfinite(value)
        attempts.append(dict(scale=scale,loss=value,checks=candidate_checks))
        if strictly_accepted(before_loss,value):accepted_scale=scale;break
    if accepted_scale==0:
        net.load_state_dict(before);optimizer.load_state_dict(moments_before)
    # Adam accepted with scaled parameters keeps the full proposal's moments,
    # exactly as v2. Negative-gradient proposals never call optimizer.step().
    _,_,after_loss,after_checks=evaluate(False)
    after=float(after_loss.detach());assert math.isfinite(after) and after<=before_loss+1e-12
    transition=schedule.finish(direction,accepted_scale>0)
    reset=None
    if transition['reset_adam']:
        old_state_keys=sorted({key for values in optimizer.state_dict()['state'].values() for key in values})
        optimizer=rebuild_adam(optimizer)
        reset=dict(event='rebuild_after_negative_gradient_proposal',reason='uniform after acceptance or rejection',
                   removed_state_fields=old_state_keys,new_state_entries=0)
    log=dict(proposal_id=proposal_id,direction_type=direction,before_loss=before_loss,after_loss=after,
        clip_before_norm=raw_norm,clip_after_norm=clipped_norm,clip_after_gTd=gtd,direction_norm=direction_norm,
        negative_gradient_expected_gTd=(-.012*clipped_norm**2 if direction=='negative_gradient' else None),
        accepted_scale=accepted_scale,attempts=attempts,checks=checks,after_checks=after_checks,
        reset_event=reset,transition=transition,S24_calls=calls,
        S24_calls_total=calls['S24_gradient_forward']+calls['S24_primal_forward'],
        inner_unroll_steps_per_S24=24,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu)
    return optimizer,log,raw
