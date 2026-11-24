"""New static integration: Adam clip preserved, recovery uses raw gradient; no torch on import."""
import copy
import math
import time
from recovery_step import DirectionSchedule, SCALES, strictly_accepted, rebuild_adam


class LedgerPersistenceError(RuntimeError):
    pass


def finite_scalar(value, name):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite '+name)
    return value


def direction_scalar_diagnostics(clipped_norm, direction_norm, actual_gtd, direction):
    finite_scalar(clipped_norm, 'clipped gradient norm')
    finite_scalar(direction_norm, 'direction norm')
    finite_scalar(actual_gtd, 'actual clipped gTd')
    expected = finite_scalar(-.012*clipped_norm**2, 'expected clipped gTd') if direction == 'negative_gradient' else None
    delta = finite_scalar(actual_gtd-expected, 'actual minus expected gTd') if expected is not None else None
    return dict(negative_gradient_expected_gTd=expected,
                actual_minus_expected_gTd=delta, zero_direction=direction_norm == 0.)


class ProposalTelemetry:
    """Caller-owned JSON ledger. Callback receives an isolated snapshot.

    Callback errors are retained and deferred until finalization. They never
    replace an active computation exception. A successful computation with any
    persistence error raises LedgerPersistenceError and cannot look successful.
    Use a fresh ledger per proposal; caller metadata survives initialization.
    """
    def __init__(self, ledger=None, callback=None):
        self.ledger = {} if ledger is None else ledger
        self.callback = callback
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.ledger.update(status='running', phase='initialization', direction_type=None,
            proposal_id=None, scale=None, events=[], completed_trials=[],
            calls={name: dict(started=0, completed=0) for name in
                   ('S24_gradient_forward', 'S24_primal_forward', 'backward')},
            operations={}, exception=None, persistence_errors=[], finalized=False,
            new_optimizer_returned=False)

    def refresh_time(self):
        self.ledger['partial_wall_seconds'] = time.perf_counter() - self.started
        self.ledger['partial_cpu_seconds'] = time.process_time() - self.cpu_started

    def publish(self):
        self.refresh_time()
        if self.callback is not None:
            try:
                self.callback(copy.deepcopy(self.ledger))
            except BaseException as exc:
                # Even a failing final callback must not replace the real error.
                self.ledger['persistence_errors'].append(dict(
                    type=type(exc).__name__, message=str(exc), phase=self.ledger['phase']))
                self.refresh_time()

    def mark(self, phase, **fields):
        self.ledger.update(phase=phase, **fields)
        self.publish()

    def operation(self, name, function, **fields):
        counts = self.ledger['calls'] if name in self.ledger['calls'] else self.ledger['operations']
        entry = counts.setdefault(name, dict(started=0, completed=0))
        entry['started'] += 1
        self.ledger['events'].append(dict(operation=name, event='started', **fields))
        self.mark(name, **fields)
        result = function()  # Exceptions propagate to run(), never converted to rejection.
        entry['completed'] += 1
        self.ledger['events'].append(dict(operation=name, event='completed', **fields))
        self.publish()
        return result

    def run(self, function):
        primary_error = None
        try:
            self.publish()
            result = function(self)
            self.ledger['status'] = 'success'
            return result
        except BaseException as exc:
            primary_error = exc
            self.ledger.update(status='failed', exception=dict(
                type=type(exc).__name__, message=str(exc), phase=self.ledger['phase'],
                direction_type=self.ledger['direction_type'], scale=self.ledger['scale']))
            raise
        finally:
            self.ledger['finalized'] = True
            if primary_error is None and self.ledger['persistence_errors']:
                self.ledger['status'] = 'persistence_failed'
            self.ledger['new_optimizer_returned'] = primary_error is None and not self.ledger['persistence_errors']
            self.publish()
            if primary_error is None and self.ledger['persistence_errors']:
                self.ledger['status'] = 'persistence_failed'
                self.ledger['new_optimizer_returned'] = False
                raise LedgerPersistenceError('Proposal persistence callback failed; inspect caller-owned ledger')


def protected_proposal(net, obj, optimizer, schedule, forward, cfg,
                       ledger=None, ledger_callback=None):
    """Same three-value return; optional external ledger survives every raise.

    Caller must save ledger on failure and replace the returned optimizer on
    success. Persistence callback performs actual storage; no file is opened here.
    """
    telemetry = ProposalTelemetry(ledger, ledger_callback)
    return telemetry.run(lambda recorder: _protected_proposal(
        net, obj, optimizer, schedule, forward, cfg, recorder))


def _protected_proposal(net, obj, optimizer, schedule, forward, cfg, telemetry):
    telemetry.mark('configuration', direction_type=schedule.current(), proposal_id=schedule.completed)
    import torch  # Future explicitly authorized numerical execution only.
    assert cfg['learning_rate'] == .012 and cfg['gradient_clip_norm'] == 5
    assert tuple(cfg['outer_line_search_scales']) == SCALES
    assert cfg['acceptance_improvement'] == 1e-12 and cfg['proposals_per_candidate'] == 600
    assert schedule.budget == 600
    assert all(group['lr'] == .012 and group.get('weight_decay', 0) == 0 for group in optimizer.param_groups)
    direction = schedule.current(); proposal_id = schedule.completed
    started = time.perf_counter(); cpu = time.process_time()

    def evaluate(gradient):
        name = 'S24_gradient_forward' if gradient else 'S24_primal_forward'
        return telemetry.operation(name, lambda: forward(net, obj, 24, gradient))

    def snapshot():
        return {name: value.detach().clone() for name, value in net.state_dict().items()}

    telemetry.operation('zero_grad', optimizer.zero_grad)
    _, _, loss, checks = evaluate(True)
    telemetry.mark('before_loss_validation')
    before_loss = float(loss.detach()); assert math.isfinite(before_loss)
    telemetry.mark('before_loss_recorded', before_loss=before_loss)
    telemetry.operation('backward', loss.backward)
    telemetry.mark('raw_gradient_validation')
    parameters = dict(net.named_parameters())
    raw = {name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
           for name, p in parameters.items()}
    assert all(bool(torch.isfinite(g).all()) for g in raw.values())
    telemetry.mark('raw_norm_validation')
    raw_norm = float(torch.cat([g.reshape(-1) for g in raw.values()]).norm())
    finite_scalar(raw_norm, 'raw gradient norm before clipping')
    if direction == 'adam':
        telemetry.operation('gradient_clip', lambda: torch.nn.utils.clip_grad_norm_(net.parameters(), 5))
    else:
        telemetry.mark('raw_negative_gradient_no_clip')
    telemetry.mark('clipped_gradient_validation')
    clipped = {name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
               for name, p in parameters.items()}
    assert all(bool(torch.isfinite(g).all()) for g in clipped.values()), 'Nonfinite clipped gradient element'
    clipped_norm = float(torch.cat([g.reshape(-1) for g in clipped.values()]).norm())
    finite_scalar(clipped_norm, 'clipped gradient norm')
    telemetry.mark('state_snapshot', clip_before_norm=raw_norm, clip_after_norm=clipped_norm)
    before = snapshot(); moments_before = copy.deepcopy(optimizer.state_dict())
    if direction == 'adam':
        telemetry.operation('adam_step', optimizer.step)
        proposal = snapshot()
    else:
        telemetry.mark('negative_gradient_proposal')
        proposal = {name: (value-.012*clipped[name] if name in parameters else value.clone())
                    for name, value in before.items()}
    telemetry.mark('direction_validation')
    vectors = {name: proposal[name]-before[name] for name in parameters}
    assert all(bool(torch.isfinite(v).all()) for v in vectors.values())
    direction_norm = float(torch.cat([v.reshape(-1) for v in vectors.values()]).norm())
    gtd = float(sum((clipped[name]*vectors[name]).sum() for name in parameters))
    diagnostics = direction_scalar_diagnostics(clipped_norm, direction_norm, gtd, direction)
    telemetry.mark('line_search', clip_after_gTd=gtd, direction_norm=direction_norm, **diagnostics)
    attempts = []; accepted_scale = 0.
    for scale in SCALES:
        telemetry.operation('trial_parameter_load', lambda: net.load_state_dict(
            {name: before[name]+scale*(proposal[name]-before[name]) for name in before}), scale=scale)
        _, _, candidate_loss, candidate_checks = evaluate(False)
        telemetry.mark('trial_loss_validation')
        value = float(candidate_loss.detach()); assert math.isfinite(value)
        attempts.append(dict(scale=scale, loss=value, checks=candidate_checks))
        telemetry.ledger['completed_trials'].append(dict(scale=scale, loss=value))
        telemetry.publish()
        if strictly_accepted(before_loss, value):
            accepted_scale = scale
            break
    telemetry.mark('acceptance_decision', accepted_scale=accepted_scale, scale=None)
    if accepted_scale == 0:
        telemetry.operation('parameter_restore', lambda: net.load_state_dict(before))
        telemetry.operation('optimizer_state_restore', lambda: optimizer.load_state_dict(moments_before))
    # Accepted Adam retains full proposed moments, as in the original module.
    _, _, after_loss, after_checks = evaluate(False)
    telemetry.mark('after_loss_validation')
    after = float(after_loss.detach()); assert math.isfinite(after) and after <= before_loss+1e-12
    transition = telemetry.operation('schedule_finish', lambda: schedule.finish(direction, accepted_scale > 0))
    transition['inherited_reset_adam'] = transition['reset_adam']
    transition['reset_adam'] = direction == 'negative_gradient' and accepted_scale > 0
    if accepted_scale == 0:
        assert all(torch.equal(snapshot()[n], before[n]) for n in before)
        import io, hashlib
        def state_hash(value):
            stream = io.BytesIO(); torch.save(value, stream)
            return hashlib.sha256(stream.getvalue()).hexdigest()
        assert state_hash(optimizer.state_dict()) == state_hash(moments_before)
        assert after == before_loss
    else:
        assert after < before_loss - 1e-12
    telemetry.mark('schedule_recorded', transition=transition, after_loss=after)
    reset = None
    if transition['reset_adam']:
        telemetry.mark('reset_state_inspection')
        old_state_keys = sorted({key for values in optimizer.state_dict()['state'].values() for key in values})
        optimizer = telemetry.operation('adam_reset', lambda: rebuild_adam(optimizer))
        reset = dict(event='rebuild_after_negative_gradient_proposal', reason='accepted raw negative-gradient recovery only',
                     removed_state_fields=old_state_keys, new_state_entries=0)
    telemetry.mark('log_construction', reset_event=reset)
    calls = {name: count['completed'] for name, count in telemetry.ledger['calls'].items()}
    log = dict(proposal_id=proposal_id, direction_type=direction, before_loss=before_loss, after_loss=after,
        clip_before_norm=raw_norm, clip_after_norm=clipped_norm, clip_after_gTd=gtd, direction_norm=direction_norm,
        **diagnostics,
        accepted_scale=accepted_scale, attempts=attempts, checks=checks, after_checks=after_checks,
        reset_event=reset, transition=transition, S24_calls=calls,
        S24_calls_total=calls['S24_gradient_forward']+calls['S24_primal_forward'],
        inner_unroll_steps_per_S24=24, wall_seconds=time.perf_counter()-started,
        cpu_seconds=time.process_time()-cpu)
    return optimizer, log, raw

