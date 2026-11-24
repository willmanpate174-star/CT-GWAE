"""Model primitives for the single frozen training entry, run_model.py."""
from __future__ import annotations
import random
import numpy as np
import torch
from data_io import DT

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class MultilayerGWEncoder(torch.nn.Module):
    def __init__(self, operators, X, K, steps):
        super().__init__()
        self.operators, self.X = operators, X
        L, d0, d = len(operators), X.shape[1], 24
        self.W0 = torch.nn.Parameter(torch.randn(d0, d, dtype=DT) * .22)
        self.W1 = torch.nn.Parameter(torch.randn(d, d, dtype=DT) * .18)
        self.fusion = torch.nn.Linear(L * d, d, dtype=DT)
        self.head = torch.nn.Linear(d, K, dtype=DT)
        torch.nn.init.normal_(self.head.weight, std=.45)
        torch.nn.init.normal_(self.head.bias, std=.12)
        self.log_eta = torch.nn.Parameter(torch.full((steps,), np.log(np.expm1(8.0)), dtype=DT))

    def forward(self):
        H = torch.tanh(torch.stack([torch.sparse.mm(op, self.X @ self.W0)
                                    for op in self.operators]))
        H = torch.tanh(torch.stack([torch.sparse.mm(op, h)
                                    for op, h in zip(self.operators, H @ self.W1)]))
        H = H.permute(1, 0, 2).reshape(len(self.X), -1)
        return self.head(torch.tanh(self.fusion(H)))

def node_features(layers, seed):
    degrees = np.column_stack([np.log1p(np.asarray(A.sum(axis=1)).ravel())
                               for A in layers])
    degrees = (degrees - degrees.mean(0)) / (degrees.std(0) + 1e-8)
    noise = np.random.default_rng(seed + 991).normal(size=(layers[0].shape[0], 16)) * .25
    return torch.tensor(np.c_[degrees, noise], dtype=DT)

def initial_transport(objective, logits):
    P = torch.softmax(logits, dim=1)
    M = (1.0 - objective.delta) * P + objective.delta / objective.K
    return objective.p[:, None] * M

def unroll(objective, logits, log_eta, create_graph=True):
    T = initial_transport(objective, logits)
    log_m = torch.log(T / objective.p[:, None])
    history = []
    for raw_eta in log_eta:
        if not create_graph:
            T = T.detach().requires_grad_(True)
            log_m = log_m.detach()
        current = objective.loss(T)
        grad = torch.autograd.grad(current, T, create_graph=create_graph,
                                   retain_graph=create_graph)[0]
        eta_source = raw_eta if create_graph else raw_eta.detach()
        eta, accepted = torch.nn.functional.softplus(eta_source), False
        if getattr(objective,'bounded_step',False):
            # Remove only a row-constant gradient (softmax invariant), then
            # smoothly limit each step's log-odds movement to log(2).
            grad=grad-grad.mean(1,keepdim=True)
            norm=grad.abs().max(1,keepdim=True).values if getattr(objective,'row_bounded',False) else grad.abs().max()
            gamma=np.log(2)/2
            eta=eta/(1+eta*norm/gamma)
            objective.audit['bounded_step_calls']=objective.audit.get('bounded_step_calls',0)+1
        for _ in range(16):
            # log_softmax is algebraically the exact M-centred KL mirror map.
            # Keeping log_m avoids a probability floor or log(0) correction.
            candidate_log_m = torch.log_softmax(log_m - eta * grad, dim=1)
            candidate = objective.p[:, None] * torch.exp(candidate_log_m)
            if getattr(objective,'bounded_step',False):
                movement=candidate_log_m-log_m
                span=float((movement.max(1).values-movement.min(1).values).max().detach())
                objective.audit['max_log_odds_change']=max(objective.audit.get('max_log_odds_change',0.),span)
                if span>np.log(2)+1e-10:raise RuntimeError('Log-odds bound violated')
            candidate_loss = objective.loss(candidate)
            # The positivity test is redundant in exact arithmetic; it rejects
            # only a floating-point underflow that would leave the KL domain.
            if (float(candidate_loss.detach()) <= float(current.detach())
                    and bool((candidate > 0).all())):
                T, log_m, accepted = candidate, candidate_log_m, True
                history.append(float(candidate_loss.detach()))
                break
            eta = eta * .5
        if not accepted:
            history.append(float(current.detach()))
    return (T if create_graph else T.detach()), history
