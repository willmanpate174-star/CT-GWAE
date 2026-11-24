"""One common inverse-mass template, jointly fitted across active layers.

qbar is the unweighted mean of the active layers' target masses. Each layer
retains its own source/target measure in GW, but B itself has no layer index.
The 1/null distortion weights, occupancy penalty and encoder are unchanged.
"""
import numpy as np
import torch
from geometric_train_model import LayerGeometricSingleGW


def torch_fit(q, diagonal, null):
    qbar = q.mean(0)
    inv = torch.where(qbar > 0, 1 / torch.where(qbar > 0, qbar, 1), 0)
    w = (1 / null); w = w / w.sum()
    h = (q.square() * inv).sum(1)
    g = (q.square() * inv.square()).sum(1)
    y = (diagonal * inv).sum(1)
    hh, gg, yy = (w * h).sum(), (w * g).sum(), (w * y).sum()
    det = gg - hh.square()
    if float(det.detach()) <= 1e-13:
        if abs(float((yy-hh).detach())) > 1e-10 * max(1.,abs(float(hh.detach()))):
            raise FloatingPointError('Unexpected degenerate common-template Gram')
        b, a = qbar.new_ones(()), qbar.new_zeros(())
    else:
        bs = torch.stack(((gg - hh * yy) / det, yy.new_ones(()), yy.new_zeros(())))
        aa = torch.stack(((yy - hh) / det, yy.new_zeros(()), yy / gg))
        gains = 2*bs + 2*aa*yy - bs.square() - 2*bs*aa*hh - aa.square()*gg
        feasible = (bs.detach() >= 0) & (aa.detach() >= 0)
        chosen = torch.where(feasible, gains.detach(), -torch.inf).argmax()
        b, a = bs[chosen], aa[chosen]
    gain = 2*b + 2*a*y - b.square() - 2*b*a*h - a.square()*g
    return dict(b=b, alpha=a, qbar=qbar, inverse_common_mass=inv,
                common_h=h, common_g=g, common_y=y, gain=gain,
                gram=torch.stack((hh, gg, yy)), gram_determinant=det)


class SharedTemplateObjective(LayerGeometricSingleGW):
    def profile(self, T):
        original = super().profile(T)
        r = torch_fit(original['q'], original['diagonal_edge_mass'], self.null)
        b, a = r.pop('b'), r.pop('alpha')
        original.update(r)
        layer = (self.nu - r['gain']) / self.null
        original.update(b=b.expand_as(self.null), alpha=a.expand_as(self.null),
                        layer_structure=layer, structure=layer.mean(),
                        b_zero_face=(b == 0).expand_as(self.null),
                        template_coefficients_are_layerwise=False,
                        template_is_identical_across_layers=True,
                        template_basis='inverse_mean_active_layer_target_mass')
        return original

    def profiles(self, T):
        r = self.profile(T)
        one = r['b'][0] + r['alpha'][0] * torch.diag(r['inverse_common_mass'])
        r['template'] = one[None].expand(len(self.null), -1, -1)
        return r

    def analytic_gradient(self, T):
        t = T.detach().requires_grad_(True)
        return torch.autograd.grad(self.loss(t), t)[0]

    def analytic_hvp(self, T, direction):
        t = T.detach().requires_grad_(True)
        g = torch.autograd.grad(self.loss(t), t, create_graph=True)[0]
        return torch.autograd.grad((g * direction).sum(), t)[0]


class SharedTemplateHard:
    def __init__(self, original):
        self.base = original

    def score(self, labels):
        r = self.base.score(labels)
        ps = r['profiles']
        q = torch.tensor([p['q'] for p in ps], dtype=torch.float64)
        d = torch.tensor([p['diagonal_edge_mass'] for p in ps], dtype=torch.float64)
        null = torch.tensor([p['null'] for p in ps], dtype=torch.float64)
        fit = torch_fit(q, d, null)
        b, a = float(fit['b']), float(fit['alpha'])
        diagonal = b + a * fit['inverse_common_mass'].numpy()
        for i, p in enumerate(ps):
            gain = float(fit['gain'][i])
            p.update(b=b, alpha=a, gain=gain, structure=(p['nu']-gain)/p['null'],
                     template_diagonal=diagonal.tolist(), template_off_diagonal=b,
                     common_q=fit['qbar'].tolist(),
                     branch='shared_template_joint_NNLS',
                     template_is_identical_across_layers=True)
        r.update(structure=float(np.mean([p['structure'] for p in ps])),
                 template_is_identical_across_layers=True)
        r['objective'] = r['structure'] + r['node_mass']
        r['score_domain'] += ':shared_template'
        return r
