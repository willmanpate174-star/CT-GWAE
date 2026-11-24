"""Three single-mechanism changes to the frozen per-layer single GW model."""
import numpy as np
import torch
from geometric_train_model import LayerGeometricSingleGW


def inverse_coeff(x, count):
    """NNLS for Gram [[1,1],[1,count]] and RHS [1,x]."""
    interior = torch.relu(x - 1) / (count - 1)
    boundary = x > count
    alpha = torch.where(boundary, x / count, interior)
    b = torch.where(boundary, torch.zeros_like(x), 1 - interior)
    return b, alpha


def ordinary_coeff(q, diagonal):
    """B=b11^T+alpha I; enumerate the interior and both nonnegative faces."""
    h = q.square().sum(1)
    cross = (q[:, :, None] * q[:, None, :] *
             (1 - torch.eye(q.shape[1], dtype=q.dtype))).sum((1, 2))
    y = diagonal.sum(1)
    det = h * cross
    if not bool((det.detach() > 0).all()):
        raise FloatingPointError('Positive soft support Gram determinant required')
    b0, a0 = (h - h * y) / det, (y - h) / det
    bs = torch.stack((b0, torch.ones_like(h), torch.zeros_like(h)))
    aa = torch.stack((a0, torch.zeros_like(h), y / h))
    gains = 2 * bs + 2 * aa * y - bs.square() - 2 * bs * aa * h - aa.square() * h
    feasible = (bs.detach() >= 0) & (aa.detach() >= 0)
    chosen = torch.where(feasible, gains.detach(), -torch.inf).argmax(0)
    b = bs.gather(0, chosen[None]).squeeze(0)
    alpha = aa.gather(0, chosen[None]).squeeze(0)
    return b, alpha, h, y


class AblationObjective(LayerGeometricSingleGW):
    def __init__(self, layers, K, variant):
        super().__init__(layers, K)
        if variant not in ('Full', 'shared_coeff', 'no_dist_norm', 'no_mass_correction'):
            raise ValueError(variant)
        self.variant = variant

    def profile(self, T):
        r = super().profile(T)
        if self.variant == 'Full':
            return r
        if self.variant == 'shared_coeff':
            weights = 1 / self.null
            pooled = (weights * r['x']).sum() / weights.sum()
            b, alpha = inverse_coeff(pooled, self.K)
            r['b'] = b.expand_as(r['x'])
            r['alpha'] = alpha.expand_as(r['x'])
            r['b_zero_threshold_excess'] = (pooled-self.K).expand_as(r['x'])
            r['template_coefficients_are_layerwise'] = False
        elif self.variant == 'no_mass_correction':
            b, alpha, h, y = ordinary_coeff(r['q'], r['diagonal_edge_mass'])
            r.update(b=b, alpha=alpha, ordinary_h=h, ordinary_y=y,
                     template_basis='ordinary_diagonal',b_zero_threshold_excess=y-1)
        b, a = r['b'], r['alpha']
        if self.variant == 'no_mass_correction':
            h, y = r['ordinary_h'], r['ordinary_y']
            gain = 2*b + 2*a*y - b.square() - 2*b*a*h - a.square()*h
        else:
            gain = 2*b + 2*a*r['x'] - b.square() - 2*b*a - self.K*a.square()
        raw = self.nu - gain
        layer = raw if self.variant == 'no_dist_norm' else raw / self.null
        r.update(gain=gain, layer_structure=layer, structure=layer.mean(),b_zero_face=b==0)
        return r

    def analytic_gradient(self, T):
        # The frozen closed-form derivative belongs to Full only. Modified
        # objectives use their own exact differentiable loss, never that formula.
        if self.variant=='Full':return super().analytic_gradient(T)
        t=T.detach().requires_grad_(True)
        return torch.autograd.grad(self.loss(t),t)[0]

    def analytic_hvp(self, T, direction):
        if self.variant=='Full':return super().analytic_hvp(T,direction)
        t=T.detach().requires_grad_(True)
        g=torch.autograd.grad(self.loss(t),t,create_graph=True)[0]
        return torch.autograd.grad((g*direction).sum(),t)[0]

    def profiles(self, T):
        r = self.profile(T)
        basis = (torch.eye(self.K, dtype=T.dtype)[None].expand(len(self.null), -1, -1)
                 if self.variant == 'no_mass_correction' else torch.diag_embed(1/r['q']))
        r['template'] = r['b'][:, None, None] + r['alpha'][:, None, None] * basis
        return r


class HardAblation:
    def __init__(self, frozen_scorer, variant):
        self.base, self.variant = frozen_scorer, variant

    def score(self, labels):
        r = self.base.score(labels)
        if self.variant == 'Full':
            return r
        ps = r['profiles']
        if self.variant == 'shared_coeff':
            w = np.array([1/p['null'] for p in ps])
            x = float(np.dot(w, [p['x'] for p in ps]) / w.sum())
            m = float(np.dot(w, [p['m'] for p in ps]) / w.sum())
            if all(p['m']==1 for p in ps):
                b, a = 1., 0.
            elif x <= 1:
                b, a = 1., 0.
            elif x <= m:
                a = (x-1)/(m-1); b = 1-a
            else:
                b, a = 0., x/m
            for p in ps:
                p.update(b=b, alpha=a, gain=2*b+2*a*p['x']-b*b-2*b*a-p['m']*a*a)
        elif self.variant == 'no_mass_correction':
            for p in ps:
                q = np.array(p['q']); h = float(q @ q)
                cross = float((q[:, None]*q[None, :]*(1-np.eye(len(q)))).sum())
                y = float(sum(p['diagonal_edge_mass']))
                if p['m'] == 1:
                    b, a = 1., 0.
                else:
                    candidates = [((h-h*y)/(h*cross), (y-h)/(h*cross)), (1.,0.), (0.,y/h)]
                    candidates = [(b,a) for b,a in candidates if b>=0 and a>=0]
                    b,a = max(candidates,key=lambda v:2*v[0]+2*v[1]*y-v[0]**2-2*v[0]*v[1]*h-v[1]**2*h)
                p.update(b=b,alpha=a,gain=2*b+2*a*y-b*b-2*b*a*h-a*a*h,
                         template_basis='ordinary_diagonal')
        for p in ps:
            raw = p['nu']-p['gain']
            p['structure'] = raw if self.variant == 'no_dist_norm' else raw/p['null']
            diagonal = np.full(r['K'],p['b'],dtype=float)
            active = np.array(p['positive_support_labels'],dtype=int)
            diagonal[active] += (p['alpha'] if self.variant == 'no_mass_correction'
                                 else p['alpha']/np.array(p['q'])[active])
            p.update(template_diagonal=diagonal.tolist(),template_off_diagonal=p['b'],
                     branch=self.variant)
        r['structure'] = float(np.mean([p['structure'] for p in ps]))
        r['objective'] = r['structure']+r['node_mass']
        r['score_domain'] += ':'+self.variant
        return r


class BObjective(AblationObjective):
    """B changes only the coefficient of the global node-occupancy penalty."""
    def __init__(self,layers,K,variant,lambda_rho=.02):
        super().__init__(layers,K,variant)
        if not np.isfinite(lambda_rho) or lambda_rho<0:
            raise ValueError('Nonnegative finite lambda_rho required')
        self.lambda_rho=float(lambda_rho)

    def parts(self,T):
        result=self.profile(T)
        mass=self.lambda_rho*(torch.relu(self.q_floor-T.sum(0))/self.q_floor).square().sum()
        return result['structure'],mass,T.new_zeros(())

    def analytic_gradient(self,T):
        t=T.detach().requires_grad_(True)
        return torch.autograd.grad(self.loss(t),t)[0]

    def analytic_hvp(self,T,direction):
        t=T.detach().requires_grad_(True)
        g=torch.autograd.grad(self.loss(t),t,create_graph=True)[0]
        return torch.autograd.grad((g*direction).sum(),t)[0]


class BHard(HardAblation):
    def __init__(self,base,variant,lambda_rho=.02):
        super().__init__(base,variant)
        self.lambda_rho=float(lambda_rho)

    def score(self,labels):
        r=super().score(labels)
        if self.lambda_rho!=.02:
            q=np.asarray(r['node_q']);floor=r['node_q_floor']
            r['node_mass']=float(self.lambda_rho*np.square(np.maximum(floor-q,0)/floor).sum())
            r['objective']=r['structure']+r['node_mass']
        r['lambda_rho']=self.lambda_rho
        return r
