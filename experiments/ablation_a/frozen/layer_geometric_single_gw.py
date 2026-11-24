"""One mechanism candidate: per-layer geometric sqrt-degree GW, common membership.
Static unregistered training kernel; no runner, truth, initialization or decoder.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch

STAT_SOURCE = Path(__file__).resolve().parent / 'layer_geometric_hard_score.py'
_spec = importlib.util.spec_from_file_location('geometric_fixed_statistics', STAT_SOURCE)
_stats = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _stats
_spec.loader.exec_module(_stats)


def _sparse(a):
    c = a.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.stack((c.row, c.col)), dtype=torch.long),
        torch.tensor(c.data.copy(), dtype=torch.float64), c.shape, dtype=torch.float64).coalesce()


class LayerGeometricSingleGW:
    def __init__(self, layers, K):
        stats = _stats.LayerGeometricFixedPartition(layers, K)
        self.n, self.K = stats.n, stats.K
        self.p = torch.full((self.n,), 1/self.n, dtype=torch.float64)
        self.delta, self.q_floor = .01, 1/(4*self.K)
        self.bounded_step, self.row_bounded = True, True
        self.active_layers = [r['layer_id'] for r in stats.layers]
        self.zero_volume_layers = list(stats.zero_volume_layers)
        self.inactive_variance_layers = list(stats.inactive_variance_layers)
        self.zero_source_nodes_by_layer = [list(r['zero_source_nodes']) for r in stats.layers]
        self.source_measures = torch.tensor(np.stack([r['mu'] for r in stats.layers]), dtype=torch.float64)
        self.raw = [_sparse(r['raw']) for r in stats.layers]
        self.nu = torch.tensor([r['nu'] for r in stats.layers], dtype=torch.float64)
        self.null = self.nu-1
        self.audit = {}

    @staticmethod
    def _finite(value, name):
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError('Nonfinite '+name)

    def profile(self, T):
        if T.shape != (self.n, self.K) or T.dtype != torch.float64 or T.device != self.p.device:
            raise ValueError('CPU float64 N by K transport required')
        self._finite(T, 'transport')
        if bool((T.detach() < 0).any()) or not torch.allclose(T.detach().sum(1), self.p, rtol=0, atol=1e-12):
            raise ValueError('Nonnegative uniform-row transport required')
        W = T/self.p[:, None]
        q = self.source_measures @ W
        self._finite(q, 'layer target masses')
        if not bool((q.detach() > 0).all()):
            raise ValueError('Strictly positive soft target mass in every layer required')
        diagonal = torch.stack([(W*torch.sparse.mm(a, W)).sum(0) for a in self.raw])
        x = (diagonal/q).sum(1)
        self._finite(x, 'layer contractions')
        if bool((x.detach() < 0).any()):
            raise FloatingPointError('Negative geometric contraction; no clipping')
        boundary = x > self.K
        interior_alpha = torch.relu(x-1)/(self.K-1)
        alpha = torch.where(boundary, x/self.K, interior_alpha)
        b = torch.where(boundary, torch.zeros_like(x), 1-interior_alpha)
        gain = 2*b+2*alpha*x-b.square()-2*b*alpha-self.K*alpha.square()
        layer_structure = (self.nu-gain)/self.null
        structure = layer_structure.mean()
        self._finite(b, 'baselines'); self._finite(alpha, 'contrasts')
        self._finite(layer_structure, 'structure')
        if bool((b.detach() < 0).any()) or bool((alpha.detach() < 0).any()):
            raise FloatingPointError('Nonnegative layer coefficients required')
        return dict(q=q, diagonal_edge_mass=diagonal, x=x, b=b, alpha=alpha,
            gain=gain, layer_structure=layer_structure, structure=structure,
            active_layers=self.active_layers, template_coefficients_are_layerwise=True,
            source_measures_are_layerwise=True, b_zero_threshold_excess=x-self.K,
            b_zero_face=boundary, template_basis='inverse_layer_target_mass')

    def parts(self, T):
        result = self.profile(T)
        node_q = T.sum(0)
        mass = .02*(torch.relu(self.q_floor-node_q)/self.q_floor).square().sum()
        self._finite(mass, 'node occupancy penalty')
        return result['structure'], mass, T.new_zeros(())

    def loss(self, T):
        structure, mass, _ = self.parts(T)
        total = structure+mass
        self._finite(total, 'complete loss')
        return total

    def profiles(self, T):
        result = self.profile(T)
        result['template'] = result['b'][:,None,None]+result['alpha'][:,None,None]*torch.diag_embed(1/result['q'])
        self._finite(result['template'], 'layer community templates')
        return result

    def analytic_gradient(self, T):
        result = self.profile(T); W = T/self.p[:,None]
        terms = []
        for i, a in enumerate(self.raw):
            q, mu = result['q'][i], self.source_measures[i]
            V, D = torch.sparse.mm(a, W), result['diagonal_edge_mass'][i]
            gx = 2*V/q[None,:]-mu[:,None]*D[None,:]/q[None,:].square()
            terms.append(-2*result['alpha'][i]*gx/self.null[i])
        structural = torch.stack(terms).mean(0)/self.p[:,None]
        node = -.04*torch.relu(self.q_floor-T.sum(0))/self.q_floor**2
        total = structural+node[None,:]
        self._finite(total, 'analytic complete gradient')
        return total

    def analytic_hvp(self, T, direction):
        """T-coordinate HVP; ReLU=0 branch at folds matches PyTorch convention.
        Not a claim of a unique classical Hessian at x=1 or node_q=floor.
        """
        if direction.shape != T.shape or direction.dtype != T.dtype or direction.device != T.device:
            raise ValueError('Direction must match transport')
        self._finite(direction, 'HVP direction')
        result = self.profile(T); W = T/self.p[:,None]; Z = direction/self.p[:,None]
        terms = []
        for i, a in enumerate(self.raw):
            q, mu = result['q'][i], self.source_measures[i]
            V, D = torch.sparse.mm(a, W), result['diagonal_edge_mass'][i]
            dq = mu @ Z; dD = 2*(Z*V).sum(0)
            gx = 2*V/q[None,:]-mu[:,None]*D[None,:]/q[None,:].square()
            dx = (gx*Z).sum()
            hx = 2*torch.sparse.mm(a,Z)/q[None,:]-2*V*dq[None,:]/q[None,:].square()
            hx = hx-mu[:,None]*dD[None,:]/q[None,:].square()+2*mu[:,None]*D[None,:]*dq[None,:]/q[None,:].pow(3)
            # Genuine x>K NNLS face uses d(alpha)/dx=1/K, not interior 1/(K-1).
            dalpha = torch.where(result['b_zero_face'][i], dx/self.K,
                torch.where(result['x'][i] > 1, dx/(self.K-1), torch.zeros_like(dx)))
            terms.append(-2*(dalpha*gx+result['alpha'][i]*hx)/self.null[i])
        structural = torch.stack(terms).mean(0)/self.p[:,None]
        active_mass = (T.sum(0) < self.q_floor).to(T.dtype)
        node = .04/self.q_floor**2*active_mass*direction.sum(0)
        total = structural+node[None,:]
        self._finite(total, 'analytic complete HVP')
        return total


