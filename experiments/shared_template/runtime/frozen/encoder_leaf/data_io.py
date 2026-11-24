"""Sparse input and profiled GW objective for the frozen V8 protocol.

The source package and its data files are read without modification. Graph
matrices remain SciPy CSR on load and PyTorch sparse COO during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


DT = torch.float64
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _torch_sparse(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.stack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data.copy(), dtype=DT)
    return torch.sparse_coo_tensor(indices, values, coo.shape, dtype=DT,
                                   check_invariants=True).coalesce()


def load_case(name: str):
    folder = SOURCE_ROOT / "数据" / name
    with np.load(folder / "graphs.npz") as z:
        L = len([key for key in z.files if key.startswith("data_")])
        layers = []
        for ell in range(L):
            indptr = z[f"indptr_{ell}"]
            n = len(indptr) - 1
            A = sp.csr_matrix((z[f"data_{ell}"], z[f"indices_{ell}"], indptr),
                              shape=(n, n), dtype=np.float64)
            # This is the sparse equivalent of np.maximum(A, A.T) in V8.
            layers.append(A.maximum(A.T).tocsr())
    with np.load(folder / "evaluation.npz") as evaluation:
        truth = evaluation["truth"].copy()
        mask = evaluation["mask"].astype(bool).copy()
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
    return layers, truth, mask, int(meta["K"])


def relation(A: sp.csr_matrix) -> sp.csr_matrix:
    degrees = np.asarray(A.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    C = (sp.diags(inv) @ A @ sp.diags(inv)).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return C


def propagation_operator(A: sp.csr_matrix) -> torch.Tensor:
    X = A + sp.eye(A.shape[0], format="csr", dtype=np.float64)
    degrees = np.asarray(X.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    return _torch_sparse(sp.diags(inv) @ X @ sp.diags(inv))


class ProfiledGW:
    def __init__(self, layers: list[sp.csr_matrix], K: int):
        if K < 2:
            raise ValueError("K must be at least 2 so that 0 < h < 1.")
        self.n, self.K = layers[0].shape[0], K
        self.p = torch.full((self.n,), 1.0 / self.n, dtype=DT)
        C = [relation(A) for A in layers]
        mu = np.array([float(matrix.sum()) / self.n**2 for matrix in C])
        nu = np.array([float(np.dot(matrix.data, matrix.data)) / self.n**2 for matrix in C])
        null = nu - mu * mu
        active = null > 1e-10
        if not active.any():
            raise ValueError("No active layer has positive null structural variance.")
        self.C = [_torch_sparse(C[i]) for i in np.flatnonzero(active)]
        self.mu = torch.tensor(mu[active], dtype=DT)
        self.null = torch.tensor(null[active], dtype=DT)
        self.active_layers = np.flatnonzero(active).tolist()
        # Fixed method definition: smoothing is part of T^(0), not the mirror step.
        self.delta = 0.01
        self.q_floor = 1.0 / (4 * K)

    def _x(self, T: torch.Tensor) -> torch.Tensor:
        return torch.stack([(T * torch.sparse.mm(C, T)).sum() for C in self.C])

    def loss(self, T: torch.Tensor) -> torch.Tensor:
        q = T.sum(0)
        h = (q * q).sum()
        x = self._x(T)
        E = (self.null - torch.relu(x - h * self.mu).square() / (h * (1.0 - h))) / self.null
        mass = 0.02 * (torch.relu(self.q_floor - q) / self.q_floor).square().sum()
        return E.mean() + mass

    def decode_templates(self, T: torch.Tensor) -> dict:
        q = T.sum(0)
        h = (q * q).sum()
        x = self._x(T)
        separated = x >= h * self.mu
        theta_in = torch.where(separated, x / h, self.mu)
        theta_out = torch.where(separated, (self.mu - x) / (1.0 - h), self.mu)
        return {
            "active_layers": [int(i) for i in self.active_layers],
            "theta_in": [float(v.detach()) for v in theta_in],
            "theta_out": [float(v.detach()) for v in theta_out],
            "separated": [bool(v.detach()) for v in separated],
        }
