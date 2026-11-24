"""Only the layer measure and its dependent statistics differ from frozen Full.
All profile, NNLS, gradient, HVP, training and MAP code is inherited unchanged.
"""
from common import ROOT
import sys
sys.path.insert(0,str(ROOT/'frozen'))
import numpy as np
import scipy.sparse as sp
import torch
from layer_geometric_single_gw import LayerGeometricSingleGW,_sparse
from layer_geometric_hard_score import LayerGeometricFixedPartition

class MeasureHard(LayerGeometricFixedPartition):
    def __init__(self,layers,K,measure):
        if measure not in ('uniform','sqrt','degree'):raise ValueError(measure)
        # Preserve the original input validation and K/N domain exactly.
        super().__init__(layers,K)
        self.measure=measure
        if measure=='sqrt':return
        self.layers=[];self.zero_volume_layers=[];self.inactive_variance_layers=[]
        for i,layer in enumerate(layers):
            a=sp.csr_matrix(layer,dtype=np.float64).copy();a.sum_duplicates();a.eliminate_zeros()
            d=np.asarray(a.sum(1)).ravel();volume=float(d.sum())
            if volume==0:self.zero_volume_layers.append(i);continue
            mu=np.full(self.n,1/self.n,dtype=np.float64) if measure=='uniform' else d/volume
            raw=(a/volume).tocsr();c=raw.tocoo();den=mu[c.row]*mu[c.col]
            if not np.isfinite(mu).all() or abs(mu.sum()-1)>1e-12 or not (den>0).all():raise ValueError('Invalid measure')
            rel=c.data/den;nu=float(np.sum(c.data*rel))
            if not np.isfinite(nu):raise FloatingPointError('Nonfinite nu')
            if nu-1<=1e-10:self.inactive_variance_layers.append(i);continue
            self.layers.append(dict(layer_id=i,raw=raw,mu=mu,nu=nu,null=nu-1,
                volume=volume,zero_source_nodes=np.flatnonzero(mu==0).tolist()))
        if not self.layers:raise ValueError('At least one positive-variance layer required')
    def score(self,labels):
        r=super().score(labels)
        r['source_measure']=self.measure
        r['score_domain']='hard_fixed_partition_per_layer_'+self.measure+'_mu_support_gram_m'
        return r

class MeasureObjective(LayerGeometricSingleGW):
    def __init__(self,layers,K,measure):
        super().__init__(layers,K)
        s=MeasureHard(layers,K,measure)
        self.measure=measure
        self.active_layers=[r['layer_id'] for r in s.layers]
        self.zero_volume_layers=list(s.zero_volume_layers)
        self.inactive_variance_layers=list(s.inactive_variance_layers)
        self.zero_source_nodes_by_layer=[r['zero_source_nodes'] for r in s.layers]
        self.source_measures=torch.tensor(np.stack([r['mu'] for r in s.layers]),dtype=torch.float64)
        self.raw=[_sparse(r['raw']) for r in s.layers]
        self.nu=torch.tensor([r['nu'] for r in s.layers],dtype=torch.float64)
        self.null=self.nu-1
        self.hard_diagnostic=s
