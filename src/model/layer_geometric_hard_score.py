"""STATIC fixed-partition diagnostic, not a trained or frozen model.
Per-layer normalized sqrt-degree geometric measure, independent inverse-q template, same GW
family and original uniform-node .02 occupancy penalty. No truth or encoder.
"""
import numpy as np
import scipy.sparse as sp


class LayerGeometricFixedPartition:
    """Prepare immutable sparse layer statistics once; labels remain unchanged.
    Inputs: aligned symmetric nonnegative zero-diagonal adjacency layers, K>=2.
    Zero-volume layers are explicitly reported and omitted from structure mean.
    Zero-degree nodes remain in the full node occupancy penalty.
    """
    def __init__(self,layers,K):
        if not isinstance(K,(int,np.integer)) or isinstance(K,(bool,np.bool_)) or K<2 or not layers:
            raise ValueError('Nonempty aligned layers and integer K>=2 required')
        self.K=int(K);self.n=layers[0].shape[0]
        if self.n<1:
            raise ValueError('Nonempty node set required')
        self.layers=[];self.zero_volume_layers=[];self.inactive_variance_layers=[]
        self.q_floor=1/(4*self.K)
        for layer_id,layer in enumerate(layers):
            a=sp.csr_matrix(layer,dtype=np.float64).copy();a.sum_duplicates();a.eliminate_zeros()
            if a.shape!=(self.n,self.n):
                raise ValueError('Aligned square layer required')
            if not np.isfinite(a.data).all() or (a.data<0).any():
                raise ValueError('Finite nonnegative edges required')
            if np.any(a.diagonal()!=0) or (a-a.T).nnz:
                raise ValueError('Symmetric zero-diagonal relation required')
            degree=np.asarray(a.sum(1)).ravel();volume=float(degree.sum())
            if not np.isfinite(degree).all() or not np.isfinite(volume):
                raise FloatingPointError('Finite degrees and volume required')
            if volume==0:
                self.zero_volume_layers.append(layer_id);continue
            root_degree=np.sqrt(degree);measure_scale=float(root_degree.sum())
            mu=root_degree/measure_scale
            if not np.isfinite(mu).all() or (mu<0).any() or abs(float(mu.sum())-1)>1e-12:
                raise FloatingPointError('Finite normalized layer geometric measure required')
            raw=(a/volume).tocsr();coo=raw.tocoo();product=mu[coo.row]*mu[coo.col]
            if not np.isfinite(product).all() or not (product>0).all():
                raise FloatingPointError('Positive finite measure at all edge endpoints required')
            relation_values=coo.data/product
            nu=float(np.sum(coo.data*relation_values))
            if not np.isfinite(relation_values).all() or not np.isfinite(nu):
                raise FloatingPointError('Finite relation and second moment required')
            if nu-1<=1e-10:
                self.inactive_variance_layers.append(layer_id);continue
            # CSR / NumPy buffers are not exposed for update by this interface.
            self.layers.append(dict(layer_id=layer_id,raw=raw,mu=mu,nu=nu,null=nu-1,
                volume=volume,zero_source_nodes=np.flatnonzero(mu==0).tolist()))
        if not self.layers:
            raise ValueError('At least one positive-variance layer required')

    def score(self,labels):
        labels=np.asarray(labels)
        if labels.shape!=(self.n,) or labels.dtype.kind not in 'iu':
            raise ValueError('One integer label per original node required')
        if not ((labels>=0)&(labels<self.K)).all():
            raise ValueError('Labels must use the original registered K domain')
        original=labels.copy();counts=np.bincount(labels.astype(np.int64),minlength=self.K)
        node_q=counts/self.n
        node_mass=float(.02*np.sum((np.maximum(self.q_floor-node_q,0)/self.q_floor)**2))
        profiles=[]
        for layer in self.layers:
            mu=layer['mu'];q=np.bincount(labels.astype(np.int64),weights=mu,minlength=self.K)
            active=np.flatnonzero(q>0);m=int(active.size)
            if not np.isfinite(q).all() or (q<0).any() or abs(float(q.sum())-1)>1e-12 or m<1:
                raise FloatingPointError('Finite nonnegative normalized layer q required')
            coo=layer['raw'].tocoo();same=labels[coo.row]==labels[coo.col]
            diagonal=np.bincount(labels[coo.row[same]].astype(np.int64),weights=coo.data[same],minlength=self.K)
            if not np.isfinite(diagonal).all() or (diagonal<0).any():
                raise FloatingPointError('Finite nonnegative same-class edge masses required')
            if np.any(diagonal[q==0]!=0):
                raise ValueError('Zero-source class cannot contain positive edge mass')
            x=float(np.sum(diagonal[active]/q[active]))
            if not np.isfinite(x) or x<0:
                raise FloatingPointError('Finite nonnegative contraction required')
            if m==1:
                if abs(x-1)>1e-12 or abs(float(q[active[0]])-1)>1e-12:
                    raise ValueError('Single positive layer-support class requires q=x=1')
                b,alpha,gain=1.,0.,1.;branch='single_support_canonical'
            elif x<=1:
                b,alpha,gain=1.,0.,1.;branch='nonseparated'
            elif x<=m:
                alpha=(x-1)/(m-1);b=1-alpha
                gain=2*b+2*alpha*x-b*b-2*b*alpha-m*alpha*alpha
                branch='independent_layer_interior'
            else:
                # Genuine NNLS b=0 face: geometric measure need not bound x<=m.
                b=0.;alpha=x/m;gain=2*alpha*x-m*alpha*alpha
                branch='independent_layer_b_zero_face'
            values=np.array([b,alpha,gain],dtype=np.float64)
            if not np.isfinite(values).all() or b<0 or alpha<0:
                raise FloatingPointError('Finite nonnegative independent coefficients required')
            structure=(layer['nu']-gain)/layer['null']
            # On q>0 support B=b11+alpha diag(1/q). Full K storage need not
            # invent inverse masses for unsupported labels. Outside support
            # entries are canonically b, which carries no structural weight.
            template_diagonal=np.full(self.K,b)
            template_diagonal[active]+=alpha/q[active]
            if not np.isfinite(structure) or not np.isfinite(template_diagonal).all():
                raise FloatingPointError('Finite layer objective/template required')
            profiles.append(dict(layer_id=layer['layer_id'],mu=mu.tolist(),nu=layer['nu'],
                null=layer['null'],q=q.tolist(),positive_support_labels=active.tolist(),m=m,
                diagonal_edge_mass=diagonal.tolist(),x=x,b=b,alpha=alpha,gain=gain,
                structure=float(structure),branch=branch,coefficients_unique=m>1,
                zero_source_nodes=layer['zero_source_nodes'],unsupported_community_labels=np.flatnonzero(q==0).tolist(),
                nnls_b_zero_threshold=m,b_zero_threshold_excess=x-m,
                template_off_diagonal=b,template_diagonal=template_diagonal.tolist(),
                unsupported_template_convention='inverse term zero outside positive q support; no fictitious inverse mass'))
        structure=float(np.mean([p['structure'] for p in profiles]));objective=structure+node_mass
        if not np.isfinite(objective) or not np.array_equal(labels,original):
            raise FloatingPointError('Finite objective and unchanged labels required')
        return dict(objective=objective,structure=structure,node_mass=node_mass,node_q=node_q.tolist(),
            node_q_floor=self.q_floor,label_counts=counts.tolist(),K=self.K,n=self.n,
            active_layers=[p['layer_id'] for p in profiles],zero_volume_layers=list(self.zero_volume_layers),
            inactive_variance_layers=list(self.inactive_variance_layers),profiles=profiles,
            labels_changed=False,score_domain='hard_fixed_partition_per_layer_geometric_mu_support_gram_m',
            training_proposals=0,encoder_calls=0,unroll_calls=0,KMeans_calls=0,truth_read=False)

