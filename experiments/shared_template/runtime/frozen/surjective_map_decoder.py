"""STATIC, unregistered: exact known-K surjective MAP projection.
Only saved positive row-stochastic assignment probabilities are inputs.
No graph, truth labels, GW search, KMeans, clipping or smoothing.
"""
import numpy as np

ROW_SUM_ATOL=1e-12

def decode(W):
    """Return (integer labels, discrete projection receipt). Never mutate W.

    Maximize sum_i log(W[i,z_i]) subject to all K labels represented.
    Existing full-K argmax returns the same label bytes with zero assignment calls.
    """
    values=np.asarray(W)
    if values.ndim!=2 or values.dtype.kind!='f':
        raise ValueError('Floating N by K probability matrix required')
    N,K=values.shape
    if K<1 or N<K:
        raise ValueError('Known K>=1 and N>=K required for a surjection')
    if not np.isfinite(values).all() or not (values>0).all():
        raise ValueError('Strictly positive finite probabilities; no zero smoothing')
    if np.max(np.abs(values.sum(axis=1)-1))>ROW_SUM_ATOL:
        raise ValueError('Row-stochastic probabilities required; no normalization')
    original=np.argmax(values,axis=1)
    before=np.bincount(original,minlength=K)
    if (before>0).all():
        return original,dict(status='already_surjective_argmax_unchanged',N=N,K=K,
            input_row_sum_atol=ROW_SUM_ATOL,assignment_calls=0,changed_nodes=0,
            counts_before=before.tolist(),counts_after=before.tolist(),
            representative_rows=[],representative_classes=[],total_log_cost=0.,
            tie_rule='numpy first-index row argmax; no Hungarian call',
            truth_read=False,graph_read=False)
    logs=np.log(values)
    maxima=logs.max(axis=1)
    costs=(maxima[:,None]-logs).T
    if not np.isfinite(costs).all():
        raise FloatingPointError('Nonfinite exact representative cost')
    from scipy.optimize import linear_sum_assignment
    classes,representatives=linear_sum_assignment(costs)
    if not np.array_equal(classes,np.arange(K)) or len(set(representatives.tolist()))!=K:
        raise RuntimeError('Assignment must cover every class with distinct rows')
    labels=original.copy()
    labels[representatives]=classes
    counts=np.bincount(labels,minlength=K)
    if not (counts>0).all():
        raise RuntimeError('Exact surjective construction failed')
    chosen_cost=float(costs[classes,representatives].sum())
    realised_cost=float((maxima-logs[np.arange(N),labels]).sum())
    if not np.isfinite(chosen_cost) or not np.isfinite(realised_cost):
        raise FloatingPointError('Nonfinite projection objective')
    return labels,dict(status='exact_known_K_surjective_MAP_projection',N=N,K=K,
        input_row_sum_atol=ROW_SUM_ATOL,assignment_calls=1,
        changed_nodes=int(np.count_nonzero(labels!=original)),
        counts_before=before.tolist(),counts_after=counts.tolist(),
        representative_rows=representatives.tolist(),representative_classes=classes.tolist(),
        total_log_cost=realised_cost,assignment_log_cost=chosen_cost,
        projection_log_likelihood=float(logs[np.arange(N),labels].sum()),
        tie_rule='numpy first-index row argmax; Hungarian ties use actual pinned SciPy implementation; no perturbation',
        truth_read=False,graph_read=False)
