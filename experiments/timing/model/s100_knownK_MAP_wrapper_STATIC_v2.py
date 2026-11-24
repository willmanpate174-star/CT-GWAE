"""STATIC shared S100 endpoint decoder. No graphs/truth/training or tunable parameters."""
def decode_and_score(T,raw_labels,K,scorer,decoder,np):
    N=T.shape[0] if getattr(T,'ndim',None)==2 else 0
    if T.ndim!=2 or T.shape!=(N,K) or T.dtype.kind!='f' or N<K or K<1:
        raise ValueError('Saved floating N by K T and N>=K>=1 required')
    if not np.isfinite(T).all() or not (T>0).all():raise ValueError('Finite strictly positive saved T required')
    if raw_labels.shape!=(N,) or raw_labels.dtype.kind not in 'iu':raise ValueError('Saved integer raw labels required')
    original=np.argmax(T,axis=1)
    if not np.array_equal(raw_labels,original):raise ValueError('Saved raw endpoint must be exact T argmax')
    W=N*T
    if not np.isfinite(W).all() or not (W>0).all() or np.max(np.abs(W.sum(axis=1)-1))>decoder.ROW_SUM_ATOL:
        raise ValueError('Exact W=N*T domain invalid; no normalization/clip/epsilon')
    before=np.bincount(original,minlength=K)
    # The original endpoint's full-K bytes are authoritative; multiplying by N
    # can merge one-ulp ties in floating arithmetic, so do not re-label this branch.
    if (before>0).all():
        labels=raw_labels.copy()
        receipt=dict(status='already_surjective_raw_argmax_unchanged',N=N,K=K,assignment_calls=0,
            changed_nodes=0,counts_before=before.tolist(),counts_after=before.tolist(),
            total_log_cost=0.,input_row_sum_atol=decoder.ROW_SUM_ATOL,truth_read=False,
            tie_rule='original saved S100 T argmax first index; no assignment')
    else:
        labels,receipt=decoder.decode(W)
        receipt['original_T_argmax_changed_nodes']=int(np.count_nonzero(labels!=original))
    receipt['raw_T_counts_before']=before.tolist()
    receipt['raw_T_full_K']=bool((before>0).all())
    if not (np.bincount(labels,minlength=K)>0).all():raise RuntimeError('Known K surjection failed')
    raw_score=scorer.score(original);score=scorer.score(labels)
    if raw_score['labels_changed'] is not False or score['labels_changed'] is not False:
        raise RuntimeError('Hard scorer cannot modify labels')
    return labels,receipt,raw_score,score
