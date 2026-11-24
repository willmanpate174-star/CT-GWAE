"""Relative paths, input loading and the experiment's evaluation conventions."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ('RDSoS', 'SCML', 'WPTTD', 'DGFM3', 'jNMF-GCL', 'DMGI', 'MultiTensor', 'InfoMGF-LA')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02*(attempt+1))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def catalog():
    return read(ROOT/'configs/catalog.json')['datasets']


def graph(item):
    import numpy as np
    from scipy import sparse
    folder = ROOT/item['path']
    for name, digest in item['sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('Input changed: ' + str(folder/name))
    n, count = item['N'], item['L']
    with np.load(folder/'graphs.npz', allow_pickle=False) as z:
        layers = [sparse.csr_matrix((z[f'data_{i}'], z[f'indices_{i}'],
                  z[f'indptr_{i}']), shape=(n, n)) for i in range(count)]
    for a in layers:
        a.check_format(full_check=True)
        if not (a.has_canonical_format and np.isfinite(a.data).all()
                and (a.data >= 0).all() and not (a-a.T).nnz
                and not np.any(a.diagonal())):
            raise ValueError('Expected nonnegative, symmetric, canonical CSR graphs')
    return layers, dict(N=n, L=count, K=item['K'])


def seed(item, method, r, j=0):
    if method == 'CT-GWAE':
        return item['model_seed_base'] + 100*r + j
    if method == 'InfoMGF-LA':
        return 931100000 + 1000*item['infomgf_ordinal'] + r
    if item['id'] == 'mnist70000':
        return item['baseline_seed_config'][method+'_seeds'][r]
    if item['suite'] == 'scalability':
        return {'SCML':85000000,'DGFM3':83000000}[method] + 10000*item['index'] + 100*r
    if item['suite'] == 'synthetic':
        return 10000 + 100*r
    return 10000 + 100000*(BASELINES.index(method)+1) + 1000*item['index'] + 100*r


def scores(labels, item):
    """Called only after predictions have been fixed and read back."""
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment
    if not item['has_truth']:
        return dict(ari=None, nmi=None, acc=None)
    with np.load(ROOT/item['path']/'evaluation.npz', allow_pickle=False) as z:
        truth, mask = z['truth'], z['mask'].astype(bool)
    y = np.asarray(labels)
    n, L = item['N'], item['L']
    if truth.shape != (n,) or mask.shape != (n,) or not mask.any():
        raise ValueError('Invalid ground truth or mask')
    if y.dtype.kind not in 'iu' or y.shape not in ((n,), (L, n)):
        raise ValueError('Invalid prediction shape or dtype')
    if item['evaluation'] == 'node':
        if y.shape != (n,):
            raise ValueError('This protocol requires one common node partition')
        t, y = truth[mask], y[mask]
    else:
        y = np.broadcast_to(y, (L, n)) if y.ndim == 1 else y
        t, y = np.tile(truth[mask], L), y[:, mask].ravel()
    _, tindex = np.unique(t, return_inverse=True)
    _, yindex = np.unique(y, return_inverse=True)
    contingency = np.zeros((tindex.max()+1, yindex.max()+1), dtype=np.int64)
    np.add.at(contingency, (tindex, yindex), 1)
    row, col = linear_sum_assignment(-contingency)
    return dict(ari=float(adjusted_rand_score(t, y)),
                nmi=float(normalized_mutual_info_score(t, y, average_method='arithmetic')),
                acc=float(contingency[row, col].sum()/len(y)))


def scores_for_method(labels, item, method):
    import numpy as np
    if method != 'InfoMGF-LA':
        return scores(labels,item)
    y=np.asarray(labels)
    if y.shape!=(10,item['N']) or y.dtype.kind not in 'iu':
        raise ValueError('InfoMGF-LA requires all ten fixed KMeans predictions')
    values=[scores(p,item) for p in y]
    return {m:float(np.mean([r[m] for r in values])) for m in ('ari','nmi','acc')}


def evaluate_lock(folder):
    import numpy as np
    folder = Path(folder)
    lock = read(folder/'prediction_lock.json')
    item = lock['dataset']
    if len(lock['winners']) != lock['repeats']:
        raise ValueError('Incomplete prediction lock')
    rows = []
    for winner in lock['winners']:
        path = (folder/winner['labels']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != winner['sha256']:
            raise ValueError('Locked prediction changed')
        rows.append(dict(r=winner['r'], seed=winner['seed'],
                         **scores_for_method(np.load(path, allow_pickle=False), item, lock['method'])))
    summary = {}
    for metric in ('ari', 'nmi', 'acc'):
        vals = [x[metric] for x in rows if x[metric] is not None]
        summary[metric] = dict(n=len(vals), mean=float(np.mean(vals)) if vals else None,
                              sample_sd=float(np.std(vals, ddof=1)) if len(vals)>1 else None)
    result = dict(dataset=item['id'], method=lock['method'],
                  diagnostic=lock['diagnostic'], metrics=rows, summary=summary)
    save(folder/'metrics.json', result)
    return result
