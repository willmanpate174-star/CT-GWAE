"""Paths, immutable receipts and input validation for the scalability study."""
from pathlib import Path
import hashlib, json, os, sys, platform

ROOT = Path(__file__).resolve().parent
METHODS = ('CT-GWAE', 'SCML', 'DGFM3')
FILES = ('graphs.npz', 'metadata.json', 'node_index.npz', 'evaluation.npz')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save(p, obj, replace=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    if replace:
        temp = p.with_name(p.name + f'.{os.getpid()}.tmp')
        with temp.open('x', encoding='utf-8') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, p)
    else:
        with p.open('x', encoding='utf-8') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())

def require(ok, message):
    if not ok: raise ValueError(message)

def code_hashes():
    manifest = read(ROOT/'provenance/package_sha256.json')
    for rel, expected in manifest.items():
        require(sha(ROOT/rel) == expected, 'Source changed: ' + rel)
    return manifest

def graph(folder):
    import numpy as np
    from scipy import sparse
    m = read(folder/'metadata.json'); n, L, k = (m[x] for x in ('N','L','K'))
    require(all(type(x) is int for x in (n,L,k)) and n >= k >= 2 and L > 0, 'Invalid N/L/K')
    with np.load(folder/'graphs.npz', allow_pickle=False) as z:
        require(set(z.files) == {f'{x}_{l}' for x in ('data','indices','indptr') for l in range(L)}, 'CSR keys')
        layers = [sparse.csr_matrix((z[f'data_{l}'], z[f'indices_{l}'], z[f'indptr_{l}']), shape=(n,n)) for l in range(L)]
    for a in layers:
        a.check_format(full_check=True)
        require(a.has_canonical_format and np.isfinite(a.data).all() and (a.data == 1).all(), 'Canonical binary CSR')
        require(not (a-a.T).nnz and not np.any(a.diagonal()), 'Symmetric zero-diagonal graphs')
    return layers, m

def fingerprints(folder):
    return {name: sha(folder/name) for name in FILES}

def catalog():
    return read(ROOT/'data/catalog.json')['datasets']

def dataset_folder(item):
    p = (ROOT/item['path']).resolve()
    require(p.is_relative_to(ROOT), 'Dataset path escapes runner')
    return p

def selected(suite):
    return [x for x in catalog() if suite == 'all' or x['suite'] == suite]

def seed(item, method, r, j=0):
    # Graphs and method seeds have disjoint, fixed ranges; never use Python hash().
    return {'CT-GWAE':81000000,'SCML':85000000,'DGFM3':83000000}[method] + 10000*item['index'] + 100*r + j

def environment():
    import importlib.metadata
    names = ('numpy','scipy','torch','scikit-learn','threadpoolctl','matplotlib','networkx')
    return dict(python=sys.version, platform=platform.platform(), packages={n:importlib.metadata.version(n) for n in names})

def valid_success(slot):
    p = slot/'success.json'
    if not p.exists(): return None
    rec = read(p); attempt = slot/rec['attempt']
    require(attempt.resolve().is_relative_to(slot.resolve()), 'Unsafe attempt')
    require(read(attempt/'success.json') == rec['record'], 'Success readback')
    for rel, digest in rec['record']['artifacts'].items():
        require(sha(attempt/rel) == digest, 'Completed artifact changed: ' + str(attempt/rel))
    return rec

def evaluate(labels, folder, L, n):
    """Original baseline evaluator, including its multilayer tiling convention."""
    import importlib.util
    sys.path.insert(0, str(ROOT/'vendor/baseline/runtime'))
    spec = importlib.util.spec_from_file_location('frozen_evaluation', ROOT/'vendor/baseline/runtime/run_one.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.evaluate(labels, folder, L, n)
