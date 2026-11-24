"""Generate sparse SBM inputs once, before any model or evaluation run."""
from study_common import *

def sbm(n, proportions, seed_value, L=4):
    import numpy as np
    from scipy import sparse
    rng = np.random.default_rng(seed_value)
    sizes = np.asarray(proportions)*n
    require(np.allclose(sizes, sizes.astype(int)), 'Integer community sizes')
    sizes = sizes.astype(int)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    permutation = rng.permutation(n)
    truth = truth[permutation]
    ids = [np.flatnonzero(truth == c) for c in range(len(sizes))]
    f = float(sum(int(x)*(int(x)-1) for x in sizes)/(n*(n-1)))
    pout = (12-19.12*f)/(n-1); pin = pout+19.12/(n-1)
    require(0 <= pout <= pin <= 1, 'SBM probabilities')
    layers = []
    for _ in range(L):
        rows, cols = [], []
        for c, a in enumerate(ids):
            for d in range(c, len(ids)):
                b = ids[d]; m, q = len(a), len(b)
                total = m*(m-1)//2 if c == d else m*q
                prob = pin if c == d else pout
                # Binomial edge count + a uniform subset is exactly Bernoulli sampling.
                # No N x N array or forced connectedness is introduced.
                count = int(rng.binomial(total, prob))
                flat = rng.choice(total, size=count, replace=False)
                if c == d:
                    starts = np.arange(m, dtype=np.int64)
                    starts = starts*(2*m-starts-1)//2
                    i = np.searchsorted(starts, flat, side='right')-1
                    j = i+1+flat-starts[i]
                else: i, j = flat//q, flat%q
                rows.append(a[i]); cols.append(b[j])
        u, v = np.concatenate(rows), np.concatenate(cols)
        matrix = sparse.csr_matrix((np.ones(2*len(u)), (np.r_[u,v],np.r_[v,u])), shape=(n,n))
        matrix.sort_indices(); layers.append(matrix)
    return layers, truth, dict(sizes=sizes.tolist(), f_N=f, p_in=pin, p_out=pout, expected_mean_degree=12)

def save_graph(folder, layers, truth, metadata):
    import numpy as np
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'graphs.npz', **{f'{key}_{i}':getattr(a,key) for i,a in enumerate(layers) for key in ('data','indices','indptr')})
    np.savez_compressed(folder/'evaluation.npz', truth=truth, mask=np.ones(len(truth),dtype=bool))
    np.savez_compressed(folder/'node_index.npz', original_ids=np.arange(len(truth)))
    save(folder/'metadata.json', metadata)

def prepare():
    target = ROOT/'data/catalog.json'
    if target.exists():
        for item in catalog(): require(fingerprints(dataset_folder(item)) == item['sha256'], 'Existing data changed')
        print('Inputs already prepared and unchanged.'); return
    items = []
    for ni,n in enumerate((1000,5000,10000)):
        for ci,(condition,proportions) in enumerate((('balanced',(.25,.25,.25,.25)),('imbalanced',(.05,.15,.30,.50)))):
            for g in range(5):
                name = f'sbm_{condition}_n{n}_g{g:02d}'
                folder = ROOT/'data/synthetic'/name
                graph_seed = 72000000+10000*ni+1000*ci+g
                if folder.exists():
                    require((folder/'prepared.json').exists(), 'Partial input exists; preserve it and inspect: '+str(folder))
                    receipt = read(folder/'prepared.json')
                    require(receipt['graph_seed'] == graph_seed and receipt['sha256'] == fingerprints(folder), 'Input receipt mismatch')
                else:
                    layers, truth, params = sbm(n,proportions,graph_seed)
                    m = dict(N=n,L=4,K=4,dataset=name,graph_seed=graph_seed,condition=condition,instance=g,**params)
                    save_graph(folder,layers,truth,m)
                    save(folder/'prepared.json', dict(graph_seed=graph_seed,sha256=fingerprints(folder)))
                graph(folder)
                items.append(dict(id=name,suite='sim',N=n,L=4,K=4,condition=condition,instance=g,index=len(items),path=str(folder.relative_to(ROOT)).replace('\\','/'),sha256=fingerprints(folder)))
                print('Prepared', name, flush=True)
    for name,n,L,k in [('amazon',7621,3,4),('arxiv_netscience',14489,13,8)]:
        folder = ROOT/'data/real'/name
        _,m = graph(folder); require((m['N'],m['L'],m['K']) == (n,L,k), 'Real dataset identity')
        items.append(dict(id=name,suite='amazon' if name == 'amazon' else 'arxiv',N=n,L=L,K=k,condition=None,instance=None,index=len(items),path=str(folder.relative_to(ROOT)).replace('\\','/'),sha256=fingerprints(folder)))
    save(target,dict(protocol='SBM30 plus Amazon plus arXiv; predeclared seeds',datasets=items))
    print('32 datasets ready; no training performed.')

if __name__ == '__main__': prepare()
