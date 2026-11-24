"""Run the frozen experiments. Use --check before training."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
import argparse
import concurrent.futures
import subprocess
import platform
import importlib.metadata
from reproduction_common import ROOT, BASELINES, read, save, sha, catalog, graph, seed, evaluate_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('real_world', 'synthetic', 'scalability'), default='real_world')
    p.add_argument('--dataset', nargs='+', help='Dataset IDs, or all')
    p.add_argument('--method', nargs='+', default=['CT-GWAE'], choices=('CT-GWAE',)+BASELINES)
    p.add_argument('--list', action='store_true')
    p.add_argument('--check', action='store_true', help='Verify files and load graphs; no training')
    p.add_argument('--diagnostic', action='store_true', help='One cold start per selected dataset/method')
    p.add_argument('--workers', type=int, default=1, help='Simultaneous single-threaded child processes')
    p.add_argument('--infomgf-python', default=os.environ.get('INFOMGF_PYTHON',sys.executable), help='Python3.11/3.12 environment with the frozen InfoMGF stack')
    p.add_argument('--output', type=Path, default=ROOT/'runs/reproduction')
    a = p.parse_args()
    items = [x for x in catalog() if x['suite']==a.suite]
    if a.list:
        for x in items:
            print(f"{x['id']:48} N={x['N']} L={x['L']} K={x['K']}")
        return
    if any(method != 'CT-GWAE' for method in a.method):
        p.error('Third-party baseline implementations are not bundled; see THIRD_PARTY.md. This runner executes CT-GWAE.')
    if not a.dataset:
        p.error('Choose --dataset <ID> or --dataset all; see --list')
    if a.dataset != ['all']:
        missing = set(a.dataset)-{x['id'] for x in items}
        if missing:
            p.error('Unknown dataset(s): '+', '.join(sorted(missing)))
        items = [x for x in items if x['id'] in a.dataset]
    if a.check:
        manifest = read(ROOT/'checksums.json')
        for name, digest in manifest.items():
            if sha(ROOT/name) != digest:
                raise ValueError('Package file changed: '+name)
        for item in items:
            graph(item)
        sys.path.insert(0, str(ROOT/'src/model'))
        from bootstrap import activate
        activate()
        print(f'CHECK PASSED: {len(items)} inputs; package hashes and model imports. No training.')
        return
    if a.workers < 1:
        p.error('--workers must be positive')
    if a.suite=='scalability' and set(a.method)-{'CT-GWAE', 'SCML', 'DGFM3'}:
        p.error('The scalability protocol contains CT-GWAE, SCML and DGFM3.')
    for item in items:
        allowed=item.get('allowed_methods',('CT-GWAE',)+BASELINES)
        if set(a.method)-set(allowed):
            p.error('Unsupported method for '+item['id']+': '+','.join(sorted(set(a.method)-set(allowed))))
    if 'InfoMGF-LA' in a.method:
        subprocess.run([a.infomgf_python,'-B',str(ROOT/'baselines/InfoMGF_LA/environment.py')],check=True)
    for name, digest in read(ROOT/'checksums.json').items():
        if name.startswith(('src/', 'baselines/', 'configs/')) and sha(ROOT/name) != digest:
            raise ValueError('Source or configuration changed: '+name)
    jobs, pools = [], []
    # Validate identities before creating output or starting a child.
    for item in items:
        for method in a.method:
            R = 1 if a.diagnostic else item['model_repeats' if method=='CT-GWAE' else 'baseline_repeats']
            J = 1 if a.diagnostic or method!='CT-GWAE' else item['model_starts']
            pools.append((item, method, R, J))
            for r in range(R):
                for j in range(J):
                    jobs.append(dict(dataset=item, method=method, r=r, j=j, seed=seed(item, method, r, j)))
    output = a.output.resolve()
    if output.exists():
        p.error('Output exists. Choose a new --output; existing results will not be overwritten.')
    output.mkdir(parents=True)
    environment = dict(python=sys.version, platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in
                  ('numpy','scipy','torch','scikit-learn','networkx','threadpoolctl')})
    save(output/'plan.json', dict(diagnostic=a.diagnostic, jobs=jobs, workers=a.workers,
                                 environment=environment))
    paths = []
    for job in jobs:
        folder = output/job['dataset']['id']/job['method']/f"r{job['r']:02d}_j{job['j']:02d}"
        folder.mkdir(parents=True)
        path = folder/'job.json'
        save(path, job)
        paths.append(path)
    def execute(path):
        with (path.parent/'output.txt').open('w', encoding='utf-8') as f:
            runtime=a.infomgf_python if read(path)['method']=='InfoMGF-LA' else sys.executable
            code = subprocess.run([runtime, '-B', str(ROOT/'src/worker.py'), str(path)],
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        print(('DONE ' if code==0 else 'FAILED ')+str(path.parent.relative_to(output)), flush=True)
        return code
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = [executor.submit(execute, path) for path in paths]
        codes = []
        for future in concurrent.futures.as_completed(futures):
            if future.cancelled():
                continue
            code = future.result()
            codes.append(code)
            if code:
                for pending in futures:
                    pending.cancel()
    if any(codes):
        save(output/'failure.json', dict(failed_jobs=sum(c!=0 for c in codes),
             finished_jobs=len(codes), planned_jobs=len(paths)))
        raise SystemExit('Some jobs failed. Outputs have been preserved; inspect their failure.json files.')
    # Every start is complete before the first evaluation label is opened.
    for item, method, R, J in pools:
        folder = output/item['id']/method
        winners = []
        for r in range(R):
            rows = [read(folder/f'r{r:02d}_j{j:02d}/success.json') for j in range(J)]
            j = min(range(J), key=lambda j: (rows[j]['objective'], j)) if method=='CT-GWAE' else 0
            chosen = rows[j]
            path = folder/f'r{r:02d}_j{j:02d}/labels.npy'
            if sha(path) != chosen['sha256']:
                raise ValueError('Prediction changed before locking')
            winners.append(dict(r=r, j=j, seed=chosen['seed'], labels=path.relative_to(folder).as_posix(),
                                sha256=chosen['sha256'], objective=chosen.get('objective')))
        save(folder/'prediction_lock.json', dict(dataset=item, method=method,
             repeats=R, diagnostic=a.diagnostic, truth_used_for_selection=False, winners=winners))
    for item, method, R, J in pools:
        result = evaluate_lock(output/item['id']/method)
        print(item['id'], method, result['summary'], flush=True)
    save(output/'success.json', dict(status='passed', diagnostic=a.diagnostic, jobs=len(jobs)))


if __name__=='__main__':
    main()
