"""Recompute ARI, arithmetic NMI and Hungarian ACC from fixed predictions."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
import argparse
from reproduction_common import ROOT, read, sha, catalog, scores_for_method, evaluate_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, help='A run folder or one dataset/method folder')
    p.add_argument('--reference', action='store_true', help='Check the included reference predictions')
    a = p.parse_args()
    if a.reference:
        import numpy as np
        datasets = {x['id']:x for x in catalog()}
        rows = read(ROOT/'results/reference_predictions.json')
        maximum = 0.0
        checked = 0
        unavailable = set()
        for row in rows:
            item = datasets[row['dataset']]
            if item['has_truth'] and not (ROOT/item['path']/'evaluation.npz').is_file():
                unavailable.add(item['id'])
                continue
            checked += 1
            path = ROOT/row['labels']
            if sha(path) != row['sha256']:
                raise ValueError('Reference prediction changed: '+str(path))
            result = scores_for_method(np.load(path, allow_pickle=False), datasets[row['dataset']], row['method'])
            for key, expected in row['expected'].items():
                if expected is None:
                    if result[key] is not None:
                        raise ValueError('Unexpected ground-truth score')
                else:
                    error = abs(result[key]-expected)
                    maximum = max(maximum, error)
                    if error > 1e-10:
                        raise ValueError(f'Reference mismatch: {row["dataset"]}/{row["method"]}/r{row["r"]} {key}: {result[key]} != {expected}')
        print(f'REFERENCE CHECK PASSED: {checked} fixed predictions; maximum metric difference {maximum:.3g}.')
        if unavailable:
            print('Separate evaluation inputs needed for: '+', '.join(sorted(unavailable)))
        return
    if a.output is None:
        p.error('Choose --output or --reference')
    locks = sorted(a.output.rglob('prediction_lock.json'))
    if not locks:
        p.error('No completed prediction lock was found')
    for path in locks:
        result = evaluate_lock(path.parent)
        print(result['dataset'], result['method'], result['summary'])


if __name__=='__main__':
    main()
