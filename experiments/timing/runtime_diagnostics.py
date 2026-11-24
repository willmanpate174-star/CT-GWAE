"""Read saved timing/resource observations; no new computation or truth access."""
import csv
import json
import math
from pathlib import Path


def diagnostics(folder, timing):
    folder = Path(folder)
    with (folder / 'proposal_runtime.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 600 or [int(r['proposal_id']) for r in rows] != list(range(600)):
        raise RuntimeError('Incomplete per-proposal execution observations.')
    cpu = math.fsum(float(r['cpu_seconds']) for r in rows)
    wall = math.fsum(float(r['wall_seconds']) for r in rows)
    if not math.isclose(cpu, timing['training_cpu_seconds'], rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError('Diagnostic/protected CPU interval sums differ.')
    if not math.isclose(wall, timing['training_wall_sum_seconds'], rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError('Diagnostic/protected wall interval sums differ.')
    result = {'proposal_cpu_to_wall_ratio': cpu / wall if wall > 0 else None}
    for key in ('real_forward_calls', 'real_gradient_calls', 'real_primal_calls', 'cache_hits'):
        values = [int(r[key]) for r in rows]
        if any(v < 0 for v in values):
            raise RuntimeError('Negative execution count.')
        result['protected_' + key] = sum(values)
    if sum(int(r['S24_calls_total']) for r in rows) != timing['S24_calls']:
        raise RuntimeError('Requested S24 counts differ.')
    samples = [json.loads(line) for line in (folder / 'resource_samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    frequencies = [int(s['cpu_frequency_kHz']) for s in samples if s.get('cpu_frequency_kHz')]
    result.update({
        'resource_samples': len(samples),
        'max_sampled_peak_rss_kB': max((s['ru_maxrss_kB'] for s in samples if s.get('ru_maxrss_kB') is not None), default=None),
        'last_sampled_major_faults': samples[-1].get('ru_majflt') if samples else None,
        'last_sampled_swaps': samples[-1].get('ru_nswap') if samples else None,
        'frequency_samples_available': len(frequencies),
        'min_sampled_frequency_kHz': min(frequencies) if frequencies else None,
        'max_sampled_frequency_kHz': max(frequencies) if frequencies else None,
    })
    return result
