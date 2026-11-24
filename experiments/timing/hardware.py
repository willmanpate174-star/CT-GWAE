"""Hardware controls and observations; no model or label access."""
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:
    resource = None  # Windows is used only for local, non-formal fixtures.
import socket
import time

from common import read, require, save, digest, ROOT


def maybe_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def integers(text):
    result = []
    for part in (text or '').split(','):
        if not part:
            continue
        ends = [int(x) for x in part.split('-')]
        result.extend(range(ends[0], ends[-1] + 1))
    return sorted(set(result))


def topology(allowed):
    result = []
    online = integers(maybe_text('/sys/devices/system/cpu/online'))
    for cpu in online or sorted(allowed):
        root = Path('/sys/devices/system/cpu/cpu%d' % cpu)
        nodes = list(root.glob('node[0-9]*'))
        result.append({
            'cpu': cpu,
            'socket': maybe_text(root / 'topology/physical_package_id'),
            'core': maybe_text(root / 'topology/core_id'),
            'numa': int(nodes[0].name[4:]) if nodes else None,
            'thread_siblings': integers(maybe_text(root / 'topology/thread_siblings_list')),
            'allocated_affinity': cpu in allowed,
        })
    return result


def cpu_models():
    text = maybe_text('/proc/cpuinfo') or ''
    return sorted(set(line.partition(':')[2].strip() for line in text.splitlines()
                      if line.startswith('model name')))


def snapshot(cpu=None):
    memory = {}
    for line in (maybe_text('/proc/meminfo') or '').splitlines():
        key, _, val = line.partition(':')
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key + '_kB'] = int(val.split()[0])
    status = {}
    for line in (maybe_text('/proc/self/status') or '').splitlines():
        key, _, val = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'VmSwap', 'Threads',
                   'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'):
            status[key] = val.strip()
    use = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    freq_root = Path('/sys/devices/system/cpu/cpu%d/cpufreq' % cpu) if cpu is not None else None
    return {
        'wall_timestamp_ns': time.time_ns(), 'monotonic_seconds': time.monotonic(),
        'process_cpu_seconds': time.process_time(),
        'loadavg': maybe_text('/proc/loadavg'), 'memory': memory, 'process': status,
        'cpu_frequency_kHz': maybe_text(freq_root / 'scaling_cur_freq') if freq_root else None,
        'frequency_governor': maybe_text(freq_root / 'scaling_governor') if freq_root else None,
        'cpu_stat': next((x for x in (maybe_text('/proc/stat') or '').splitlines()
                          if x.startswith('cpu%d ' % cpu)), None) if cpu is not None else None,
        'memory_pressure': maybe_text('/proc/pressure/memory'),
        'ru_maxrss_kB': use.ru_maxrss if use else None, 'ru_nswap': use.ru_nswap if use else None,
        'ru_majflt': use.ru_majflt if use else None, 'ru_minflt': use.ru_minflt if use else None,
        'ru_nvcsw': use.ru_nvcsw if use else None, 'ru_nivcsw': use.ru_nivcsw if use else None,
        'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    }


def create_lock(output):
    require(platform.system() == 'Linux', 'Run the timing controller in the allocated Linux Slurm job.')
    require(os.environ.get('SLURM_JOB_ID'), 'No Slurm allocation; never train on the admin node.')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) == 20,
            'This benchmark requests one allocation of20 CPUs, with one active training thread.')
    allowed = set(os.sched_getaffinity(0))
    require(allowed, 'Empty allocated CPU affinity.')
    topo = topology(allowed)
    models = cpu_models()
    require(len(models) == 1, 'The selected node has multiple CPU model strings; no timing started.')
    candidates = [x for x in topo if x['cpu'] in allowed]
    require(candidates, 'Cannot map allocated CPUs to topology.')
    # Prefer a NUMA region with the most reserved CPUs and an allocated sibling set.
    def key(t):
        same = [q for q in topo if q['numa'] == t['numa']]
        reserved = sum(q['cpu'] in allowed for q in same)
        siblings_reserved = set(t['thread_siblings'] or [t['cpu']]).issubset(allowed)
        return (not siblings_reserved, -reserved, t['cpu'])
    chosen = min(candidates, key=key)
    same_numa = [q['cpu'] for q in topo if q['numa'] == chosen['numa']]
    same_socket = [q['cpu'] for q in topo if q['socket'] == chosen['socket']]
    lock = {
        'status': 'locked', 'host': socket.gethostname(), 'cpu_model': models[0],
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'allocated_affinity': sorted(allowed), 'training_cpu_id': chosen['cpu'],
        'training_socket': chosen['socket'], 'training_physical_core': chosen['core'],
        'training_numa_node': chosen['numa'],
        'thread_siblings_allocated': set(chosen['thread_siblings'] or [chosen['cpu']]).issubset(allowed),
        'entire_numa_affinity_reserved': set(same_numa).issubset(allowed),
        'entire_socket_affinity_reserved': set(same_socket).issubset(allowed),
        'topology': topo, 'training_threads': 1, 'simultaneous_trainings': 1,
        'resource_control': 'same physical node, fixed logical CPU, fixed1thread;20reserved CPU slots',
        'whole_node_exclusive': False,
        'memory_policy': 'OS default first-touch from pinned worker; no claimed full memory isolation',
        'plan_sha256': digest(ROOT / 'plan_before_runs.json'),
        'before_first_task': snapshot(chosen['cpu']),
    }
    save(Path(output) / 'hardware_lock.json', lock)
    return lock


def pin_and_verify(output):
    path = Path(output) / 'hardware_lock.json'
    require(path.is_file(), 'Hardware lock missing; use01_submit.sh, not a direct worker command.')
    lock = read(path)
    require(socket.gethostname() == lock['host'], 'Physical node changed; artifacts retained.')
    require(cpu_models() == [lock['cpu_model']], 'CPU model changed; artifacts retained.')
    require(os.environ.get('SLURM_JOB_ID') == lock['slurm_job_id'], 'Slurm allocation changed.')
    require(lock['plan_sha256'] == digest(ROOT / 'plan_before_runs.json'), 'Hardware/plan binding changed.')
    cpu = int(lock['training_cpu_id'])
    require(cpu in os.sched_getaffinity(0), 'Locked training CPU is outside this process allocation.')
    os.sched_setaffinity(0, {cpu})
    require(os.sched_getaffinity(0) == {cpu}, 'Failed to pin the training CPU.')
    return {'status': 'matched', 'host': lock['host'], 'cpu_model': lock['cpu_model'],
            'training_cpu_id': cpu, 'hardware_lock_sha256': digest(path),
            'affinity': sorted(os.sched_getaffinity(0)), 'snapshot': snapshot(cpu)}
