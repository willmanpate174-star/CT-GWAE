"""One Slurm allocation;50 predetermined cold starts run sequentially on one CPU."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from datetime import datetime, timezone
import getpass
import importlib
import json
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from common import ROOT, digest, folder_for, pinned_environment, plan, read, require, save, verify_package


def numerical_import_check():
    names = [('numpy','numpy'),('scipy','scipy'),('torch','torch'),
             ('scikit-learn','sklearn'),('threadpoolctl','threadpoolctl')]
    imported = {key: str(importlib.import_module(module).__version__) for key,module in names}
    from common import NUMERICAL_PACKAGES
    require(imported == NUMERICAL_PACKAGES, 'Imported numerical package versions differ: '+json.dumps(imported))
    return imported


def existing_output():
    require((ROOT / 'submission.json').is_file(), 'No submission found. Run01_submit.sh first.')
    return Path(read(ROOT / 'submission.json')['output'])


def submission_command(output, python):
    command = ['sbatch', '--parsable', '--partition=' + os.environ['SLURM_PARTITION'],
               '--account=' + os.environ['SLURM_ACCOUNT'],
               '--qos=' + os.environ['SLURM_QOS'],
               '--nodes=1', '--ntasks=1', '--cpus-per-task=20', '--mem=16G',
               '--time=0', '--no-requeue', '--distribution=block:block',
               '--job-name=CTGWAE-timing', '--chdir=' + str(ROOT),
               '--output=' + str(output / 'logs/benchmark_%j.out'),
               '--error=' + str(output / 'logs/benchmark_%j.err')]
    node = os.environ.get('RUNTIME_NODE', '').strip()
    if node:
        require(all(c.isalnum() or c in '_-' for c in node), 'RUNTIME_NODE must be one physical node name.')
        command.append('--nodelist=' + node)
    command += [str(ROOT / 'batch_job.sh'), str(ROOT), str(python), str(output)]
    return command


def submit(output_arg=None):
    require(sys.platform.startswith('linux'), 'Submit from the school Linux WebShell.')
    require(all(os.environ.get(k) for k in ('SLURM_PARTITION','SLURM_ACCOUNT','SLURM_QOS')),
            'Set SLURM_PARTITION, SLURM_ACCOUNT and SLURM_QOS for your cluster.')
    check = verify_package()
    environment = pinned_environment()
    numerical_import_check()
    if (ROOT / 'submission.json').is_file():
        q = read(ROOT / 'submission.json')
        print('Already submitted. No new job created:', q['job_id'])
        print('Output:', q['output'])
        return
    require(not (ROOT / 'submission_claim.json').exists(),
            'A prior submission claim exists. Inspect submission_failure.json and squeue; do not restart blindly.')
    require(shutil.which('sbatch') and shutil.which('srun'), 'Slurm commands not found.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = (Path(output_arg).expanduser() if output_arg else
              Path.home() / ('CTGWAE_Timing50k_results_' + stamp)).resolve()
    require(not output.exists(), 'Existing output directory is preserved; choose a new path.')
    command = submission_command(output, sys.executable)
    save(ROOT / 'submission_claim.json', {'output': str(output), 'manifest': check['manifest'], 'command': command})
    try:
        (output / 'logs').mkdir(parents=True)
        save(output / 'run_plan.json', {'plan': plan(), 'manifest_sha256': check['manifest'],
             'environment_at_submission': environment, 'command': command,
             'whole_node_exclusive': False, 'reserved_cpus': 20, 'training_threads': 1,
             'simultaneous_jobs': 1, 'simultaneous_trainings': 1})
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        require(result.returncode == 0, 'sbatch failed before training:\n' + result.stdout + result.stderr)
        identifiers = [line.strip().split(';')[0] for line in result.stdout.splitlines()
                       if re.fullmatch(r'\d+(?:;[^\s]+)?', line.strip())]
        require(len(identifiers) == 1, 'Unexpected sbatch response: ' + result.stdout)
        job = identifiers[0]
        receipt = {'status': 'submitted', 'job_id': job, 'jobs': {'benchmark': job},
                   'output': str(output), 'tasks': 50, 'node_counts': plan()['node_counts'],
                   'reserved_cpus': 20, 'training_threads': 1, 'simultaneous_trainings': 1,
                   'max_jobs': 1, 'max_cpus': 20, 'Slurm_time_request': '0=UNLIMITED',
                   'no_requeue': True, 'training_code_changed': False,
                   'plan_sha256': check['plan'], 'manifest_sha256': check['manifest']}
        save(output / 'submission.json', receipt)
        save(ROOT / 'submission.json', receipt)
        (ROOT / 'last_output.txt').write_text(str(output) + '\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2), flush=True)
        print('One node, one locked training CPU,50sequential600-proposal runs -> replay -> small ZIP.', flush=True)
        print('20CPUs reserved; one active training thread. Shared-node isolation is logged, not assumed.', flush=True)
    except BaseException as error:
        failure = {'status': 'submission_incomplete', 'output': str(output),
                   'error': repr(error), 'traceback': traceback.format_exc()}
        save(ROOT / 'submission_failure.json', failure)
        if output.exists():
            save(output / 'submission_failure.json', failure)
        raise


def sequential(output):
    from hardware import create_lock, snapshot
    from collect import collect
    verify_package()
    pinned_environment()
    require(not (output / 'controller_claim.json').exists(),
            'This run already has a controller claim; existing partial results are never overwritten.')
    save(output / 'controller_claim.json', {'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'),
         'command': sys.argv, 'plan_sha256': digest(ROOT / 'plan_before_runs.json')})
    active = None
    state = 'failed'
    def interrupted(signum, _frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        raise InterruptedError('Received signal%d; no automatic replacement or restart.' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        lock = create_lock(output)
        print('HARDWARE LOCK:', json.dumps({k: lock[k] for k in
              ('host', 'cpu_model', 'training_cpu_id', 'training_numa_node', 'whole_node_exclusive')},
              ensure_ascii=False), flush=True)
        tasks = {t['task_index']: t for t in plan()['tasks']}
        for number, index in enumerate(plan()['execution_order'], 1):
            item = tasks[index]
            folder = folder_for(output, item)
            require(not folder.exists(), 'Task folder already exists; preserve it rather than retrying: ' + str(folder))
            folder.mkdir(parents=True)
            save(output / 'controller_state.json', {'status': 'running', 'sequence': number, 'total': 50,
                 'task': item, 'host': lock['host'], 'cpu_model': lock['cpu_model'],
                 'training_cpu_id': lock['training_cpu_id'], 'before_task': snapshot(lock['training_cpu_id'])}, replace=True)
            with (folder / 'worker.log').open('x', encoding='utf-8') as stream:
                command = [sys.executable, '-B', str(ROOT / 'task.py'), 'worker',
                           '--task-index', str(index), '--output', str(output)]
                active = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                print('START %d/50' % number, item['id'], 'pid=', active.pid, flush=True)
                while active.poll() is None:
                    time.sleep(2)
                code = active.returncode
            active = None
            require(code == 0, 'Task failed; remaining tasks were not started. Inspect ' + str(folder / 'worker.log'))
            from common import check_success
            receipt = check_success(folder)
            require(receipt['task'] == item and not receipt['fixture_only'], 'Task identity or fixture status differs.')
            print('PASS %d/50' % number, item['id'],
                  round(receipt['timing']['training_cpu_seconds'], 3), 'CPU seconds', flush=True)
        state = 'complete'
        save(output / 'controller_terminal.json', {'status': state, 'completed': 50, 'expected': 50})
    except BaseException as error:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        failure = {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc(),
                   'no_replacement': True, 'no_automatic_restart': True}
        save(output / 'controller_failure.json', failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
    finally:
        save(output / 'controller_state.json', {'status': state, 'total': 50,
             'job_id': os.environ.get('SLURM_JOB_ID')}, replace=True)
        # Export even on interruption; incomplete outputs remain explicitly incomplete.
        try:
            collected = collect(output)
            if state == 'complete':
                require(collected, 'Final rehash/collection is incomplete; inspect summary/failures.json.')
        except BaseException as error:
            save(output / 'collection_failure.json', {'status': 'failed', 'error': repr(error),
                 'traceback': traceback.format_exc(), 'task_artifacts_retained': True})
            print('Collection failed; original task files retained:', repr(error), flush=True)
            if state == 'complete':
                save(output / 'controller_state.json', {'status': 'collection_failed', 'total': 50}, replace=True)
                raise


def status(output):
    from collect import current_state
    counts, groups = current_state(output)
    for n, q in sorted(groups.items(), key=lambda x: int(x[0])):
        print('N=%6d independently passed %d/%d' % (int(n), q['passed'], q['total']))
    print(json.dumps(counts, indent=2), flush=True)
    lock_path = output / 'hardware_lock.json'
    if lock_path.exists():
        lock = read(lock_path)
        print('Fixed node/CPU/core:', lock['host'], lock['cpu_model'], lock['training_cpu_id'])
    for item in plan()['tasks']:
        folder = folder_for(output, item)
        if not (folder / 'claim.json').exists() or (folder / 'success.json').exists():
            continue
        q = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
        print(item['id'], '%s/600' % q.get('completed', 0), 'phase=', q.get('phase', 'unknown'))
        if (folder / 'failure.json').exists():
            print((folder / 'failure.json').read_text(encoding='utf-8')[:1600])
    if (output / 'submission.json').exists():
        job = str(read(output / 'submission.json')['job_id'])
        subprocess.run(['squeue', '-j', job, '-o', '%.22i %.10T %.14M %.6C %.16l %.30R'], check=False)
    print('Output:', output)
    if (output / 'last_collection.json').exists():
        print('SMALL ZIP:', read(output / 'last_collection.json')['archive'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'submit', 'sequential', 'status', 'collect', 'archive'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        print(json.dumps({**verify_package(), 'environment': pinned_environment(),
                          'actual_import_versions': numerical_import_check()}, indent=2))
    elif args.action == 'submit':
        submit(args.output)
    else:
        output = args.output.resolve() if args.output else existing_output()
        if args.action == 'sequential':
            sequential(output)
        elif args.action == 'status':
            status(output)
        elif args.action == 'collect':
            from collect import collect
            collect(output)
        else:
            require((output / 'last_collection.json').exists(), 'No ZIP yet. Run03_results.sh to export one.')
            q = read(output / 'last_collection.json')
            print('Result:', q['status'], '%d/50' % q['passed'])
            print('DOWNLOAD:', q['archive'])


if __name__ == '__main__':
    main()
