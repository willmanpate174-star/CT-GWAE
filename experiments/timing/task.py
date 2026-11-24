"""Run one fixed training and an independent one-thread endpoint replay."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import csv
import gc
from pathlib import Path
import subprocess
import sys
import time
import traceback

from execution_adapter import Execution
from hardware import pin_and_verify, snapshot
import json

from common import (ROOT, check_success, digest, folder_for, load_graph, pinned_environment,
                    plan, read, require, save, timing_record, verify_package)


def runtime():
    sys.path.insert(0, str(ROOT / "model"))
    from bootstrap import activate
    values = activate()
    torch = values[6]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    return values


def prepare_input(item, folder):
    specification = read(ROOT / "input_catalog.json")[item["id"]]
    source = ROOT / specification["path"]
    for name,wanted in specification["sha256"].items():
        require(digest(source/name)==wanted, "Presealed graph changed: "+item["id"]+"/"+name)
    return source


def endpoints(net, obj, B, torch, np, folder):
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    selected = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)["network"]
    arrays, losses = {}, {}
    for name, state in (("initial", initial), ("selected", selected)):
        net.load_state_dict(state, strict=True)
        for steps in (24, 100):
            T, history, loss, checks = B.forward(net, obj, steps, False)
            arrays[f"{name}_S{steps}"] = T.detach().numpy().copy()
            losses[f"{name}_S{steps}"] = float(loss)
    np.savez_compressed(folder / "reference_endpoints.npz", **arrays)
    return arrays["selected_S100"], losses


def train(item, folder, input_path):
    B, H, C, Schedule, wrapper, decoder, torch, np, cfg, support = runtime()
    layers, metadata = load_graph(input_path, item)
    for name in ("proposal_logs", "proposal_ledgers"):
        (folder / name).mkdir()
    original_save = H.save
    execution=Execution(B)
    execution.install()
    proposal_started=time.perf_counter()
    previous_stats={}
    original_protected = C.protected_proposal
    def counted_proposal(*args, **kwargs):
        before = execution.stats()
        result = original_protected(*args, **kwargs)
        after = execution.stats()
        result[1]["runtime_call_counts"] = {k:after.get(k,0)-before.get(k,0) for k in after}
        return result
    C.protected_proposal = counted_proposal

    def persist(path, value):
        nonlocal previous_stats
        if Path(path).parent.name == "proposal_logs":
            stats=execution.stats()
            value["execution_memo"]={k:stats.get(k,0)-previous_stats.get(k,0) for k in stats}
            previous_stats=stats
        original_save(path, value)
        if Path(path).parent.name == "proposal_logs":
            completed = int(Path(path).stem) + 1
            save(folder / "progress.json", {
                "phase": "training", "completed": completed, "total": 600,
                "after_loss": value["after_loss"], "N": item["N"],
                "condition": item["condition"], "seed": item["model_seed"],
                "elapsed_training_loop_seconds":time.perf_counter()-proposal_started,
                "last_proposal_wall_seconds":value["wall_seconds"],
                "last_proposal_cpu_seconds":value["cpu_seconds"],
                "S24_calls_requested":value["S24_calls_total"],
                "memo":execution.stats(),"training_threads":1,
            }, replace=True)
            if completed == 1 or completed % 10 == 0:
                print(item["id"], completed, "/600", flush=True)
                cpu = next(iter(os.sched_getaffinity(0))) if hasattr(os,"sched_getaffinity") else None
                with (folder/"resource_samples.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps({"completed":completed,**snapshot(cpu)},allow_nan=False)+"\n")

    H.save = persist
    slot = dict(seed=item["model_seed"], mode="restored", variant="original-control", proposals=600)
    try:
        net, obj, training = H.train_candidate(slot, layers, metadata, B, torch, Schedule,
                                              C, folder, cfg, {})
        save(folder/"execution_report.json",execution.report())
    finally:
        H.save=original_save
        C.protected_proposal=original_protected
        execution.restore()
    terminal = torch.load(folder / "terminal.pt", map_location="cpu", weights_only=True)
    require(terminal["schedule"]["completed"] == 600, "Terminal proposal counter differs.")
    timing = timing_record(training)
    transport, losses = endpoints(net, obj, B, torch, np, folder)
    require(abs(losses["selected_S24"] - training["selected_loss"]) < 1e-10,
            "Selected checkpoint objective differs.")
    raw = transport.argmax(1).astype(np.int64)
    labels, decoding, raw_score, hard = wrapper.decode_and_score(
        transport, raw, 4, obj.hard_diagnostic, decoder, np)
    np.save(folder / "labels.npy", labels, allow_pickle=False)
    save(folder / "decoder.json", decoding)
    save(folder / "endpoint_losses.json", losses)
    with (folder / "proposal_timing.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["proposal_id", "cpu_seconds", "wall_seconds", "S24_calls_total",
                  "direction_type", "accepted_scale", "before_loss", "after_loss"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in training["trace"])
    with (folder/"proposal_runtime.csv").open("x",encoding="utf-8",newline="") as stream:
        keys=["proposal_id","cpu_seconds","wall_seconds","S24_calls_total","real_forward_calls","real_gradient_calls","real_primal_calls","cache_hits"]
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in training["trace"]:
            counts=row["runtime_call_counts"]
            writer.writerow({**{k:row[k] for k in keys[:4]},**{k:counts.get(k,0) for k in keys[4:]}})
    save(folder / "timing.json", timing)
    return timing


def replay(item, folder):
    import numpy as np

    hardware=pin_and_verify(folder.parent.parent)
    save(folder/"hardware_check_replay.json",hardware)
    verify_package()
    pinned_environment()
    specification = read(folder / "input_receipt.json")
    for name, wanted in specification["sha256"].items():
        require(digest(Path(specification["path"]) / name) == wanted, "Replay input changed.")
    layers, metadata = load_graph(specification["path"], item)
    B, H, C, Schedule, wrapper, decoder, torch, np, cfg, support = runtime()
    net, obj, initial_state = B.construct(
        layers, 4, item["model_seed"], "restored", "original-control")
    sys.path.insert(0,str(ROOT/"execution"))
    from csr_runtime import apply_csr
    apply_csr(net,obj,cached_backward=True)
    import hashlib
    feature_sha = hashlib.sha256(net.X.numpy().tobytes()).hexdigest()
    require(feature_sha == read(folder / "initialization.json")["feature_sha256"],
            "Independent structural input features differ.")
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    require(B.state_digest(initial_state) == B.state_digest(initial), "Initial cold state differs.")
    selected = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)["network"]
    comparisons = {}
    with np.load(folder / "reference_endpoints.npz", allow_pickle=False) as arrays:
        for name, state in (("initial", initial), ("selected", selected)):
            net.load_state_dict(state, strict=True)
            for steps in (24, 100):
                key = f"{name}_S{steps}"
                T, history, loss, checks = B.forward(net, obj, steps, False)
                actual = T.detach().numpy()
                reference = arrays[key]
                comparisons[key] = {
                    "exact": bool(np.array_equal(actual, reference)),
                    "maximum_absolute_error": float(np.max(np.abs(actual - reference))),
                }
                if key == "selected_S100":
                    raw = actual.argmax(1).astype(np.int64)
                    labels, decoding, raw_score, hard = wrapper.decode_and_score(
                        actual, raw, 4, obj.hard_diagnostic, decoder, np)
                    require(np.array_equal(labels, np.load(folder / "labels.npy", allow_pickle=False)),
                            "Independent MAP labels differ.")
    save(folder / "replay_comparison.json", comparisons)
    require(all(row["exact"] for row in comparisons.values()), "Independent endpoint replay differs.")
    save(folder / "replay_receipt.json", {
        "status": "passed", "fresh_process": True, "threads": 1, "comparisons": comparisons,
        "MAP_exact": True, "truth_access": False, "training": False,
        "memo_used":False,"layout":"CSR","execution_sha256":digest(ROOT/"execution_protocol.json"),
    })


def worker(item, folder, fixture=False):
    hardware=pin_and_verify(folder.parent.parent)
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/"hardware_check.json",hardware)
    save(folder / "claim.json", {
        "pid": os.getpid(), "task": item, "fixture_only": fixture,
        "command":sys.argv,"owner":plan()["owner"],"unrestricted_wall_time_request":True,
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID"),
        "plan_sha256": digest(ROOT / "plan_before_runs.json"),
    })
    started, cpu = time.perf_counter(), time.process_time()
    passed = False
    try:
        save(folder/"progress.json",{"phase":"input_preparation","completed":0,"total":600,
            "N":item["N"],"condition":item["condition"],"seed":item["model_seed"]},replace=True)
        verify_package()
        environment = pinned_environment()
        save(folder / "environment.json", environment)
        source = prepare_input(item, folder)
        fingerprints = {name: digest(source / name)
                        for name in ("graphs.npz", "metadata.json", "node_index.npz", "evaluation.npz")}
        save(folder / "input_receipt.json", {
            "path": str(source.resolve()), "sha256": fingerprints,
            "graph_seed": item["graph_seed"], "labels_read_for_evaluation": False,
        })
        save(folder/"progress.json",{"phase":"construction_and_training","completed":0,"total":600,
            "N":item["N"],"condition":item["condition"],"seed":item["model_seed"]},replace=True)
        timing = train(item, folder, source)
        save(folder / "progress.json", {
            "phase": "independent_replay", "completed": 600, "total": 600,
            "N": item["N"], "condition": item["condition"], "seed": item["model_seed"],
        }, replace=True)
        gc.collect()
        with (folder / "replay.log").open("x", encoding="utf-8") as stream:
            command = [sys.executable, "-B", str(ROOT / "task.py"), "replay", "--folder", str(folder)]
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        require(result.returncode == 0, "Independent replay failed; inspect replay.log.")
        require(read(folder / "replay_receipt.json")["status"] == "passed", "Replay receipt missing.")
        files = ("initial.pt", "checkpoint.pt", "terminal.pt", "training.json", "training_transport.npz",
                 "reference_endpoints.npz", "labels.npy", "decoder.json", "endpoint_losses.json",
                 "proposal_timing.csv", "timing.json", "replay_receipt.json", "input_receipt.json",
                 "environment.json", "execution_report.json", "hardware_check.json",
                 "hardware_check_replay.json", "resource_samples.jsonl", "proposal_runtime.csv")
        save(folder / "success.json", {
            "status": "passed", "task": item, "timing": timing, "fixture_only": fixture,
            "plan_sha256": digest(ROOT / "plan_before_runs.json"),
            "artifacts": {name: digest(folder / name) for name in files},
            "truth_access": False, "new_cold_start": True,
        })
        save(folder / "progress.json", {
            "phase": "passed", "completed": 600, "total": 600,
            "N": item["N"], "condition": item["condition"], "seed": item["model_seed"],
        }, replace=True)
        passed = True
    except BaseException as error:
        save(folder / "failure.json", {
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
            "task": item, "fixture_only": fixture, "no_replacement": True,
        })
        raise
    finally:
        save(folder / "process_cost.json", {
            "status": "passed" if passed else "failed",
            "total_worker_wall_seconds": time.perf_counter() - started,
            "total_worker_cpu_seconds": time.process_time() - cpu,
            "scope": "Whole worker process, distinct from the Fig. 15 proposal CPU metric.",
            "resource_at_exit":snapshot(next(iter(os.sched_getaffinity(0))) if hasattr(os,"sched_getaffinity") else None),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("worker", "replay"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.action == "replay":
        item = read(args.folder / "claim.json")["task"]
        replay(item, args.folder)
    else:
        item = next(t for t in plan()["tasks"] if t["task_index"]==args.task_index)
        worker(item, folder_for(args.output, item))


if __name__ == "__main__":
    main()

