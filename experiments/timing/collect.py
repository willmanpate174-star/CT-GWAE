"""Validate completed outputs and export small timing results."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics
import zipfile
from runtime_diagnostics import diagnostics

from common import (ROOT, check_success, digest, folder_for, plan, read, require, save,
                    timing_record, verify_package)


def current_state(output):
    counts = {"passed": 0, "failed": 0, "trained600": 0, "partial": 0, "not_started": 0,"blocked":0}
    by_size = {}
    for item in plan()["tasks"]:
        folder = folder_for(output, item)
        progress = read(folder / "progress.json") if (folder / "progress.json").exists() else {}
        if progress.get("completed") == 600:
            counts["trained600"] += 1
        state = "not_started"
        if (folder / "success.json").exists():
            state = "passed"
        elif (folder / "blocked.json").exists():
            state="blocked"
        elif (folder / "failure.json").exists():
            state = "failed"
        elif (folder / "claim.json").exists():
            state = "partial"
        counts[state] += 1
        group = by_size.setdefault(str(item["N"]), {"passed": 0, "total": sum(t["N"]==item["N"] for t in plan()["tasks"])})
        group["passed"] += state == "passed"
    return counts, by_size


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(output):
    import numpy as np

    verify_package()
    output = Path(output).resolve()
    require(output.is_dir(), "Output directory is absent.")
    moment = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = output / ("summary_" + moment)
    destination.mkdir()
    rows, failures, environments = [], [], []
    lock=read(output/"hardware_lock.json") if (output/"hardware_lock.json").exists() else None
    for item in plan()["tasks"]:
        folder = folder_for(output, item)
        if not (folder / "success.json").exists():
            failures.append({
                "task_index": item["task_index"], "id": item["id"], "N": item["N"],
                "condition": item["condition"],
                "failure": read(folder / "failure.json") if (folder / "failure.json").exists()
                           else "No passed receipt. Inspect preserved task/Slurm logs.",
            })
            continue
        try:
            receipt = check_success(folder)
            require(not receipt["fixture_only"] and receipt["task"] == item, "Task identity differs.")
            training = read(folder / "training.json")
            timing = timing_record(training)
            require(timing == receipt["timing"], "Timing readback differs.")
            environment = read(folder / "environment.json")
            require(lock is not None,"Hardware lock missing.")
            require(environment["host"]==lock["host"] and environment["cpu_model"]==lock["cpu_model"],"Node or CPU changed.")
            require(environment["CPU_affinity"]==[lock["training_cpu_id"]],"Training CPU affinity changed.")
            for name in ("hardware_check.json","hardware_check_replay.json"):
                check=read(folder/name)
                require(check["hardware_lock_sha256"]==digest(output/"hardware_lock.json"),"Hardware-lock binding differs.")
                require(check["affinity"]==[lock["training_cpu_id"]],"Replay/training core differs.")
            replay=read(folder/"replay_receipt.json")
            require(replay["status"]=="passed" and replay["fresh_process"] and not replay["memo_used"] and replay["MAP_exact"],"Independent replay gate failed.")
            require(all(v["exact"] for v in replay["comparisons"].values()),"Endpoint replay differs.")
            environments.append(environment)
            rows.append({
                "owner":plan()["owner"],"execution_sha256":digest(ROOT/"execution_protocol.json"),
                "id": item["id"], "N": item["N"], "condition": item["condition"],
                "instance": item["instance"], "kind": item["kind"], "r": 0, "j": 0,
                "model_seed": item["model_seed"], "graph_seed": item["graph_seed"],
                **timing, **diagnostics(folder,timing), "training_cpu_id":lock["training_cpu_id"],
                "host": environment["host"], "cpu_model": environment["cpu_model"],
                "independent_replay": "passed",
            })
        except Exception as error:
            failures.append({"task_index": item["task_index"], "id": item["id"],
                             "N": item["N"], "condition": item["condition"], "failure": repr(error)})
    cpu_models = sorted({env["cpu_model"] for env in environments})
    hosts = sorted({env["host"] for env in environments})
    comparable = len(cpu_models) == 1 and len(hosts)==1
    summaries = []
    for condition in ("balanced", "imbalanced"):
        for n in plan()["node_counts"]:
            group = [row for row in rows if row["condition"] == condition and row["N"] == n]
            expected=sum(t["condition"]==condition and t["N"]==n for t in plan()["tasks"])
            complete = len(group) == expected and comparable
            values = [row["training_cpu_seconds"] for row in group]
            summaries.append({
                "condition": condition, "N": n, "independent_graphs_passed": len(group),
                "expected_graphs": expected, "status": "complete" if complete else "incomplete",
                "mean_training_cpu_seconds": statistics.mean(values) if complete else "",
                "sample_sd_cpu_seconds": statistics.stdev(values) if complete else "",
                "mean_cpu_seconds_per_1000_nodes": statistics.mean(values) * 1000 / n if complete else "",
            })
    slopes = []
    for condition in ("balanced", "imbalanced"):
        group = [row for row in summaries if row["condition"] == condition]
        if all(row["status"] == "complete" for row in group):
            x = np.log([row["N"] for row in group])
            y = np.log([row["mean_training_cpu_seconds"] for row in group])
            exponent, intercept = np.polyfit(x, y, 1)
            slopes.append({
                "condition": condition, "node_count_points": len(plan()["node_counts"]),
                "log_log_exponent": float(exponent), "log_intercept": float(intercept),
                "definition": "OLS log(mean proposal CPU seconds) versus log(N); descriptive.",
            })
    write_csv(destination / "training_timing_graph_level.csv", rows)
    write_csv(destination / "training_time_summary.csv", summaries)
    write_csv(destination / "observed_scaling_slopes.csv", slopes)
    complete = len(rows) == 50 and not failures
    save(destination / "completion.json", {
        "status": "complete" if complete else "incomplete",
        "passed": len(rows), "expected": 50, "incomplete_or_invalid_tasks": len(failures),
        "actual_failure_files":sum((folder_for(output,t)/"failure.json").exists() for t in plan()["tasks"]),
        "same_cpu_model": comparable, "same_node":len(hosts)==1,
        "owner":plan()["owner"],"hardware_stratification_required":len(cpu_models)>1, "cpu_models": cpu_models, "hosts": hosts,
        "plan_sha256": digest(ROOT / "plan_before_runs.json"),
        "manifest_sha256": digest(ROOT / "manifest.json"),
        "old_timings_used_in_new_summary": False,
        "new_cold_starts": len(rows), "truth_evaluation": False,
        "hardware_lock":lock,"same_locked_training_core":bool(lock and len(rows)>0),
        "shared_node_not_exclusive":True,
    })
    save(destination / "failures.json", failures)
    archive = output.parent / ("CTGWAE_Timing50k_djh_small_" + moment + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for file in sorted(destination.iterdir()):
            bundle.write(file, "summary/" + file.name)
        for name in ("plan_before_runs.json", "manifest.json", "timing_scope.json", "README_CN.md",
                     "global_plan_before_split.json","execution_protocol.json","source_binding.json","input_catalog.json","VALIDATION.json"):
            bundle.write(ROOT / name, "protocol/" + name)
        for name in ("submission.json", "submission_failure.json", "run_plan.json", "hardware_lock.json", "controller_state.json", "controller_terminal.json", "controller_failure.json", "collection_failure.json"):
            if (output / name).exists():
                bundle.write(output / name, "submission/" + name)
        small_files = (
            "claim.json", "input_receipt.json", "environment.json", "progress.json",
            "timing.json", "proposal_timing.csv", "success.json", "failure.json",
            "replay_receipt.json", "replay_comparison.json", "endpoint_losses.json",
            "decoder.json", "process_cost.json", "replay.log", "worker.log", "execution_report.json",
            "hardware_check.json", "hardware_check_replay.json", "resource_samples.jsonl", "proposal_runtime.csv",
        )
        for item in plan()["tasks"]:
            folder = folder_for(output, item)
            for name in small_files:
                if (folder / name).is_file():
                    bundle.write(folder / name, f"tasks/task_{item['task_index']:03d}/" + name)
        for log in sorted((output / "logs").glob("*")):
            if log.is_file() and log.stat().st_size < 2 * 1024 * 1024:
                bundle.write(log, "logs/" + log.name)
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None, "Archive integrity failed.")
    save(output / "last_collection.json", {
        "status": "complete" if complete else "incomplete", "archive": str(archive),
        "summary": str(destination), "passed": len(rows), "expected": 50,
    }, replace=True)
    print("Passed:", len(rows), "/50", flush=True)
    print("Result:", "complete" if complete else "incomplete; inspect summary/failures.json", flush=True)
    print("DOWNLOAD:", archive, flush=True)
    print("MiB:", round(archive.stat().st_size / 1024**2, 3), flush=True)
    return complete

