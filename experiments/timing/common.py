"""Shared, source-checked utilities for the Fig. 15 timing extension."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
NUMERICAL_PACKAGES = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "torch": "2.14.0+cpu",
    "scikit-learn": "1.9.0",
    "threadpoolctl": "3.6.0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if not replace:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        return
    handle, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def plan():
    return read(ROOT / "plan_before_runs.json")


def pinned_environment():
    require(sys.version_info[:2] == (3, 12), "Use the original Python 3.12 scalability environment.")
    packages = {name: importlib.metadata.version(name) for name in NUMERICAL_PACKAGES}
    require(packages == NUMERICAL_PACKAGES,
            "Numerical environment differs from Fig. 15: " + json.dumps(packages))
    cpu_model = platform.processor()
    cpu_file = Path("/proc/cpuinfo")
    if cpu_file.exists():
        fields = [line.partition(":")[2].strip()
                  for line in cpu_file.read_text().splitlines() if line.startswith("model name")]
        cpu_model = fields[0] if fields else cpu_model
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "host": socket.gethostname(), "cpu_model": cpu_model,
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
        "threads_per_training": 1,
        "execution_sha256":digest(ROOT/"execution_protocol.json"),
        "CPU_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
    }


def verify_package():
    expected = read(ROOT / "manifest.json")["sha256"]
    for relative, wanted in expected.items():
        file = (ROOT / relative).resolve()
        require(file.is_relative_to(ROOT) and file.is_file(), "Missing package file: " + relative)
        require(digest(file) == wanted, "Package hash mismatch: " + relative)
    lineage = read(ROOT / "model/lineage.json")
    for relative, info in lineage.items():
        require(digest(ROOT / "model" / relative) == info["sha256"], "Frozen numerical source changed: " + relative)
    p = plan()
    require(len(p["tasks"]) == 50 and p["node_counts"] == [1000,5000,10000,20000,50000], "Invalid timing scope.")
    originals = {t["task_index"]: t for t in read(ROOT / "global_plan_before_split.json")["tasks"]}
    require(digest(ROOT / "global_plan_before_split.json") == p["original_plan_sha256"], "Original plan changed.")
    require(all(t == originals[t["task_index"]] for t in p["tasks"]), "Registered graph/seed identity changed.")
    require(len({t["model_seed"] for t in p["tasks"]}) == 50, "Duplicate model seed.")
    require(len(p["execution_order"]) == 50 and set(p["execution_order"]) == {t["task_index"] for t in p["tasks"]}, "Task order coverage changed.")
    for n in p["node_counts"]:
        for c in p["conditions"]:
            require(sum(t["N"]==n and t["condition"]==c for t in p["tasks"]) == 5, "Expected5graphs per cell.")
    require(p["proposals"]==600 and p["S24"]==24 and p["S100"]==100 and p["training_threads"]==1, "Frozen budgets changed.")
    catalog = read(ROOT / "input_catalog.json")
    for t in p["tasks"]:
        spec = catalog[t["id"]]
        folder = (ROOT / spec["path"]).resolve()
        require(folder.is_relative_to(ROOT), "Input path escapes package.")
        for name,h in spec["sha256"].items():
            require(digest(folder/name)==h, "Sealed input changed: "+t["id"]+"/"+name)
    return {"status":"verified","manifest":digest(ROOT/"manifest.json"),"plan":digest(ROOT/"plan_before_runs.json"),
            "tasks":50,"node_counts":p["node_counts"],"graphs_per_cell":5,"training_threads":1,
            "reserved_cpus":20,"simultaneous_jobs":1,"simultaneous_trainings":1,"training":False,"truth_access":False}


def load_graph(folder, item):
    import numpy as np
    from scipy import sparse

    folder = Path(folder)
    metadata = read(folder / "metadata.json")
    require((metadata["N"], metadata["L"], metadata["K"]) == (item["N"], 4, 4),
            "Graph dimensions differ from the plan.")
    with np.load(folder / "graphs.npz", allow_pickle=False) as arrays:
        layers = [sparse.csr_matrix(
            (arrays[f"data_{index}"], arrays[f"indices_{index}"], arrays[f"indptr_{index}"]),
            shape=(item["N"], item["N"])) for index in range(4)]
    for layer in layers:
        layer.check_format(full_check=True)
        require(layer.has_canonical_format and np.isfinite(layer.data).all()
                and (layer.data == 1).all() and not (layer - layer.T).nnz
                and not layer.diagonal().any(), "Expected binary undirected loop-free CSR layers.")
    return layers, metadata


def timing_record(training):
    import math

    trace = training["trace"]
    require(training["proposals"] == 600 and len(trace) == 600, "Incomplete training budget.")
    require([row["proposal_id"] for row in trace] == list(range(600)), "Proposal ordering differs.")
    require(all(row["inner_unroll_steps_per_S24"] == 24 for row in trace), "S24 changed.")
    cpu = [float(row["cpu_seconds"]) for row in trace]
    wall = [float(row["wall_seconds"]) for row in trace]
    require(all(math.isfinite(value) and value >= 0 for value in cpu + wall), "Invalid timing.")
    return {
        "training_cpu_seconds": sum(cpu), "training_wall_sum_seconds": sum(wall),
        "S24_calls": sum(row["S24_calls_total"] for row in trace), "proposals": 600,
        "accepted": training["accepted"], "rejected": 600 - training["accepted"],
        "selected_epoch": training["selected_epoch"],
    }


def check_success(folder):
    folder = Path(folder)
    receipt = read(folder / "success.json")
    require(receipt["status"] == "passed", "Candidate is not passed.")
    require(receipt["plan_sha256"] == digest(ROOT / "plan_before_runs.json"), "Plan binding differs.")
    for relative, wanted in receipt["artifacts"].items():
        require(digest(folder / relative) == wanted, "Completed artifact changed: " + relative)
    return receipt


def folder_for(output, task):
    return Path(output) / "tasks" / f"task_{task['task_index']:03d}"

