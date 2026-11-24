#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
PYTHON_BIN="$2"
OUTPUT="$3"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec srun --ntasks=1 --cpus-per-task=20 --cpu-bind=cores --distribution=block:block --kill-on-bad-exit=1 \
  "$PYTHON_BIN" -B "$ROOT/run.py" sequential --output "$OUTPUT"
