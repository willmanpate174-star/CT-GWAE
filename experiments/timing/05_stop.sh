#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/server.env"
JOB_ID="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["job_id"])' "$ROOT/submission.json")"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Invalid stored job ID"; exit 1; }
scancel "$JOB_ID"
echo "Stopped only this package job: $JOB_ID; existing partial outputs retained."
