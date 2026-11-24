#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON_BIN:-python}" run.py verify
