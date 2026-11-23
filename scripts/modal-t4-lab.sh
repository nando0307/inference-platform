#!/bin/bash
# Local-only preparation and guarded Modal launch for modal_t4_lab.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${2:-diagnostic}"

usage() {
  printf 'Usage: %s {preflight|launch|apps|stop} [mode|app-id]\n' "$0"
  printf 'Modes: diagnostic, single-small, single-large, shared\n'
}

case "${1:-}" in
  preflight)
    modal --version
    modal profile current >/dev/null
    modal token info >/dev/null 2>&1
    printf 'Modal profile selected; authentication accepted (token details suppressed).\n'
    python3 - <<'PY'
from pathlib import Path
import os

root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
models = {
    "Qwen2.5-0.5B-Instruct": "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen3-1.7B": "models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
for name, relative in models.items():
    snapshot = root / relative
    print(f"{name}: {'available' if snapshot.is_dir() and any(snapshot.glob('*.safetensors')) else 'missing'} ({snapshot})")
PY
    ;;
  launch)
    case "$MODE" in diagnostic|single-small|single-large|shared) ;; *) usage >&2; exit 2 ;; esac
    printf 'About to allocate one T4 for mode=%s (maximum 16 minutes).\n' "$MODE" >&2
    printf 'Confirm credits, budget, and spending limits in Modal before proceeding.\n' >&2
    modal run "$ROOT/modal_t4_lab.py" --mode "$MODE"
    ;;
  apps)
    modal app list
    ;;
  stop)
    app_id="${2:-}"
    if [[ ! "$app_id" =~ ^ap-[[:alnum:]]+$ ]]; then
      printf 'Pass the exact app ID created by this run (expected ap-*); no broad stop is allowed.\n' >&2
      exit 2
    fi
    modal app stop "$app_id"
    ;;
  *) usage >&2; exit 2 ;;
esac
