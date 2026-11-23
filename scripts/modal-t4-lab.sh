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
