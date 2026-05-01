#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$ROOT/artifacts/conformance/error_taxonomy_matrix.v1.json"
MARKDOWN="$ROOT/artifacts/conformance/error_taxonomy_matrix.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_error_taxonomy_gate.e2e.json"
ENFORCE=0
CASE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      REPORT="$2"
      shift 2
      ;;
    --markdown)
      MARKDOWN="$2"
      shift 2
      ;;
    --e2e)
      E2E="$2"
      shift 2
      ;;
    --case)
      CASE_ID="$2"
      shift 2
      ;;
    --enforce)
      ENFORCE=1
      shift
      ;;
    -h|--help)
      echo "Usage: run_error_taxonomy_gate.sh [--report <json>] [--markdown <md>] [--e2e <json>] [--case <case_id>] [--enforce]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=(
  --root "$ROOT"
  --report "$REPORT"
  --markdown "$MARKDOWN"
  --e2e "$E2E"
)

if [[ -n "$CASE_ID" ]]; then
  ARGS+=(--case "$CASE_ID")
fi

if [[ "$ENFORCE" -eq 1 ]]; then
  ARGS+=(--enforce)
fi

cargo run -p fj-conformance --bin fj_error_taxonomy_gate -- "${ARGS[@]}"
"$ROOT/scripts/validate_e2e_logs.sh" "$E2E"
