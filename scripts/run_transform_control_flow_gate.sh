#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$ROOT/artifacts/conformance/transform_control_flow_matrix.v1.json"
MARKDOWN="$ROOT/artifacts/conformance/transform_control_flow_matrix.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_transform_control_flow_gate.e2e.json"
ENFORCE=0

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
    --enforce)
      ENFORCE=1
      shift
      ;;
    -h|--help)
      echo "Usage: run_transform_control_flow_gate.sh [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
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

if [[ "$ENFORCE" -eq 1 ]]; then
  ARGS+=(--enforce)
fi

cargo run -p fj-conformance --bin fj_transform_control_flow_gate -- "${ARGS[@]}"
"$ROOT/scripts/validate_e2e_logs.sh" "$E2E"
