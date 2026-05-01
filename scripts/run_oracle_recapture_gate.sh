#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_PATH="${FJ_ORACLE_MATRIX:-$ROOT_DIR/artifacts/conformance/oracle_recapture_matrix.v1.json}"
DRIFT_PATH="${FJ_ORACLE_DRIFT:-$ROOT_DIR/artifacts/conformance/oracle_drift_report.v1.json}"
MARKDOWN_PATH="${FJ_ORACLE_MARKDOWN:-$ROOT_DIR/artifacts/conformance/oracle_recapture_matrix.v1.md}"
E2E_PATH="${FJ_ORACLE_E2E:-$ROOT_DIR/artifacts/e2e/e2e_oracle_recapture_gate.e2e.json}"
BASELINE_PATH=""
ENFORCE=0
REQUIRE_BASELINE=0

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_oracle_recapture_gate.sh [--baseline matrix.json] [--require-baseline] [--enforce]

Builds the oracle recapture matrix, drift report, markdown preview, and shared-schema
E2E forensic log for frankenjax-cstq.1.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      if [[ $# -lt 2 ]]; then
        echo "error: --baseline requires a path" >&2
        exit 2
      fi
      BASELINE_PATH="$2"
      shift 2
      ;;
    --require-baseline)
      REQUIRE_BASELINE=1
      shift
      ;;
    --enforce)
      ENFORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CMD=(
  cargo run -p fj-conformance --bin fj_oracle_recapture_gate --
  --root "$ROOT_DIR"
  --matrix "$MATRIX_PATH"
  --drift "$DRIFT_PATH"
  --markdown "$MARKDOWN_PATH"
  --e2e "$E2E_PATH"
)

if [[ -n "$BASELINE_PATH" ]]; then
  CMD+=(--baseline "$BASELINE_PATH")
fi
if [[ "$REQUIRE_BASELINE" -eq 1 ]]; then
  CMD+=(--require-baseline)
fi
if [[ "$ENFORCE" -eq 1 ]]; then
  CMD+=(--enforce)
fi

"${CMD[@]}"
