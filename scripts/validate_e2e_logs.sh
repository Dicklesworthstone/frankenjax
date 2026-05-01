#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_PATH=""
ARGS=()

usage() {
  cat <<'USAGE'
Usage: ./scripts/validate_e2e_logs.sh [--output <report.json>] [log-or-dir ...]

Validates FrankenJAX shared E2E forensic logs against the Rust validator.
If no log-or-dir is supplied, validates the bootstrap contract sample.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      if [[ $# -lt 2 ]]; then
        echo "error: --output requires a path" >&2
        exit 2
      fi
      REPORT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS+=("$ROOT_DIR/artifacts/e2e/e2e_forensic_log_contract_bootstrap.e2e.json")
fi

CMD=(
  cargo run -p fj-conformance --bin fj_e2e_log_check --
  --root "$ROOT_DIR"
  --json
)

if [[ -n "$REPORT_PATH" ]]; then
  CMD+=(--output "$REPORT_PATH")
fi

CMD+=("${ARGS[@]}")

"${CMD[@]}"
