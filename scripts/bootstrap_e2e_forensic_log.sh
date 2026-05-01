#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_PATH="${FJ_E2E_BOOTSTRAP_REPORT:-$ROOT_DIR/artifacts/e2e/e2e_forensic_log_contract_bootstrap.validation.json}"

cargo test -p fj-conformance --test e2e_forensic_log_contract -- \
  bootstrap_e2e_forensic_log_sample_validates --exact --nocapture

"$ROOT_DIR/scripts/validate_e2e_logs.sh" \
  --output "$REPORT_PATH" \
  "$ROOT_DIR/artifacts/e2e/e2e_forensic_log_contract_bootstrap.e2e.json"

printf 'bootstrap validation report: %s\n' "$REPORT_PATH"
