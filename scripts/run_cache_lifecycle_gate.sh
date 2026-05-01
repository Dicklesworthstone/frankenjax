#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="$ROOT/artifacts/conformance/cache_legacy_parity_ledger.v1.json"
REPORT="$ROOT/artifacts/conformance/cache_lifecycle_report.v1.json"
MARKDOWN="$ROOT/artifacts/conformance/cache_legacy_parity_ledger.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json"
ENFORCE=()

while (($#)); do
  case "$1" in
    --ledger)
      if [[ $# -lt 2 ]]; then
        echo "error: --ledger requires a path" >&2
        exit 2
      fi
      LEDGER="$2"
      shift 2
      ;;
    --report)
      if [[ $# -lt 2 ]]; then
        echo "error: --report requires a path" >&2
        exit 2
      fi
      REPORT="$2"
      shift 2
      ;;
    --markdown)
      if [[ $# -lt 2 ]]; then
        echo "error: --markdown requires a path" >&2
        exit 2
      fi
      MARKDOWN="$2"
      shift 2
      ;;
    --e2e)
      if [[ $# -lt 2 ]]; then
        echo "error: --e2e requires a path" >&2
        exit 2
      fi
      E2E="$2"
      shift 2
      ;;
    --enforce)
      ENFORCE=(--enforce)
      shift
      ;;
    -h|--help)
      echo "Usage: run_cache_lifecycle_gate.sh [--ledger <json>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cargo run -p fj-conformance --bin fj_cache_lifecycle_gate -- \
  --root "$ROOT" \
  --ledger "$LEDGER" \
  --report "$REPORT" \
  --markdown "$MARKDOWN" \
  --e2e "$E2E" \
  "${ENFORCE[@]}"
