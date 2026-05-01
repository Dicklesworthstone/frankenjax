#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$ROOT/artifacts/conformance/architecture_boundary_decision.v1.json"
MARKDOWN="$ROOT/artifacts/conformance/architecture_boundary_decision.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_architecture_boundary_gate.e2e.json"
ENFORCE=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --report requires a path" >&2
        exit 2
      fi
      REPORT="$2"
      shift 2
      ;;
    --markdown)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --markdown requires a path" >&2
        exit 2
      fi
      MARKDOWN="$2"
      shift 2
      ;;
    --e2e)
      if [[ $# -lt 2 || "$2" == --* ]]; then
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
      echo "Usage: run_architecture_boundary_gate.sh [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cargo run -p fj-conformance --bin fj_architecture_boundary_gate -- \
  --root "$ROOT" \
  --report "$REPORT" \
  --markdown "$MARKDOWN" \
  --e2e "$E2E" \
  "${ENFORCE[@]}"
