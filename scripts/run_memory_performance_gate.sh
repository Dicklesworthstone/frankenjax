#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$ROOT/artifacts/performance/memory_performance_gate.v1.json"
MARKDOWN="$ROOT/artifacts/performance/memory_performance_gate.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_memory_performance_gate.e2e.json"
ENFORCE=()

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
      ENFORCE=(--enforce)
      shift
      ;;
    -h|--help)
      echo "Usage: run_memory_performance_gate.sh [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cargo run -p fj-conformance --bin fj_memory_performance_gate -- \
  --root "$ROOT" \
  --report "$REPORT" \
  --markdown "$MARKDOWN" \
  --e2e "$E2E" \
  "${ENFORCE[@]}"
