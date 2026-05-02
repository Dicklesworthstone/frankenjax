#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON="$ROOT/artifacts/e2e/e2e_api_readme_quickstart.e2e.json"
ENFORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --json requires a path" >&2
        exit 2
      fi
      JSON="$2"
      shift 2
      ;;
    --enforce)
      ENFORCE=1
      shift
      ;;
    -h|--help)
      echo "Usage: run_api_readme_examples.sh [--json <path>] [--enforce]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cargo run -p fj-api --example readme_quickstart -- --json "$JSON"

if [[ "$ENFORCE" -eq 1 ]]; then
  "$ROOT/scripts/validate_e2e_logs.sh" "$JSON"
fi
