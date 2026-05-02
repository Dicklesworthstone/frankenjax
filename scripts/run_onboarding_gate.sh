#!/usr/bin/env bash
# run_onboarding_gate.sh - fresh-clone onboarding command inventory gate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="$ROOT/artifacts/conformance/onboarding_command_inventory.v1.json"
E2E="$ROOT/artifacts/e2e/e2e_onboarding_gate.e2e.json"
ENFORCE=0

usage() {
  cat <<'USAGE'
Usage: run_onboarding_gate.sh [--inventory <json>] [--e2e <json>] [--enforce]

Generates and validates the frankenjax-cstq.18 onboarding command inventory gate.
The gate validates documented command anchors, script paths, replay commands,
skip rationales, evidence refs, environment allowlists, and E2E log shape.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inventory)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --inventory requires a path" >&2
        exit 2
      fi
      INVENTORY="$2"
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
      ENFORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ARGS=(
  --root "$ROOT"
  --inventory "$INVENTORY"
  --e2e "$E2E"
)
if [[ "$ENFORCE" -eq 1 ]]; then
  ARGS+=(--enforce)
fi

cargo run -p fj-conformance --bin fj_onboarding_gate -- "${ARGS[@]}"
"$ROOT/scripts/validate_e2e_logs.sh" "$E2E"
