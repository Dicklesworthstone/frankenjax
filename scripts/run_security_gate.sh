#!/usr/bin/env bash
# run_security_gate.sh - security/adversarial evidence gate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREAT_MODEL="$ROOT/artifacts/conformance/security_threat_model.v1.json"
REPORT="$ROOT/artifacts/conformance/security_adversarial_gate.v1.json"
MARKDOWN="$ROOT/artifacts/conformance/security_adversarial_gate.v1.md"
E2E="$ROOT/artifacts/e2e/e2e_security_gate.e2e.json"
ENFORCE=0
CATEGORY=""
FAMILY=""

usage() {
  cat <<'USAGE'
Usage: run_security_gate.sh [--threat-model <json>] [--report <json>] [--markdown <md>] [--e2e <json>] [--category <id>] [--family <id>] [--enforce]

Generates and validates the frankenjax-cstq.17 security/adversarial evidence gate.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threat-model)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --threat-model requires a path" >&2
        exit 2
      fi
      THREAT_MODEL="$2"
      shift 2
      ;;
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
        echo "error: $1 requires a path" >&2
        exit 2
      fi
      E2E="$2"
      shift 2
      ;;
    --category)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --category requires an id" >&2
        exit 2
      fi
      CATEGORY="$2"
      shift 2
      ;;
    --family)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --family requires an id" >&2
        exit 2
      fi
      FAMILY="$2"
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
  --threat-model "$THREAT_MODEL"
  --report "$REPORT"
  --markdown "$MARKDOWN"
  --e2e "$E2E"
)

if [[ -n "$CATEGORY" ]]; then
  ARGS+=(--category "$CATEGORY")
fi
if [[ -n "$FAMILY" ]]; then
  ARGS+=(--family "$FAMILY")
fi
if [[ "$ENFORCE" -eq 1 ]]; then
  ARGS+=(--enforce)
fi

cargo run -p fj-conformance --bin fj_security_gate -- "${ARGS[@]}"

"$ROOT/scripts/validate_e2e_logs.sh" "$E2E"
