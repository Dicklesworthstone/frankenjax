#!/usr/bin/env bash
# nightly_scrub.sh — Full integrity scan of all protected artifacts
#
# Designed to run as a nightly cron job or scheduled CI pipeline.
# Scans all durability sidecars and verifies artifact integrity.
#
# Usage:
#   ./scripts/nightly_scrub.sh [--json] [--alert-on-failure]
#
# Cron template:
#   0 3 * * * cd /path/to/frankenjax && ./scripts/nightly_scrub.sh --json >> artifacts/ci/nightly_scrub.log 2>&1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_OUTPUT=0
ALERT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON_OUTPUT=1; shift ;;
    --alert-on-failure) ALERT=1; shift ;;
    -h|--help) echo "Usage: $0 [--json] [--alert-on-failure]"; exit 0 ;;
    *) shift ;;
  esac
done

TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
DURABILITY_DIR="$ROOT_DIR/artifacts/durability"
PASS=0
FAIL=0
MISSING=0
RESULTS=()

if [[ ! -d "$DURABILITY_DIR" ]]; then
  if [[ $JSON_OUTPUT -eq 1 ]]; then
    echo "{\"schema_version\":\"frankenjax.nightly-scrub.v1\",\"timestamp\":\"$TIMESTAMP\",\"status\":\"skip\",\"reason\":\"no durability directory\"}"
  else
    echo "[nightly-scrub] No durability directory found — nothing to scan"
  fi
  exit 0
fi

# Build fj_durability if needed
CARGO_TARGET="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"
DURABILITY_BIN="$CARGO_TARGET/debug/fj_durability"
if [[ ! -x "$DURABILITY_BIN" ]]; then
  cargo build --bin fj_durability -p fj-conformance 2>/dev/null || {
    echo "[nightly-scrub] WARNING: could not build fj_durability"
    exit 1
  }
  DURABILITY_BIN="$CARGO_TARGET/debug/fj_durability"
fi

# Find all artifact/sidecar pairs
while IFS= read -r -d '' sidecar; do
  stem=$(basename "$sidecar" .sidecar.json)
  parent=$(dirname "$sidecar")

  # Resolve the original artifact from the sidecar's recorded artifact_path or fallback directories
  artifact=""
  recorded_path=$(jq -r '.artifact_path // empty' "$sidecar" 2>/dev/null || true)
  recorded_rel="${recorded_path#/data/projects/frankenjax/}"
  recorded_rel="${recorded_rel#/dp/frankenjax/}"
  if [[ -n "$recorded_rel" && -f "$ROOT_DIR/$recorded_rel" ]]; then
    artifact="$ROOT_DIR/$recorded_rel"
  elif [[ -n "$recorded_path" && -f "$recorded_path" ]]; then
    artifact="$recorded_path"
  else
    for dir in \
      "$ROOT_DIR/crates/fj-conformance/fixtures" \
      "$ROOT_DIR/crates/fj-conformance/fixtures/rng" \
      "$ROOT_DIR/crates/fj-conformance/fixtures/transforms" \
      "$ROOT_DIR/artifacts/conformance" \
      "$ROOT_DIR/artifacts/ci" \
      "$ROOT_DIR/artifacts/performance" \
      "$ROOT_DIR/artifacts/e2e" \
      "$ROOT_DIR/artifacts/e2e/golden_journeys" \
      "$ROOT_DIR/artifacts/performance/evidence"; do
      candidate="$dir/${stem}.json"
      if [[ -f "$candidate" ]]; then
        artifact="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$artifact" ]]; then
    MISSING=$((MISSING + 1))
    RESULTS+=("{\"sidecar\":\"$sidecar\",\"status\":\"missing_artifact\"}")
    continue
  fi

  report="$parent/${stem}.nightly_scrub.json"
  if "$DURABILITY_BIN" scrub --artifact "$artifact" --sidecar "$sidecar" --report "$report" 2>/dev/null; then
    PASS=$((PASS + 1))
    RESULTS+=("{\"artifact\":\"$artifact\",\"status\":\"pass\"}")
  else
    FAIL=$((FAIL + 1))
    RESULTS+=("{\"artifact\":\"$artifact\",\"status\":\"fail\"}")
  fi
done < <(find "$DURABILITY_DIR" -name "*.sidecar.json" -type f -print0 2>/dev/null)

OVERALL="pass"
if [[ $FAIL -gt 0 ]]; then
  OVERALL="fail"
fi

if [[ $JSON_OUTPUT -eq 1 ]]; then
  RESULTS_JSON=$(printf '%s\n' "${RESULTS[@]}" | python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
  cat <<EOF
{
  "schema_version": "frankenjax.nightly-scrub.v1",
  "timestamp": "$TIMESTAMP",
  "status": "$OVERALL",
  "passed": $PASS,
  "failed": $FAIL,
  "missing_artifact": $MISSING,
  "results": $RESULTS_JSON
}
EOF
else
  echo "[nightly-scrub] $TIMESTAMP"
  echo "[nightly-scrub] Results: $PASS passed, $FAIL failed, $MISSING missing artifact"
  echo "[nightly-scrub] Overall: $(echo $OVERALL | tr '[:lower:]' '[:upper:]')"
fi

if [[ $ALERT -eq 1 && $FAIL -gt 0 ]]; then
  echo "[nightly-scrub] ALERT: $FAIL artifact(s) failed integrity check!" >&2
fi

[[ $FAIL -eq 0 ]]
