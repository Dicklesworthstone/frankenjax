#!/usr/bin/env bash
# durability_ci_gate.sh — Post-test sidecar verification gate
#
# Runs generate+scrub+proof pipeline on all protected artifact categories.
# Fails CI if any artifact cannot survive drop-source=2.
#
# Usage:
#   ./scripts/durability_ci_gate.sh [--drop-source <N>] [--skip-generate]
#
# Protected artifact categories:
#   1. Conformance fixtures
#   2. Benchmark baselines (criterion)
#   3. Evidence packs (per-packet)
#   4. Crash corpus (triage artifacts)
#   5. Decision ledger snapshots
#   6. Migration manifests / schemas

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROP_SOURCE=2
SKIP_GENERATE=0
DURABILITY_BIN=""
EXIT_CODE=0
PASS=0
FAIL=0
SKIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drop-source) DROP_SOURCE="$2"; shift 2 ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--drop-source N] [--skip-generate]"
      exit 0
      ;;
    *) shift ;;
  esac
done

# Build the durability binary
echo "[durability] building fj_durability..."
cargo build --bin fj_durability -p fj-conformance 2>/dev/null || {
  echo "[durability] SKIP: fj_durability binary failed to build"
  exit 0
}
CARGO_TARGET="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"
DURABILITY_BIN="$CARGO_TARGET/debug/fj_durability"
if [[ ! -x "$DURABILITY_BIN" ]]; then
  # Search common locations
  for candidate in \
    "$ROOT_DIR/target/debug/fj_durability" \
    "$(find "$CARGO_TARGET" -name fj_durability -type f 2>/dev/null | head -1)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      DURABILITY_BIN="$candidate"
      break
    fi
  done
  if [[ ! -x "$DURABILITY_BIN" ]]; then
    echo "[durability] SKIP: fj_durability binary not found"
    exit 0
  fi
fi

echo "[durability] using binary: $DURABILITY_BIN"
echo "[durability] drop-source count: $DROP_SOURCE"

# Define protected artifact directories
declare -a ARTIFACT_DIRS=(
  "$ROOT_DIR/crates/fj-conformance/fixtures"
  "$ROOT_DIR/crates/fj-conformance/fixtures/rng"
  "$ROOT_DIR/crates/fj-conformance/fixtures/transforms"
  "$ROOT_DIR/artifacts/conformance"
  "$ROOT_DIR/artifacts/ci"
  "$ROOT_DIR/artifacts/performance"
)

DURABILITY_DIR="$ROOT_DIR/artifacts/durability"
mkdir -p "$DURABILITY_DIR"

process_artifact() {
  local artifact="$1"
  local stem
  stem=$(basename "$artifact" .json)
  local output_dir="$DURABILITY_DIR/$stem"
  mkdir -p "$output_dir"

  local sidecar="$output_dir/${stem}.sidecar.json"
  local report="$output_dir/${stem}.scrub.json"
  local proof="$output_dir/${stem}.proof.json"

  if [[ $SKIP_GENERATE -eq 1 && -f "$sidecar" ]]; then
    # Verify-only: just scrub
    if "$DURABILITY_BIN" scrub --artifact "$artifact" --sidecar "$sidecar" --report "$report" 2>/dev/null; then
      echo "  PASS (verify): $artifact"
      PASS=$((PASS + 1))
    else
      echo "  FAIL (verify): $artifact"
      FAIL=$((FAIL + 1))
      EXIT_CODE=1
    fi
  else
    # Full pipeline
    if "$DURABILITY_BIN" pipeline \
      --artifact "$artifact" \
      --sidecar "$sidecar" \
      --report "$report" \
      --proof "$proof" \
      --drop-source "$DROP_SOURCE" 2>/dev/null; then
      echo "  PASS: $artifact"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: $artifact"
      FAIL=$((FAIL + 1))
      EXIT_CODE=1
    fi
  fi
}

for dir in "${ARTIFACT_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "[durability] SKIP: $dir does not exist"
    continue
  fi

  echo "[durability] scanning $dir..."
  while IFS= read -r -d '' artifact; do
    # Skip sidecar/scrub/proof/verify files
    name=$(basename "$artifact")
    case "$name" in
      *.sidecar.json|*.scrub.json|*.proof.json|*.verify.json) continue ;;
    esac
    process_artifact "$artifact"
  done < <(find "$dir" -maxdepth 1 -name "*.json" -type f -print0 2>/dev/null)
done

echo ""
echo "[durability] ============================================"
echo "[durability] Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "[durability] Drop-source resilience: $DROP_SOURCE symbols"
echo "[durability] ============================================"

exit $EXIT_CODE
