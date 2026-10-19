#!/usr/bin/env bash
# run_durability_coverage_gate.sh — RaptorQ durability coverage gate
#
# Scans all long-lived artifact directories and verifies each primary artifact
# has a corresponding sidecar/scrub/proof triplet. Emits E2E forensic log.
#
# Usage:
#   ./scripts/run_durability_coverage_gate.sh [--enforce] [--json <path>]
#
# Exit codes:
#   0 - All required artifacts have durability coverage
#   1 - Coverage gaps found (missing triplets)
#   2 - Usage error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENFORCE=0
JSON="$ROOT/artifacts/e2e/e2e_durability_coverage_gate.e2e.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enforce) ENFORCE=1; shift ;;
    --json)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --json requires a path" >&2
        exit 2
      fi
      JSON="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--enforce] [--json <path>]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Authoritative long-lived artifact directories (per durability_coverage_policy.v1.json)
declare -a ARTIFACT_SCOPES=(
  "$ROOT/crates/fj-conformance/fixtures:2"
  "$ROOT/artifacts/conformance:1"
  "$ROOT/artifacts/ci:1"
  "$ROOT/artifacts/performance:1"
)

# Artifact patterns to EXCLUDE from durability requirements
# (generated files, sidecar/scrub/proof themselves, etc.)
EXCLUDE_PATTERNS=(
  "*.sidecar.json"
  "*.scrub.json"
  "*.proof.json"
  "*.decode-proof.json"
  "*.verify.json"
  "*.stdout.log"
  "*.stderr.log"
)

resolve_triplet_file() {
  local artifact="$1"
  local suffix="$2"
  local stem="${artifact%.json}"
  local base
  base=$(basename "$stem")

  if [[ -f "${stem}.${suffix}.json" ]]; then
    echo "${stem}.${suffix}.json"
  elif [[ -f "$ROOT/artifacts/durability/${base}/${base}.${suffix}.json" ]]; then
    echo "$ROOT/artifacts/durability/${base}/${base}.${suffix}.json"
  elif [[ -f "$ROOT/artifacts/durability/${base}.${suffix}.json" ]]; then
    echo "$ROOT/artifacts/durability/${base}.${suffix}.json"
  else
    echo ""
  fi
}

count_artifacts() {
  local total=0
  local covered=0
  local missing_sidecar=0
  local missing_scrub=0
  local missing_proof=0
  local missing_all=0
  local stale_triplets=0

  for scope in "${ARTIFACT_SCOPES[@]}"; do
    local dir="${scope%:*}"
    local depth="${scope##*:}"
    if [[ ! -d "$dir" ]]; then
      continue
    fi

    while IFS= read -r -d '' artifact; do
      name=$(basename "$artifact")

      # Skip excluded patterns
      skip=0
      for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$name" == $pattern ]]; then
          skip=1
          break
        fi
      done
      [[ $skip -eq 1 ]] && continue

      total=$((total + 1))

      sidecar_path=$(resolve_triplet_file "$artifact" "sidecar")
      scrub_path=$(resolve_triplet_file "$artifact" "scrub")
      proof_path=$(resolve_triplet_file "$artifact" "proof")

      has_sidecar=0
      has_scrub=0
      has_proof=0
      [[ -n "$sidecar_path" ]] && has_sidecar=1
      [[ -n "$scrub_path" ]] && has_scrub=1
      [[ -n "$proof_path" ]] && has_proof=1

      if [[ $has_sidecar -eq 1 && $has_scrub -eq 1 && $has_proof -eq 1 ]]; then
        actual_sha=$(sha256sum "$artifact" | awk '{print $1}')
        sc_sha=$(jq -r '.artifact_sha256_hex // empty' "$sidecar_path" 2>/dev/null || true)
        sr_sha=$(jq -r '.decoded_sha256_hex // empty' "$scrub_path" 2>/dev/null || true)
        sr_ok=$(jq -r '.decoded_matches_expected // false' "$scrub_path" 2>/dev/null || true)
        pr_sha=$(jq -r '.decoded_sha256_hex // empty' "$proof_path" 2>/dev/null || true)
        pr_ok=$(jq -r '.recovered // false' "$proof_path" 2>/dev/null || true)

        if [[ "$actual_sha" == "$sc_sha" && "$actual_sha" == "$sr_sha" && "$sr_ok" == "true" && "$actual_sha" == "$pr_sha" && "$pr_ok" == "true" ]]; then
          covered=$((covered + 1))
        else
          stale_triplets=$((stale_triplets + 1))
        fi
      else
        if [[ $has_sidecar -eq 0 && $has_scrub -eq 0 && $has_proof -eq 0 ]]; then
          missing_all=$((missing_all + 1))
        fi
        [[ $has_sidecar -eq 0 ]] && missing_sidecar=$((missing_sidecar + 1))
        [[ $has_scrub -eq 0 ]] && missing_scrub=$((missing_scrub + 1))
        [[ $has_proof -eq 0 ]] && missing_proof=$((missing_proof + 1))
      fi
    done < <(find "$dir" -maxdepth "$depth" -name "*.json" -type f -print0 2>/dev/null)
  done

  echo "$total,$covered,$missing_sidecar,$missing_scrub,$missing_proof,$missing_all,$stale_triplets"
}

# Run coverage check
result=$(count_artifacts)
total=$(echo "$result" | cut -d, -f1)
covered=$(echo "$result" | cut -d, -f2)
missing_sidecar=$(echo "$result" | cut -d, -f3)
missing_scrub=$(echo "$result" | cut -d, -f4)
missing_proof=$(echo "$result" | cut -d, -f5)
missing_all=$(echo "$result" | cut -d, -f6)
stale_triplets=$(echo "$result" | cut -d, -f7)

uncovered=$((total - covered))
if [[ $total -gt 0 ]]; then
  coverage_pct=$((covered * 100 / total))
else
  coverage_pct=100
fi

# Determine status
if [[ $uncovered -eq 0 && $stale_triplets -eq 0 ]]; then
  status="pass"
else
  status="fail"
fi

# Generate E2E log
mkdir -p "$(dirname "$JSON")"
cat > "$JSON" <<EOF
{
  "schema_version": "frankenjax.e2e-forensic-log.v1",
  "bead_id": "frankenjax-cstq.5",
  "scenario_id": "e2e_durability_coverage_gate",
  "test_id": "durability_coverage_gate",
  "packet_id": null,
  "command": ["./scripts/run_durability_coverage_gate.sh", "--enforce"],
  "working_dir": "$ROOT",
  "environment": {
    "os": "$(uname -s | tr '[:upper:]' '[:lower:]')",
    "arch": "$(uname -m)",
    "timestamp_unix_ms": $(date +%s)000
  },
  "feature_flags": [],
  "fixture_ids": [],
  "oracle_ids": [],
  "transform_stack": [],
  "mode": "strict",
  "inputs": {
    "artifact_directories": [
      "crates/fj-conformance/fixtures",
      "artifacts/conformance",
      "artifacts/ci",
      "artifacts/performance"
    ]
  },
  "expected": {
    "coverage_requirement": "all_triplets_sha256_fresh",
    "minimum_coverage_pct": 100
  },
  "actual": {
    "total_artifacts": $total,
    "covered_artifacts": $covered,
    "uncovered_artifacts": $uncovered,
    "coverage_pct": $coverage_pct,
    "missing_sidecar": $missing_sidecar,
    "missing_scrub": $missing_scrub,
    "missing_proof": $missing_proof,
    "missing_all": $missing_all,
    "stale_triplets": $stale_triplets
  },
  "tolerance": {
    "policy_id": "all_triplets_required",
    "notes": "every primary artifact requires sidecar + scrub + proof matching current on-disk SHA-256"
  },
  "error": $(if [[ "$status" == "fail" ]]; then echo '"coverage_or_freshness_gap"'; else echo 'null'; fi),
  "timings": {
    "total_ms": 0
  },
  "allocations": {
    "measurement_backend": "not_measured"
  },
  "artifacts": [],
  "replay_command": "./scripts/run_durability_coverage_gate.sh --enforce",
  "status": "$status",
  "failure_summary": $(if [[ "$status" == "fail" ]]; then echo "\"$uncovered artifacts missing or stale durability triplets (stale=$stale_triplets)\""; else echo 'null'; fi),
  "redactions": [],
  "metadata": {
    "bead": {
      "id": "frankenjax-cstq.5",
      "title": "Enforce RaptorQ coverage for all long-lived artifacts"
    }
  }
}
EOF

# Output summary
echo "===== Durability Coverage Gate ====="
echo "Total artifacts:     $total"
echo "Covered (fresh):     $covered"
echo "Uncovered/Stale:     $uncovered"
echo "Coverage:            $coverage_pct%"
echo ""
echo "Missing / Stale breakdown:"
echo "  - Missing sidecar: $missing_sidecar"
echo "  - Missing scrub:   $missing_scrub"
echo "  - Missing proof:   $missing_proof"
echo "  - Missing all:     $missing_all"
echo "  - Stale SHA-256:   $stale_triplets"
echo ""
echo "Status: $status"
echo "E2E log: $JSON"
echo "===================================="

if [[ "$ENFORCE" -eq 1 && "$status" == "fail" ]]; then
  echo ""
  echo "FAIL: durability coverage gate requires 100% fresh triplet coverage"
  exit 1
fi

exit 0
