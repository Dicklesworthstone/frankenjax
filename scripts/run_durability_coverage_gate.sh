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

# Authoritative long-lived artifact directories (per cstq.5 acceptance)
declare -a ARTIFACT_DIRS=(
  # Conformance artifacts
  "$ROOT/artifacts/conformance"
  # E2E forensic logs
  "$ROOT/artifacts/e2e"
  # CI reliability budgets
  "$ROOT/artifacts/ci"
  # Performance baselines and gates
  "$ROOT/artifacts/performance"
  # Phase2C migration manifests
  "$ROOT/artifacts/phase2c"
  # Schema definitions
  "$ROOT/artifacts/schemas"
)

# Artifact patterns to EXCLUDE from durability requirements
# (generated files, sidecar/scrub/proof themselves, etc.)
EXCLUDE_PATTERNS=(
  "*.sidecar.json"
  "*.scrub.json"
  "*.proof.json"
  "*.verify.json"
  "*.stdout.log"
  "*.stderr.log"
)

count_artifacts() {
  local total=0
  local covered=0
  local missing_sidecar=0
  local missing_scrub=0
  local missing_proof=0
  local missing_all=0
  local details=""

  for dir in "${ARTIFACT_DIRS[@]}"; do
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

      stem="${artifact%.json}"
      has_sidecar=0
      has_scrub=0
      has_proof=0

      # Check for sidecar in same dir or durability dir
      if [[ -f "${stem}.sidecar.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem").sidecar.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem")/$(basename "$stem").sidecar.json" ]]; then
        has_sidecar=1
      fi

      # Check for scrub report
      if [[ -f "${stem}.scrub.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem").scrub.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem")/$(basename "$stem").scrub.json" ]]; then
        has_scrub=1
      fi

      # Check for decode proof
      if [[ -f "${stem}.proof.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem").proof.json" ]] || \
         [[ -f "$ROOT/artifacts/durability/$(basename "$stem")/$(basename "$stem").proof.json" ]]; then
        has_proof=1
      fi

      if [[ $has_sidecar -eq 1 && $has_scrub -eq 1 && $has_proof -eq 1 ]]; then
        covered=$((covered + 1))
      else
        if [[ $has_sidecar -eq 0 && $has_scrub -eq 0 && $has_proof -eq 0 ]]; then
          missing_all=$((missing_all + 1))
        fi
        [[ $has_sidecar -eq 0 ]] && missing_sidecar=$((missing_sidecar + 1))
        [[ $has_scrub -eq 0 ]] && missing_scrub=$((missing_scrub + 1))
        [[ $has_proof -eq 0 ]] && missing_proof=$((missing_proof + 1))
        details="$details$(basename "$artifact"): sidecar=$has_sidecar scrub=$has_scrub proof=$has_proof\n"
      fi
    done < <(find "$dir" -maxdepth 2 -name "*.json" -type f -print0 2>/dev/null)
  done

  echo "$total,$covered,$missing_sidecar,$missing_scrub,$missing_proof,$missing_all"
}

# Run coverage check
result=$(count_artifacts)
total=$(echo "$result" | cut -d, -f1)
covered=$(echo "$result" | cut -d, -f2)
missing_sidecar=$(echo "$result" | cut -d, -f3)
missing_scrub=$(echo "$result" | cut -d, -f4)
missing_proof=$(echo "$result" | cut -d, -f5)
missing_all=$(echo "$result" | cut -d, -f6)

uncovered=$((total - covered))
if [[ $total -gt 0 ]]; then
  coverage_pct=$((covered * 100 / total))
else
  coverage_pct=100
fi

# Determine status
if [[ $uncovered -eq 0 ]]; then
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
      "artifacts/conformance",
      "artifacts/e2e",
      "artifacts/ci",
      "artifacts/performance",
      "artifacts/phase2c",
      "artifacts/schemas"
    ]
  },
  "expected": {
    "coverage_requirement": "all_triplets",
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
    "missing_all": $missing_all
  },
  "tolerance": {
    "policy_id": "all_triplets_required",
    "notes": "every primary artifact requires sidecar + scrub + proof"
  },
  "error": $(if [[ "$status" == "fail" ]]; then echo '"coverage_gap"'; else echo 'null'; fi),
  "timings": {
    "total_ms": 0
  },
  "allocations": {
    "measurement_backend": "not_measured"
  },
  "artifacts": [],
  "replay_command": "./scripts/run_durability_coverage_gate.sh --enforce",
  "status": "$status",
  "failure_summary": $(if [[ "$status" == "fail" ]]; then echo "\"$uncovered artifacts missing durability triplets\""; else echo 'null'; fi),
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
echo "Covered (triplet):   $covered"
echo "Uncovered:           $uncovered"
echo "Coverage:            $coverage_pct%"
echo ""
echo "Missing breakdown:"
echo "  - Missing sidecar: $missing_sidecar"
echo "  - Missing scrub:   $missing_scrub"
echo "  - Missing proof:   $missing_proof"
echo "  - Missing all:     $missing_all"
echo ""
echo "Status: $status"
echo "E2E log: $JSON"
echo "===================================="

if [[ "$ENFORCE" -eq 1 && "$status" == "fail" ]]; then
  echo ""
  echo "FAIL: durability coverage gate requires 100% triplet coverage"
  exit 1
fi

exit 0
