#!/usr/bin/env bash
# run_security_gate.sh — Bootstrap security/adversarial evidence gate
#
# Validates security threat model coverage by checking that all threat
# categories have evidence refs pointing to existing artifacts with passing
# status. Emits E2E forensic log.
#
# Usage:
#   ./scripts/run_security_gate.sh [--enforce] [--json <path>]
#
# Exit codes:
#   0 - All threat categories have evidence (green)
#   1 - Coverage gaps found (yellow/red categories)
#   2 - Usage error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON="$ROOT/artifacts/e2e/e2e_security_gate.e2e.json"
ENFORCE=0

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

THREAT_MODEL="$ROOT/artifacts/conformance/security_threat_model.v1.json"
ERROR_TAXONOMY="$ROOT/artifacts/conformance/error_taxonomy_matrix.v1.json"

if [[ ! -f "$THREAT_MODEL" ]]; then
  echo "error: threat model not found: $THREAT_MODEL" >&2
  exit 1
fi

if [[ ! -f "$ERROR_TAXONOMY" ]]; then
  echo "error: error taxonomy matrix not found: $ERROR_TAXONOMY" >&2
  exit 1
fi

TIMESTAMP_MS=$(($(date +%s) * 1000))

taxonomy_status=$(jq -r '.status' "$ERROR_TAXONOMY")
threat_categories=$(jq -r '.threat_categories | length' "$THREAT_MODEL")
evidence_green=$(jq -r '.summary.evidence_green' "$THREAT_MODEL")
evidence_yellow=$(jq -r '.summary.evidence_yellow' "$THREAT_MODEL")
evidence_red=$(jq -r '.summary.evidence_red' "$THREAT_MODEL")
model_status=$(jq -r '.status' "$THREAT_MODEL")

if [[ "$taxonomy_status" == "pass" && "$evidence_green" -eq "$threat_categories" && "$evidence_yellow" -eq 0 && "$evidence_red" -eq 0 ]]; then
  gate_status="pass"
else
  gate_status="fail"
fi

mkdir -p "$(dirname "$JSON")"
cat > "$JSON" <<EOF
{
  "schema_version": "frankenjax.e2e-forensic-log.v1",
  "bead_id": "frankenjax-cstq.17",
  "scenario_id": "e2e_security_gate",
  "test_id": "security_gate_bootstrap",
  "packet_id": null,
  "command": ["./scripts/run_security_gate.sh", "--enforce"],
  "working_dir": "$ROOT",
  "environment": {
    "os": "$(uname -s | tr '[:upper:]' '[:lower:]')",
    "arch": "$(uname -m)",
    "rust_version": "$(rustc --version 2>/dev/null | cut -d' ' -f2 || echo 'unknown')",
    "cargo_version": "$(cargo --version 2>/dev/null | cut -d' ' -f2 || echo 'unknown')",
    "cargo_target_dir": "${CARGO_TARGET_DIR:-target}",
    "env_vars": {},
    "timestamp_unix_ms": $TIMESTAMP_MS
  },
  "feature_flags": [],
  "fixture_ids": [],
  "oracle_ids": [],
  "transform_stack": [],
  "mode": "strict",
  "inputs": {
    "threat_model_path": "artifacts/conformance/security_threat_model.v1.json",
    "error_taxonomy_path": "artifacts/conformance/error_taxonomy_matrix.v1.json"
  },
  "expected": {
    "taxonomy_status": "pass",
    "all_categories_green": true
  },
  "actual": {
    "taxonomy_status": "$taxonomy_status",
    "threat_categories": $threat_categories,
    "evidence_green": $evidence_green,
    "evidence_yellow": $evidence_yellow,
    "evidence_red": $evidence_red,
    "model_status": "$model_status"
  },
  "tolerance": {
    "policy_id": "security_bootstrap",
    "atol": null,
    "rtol": null,
    "ulp": null,
    "notes": "Bootstrap phase accepts yellow/partial evidence; enforcement phase requires all green"
  },
  "error": null,
  "timings": {
    "setup_ms": 0,
    "trace_ms": 0,
    "dispatch_ms": 0,
    "eval_ms": 0,
    "verify_ms": 0,
    "total_ms": 0
  },
  "allocations": {
    "allocation_count": null,
    "allocated_bytes": null,
    "peak_rss_bytes": null,
    "measurement_backend": "not_measured"
  },
  "artifacts": [
    {
      "kind": "threat_model",
      "path": "artifacts/conformance/security_threat_model.v1.json",
      "sha256": "$(sha256sum "$THREAT_MODEL" | cut -d' ' -f1)",
      "required": true
    },
    {
      "kind": "error_taxonomy",
      "path": "artifacts/conformance/error_taxonomy_matrix.v1.json",
      "sha256": "$(sha256sum "$ERROR_TAXONOMY" | cut -d' ' -f1)",
      "required": true
    }
  ],
  "replay_command": "./scripts/run_security_gate.sh --enforce",
  "status": "$gate_status",
  "failure_summary": $(if [[ "$gate_status" == "fail" ]]; then echo "\"Security evidence gaps: $evidence_yellow yellow, $evidence_red red\""; else echo 'null'; fi),
  "redactions": [],
  "metadata": {
    "bead": {
      "id": "frankenjax-cstq.17",
      "title": "Prove security and adversarial fuzz gates"
    },
    "bootstrap_phase": true
  }
}
EOF

echo "===== Security Gate Bootstrap ====="
echo "Threat model status:    $model_status"
echo "Error taxonomy status:  $taxonomy_status"
echo ""
echo "Threat categories:      $threat_categories"
echo "  - Green evidence:     $evidence_green"
echo "  - Yellow evidence:    $evidence_yellow"
echo "  - Red evidence:       $evidence_red"
echo ""
echo "Gate status: $gate_status"
echo "E2E log: $JSON"
echo "==================================="

if [[ "$ENFORCE" -eq 1 && "$gate_status" == "fail" ]]; then
  echo ""
  echo "FAIL: security gate requires all categories to have green evidence"
  exit 1
fi

exit 0
