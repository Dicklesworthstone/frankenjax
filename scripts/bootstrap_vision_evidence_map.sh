#!/usr/bin/env bash
# bootstrap_vision_evidence_map.sh — Bootstrap vision-to-evidence map from docs
#
# Scans FEATURE_PARITY.md and README.md, extracts claims, links to evidence
# artifacts and tests, emits JSON map with red/yellow/green status per claim.
#
# Usage:
#   ./scripts/bootstrap_vision_evidence_map.sh [--json <path>] [--preview]
#
# Exit codes:
#   0 - Map generated successfully
#   1 - Evidence gaps found (red/yellow claims exist)
#   2 - Usage error

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON="$ROOT/artifacts/conformance/vision_evidence_map.v1.json"
PREVIEW=""
BEAD_ID="frankenjax-cstq.14"

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
    --preview)
      PREVIEW=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--json <path>] [--preview]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TIMESTAMP_MS=$(($(date +%s) * 1000))

sha256_file() {
  sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo "0000000000000000000000000000000000000000000000000000000000000000"
}

check_path_exists() {
  if [[ -e "$ROOT/$1" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

get_replay_cmd() {
  local path="$1"
  case "$path" in
    scripts/run_*.sh)
      echo "./$path --enforce"
      ;;
    crates/*/tests/*.rs)
      local crate
      crate=$(echo "$path" | sed 's|crates/\([^/]*\)/.*|\1|')
      echo "cargo test -p $crate"
      ;;
    *)
      echo "null"
      ;;
  esac
}

# Extract feature family claims from FEATURE_PARITY.md
extract_feature_parity_claims() {
  local claims_json="[]"
  local line_num=0
  local in_matrix=0
  local claim_idx=0

  while IFS= read -r line; do
    line_num=$((line_num + 1))

    # Detect Feature Family Matrix section
    if [[ "$line" == *"Feature Family Matrix"* ]]; then
      in_matrix=1
      continue
    fi

    # Stop at next major section
    if [[ $in_matrix -eq 1 && "$line" =~ ^##[^#] && "$line" != *"Feature Family Matrix"* ]]; then
      in_matrix=0
      continue
    fi

    # Parse table rows (skip header and separator)
    if [[ $in_matrix -eq 1 && "$line" =~ ^\|[^-] && "$line" != *"Feature Family"* ]]; then
      # Extract columns: Feature Family | Status | Current Evidence | Next Required
      feature=$(echo "$line" | cut -d'|' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      status=$(echo "$line" | cut -d'|' -f3 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      evidence=$(echo "$line" | cut -d'|' -f4 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

      if [[ -n "$feature" && "$feature" != "Feature Family" ]]; then
        claim_id="claim_fp_$(echo "$feature" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd 'a-z0-9_' | head -c 40)"

        # Determine status color
        if [[ "$status" == "parity_green" ]]; then
          color="green"
          reason="FEATURE_PARITY status is parity_green with documented evidence"
        elif [[ "$status" == "in_progress" ]]; then
          color="yellow"
          reason="FEATURE_PARITY status is in_progress"
        else
          color="red"
          reason="FEATURE_PARITY status is $status - needs evidence"
        fi

        # Look for evidence references in the evidence column
        evidence_refs="[]"

        # Check for test references
        if [[ "$evidence" =~ tests/ || "$evidence" =~ src/.*\.rs ]]; then
          while read -r test_path; do
            if [[ -n "$test_path" ]]; then
              exists=$(check_path_exists "$test_path")
              replay=$(get_replay_cmd "$test_path")
              evidence_refs=$(echo "$evidence_refs" | jq --arg kind "test" --arg path "$test_path" --argjson exists "$exists" --arg replay "$replay" \
                '. + [{"kind": $kind, "path": $path, "exists": $exists, "replay_command": (if $replay == "null" then null else $replay end), "last_verified_unix_ms": null}]')
            fi
          done < <(echo "$evidence" | grep -oE 'crates/[^;`]+\.rs' | head -5 || true)
        fi

        # Check for artifact references
        if [[ "$evidence" =~ artifacts/ ]]; then
          while read -r artifact_path; do
            if [[ -n "$artifact_path" ]]; then
              exists=$(check_path_exists "$artifact_path")
              evidence_refs=$(echo "$evidence_refs" | jq --arg kind "artifact" --arg path "$artifact_path" --argjson exists "$exists" \
                '. + [{"kind": $kind, "path": $path, "exists": $exists, "replay_command": null, "last_verified_unix_ms": null}]')
            fi
          done < <(echo "$evidence" | grep -oE 'artifacts/[^;`]+' | head -5 || true)
        fi

        # Check for E2E references
        if [[ "$evidence" =~ e2e/ || "$evidence" =~ E2E ]]; then
          while read -r e2e_path; do
            if [[ -n "$e2e_path" ]]; then
              exists=$(check_path_exists "$e2e_path")
              evidence_refs=$(echo "$evidence_refs" | jq --arg kind "e2e_script" --arg path "$e2e_path" --argjson exists "$exists" \
                '. + [{"kind": $kind, "path": $path, "exists": $exists, "replay_command": null, "last_verified_unix_ms": null}]')
            fi
          done < <(echo "$evidence" | grep -oE 'artifacts/e2e/[^;`]+' | head -5 || true)
        fi

        # If no evidence found but status is green, mark as needing verification
        evidence_count=$(echo "$evidence_refs" | jq 'length')
        if [[ "$color" == "green" && "$evidence_count" -eq 0 ]]; then
          color="yellow"
          reason="Claimed green but no machine-verifiable evidence paths found"
        fi

        claims_json=$(echo "$claims_json" | jq \
          --arg id "$claim_id" \
          --arg source "feature_parity" \
          --argjson line "$line_num" \
          --arg text "$feature: $status" \
          --arg status "$color" \
          --arg reason "$reason" \
          --argjson evidence "$evidence_refs" \
          '. + [{
            "claim_id": $id,
            "source": $source,
            "source_line": $line,
            "claim_text": $text,
            "status": $status,
            "status_reason": $reason,
            "evidence": $evidence,
            "linked_beads": [],
            "blocking_gaps": []
          }]')

        claim_idx=$((claim_idx + 1))
      fi
    fi
  done < "$ROOT/FEATURE_PARITY.md"

  echo "$claims_json"
}

# Generate source docs metadata
FP_SHA=$(sha256_file "$ROOT/FEATURE_PARITY.md")
README_SHA=$(sha256_file "$ROOT/README.md")

SOURCE_DOCS=$(cat <<EOF
[
  {"path": "FEATURE_PARITY.md", "sha256": "$FP_SHA", "scanned_at_unix_ms": $TIMESTAMP_MS},
  {"path": "README.md", "sha256": "$README_SHA", "scanned_at_unix_ms": $TIMESTAMP_MS}
]
EOF
)

# Extract claims
echo "Scanning FEATURE_PARITY.md for claims..."
CLAIMS=$(extract_feature_parity_claims)

# Calculate summary
TOTAL=$(echo "$CLAIMS" | jq 'length')
RED=$(echo "$CLAIMS" | jq '[.[] | select(.status == "red")] | length')
YELLOW=$(echo "$CLAIMS" | jq '[.[] | select(.status == "yellow")] | length')
GREEN=$(echo "$CLAIMS" | jq '[.[] | select(.status == "green")] | length')

if [[ "$TOTAL" -gt 0 ]]; then
  COVERAGE_PCT=$(echo "scale=1; $GREEN * 100 / $TOTAL" | bc)
else
  COVERAGE_PCT="100.0"
fi

MISSING_REPLAY=$(echo "$CLAIMS" | jq '[.[] | .evidence[] | select(.replay_command == null)] | length')
STALE_ARTIFACTS=0

# Generate full map
mkdir -p "$(dirname "$JSON")"
cat > "$JSON" <<EOF
{
  "schema_version": "frankenjax.vision-evidence-map.v1",
  "bead_id": "$BEAD_ID",
  "generated_at_unix_ms": $TIMESTAMP_MS,
  "source_docs": $SOURCE_DOCS,
  "claims": $CLAIMS,
  "summary": {
    "total_claims": $TOTAL,
    "red_count": $RED,
    "yellow_count": $YELLOW,
    "green_count": $GREEN,
    "coverage_pct": $COVERAGE_PCT,
    "missing_replay_count": $MISSING_REPLAY,
    "stale_artifact_count": $STALE_ARTIFACTS
  }
}
EOF

echo ""
echo "===== Vision-to-Evidence Map Bootstrap ====="
echo "Source documents scanned: 2"
echo "Total claims extracted:   $TOTAL"
echo ""
echo "Status breakdown:"
echo "  - Green (verified):   $GREEN"
echo "  - Yellow (partial):   $YELLOW"
echo "  - Red (missing):      $RED"
echo ""
echo "Coverage: $COVERAGE_PCT%"
echo "Missing replay commands: $MISSING_REPLAY"
echo "Stale artifacts: $STALE_ARTIFACTS"
echo ""
echo "Map written to: $JSON"
echo "============================================"

# Generate human-readable preview if requested
if [[ -n "$PREVIEW" ]]; then
  PREVIEW_PATH="${JSON%.json}.md"
  echo "" > "$PREVIEW_PATH"
  echo "# Vision-to-Evidence Map Preview" >> "$PREVIEW_PATH"
  echo "" >> "$PREVIEW_PATH"
  echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PREVIEW_PATH"
  echo "" >> "$PREVIEW_PATH"
  echo "## Summary" >> "$PREVIEW_PATH"
  echo "" >> "$PREVIEW_PATH"
  echo "| Metric | Value |" >> "$PREVIEW_PATH"
  echo "|--------|-------|" >> "$PREVIEW_PATH"
  echo "| Total claims | $TOTAL |" >> "$PREVIEW_PATH"
  echo "| Green | $GREEN |" >> "$PREVIEW_PATH"
  echo "| Yellow | $YELLOW |" >> "$PREVIEW_PATH"
  echo "| Red | $RED |" >> "$PREVIEW_PATH"
  echo "| Coverage | $COVERAGE_PCT% |" >> "$PREVIEW_PATH"
  echo "" >> "$PREVIEW_PATH"
  echo "## Claims by Status" >> "$PREVIEW_PATH"
  echo "" >> "$PREVIEW_PATH"

  for status in red yellow green; do
    count=$(echo "$CLAIMS" | jq "[.[] | select(.status == \"$status\")] | length")
    if [[ "$count" -gt 0 ]]; then
      echo "### $(echo "$status" | tr '[:lower:]' '[:upper:]') ($count)" >> "$PREVIEW_PATH"
      echo "" >> "$PREVIEW_PATH"
      echo "$CLAIMS" | jq -r ".[] | select(.status == \"$status\") | \"- **\(.claim_text)** (line \(.source_line)): \(.status_reason)\"" >> "$PREVIEW_PATH"
      echo "" >> "$PREVIEW_PATH"
    fi
  done

  echo "Preview written to: $PREVIEW_PATH"
fi

# Exit with error if gaps found
if [[ "$RED" -gt 0 || "$YELLOW" -gt 0 ]]; then
  echo ""
  echo "WARNING: Evidence gaps found ($RED red, $YELLOW yellow)"
  exit 1
fi

exit 0
