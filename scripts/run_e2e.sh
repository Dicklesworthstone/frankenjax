#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${FJ_E2E_ARTIFACT_DIR:-$ROOT_DIR/artifacts/e2e}"
CURRENT_FORENSIC_SCHEMA="frankenjax.e2e-forensic-log.v1"
BEAD_ID="${FJ_E2E_BEAD_ID:-unassigned}"
PACKET_FILTER=""
SCENARIO_FILTER=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_e2e.sh [--packet P2C-001] [--scenario e2e_p2c001_*]

Options:
  --packet <P2C-###>     Run only scenarios associated with one packet.
  --scenario <id>        Run only one exact scenario id.
  -h, --help             Show this help.

Examples:
  ./scripts/run_e2e.sh
  ./scripts/run_e2e.sh --packet P2C-001
  ./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline
USAGE
}

normalize_packet() {
  local raw="$1"
  local upper
  upper="$(printf '%s' "$raw" | tr '[:lower:]' '[:upper:]')"
  upper="${upper//_/\-}"
  if [[ "$upper" =~ ^P2C-?([0-9]{3})$ ]]; then
    printf 'P2C-%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf '%s' "$upper"
}

packet_from_name() {
  local name="$1"
  if [[ "$name" =~ p2c([0-9]{3}) ]]; then
    printf 'P2C-%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf 'UNSPECIFIED'
}

needs_fallback_log() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    return 0
  fi
  ! grep -Eq "\"schema_version\"[[:space:]]*:[[:space:]]*\"$CURRENT_FORENSIC_SCHEMA\"" "$log_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --packet)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --packet requires a value" >&2
        exit 2
      fi
      PACKET_FILTER="$(normalize_packet "$2")"
      shift 2
      ;;
    --scenario)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "error: --scenario requires a value" >&2
        exit 2
      fi
      SCENARIO_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$ARTIFACT_DIR"

mapfile -t E2E_FILES < <(find "$ROOT_DIR/crates" -type f -path '*/tests/e2e*.rs' | sort)
if [[ ${#E2E_FILES[@]} -eq 0 ]]; then
  echo "No e2e test binaries discovered (expected files matching crates/*/tests/e2e*.rs)." >&2
  exit 1
fi

SCENARIOS=()
for test_file in "${E2E_FILES[@]}"; do
  crate_dir="$(dirname "$(dirname "$test_file")")"
  pkg="$(basename "$crate_dir")"
  test_bin="$(basename "$test_file" .rs)"

  mapfile -t discovered < <(
    cargo test -p "$pkg" --test "$test_bin" -- --list 2>/dev/null \
      | awk -F': test' '/^e2e_[A-Za-z0-9_]+: test$/ {print $1}'
  )

  for scenario in "${discovered[@]}"; do
    packet="$(packet_from_name "$scenario")"
    if [[ "$packet" == "UNSPECIFIED" ]]; then
      packet="$(packet_from_name "$test_bin")"
    fi
    SCENARIOS+=("$pkg|$test_bin|$scenario|$packet")
  done
done

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  echo "No e2e scenarios discovered from e2e test binaries." >&2
  exit 1
fi

SELECTED=()
for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r pkg test_bin scenario packet <<<"$entry"

  if [[ -n "$PACKET_FILTER" && "$packet" != "$PACKET_FILTER" ]]; then
    continue
  fi
  if [[ -n "$SCENARIO_FILTER" && "$scenario" != "$SCENARIO_FILTER" ]]; then
    continue
  fi

  SELECTED+=("$entry")
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "No scenarios matched filters (packet='${PACKET_FILTER:-*}', scenario='${SCENARIO_FILTER:-*}')." >&2
  exit 1
fi

PASS_COUNT=0
FAIL_COUNT=0
SUMMARY_ROWS=()

printf 'Running %d E2E scenario(s)...\n' "${#SELECTED[@]}"

for entry in "${SELECTED[@]}"; do
  IFS='|' read -r pkg test_bin scenario packet <<<"$entry"

  stdout_log="$ARTIFACT_DIR/${scenario}.stdout.log"
  forensic_log="$ARTIFACT_DIR/${scenario}.e2e.json"
  replay_cmd="cargo test -p $pkg --test $test_bin -- $scenario --exact --nocapture"

  start_ms="$(date +%s%3N)"
  set +e
  FJ_E2E_ARTIFACT_DIR="$ARTIFACT_DIR" \
    cargo test -p "$pkg" --test "$test_bin" -- "$scenario" --exact --nocapture \
    >"$stdout_log" 2>&1
  rc=$?
  set -e
  end_ms="$(date +%s%3N)"
  duration_ms=$((end_ms - start_ms))

  if needs_fallback_log "$forensic_log"; then
    stdout_hash="$(sha256sum "$stdout_log" | awk '{print $1}')"
    status="$( [[ $rc -eq 0 ]] && echo pass || echo fail )"
    failure_summary_json="null"
    if [[ $rc -ne 0 ]]; then
      failure_summary_json="\"cargo test exited $rc; see stdout log\""
    fi
    cat >"$forensic_log" <<JSON
{
  "schema_version": "$CURRENT_FORENSIC_SCHEMA",
  "bead_id": "$BEAD_ID",
  "scenario_id": "$scenario",
  "test_id": "$test_bin::$scenario",
  "packet_id": "$packet",
  "command": ["cargo", "test", "-p", "$pkg", "--test", "$test_bin", "--", "$scenario", "--exact", "--nocapture"],
  "working_dir": "$ROOT_DIR",
  "environment": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "rust_version": "$(rustc --version 2>/dev/null || printf 'rustc <unknown>')",
    "cargo_version": "$(cargo --version 2>/dev/null || printf 'cargo <unknown>')",
    "cargo_target_dir": "${CARGO_TARGET_DIR:-<default>}",
    "env_vars": {},
    "timestamp_unix_ms": $end_ms
  },
  "feature_flags": [],
  "fixture_ids": [],
  "oracle_ids": [],
  "transform_stack": [],
  "mode": "strict",
  "inputs": {"scenario": "$scenario", "packet": "$packet"},
  "expected": {"exit_code": 0},
  "actual": {"exit_code": $rc},
  "tolerance": {"policy_id": "exact_exit_code", "atol": null, "rtol": null, "ulp": null, "notes": "fallback log emitted by scripts/run_e2e.sh"},
  "error": {"expected": null, "actual": null, "taxonomy_class": "none"},
  "timings": {"setup_ms": 0, "trace_ms": 0, "dispatch_ms": 0, "eval_ms": 0, "verify_ms": $duration_ms, "total_ms": $duration_ms},
  "allocations": {"allocation_count": null, "allocated_bytes": null, "peak_rss_bytes": null, "measurement_backend": "not_measured"},
  "artifacts": [{"kind": "stdout_log", "path": "$stdout_log", "sha256": "$stdout_hash", "required": true}],
  "replay_command": "$replay_cmd",
  "status": "$status",
  "failure_summary": $failure_summary_json,
  "redactions": [],
  "metadata": {
    "emitter": "scripts/run_e2e.sh",
    "fallback": true,
    "fallback_reason": "scenario did not emit a current-schema forensic log",
    "package": "$pkg",
    "test_binary": "$test_bin"
  }
}
JSON
  fi

  if [[ $rc -eq 0 ]]; then
    ((PASS_COUNT += 1))
    SUMMARY_ROWS+=("PASS | $packet | $scenario | ${duration_ms}ms")
    printf '[PASS] %s (%s)\n' "$scenario" "$packet"
  else
    ((FAIL_COUNT += 1))
    SUMMARY_ROWS+=("FAIL | $packet | $scenario | ${duration_ms}ms | replay: $replay_cmd")
    printf '[FAIL] %s (%s)\n' "$scenario" "$packet"
    printf '  replay: %s\n' "$replay_cmd"
    printf '  stdout: %s\n' "$stdout_log"
  fi
done

TOTAL_COUNT=$((PASS_COUNT + FAIL_COUNT))
printf '\nE2E Summary\n'
printf '  total: %d\n' "$TOTAL_COUNT"
printf '  pass:  %d\n' "$PASS_COUNT"
printf '  fail:  %d\n' "$FAIL_COUNT"
printf '  logs:  %s\n' "$ARTIFACT_DIR"

for row in "${SUMMARY_ROWS[@]}"; do
  printf '  %s\n' "$row"
done

if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
