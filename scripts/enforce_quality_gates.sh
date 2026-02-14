#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGETS_JSON="$ROOT_DIR/artifacts/ci/reliability_budgets.v1.json"
COVERAGE_RAW_JSON="$ROOT_DIR/artifacts/ci/coverage_raw_llvmcov.v1.json"
COVERAGE_REPORT_JSON="$ROOT_DIR/artifacts/ci/coverage_report.v1.json"
COVERAGE_TREND_JSON="$ROOT_DIR/artifacts/ci/coverage_trend.v1.json"
FLAKE_REPORT_JSON="$ROOT_DIR/artifacts/ci/flake_report.v1.json"
RUNTIME_REPORT_JSON="$ROOT_DIR/artifacts/ci/runtime_report.v1.json"
GATE_REPORT_JSON="$ROOT_DIR/artifacts/ci/reliability_gate_report.v1.json"

SKIP_COVERAGE=0
SKIP_FLAKE=0
SKIP_RUNTIME=0
FLAKE_RUNS_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/enforce_quality_gates.sh [options]

Options:
  --budgets <path>         Reliability budget JSON path.
  --skip-coverage          Skip coverage gate.
  --skip-flake             Skip flake gate.
  --skip-runtime           Skip runtime gate.
  --flake-runs <n>         Override flake runs (default from budgets).
  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budgets)
      BUDGETS_JSON="$2"
      shift 2
      ;;
    --skip-coverage)
      SKIP_COVERAGE=1
      shift
      ;;
    --skip-flake)
      SKIP_FLAKE=1
      shift
      ;;
    --skip-runtime)
      SKIP_RUNTIME=1
      shift
      ;;
    --flake-runs)
      FLAKE_RUNS_OVERRIDE="$2"
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

mkdir -p "$ROOT_DIR/artifacts/ci"

if [[ ! -f "$BUDGETS_JSON" ]]; then
  echo "error: budgets file not found at $BUDGETS_JSON" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

coverage_pass=true
flake_pass=true
runtime_pass=true

coverage_failures_tmp="$(mktemp)"
runtime_entries_tmp="$(mktemp)"
coverage_entries_tmp="$(mktemp)"
trap 'rm -f "$coverage_failures_tmp" "$runtime_entries_tmp" "$coverage_entries_tmp"' EXIT

start_pipeline_s="$(date +%s)"

if [[ $SKIP_COVERAGE -eq 0 ]]; then
  echo "[coverage] running cargo llvm-cov..."
  cargo llvm-cov --workspace --json --summary-only --output-path "$COVERAGE_RAW_JSON" >/dev/null

  mapfile -t crates < <(jq -r '.coverage | keys[]' "$BUDGETS_JSON")
  for crate in "${crates[@]}"; do
    line_floor="$(jq -r ".coverage[\"$crate\"].line_min_percent" "$BUDGETS_JSON")"
    branch_floor="$(jq -r ".coverage[\"$crate\"].branch_min_percent" "$BUDGETS_JSON")"

    line_count="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.lines.count] | add // 0" "$COVERAGE_RAW_JSON")"
    line_covered="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.lines.covered] | add // 0" "$COVERAGE_RAW_JSON")"

    branch_count="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.branches.count] | add // 0" "$COVERAGE_RAW_JSON")"
    branch_covered="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.branches.covered] | add // 0" "$COVERAGE_RAW_JSON")"

    region_count="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.regions.count] | add // 0" "$COVERAGE_RAW_JSON")"
    region_covered="$(jq "[.data[0].files[] | select(.filename|contains(\"/crates/$crate/\")) | .summary.regions.covered] | add // 0" "$COVERAGE_RAW_JSON")"

    line_percent="0"
    if [[ "$line_count" -gt 0 ]]; then
      line_percent="$(awk -v c="$line_covered" -v t="$line_count" 'BEGIN { printf "%.4f", 100.0 * c / t }')"
    fi

    branch_source="branches"
    branch_percent="0"
    if [[ "$branch_count" -gt 0 ]]; then
      branch_percent="$(awk -v c="$branch_covered" -v t="$branch_count" 'BEGIN { printf "%.4f", 100.0 * c / t }')"
    else
      branch_source="regions"
      if [[ "$region_count" -gt 0 ]]; then
        branch_percent="$(awk -v c="$region_covered" -v t="$region_count" 'BEGIN { printf "%.4f", 100.0 * c / t }')"
      fi
    fi

    line_ok="$(awk -v p="$line_percent" -v f="$line_floor" 'BEGIN { print (p + 0 >= f + 0) ? "true" : "false" }')"
    branch_ok="$(awk -v p="$branch_percent" -v f="$branch_floor" 'BEGIN { print (p + 0 >= f + 0) ? "true" : "false" }')"

    if [[ "$line_ok" != "true" || "$branch_ok" != "true" ]]; then
      coverage_pass=false
      printf '%s\n' "$crate" >>"$coverage_failures_tmp"
    fi

    jq -nc \
      --arg crate "$crate" \
      --argjson line_percent "$line_percent" \
      --argjson line_floor "$line_floor" \
      --argjson branch_percent "$branch_percent" \
      --argjson branch_floor "$branch_floor" \
      --arg branch_source "$branch_source" \
      --argjson line_ok "$line_ok" \
      --argjson branch_ok "$branch_ok" \
      '{
        crate: $crate,
        line_percent: $line_percent,
        line_floor: $line_floor,
        branch_percent: $branch_percent,
        branch_floor: $branch_floor,
        branch_source: $branch_source,
        line_ok: $line_ok,
        branch_ok: $branch_ok
      }' \
      >>"$coverage_entries_tmp"
  done

  jq -s \
    --arg schema_version "frankenjax.coverage-report.v1" \
    --arg raw_report_path "$COVERAGE_RAW_JSON" \
    --argjson passed "$coverage_pass" \
    '{
      schema_version: $schema_version,
      raw_report_path: $raw_report_path,
      passed: $passed,
      crates: .
    }' \
    "$coverage_entries_tmp" >"$COVERAGE_REPORT_JSON"

  if [[ -f "$COVERAGE_TREND_JSON" ]]; then
    for crate in "${crates[@]}"; do
      prev_line_percent="$(
        jq -r \
          --arg crate "$crate" \
          '.snapshots[-1].crates[]? | select(.crate == $crate) | .line_percent // empty' \
          "$COVERAGE_TREND_JSON"
      )"
      if [[ -z "$prev_line_percent" ]]; then
        continue
      fi

      curr_line_percent="$(
        jq -r \
          --arg crate "$crate" \
          '.crates[] | select(.crate == $crate) | .line_percent' \
          "$COVERAGE_REPORT_JSON"
      )"

      regressed="$(
        awk -v prev="$prev_line_percent" -v curr="$curr_line_percent" \
          'BEGIN { print ((prev - curr) > 2.0) ? "true" : "false" }'
      )"
      if [[ "$regressed" == "true" ]]; then
        coverage_pass=false
        echo "[coverage] regression >2% for $crate: prev=$prev_line_percent current=$curr_line_percent" >&2
      fi
    done
  fi

  snapshot_tmp="$(mktemp)"
  trend_tmp="$(mktemp)"
  jq -n \
    --argjson ts_unix_ms "$(date +%s%3N)" \
    --slurpfile report "$COVERAGE_REPORT_JSON" \
    '{
      ts_unix_ms: $ts_unix_ms,
      crates: $report[0].crates
    }' >"$snapshot_tmp"

  if [[ -s "$COVERAGE_TREND_JSON" ]] \
    && jq -e '.schema_version == "frankenjax.coverage-trend.v1" and (.snapshots | type == "array")' \
      "$COVERAGE_TREND_JSON" >/dev/null 2>&1; then
    jq --slurpfile snap "$snapshot_tmp" '.snapshots += $snap' "$COVERAGE_TREND_JSON" >"$trend_tmp"
  else
    jq --slurpfile snap "$snapshot_tmp" \
      '{schema_version: "frankenjax.coverage-trend.v1", snapshots: $snap}' >"$trend_tmp"
  fi

  mv "$trend_tmp" "$COVERAGE_TREND_JSON"
  rm -f "$snapshot_tmp"

  if [[ "$coverage_pass" != "true" ]]; then
    echo "[coverage] failed floors for crate(s): $(paste -sd, "$coverage_failures_tmp")" >&2
  else
    echo "[coverage] all crate floors satisfied"
  fi
else
  echo "[coverage] skipped"
fi

if [[ $SKIP_FLAKE -eq 0 ]]; then
  flake_runs="$(jq -r '.flake.runs' "$BUDGETS_JSON")"
  if [[ -n "$FLAKE_RUNS_OVERRIDE" ]]; then
    flake_runs="$FLAKE_RUNS_OVERRIDE"
  fi
  max_suite_flake_rate="$(jq -r '.flake.max_suite_flake_rate' "$BUDGETS_JSON")"

  echo "[flake] running detector for ${flake_runs} runs..."
  set +e
  "$ROOT_DIR/scripts/detect_flakes.sh" \
    --runs "$flake_runs" \
    --out "$FLAKE_REPORT_JSON" \
    --log-dir "$ROOT_DIR/artifacts/ci/flake-runs"
  detector_rc=$?
  set -e

  if [[ ! -f "$FLAKE_REPORT_JSON" ]]; then
    echo "[flake] missing flake report" >&2
    flake_pass=false
  else
    suite_flake_rate="$(jq -r '.suite_flake_rate' "$FLAKE_REPORT_JSON")"
    rate_ok="$(awk -v p="$suite_flake_rate" -v f="$max_suite_flake_rate" 'BEGIN { print (p + 0 <= f + 0) ? "true" : "false" }')"
    if [[ $detector_rc -ne 0 || "$rate_ok" != "true" ]]; then
      flake_pass=false
      echo "[flake] gate failed (rate=${suite_flake_rate}, max=${max_suite_flake_rate})" >&2
    else
      echo "[flake] gate passed"
    fi
  fi
else
  echo "[flake] skipped"
fi

if [[ $SKIP_RUNTIME -eq 0 ]]; then
  RUNTIME_LOG_DIR="$ROOT_DIR/artifacts/ci/runtime-logs"
  mkdir -p "$RUNTIME_LOG_DIR"

  run_runtime_check() {
    local key="$1"
    local command="$2"
    local budget
    budget="$(jq -r ".runtime_seconds[\"$key\"]" "$BUDGETS_JSON")"

    local log_file="$RUNTIME_LOG_DIR/${key}.log"
    local start_s end_s duration_s rc status

    start_s="$(date +%s)"
    set +e
    bash -lc "cd '$ROOT_DIR' && $command" >"$log_file" 2>&1
    rc=$?
    set -e
    end_s="$(date +%s)"
    duration_s=$((end_s - start_s))

    status="pass"
    if [[ $rc -ne 0 || $duration_s -gt $budget ]]; then
      status="fail"
      runtime_pass=false
    fi

    jq -nc \
      --arg key "$key" \
      --arg command "$command" \
      --argjson budget_seconds "$budget" \
      --argjson duration_seconds "$duration_s" \
      --argjson exit_code "$rc" \
      --arg status "$status" \
      --arg log_file "$log_file" \
      '{
        key: $key,
        command: $command,
        budget_seconds: $budget_seconds,
        duration_seconds: $duration_seconds,
        exit_code: $exit_code,
        status: $status,
        log_file: $log_file
      }' \
      >>"$runtime_entries_tmp"

    echo "[runtime] $key => $status (${duration_s}s <= ${budget}s?)"
  }

  run_runtime_check "unit_tests_workspace" "cargo test --workspace --quiet"
  run_runtime_check "property_tests" "cargo test --workspace --quiet"
  run_runtime_check "differential_oracle" "cargo test -p fj-conformance --test transforms -- --nocapture"
  run_runtime_check "e2e_scenarios" "./scripts/run_e2e.sh --packet P2C-001"

  jq -s \
    --arg schema_version "frankenjax.runtime-report.v1" \
    --argjson passed "$runtime_pass" \
    '{
      schema_version: $schema_version,
      passed: $passed,
      checks: .
    }' \
    "$runtime_entries_tmp" >"$RUNTIME_REPORT_JSON"
else
  echo "[runtime] skipped"
fi

end_pipeline_s="$(date +%s)"
pipeline_duration_s=$((end_pipeline_s - start_pipeline_s))
full_budget_s="$(jq -r '.runtime_seconds.full_pipeline' "$BUDGETS_JSON")"
full_pipeline_ok="$(awk -v p="$pipeline_duration_s" -v f="$full_budget_s" 'BEGIN { print (p + 0 <= f + 0) ? "true" : "false" }')"

if [[ "$full_pipeline_ok" != "true" ]]; then
  runtime_pass=false
  echo "[runtime] full pipeline budget exceeded (${pipeline_duration_s}s > ${full_budget_s}s)" >&2
fi

overall_pass=true
if [[ $SKIP_COVERAGE -eq 0 && "$coverage_pass" != "true" ]]; then
  overall_pass=false
fi
if [[ $SKIP_FLAKE -eq 0 && "$flake_pass" != "true" ]]; then
  overall_pass=false
fi
if [[ $SKIP_RUNTIME -eq 0 && "$runtime_pass" != "true" ]]; then
  overall_pass=false
fi

generated_at_ms="$(date +%s%3N)"

jq -n \
  --arg schema_version "frankenjax.reliability-gate-report.v1" \
  --arg budgets_path "$BUDGETS_JSON" \
  --arg coverage_report "$COVERAGE_REPORT_JSON" \
  --arg flake_report "$FLAKE_REPORT_JSON" \
  --arg runtime_report "$RUNTIME_REPORT_JSON" \
  --argjson coverage_enabled "$((1 - SKIP_COVERAGE))" \
  --argjson flake_enabled "$((1 - SKIP_FLAKE))" \
  --argjson runtime_enabled "$((1 - SKIP_RUNTIME))" \
  --argjson coverage_pass "$coverage_pass" \
  --argjson flake_pass "$flake_pass" \
  --argjson runtime_pass "$runtime_pass" \
  --argjson full_pipeline_duration_seconds "$pipeline_duration_s" \
  --argjson full_pipeline_budget_seconds "$full_budget_s" \
  --argjson generated_at_unix_ms "$generated_at_ms" \
  --argjson overall_passed "$overall_pass" \
  '{
    schema_version: $schema_version,
    generated_at_unix_ms: $generated_at_unix_ms,
    budgets_path: $budgets_path,
    coverage: {
      enabled: ($coverage_enabled == 1),
      passed: $coverage_pass,
      report: $coverage_report
    },
    flake: {
      enabled: ($flake_enabled == 1),
      passed: $flake_pass,
      report: $flake_report
    },
    runtime: {
      enabled: ($runtime_enabled == 1),
      passed: $runtime_pass,
      report: $runtime_report,
      full_pipeline_duration_seconds: $full_pipeline_duration_seconds,
      full_pipeline_budget_seconds: $full_pipeline_budget_seconds
    },
    overall_passed: $overall_passed
  }' >"$GATE_REPORT_JSON"

echo "Reliability gate report written: $GATE_REPORT_JSON"

if [[ "$overall_pass" != "true" ]]; then
  exit 1
fi
