#!/usr/bin/env bash
set -euo pipefail

# FrankenJAX Performance Regression Gate
#
# Compares criterion benchmark results between a saved baseline and the current
# working tree. Fails if any benchmark's p95 regresses beyond the threshold
# (default 5%) unless a risk-note justification exists.
#
# Usage:
#   ./scripts/check_perf_regression.sh                     # full run
#   ./scripts/check_perf_regression.sh --save-baseline      # save current as baseline
#   ./scripts/check_perf_regression.sh --threshold 10       # custom threshold
#   ./scripts/check_perf_regression.sh --skip-run           # compare existing data only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUDGETS_JSON="$ROOT_DIR/artifacts/ci/reliability_budgets.v1.json"
PERF_REPORT_JSON="$ROOT_DIR/artifacts/ci/perf_regression_report.v1.json"
CRITERION_TARGET="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"
CRITERION_DIR="$CRITERION_TARGET/criterion"

SAVE_BASELINE=0
SKIP_RUN=0
VERIFY_REGIMES=0
THRESHOLD_OVERRIDE=""
BASELINE_NAME=""
BENCH_FILTER=""
RISK_NOTES_DIR="$ROOT_DIR/artifacts/performance/risk_notes"
GLOBAL_PERF_GATE_JSON="$ROOT_DIR/artifacts/performance/global_performance_gate.v1.json"

usage() {
  cat <<'USAGE'
Usage: ./scripts/check_perf_regression.sh [options]

Options:
  --save-baseline          Save current results as the baseline (no comparison).
  --baseline-name <name>   Baseline save name (default from budgets).
  --threshold <percent>    Override max p95 regression percent.
  --bench-filter <filter>  Only compare benchmarks matching this substring.
  --skip-run               Skip running benchmarks; compare existing data.
  --verify-regimes         Verify PrecisionPolicy (BitExactPortable vs HardwareFmaAccelerated)
                           and Stockham batched FFT regimes + vs-JAX 0.10.2 scoreboard.
  --budgets <path>         Reliability budget JSON path.
  -h, --help               Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-baseline)
      SAVE_BASELINE=1
      shift
      ;;
    --baseline-name)
      BASELINE_NAME="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD_OVERRIDE="$2"
      shift 2
      ;;
    --bench-filter)
      BENCH_FILTER="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --verify-regimes)
      VERIFY_REGIMES=1
      shift
      ;;
    --budgets)
      BUDGETS_JSON="$2"
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

if [[ ! -f "$BUDGETS_JSON" ]]; then
  echo "error: budgets file not found at $BUDGETS_JSON" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

if [[ $VERIFY_REGIMES -eq 1 ]]; then
  echo "[perf] verifying global_performance_gate.v1.json PrecisionPolicy regimes and vs-JAX scoreboard..."
  if [[ ! -f "$GLOBAL_PERF_GATE_JSON" ]]; then
    echo "error: global performance gate not found at $GLOBAL_PERF_GATE_JSON" >&2
    exit 1
  fi
  jq -e '
    .precision_policy_regimes.default_policy == "BitExactPortable" and
    .precision_policy_regimes.accelerated_policy == "HardwareFmaAccelerated" and
    (.precision_policy_regimes.gemm_cholesky_exp | length >= 4) and
    ([.precision_policy_regimes.gemm_cholesky_exp[] | select(.speedup_x >= .required_min_speedup_x and .status == "pass")] | length) == (.precision_policy_regimes.gemm_cholesky_exp | length) and
    (.precision_policy_regimes.stockham_batched_fft | length >= 3) and
    ([.precision_policy_regimes.stockham_batched_fft[] | select(.portable_speedup_vs_bluestein_x >= .required_min_speedup_x and .fma_speedup_vs_bluestein_x >= .required_min_speedup_x and .status == "pass")] | length) == (.precision_policy_regimes.stockham_batched_fft | length) and
    (.vs_jax_scoreboard.families | length >= 10)
  ' "$GLOBAL_PERF_GATE_JSON" >/dev/null
  if [[ $SKIP_RUN -eq 0 ]]; then
    echo "[perf] running live PrecisionPolicy and Stockham FFT regime benchmark suite..."
    cargo test --release -p frankenjax-lax --lib bench_rc26_18_all_regimes -- --ignored --nocapture
  fi
  echo "PASS: PrecisionPolicy regimes and vs-JAX 0.10.2 scoreboard verified"
  exit 0
fi

# Read budget defaults
threshold="$(jq -r '.perf_regression.max_p95_regression_percent // 5.0' "$BUDGETS_JSON")"
if [[ -n "$THRESHOLD_OVERRIDE" ]]; then
  threshold="$THRESHOLD_OVERRIDE"
fi

baseline_name="${BASELINE_NAME:-$(jq -r '.perf_regression.baseline_save_name // "main"' "$BUDGETS_JSON")}"
budget_bench_filter="$(jq -r '.perf_regression.bench_filter // ""' "$BUDGETS_JSON")"
if [[ -z "$BENCH_FILTER" ]]; then
  BENCH_FILTER="$budget_bench_filter"
fi

# -----------------------------------------------------------------------
# Save baseline mode
# -----------------------------------------------------------------------
if [[ $SAVE_BASELINE -eq 1 ]]; then
  echo "[perf] saving baseline as '$baseline_name'..."
  if [[ $SKIP_RUN -eq 0 ]]; then
    cargo bench --bench dispatch_baseline -- --save-baseline "$baseline_name"
  else
    echo "[perf] --skip-run: assuming criterion data already present"
    # Copy current results as baseline
    if [[ -d "$CRITERION_DIR" ]]; then
      for bench_dir in "$CRITERION_DIR"/*/; do
        if [[ -d "$bench_dir/new" ]]; then
          mkdir -p "$bench_dir/$baseline_name"
          cp -r "$bench_dir/new/"* "$bench_dir/$baseline_name/"
        fi
      done
    fi
  fi
  echo "[perf] baseline '$baseline_name' saved"
  exit 0
fi

# -----------------------------------------------------------------------
# Comparison mode
# -----------------------------------------------------------------------
echo "[perf] regression threshold: ${threshold}%"
echo "[perf] baseline: $baseline_name"

if [[ $SKIP_RUN -eq 0 ]]; then
  echo "[perf] running benchmarks..."
  cargo bench --bench dispatch_baseline -- --baseline "$baseline_name" 2>&1 | tee "$ROOT_DIR/artifacts/ci/bench_output.log"
fi

# -----------------------------------------------------------------------
# Parse criterion JSON estimates
# -----------------------------------------------------------------------
generated_at_ms="$(date +%s%3N)"
candidate_id="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
baseline_id="$baseline_name"

benchmarks_tmp="$(mktemp)"
regressions_tmp="$(mktemp)"
trap 'rm -f "$benchmarks_tmp" "$regressions_tmp"' EXIT

overall_pass=true

if [[ ! -d "$CRITERION_DIR" ]]; then
  echo "error: criterion directory not found at $CRITERION_DIR" >&2
  echo "       run benchmarks first or use --skip-run with existing data" >&2
  exit 1
fi

# Walk criterion output directories
find "$CRITERION_DIR" -name "estimates.json" -path "*/new/*" | sort | while read -r est_file; do
  # Extract bench group and function from path
  # Pattern: criterion/<group>/<function>/new/estimates.json
  rel_path="${est_file#"$CRITERION_DIR"/}"
  bench_group="$(echo "$rel_path" | cut -d/ -f1)"
  bench_func="$(echo "$rel_path" | cut -d/ -f2)"
  bench_id="${bench_group}/${bench_func}"

  # Apply filter
  if [[ -n "$BENCH_FILTER" && "$bench_id" != *"$BENCH_FILTER"* ]]; then
    continue
  fi

  # Read candidate p95 (upper confidence bound from slope or mean)
  candidate_p95_ns="$(jq '.slope.confidence_interval.upper_bound // .mean.confidence_interval.upper_bound // 0' "$est_file")"

  # Read baseline p95
  baseline_est_file="${est_file/\/new\//\/$baseline_name\/}"
  if [[ ! -f "$baseline_est_file" ]]; then
    # No baseline — treat as new benchmark, always pass
    jq -nc \
      --arg bench_id "$bench_id" \
      --arg group "$bench_group" \
      --argjson baseline_p95_ns 0 \
      --argjson candidate_p95_ns "$candidate_p95_ns" \
      --argjson delta_percent 0 \
      --arg status "pass" \
      '{
        bench_id: $bench_id,
        group: $group,
        baseline_p95_ns: $baseline_p95_ns,
        candidate_p95_ns: $candidate_p95_ns,
        delta_percent: $delta_percent,
        status: $status
      }' >>"$benchmarks_tmp"
    continue
  fi

  baseline_p95_ns="$(jq '.slope.confidence_interval.upper_bound // .mean.confidence_interval.upper_bound // 0' "$baseline_est_file")"

  # Compute delta percent
  if [[ "$(awk -v b="$baseline_p95_ns" 'BEGIN { print (b > 0) ? "yes" : "no" }')" == "yes" ]]; then
    delta_percent="$(awk -v c="$candidate_p95_ns" -v b="$baseline_p95_ns" 'BEGIN { printf "%.4f", (c - b) / b * 100.0 }')"
  else
    delta_percent="0"
  fi

  # Determine status
  is_regressed="$(awk -v d="$delta_percent" -v t="$threshold" 'BEGIN { print (d > t) ? "yes" : "no" }')"
  is_improved="$(awk -v d="$delta_percent" 'BEGIN { print (d < -2.0) ? "yes" : "no" }')"

  status="pass"
  risk_note_ref=""

  if [[ "$is_regressed" == "yes" ]]; then
    # Check for risk note
    safe_bench_id="${bench_id//\//_}"
    risk_note_path="$RISK_NOTES_DIR/${safe_bench_id}.risk_note.json"
    if [[ -f "$risk_note_path" ]]; then
      status="risk_noted"
      risk_note_ref="$risk_note_path"
    else
      status="regressed"
      overall_pass=false
    fi

    justified="false"
    if [[ -n "$risk_note_ref" ]]; then
      justified="true"
    fi

    # Record regression
    if [[ -n "$risk_note_ref" ]]; then
      jq -nc \
        --arg bench_id "$bench_id" \
        --argjson delta_percent "$delta_percent" \
        --argjson justified "$justified" \
        --arg risk_note_ref "$risk_note_ref" \
        '{
          bench_id: $bench_id,
          delta_percent: $delta_percent,
          justified: $justified,
          risk_note_ref: $risk_note_ref
        }' >>"$regressions_tmp"
    else
      jq -nc \
        --arg bench_id "$bench_id" \
        --argjson delta_percent "$delta_percent" \
        --argjson justified "$justified" \
        '{
          bench_id: $bench_id,
          delta_percent: $delta_percent,
          justified: $justified
        }' >>"$regressions_tmp"
    fi
  elif [[ "$is_improved" == "yes" ]]; then
    status="improved"
  fi

  # Record benchmark entry
  if [[ -n "$risk_note_ref" ]]; then
    jq -nc \
      --arg bench_id "$bench_id" \
      --arg group "$bench_group" \
      --argjson baseline_p95_ns "$baseline_p95_ns" \
      --argjson candidate_p95_ns "$candidate_p95_ns" \
      --argjson delta_percent "$delta_percent" \
      --arg status "$status" \
      --arg risk_note_ref "$risk_note_ref" \
      '{
        bench_id: $bench_id,
        group: $group,
        baseline_p95_ns: $baseline_p95_ns,
        candidate_p95_ns: $candidate_p95_ns,
        delta_percent: $delta_percent,
        status: $status,
        risk_note_ref: $risk_note_ref
      }' >>"$benchmarks_tmp"
  else
    jq -nc \
      --arg bench_id "$bench_id" \
      --arg group "$bench_group" \
      --argjson baseline_p95_ns "$baseline_p95_ns" \
      --argjson candidate_p95_ns "$candidate_p95_ns" \
      --argjson delta_percent "$delta_percent" \
      --arg status "$status" \
      '{
        bench_id: $bench_id,
        group: $group,
        baseline_p95_ns: $baseline_p95_ns,
        candidate_p95_ns: $candidate_p95_ns,
        delta_percent: $delta_percent,
        status: $status
      }' >>"$benchmarks_tmp"
  fi

  echo "[perf] $bench_id: delta=${delta_percent}% status=$status"
done

# Build report
mkdir -p "$ROOT_DIR/artifacts/ci"

# Handle empty files (no benchmarks found)
if [[ ! -s "$benchmarks_tmp" ]]; then
  echo "error: no benchmark data found in $CRITERION_DIR" >&2
  echo '[]' >"$benchmarks_tmp"
  overall_pass=false
fi

if [[ ! -s "$regressions_tmp" ]]; then
  echo '[]' >"$regressions_tmp"
fi

benchmarks_json="$(jq -s '.' "$benchmarks_tmp")"
regressions_json="$(jq -s '.' "$regressions_tmp")"

jq -n \
  --arg schema_version "frankenjax.perf-delta.v1" \
  --argjson generated_at_unix_ms "$generated_at_ms" \
  --arg baseline_id "$baseline_id" \
  --arg candidate_id "$candidate_id" \
  --argjson regression_threshold_percent "$threshold" \
  --argjson benchmarks "$benchmarks_json" \
  --argjson regressions "$regressions_json" \
  --arg overall_status "$(if [[ "$overall_pass" == "true" ]]; then echo pass; else echo fail; fi)" \
  '{
    schema_version: $schema_version,
    generated_at_unix_ms: $generated_at_unix_ms,
    baseline_id: $baseline_id,
    candidate_id: $candidate_id,
    regression_threshold_percent: $regression_threshold_percent,
    benchmarks: $benchmarks,
    regressions: $regressions,
    overall_status: $overall_status
  }' >"$PERF_REPORT_JSON"

echo ""
echo "Performance regression report: $PERF_REPORT_JSON"

regression_count="$(echo "$regressions_json" | jq 'length')"
unjustified_count="$(echo "$regressions_json" | jq '[.[] | select(.justified == false)] | length')"

if [[ "$regression_count" -gt 0 ]]; then
  echo "Regressions: $regression_count ($unjustified_count unjustified)"
fi

if [[ "$overall_pass" != "true" ]]; then
  echo ""
  echo "FAIL: $unjustified_count benchmark(s) regressed >${threshold}% without risk-note justification" >&2
  exit 1
else
  echo "PASS: no unjustified regressions above ${threshold}%"
fi
