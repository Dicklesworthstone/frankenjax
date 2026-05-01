# Profiling Workflow

## Benchmark Suite

The global performance gate is tracked in
`artifacts/performance/global_performance_gate.v1.json`. It normalizes the
current baseline across trace, compile/dispatch, execute, cold-cache,
warm-cache, and memory phases without inventing measurements.

Baseline numbers come from
`artifacts/performance/benchmark_baselines_v2_2026-03-12.json`, which captures
82 Criterion benchmarks across seven suites. Memory evidence comes from
`artifacts/performance/memory_performance_gate.v1.json`, which records procfs
RSS measurements for smoke workloads that exercise trace, dispatch, AD, vmap,
FFT, linalg, cache hit/miss, and durability paths.

| Phase | Suites | What it gates |
|-------|--------|---------------|
| `trace` | `jaxpr_fingerprint`, `jaxpr_validation` | Jaxpr capture-adjacent hashing and validation overhead |
| `compile_dispatch` | `transform_composition`, `dispatch_latency` | Transform proof checks and dispatch wrapper latency |
| `execute` | `backend_cpu`, `lax_eval` | CPU backend execution and primitive evaluator throughput |
| `cold_cache` | `cache_subsystem` | Cache misses and cache-key construction |
| `warm_cache` | `cache_subsystem`, `jaxpr_fingerprint` | Cache hits and cached fingerprint retrieval |
| `memory` | `memory_performance_gate` | RSS budget for trace/dispatch/AD/vmap/FFT/linalg/cache/durability smoke workloads |

The dispatch benchmark file remains the densest single suite and covers these
metric categories:

| Group | Benchmarks | What it measures |
|-------|-----------|------------------|
| `dispatch_latency` | jit/grad/vmap x scalar/vector + compositions | End-to-end dispatch overhead |
| `eval_jaxpr_throughput` | 10/100/1000 equation chains | Interpreter throughput scaling |
| `transform_composition` | single/depth-2/depth-3/empty | Composition verification cost |
| `cache_key_generation` | simple/medium/large/hardened | Cache key derivation cost |
| `ledger_append` | single/burst-100 | Evidence ledger write throughput |
| `jaxpr_fingerprint` | 1/10/100 eq + cached lookup | Fingerprint computation + OnceLock cache |
| `jaxpr_validation` | 1/10/100 eq | Well-formedness check cost |

## Running Benchmarks

Full suite:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-dispatch \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench --bench dispatch_baseline
```

Single group:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-dispatch \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench --bench dispatch_baseline -- dispatch_latency
```

Single benchmark:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-dispatch \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench --bench dispatch_baseline -- "jit/scalar_add"
```

Cross-crate phase probes:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-cache \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench -p fj-cache --bench cache_baseline

CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-backend \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench -p fj-backend-cpu --bench backend_baseline

CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-lax \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline
```

## Saving a Baseline

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-baseline \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- ./scripts/check_perf_regression.sh --save-baseline
```

This saves criterion results under the baseline name from `reliability_budgets.v1.json` (default: `main`). Custom name:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-baseline \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- ./scripts/check_perf_regression.sh --save-baseline --baseline-name pre-optimization
```

## Checking for Regressions

After making changes, compare against the saved baseline:

```bash
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-candidate \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- ./scripts/check_perf_regression.sh
```

The gate fails if any benchmark's p95 regresses more than 5% (configurable in
`reliability_budgets.v1.json`) without a risk-note justification. Optimization
work must keep the same loop every time: baseline, profile, change one lever,
prove behavior unchanged with conformance/invariant tests, then re-baseline.

## Justifying an Accepted Regression

If a regression is intentional (e.g., correctness fix that costs throughput), create a risk note:

```bash
mkdir -p artifacts/performance/risk_notes
```

Create `artifacts/performance/risk_notes/<group>_<function>.risk_note.json` following `artifacts/schemas/risk_note.v1.schema.json`.

## CI Integration

The perf regression gate is integrated into `scripts/enforce_quality_gates.sh`:

```bash
./scripts/enforce_quality_gates.sh                 # runs all gates including perf
./scripts/enforce_quality_gates.sh --skip-perf      # skip perf gate
```

## Evidence Artifact

Each gate run emits `artifacts/ci/perf_regression_report.v1.json` conforming to `artifacts/schemas/perf_delta.v1.schema.json`. Fields:

- `baseline_id` / `candidate_id`: git refs being compared
- `benchmarks[]`: per-benchmark p95 values and delta percentages
- `regressions[]`: benchmarks exceeding threshold, with justification status
- `overall_status`: `pass` or `fail`

`artifacts/performance/global_performance_gate.v1.json` is the phase-level
coverage artifact. It must keep measured phases tied to existing Criterion
baseline values and must keep memory tied to `memory_performance_gate.v1.json`
RSS measurements. Allocation counts remain non-synthetic: if an allocator-level
counter backend is added later, it must be recorded as an additional measured
backend rather than inferred from payload sizes.

## Example Profiling Session

```bash
# 1. Save baseline on clean main
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-baseline \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- ./scripts/check_perf_regression.sh --save-baseline

# 2. Make one optimization change

# 3. Run comparison
CARGO_TARGET_DIR=/data/tmp/frankenjax-perf-candidate \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  rch exec -- ./scripts/check_perf_regression.sh

# 4. Review report
jq . artifacts/ci/perf_regression_report.v1.json
```
