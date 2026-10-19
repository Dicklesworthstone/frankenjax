# FEATURE_PARITY

Audit timestamp: **2026-10-19**, reconciled across `rc26` (`cargo test --workspace` passed; live inventory: 17 workspace crates, 162 canonical primitive variants in `Primitive::ALL` = 157 V1 local eval/AD primitives plus 5 pmap collectives that fail closed without multi-device context, plus 1 FFI `CustomCall` variant in `Primitive::FFI_PRIMITIVES` = 163 total `Primitive` enum variants, 11 dtypes, 121 `ProgramSpec` variants, 30 active IEEE-754-safe e-graph rules by default plus 61 opt-in `numerically_unsafe_rules()` for 91 total, 341,813 Rust library source lines under `crates/*/src` and 480,784 total Rust lines under `crates/`, 9,425+ static Rust test/proptest markers, 174 conformance test files, and 861 committed JAX oracle fixture cases across 6 fixture bundles).

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Reality-Check Follow-Up Tracker

The 2026-05-01 and 2026-10-19 reality checks reconciled implementation claims and closed all architectural and evidence gaps via explicit beads:

| Bead | Scope | Current State |
|---|---|---|
| `frankenjax-fcxy.1` | Reconcile README, CHANGELOG, FEATURE_PARITY, and TODO status with live implementation evidence | closed |
| `frankenjax-fcxy.2` | Complete Phase2C packet topology and durability proof coverage | closed |
| `frankenjax-fcxy.3` | Strengthen transform-composition verification beyond ledger hygiene | closed |
| `frankenjax-fcxy.4` | Replace or explicitly gate composed-grad finite-difference fallback | closed |
| `frankenjax-fcxy.5` | Define and enforce global performance baseline gates | closed |
| `frankenjax-rc26-n1emn` | 2026-10 Reality-Check & Ambition Program: Symbolic AD, Full Tracing, Connected Spine, Proof TTL, & Honest Gates | closed |

## Feature Family Matrix

| Feature Family | Status | Current Evidence | Next Required Artifact |
|---|---|---|---|
| Canonical IR + TTL model | parity_green | Canonical IR/value model in `crates/fj-core/src/lib.rs`; cryptographic per-step `TransformStepProof` (binding `input_jaxpr_fingerprint` -> `output_jaxpr_fingerprint` + `step_commit_sha256`) in `crates/fj-dispatch/src/lib.rs`; E2E traces in `artifacts/e2e/`; composition verifier checks evidence count, non-empty evidence, evidence-to-transform binding, duplicate evidence IDs, and cryptographic step chain continuity; TTL semantic proof gate in `artifacts/conformance/ttl_semantic_proof_matrix.v1.json` | Expand structural oracle comparisons to larger program families |
| Primitive semantics (162 canonical variants) | parity_green | `Primitive` enum (163 ops: 162 canonical variants in `Primitive::ALL` + 1 FFI `CustomCall` = 163 enum variants) in `crates/fj-core/src/lib.rs`; 157 non-pmap V1 local variants (+ `CustomCall`) have evaluator + extensive tests in `crates/fj-lax/src/lib.rs`; 5 pmap collectives are represented for IR parity and fail closed with typed pmap-context errors; all workspace tests green | Continue expanding oracle-backed primitive fixture families and keep pmap collectives fail-closed until multi-device backend support lands |
| Interpreter path over canonical IR | parity_green | Interpreter/eval coverage in `crates/fj-interpreters/src/lib.rs` and staging tests; multi-output support via sub_jaxprs; `CompiledProgram` pre-lowered execution plan (`eval_jaxpr_cached`) | Add broader higher-rank oracle parity fixtures |
| Dispatch path + transform wrappers (`jit`/`grad`/`vmap`) | parity_green | Dispatch + composition tests in `crates/fj-dispatch/src/lib.rs`; symbolic `grad_to_jaxpr` IR differentiation enables exact arbitrary-order `grad(grad(f))`, `vmap(grad(f))`, and `grad(vmap(f))` with zero finite-difference fallback; bounded 256-entry LRU `CompiledProgram` cache; e-graph optimization enabled by default on `jit`; `backend=cpu-parallel` routes multi-equation DAGs through `fj-backend-cpu::CpuBackend` | Extend parity report against broader legacy transform matrix |
| Cache-key determinism + strict/hardened split | parity_green | Determinism/mode-split tests in `crates/fj-cache/src/lib.rs`; strict-vs-hardened E2E in `artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json` | Component-by-component parity ledger against legacy cache-key behavior |
| Cross-crate error taxonomy | parity_green | Error taxonomy matrix gate in `artifacts/conformance/error_taxonomy_matrix.v1.json`; E2E forensic log in `artifacts/e2e/e2e_error_taxonomy_gate.e2e.json`; covers IR validation, transform proof, primitive arity/type/shape, interpreter missing variables, cache strict/hardened unknown features, vmap axis mismatch, durability missing artifacts, unsupported transform tails, and unsupported control-flow rows | Keep new public error boundaries wired into the taxonomy gate before counting them as complete |
| Security/adversarial gates | parity_green | Security adversarial gate in `artifacts/conformance/security_adversarial_gate.v1.json`; refreshed threat model in `artifacts/conformance/security_threat_model.v1.json`; E2E forensic log in `artifacts/e2e/e2e_security_gate.e2e.json`; live corpus replay proves 9/9 threat categories green, 19/19 fuzz seed families complete, 10/10 adversarial rows typed and panic-free, and 0 open P0 crash-index entries | Keep new parser/FFI/durability/cache/ledger boundaries wired into the security gate before counting them as complete |
| Fresh-clone onboarding command gates | parity_green | Onboarding command inventory in `artifacts/conformance/onboarding_command_inventory.v1.json`; E2E forensic log in `artifacts/e2e/e2e_onboarding_gate.e2e.json`; schema/examples in `artifacts/schemas/onboarding_command_inventory.v1.schema.json`; proves README/key-document command anchors, script paths, replay commands, skip rationales, evidence refs, and safe environment allowlists | Keep every new user-facing command wired into the inventory before counting docs-green |
| Decision/evidence ledger foundation | parity_green | Ledger/test coverage in `crates/fj-ledger/src/lib.rs`; wired into `crates/fj-dispatch/src/lib.rs` and `crates/fj-runtime/src/lib.rs`; decision-ledger calibration report in `artifacts/conformance/decision_ledger_calibration.v1.json`; E2E forensic log in `artifacts/e2e/e2e_decision_ledger_gate.e2e.json`; proves 9 live decision classes and 40-trial calibration buckets | Keep every new consequential runtime decision wired into the calibration gate before counting ledger-green |
| Conformance harness + transform bundle runner | parity_green | Harness/reporting code in `crates/fj-conformance/src/lib.rs`; 861 total oracle cases (613 transform + 25 RNG + 46 linalg/FFT + 15 composition + 162 dtype-promotion) captured from JAX 0.9.2 across 6 fixture bundles in `crates/fj-conformance/fixtures/`; oracle recapture matrix in `artifacts/conformance/oracle_recapture_matrix.v1.json` | Continue expanding complex-valued AD gradient fixture coverage |
| Legacy fixture capture automation | parity_green | Capture pipeline script in `crates/fj-conformance/scripts/capture_legacy_fixtures.py` and `crates/fj-conformance/src/oracle_recapture.rs`; strict-mode capture completed with JAX 0.9.2 in uv venv (Python 3.12); drift report in `artifacts/conformance/oracle_drift_report.v1.json` | Automate periodic re-capture for regression detection |
| `jit` transform semantics | parity_green | Transform fixture + API E2E coverage in `crates/fj-conformance/tests/transforms.rs` and `crates/fj-conformance/tests/e2e_p2c002.rs`; partial evaluation + staging + e-graph optimization + LRU `CompiledProgram` caching in `crates/fj-dispatch/src/lib.rs` | Expand to broader oracle slices |
| `grad` transform semantics | parity_green | Symbolic `grad_to_jaxpr` IR differentiation + tape-based reverse-mode AD in `crates/fj-ad/src/lib.rs`; all 157 non-pmap V1 local canonical primitives (+ `CustomCall`) with analytical VJP+JVP rules including linalg (`Cholesky`, `Qr`, `Svd`, `Eigh`, `Eig`, `TriangularSolve`, `Lu`, `Solve`, `Det`, `Slogdet`), FFT (`Fft`, `Ifft`, `Rfft`, `Irfft`), and `Cbrt`/`Erf`/`Erfc`/`ErfInv`/`Expm1`/`Log1p`/`Logistic` in `grad_to_jaxpr`; exact forward-over-reverse `hessian_jaxpr` | Expand numerical verification coverage for complex AD rules |
| `vmap` transform semantics | parity_green | BatchTrace path in `crates/fj-dispatch/src/batching.rs`; per-primitive batching rules; in_axes/out_axes support; conformance in `crates/fj-conformance/tests/vmap_conformance.rs` and `crates/fj-conformance/tests/multirank_conformance.rs`; transform/control-flow matrix gate in `artifacts/conformance/transform_control_flow_matrix.v1.json` | Expand oracle fixture breadth for higher-rank and larger batch shapes |
| RNG implementation (ThreeFry + samplers) | parity_green | ThreeFry2x32/key/split/fold_in and 28 distributions in `crates/fj-lax/src/threefry.rs` and `crates/fj-api/src/lib.rs`; uniform/normal match JAX's f32 partitionable path; KS, chi-squared, binomial statistical tests | RNG conformance integrated into oracle fixture families |
| RNG determinism vs JAX oracle | parity_green | RNG fixture bundle in `crates/fj-conformance/fixtures/rng/rng_determinism.v1.json` with 25 cases across 5 seeds; all 25/25 pass; statistical and determinism tests in `crates/fj-conformance/tests/random_determinism.rs` | Capture additional JAX oracle fixtures for distribution families |
| RaptorQ sidecar durability pipeline | parity_green | Durability implementation in `crates/fj-conformance/src/durability.rs`; CLI in `crates/fj-conformance/src/bin/fj_durability.rs`; 38/38 required policy artifacts and 126/126 total sidecars under `artifacts/durability/` verified SHA-256 fresh on disk by `crates/fj-conformance/tests/durability_coverage.rs` and `artifacts/conformance/durability_coverage_policy.v1.json` | Maintain 100% SHA-256 fresh triplet coverage on every artifact update |
| DType system (11 types + promotion rules) | parity_green | BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128 in `crates/fj-core/src/lib.rs`; type promotion rules match JAX lattice; 162 JAX oracle dtype promotion cases in `crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json` | Expand oracle fixture coverage for complex dtype pairs |
| AD completeness | parity_green | All 157 non-pmap V1 local canonical primitives (+ `CustomCall`) have analytical VJP+JVP rules in `crates/fj-ad/src/lib.rs` with zero finite-difference fallback (`crates/fj-conformance/tests/ad_numerical_verification.rs`); exact symbolic `grad_to_jaxpr` and forward-over-reverse `hessian_jaxpr` | Expand to higher-rank complex-valued AD verification |
| Numerical stability + platform determinism | parity_green | Live-measured numerical-stability matrix in `artifacts/conformance/numerical_stability_matrix.v1.json`; E2E forensic log in `artifacts/e2e/e2e_numerical_stability_gate.e2e.json`; tests in `crates/fj-conformance/tests/numerical_stability_gate.rs`; proves 11 guardrail families with live execution measurements | Keep new numerical, RNG, serialization, and AD guardrails wired into this gate before counting stability-green |
| Control flow (`cond`/`scan`/`while`/`fori_loop`/`switch`) | parity_green | All control flow primitives implemented in `crates/fj-lax/src/lib.rs` and `crates/fj-interpreters/src/lib.rs`; transform/control-flow matrix gate in `artifacts/conformance/transform_control_flow_matrix.v1.json` passes 21 rows (`crates/fj-conformance/tests/transform_control_flow_gate.rs`) | Expand beyond V1 with broader higher-rank functional control-flow oracle fixtures |
| Python bindings (`fj-py`) | parity_green | PyO3 module in `crates/fj-py/src/lib.rs` exposes `PyValue`, `PyTracer` with 27 operations, Python callable `make_jaxpr`, Python PyTree flatten/unflatten (`list`, `tuple`, `dict`), `jit`, `grad`, `vmap`, `value_and_grad`, and `checkpoint` with 26 Rust unit tests and `crates/fj-py/tests/smoke_test.py` | Expand Python wheel packaging for multi-platform release artifacts |
| Tracing from user code + nested transform tracing | parity_green | `make_jaxpr()` and `make_jaxpr_fallible()` in `crates/fj-trace/src/lib.rs` expose all 157 V1 local canonical primitives (+ `CustomCall`) on `TracerRef`; `crates/fj-api/src/lib.rs` exposes `nn`, `random`, `tree_util`, `linalg`, `fft`, and tensor creation helpers | Expand to deeper multi-output control-flow tracing ergonomics |
| E-graph optimization pipeline | parity_green | E-graph language (70+ node types), 30 active IEEE-754-safe algebraic rewrite rules by default (+ 61 opt-in `numerically_unsafe_rules()`, 91 total) in `crates/fj-egraph/src/lib.rs`; wired into `jit` dispatch by default (`egraph_optimize=true`); conformance tests in `crates/fj-conformance/tests/egraph_preserves_semantics.rs` | Expand shape-aware algebraic rewrites |
| Special functions + linear algebra + FFT | parity_green | Special math (`Erf`, `Erfc`, `Cbrt`, `Lgamma`, `Digamma`, `ErfInv`, `IsFinite`, `IntegerPow`, `Nextafter`, `BesselI0e`, `BesselI1e`), all 10 linalg ops in `crates/fj-lax/src/linalg.rs`, and all 4 FFT ops in `crates/fj-lax/src/fft.rs`; 46 JAX-captured parity fixtures in `crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json` verified by `crates/fj-conformance/tests/linalg_fft_oracle_parity.rs` | Expand complex-valued AD verification beyond current FFT VJP/JVP rows |
| CPU parallel backend | parity_green | Hybrid dependency-wave executor in `crates/fj-backend-cpu/src/executor.rs`; wired into `crates/fj-dispatch/src/lib.rs` via `backend=cpu-parallel` or `cpu_parallel=true` compile option; `fj-backend-gpu` fail-closed contract in `crates/fj-backend-gpu/src/lib.rs` | Expand tensor-heavy benchmarks to dot/FFT/reduction primitive mixes |

## Performance Evidence Updates

### 2026-05-01: Global performance gate (`frankenjax-fcxy.5`)

- Scope: phase-level performance evidence for trace, compile/dispatch, execute, cold-cache, warm-cache, and memory coverage.
- Baseline: `artifacts/performance/benchmark_baselines_v2_2026-03-12.json` records 82 Criterion benchmarks across dispatch, LAX evaluation, cache, API, backend CPU, FFI, partial evaluation, DCE, and staging suites.
- Gate artifact: `artifacts/performance/global_performance_gate.v1.json` maps required phases to existing measured benchmarks and links the `memory` phase to `artifacts/performance/memory_performance_gate.v1.json`, which records Linux procfs RSS evidence for trace, dispatch, AD, vmap, FFT, linalg, cache hit/miss, and durability workloads.
- Optimization queue: `artifacts/performance/optimization_hotspot_scoreboard.v1.json` ranks vmap multiplier, AD tape/backward map, tensor materialization, shape kernels, cache-key hashing, e-graph saturation, FFT/linalg/reduction mixes, and durability encode/decode by measured p95/p99/RSS plus confidence, then creates `br` follow-up beads only for rows scoring at or above `2.0`.
- Policy: p95 regressions above 5% require a risk note; optimization work must baseline, profile, change one lever, prove behavior unchanged, and re-baseline.
- Validation: `crates/fj-conformance/tests/artifact_schemas.rs` includes a coverage test that rejects missing phases, missing benchmark references for measured phases, unmeasured memory rows, synthetic memory numbers, and missing policy flags.

### 2026-05-01: Memory performance gate (`frankenjax-cstq.4`)

- Scope: RSS-gated smoke coverage for trace fingerprinting, JIT dispatch, scalar AD, vector vmap, FFT, Cholesky linalg, cache hit/miss, and durability sidecar recovery.
- Gate artifacts: `artifacts/performance/memory_performance_gate.v1.json`, `artifacts/performance/memory_performance_gate.v1.md`, and `artifacts/e2e/e2e_memory_performance_gate.e2e.json`.
- Durability witness: `artifacts/performance/memory_durability_probe.v1.json` plus `.sidecar.json`, `.scrub.json`, and `.proof.json` prove the durability workload inside the memory gate.
- Validation: `cargo test -p fj-conformance --test memory_performance_gate -- --nocapture`, `./scripts/run_memory_performance_gate.sh --enforce`, and `./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_memory_performance_gate.e2e.json`.

### 2026-05-01: Transform/control-flow matrix gate (`frankenjax-cstq.2`)

- Scope: V1 advanced transform/control-flow behavior matrix for `jit`, `grad`, `vmap`, `value_and_grad`, `jacobian`, `hessian`, nested grad/vmap, `cond`, `scan`, `while`, `switch`, scalar/tensor inputs, multi-carry state, multi-output returns, dtype-mixed rows, and explicit error rows.
- Gate artifacts: `artifacts/conformance/transform_control_flow_matrix.v1.json`, `artifacts/conformance/transform_control_flow_matrix.v1.md`, and `artifacts/e2e/e2e_transform_control_flow_gate.e2e.json`.
- Result: 21 rows pass: 18 supported rows execute under strict mode and 3 unsupported V1 rows fail closed with typed `TransformExecutionError` classes.
- Performance sentinels: `perf_vmap_scan_loop_stack`, `perf_vmap_while_loop_stack`, `perf_jit_vmap_grad_cond`, and `perf_batched_switch` record p50/p95/p99 nanosecond timings plus Linux procfs peak RSS.
- Validation: `cargo test -p fj-conformance --test transform_control_flow_gate -- --nocapture`, `./scripts/run_transform_control_flow_gate.sh --enforce`, and `./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_transform_control_flow_gate.e2e.json`.

### 2026-05-01: Error taxonomy matrix gate (`frankenjax-cstq.8`)

- Scope: cross-crate typed error taxonomy for user-visible strict/hardened boundaries.
- Gate artifacts: `artifacts/conformance/error_taxonomy_matrix.v1.json`, `artifacts/conformance/error_taxonomy_matrix.v1.md`, and `artifacts/e2e/e2e_error_taxonomy_gate.e2e.json`.
- Result: required rows pass for IR validation, transform proof hygiene, primitive arity/type/shape errors, interpreter missing variables, cache unknown-feature policy, vmap axis mismatch, durability missing artifacts, unsupported transform tails, and unsupported transform/control-flow rows.
- Strict/hardened policy: the only allowlisted divergence is cache unknown-feature handling: strict mode rejects unknown incompatible features, while hardened mode preserves them in the deterministic key payload.
- Validation: `cargo test -p fj-conformance --test error_taxonomy_gate -- --nocapture`, `./scripts/run_error_taxonomy_gate.sh --enforce`, and `./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_error_taxonomy_gate.e2e.json`.

### 2026-04-27: `frankenjax-3oq` FFT radix-2 fast path

- Scope: `fj-lax` FFT family (`Fft`, `Ifft`, `Rfft`, `Irfft`) for power-of-two last-axis lengths.
- Baseline: Criterion `eval/*fft_256*` before the change measured `fft=1.1763ms`, `ifft=1.0899ms`, `rfft=1.1658ms`, `irfft=1.0762ms`.
- Profile: `perf record` on the baseline showed `__sincos_fma` as the dominant sample bucket (`74.93%` children, `69.49%` self), under the direct DFT/IDFT loops.
- Opportunity score: impact 5, confidence 5, effort 2 => 12.5.
- Lever: radix-2 Cooley-Tukey path for power-of-two sizes; direct DFT/IDFT remains the exact fallback for non-power-of-two sizes.
- Re-baseline on the same host: `fft=5.7289us` (-99.454%), `ifft=5.7560us` (-99.429%), `rfft=4.0915us` (-99.617%), `irfft=4.7985us` (-99.514%).
- Isomorphism proof: ordering preserved by per-batch in-place butterflies and unchanged output indexing; tie-breaking N/A; RNG N/A; floating-point results are mathematically the same DFT/IDFT with normal floating-point order differences, guarded by direct-reference tests for the fast path and exact fallback tests for non-power-of-two lengths.

### 2026-04-27: `frankenjax-b3xy` batched-index `vmap(switch)` selection

- Scope: `fj-dispatch` BatchTrace handling for primitive-form `Switch` and `Switch` equations carrying sub-Jaxprs when the switch index is batched.
- Baseline: Criterion `vmap_switch/batched_index_128` measured `34.856us` before the change.
- Profile: the source hotspot was the batched-index `Switch` path in `batch_switch_sub_jaxprs`, which sliced every batch element and ran only the selected branch per slice; the benchmark exercises 128 batched indices across identity/add/mul branches.
- Opportunity score: impact 3, confidence 5, effort 2 => 7.5.
- Lever: evaluate each switch branch once with batched inputs, then copy the selected row for each batch index; primitive-form switch values use the same row-selection helper.
- Re-baseline on the same host and target dir: `19.835us` (-44.011%).
- Isomorphism proof: ordering preserved because switch branches are pure Jaxprs and selection still follows the original batch order; tie-breaking/clamping unchanged via the existing `scalar_to_switch_index` rules for negative and high indices; floating-point/RNG N/A for the measured integer branches; golden behavior guarded by primitive and sub-Jaxpr batched switch tests plus conformance `vmap(switch)` coverage.

### 2026-04-28: `frankenjax-fnzr` primitive scalar-sequence `vmap(scan)` folds

- Scope: `fj-dispatch` BatchTrace handling for primitive-form `Scan` when `vmap` maps scalar sequence bodies (`add`, `sub`, `mul`) over batched scalar carries and rank-1/rank-2 `xs`.
- Baseline: Criterion `vmap_scan/shared_init_batched_xs_128x64` measured `313.44us` before the change.
- Profile: the generic transpose-and-vectorized-scan attempt regressed to `462.99us`; the actual source hotspot was the generic per-batch fallback allocating/slicing 128 independent scan executions for a scalar row-fold workload.
- Opportunity score: impact 4, confidence 5, effort 2 => 10.0.
- Lever: fold scalar rows directly from the batched `xs` tensor for supported primitive scan bodies, preserving the existing fallback for higher-rank carries, unsupported body ops, and generalized control-flow scans.
- Re-baseline on the same worker and target dir after narrowing the literal fast path to exact I64/F64 semantics: `35.901us` (-88.546%).
- Isomorphism proof: ordering preserved by iterating scan positions in the same forward/reverse order per batch row; tie-breaking unchanged/N/A; floating-point follows the same left-fold operation order within each row; RNG N/A; behavior guarded by leading-axis, nonzero-axis, scalar-`xs`, and existing dispatch-suite `vmap_scan_batched_carry_and_xs` tests.

## Required Evidence Per Family

1. Differential conformance report.
2. Invariant checklist entry/update.
3. Benchmark delta report for perf-sensitive changes.
4. Risk note update if compatibility/security surface changed.

## Coverage Objective

Target: 100% coverage for declared V1 scope, with explicit parity exceptions documented by artifact and linked to open bead IDs.
