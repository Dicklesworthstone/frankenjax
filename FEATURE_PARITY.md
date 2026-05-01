# FEATURE_PARITY

Audit timestamp: **2026-05-01** (`cargo test --workspace` passed via `rch`; live inventory: 15 workspace crates, 110 primitives, 11 dtypes, 162,733 Rust source lines under `crates/`, 4,416 static Rust test/proptest markers, 115 conformance test files, and 848 committed JAX oracle fixture cases).

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Reality-Check Follow-Up Tracker

The 2026-05-01 reality check found that the implementation is substantial and workspace-green, but several "all green" claims needed narrower language. Remaining parity and evidence gaps are now tracked by explicit beads:

| Bead | Scope | Current State |
|---|---|---|
| `frankenjax-fcxy.1` | Reconcile README, CHANGELOG, FEATURE_PARITY, and TODO status with live implementation evidence | in progress |
| `frankenjax-fcxy.2` | Complete Phase2C packet topology and durability proof coverage | open |
| `frankenjax-fcxy.3` | Strengthen transform-composition verification beyond ledger hygiene | closed |
| `frankenjax-fcxy.4` | Replace or explicitly gate composed-grad finite-difference fallback | closed |
| `frankenjax-fcxy.5` | Define and enforce global performance baseline gates | open |

## Feature Family Matrix

| Feature Family | Status | Current Evidence | Next Required Artifact |
|---|---|---|---|
| Canonical IR + TTL model | parity_green | Canonical IR/value model in `crates/fj-core/src/lib.rs`; transform evidence flow in `crates/fj-dispatch/src/lib.rs`; E2E traces in `artifacts/e2e/`; current composition verifier checks evidence count, non-empty evidence, evidence-to-transform binding, duplicate evidence IDs, and evidence-bound stack signatures | Expand structural oracle comparisons to larger program families |
| Primitive semantics (110 ops) | parity_green | `Primitive` enum (110 ops) in `crates/fj-core/src/lib.rs`; evaluator + extensive tests in `crates/fj-lax/src/lib.rs`; all workspace tests green | Continue expanding oracle-backed primitive fixture families |
| Interpreter path over canonical IR | parity_green | Interpreter/eval coverage in `crates/fj-interpreters/src/lib.rs` and staging tests; multi-output support via sub_jaxprs | Add broader higher-rank oracle parity fixtures |
| Dispatch path + transform wrappers (`jit`/`grad`/`vmap`) | parity_green | Dispatch + composition tests in `crates/fj-dispatch/src/lib.rs`; e-graph optimization wired via `egraph_optimize` compile option; BatchTrace fast path for default `vmap`; `grad(jit(f))` uses symbolic AD and remaining finite-difference grad fallback is gateable with `allow_finite_diff_grad_fallback=false` or `deny` | Extend parity report against broader legacy transform matrix |
| Cache-key determinism + strict/hardened split | parity_green | Determinism/mode-split tests in `crates/fj-cache/src/lib.rs`; strict-vs-hardened E2E in `artifacts/e2e/` | Component-by-component parity ledger against legacy cache-key behavior |
| Decision/evidence ledger foundation | parity_green | Ledger/test coverage in `crates/fj-ledger/src/lib.rs`; audit-trail E2E in `artifacts/e2e/e2e_p2c004_evidence_ledger_audit_trail.e2e.json` | Add calibration/drift confidence reporting artifacts |
| Conformance harness + transform bundle runner | parity_green | Harness/reporting code in `crates/fj-conformance/src/lib.rs`; 613 transform fixtures + 25 RNG fixtures + 33 linalg/FFT fixtures + 15 composition fixtures + 162 dtype-promotion fixtures captured from JAX 0.9.2; smoke/integration tests; linalg/FFT/durability/e-graph conformance tests; explicit higher-rank/edge-case oracle coverage in `tests/linalg_oracle.rs` and `tests/fft_oracle.rs` | Capture matching higher-rank/edge-case JAX fixtures so these scenarios also live in the fixture-backed parity bundles |
| Legacy fixture capture automation | parity_green | Capture pipeline script in `crates/fj-conformance/scripts/capture_legacy_fixtures.py` (strict + fallback modes); strict-mode capture completed with JAX 0.9.2 in uv venv (Python 3.12); `_as_u32_list` updated for JAX 0.9+ PRNG key API | Automate periodic re-capture for regression detection |
| `jit` transform semantics | parity_green | Transform fixture + API E2E coverage in `crates/fj-conformance/tests/transforms.rs` and `crates/fj-conformance/tests/e2e_p2c002.rs` | Expand to broader oracle slices |
| `grad` transform semantics | parity_green | Tape-based reverse-mode AD in `crates/fj-ad/src/lib.rs`; all 110 primitives with VJP+JVP rules including linalg (Cholesky, QR, SVD, Eigh, TriangularSolve) and FFT (Fft, Ifft, Rfft, Irfft); custom_vjp/custom_jvp registration; Jacobian/Hessian; value_and_grad shared forward pass | Expand numerical verification coverage for complex AD rules |
| `vmap` transform semantics | in_progress | BatchTrace path in `crates/fj-dispatch/src/batching.rs`; per-primitive batching rules; in_axes/out_axes support; conformance in `crates/fj-conformance/tests/vmap_conformance.rs` and `multirank_conformance.rs`; README limitations still identify more complex iterative compositions as unfinished | Complete parity for advanced batching/control-flow compositions |
| RNG implementation (ThreeFry + samplers) | parity_green | ThreeFry2x32/key/split/fold_in/uniform/normal/bernoulli/categorical in `crates/fj-lax/src/threefry.rs`; fold_in uses JAX threefry_seed convention `[0, data]`; uniform/normal match JAX's f32 partitionable path (XOR-based bits + mantissa conversion, erfinv for normal); KS, chi-squared, binomial statistical tests | RNG conformance integrated into oracle fixture families |
| RNG determinism vs JAX oracle | parity_green | RNG fixture bundle (`rng_determinism.v1.json`) with 25 cases across 5 seeds; **all 25/25 pass**; **34 threefry tests** including multi-level split independence, fold_in composition, exponential/truncated normal distributions, multi-seed KS/normal statistics, split correlation tests | Capture additional JAX oracle fixtures for distribution families |
| RaptorQ sidecar durability pipeline | in_progress | Durability implementation in `crates/fj-conformance/src/durability.rs`; CLI in `crates/fj-conformance/src/bin/fj_durability.rs`; sidecar/scrub/proof artifacts under `artifacts/durability/`; automated durability coverage tests for conformance fixtures, CI budgets, and parity reports in `tests/durability_coverage.rs` | Complete all-long-lived-artifact and Phase2C packet coverage under `frankenjax-fcxy.2` |
| DType system (11 types + promotion rules) | parity_green | BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128 in `crates/fj-core/src/lib.rs`; native scalar literals for F32/F64/half precision; type promotion rules match JAX lattice; 162 JAX oracle dtype promotion cases with scalar F32 value checks; tensor-level promotion tests; **Complex64/Complex128 promotion tests** (Complex64+F64→Complex128 fix, complex mul/div correctness) | Expand oracle fixture coverage for complex dtype pairs |
| AD completeness | parity_green | All 110 primitives with VJP+JVP rules; linalg decompositions and FFT fully implemented; **21 VJP numerical tests** (including ill-conditioned Cholesky 3x3, near-singular QR/SVD, TriangularSolve near-zero, exp/log/div boundary) + **14 JVP numerical tests** (including rectangular QR, 3x3 SVD/Eigh, near-singular triangular solve, exp/log boundary); Eigh VJP stabilized for clustered eigenvalues | Expand to higher-rank and complex-valued AD verification |
| Control flow (`cond`/`scan`/`while`/`fori_loop`/`switch`) | in_progress | All control flow primitives implemented in `crates/fj-lax/src/lib.rs`; **31 conformance tests** covering cond, scan, while, fori_loop, switch; transform compositions: grad+cond, grad+scan, grad+while, jit+grad+cond, jit+grad+scan, vmap+cond, **vmap(grad(cond))**, **vmap(grad(scan))**, **jit(vmap(grad(cond/scan)))**, **grad(grad(f))**, **vmap(grad(grad(f)))**, **vmap(grad(while))**; primitive scalar-sequence `vmap(scan)` uses direct row folds for add/sub/mul bodies; batched-index `vmap(switch)` evaluates branches once over the batch and selects rows with JAX clamping semantics | Further nested compositions with advanced scan/while control flow |
| Tracing from user code + nested transform tracing | parity_green | `make_jaxpr()` and `make_jaxpr_fallible()` in `crates/fj-trace/src/lib.rs`; nested trace context simulation; **69 trace tests** including multi-input broadcasting, unary chain shape preservation, reduction shape changes, mixed dtype promotion, diamond DAG, multi-output, multiple reductions | Expand to nested trace contexts for transform composition |
| E-graph optimization pipeline | parity_green | E-graph language (70+ node types), 87 algebraic rewrite rules, `optimize_jaxpr()` with equality saturation; wired into dispatch via `egraph_optimize` compile option; shared extraction dedupe for multi-output algebraic regions; 48 unit tests in `crates/fj-egraph/src/lib.rs`; **22 optimization-preserving conformance tests** including multi-equation (sin², cascade, exp-log-square), idempotence (abs, reciprocal), tensor ops in `tests/egraph_preserves_semantics.rs` | Expand multi-output optimization coverage; expand to shape-aware rewrites |
| Special functions + linear algebra + FFT | parity_green | Cbrt, Lgamma, Digamma, ErfInv, IsFinite, IntegerPow, Nextafter implemented; Cholesky, QR, SVD, TriangularSolve, Eigh fully implemented in `crates/fj-lax/src/linalg.rs`; Fft/Ifft/Rfft/Irfft implemented in `crates/fj-lax/src/fft.rs`; 33 JAX-captured parity fixtures in `fixtures/linalg_fft_oracle.v1.json` generated by `crates/fj-conformance/scripts/capture_linalg_fft_oracle.py` (JAX 0.9.2, Python 3.12.13) with runner coverage in `tests/linalg_fft_oracle_parity.rs`; additional higher-rank and edge-case coverage in `tests/linalg_oracle.rs`, `tests/fft_oracle.rs`, and `tests/linalg_higher_rank.rs` | Expand the fixture-backed bundle further with higher-rank and edge-case cases so it fully matches the broader manual oracle surface |
| CPU parallel backend | parity_green | Hybrid dependency-wave executor in `crates/fj-backend-cpu/src/executor.rs`; wide DAGs under 128 equations keep scan scheduling, longer pure segments use dependency-count consumer wakeups keyed by local producer index; ready-wave parallelism now stays tensor/cost aware by keeping scalar waves below 256 equations sequential and only parallelizing tensor waves when aggregate or per-input element counts justify Rayon; tests cover out-of-order dependencies, control-flow barrier ordering, missing segment inputs, long-chain dependency-count execution, long branched fan-in execution, long-segment missing-input errors, and tensor ready-wave cost gates; ordering/barrier isomorphism preserved because only pure single-output segments are dependency-scheduled and ready waves still commit in equation-index order; `backend_execute/dependency_chain_512` improved from 92.011us to 78.885us (-13.9%) and measured 74.075us in the latest guardrail; `backend_scheduler_cutover/dependency_chain/255` improved from 138.50us scan path to 38.374us (-72.4%) after lowering cutover to 128; branched fan-in benchmarks improved from 2.8725ms to 21.549us for 16x8, 3.8151ms to 41.504us for 32x8, and 2.7627ms to 43.671us for 64x4; new 128x2 fan-in guardrail measured 53.646us; `wide_parallel_64` improved from 1.5485ms to 13.703us; tensor ready-wave tuning improved `16x4x4` from 1.7003ms to 35.357us, `16x4x64` from 1.8245ms to 295.62us, and `32x4x64` from 2.4386ms to 531.26us while preserving large-tensor parallel speed (`16x4x1024` 2.4745ms -> 2.4714ms, `16x4x4096` 5.2948ms -> 5.2378ms) | Expand tensor-heavy benchmarks to dot/FFT/reduction primitive mixes |

## Performance Evidence Updates

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
