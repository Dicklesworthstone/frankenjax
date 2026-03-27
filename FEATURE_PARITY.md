# FEATURE_PARITY

Audit timestamp: **2026-03-27** (`cargo fmt --check` passed locally; `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p fj-conformance --test linalg_oracle -- --nocapture`, and `cargo test -p fj-conformance --test fft_oracle -- --nocapture` passed via `rch`).

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Feature Family Matrix

| Feature Family | Status | Current Evidence | Next Required Artifact |
|---|---|---|---|
| Canonical IR + TTL model | parity_green | Canonical IR/value model in `crates/fj-core/src/lib.rs`; transform evidence flow in `crates/fj-dispatch/src/lib.rs`; E2E traces in `artifacts/e2e/` | Expand structural oracle comparisons to larger program families |
| Primitive semantics (110 ops) | parity_green | `Primitive` enum (110 ops) in `crates/fj-core/src/lib.rs`; evaluator + extensive tests in `crates/fj-lax/src/lib.rs`; all workspace tests green | Continue expanding oracle-backed primitive fixture families |
| Interpreter path over canonical IR | parity_green | Interpreter/eval coverage in `crates/fj-interpreters/src/lib.rs` and staging tests; multi-output support via sub_jaxprs | Add broader higher-rank oracle parity fixtures |
| Dispatch path + transform wrappers (`jit`/`grad`/`vmap`) | parity_green | Dispatch + composition tests in `crates/fj-dispatch/src/lib.rs`; e-graph optimization wired via `egraph_optimize` compile option; 55+ dispatch tests | Extend parity report against broader legacy transform matrix |
| Cache-key determinism + strict/hardened split | parity_green | Determinism/mode-split tests in `crates/fj-cache/src/lib.rs`; strict-vs-hardened E2E in `artifacts/e2e/` | Component-by-component parity ledger against legacy cache-key behavior |
| Decision/evidence ledger foundation | parity_green | Ledger/test coverage in `crates/fj-ledger/src/lib.rs`; audit-trail E2E in `artifacts/e2e/e2e_p2c004_evidence_ledger_audit_trail.e2e.json` | Add calibration/drift confidence reporting artifacts |
| Conformance harness + transform bundle runner | parity_green | Harness/reporting code in `crates/fj-conformance/src/lib.rs`; 611 transform fixtures + 25 RNG fixtures captured from JAX 0.9.2; smoke/integration tests; linalg/FFT/durability/e-graph conformance tests; explicit higher-rank/edge-case oracle coverage in `tests/linalg_oracle.rs` and `tests/fft_oracle.rs` (4x4 Cholesky, 4x3 QR, 3x2 SVD, repeated-eigenvalue Eigh, transpose+unit-diagonal triangular solve, rank-2 FFT batching, RFFT zero-padding, odd-length IRFFT roundtrip) | Capture matching higher-rank/edge-case JAX fixtures so these scenarios also live in the fixture-backed parity bundles |
| Legacy fixture capture automation | parity_green | Capture pipeline script in `crates/fj-conformance/scripts/capture_legacy_fixtures.py` (strict + fallback modes); strict-mode capture completed with JAX 0.9.2 in uv venv (Python 3.12); `_as_u32_list` updated for JAX 0.9+ PRNG key API | Automate periodic re-capture for regression detection |
| `jit` transform semantics | parity_green | Transform fixture + API E2E coverage in `crates/fj-conformance/tests/transforms.rs` and `crates/fj-conformance/tests/e2e_p2c002.rs` | Expand to broader oracle slices |
| `grad` transform semantics | parity_green | Tape-based reverse-mode AD in `crates/fj-ad/src/lib.rs`; all 110 primitives with VJP+JVP rules including linalg (Cholesky, QR, SVD, Eigh, TriangularSolve) and FFT (Fft, Ifft, Rfft, Irfft); custom_vjp/custom_jvp registration; Jacobian/Hessian; value_and_grad shared forward pass | Expand numerical verification coverage for complex AD rules |
| `vmap` transform semantics | parity_green | BatchTrace path in `crates/fj-dispatch/src/batching.rs`; per-primitive batching rules; in_axes/out_axes support; conformance in `crates/fj-conformance/tests/vmap_conformance.rs` and `multirank_conformance.rs` | Complete parity for advanced batching/control-flow compositions |
| RNG implementation (ThreeFry + samplers) | parity_green | ThreeFry2x32/key/split/fold_in/uniform/normal/bernoulli/categorical in `crates/fj-lax/src/threefry.rs`; fold_in uses JAX threefry_seed convention `[0, data]`; uniform/normal match JAX's f32 partitionable path (XOR-based bits + mantissa conversion, erfinv for normal); KS, chi-squared, binomial statistical tests | RNG conformance integrated into oracle fixture families |
| RNG determinism vs JAX oracle | parity_green | RNG fixture bundle (`rng_determinism.v1.json`) with 25 cases across 5 seeds; **all 25/25 pass** (key, split, fold_in, uniform, normal); determinism conformance tests in `crates/fj-conformance/tests/random_determinism.rs` | Expand to additional seed/distribution families |
| RaptorQ sidecar durability pipeline | parity_green | Durability implementation in `crates/fj-conformance/src/durability.rs`; CLI in `crates/fj-conformance/src/bin/fj_durability.rs`; sidecar/scrub/proof artifacts under `artifacts/durability/`; automated durability coverage tests for all conformance fixtures, CI budgets, and parity reports in `tests/durability_coverage.rs` | Expand to benchmark delta artifacts and migration manifests |
| DType system (11 types + promotion rules) | parity_green | BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128 in `crates/fj-core/src/lib.rs`; type promotion rules match JAX lattice (BF16+BF16→BF16, F16+F16→F16, half+int→half, U32+F32→F32); 162 JAX oracle dtype promotion cases; 32 core-type + **66/66 tensor-level** promotion tests pass in `tests/dtype_promotion_oracle_parity.rs` | Complete complex dtype promotion coverage |
| AD completeness | parity_green | All 110 primitives with VJP+JVP rules; linalg decompositions (Cholesky, QR, SVD, Eigh, TriangularSolve) and FFT (Fft, Ifft, Rfft, Irfft) fully implemented with multi-output support; custom_vjp/custom_jvp registration; Jacobian/Hessian matrix computation; value_and_grad; ReduceWindow VJP fully implemented; **10 numerical VJP verification tests** (Cholesky 2x2+3x3, QR, SVD, Eigh, TriangularSolve, FFT, RFFT, IFFT, IRFFT) + 5 JVP tests | Further edge-case AD verification (ill-conditioned, denormals) |
| Control flow (`cond`/`scan`/`while`/`fori_loop`/`switch`) | parity_green | All control flow primitives implemented in `crates/fj-lax/src/lib.rs`; **20 conformance tests** covering cond, scan, while, fori_loop, switch; transform compositions: grad+cond, grad+scan, grad+while, jit+grad+cond, jit+grad+scan, vmap+cond; 27 E2E fixture cases | Expand nested multi-transform compositions (grad(vmap(cond)), etc.) |
| Tracing from user code + nested transform tracing | parity_green | `make_jaxpr()` and `make_jaxpr_fallible()` in `crates/fj-trace/src/lib.rs`; nested trace context simulation; re-export via `crates/fj-api/src/lib.rs` | Broaden trace-time validation evidence |
| E-graph optimization pipeline | parity_green | E-graph language (70+ node types), 87 algebraic rewrite rules, `optimize_jaxpr()` with equality saturation; wired into dispatch via `egraph_optimize` compile option; 47 unit tests in `crates/fj-egraph/src/lib.rs`; **22 optimization-preserving conformance tests** including multi-equation (sin², cascade, exp-log-square), idempotence (abs, reciprocal), tensor ops in `tests/egraph_preserves_semantics.rs` | Support multi-output jaxpr optimization; expand to shape-aware rewrites |
| Special functions + linear algebra + FFT | parity_green | Cbrt, Lgamma, Digamma, ErfInv, IsFinite, IntegerPow, Nextafter implemented; Cholesky, QR, SVD, TriangularSolve, Eigh fully implemented in `crates/fj-lax/src/linalg.rs`; Fft/Ifft/Rfft/Irfft implemented in `crates/fj-lax/src/fft.rs`; 21 hand-verified parity fixtures in `fixtures/linalg_fft_oracle.v1.json` with runner coverage in `tests/linalg_fft_oracle_parity.rs`; additional hand-verified higher-rank and edge-case coverage in `tests/linalg_oracle.rs`, `tests/fft_oracle.rs`, and `tests/linalg_higher_rank.rs` | Replace the hand-verified `linalg_fft_oracle.v1.json` bundle with a reproducible JAX-captured generator, then expand it with higher-rank/edge-case cases so fixture-backed parity matches the broader manual oracle surface |
| CPU parallel backend | parity_green | Dependency-wave parallel executor in `crates/fj-backend-cpu/src/lib.rs`; replaces sequential interpreter | Profile and optimize wave scheduling |

## Required Evidence Per Family

1. Differential conformance report.
2. Invariant checklist entry/update.
3. Benchmark delta report for perf-sensitive changes.
4. Risk note update if compatibility/security surface changed.

## Coverage Objective

Target: 100% coverage for declared V1 scope, with explicit parity exceptions documented by artifact and linked to open bead IDs.
