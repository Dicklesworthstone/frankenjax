# FEATURE_PARITY

Audit timestamp: **2026-03-15** (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` passed via `rch`).

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
| Conformance harness + transform bundle runner | parity_green | Harness/reporting code in `crates/fj-conformance/src/lib.rs`; 611 transform fixtures + 25 RNG fixtures captured from JAX 0.9.2; smoke/integration tests; linalg/FFT/durability/e-graph conformance tests | Expand to broader oracle slices and multi-rank fixture families |
| Legacy fixture capture automation | parity_green | Capture pipeline script in `crates/fj-conformance/scripts/capture_legacy_fixtures.py` (strict + fallback modes); strict-mode capture completed with JAX 0.9.2 in uv venv (Python 3.12); `_as_u32_list` updated for JAX 0.9+ PRNG key API | Automate periodic re-capture for regression detection |
| `jit` transform semantics | parity_green | Transform fixture + API E2E coverage in `crates/fj-conformance/tests/transforms.rs` and `crates/fj-conformance/tests/e2e_p2c002.rs` | Expand to broader oracle slices |
| `grad` transform semantics | parity_green | Tape-based reverse-mode AD in `crates/fj-ad/src/lib.rs`; all 110 primitives with VJP+JVP rules including linalg (Cholesky, QR, SVD, Eigh, TriangularSolve) and FFT (Fft, Ifft, Rfft, Irfft); custom_vjp/custom_jvp registration; Jacobian/Hessian; value_and_grad shared forward pass | Expand numerical verification coverage for complex AD rules |
| `vmap` transform semantics | parity_green | BatchTrace path in `crates/fj-dispatch/src/batching.rs`; per-primitive batching rules; in_axes/out_axes support; conformance in `crates/fj-conformance/tests/vmap_conformance.rs` and `multirank_conformance.rs` | Complete parity for advanced batching/control-flow compositions |
| RNG implementation (ThreeFry + samplers) | parity_green | ThreeFry2x32/key/split/fold_in/uniform/normal/bernoulli/categorical in `crates/fj-lax/src/threefry.rs`; KS, chi-squared, binomial statistical tests | RNG conformance integrated into oracle fixture families |
| RNG determinism vs JAX oracle | parity_green | RNG fixture bundle (`rng_determinism.v1.json`) with 20+ cases; determinism conformance tests in `crates/fj-conformance/tests/random_determinism.rs`; E2E artifact at `artifacts/e2e/e2e_rng_determinism.e2e.json` | Expand to additional seed/distribution families |
| RaptorQ sidecar durability pipeline | parity_green | Durability implementation in `crates/fj-conformance/src/durability.rs`; CLI in `crates/fj-conformance/src/bin/fj_durability.rs`; sidecar/scrub/proof artifacts under `artifacts/durability/`; automated durability coverage tests for all conformance fixtures, CI budgets, and parity reports in `tests/durability_coverage.rs` | Expand to benchmark delta artifacts and migration manifests |
| DType system (11 types + promotion rules) | parity_green | BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128 in `crates/fj-core/src/lib.rs`; type promotion rules implemented; 162 JAX oracle dtype promotion cases in `fixtures/dtype_promotion_oracle.v1.json`; 32 core-type parity tests pass in `tests/dtype_promotion_oracle_parity.rs` | Expand F32/BF16/F16 tensor-level promotion coverage |
| AD completeness | parity_green | All 110 primitives with VJP+JVP rules; linalg decompositions (Cholesky, QR, SVD, Eigh, TriangularSolve) and FFT (Fft, Ifft, Rfft, Irfft) fully implemented with multi-output support; custom_vjp/custom_jvp registration; Jacobian/Hessian matrix computation; value_and_grad; ReduceWindow VJP fully implemented | Expand oracle-backed numerical verification for linalg/FFT AD rules |
| Control flow (`cond`/`scan`/`while`/`fori_loop`/`switch`) | parity_green | All control flow primitives implemented in `crates/fj-lax/src/lib.rs`; scan supports named body ops; while_loop with functional body; fori_loop; switch | Add full control-flow fixture families and transform-composition parity reports |
| Tracing from user code + nested transform tracing | parity_green | `make_jaxpr()` and `make_jaxpr_fallible()` in `crates/fj-trace/src/lib.rs`; nested trace context simulation; re-export via `crates/fj-api/src/lib.rs` | Broaden trace-time validation evidence |
| E-graph optimization pipeline | parity_green | E-graph language (70+ node types), 80+ algebraic rewrite rules, `optimize_jaxpr()` with equality saturation; wired into dispatch via `egraph_optimize` compile option; 47 unit tests in `crates/fj-egraph/src/lib.rs`; 12 optimization-preserving conformance tests in `crates/fj-conformance/tests/egraph_preserves_semantics.rs` | Expand to multi-equation optimization chains and tensor-valued programs |
| Special functions + linear algebra + FFT | parity_green | Cbrt, Lgamma, Digamma, ErfInv, IsFinite, IntegerPow, Nextafter implemented; Cholesky, QR, SVD, TriangularSolve, Eigh fully implemented in `crates/fj-lax/src/linalg.rs`; Fft/Ifft/Rfft/Irfft implemented in `crates/fj-lax/src/fft.rs`; 21 JAX oracle parity fixtures in `fixtures/linalg_fft_oracle.v1.json` with parity test in `tests/linalg_fft_oracle_parity.rs` | Expand to higher-rank matrices and edge cases |
| CPU parallel backend | parity_green | Dependency-wave parallel executor in `crates/fj-backend-cpu/src/lib.rs`; replaces sequential interpreter | Profile and optimize wave scheduling |

## Required Evidence Per Family

1. Differential conformance report.
2. Invariant checklist entry/update.
3. Benchmark delta report for perf-sensitive changes.
4. Risk note update if compatibility/security surface changed.

## Coverage Objective

Target: 100% coverage for declared V1 scope, with explicit parity exceptions documented by artifact and linked to open bead IDs.
