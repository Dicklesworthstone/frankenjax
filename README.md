# FrankenJAX

<div align="center">
  <img src="frankenjax_illustration.webp" alt="FrankenJAX - Clean-room Rust reimplementation of JAX transform semantics">
</div>

FrankenJAX is a clean-room Rust reimplementation targeting semantic fidelity, mathematical rigor, operational safety, and profile-proven performance.

## Core Identity

Trace Transform Ledger (TTL): canonical JAXPR-like IR with transform-composition evidence for `jit`, `grad`, and `vmap`.

Transform composition semantics are non-negotiable.

## Method Stack

This project applies four disciplines for meaningful changes:

1. `alien-artifact-coding`
2. `extreme-software-optimization`
3. `RaptorQ-everywhere`
4. strict/hardened compatibility-security mode split

## Legacy Oracle

- `/dp/frankenjax/legacy_jax_code/jax`

## Exemplar Reference

- `references/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

## Workspace Crates

- `fj-core` — canonical IR, core value/types, primitive definitions
- `fj-trace` — tracing context, tracer model, primitive shape inference
- `fj-interpreters` — scoped primitive interpreter and environment model
- `fj-dispatch` — transform-order-sensitive dispatch and composition checks
- `fj-lax` — LAX primitive evaluation and shape-aware tensor behavior
- `fj-cache` — deterministic cache keys and strict/hardened cache policy gates
- `fj-runtime` — runtime value model and execution surfaces
- `fj-conformance` — fixture harness, oracle capture pipeline, parity reporting
- `fj-ledger` — decision/evidence ledger and audit primitives
- `fj-ad` — automatic differentiation (VJP/JVP and transform wiring)
- `fj-egraph` — e-graph rewrite and equality-saturation optimization scaffolding
- `fj-test-utils` — shared fixtures, deterministic testing helpers, logging contracts
- `fj-api` — user-facing API composition layer
- `fj-backend-cpu` — CPU backend execution plumbing
- `fj-ffi` — FFI surface and interop boundaries

## Current Status

Implemented foundation + differential harness slice:

- 15-crate Rust workspace with end-to-end trace → dispatch → runtime pipeline
- canonical IR + tensor-aware runtime value model with transform ledger evidence
- transform composition proof checks and order-sensitive dispatch execution
- `Primitive` enum currently defines **110** operations in `fj-core`
- 11 DTypes (BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128) with type promotion
- tape-based reverse-mode AD: all 110 primitives with VJP+JVP rules, Jacobian/Hessian, value_and_grad
- vmap with per-primitive batching rules, in_axes/out_axes, BatchTrace interpreter
- e-graph equality saturation optimizer (80+ algebraic rewrite rules) wired into dispatch
- ThreeFry2x32 RNG with key/split/fold_in/uniform/normal/bernoulli/categorical
- control flow: cond, scan, while_loop, fori_loop, switch
- tracing: make_jaxpr from Rust closures, nested trace contexts
- CPU parallel backend with dependency-wave executor
- deterministic cache-key module with strict/hardened gate behavior
- decision/evidence ledger primitives with loss-matrix actions
- conformance fixture bundle runner for `jit`/`grad`/`vmap`, including multi-rank fixture families
- durability pipeline (`generate` / `scrub` / `proof` / `pipeline`) for long-lived artifacts
- test surface currently includes **1576** `#[test]` cases across workspace crates, plus proptest/property suites

## Fixture Capture

Regenerate transform fixtures:

```bash
python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json
```

Notes:

- strict legacy capture requires compatible `jax` + `jaxlib`
- when unavailable, script falls back to deterministic analytical capture unless `--strict` is set

## E2E Orchestration

Run all discovered E2E scenarios:

```bash
./scripts/run_e2e.sh
```

Run one packet:

```bash
./scripts/run_e2e.sh --packet P2C-001
```

Run one scenario:

```bash
./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline
```

Each scenario emits a forensic log at `artifacts/e2e/<scenario>.e2e.json` with replay command, input capture, intermediate states, output capture, and timing.

## Reliability Gates

Budgets are defined in `artifacts/ci/reliability_budgets.v1.json`.

Run full reliability gates (coverage + flake + runtime + crash triage):

```bash
./scripts/enforce_quality_gates.sh
```

Run targeted gates during local iteration:

```bash
./scripts/enforce_quality_gates.sh --skip-coverage --flake-runs 3
```

Flake detector standalone:

```bash
./scripts/detect_flakes.sh --runs 10
```

Crash triage gate standalone (fails on open/new `P0` based on crash index):

```bash
./scripts/enforce_quality_gates.sh --skip-coverage --skip-flake --skip-runtime
```

## Fuzzing + Crash Triage

Build all fuzz targets:

```bash
cd crates/fj-conformance/fuzz
cargo fuzz build
```

Run one target with seed corpus:

```bash
cd crates/fj-conformance/fuzz
cargo fuzz run ir_deserializer corpus/seed/ir_deserializer
```

Triage a crash artifact (hash + minimize + durability sidecars + index update):

```bash
./scripts/triage_crash.sh \
  --target ir_deserializer \
  --input crates/fj-conformance/fuzz/artifacts/ir_deserializer/crash-<hash> \
  --bead bd-3dl.7
```

## Durability Commands

Generate sidecar only:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  generate --artifact <artifact_path> --sidecar <sidecar_path>
```

Scrub sidecar:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  scrub --artifact <artifact_path> --sidecar <sidecar_path> --report <scrub_report_path>
```

Generate decode proof:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  proof --artifact <artifact_path> --sidecar <sidecar_path> --proof <decode_proof_path> --drop-source 2
```

All-in-one pipeline:

```bash
cargo run -p fj-conformance --bin fj_durability -- \
  pipeline --artifact <artifact_path> --sidecar <sidecar_path> --report <scrub_report_path> --proof <decode_proof_path>
```

## Key Documents

- `AGENTS.md`
- `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`
- `PLAN_TO_PORT_JAX_TO_RUST.md`
- `EXISTING_JAX_STRUCTURE.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`

## Verification Commands

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo test -p fj-conformance -- --nocapture
cargo bench
```
