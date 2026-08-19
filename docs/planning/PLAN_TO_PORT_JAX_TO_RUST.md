# PLAN_TO_PORT_JAX_TO_RUST

## 0. Source-of-Truth Workflow

This repository follows the `porting-to-rust` spec-first workflow:

1. Extract legacy behavior into explicit specs.
2. Implement from spec (never line-by-line translation).
3. Verify with differential conformance against legacy oracle.
4. Optimize only with behavior-isomorphism proof artifacts.

Reference exemplar imported locally:

- `references/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

## 1. Legacy Oracle

- Local: `/dp/frankenjax/legacy_jax_code/jax`
- Upstream: `https://github.com/jax-ml/jax`

## 2. Port Scope (Target)

- JAX trace/Jaxpr semantics for scoped primitives and transforms.
- Transform composition correctness for `jit`, `grad`, `vmap`.
- Deterministic cache-key semantics for compilation/dispatch.
- Differential conformance harness with reproducible fixture artifacts.
- Benchmark harness for trace/compile/execute latency and cache warm/cold behavior.

## 3. Explicit Exclusions (Current)

- Full jaxlib replacement and TPU plugin breadth.
- Full ecosystem parity beyond scoped V1 contract.
- Distributed plugin/runtime breadth not needed for core migration acceptance.

## 4. Mandatory Method Stack Application

Every meaningful implementation decision must emit artifacts for:

1. `alien-artifact-coding`: decision-theoretic contracts + evidence ledgers + confidence claims.
2. `extreme-software-optimization`: baseline/profile/one-lever/proof/rebaseline loop.
3. `RaptorQ-everywhere`: sidecars + scrub reports + decode proof path.
4. Compatibility-security doctrine: strict/hardened mode split + fail-closed on unknown incompatibilities.

## 5. Execution Phases

### Phase A: Spec Lock

- Complete and continuously maintain:
  - `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`
  - `EXISTING_JAX_STRUCTURE.md`
  - `PROPOSED_ARCHITECTURE.md`
  - `FEATURE_PARITY.md`
- For each subsystem, list exact legacy file/function anchors.

Exit criteria:

- Spec sections map directly to crate/module ownership.
- Each compatibility-critical behavior has at least one planned fixture family.

### Phase B: First Vertical Slice (in progress)

- Canonical IR and transform ledger models (`fj-core`).
- Scalar primitive interpreter path (`fj-lax` + `fj-interpreters`).
- Deterministic cache-key derivation with strict/hardened gates (`fj-cache`).
- Decision/evidence ledger foundation (`fj-ledger`).
- Dispatch path stitching the above (`fj-dispatch`).
- Conformance manifest/report scaffolding (`fj-conformance`).

Exit criteria:

- End-to-end deterministic dispatch for scoped scalar programs.
- Cache key is deterministic and strict-mode fail-closed behavior enforced.
- Conformance harness emits machine-readable parity report skeleton.

### Phase C: Legacy Extraction Wave

- Extract concrete behaviors from:
  - `jax/_src/api.py`
  - `jax/_src/interpreters/{ad,batching,partial_eval}.py`
  - `jax/_src/{dispatch,compiler,cache_key,compilation_cache}.py`
  - `jax/tests/{jax_jit_test,lax_autodiff_test,lax_vmap_test,lax_test,random_test}.py`
- Create normalized fixture corpus grouped by feature family.

Exit criteria:

- Differential fixtures generated for each feature family.
- Expected behavior and edge-case tables committed into spec docs.

### Phase D: Coverage Expansion

- Expand primitive coverage and transform support.
- Implement strict/hardened policy matrix and drift gates.
- Add adversarial and property-based tests for high-risk state transitions.

Exit criteria:

- `FEATURE_PARITY.md` reaches `parity_green` for scoped feature families.

### Phase E: Perf + Durability Hardening

- Benchmark warm/cold cache and transform overhead.
- Profile real hotspots, apply one optimization lever per change.
- Add RaptorQ sidecars for long-lived conformance/benchmark bundles.

Exit criteria:

- No regression across correctness and performance gates.
- Durability artifacts validated (manifest + scrub + decode proof path).

## 6. Conformance Contract

Per feature family, produce:

1. Differential fixture report.
2. Invariant checklist update.
3. Benchmark delta report (if performance-sensitive).
4. Risk note update when compatibility or threat surface changes.

## 7. Tooling and Verification Contract

Required after meaningful code changes:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

When benches/conformance suites mature:

```bash
cargo test -p fj-conformance -- --nocapture
cargo bench
```

## 8. Integration Commitments

- Asupersync integration points are first-class for cancellation, checkpoints, and budget-aware orchestration.
- FrankenTUI integration points are first-class for operational telemetry and progressive disclosure of evidence.

These are implemented incrementally behind explicit integration boundaries so API instability does not block core parity work.
