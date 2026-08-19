# COMPREHENSIVE SPECIFICATION FOR FRANKENJAX V1

> A clean-room Rust reimplementation of JAX transform semantics for scoped APIs,
> with Trace Transform Ledger rigor, fail-closed compatibility behavior, and
> profile-proven performance.

---

## 0. How To Read This Document

This is the top-level specification for FrankenJAX V1.

It is intentionally modeled after the rigor and structure of:

- `references/COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md`

This document consolidates and supersedes implementation intent across:

- `PLAN_TO_PORT_JAX_TO_RUST.md`
- `EXISTING_JAX_STRUCTURE.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`

When conflicts exist, this document wins.

### 0.1 Normative language

The following terms are normative:

- MUST / MUST NOT: mandatory requirements.
- SHOULD / SHOULD NOT: strong default; deviations require explicit rationale.
- MAY: optional behavior.

### 0.2 Scope doctrine

This specification defines the full V1 target for scoped compatibility.

Implementation is phased for sequencing, not to quietly drop required behavior.

## 1. Project Identity

FrankenJAX is a memory-safe, clean-room Rust system whose crown-jewel abstraction is:

- Trace Transform Ledger (TTL): canonical IR + transform-composition evidence for `jit`, `grad`, `vmap`.

Legacy behavioral oracle:

- `/dp/frankenjax/legacy_jax_code/jax`
- upstream `jax-ml/jax`

Non-regression imperative:

- transform composition semantics are non-negotiable.

## 2. Prime Directives

FrankenJAX MUST be simultaneously:

1. Behaviorally trustworthy for scoped compatibility.
2. Mathematically explicit in decision and risk handling.
3. Operationally resilient with durable evidence artifacts.
4. Performance-competitive via profile-and-proof optimization.

## 3. Compatibility Doctrine (Mode Split)

### 3.1 Strict mode

- maximize observable compatibility for scoped APIs
- fail closed on unknown incompatible metadata/features
- no behavior-changing repair heuristics

### 3.2 Hardened mode

- preserve outward API contract
- allow bounded defensive handling
- require audit-evidence entries for policy deviations

### 3.3 Compatibility matrix requirement

Every compatibility-sensitive subsystem MUST maintain explicit drift-gate rows,
including at minimum:

- cache-key surface
- transform-order semantics
- shape/dtype canonicalization path
- error-path behavior for malformed metadata

## 4. Architecture Contract

Target architecture:

`user API -> trace -> canonical IR -> transform stack -> lowering -> runtime backend`

Current crate realization:

- `fj-core`
- `fj-lax`
- `fj-interpreters`
- `fj-cache`
- `fj-ledger`
- `fj-dispatch`
- `fj-runtime`
- `fj-conformance`

## 5. Canonical IR and TTL Contract

### 5.1 Canonical representation

TTL MUST carry:

- canonical `Jaxpr`-like program form
- ordered transform stack
- transform evidence references

### 5.2 Determinism invariant

Given identical canonicalized inputs and transform stack, TTL signatures MUST be deterministic.

### 5.3 Composition invariant

Applying transforms in different orders MUST not be treated as equivalent unless proven equivalent by explicit rules.

## 6. Cache-Key Contract

FrankenJAX cache keys MUST be deterministic hashes over a canonicalized input bundle that includes:

- canonical program representation
- transform stack ordering
- backend identity
- normalized compile options
- optional custom hook identity bits
- compatibility-sensitive unknown feature surface

Strict mode MUST reject unknown incompatible features.

Hardened mode MAY proceed with bounded handling, but MUST log evidence and include unknown features in key material.

## 7. Decision and Evidence Ledger Contract

For consequential runtime decisions (admission/fallback/cache policy), the system MUST capture:

1. state summary
2. evidence signals
3. loss matrix
4. posterior/confidence values
5. selected action
6. timestamp and mode

Decision rules SHOULD minimize expected loss under asymmetric error costs.

## 8. Conformance Contract

### 8.1 Fixture families

At minimum, V1 conformance MUST include differential fixture families for:

- `jit`
- `grad`
- `vmap`
- `lax` scoped primitives
- `random` determinism

### 8.2 Oracle anchors

Fixture extraction is anchored to:

- `tests/jax_jit_test.py`
- `tests/lax_autodiff_test.py`
- `tests/lax_vmap_test.py`
- `tests/lax_test.py`
- `tests/random_test.py`

### 8.3 Report artifact

Each conformance run MUST emit a machine-readable parity report containing:

- total cases
- matched cases
- mismatched cases
- mode (strict/hardened)
- fixture family metadata

## 9. Security Contract

Security focus:

- cache confusion resistance
- transform-order vulnerability prevention
- malformed graph/shape signature handling

Minimum controls:

1. fail-closed behavior for unknown incompatible features (strict mode)
2. adversarial fixture coverage for high-risk transitions
3. deterministic audit logs for recovery and override actions
4. explicit threat-model notes for major subsystems

## 10. Performance Contract

Metrics MUST be tracked separately for:

- trace latency
- compile latency
- execute latency

Optimization loop is mandatory:

1. baseline p50/p95/p99 + memory
2. profile hotspot
3. one optimization lever per change
4. behavior-isomorphism verification
5. rebaseline and delta artifact

## 11. RaptorQ-Everywhere Durability Contract

Long-lived artifacts MUST have repair-symbol sidecars:

- conformance fixture bundles
- benchmark baseline bundles
- migration manifests
- reproducibility ledgers
- long-lived state snapshots

Required durability artifacts:

1. symbol generation manifest
2. integrity scrub report
3. decode proof artifact for each recovery event

## 12. Asupersync Integration Contract

FrankenJAX runtime orchestration MUST support:

- cancellation-aware checkpoints
- budget/deadline propagation
- deterministic outcome capture for decision and audit flows

Integration MUST remain capability-explicit and avoid ambient authority.

## 13. FrankenTUI Integration Contract

Operational interfaces SHOULD expose:

- transform stack state
- cache decision state
- parity/benchmark status
- evidence-ledger progressive disclosure

UI components MUST remain optional adapters, not hard runtime dependencies.

## 14. Milestones

### M0 - Foundation (completed)

- canonical IR + TTL data model (including tensor-aware runtime values)
- deterministic transform composition proof checker
- deterministic cache-key module
- evidence-ledger module
- conformance manifest/report scaffolding
- RaptorQ sidecar/scrub/decode-proof pipeline for initial long-lived artifacts

### M1 - Differential Core (in progress)

- first differential fixtures for `jit`/`grad`/`vmap`
- oracle-vs-target parity report generation

### M2 - Coverage Expansion

- broaden primitive coverage
- tighten transform invariants
- add adversarial/property suites

### M3 - Hardening and Durability

- benchmark regression gates
- RaptorQ artifact sidecars and scrub pipeline

## 15. Acceptance Gates

V1 readiness requires all gates green:

1. Compatibility gate: scoped parity report passes.
2. Security gate: adversarial/high-risk checks pass.
3. Performance gate: budgets pass with no semantic regressions.
4. Durability gate: sidecar + scrub + decode-proof artifacts validated.

## 16. Current Residual Risks

1. Transform semantics are validated against current fixture bundle but not yet against full strict legacy capture from real `jax` + `jaxlib`.
2. Cache-key surface is scaffolded but not fully aligned with all legacy hash components/signals.
3. RaptorQ durability is implemented for initial evidence artifacts but not yet expanded to all required long-lived bundles.
4. Asupersync/FrankenTUI bridges are foundational and need deeper runtime/UI integration.
