# TODO_SUPER_DETAILED_PORT_AND_PARITY

Last updated: 2026-05-01
Owner: Codex session
Goal: Complete all identified next-step streams end-to-end in one pass.

## Legend

- [ ] not started
- [~] in progress
- [x] completed
- [!] blocked/needs follow-up

## 2026-05-01 Reality-Check Follow-Up TODO

Tracker parent: `frankenjax-fcxy`

### RC0. Tracker and coordination
- [x] Confirm live tracker state before adding work.
- [x] Verify open queue was empty before creating follow-up beads.
- [x] Create parent epic `frankenjax-fcxy`.
- [x] Create docs/status reconciliation bead `frankenjax-fcxy.1`.
- [x] Create Phase2C topology/durability bead `frankenjax-fcxy.2`.
- [x] Create TTL verification hardening bead `frankenjax-fcxy.3`.
- [x] Create composed-grad fallback bead `frankenjax-fcxy.4`.
- [x] Create global performance gate bead `frankenjax-fcxy.5`.
- [x] Claim `frankenjax-fcxy.1` first.
- [!] Agent Mail write-path bootstrap failed with a database error; continue with `br` and local file discipline.

### RC1. Live evidence inventory
- [x] Confirm current HEAD short hash and commit subject.
- [x] Count workspace crates.
- [x] Count `Primitive` variants.
- [x] Count `DType` variants.
- [x] Count Rust source lines under `crates/`.
- [x] Count static Rust test/proptest markers.
- [x] Count conformance test files.
- [x] Count transform fixture cases.
- [x] Count RNG fixture cases.
- [x] Count linalg/FFT fixture cases.
- [x] Count transform-composition fixture cases.
- [x] Count dtype-promotion fixture cases.
- [x] Run `cargo test --workspace` through `rch`.

### RC2. README reality reconciliation (`frankenjax-fcxy.1`)
- [x] Replace stale 846 fixture claims with live 848 fixture count.
- [x] Replace stale 4,280 test claim with live static-marker wording.
- [x] Replace over-broad "all green" status language where a tracked limitation remains.
- [x] Clarify TTL as auditable evidence/provenance rather than a formal semantic proof.
- [x] Clarify RaptorQ durability as implemented for current evidence bundles, with all-long-lived-artifact expansion still tracked.
- [x] Preserve explicit limitations: CPU-only, no XLA lowering, advanced `vmap` plus iterative-control-flow compositions still incomplete.
- [x] Update fixture-family table totals.
- [x] Update FAQ oracle fixture count.

### RC3. Feature parity reconciliation (`frankenjax-fcxy.1`)
- [x] Update audit timestamp and command evidence.
- [x] Add a reality-check gap tracker section with bead IDs.
- [x] Update fixture and test evidence counts.
- [x] Downgrade TTL, dispatch composition, `vmap`, control flow, and RaptorQ rows where the next artifact is a real parity gap rather than routine expansion.
- [x] Keep declared V1 successes marked green only where live evidence supports them.

### RC4. CHANGELOG reconciliation (`frankenjax-fcxy.1`)
- [x] Update latest commit from stale March hash to current HEAD.
- [x] Update current state numbers to live counts.
- [x] Add 2026-05-01 reality-check recalibration entry.
- [x] List follow-up beads that now own remaining gaps.

### RC5. Phase2C topology completion (`frankenjax-fcxy.2`)
- [ ] Claim `frankenjax-fcxy.2` after `frankenjax-fcxy.1` lands.
- [ ] Build packet inventory for `FJ-P2C-001` through `FJ-P2C-008`.
- [ ] Compare each packet against required topology in `PHASE2C_EXTRACTION_PACKET.md`.
- [ ] Add or update missing `fixture_manifest` artifacts.
- [ ] Add or update missing `parity_gate` artifacts.
- [ ] Add or update missing `parity_report` artifacts.
- [ ] Add or update missing `risk_note` artifacts.
- [ ] Add or update sidecar/proof manifest references.
- [ ] Add a topology coverage test or machine-readable checker if the existing harness has a natural home.
- [ ] Run targeted artifact/schema tests.

### RC6. TTL verification hardening (`frankenjax-fcxy.3`)
- [ ] Claim `frankenjax-fcxy.3` after Phase2C docs/artifacts are stable or if code work becomes the narrower next slice.
- [ ] Inspect `TraceTransformLedger` construction APIs.
- [ ] Identify what evidence strings currently encode for `jit`, `grad`, `vmap`, nested transforms, and empty stacks.
- [ ] Define a stronger valid-evidence contract that preserves existing valid ledgers.
- [ ] Reject evidence entries that do not bind to the expected transform.
- [ ] Reject duplicate/stale evidence where the stack signature does not match.
- [ ] Keep composition proof hashing deterministic.
- [ ] Add unit tests for valid single transforms.
- [ ] Add unit tests for valid nested transforms.
- [ ] Add rejection tests for mismatched transform evidence.
- [ ] Add rejection tests for stale stack signatures.
- [ ] Run `cargo test -p fj-core transform_composition`.

### RC7. Composed-grad fallback (`frankenjax-fcxy.4`)
- [ ] Claim `frankenjax-fcxy.4` after TTL hardening or if dispatch proves independently tractable.
- [ ] Inspect `execute_grad` tail-transform fallback behavior in `fj-dispatch`.
- [ ] Identify strict-mode versus hardened-mode behavior knobs available in dispatch options.
- [ ] Prefer symbolic composition for tractable `grad(jit(...))` or `jit(grad(...))` cases.
- [ ] If symbolic coverage cannot be completed in one slice, make finite difference opt-in or hardened-only.
- [ ] Add error messaging that explains non-strict fallback requirements.
- [ ] Add regression test for composed grad with tail `jit`.
- [ ] Add regression test for composed grad with tail `vmap` where supported.
- [ ] Add conformance coverage or explicit limitation entry for unsupported cases.
- [ ] Run `cargo test -p fj-dispatch grad`.

### RC8. Global performance gates (`frankenjax-fcxy.5`)
- [ ] Claim `frankenjax-fcxy.5` after correctness gates are not moving.
- [ ] Inventory existing Criterion benchmarks and performance artifacts.
- [ ] Define schema for trace, compile/dispatch, execute, cold-cache, warm-cache, and memory fields.
- [ ] Capture or normalize current baselines without inventing unmeasured speed claims.
- [ ] Add parse/coverage tests for baseline artifacts.
- [ ] Document exact `rch`/Criterion commands for re-baselining.
- [ ] Add pass/fail tolerance policy that distinguishes noise from regression.
- [ ] Record an isomorphism proof template for any actual optimization lever.

### RC9. Verification and landing for this follow-up
- [ ] Run `cargo fmt --check`.
- [ ] Run `cargo check --workspace --all-targets` through `rch`.
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings` through `rch`.
- [ ] Run targeted tests for touched crates.
- [ ] Run `cargo test --workspace` through `rch`.
- [ ] Run `ubs` on changed source/docs files if applicable.
- [ ] Close completed beads.
- [ ] Leave unfinished beads open with exact next steps.
- [ ] Run `br sync --flush-only`.
- [ ] Do not stage or commit unrelated dirty artifact/log files.

## Stream A: Legacy Fixture Capture + Conformance Wiring

### A0. Task management and design lock
- [x] Create main TODO tracker in-repo.
- [x] Freeze fixture schema fields and versioning strategy.
- [x] Define minimal first-pass case family set (`jit`, `grad`, `vmap`) with composition cases.
- [x] Add TODO sync points after each major code edit cluster.

### A1. Legacy oracle capture script
- [x] Create capture script path (`crates/fj-conformance/scripts/capture_legacy_fixtures.py`).
- [x] Implement robust JAX import strategy using local legacy tree path.
- [x] Implement conversion helpers (arrays/scalars -> JSON fixture values).
- [x] Implement case executions:
  - [x] `jit_add_scalar`
  - [x] `jit_compose_grad_square_plus_linear`
  - [x] `grad_square_scalar`
  - [x] `grad_square_plus_linear_scalar`
  - [x] `vmap_add_one_vector`
  - [x] `vmap_grad_square_vector`
- [x] Emit deterministic JSON ordering and metadata header.
- [x] Add CLI flags (`--output`, `--strict`, `--skip-existing`).
- [x] Add script-level fallback path when `jax/jaxlib` unavailable.
- [x] Add strict-mode hard-fail behavior.

### A2. Fixture bundle materialization
- [x] Add committed fixture bundle JSON for deterministic CI runs.
- [x] Ensure fixture values align with script output schema.
- [x] Add fixture README docs for regeneration.

### A3. Rust conformance harness expansion
- [x] Add fixture data model structs in `fj-conformance`.
- [x] Implement fixture loader + schema validation.
- [x] Implement case executor via `fj-dispatch`.
- [x] Implement numeric comparison policy (absolute + relative tolerance).
- [x] Produce machine-readable parity report artifact struct.
- [x] Add parity family summary aggregation.

### A4. Conformance tests and reports
- [x] Add integration test for full transform fixture suite.
- [x] Add test for parity mismatch detection path.
- [x] Add fixture parsing + conversion tests.
- [x] Add report serialization/deserialization test path.

## Stream B: Core IR / Interpreter Expansion + Transform Invariants

### B0. Type-system and data model design
- [x] Introduce first-class runtime `Value` model (scalar + tensor).
- [x] Introduce tensor value struct with shape/data validation.
- [x] Preserve existing literal model for jaxpr constants.
- [x] Ensure serde derives for fixture compatibility.

### B1. Core invariants
- [x] Add transform composition invariant checker API.
- [x] Add invariant errors:
  - [x] evidence cardinality mismatch
  - [x] empty evidence entry rejection
  - [x] unsupported transform sequence for current engine capabilities
- [x] Add transform composition proof artifact struct.

### B2. Primitive semantics expansion
- [x] Extend `fj-lax` primitives to `Value` operations.
- [x] Support scalar-scalar operations.
- [x] Support tensor-tensor elementwise operations.
- [x] Support scalar-tensor broadcasting.
- [x] Support `reduce_sum` on scalar/tensor values.
- [x] Support vector `dot` semantics.
- [x] Add dimensionality/type mismatch errors.

### B3. Interpreter expansion
- [x] Update `fj-interpreters` to evaluate `Value` outputs.
- [x] Preserve deterministic env ordering and behavior.
- [x] Add tests for tensor arithmetic and broadcasting.

### B4. Dispatch transform execution model
- [x] Update dispatch request/response to `Value` args/outputs.
- [x] Insert transform invariant validation before execution.
- [x] Implement transform wrapper execution order semantics:
  - [x] `jit` (identity wrapper with evidence)
  - [x] `grad` (numerical gradient wrapper for scalar first arg)
  - [x] `vmap` (map over first-axis tensor values)
- [x] Ensure order-sensitive composition behavior is explicit and test-covered.
- [x] Add dispatch tests for:
  - [x] `jit`
  - [x] `grad`
  - [x] `vmap`
  - [x] composition order behavior

### B5. Runtime/ledger integration
- [x] Attach transform proof data to evidence ledger signals.
- [x] Ensure strict/hardened mode continues to propagate.

## Stream C: RaptorQ Sidecars + Scrub + Decode Proof

### C0. Durability architecture in code
- [x] Add durability module in `fj-conformance`.
- [x] Add sidecar schema (`manifest + symbols + integrity metadata`).
- [x] Add scrub report schema.
- [x] Add decode proof schema.

### C1. Asupersync RaptorQ integration
- [x] Add dependency wiring for `asupersync` in conformance crate.
- [x] Implement `encode_artifact_with_sidecar` using `EncodingPipeline`.
- [x] Implement `decode_artifact_from_sidecar` using `DecodingPipeline`.
- [x] Implement canonical hash verification.

### C2. Scrub pipeline
- [x] Implement sidecar integrity scrub function.
- [x] Validate symbol metadata consistency checks.
- [x] Validate full decode and hash match.
- [x] Emit structured scrub report.

### C3. Recovery + decode proof
- [x] Implement lossy recovery simulation (drop source symbols, recover from retained set).
- [x] Implement fallback recovery path (repair-drop proof if needed).
- [x] Emit decode proof artifact with missing symbol set and recovery result.
- [x] Add failure-capable proof shape for insufficient-symbol events.

### C4. CLI and artifact generation
- [x] Add CLI tool/binary for sidecar generation + scrub + proof output.
- [x] Wire generation for:
  - [x] `crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json`
  - [x] `artifacts/performance/dispatch_baseline_2026-02-13.md`
- [x] Commit generated sidecar/manifests/reports.

### C5. Durability tests
- [x] Add round-trip sidecar encode/decode unit tests.
- [x] Add scrub positive-path test.
- [x] Add decode-proof recovery test with simulated loss.

## Stream D: Docs / Parity Matrix / TODO hygiene

### D0. Spec and parity docs
- [x] Update `FEATURE_PARITY.md` statuses with newly completed items.
- [x] Update `PROPOSED_ARCHITECTURE.md` with durability implementation details.
- [x] Update `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md` implementation progress section.

### D1. Operational docs
- [x] Document fixture regeneration command.
- [x] Document sidecar generation/scrub commands.
- [x] Document known current limitations and exact next gaps.

### D2. TODO maintenance
- [x] Mark completed tasks as work progresses.
- [x] Add newly discovered subtasks immediately.
- [x] Perform final TODO integrity pass before finishing.

## Stream E: Verification and landing

### E0. Required checks
- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p fj-conformance -- --nocapture`
- [x] `cargo bench`

### E1. Method-stack artifact checks
- [x] Confirm alien-artifact evidence artifacts produced/updated.
- [x] Confirm extreme optimization baseline/proof artifacts produced/updated.
- [x] Confirm RaptorQ sidecar artifacts produced/updated.
- [x] Confirm compatibility-security drift gates reflected in code/docs.

### E2. Session landing
- [x] Confirm no destructive commands were executed.
- [x] Summarize changes with rationale.
- [x] List residual risks and highest-value next steps.
- [x] Ensure TODO file reflects final reality with no stale statuses.
