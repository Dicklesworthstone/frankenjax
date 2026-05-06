# TODO_SUPER_DETAILED_PORT_AND_PARITY

Last updated: 2026-05-01
Owner: Codex session
Goal: Complete all identified next-step streams end-to-end in one pass.

## Legend

- [ ] not started
- [~] in progress
- [x] completed
- [!] blocked/needs follow-up

## 2026-05-01 Full Reality Check, Ambition Rounds, and Bead Plan

Tracker parent: `frankenjax-cstq`

### RCB0. Skill protocol and source ingestion
- [x] Apply `reality-check-for-project` instead of jumping directly to implementation.
- [x] Read global `/dp/AGENTS.md`.
- [x] Read repo-local `AGENTS.md`.
- [x] Read `README.md` fully, including current limitations and verification sections.
- [x] Read `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`.
- [x] Read `PLAN_TO_PORT_JAX_TO_RUST.md`.
- [x] Read `PROPOSED_ARCHITECTURE.md`.
- [x] Read `FEATURE_PARITY.md`.
- [x] Read `PHASE2C_EXTRACTION_PACKET.md`.
- [x] Read `EXISTING_JAX_STRUCTURE.md`.
- [x] Read `EXHAUSTIVE_LEGACY_ANALYSIS.md`.
- [x] Read `CHANGELOG.md`.
- [x] Read `UPGRADE_LOG.md`.
- [x] Read conformance fixture README notes.
- [x] Confirm current tracker state before creating new beads: `br ready --json` returned `[]`.
- [x] Confirm `bv --robot-triage` before this pass reported 314 closed issues, 0 open issues, and no ready work.
- [!] Agent Mail write-path bootstrap still fails with a database error; use `br`, JSONL, git status, and scoped file edits instead.

### RCB1. Reality-check answer
- [x] Current implementation is substantial, not a skeleton: 15 workspace crates, 115 declared primitives, strict/hardened infrastructure, conformance harnesses, many E2E/durability artifacts, and recently green workspace validation evidence.
- [x] Current implementation is not yet "100 percent V1 reality" because several spec-level promises still need stronger machine evidence or actual closure.
- [x] The right next work is not another broad rewrite. It is a focused graph of oracle recapture, transform composition, semantic proof, memory/perf gating, durability inventory, cache-key parity, facade/API proof, and audit beads.
- [x] GPU/TPU/XLA replacement remains excluded scope, not a failure, unless future docs claim it as V1.
- [x] CPU-only/no-XLA limitations remain correctly documented and should stay explicit.

### RCB2. Vision checklist against current evidence

| Vision claim | Current reality | Gap owner |
|---|---|---|
| Clean-room Rust implementation of scoped JAX transform semantics | Broadly implemented for declared primitives and core transforms | `frankenjax-cstq.1`, `frankenjax-cstq.2`, `frankenjax-cstq.7` |
| Transform composition semantics are non-negotiable | Evidence/proof checks exist and were hardened, but semantic replay against oracle structures is not yet complete | `frankenjax-cstq.2`, `frankenjax-cstq.3` |
| Strict/hardened compatibility split is enforced | Present in cache/dispatch paths, but cross-crate error taxonomy is distributed | `frankenjax-cstq.6`, `frankenjax-cstq.8` |
| Profile-proven performance with trace/compile/execute and memory | Global gate exists for measured runtime phases and measured RSS memory workloads; hotspot prioritization remains tracked separately | `frankenjax-cstq.4`, `frankenjax-cstq.11` |
| RaptorQ sidecars for all long-lived artifacts | Many sidecars/proofs exist, including Phase2C packets and E2E artifacts; authoritative all-artifact coverage needs machine enforcement | `frankenjax-cstq.5` |
| Cache-key soundness against legacy behavior | Deterministic Rust keying exists; component-by-component legacy ledger is still needed | `frankenjax-cstq.6` |
| Top-level usable library/API story | Multiple crates and examples exist; top-level facade and README executable proof remain ambiguous | `frankenjax-cstq.9`, `frankenjax-cstq.12` |
| Backend, FFI, asupersync, FrankenTUI integrations | Foundational/optional paths exist; feature-combination and external-contract proof needs tightening | `frankenjax-cstq.10` |
| No hidden stubs or stale proof claims | Prior docs were recalibrated, but a fresh stub/dead-claim audit is still needed after the latest ambition graph | `frankenjax-cstq.13` |
| Docs/specs/beads remain aligned over time | Manual docs are currently updated, but no machine-checked vision-to-evidence coverage dashboard exists | `frankenjax-cstq.14` |

### RCB3. Gap inventory converted to bridge lanes
- [x] Lane A: Evidence truth source and drift detection.
  - [x] `frankenjax-cstq.1` recaptures oracle matrix and drift gates.
  - [x] `frankenjax-cstq.14` later makes vision-to-evidence coverage machine checked.
- [x] Lane B: Transform correctness and proof closure.
  - [x] `frankenjax-cstq.2` closes advanced transform/control-flow composition parity.
  - [x] `frankenjax-cstq.3` adds semantic TTL replay instead of relying on provenance only.
  - [x] `frankenjax-cstq.7` expands higher-rank and complex AD/linalg/FFT fixture-backed parity.
  - [x] `frankenjax-cstq.8` centralizes strict/hardened error taxonomy conformance.
- [x] Lane C: Performance, memory, and durability truth.
  - [x] `frankenjax-cstq.4` replaces `memory: not_measured` with real heap/RSS/allocation evidence.
  - [x] `frankenjax-cstq.5` enforces RaptorQ coverage for all long-lived artifacts.
  - [x] `frankenjax-cstq.11` ranks optimization work only after measured baseline/profile evidence.
- [x] Lane D: User-facing and architecture closure.
  - [x] `frankenjax-cstq.12` decides whether target crate boundaries should be extracted or explicitly retained.
  - [x] `frankenjax-cstq.9` proves documented user entry points and README examples.
  - [x] `frankenjax-cstq.10` proves backend, FFI, and optional feature contracts.
- [x] Lane E: Fresh audit.
  - [x] `frankenjax-cstq.13` searches for hidden mocks, stubs, stale claims, and false "done" surfaces.

### RCB4. Phase 3a bead creation pass
- [x] Create parent epic `frankenjax-cstq`.
- [x] Create `frankenjax-cstq.1` for oracle parity recapture matrix and drift gate.
- [x] Create `frankenjax-cstq.2` for advanced transform/control-flow composition parity.
- [x] Create `frankenjax-cstq.3` for TTL semantic verifier and structural oracle replay.
- [x] Create `frankenjax-cstq.4` for memory/allocation performance gate.
- [x] Create `frankenjax-cstq.5` for all-long-lived-artifact RaptorQ coverage.
- [x] Create `frankenjax-cstq.6` for cache-key legacy parity ledger and lifecycle conformance.
- [x] Create `frankenjax-cstq.7` for higher-rank and complex AD/linalg/FFT fixtures.
- [x] Create `frankenjax-cstq.8` for cross-crate error taxonomy conformance.
- [x] Create `frankenjax-cstq.9` for top-level user facade and README executable proof.
- [x] Create `frankenjax-cstq.10` for backend, FFI, and optional integration contracts.
- [x] Create `frankenjax-cstq.11` for optimization hotspot scoreboard and one-lever queue.
- [x] Create `frankenjax-cstq.12` for transform/lowering/API facade boundary decision.
- [x] Create `frankenjax-cstq.13` for mock/stub/dead-claim audit.
- [x] Create `frankenjax-cstq.14` for machine-checked vision-to-evidence coverage dashboard.
- [x] Mark parent epic `frankenjax-cstq` as `in_progress` after child graph creation; it stays open until the child program is executed.

### RCB5. Ambition Round 1: make claims machine-checkable
- [x] Strengthen the plan from "update docs" to "add checks that reject unsupported claims."
- [x] Require `frankenjax-cstq.1` to emit drift reports, not just recapture fixtures.
- [x] Require `frankenjax-cstq.5` to define an authoritative artifact inventory, not just generate more sidecars.
- [x] Require `frankenjax-cstq.14` to connect spec MUST clauses, parity rows, tests, artifacts, and bead IDs.

### RCB6. Ambition Round 2: promote parity claims only after oracle or fail-closed evidence
- [x] Keep advanced `vmap` plus iterative control flow as a real open parity lane until oracle matrix rows pass.
- [x] Require unsupported transform rows to fail closed explicitly, not silently degrade.
- [x] Require TTL semantic verification to bind transformed IR/evidence to structural oracle replay.
- [x] Require cache-key parity to map every legacy component to Rust material or explicit excluded scope.

### RCB7. Ambition Round 3: optimize only from measured bottlenecks
- [x] Preserve the existing performance doctrine: baseline, profile, one lever, behavior proof, rebaseline.
- [x] Treat memory/allocation as a missing measurement phase, not as an optimization claim.
- [x] Add an optimization scoreboard bead before creating more speedup beads.
- [x] Include vmap multiplier, AD tape/backward map, tensor materialization, shape kernels, cache hashing, egraph saturation, FFT/linalg/reduction mixes, and durability encode/decode in the scoreboard scope.

### RCB8. Phase 5 plan-space refinement rounds
- [x] Round 1: bv found the created graph cycle-free but too flat; add real sequencing edges.
- [x] Round 2: ensure oracle recapture blocks advanced transform, TTL proof, and higher-rank fixture work.
- [x] Round 3: ensure memory baseline blocks optimization scoreboard work.
- [x] Round 4: ensure architecture-boundary decision blocks facade/API proof, which blocks backend/FFI optional-contract proof.
- [x] Round 5: ensure final vision dashboard depends on oracle, durability, cache, and error-taxonomy evidence.
- [x] Stop refinement after no additional dependency edge is necessary for the current gap model.

### RCB9. Refined dependency graph
- [x] `frankenjax-cstq.2` depends on `frankenjax-cstq.1`.
- [x] `frankenjax-cstq.3` depends on `frankenjax-cstq.1`.
- [x] `frankenjax-cstq.3` depends on `frankenjax-cstq.2`.
- [x] `frankenjax-cstq.7` depends on `frankenjax-cstq.1`.
- [x] `frankenjax-cstq.9` depends on `frankenjax-cstq.12`.
- [x] `frankenjax-cstq.10` depends on `frankenjax-cstq.9`.
- [x] `frankenjax-cstq.11` depends on `frankenjax-cstq.4`.
- [x] `frankenjax-cstq.14` depends on `frankenjax-cstq.1`.
- [x] `frankenjax-cstq.14` depends on `frankenjax-cstq.5`.
- [x] `frankenjax-cstq.14` depends on `frankenjax-cstq.6`.
- [x] `frankenjax-cstq.14` depends on `frankenjax-cstq.8`.

### RCB10. Next implementation order from the refined graph
- [ ] First ready slice: `frankenjax-cstq.1` oracle parity recapture matrix and drift gate.
- [ ] Parallel-safe ready slices after/around it: `frankenjax-cstq.4`, `frankenjax-cstq.5`, `frankenjax-cstq.6`, `frankenjax-cstq.8`, `frankenjax-cstq.12`, `frankenjax-cstq.13`.
- [ ] After `frankenjax-cstq.1`: proceed to `frankenjax-cstq.2` and `frankenjax-cstq.7`.
- [ ] After `frankenjax-cstq.2`: proceed to `frankenjax-cstq.3`.
- [ ] After `frankenjax-cstq.4`: proceed to `frankenjax-cstq.11`.
- [ ] After `frankenjax-cstq.12`: proceed to `frankenjax-cstq.9`, then `frankenjax-cstq.10`.
- [ ] After oracle/durability/cache/error evidence: proceed to `frankenjax-cstq.14`.

### RCB11. Validation checklist for this planning/bead phase
- [x] Re-run `br dep cycles --json`; result: no cycles.
- [x] Re-run `br ready --json`; result: ready children are `frankenjax-cstq.1`, `.4`, `.5`, `.6`, `.8`, `.12`, `.13`.
- [x] Re-run `bv --robot-triage`; result: 15 open, 8 actionable, 7 blocked, 1 in progress, top pick `frankenjax-cstq.1`.
- [x] Run `git diff --check`.
- [!] Run `br sync --flush-only`; it still refuses export because the DB has 328 issues while JSONL has 329 and the stale guard says export would lose `frankenjax-h8l2`. Do not force without explicit permission; use the auto-updated JSONL records from `br create`/`br dep add`/`br update`.
- [x] Stage `TODO_SUPER_DETAILED_PORT_AND_PARITY.md` and `.beads/` changes only.
- [x] Commit and push to `main`.
- [x] Mirror `main` to legacy `master`.
- [x] Leave unrelated untracked fuzz directory untouched.

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
- [x] Claim `frankenjax-fcxy.2` after `frankenjax-fcxy.1` lands.
- [x] Build packet inventory for `FJ-P2C-001` through `FJ-P2C-008`.
- [x] Compare each packet against required topology in `PHASE2C_EXTRACTION_PACKET.md`.
- [x] Add or update missing `fixture_manifest` artifacts.
- [x] Add or update missing `parity_gate` artifacts.
- [x] Add or update missing `parity_report` artifacts.
- [x] Add or update missing `risk_note` artifacts.
- [x] Add or update sidecar/proof manifest references.
- [x] Add a topology coverage test or machine-readable checker if the existing harness has a natural home.
- [x] Run targeted artifact/schema tests.

### RC6. TTL verification hardening (`frankenjax-fcxy.3`)
- [x] Claim `frankenjax-fcxy.3` after Phase2C docs/artifacts are stable or if code work becomes the narrower next slice.
- [x] Inspect `TraceTransformLedger` construction APIs.
- [x] Identify what evidence strings currently encode for `jit`, `grad`, `vmap`, nested transforms, and empty stacks.
- [x] Define a stronger valid-evidence contract that preserves existing valid ledgers.
- [x] Reject evidence entries that do not bind to the expected transform.
- [x] Reject duplicate/stale evidence where the stack signature does not match.
- [x] Keep composition proof hashing deterministic.
- [x] Add unit tests for valid single transforms.
- [x] Add unit tests for valid nested transforms.
- [x] Add rejection tests for mismatched transform evidence.
- [x] Add rejection tests for stale stack signatures.
- [x] Run `cargo test -p fj-core transform_composition`.

### RC7. Composed-grad fallback (`frankenjax-fcxy.4`)
- [x] Claim `frankenjax-fcxy.4` after TTL hardening or if dispatch proves independently tractable.
- [x] Inspect `execute_grad` tail-transform fallback behavior in `fj-dispatch`.
- [x] Identify strict-mode versus hardened-mode behavior knobs available in dispatch options.
- [x] Prefer symbolic composition for tractable `grad(jit(...))` or `jit(grad(...))` cases.
- [x] If symbolic coverage cannot be completed in one slice, make finite difference explicitly gateable.
- [x] Add error messaging that explains non-strict fallback requirements.
- [x] Add regression test for composed grad with tail `jit`.
- [x] Add regression test for composed grad with tail `vmap` where supported.
- [x] Add conformance coverage or explicit limitation entry for unsupported cases.
- [x] Run `cargo test -p fj-dispatch grad`.

### RC8. Global performance gates (`frankenjax-fcxy.5`)
- [x] Claim `frankenjax-fcxy.5` after correctness gates are not moving.
- [x] Inventory existing Criterion benchmarks and performance artifacts.
- [x] Define schema for trace, compile/dispatch, execute, cold-cache, warm-cache, and memory fields.
- [x] Capture or normalize current baselines without inventing unmeasured speed claims.
- [x] Add parse/coverage tests for baseline artifacts.
- [x] Document exact `rch`/Criterion commands for re-baselining.
- [x] Add pass/fail tolerance policy that distinguishes noise from regression.
- [x] Record an isomorphism proof template for any actual optimization lever.

### RC9. Verification and landing for this follow-up
- [x] Run `cargo fmt --check`.
- [x] Run `cargo check --workspace --all-targets` through `rch`.
- [x] Run `cargo clippy --workspace --all-targets -- -D warnings` through `rch`.
- [x] Run targeted tests for touched crates.
- [x] Run `cargo test --workspace` through `rch`.
- [!] Run `ubs` on changed source/docs files if applicable; full changed-file scan exits 1 on test-harness `expect`/`panic`/`assert` inventory in `artifact_schemas.rs`, while the changed production file `crates/fj-dispatch/src/lib.rs` exits 0 with no critical findings.
- [x] Close completed beads.
- [x] Leave unfinished beads open with exact next steps; no `frankenjax-fcxy` child beads remain open.
- [!] Run `br sync --flush-only`; attempted after `br sync --import-only` and `br sync --rebuild`, but `br` still refuses export because its stale guard claims `frankenjax-h8l2` is missing even though JSONL/no-DB and `br list --status closed` both show it. Current `.beads/issues.jsonl` contains the closed `frankenjax-fcxy` and `frankenjax-fcxy.5` records.
- [x] Do not stage or commit unrelated dirty artifact/log files.

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
