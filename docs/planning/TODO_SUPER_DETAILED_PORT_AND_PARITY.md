# TODO_SUPER_DETAILED_PORT_AND_PARITY

Last updated: 2026-10-13 (`HEAD = 4f04dd82ace83f609eedd232839244ce872bb4ce`)
Owner: Antigravity session (`0bfbe145-820c-4bbf-848d-90bf02295c16`)
Goal: End-to-end `/reality-check-for-project` audit, Bridge Plan, Ambition Rounds (1–3), Bead Creation (`frankenjax-rc26-n1emn`), and Plan-Space Refinement (Rounds 1–5).

## Legend

- [ ] not started
- [~] in progress
- [x] completed
- [!] blocked/needs follow-up

## 2026-10-13 Comprehensive Reality Check, Bridge Plan, Ambition Rounds (1–3), and Refined Bead Graph

Tracker parent: `frankenjax-rc26-n1emn` (`in_progress`)
Companion Vision vs Reality Matrix: `artifacts/conformance/vision_reality_checklist.v1.md`

### RC26-0. Phase 0 Source Ingestion & Ground-Truth Audit
- [x] Read `/data/projects/AGENTS.md` (all 664 lines, including Named Reward-Hacking Patterns and Work-Graph Discipline).
- [x] Read `/data/projects/frankenjax/AGENTS.md` (all 867 lines).
- [x] Read `/data/projects/frankenjax/README.md` (all 1,213 lines).
- [x] Read `/data/projects/jeffreys-skills-private/.claude/skills/reality-check-for-project/SKILL.md` and all reference guides (`GAP-ANALYSIS.md`, `BRIDGE-PLAN.md`, `VISION-EXTRACTION.md`, `AMBITION-ROUNDS.md`, `LESSONS-FROM-SESSIONS.md`, `SELF-TEST.md`).
- [x] Read all planning and evidence documents in `docs/planning/`, `CHANGELOG.md`, `docs/NEGATIVE_EVIDENCE.md`, and `artifacts/conformance/vision_reality_checklist.v1.md`.
- [x] Audit every crate in `crates/` (`fj-core`, `fj-trace`, `fj-interpreters`, `fj-dispatch`, `fj-lax`, `fj-cache`, `fj-runtime`, `fj-conformance`, `fj-ledger`, `fj-ad`, `fj-egraph`, `fj-test-utils`, `fj-api`, `fj-backend-cpu`, `fj-backend-gpu`, `fj-ffi`, `fj-py`) and all gate/durability artifacts on disk.

### RC26-1. Phase 1 Reality Check Answers (Code vs. Vision Gap Analysis)
- [x] **What IS genuinely working:**
  - **17 workspace crates** (`Cargo.toml:4-22`), **321,303 source LOC** in `crates/*/src/**/*.rs` (**459,828 total Rust LOC** in `crates/**/*.rs`), **9,355 `#[test]` functions** + **23 `proptest!` blocks** across **173 conformance test files**, and **861 JAX 0.9.2 oracle fixture cases** across 5 fixture bundles.
  - **162 canonical `Primitive` variants** (`crates/fj-core/src/lib.rs:78-435`): all **157 V1 local primitives** have real pure-Rust evaluators in `fj-lax` (`157,522` LOC across 30 files), and the 5 `pmap` collectives fail closed with typed `EvalError::Unsupported`.
  - **1st-order eager AD:** 115 primitives have analytical VJP + JVP rules in `fj-ad`, plus 35 non-differentiable primitives returning `zeros_like`.
  - **Compiled slot-indexed interpreter (`frankenjax-so4wo`):** `CompiledJaxpr::eval` in `crates/fj-interpreters/src/lib.rs` executes scalar/small-tensor programs in `~180.6 ns`, and `fj-lax` beats JAX 0.10.2 XLA-CPU by `1.2x–230x` on QR (`30x`), LU (`12.9x`), slogdet/det (`7.3x/5.9x`), solve (`5.2x`), radix sort/argsort (`8x–94x`), 4D window pooling (`17x–141x`), i64 dot (`230x`), and cumsum/cumprod (`3.1x–3.9x`).
- [x] **What is NOT working, disconnected, theatrical, or broken in code/docs:**
  1. **Compiler Lint Gate & Zero-Panic Library Safety Are Broken (`rc26.1`, `rc26.2`, `rc26.3`):**
     - `rch exec -- cargo clippy --workspace --all-targets -- -D warnings` fails with **125 errors** in `fj-lax` (123 `clippy::doc_lazy_continuation` + `benches/i64_matmul_speed.rs:245` + `crates/fj-lax/src/fft.rs:1664`).
     - Contrary to `README.md:1107` (*"No panics in library code... The only unwrap() calls are in test code"*), non-test library code contains **28 `.unwrap()`**, **48 `.expect()`**, **5 `panic!()`**, **17 `unreachable!()`**, and **24 `assert!()`** calls, plus benchmark modules `cz0g0_f32accum_evidence.rs` and `cz0g0_fma_evidence.rs` compiled into production `fj-lax` (`crates/fj-lax/src/lib.rs:32-35`).
     - Documentation across `README.md`, `AGENTS.md`, `FEATURE_PARITY.md`, `PARITY-COVERAGE.md`, and `CHANGELOG.md` contradicts itself on crate counts (`8`/`11`/`15`/`16` vs `17`), primitive counts (`110`/`118`/`157` vs `162`), e-graph rules (`86`/`87` vs `30 safe / 91 total`), LOC, and test counts; `scripts/validate_feature_parity.sh` and `scripts/bootstrap_vision_evidence_map.sh` fail when run.
  2. **Tracing (`fj-trace`) & Public API (`fj-api`) Cannot Trace Most of the Language (`rc26.4`, `rc26.5`, `rc26.6`, `rc26.7`):**
     - `crates/fj-api/src/transforms.rs:554-662` (`eval_shape`) does not call `fj_lax::eval_abstract_primitive`; it uses a broken 37-op match (wrong dtype for comparisons, wrong shape for axis reductions and permuted transpose) and a wildcard `_ => Ok(vec![inputs[0].clone()])` for the remaining 125 primitives!
     - `crates/fj-trace/src/lib.rs:4517-4838` (`TracerRef`) has **0 named primitive methods**, only 4 operator traits (`Add`, `Sub`, `Mul`, `Neg` — no `Div`/`Rem`), cannot trace `Atom::Lit` constants (`L2541-2545`), fails on 0-input primitives (`Iota`, `BroadcastedIota` at `L1582, 1620`), hardcodes `sub_jaxprs: vec![]` (`L2643`), and has a 22-op wildcard `_ => Ok(vec![inputs[0].clone()])` in `infer_primitive_output_avals` (`L2431-2436`).
     - `fj_lax::{nn, threefry, tree_util}` are standalone eager `&[f64]` / `TreeNode::Leaf(f64)` functions disconnected from `Primitive`, `fj-trace`, and `fj-api`.
  3. **AD Composition Is Eager Tape + Numerical Finite Differences (`1e-5`/`1e-6`), Not a Symbolic `Jaxpr -> Jaxpr` Transform (`rc26.8`, `rc26.9`, `rc26.10`):**
     - `fj-ad` evaluates a `Jaxpr` eagerly into `Vec<Value>` via a runtime `Tape` and never produces a derivative `Jaxpr`.
     - In `crates/fj-dispatch/src/lib.rs:777-786, 830-876`, whenever `tail` contains a non-`Jit` transform (`[Grad]` or `[Vmap]`), `execute_grad` falls back to **`1e-6` central finite differences** (`execute_grad_finite_diff`), which only perturbs `args[0]` and rejects tensors (`NonScalarGradientInput`).
     - **`grad(vmap(f))` ALWAYS fails** with `NonScalarGradientInput` (`L781`), `vmap(grad(f))` bypasses `BatchTrace` (`L983`) for a sequential `O(batch)` loop (`L1382-1496`), and `hessian_jaxpr` (`crates/fj-ad/src/lib.rs:13631-13807`) is `1e-5` finite differences in the general case.
     - `fj-ad` completely ignores `eqn.sub_jaxprs` (`crates/fj-ad/src/lib.rs:1877, 7240-7880, 10187`), and `AssociativeScan` (`L7683, 7703`) and `ReduceWindow` `"prod"` JVP (`L11168`) have mathematical bugs.
  4. **Disconnected Subsystem Islands (`rc26.11`, `rc26.12`):**
     - `partial_eval_jaxpr` (`crates/fj-interpreters/src/partial_eval.rs:41`) and `stage_jaxpr` (`staging.rs:33`) have **0 call sites** in `fj-dispatch`, `fj-api`, or `fj-ad`.
     - `fj-cache` storage managers (`CacheManager`, `InMemoryCache`, `FileCache`, `LruCache`, `TtlLruCache` in `crates/fj-cache/src/lib.rs:295-600`) have **0 call sites** in `fj-dispatch` or `fj-api`.
     - `fj-backend-cpu` (`CpuBackend::execute_parallel` in `crates/fj-backend-cpu/src/executor.rs:347-460`) has **0 call sites** in `fj-dispatch` or `fj-api`.
     - Default `fj_api::jit(jaxpr).call()` (`crates/fj-api/src/transforms.rs:888-891`) bypasses `fj-dispatch`, `fj-backend-cpu`, `fj-cache`, `TraceTransformLedger`, `fj-ledger`, and `fj-egraph`.
  5. **Theatrical Crown Jewels & Isolated Bindings (`rc26.13`, `rc26.14`, `rc26.15`):**
     - `TraceTransformLedger` (`crates/fj-core/src/lib.rs:5794-5999`) holds no output IR and only checks that `transform_evidence: Vec<String>` contains the token `"jit"`, `"grad"`, `"vmap"`, or `"pmap"`.
     - `fj-ledger` runs *after* execution (`crates/fj-dispatch/src/lib.rs:659-710`) with a hardcoded formula `(eqn_count + 2*depth)/(eqn_count + 2*depth + 20)` whose decision is ignored; `LogDomainPosterior`, `ConformalPredictor`, `CalibrationReport`, and `EProcess` have 0 production callers.
     - `fj-runtime` has 0 production callers and `frankentui_bridge` (`crates/fj-runtime/src/lib.rs:67-88`) is a fake stub (`let _type_check = std::any::type_name::<ftui::Style>();`).
     - `fj-ffi` has 0 `extern "C"` exports; `fj-py` fails to build on host Python 3.14, cannot trace Python callables, and runs `1e-6` finite differences on Python functions (`crates/fj-py/src/lib.rs:711-754`).
  6. **Potemkin Conformance Gates, Broken Durability Sidecars, & Blocked `+fma`/FFT Perf (`rc26.16`, `rc26.17`, `rc26.18`):**
     - `security_adversarial.rs`, `numerical_stability.rs`, `decision_ledger.rs`, `onboarding_command.rs`, and `scripts/check_perf_regression.sh` serialize hardcoded static structs or grep a March 12 JSON file without running the workloads.
     - **61 of 96** sidecars in `artifacts/durability/` have stale `source_sha256` hashes, **0 of 6** conformance fixture bundles have a valid sidecar on disk, `oracle_recapture_matrix.v1.json` has 4/5 mismatched fixture SHA-256s on disk, and `./scripts/run_durability_coverage_gate.sh` fails at **16% coverage** (`92/545`).
- [x] **Would completing the 12 previously open beads achieve all project goals?**
  - **Categorically NO.** All 12 previously open beads were narrow `fj-lax` micro-optimizations or lint notes from the June–July 2026 perf campaign (2 of which — `frankenjax-dedicated-gemv-h36uj` and `frankenjax-simd-reduce-window-maxmin-od11p` — were already resolved/rejected in their notes and have now been closed with cited evidence). Completing the remaining 10 perf beads would have left **100% of the tracing, symbolic-AD, transform-composition, caching, backend, TTL, ledger, FFI/Python, gate-honesty, and durability gaps untouched.**

### RC26-2. Phase 2 Bridge Plan & Phase 3a Bead Creation (`frankenjax-rc26-n1emn`)
- [x] Close resolved/rejected stale perf beads with cited evidence from `docs/NEGATIVE_EVIDENCE.md`:
  - [x] Close `frankenjax-dedicated-gemv-h36uj` (ProudSalmon 2026-06-29 bench-backed 0.83x regression; existing row-block threaded `matmul_2d` `N==1` already at 1.08x near-parity with JAX).
  - [x] Close `frankenjax-simd-reduce-window-maxmin-od11p` (SlateHarrier 2026-06-25 outer-dim threaded monotonic deque 4.24x speedup flipping `reduce_window` max/min to a win vs JAX).
  - [x] Link `frankenjax-special-fn-rational-3gsc5` to its documented blocker `frankenjax-cntiy`.
- [x] Create parent epic `frankenjax-rc26-n1emn` (`in_progress`) and 18 self-contained child beads across 6 execution tracks:
  - **Track 1 — Workspace Hygiene, Compiler Lint Gate, Zero-Panic Library Safety, & Doc Truth:**
    - [ ] `frankenjax-rc26-n1emn.1` (`P0`): Fix all 125 `cargo clippy` errors in `fj-lax` and `#[cfg(test)]`-gate embedded benchmark evidence modules (`cz0g0_*`).
    - [ ] `frankenjax-rc26-n1emn.2` (`P1`): Eliminate all 122 non-test library `.unwrap()`, `.expect()`, `panic!()`, `unreachable!()`, and `assert!()` calls across all 14 core crates and enforce `deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)`.
    - [ ] `frankenjax-rc26-n1emn.3` (`P1`): Reconcile all documentation contradictions across `README.md`, `AGENTS.md`, `FEATURE_PARITY.md`, `PARITY-COVERAGE.md`, `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`, and `CHANGELOG.md`, fix `validate_feature_parity.sh` / `bootstrap_vision_evidence_map.sh`, and add `doc_metric_truth_gate.rs`.
  - **Track 2 — Tracing (`fj-trace`) & Public API (`fj-api`) Ergonomic and Semantic Completeness:**
    - [ ] `frankenjax-rc26-n1emn.4` (`P0`): Wire `fj-api::eval_shape` directly to `fj_lax::eval_abstract_primitive` and remove the broken 37-op match + 125-op `inputs[0].clone()` fallback.
    - [ ] `frankenjax-rc26-n1emn.5` (`P0`): Support `Atom::Lit` constants, scalar-tracer operator mixing, and zero-input primitives (`Iota`, `BroadcastedIota`) in `fj-trace::make_jaxpr`.
    - [ ] `frankenjax-rc26-n1emn.6` (`P1`): Complete `TracerRef` primitive method surface (all 157 V1 local ops + `Div`/`Rem`) and eliminate the 22-op wildcard aval fallback in `fj-trace`.
    - [ ] `frankenjax-rc26-n1emn.7` (`P1`): Trace higher-order control flow (`Cond`, `Scan`, `While`, `Switch` with `sub_jaxprs`) in `make_jaxpr` and expose traceable `fj_api::{nn, random, tree_util}` modules.
  - **Track 3 — True Symbolic Jaxpr-to-Jaxpr Autodiff & Arbitrary-Order Transform Composition:**
    - [ ] `frankenjax-rc26-n1emn.8` (`P0`): Implement symbolic Jaxpr-to-Jaxpr differentiation (`build_jvp_jaxpr` & `build_grad_jaxpr`) with activity analysis and dead-cotangent pruning in `fj-ad`.
    - [ ] `frankenjax-rc26-n1emn.9` (`P0`): Replace `1e-6`/`1e-5` finite-difference fallbacks in `fj-dispatch::execute_grad` and `fj-ad::hessian_jaxpr` with symbolic AD composition (`grad(grad(f))`, `grad(vmap(f))`, `vmap(grad(f))` via `BatchTrace`, multi-arg `argnums`, symbolic `hessian`).
    - [ ] `frankenjax-rc26-n1emn.10` (`P0`): Implement `sub_jaxprs`-aware JVP/VJP for `Cond`, `Scan`, `While`, `Switch` and fix `AssociativeScan` (`"max"`/`"min"` VJP, `"prod"` JVP) and `ReduceWindow` (`"prod"` JVP) mathematical bugs in `fj-ad`.
  - **Track 4 — Connected Execution Spine: Partial Eval, Persistent Cache Store, Wave-Parallel CPU Backend, & E-Graph Fusion:**
    - [ ] `frankenjax-rc26-n1emn.11` (`P0`): Wire `partial_eval_jaxpr`/`stage_jaxpr` (`static_argnums`) and `fj-cache` (`LruCache`/`FileCache`) into `fj-dispatch` and `fj-api::jit`.
    - [ ] `frankenjax-rc26-n1emn.12` (`P1`): Integrate `fj-backend-cpu` wave-parallel DAG execution, `fj-egraph` cost-guided loop fusion, and `frankenjax-jjb1h` liveness-driven buffer donation into `dispatch` and `CompiledJaxpr`.
  - **Track 5 — Crown-Jewel Proof-Carrying TTL, Live Decision/Runtime Governance, & FFI/Python Surfaces:**
    - [ ] `frankenjax-rc26-n1emn.13` (`P0`): Upgrade `TraceTransformLedger` from string-token substring checks to a cryptographic Proof-Carrying IR Transform Ledger (TTL v2) binding per-step input/output `Jaxpr` SHA-256 hashes, `AbstractValue` signatures, and `TransformRewriteWitness` records.
    - [ ] `frankenjax-rc26-n1emn.14` (`P1`): Wire `fj-ledger` (`LogDomainPosterior`, `ConformalPredictor`, `EProcess`, `CalibrationReport`) and `fj-runtime` (`RuntimeAdmissionModel`, `asupersync`, `frankentui`) into pre-execution dispatch routing and `fj-api` telemetry.
    - [ ] `frankenjax-rc26-n1emn.15` (`P1`): Expose real `extern "C"` ABI exports in `fj-ffi` + `Primitive::CustomCall` dispatch, and upgrade `fj-py` for Python 3.14 compatibility and real Python callable tracing via `PyTracer`.
  - **Track 6 — Gate Honesty, Artifact Freshness, RaptorQ Durability Repair, & vs-JAX Performance Architecture:**
    - [ ] `frankenjax-rc26-n1emn.16` (`P0`): Replace static/hardcoded gate generators (`security_adversarial.rs`, `numerical_stability.rs`, `decision_ledger.rs`, `onboarding_command.rs`) with live workload execution and assert committed JSON artifact freshness in `oracle_recapture_gate` and `architecture_boundary_gate`.
    - [ ] `frankenjax-rc26-n1emn.17` (`P0`): Unify RaptorQ durability policy, regenerate 61 stale sidecars + 6 missing fixture/oracle sidecar triplets, and enforce on-disk `source_sha256` verification in `tests/durability_coverage.rs` and `scripts/run_durability_coverage_gate.sh`.
    - [ ] `frankenjax-rc26-n1emn.18` (`P1`): Implement `PrecisionPolicy::{BitExactPortable, HardwareFmaAccelerated}` runtime SIMD/FMA dispatch (unblocking `frankenjax-cntiy`), Stockham mixed-radix (`radix-2/3/4/5/8`) plan-hoisted batched FFT (unblocking `frankenjax-murmw`), and live vs-JAX 0.10.2 performance gate reporting.

### RC26-3. Phase 4 Ambition Rounds (1–3) Applied In-Place
- [x] **Ambition Round 1 — Architectural Integrity over Workarounds:**
  - Rejected patching `execute_grad_finite_diff` to perturb tensors; escalated `rc26.8` + `rc26.9` to true symbolic Jaxpr-to-Jaxpr differentiation (`build_jvp_jaxpr` & `build_grad_jaxpr`) so `grad(grad(f))`, `grad(vmap(f))`, `vmap(grad(f))`, and `hessian` are 100% symbolic IR transformations with zero `1e-5`/`1e-6` truncation error.
  - Escalated `fj-trace` (`rc26.5`–`rc26.7`) from adding a few helper functions to complete 157-primitive `TracerRef` coverage, `Atom::Lit` constant binding, zero-input generator tracing, nested `sub_jaxprs` closure tracing, and generic `PyTree<T>` support across `Value` and `TracerRef`.
- [x] **Ambition Round 2 — Genuine Crown-Jewel Proofs & Zero-Potemkin Verification:**
  - Escalated `TraceTransformLedger` (`rc26.13`) from string token matching (`evidence.contains("jit")`) to a Merkle-committed Proof-Carrying IR Transform Ledger verifying `output_jaxpr_sha256 == next.input_jaxpr_sha256` chain continuity and abstract shape/dtype preservation across `Jit`, `Grad`, and `Vmap`.
  - Escalated `fj-ledger` and `fj-runtime` (`rc26.14`) from post-execution dead telemetry to pre-execution decision-theoretic governance using `LogDomainPosterior`, `ConformalPredictor` quantile bounds, and sequential `EProcess` drift guards.
  - Escalated `fj-conformance` gates (`rc26.16`, `rc26.17`) to ban static struct serialization (`actual_error_class: spec.expected_error_class.to_owned()`) and enforce live `source_sha256` verification on all committed RaptorQ sidecar triplets.
- [x] **Ambition Round 3 — Algorithmic & Performance Leapfrogging:**
  - Added Activity Analysis & Dead-Cotangent Pruning to `build_grad_jaxpr` (`rc26.8`) so symbolic gradient `Jaxpr`s eliminate inactive primal/adjoint subgraphs before `CompiledJaxpr` lowering.
  - Combined `fj-egraph` multi-op elementwise DAG fusion, `fj-backend-cpu` topological wave-parallel execution, and `frankenjax-jjb1h` `Arc::get_mut` liveness-driven buffer donation in `rc26.12`.
  - Resolved the `frankenjax-cntiy` (`+fma` vs golden SHA-256) architectural deadlock in `rc26.18` via explicit `PrecisionPolicy::{BitExactPortable, HardwareFmaAccelerated}` runtime CPUID dispatch alongside Stockham mixed-radix (`radix-2/3/4/5/8`) plan-hoisted batched FFT (`frankenjax-murmw`).

### RC26-4. Phase 5 Plan-Space Refinement Rounds (1–5)
- [x] **Refinement Round 1 (Coverage & Stale-Bead Hygiene):** Verified 24/24 in-scope Vision Checklist rows have explicit bead owners (`0 NO_BEAD` rows); closed the 2 already-resolved/rejected perf beads (`frankenjax-dedicated-gemv-h36uj` and `frankenjax-simd-reduce-window-maxmin-od11p`) with cited `docs/NEGATIVE_EVIDENCE.md` records; linked `frankenjax-special-fn-rational-3gsc5` to `frankenjax-cntiy`.
- [x] **Refinement Round 2 (Self-Containment & File/Line Grounding):** Verified every `rc26.1`–`rc26.18` bead description specifies exact file paths, line numbers, root-cause diagnosis, concrete implementation steps, and positive + negative test assertions.
- [x] **Refinement Round 3 (Dependency Graph Unification):** Linked orphan open perf beads (`frankenjax-jjb1h`, `frankenjax-fgk6m`, `frankenjax-jyaz6`, `frankenjax-small-batched-gemm-q032w`, `frankenjax-cntiy`, `frankenjax-murmw`, `frankenjax-mcqr.111`) into the unified `rc26` + `mcqr` dependency DAG so `rc26.1` (fixing `fj-lax` clippy) unblocks all `fj-lax` work and `rc26.18` unblocks `cntiy` and `murmw`.
- [x] **Refinement Round 4 (Parent-Child `bv` Mutual-Wait Elimination):** `bv --robot-triage` revealed that adding explicit `blocks` edges from parent `frankenjax-rc26-n1emn` to its children while `--parent` creates `child -> parent` edges caused a mutual wait in `bv` (`actionable_count: 0`). Removed the redundant `parent -> child` `blocks` edges and marked `frankenjax-rc26-n1emn` as `in_progress`, restoring `actionable_count: 4` and clean top-pick ranking (`rc26.1`, `rc26.4`, `rc26.5`).
- [x] **Refinement Round 5 (Final Convergence Verification):**
  - `br dep cycles --json`: `0` cycles.
  - `br doctor`: DB and `.beads/issues.jsonl` in sync (`2,066` total records: `2,037` closed, `28` open, `1` in_progress).
  - `bv --robot-triage`: Top 3 actionable picks are `frankenjax-rc26-n1emn.1` (score `0.5566`, unblocks 7 downstream items), `frankenjax-rc26-n1emn.4` (score `0.1937`, unblocks `rc26.6` & `rc26.8`), and `frankenjax-rc26-n1emn.5` (score `0.1543`, unblocks `rc26.6` & `rc26.7`).

### RC26-5. Converged Dependency Graph & Execution Order
- **Immediate Parallel Wave 1 (`br ready` P0 roots):**
  - `frankenjax-rc26-n1emn.1`: Fix 125 `fj-lax` clippy errors + gate embedded bench modules (unblocks `.2`, `.3`, `.10`, `.11`, `.18`, `mcqr.111`, `jjb1h`, `fgk6m`, `jyaz6`).
  - `frankenjax-rc26-n1emn.4`: Wire `fj-api::eval_shape` to `fj_lax::eval_abstract_primitive` (unblocks `.6`, `.8`).
  - `frankenjax-rc26-n1emn.5`: Support `Atom::Lit` constants, scalar-tracer mixing, and 0-input primitives in `fj-trace` (unblocks `.6`, `.7`).
- **Wave 2 (Symbolic AD Core, Full `TracerRef`, Staging/Cache Spine, & Precision/FFT Architecture):**
  - `frankenjax-rc26-n1emn.2` (depends on `.1`): Zero-panic library safety (`deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)`).
  - `frankenjax-rc26-n1emn.6` (depends on `.4`, `.5`): Complete 157-op `TracerRef` method surface & remove 22-op aval wildcard.
  - `frankenjax-rc26-n1emn.8` (depends on `.4`): Symbolic `build_jvp_jaxpr` & `build_grad_jaxpr` with activity pruning.
  - `frankenjax-rc26-n1emn.10` (depends on `.1`): `sub_jaxprs`-aware JVP/VJP for `Cond`/`Scan`/`While`/`Switch` + `AssociativeScan`/`ReduceWindow` AD fixes.
  - `frankenjax-rc26-n1emn.11` (depends on `.1`): Wire `partial_eval_jaxpr`/`stage_jaxpr` (`static_argnums`) + `fj-cache` (`LruCache`/`FileCache`) into `fj-dispatch` & `fj-api::jit`.
  - `frankenjax-rc26-n1emn.18` (depends on `.1`): `PrecisionPolicy` SIMD/FMA dispatch + Stockham mixed-radix batched FFT + live vs-JAX perf gate (unblocks `cntiy`, `murmw`, `murmw.1`, `q032w`, `3gsc5`).
  - `frankenjax-jjb1h` (depends on `.1`): Liveness-driven output buffer donation (`Arc::get_mut`).
- **Wave 3 (Higher-Order Symbolic AD Composition, Control-Flow Tracing, & Wave-Parallel/Fused Execution):**
  - `frankenjax-rc26-n1emn.7` (depends on `.5`, `.6`): Higher-order `sub_jaxprs` tracing in `make_jaxpr` + `fj_api::{nn, random, tree_util}`.
  - `frankenjax-rc26-n1emn.9` (depends on `.8`, `.10`): Eliminate `1e-6`/`1e-5` finite-difference fallbacks in `execute_grad` and `hessian_jaxpr`.
  - `frankenjax-rc26-n1emn.12` (depends on `.11`, `jjb1h`): Integrate `fj-backend-cpu` wave-parallel DAG execution + `fj-egraph` fusion + buffer donation.
- **Wave 4 (Proof-Carrying TTL v2, Pre-Execution Ledger Governance, & FFI/Python Bindings):**
  - `frankenjax-rc26-n1emn.13` (depends on `.8`, `.9`, `.11`): Cryptographic Proof-Carrying IR Transform Ledger (TTL v2).
  - `frankenjax-rc26-n1emn.14` (depends on `.11`, `.13`): Pre-execution `fj-ledger` + `fj-runtime` admission/routing + real `frankentui` rendering.
  - `frankenjax-rc26-n1emn.15` (depends on `.6`, `.9`): `fj-ffi` `extern "C"` ABI + `CustomCall` dispatch & `fj-py` Python 3.14 + `PyTracer` callable tracing.
- **Wave 5 (Live Conformance Gates, 100% RaptorQ Durability Triplets, & Final Doc-Truth Gate):**
  - `frankenjax-rc26-n1emn.16` (depends on `.13`, `.14`): Replace static gate struct generators with live workload execution & refresh stale conformance JSON artifacts.
  - `frankenjax-rc26-n1emn.17` (depends on `.16`): Regenerate all 61 stale + 6 missing fixture RaptorQ sidecar triplets & enforce `source_sha256` verification in `durability_coverage`.
  - `frankenjax-rc26-n1emn.3` (depends on `.1`, `.16`, `.17`): Reconcile all docs, fix `validate_feature_parity.sh` & `bootstrap_vision_evidence_map.sh`, and enforce `doc_metric_truth_gate`.

---

## 2026-05-01 Full Reality Check, Ambition Rounds, and Bead Plan (Historical Archive)

Tracker parent: `frankenjax-cstq`

### RCB0. Skill protocol and source ingestion
- [x] Apply `reality-check-for-project` instead of jumping directly to implementation.
- [x] Read global `/dp/AGENTS.md`.
- [x] Read repo-local `AGENTS.md`.
- [x] Read `README.md` fully, including current limitations and verification sections.
- [x] Read `docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md`.
- [x] Read `docs/planning/PLAN_TO_PORT_JAX_TO_RUST.md`.
- [x] Read `docs/planning/PROPOSED_ARCHITECTURE.md`.
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
- [x] Update `docs/planning/PROPOSED_ARCHITECTURE.md` with durability implementation details.
- [x] Update `docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md` implementation progress section.

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
