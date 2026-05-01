# EXISTING_JAX_STRUCTURE

DOC-PASS-00 baseline matrix: `artifacts/docs/bd-3dl.23.1_gap_matrix.v1.md`
DOC-PASS-01 module/package cartography snapshot: `bd-3dl.23.2` (2026-02-14)

## 1. Legacy Oracle

- Root: `/data/projects/frankenjax/legacy_jax_code/jax`
- Legacy alias path used in older docs/scripts: `/dp/frankenjax/legacy_jax_code/jax`
- Upstream: `jax-ml/jax`

## 2. High-Value File/Function Anchors

### Transform API entry points

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/api.py` | `jit`, `vmap`, `grad`, `value_and_grad`, `_vjp` | User-visible transform contract boundary | `pjit.py`, `stages.py`, interpreter transforms |

### JIT and staging/lowering path

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/pjit.py` | `JitWrapped`, `_parse_jit_arguments`, `_run_python_pjit` | JIT argument normalization and staged execution boundary | `dispatch.py`, compilation/cache modules |
| `jax/_src/stages.py` | `Traced` | Trace materialization and lowering metadata boundary | `partial_eval.py`, interpreter/lowering pipeline |

### Trace/Jaxpr construction and partial-eval boundary

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/core.py` | `Trace`, `Tracer`, `Jaxpr` | Canonical IR typing and tracer lifecycle boundary | AD, batching, dispatch interpreters |
| `jax/_src/interpreters/partial_eval.py` | `trace_to_jaxpr` | Trace-to-IR lowering + residual handling boundary | `dispatch.py`, `ad.py`, `batching.py` |

### AD (reverse/forward transform semantics)

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/interpreters/ad.py` | `linearize_subtrace`, `linearize_jaxpr`, `backward_pass3` | Reverse/forward transform semantics boundary | API transform composition (`grad`, `value_and_grad`) |

### Batching / vmap semantics

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/interpreters/batching.py` | `BatchTrace`, `BatchTracer`, `batch_subtrace`, `batch_jaxpr` | Batch-axis semantics boundary | API transform composition (`vmap`) and dispatch path |

### Cache-key and compilation cache semantics

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/cache_key.py` | `get`, `_hash_serialized_compile_options`, `_hash_xla_flags`, `_hash_accelerator_config` | Cache identity construction boundary | `compiler.py`, `compilation_cache.py`, backend/device metadata |
| `jax/_src/compiler.py` | `compile_or_get_cached`, `_resolve_compilation_strategy`, `_cache_read`, `_cache_write` | Compile/cache lifecycle boundary | `dispatch.py`, backend bridge, persistent cache storage |
| `jax/_src/compilation_cache.py` | `get_cache_key`, `get_executable_and_time`, `put_executable_and_time` | Persistent cache storage boundary | compiler strategy + backend-specific executable handling |

### Dispatch-level memoization and trace caching

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/dispatch.py` | `xla_primitive_callable`, `_is_supported_cross_host_transfer` | Runtime dispatch + memoization boundary | compiler/cache modules + backend bridge |
| `jax/_src/interpreters/partial_eval.py` | `trace_to_jaxpr` | Trace caching boundary (pre-dispatch IR reuse) | staging + dispatch + transform interpreters |

### Backend bridge and FFI boundary

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/xla_bridge.py` | backend client resolution | backend selection + runtime protocol boundary | `dispatch.py`, `compiler.py`, device runtime |
| `jax/_src/ffi.py`, `jaxlib/*.cc` | FFI registration/callback surfaces | native bridge lifetime and callback safety boundary | runtime client/device/program ownership |

## 3. Semantic Hotspots (Non-Negotiable)

1. Transform composition semantics for `jit`, `grad`, `vmap`.
2. Deterministic trace/Jaxpr construction and lowering metadata.
3. Cache-key soundness and compatibility-sensitive inputs.
4. Dispatch cache behavior and memoization invariants.
5. Error-path behavior when metadata or shape contracts are incompatible.

## 4. Conformance Fixture Family Anchors

- `jit`: `tests/jax_jit_test.py`
- `grad`: `tests/lax_autodiff_test.py` and `tests/lax_test.py`
- `vmap`: `tests/lax_vmap_test.py` and `tests/lax_vmap_op_test.py`
- `lax primitives`: `tests/lax_test.py`
- `random`: `tests/random_test.py` and `tests/random_lax_test.py`

## 5. Compatibility-Critical Inputs for Cache Keying

The legacy cache key uses deterministic hashing over a canonicalized input bundle that includes:

- canonicalized module bytecode
- backend/version fields
- normalized compile options
- normalized XLA flags
- accelerator topology/device descriptors
- optional runtime custom hook bits

FrankenJAX MUST preserve input-surface semantics for scoped compatibility.

## 6. Security and Reliability Risk Areas

- cache confusion via incompatible/unknown metadata
- transform-order drift leading to semantic mismatch
- malformed shape/graph signatures
- stale cache artifacts or corruption in persistent storage

## 7. Extraction Boundary (Current)

Included now:

- transform API semantics and composition invariants
- deterministic key derivation contract
- core conformance fixture families

Deferred but tracked:

- broad plugin/distributed backend breadth
- long-tail API surfaces outside scoped parity matrix

## 8. FrankenJAX Workspace Topology (Current Rust Slice)

### 8.1 Crate Ownership and Boundary Matrix

| Crate | Owns | Depends on (normal) | Boundary contract |
|---|---|---|---|
| `fj-core` | canonical IR (`Jaxpr`), TTL (`TraceTransformLedger`), transforms/types | none | root semantic model; no runtime/backend side effects |
| `fj-trace` | trace context, shape inference, Jaxpr materialization from traces | `fj-core` | trace-to-IR boundary; tracer lifecycle and abstract value management |
| `fj-lax` | primitive eval semantics (`eval_primitive`) | `fj-core` | pure primitive semantics over core values |
| `fj-interpreters` | IR interpreter (`eval_jaxpr`) | `fj-core`, `fj-lax` | executes canonical IR without transform orchestration |
| `fj-ad` | autodiff path (`grad_first`, tape/backward logic) | `fj-core`, `fj-lax` | gradient semantics; consumes IR/value model |
| `fj-cache` | deterministic cache key derivation | `fj-core` | strict/hardened key gate for unknown metadata |
| `fj-ledger` | evidence/decision ledger + calibration logic | `fj-core` | policy/evidence state, no dispatch execution |
| `fj-dispatch` | orchestration (`dispatch`) and transform wrapper execution | `fj-ad`, `fj-backend-cpu`, `fj-cache`, `fj-core`, `fj-egraph`, `fj-interpreters`, `fj-lax`, `fj-ledger`, `fj-runtime`, `fj-trace` | integration choke point for transform order, cache behavior, backend routing, and optimization toggles |
| `fj-runtime` | admission policy model + optional bridges | `fj-core`, `fj-ledger` | operational runtime decision boundary |
| `fj-egraph` | e-graph optimizer + IR round trip | `fj-core` | optimization sandbox integrated through explicit dispatch compile options |
| `fj-backend-cpu` | dependency-wave CPU backend | `fj-core`, `fj-interpreters`, `fj-lax`, `fj-runtime` | always-available CPU execution backend, separate from facade/API surface |
| `fj-api` | user-facing transform facade | `fj-ad`, `fj-core`, `fj-dispatch`, `fj-trace` | V1 public Rust API for `jit`/`grad`/`vmap` and derivative helpers |
| `fj-ffi` | C FFI call interface | `fj-core` | isolated native interop boundary |
| `fj-conformance` | fixture/parity harness, durability pipeline, E2E logs, boundary gates | `fj-cache`, `fj-core`, `fj-dispatch`, `fj-lax` | oracle comparison, durability, and replayable evidence generation |
| `fj-test-utils` | common test log schema + fixture IDs | none | cross-crate test contract only (dev dependency) |

### 8.2 Cross-Module Dependency Direction Map (Cycle-Checked)

Normal dependency edges (from `cargo metadata --no-deps --format-version 1`):

- `fj-core -> (none)`
- `fj-test-utils -> (none)`
- `fj-trace -> fj-core`
- `fj-interpreters -> fj-core`, `fj-interpreters -> fj-lax`
- `fj-lax -> fj-core`
- `fj-dispatch -> fj-ad`, `fj-dispatch -> fj-backend-cpu`, `fj-dispatch -> fj-cache`, `fj-dispatch -> fj-core`, `fj-dispatch -> fj-egraph`, `fj-dispatch -> fj-interpreters`, `fj-dispatch -> fj-lax`, `fj-dispatch -> fj-ledger`, `fj-dispatch -> fj-runtime`, `fj-dispatch -> fj-trace`
- `fj-ad -> fj-core`, `fj-ad -> fj-lax`
- `fj-backend-cpu -> fj-core`, `fj-backend-cpu -> fj-interpreters`, `fj-backend-cpu -> fj-lax`, `fj-backend-cpu -> fj-runtime`
- `fj-runtime -> fj-core`, `fj-runtime -> fj-ledger`
- `fj-cache -> fj-core`
- `fj-ledger -> fj-core`
- `fj-egraph -> fj-core`
- `fj-conformance -> fj-cache`, `fj-conformance -> fj-core`, `fj-conformance -> fj-dispatch`, `fj-conformance -> fj-lax`
- `fj-api -> fj-ad`, `fj-api -> fj-core`, `fj-api -> fj-dispatch`, `fj-api -> fj-trace`
- `fj-ffi -> fj-core`

Cycle check:
- `./scripts/run_architecture_boundary_gate.sh --enforce` reports `15` crates, `34` normal edges, `6` accepted boundary decisions, and `0` issues.
- Result confirms acyclic layering for the current workspace crate graph and rejects production dependencies on `fj-conformance`.

### 8.3 Layering Constraints (Current)

| Layer | Crates | Allowed outbound edges | Forbidden patterns |
|---|---|---|---|
| `L0 model` | `fj-core`, `fj-test-utils` | none for production model code; test-utils is dev-only scaffolding | any dependency on higher execution/runtime/harness layers |
| `L1 semantics` | `fj-trace`, `fj-lax`, `fj-cache`, `fj-ledger` | `-> fj-core` | cross-calls into dispatch/harness |
| `L2 execution` | `fj-interpreters`, `fj-ad`, `fj-runtime`, `fj-egraph` | `-> L0/L1` | reverse dependency from `fj-core` into execution |
| `L3 facade/backend` | `fj-dispatch`, `fj-api`, `fj-backend-cpu`, `fj-ffi` | `-> L0/L1/L2` as needed | semantic crates depending on facade/backend crates |
| `L4 ops/harness` | `fj-conformance` | `-> L0/L1/L2/L3` for evidence generation | production execution paths depending on conformance harness |

## 9. Hidden/Implicit Coupling Register

| Coupling | Why implicit | Risk | Mitigation target |
|---|---|---|---|
| Transform-order law split between `fj-core::verify_transform_composition` and `fj-dispatch` transform execution | validation and execution are in separate crates with shared assumptions | semantic drift between proof and execution | add shared transform contract tests in `fj-conformance` |
| Strict/hardened mode semantics span `fj-core`, `fj-cache`, `fj-dispatch`, `fj-runtime` | mode flag is propagated, not centrally enforced | inconsistent fail-closed behavior | centralize mode contract matrix and cross-crate invariant tests |
| `fj-test-utils` schema contract is global across crates | test log shape is shared via dev dependency only | silent logging drift across crate tests | keep schema tests mandatory in every crate |
| `fj-conformance` directly depends on implementation crates including `fj-dispatch`, `fj-cache`, `fj-core`, and `fj-lax` | harness executes current integration and evidence choke points | harness and execution semantics can co-evolve accidentally | add explicit fixture-versioning and drift-class gates |
| `fj-egraph` optimization path is wired through an explicit dispatch compile option | optimization correctness depends on the option gate staying explicit | future broader integration may bypass parity gates | require conformance parity run before promoting optimization paths |

## 10. Conflict Check: Current vs Target Architecture

| Target architecture component | Current owner/location | Status | Conflict impact |
|---|---|---|---|
| `trace` crate (`fj-trace`) | dedicated crate (`crates/fj-trace`) owns tracer IDs, abstract values, and trace context materialization | satisfied (crate present) | remaining conflict is explicit lowering/backend split, not trace crate existence |
| `canonical IR` crate (`fj-ir`) | represented in `fj-core` | partially satisfied | IR ownership exists but not isolated as planned crate |
| `transform stack` crate (`fj-transforms`) | folded into `fj-core`, `fj-trace`, `fj-ad`, `fj-dispatch`, and `fj-api` | explicitly deferred | extraction waits for advanced transform/control-flow parity and facade replay evidence |
| `lowering` crate (`fj-lowering`) | folded into interpreter/dispatch/backend path | explicitly deferred | V1 has no XLA/compiler lowering contract; avoid exposing an unused surface |
| `backend` crate (`fj-backend-cpu`) | dedicated CPU backend crate | satisfied for V1 CPU scope | future backend plurality may justify more routing/facade work |
| top-level user API facade | `fj-api` | satisfied for V1 Rust facade | release/package naming can be revisited after public examples are replay-proven |
| FFI boundary | `fj-ffi` | satisfied | native interop remains isolated from pure Rust transform crates |

Conflict-check verdict:
- Current crate dependency graph is acyclic and layered.
- `fj-api`, `fj-backend-cpu`, `fj-ffi`, and `fj-conformance` are explicit accepted boundaries for V1.
- Dedicated `fj-transforms` and `fj-lowering` crates are not optimal yet because they would mix parity work with crate churn before user-facing examples and advanced transform/control-flow evidence are green.
- The executable decision source is `artifacts/conformance/architecture_boundary_decision.v1.json`, generated by `./scripts/run_architecture_boundary_gate.sh --enforce`.

## 11. Symbol/API Census and Surface Classification

Census scope for V1 parity slice:
- legacy anchors in section `2` (transform, trace, cache, dispatch, backend/ffi bands),
- Rust workspace exported symbols in `fj-core`, `fj-trace`, `fj-lax`, `fj-interpreters`, `fj-ad`, `fj-cache`, `fj-dispatch`, `fj-ledger`, `fj-runtime`, `fj-conformance`, `fj-egraph`, `fj-api`, `fj-backend-cpu`, and `fj-ffi`.

### 11.1 Surface Classes

| Class | Meaning | Stability expectation | Break policy |
|---|---|---|---|
| `S0 user-contract` | user-observable transform behavior/API contract | strict compatibility target | break only with explicit parity exception |
| `S1 compatibility-critical internal` | internal symbol that changes user-visible outcomes | high stability target | guarded by differential/parity gates |
| `S2 workspace-internal` | implementation helper surface | medium stability target | may change if invariants + tests hold |
| `S3 experimental/internal` | optional or not yet integrated surface | low stability target | must not affect strict-mode user contract |

### 11.2 Legacy Symbol Census (Scoped Families)

| Family | Representative symbols | Class | User visibility | Regression tag |
|---|---|---|---|---|
| Transform API front door | `jit`, `vmap`, `grad`, `value_and_grad`, `_vjp` (`jax/_src/api.py`) | `S0` (`jit`/`vmap`/`grad`), `S1` (`_vjp`) | direct | `P0-transform-order` |
| Trace/Jaxpr construction | `Trace`, `Tracer`, `Jaxpr` (`jax/_src/core.py`) | `S1` | indirect but semantics-critical | `P0-trace-shape` |
| Partial-eval boundary | `trace_to_jaxpr` (`jax/_src/interpreters/partial_eval.py`) | `S1` | indirect | `P0-lowering-determinism` |
| AD transform internals | `linearize_subtrace`, `linearize_jaxpr`, `backward_pass3` (`jax/_src/interpreters/ad.py`) | `S1` | indirect (`grad`) | `P1-ad-correctness` |
| Batching transform internals | `BatchTrace`, `BatchTracer`, `batch_subtrace`, `batch_jaxpr` (`jax/_src/interpreters/batching.py`) | `S1` | indirect (`vmap`) | `P1-batch-axis` |
| Cache key construction | `get`, `_hash_serialized_compile_options`, `_hash_xla_flags`, `_hash_accelerator_config` (`jax/_src/cache_key.py`) | `S1` | indirect | `P0-cache-key` |
| Compile/cache lifecycle | `compile_or_get_cached`, `_resolve_compilation_strategy`, `_cache_read`, `_cache_write` (`jax/_src/compiler.py`) | `S1` | indirect | `P0-cache-lifecycle` |
| Persistent compilation cache | `get_cache_key`, `get_executable_and_time`, `put_executable_and_time` (`jax/_src/compilation_cache.py`) | `S1` | indirect | `P0-cache-persistence` |
| Dispatch callable path | `xla_primitive_callable`, `_is_supported_cross_host_transfer` (`jax/_src/dispatch.py`) | `S1` | indirect | `P0-dispatch-routing` |
| Backend/FFI bridge | backend bridge and FFI surfaces (`jax/_src/xla_bridge.py`, `jax/_src/ffi.py`, `jaxlib/*.cc`) | `S1`/`S3` | indirect | `P0-backend-ffi` |

### 11.3 Rust Workspace Symbol Census (Current Slice)

`rg '^pub (struct|enum|fn|mod|type|trait)'` count snapshot (2026-02-15):
- `fj-core=31`, `fj-trace=12`, `fj-interpreters=15`, `fj-dispatch=6`, `fj-lax=2`, `fj-cache=7`, `fj-runtime=3`, `fj-conformance=43`, `fj-ledger=15`, `fj-ad=3`, `fj-egraph=5`, `fj-test-utils=10`.

| Family | Key symbols | Class | Boundary note | Regression tag |
|---|---|---|---|---|
| IR + transform proof | `Jaxpr`, `TraceTransformLedger`, `TransformCompositionProof`, `verify_transform_composition` (`fj-core`) | `S1` | public in crate, but stability governed by parity semantics | `P0-transform-proof` |
| Trace abstraction and capture | `TracerId`, `AbstractValue`, `JaxprTrace`, `ClosedJaxpr`, `TraceError`, `SimpleTraceContext` (`fj-trace`) | `S1` | dedicated trace boundary now explicit in crate ownership | `P0-trace-shape` |
| Primitive semantics | `eval_primitive`, `EvalError` (`fj-lax`) | `S1` | primitive semantics feed all execution paths | `P0-primitive-semantics` |
| Interpreter execution | `eval_jaxpr`, `InterpreterError` (`fj-interpreters`) | `S1` | execution core for transform wrappers | `P0-ir-exec` |
| AD execution | `grad_jaxpr`, `grad_first`, `AdError` (`fj-ad`) | `S1` | drives `grad` transform behavior | `P1-ad-numerics` |
| Cache keying | `build_cache_key`, `build_cache_key_ref`, `compatibility_matrix_row`, `CacheKeyError` (`fj-cache`) | `S1` | strict/hardened compatibility gate | `P0-cache-key` |
| Dispatch integration | `dispatch`, `DispatchRequest`, `DispatchResponse`, `DispatchError` (`fj-dispatch`) | `S0`/`S1` | integration choke point for user-observable behavior | `P0-dispatch-order` |
| Decision ledger/runtime policy | `EvidenceLedger`, `recommend_action`, `RuntimeAdmissionModel` (`fj-ledger`, `fj-runtime`) | `S1`/`S2` | policy can alter runtime acceptance path | `P1-policy-drift` |
| Conformance and durability | `run_transform_fixture_bundle`, `capture_transform_fixture_bundle_with_oracle`, `encode_artifact_to_sidecar`, `scrub_sidecar`, `generate_decode_proof` (`fj-conformance`) | `S1` | verification plane; guards parity and durability evidence | `P0-parity-gate` |
| E-graph optimization | `optimize_jaxpr`, `algebraic_rules` (`fj-egraph`) | `S3` | currently isolated from dispatch path | `P2-future-optimizer` |

### 11.4 High-Regression Watchlist (Tagged)

| Symbol/band | Why high regression risk | Required guardrail |
|---|---|---|
| `jit`/`grad`/`vmap` + `verify_transform_composition` + `dispatch` | transform-order drift changes user semantics | transform composition differential fixtures + strict drift gate |
| `cache_key.py:get` + `fj-cache::build_cache_key` | cache confusion/collision risk and compatibility drift | component-level cache-key parity ledger |
| `trace_to_jaxpr` + `fj-interpreters::eval_jaxpr` | non-deterministic lowering/execution drift | deterministic trace fixtures and replay corpus |
| `fj-dispatch::dispatch` error path | fail-open vs fail-closed ambiguity | strict/hardened divergence report with allowlist |
| durability pipeline (`encode`/`scrub`/`proof`) | evidence corruption can mask regressions | decode-proof gate on all long-lived artifacts |

Boundary clarification:
- There is not yet a top-level `frankenjax` facade crate; current `pub` Rust symbols are workspace-public, not final end-user API guarantees.
- For V1 compatibility claims, `S0` and `S1` surfaces are normative; `S2`/`S3` remain implementation-detail unless explicitly promoted.

## 12. Data Model, State, and Invariant Mapping (DOC-PASS-03)

### 12.1 Legacy-to-Rust Data Model Mapping

| Legacy model band | Rust owner | Core fields/state | Invariant obligations |
|---|---|---|---|
| Tracer/Jaxpr graph (`core.py`) | `fj-core` (`Jaxpr`, `Equation`, `VarId`, `TraceTransformLedger`) | invars/outvars, equation edges, transform list | var def-use integrity, deterministic fingerprint/signature |
| Partial-eval state (`partial_eval.py`) | `fj-interpreters` + `fj-core` | trace-to-jaxpr boundary, residual equation ordering | stable lowering order and explicit residual handling |
| AD tape/linearization (`ad.py`) | `fj-ad` | primal values, tangent/cotangent propagation | gradient path consistency and scalar-output constraints |
| Batching state (`batching.py`) | `fj-dispatch` (current slice; `execute_vmap`) | mapped axis semantics and batched argument slicing | vmap order semantics and output-shape consistency |
| Cache identity and lifecycle (`cache_key.py`, `compiler.py`) | `fj-cache` + `fj-dispatch` | key input bundle, backend/options metadata, lookup/write path | strict fail-closed on unknown incompatibilities |
| Dispatch/runtime envelope (`dispatch.py`, bridge bands) | `fj-dispatch`, `fj-runtime` | request, transform stack, admission decision state | transform order preservation and policy traceability |

### 12.2 Critical State Machines

| Component | State sequence | Safety check | Failure semantics |
|---|---|---|---|
| Trace to executable IR | `build trace -> append equations -> seal outvars -> evaluate` | equation inputs must reference defined vars | strict: error and stop; hardened: audit + bounded handling |
| Transform stack execution | `verify composition -> apply transform wrappers -> evaluate root jaxpr` | transform/evidence cardinality and order | strict: reject on mismatch; hardened: log bounded fallback only |
| Cache/dispatch path | `canonicalize request -> key hash -> cache lookup -> hit/miss branch` | key determinism + compatibility gate | strict: fail closed on unknown incompatible metadata |
| Admission decision | `collect evidence -> posterior -> recommend action -> execute` | posterior/action consistency with loss matrix | inconsistent action path is denied and logged |
| Durability verification | `generate sidecar -> scrub integrity -> decode proof` | sidecar/source hash parity + decode success | decode/scrub mismatch blocks artifact acceptance |

### 12.3 Invariant Violation and Recovery Map

| Invariant family | Violation signal | Strict behavior | Hardened behavior |
|---|---|---|---|
| Transform order and composition | unsupported sequence, evidence mismatch | reject request | bounded fallback only when explicitly allowed + ledger entry |
| Cache identity soundness | unknown incompatible feature or unstable key input | reject key path | continue only with explicit compatibility event |
| Shape/graph integrity | malformed vars/shapes/graph signatures | fail-closed | fail-closed with diagnostic context |
| Dispatch policy traceability | missing evidence or ambiguous decision path | abort and emit deterministic error | abort and emit deterministic error + additional policy metadata |
| Durability integrity | scrub hash mismatch or decode-proof failure | block release | block release |

## 13. Execution-Path Tracing and Control-Flow Narratives (DOC-PASS-04)

### 13.1 Workflow Trace A: Transform API Request -> Dispatch Result

Canonical control flow (`jit`/`grad`/`vmap` request):

1. User-facing transform entry (`jax/_src/api.py` analog) receives function + args.
2. Trace/IR boundary constructs canonical program representation (`fj-core` TTL/Jaxpr path).
3. `fj-core::verify_transform_composition` validates transform order + evidence cardinality.
4. `fj-cache` derives compatibility-gated cache identity.
5. `fj-dispatch::dispatch` executes transform wrappers in declared stack order.
6. `fj-interpreters::eval_jaxpr` evaluates terminal IR path.
7. `fj-ledger` records evidence/decision metadata and response returns.

Branch points and outcomes:

| Branch point | Condition | Strict branch | Hardened branch |
|---|---|---|---|
| composition verification | unsupported transform sequence or evidence mismatch | reject request immediately | bounded fallback only if explicitly allowlisted; always log divergence |
| cache compatibility gate | unknown incompatible metadata in key input | fail-closed before dispatch | compatibility event emitted; continuation only through bounded audited path |
| `grad` wrapper execution | AD path unsupported for current composition | deterministic error | bounded finite-difference fallback if allowed by packet policy |
| interpreter execution | malformed var/shape graph | deterministic fail-closed error | fail-closed with richer diagnostics |

### 13.2 Workflow Trace B: Cache Hit/Miss Runtime Branching

1. Canonical request fingerprint materialized.
2. Cache key computed from backend/options/flags/shape context.
3. Lookup branch:
   - hit -> reuse cached executable/derived path.
   - miss -> compile/evaluate path, then cache write.
4. Dispatch returns output + cache-key metadata.

Error/fallback branches:

| Branch point | Failure mode | Recovery law |
|---|---|---|
| key derivation | non-canonical or incompatible input bundle | strict fail-closed; hardened only via explicit compatibility event |
| cache read | stale/corrupt cache artifact | bypass cache, re-evaluate, and emit cache-integrity event |
| cache write | write with unstable key material | block write path and preserve run result without poisoning cache |

### 13.3 Workflow Trace C: Conformance and Differential Validation

1. Fixture bundle load (`fj-conformance` fixture family).
2. Case execution through dispatch pipeline.
3. Comparator branch (`exact`, `approx`, `shape`, `type`) selects drift class.
4. Parity report emitted with strict/hardened outcome split.
5. Failure branch emits minimized repro + mismatch taxonomy.

Failure branch obligations:
- no silent downgrade of mismatch severity,
- include replay pointers for scenario-level reproduction,
- preserve strict/hardened divergence explicitly.

### 13.4 Workflow Trace D: Durability Sidecar Pipeline

1. `generate`: encode artifact to sidecar symbols.
2. `scrub`: decode + hash-compare integrity.
3. `proof`: decode under symbol-loss simulation.
4. `gate`: artifact accepted only if scrub and proof pass.

Failure branches:

| Branch point | Failure signal | Required action |
|---|---|---|
| scrub | hash mismatch | mark artifact invalid, block release path |
| proof | decode failure under configured symbol loss | block release, require sidecar regeneration/recovery evidence |
| versioning | sidecar schema mismatch | fail-closed until migration path is explicitly approved |

### 13.5 Verification-Bead Links

Execution-path narratives in this section are expected to be validated and cross-walked by:
- `bd-3dl.12.5` (unit/property tests + structured logging),
- `bd-3dl.12.6` (differential/metamorphic/adversarial validation),
- `bd-3dl.12.7` (E2E scenario scripts + replay/forensics),
- `bd-3dl.23.10` (unit/E2E test corpus and logging evidence crosswalk),
- `bd-3dl.23.11` (EXISTING_JAX_STRUCTURE expansion draft integration).

## 14. Concurrency/Lifecycle Semantics and Ordering Guarantees (DOC-PASS-06)

### 14.1 Current Concurrency Model (Code-Anchored)

Current Rust execution in the scoped transform path is intentionally single-threaded and sequencing-driven. No direct `Arc`/`Mutex`/`RwLock`/thread-spawn primitives are present in `fj-trace`, `fj-interpreters`, `fj-dispatch`, `fj-cache`, `fj-ledger`, `fj-runtime`, or `fj-conformance::durability`.

| Component | Lifecycle contract | Ordering guarantee | Code anchors |
|---|---|---|---|
| Trace frame lifecycle | nested traces are explicit stack frames; root frame is non-popable; finalize requires exactly one frame | strict LIFO frame discipline | `fj-trace::SimpleTraceContext::{push_subtrace,pop_subtrace,finalize}` |
| Trace materialization | each primitive appends one equation to active frame and updates `last_output_ids` | equation insertion order is preserved from trace execution | `fj-trace::TraceContext::process_primitive` |
| IR interpreter lifecycle | env is initialized as `constvars -> invars`, then equations evaluated, then outvars resolved | equation execution is source order; each equation writes its outputs before next equation | `fj-interpreters::eval_jaxpr_with_consts` |
| Partial-eval staging lifecycle | classify equations known/unknown, then produce known/unknown sub-jaxprs plus residual bridge | unknown inputs are ordered as `residuals ++ unknown_inputs` | `fj-interpreters::partial_eval_jaxpr` |
| Staged execution lifecycle | `stage_jaxpr` computes residuals first; `execute_staged` feeds residuals before dynamic args | known-first then unknown execution ordering is fixed | `fj-interpreters::staging::{stage_jaxpr,execute_staged}` |
| Transform stack lifecycle | stack/evidence are verified before execution; recursion peels `head` then `tail` | transform application order is exact ledger order (`head -> tail`) | `fj-core::verify_transform_composition`, `fj-dispatch::execute_with_transforms` |
| Vectorized transform lifecycle | each mapped slice is evaluated, collected, then stacked | `vmap` preserves axis-0 iteration order (`0..lead_len`) and output slot order | `fj-dispatch::execute_vmap` |
| Cache identity lifecycle | canonical payload streamed in fixed field order, then SHA-256 hashed | transform order and `compile_options` lexical order are part of deterministic key material | `fj-cache::build_cache_key_ref`, `fj-cache::hash_canonical_payload_ref` |
| Decision/evidence lifecycle | one `DecisionRecord` computed per dispatch and appended to ledger | entry append order equals dispatch completion order | `fj-dispatch::dispatch`, `fj-ledger::EvidenceLedger::append` |
| Durability lifecycle | sidecar generation, scrub integrity check, and decode-proof are separate explicit phases | symbol records are normalized (`sort_by_key(sbn,esi)`), then decode checks are deterministic over selected records | `fj-conformance::durability::{encode_artifact_to_sidecar,scrub_sidecar,generate_decode_proof}` |

### 14.2 Ordering and Lifecycle Invariants (Operational Form)

1. Transform/evidence cardinality must match before dispatch; mismatch is a hard error.
2. Transform evidence entries must be non-empty; empty evidence is rejected.
3. Current engine allows at most one `grad` and at most one `vmap` per stack.
4. Transform execution is order-sensitive by construction: `execute_with_transforms` applies transforms in stack order and recurses on the tail.
5. `vmap` requires consistent leading-axis length for mapped tensors and preserves mapped-element order in output stacking.
6. `grad` on nested tail transforms currently uses finite-difference fallback; this is an explicit lifecycle fork, not implicit behavior.
7. Jaxpr execution depends on equation order; reordering equations changes def-use timing and is not semantics-preserving by default.
8. Cache keys are deterministic only if canonical field order is preserved (`mode`, `backend`, `transforms`, `compile`, `hook`, `unknown`, `jaxpr`) and `compile_options` remains ordered.
9. Trace finalization is valid only when all nested subtraces are closed; otherwise finalization rejects with `NestedTraceNotClosed`.
10. Durability acceptance is fail-closed: scrub mismatch or decode-proof failure blocks artifact acceptance.

### 14.3 Race/Regression Surfaces (If Parallelism or Refactors Are Introduced)

| Surface | Why risky | Regression mode |
|---|---|---|
| Parallelizing equation evaluation in interpreter | env writes are currently sequential and dependency-ordered | use-before-define or output drift |
| Parallelizing `vmap` slice execution without stable join order | output tensor assembly assumes index-stable ordering | batch permutation drift |
| Replacing ordered maps in cache key material | hash payload currently depends on deterministic map iteration | cache key nondeterminism/collision-like misses |
| Relaxing trace frame lifecycle checks | nested frame closure is currently explicit | leaked/ambiguous trace state at finalize |
| Concurrent writes to sidecar/scrub/proof outputs | durability artifacts are consumed as consistent triplets | torn or mismatched evidence bundles |
| Introducing async cancellation into execution path | `asupersync` bridge is optional and currently observational | partially applied execution without deterministic ledger semantics |

### 14.4 Existing Test Coverage and Remaining Gaps

Coverage already present for lifecycle/ordering contracts:
- transform ordering behavior: `fj-dispatch` test `transform_order_is_explicit`.
- transform stack admissibility and evidence shape: `fj-core` transform composition tests (double `grad`/`vmap` rejection, evidence-cardinality checks).
- trace frame closure lifecycle: `fj-trace` test `finalize_rejects_unclosed_subtrace`.
- staging lifecycle equivalence and split behavior: `fj-interpreters` staging/partial-eval roundtrip tests.
- durability phase behavior: `fj-conformance::durability` unit tests for sidecar/scrub/proof workflow.

Current gap to track:
- No explicit multi-threaded execution tests exist in this slice because execution is deliberately single-threaded. If parallelism is introduced in interpreter/dispatch/vmap/durability paths, ordering invariants above must be re-proven with dedicated concurrent regression tests before promoting behavior to `S0`/`S1` contracts.

## 15. Error Taxonomy, Failure Modes, and Recovery Semantics (DOC-PASS-07)

### 15.1 Error Taxonomy by Boundary

| Boundary | Error surface | Trigger class | User-visible contract |
|---|---|---|---|
| Tensor/value model | `fj-core::ValueError` | shape overflow, element-count mismatch, invalid axis slicing/stacking | deterministic structural error text (`shape`, `expected`, `actual`) |
| IR well-formedness | `fj-core::JaxprValidationError` | duplicate bindings, unbound input vars, output shadowing, unknown outvars | fail-closed IR validation rejection |
| Transform composition proof | `fj-core::TransformCompositionError` | transform/evidence cardinality mismatch, empty evidence, unsupported stack sequence | reject dispatch before execution |
| Primitive semantics | `fj-lax::EvalError` | arity/type/shape mismatches, unsupported primitive behavior | primitive-specific failure with explicit primitive name |
| Interpreter execution | `fj-interpreters::InterpreterError` | input/const arity mismatch, missing variable, invalid primitive output arity | deterministic execution failure (no silent coercion) |
| Partial evaluation split | `fj-interpreters::PartialEvalError` | unknown-mask length mismatch, undefined var, residual mismatch | staging split fails before runtime execution |
| Staging orchestration | `fj-interpreters::StagingError` | wrapped partial-eval/known-eval/unknown-eval failures | phase-attributed staging failure (`partial eval`, `known eval`, `unknown eval`) |
| Dispatch integration | `fj-dispatch::DispatchError` | wrapped cache/interpreter/transform-invariant/transform-execution failures | top-level dispatch error channel with boundary prefix |
| Cache key compatibility gate | `fj-cache::CacheKeyError::UnknownIncompatibleFeatures` | strict-mode unknown incompatible metadata | strict-mode fail-closed cache admission rejection |
| Durability evidence pipeline | `fj-conformance::durability::DurabilityError` | IO/JSON/config/encode/decode/integrity failures | scrub/proof integrity failures block artifact acceptance |

### 15.2 Failure-Mode Matrix (Trigger -> Impact -> Recovery)

| Trigger | Primary error type | Impact | Recovery law (strict) | Recovery law (hardened) |
|---|---|---|---|---|
| Transform evidence count != transform count | `TransformCompositionError::EvidenceCountMismatch` | request not executable | reject request | reject request |
| Empty transform evidence element | `TransformCompositionError::EmptyEvidence` | unprovable stack provenance | reject request | reject request |
| Unsupported transform sequence (e.g., double `grad`) | `TransformCompositionError::UnsupportedSequence` | undefined transform semantics | reject request | reject request |
| Strict mode receives unknown incompatible cache features | `CacheKeyError::UnknownIncompatibleFeatures` | cache identity cannot be trusted | fail-closed before dispatch | permitted (but features included in key payload) |
| Primitive receives wrong arity/type/shape | `EvalError::{ArityMismatch,TypeMismatch,ShapeMismatch}` | primitive semantics undefined | fail evaluation and bubble to dispatch | fail evaluation and bubble to dispatch |
| Interpreter missing bound variable or arity mismatch | `InterpreterError::{MissingVariable,InputArity,ConstArity}` | program cannot evaluate deterministically | fail evaluation | fail evaluation |
| `grad` receives non-scalar input/output in current engine constraints | `TransformExecutionError::{NonScalarGradientInput,NonScalarGradientOutput}` | gradient path invalid | fail transform execution | fail transform execution |
| `vmap` leading-dimension mismatch or empty mapped axis | `TransformExecutionError::{VmapMismatchedLeadingDimension,EmptyVmapOutput,...}` | batch semantics invalid | fail transform execution | fail transform execution |
| Staging split metadata inconsistent with input mask | `PartialEvalError::InputMaskMismatch` / `StagingError::PartialEval` | staged execution plan invalid | reject staging | reject staging |
| Durability scrub hash mismatch or decode-proof failure | `DurabilityError::Integrity` / `DurabilityError::Decode` | artifact trust lost | block release/acceptance | block release/acceptance |

### 15.3 User-Facing Error Semantics

User-visible behavior in the current Rust slice is deterministic and boundary-tagged:
- Dispatch errors are wrapped with explicit boundary prefixes (`cache key error`, `interpreter error`, `transform invariant error`, `transform execution error`) via `fj-dispatch::DispatchError`.
- Primitive and interpreter errors include expected/actual details and primitive names, enabling replay and fixture triage without ambiguous messages.
- Staging and durability errors preserve phase context (`partial eval`, `known eval`, `unknown eval`, `encode`, `decode`, `integrity`) for direct failure localization.
- Strict/hardened divergence is explicit at the cache-compatibility gate: strict rejects unknown incompatible metadata; hardened accepts and hashes it as canonical key material.

### 15.4 Recovery and Fail-Closed Rules

| Error family | Default posture | Allowed recovery |
|---|---|---|
| IR/transform proof violations | fail-closed | none without explicit contract change and new parity evidence |
| Primitive/interpreter/staging semantic violations | fail-closed | fix inputs/program shape; no implicit semantic patching |
| Cache compatibility gate failures | fail-closed in strict | hardened-mode continuation only via explicit compatibility path |
| Cache read corruption (runtime branch) | bounded recovery | bypass cache and re-evaluate, do not write poisoned entries |
| Durability integrity/proof failures | fail-closed | regenerate sidecar/proof artifacts and re-verify |

### 15.5 Coverage and Audit Hooks

Coverage currently present for error semantics and fail-closed behavior:
- `fj-core`: validation and transform-composition rejection tests (duplicate/unbound/shadowing/evidence mismatch/double transform cases).
- `fj-dispatch`: strict-mode unknown-feature fail-closed test and transform-order/transform-execution error-path tests.
- `fj-interpreters`: arity/missing-variable/staging/partial-eval error tests.
- `fj-lax`: arity/type/shape mismatch tests for primitive evaluation.
- `fj-conformance::durability`: scrub/proof error-path tests and integrity checks.

Outstanding gap:
- A single consolidated cross-crate “error taxonomy conformance” suite is not yet present. Current guarantees are distributed across crate-local tests and should be unified in `fj-conformance` for regression-proof, packet-level enforcement.

## 16. Security/Compatibility Edge Cases and Undefined Zones (DOC-PASS-08)

### 16.1 Security/Compatibility Edge-Case Matrix

| Edge case | Threat class | Strict behavior | Hardened behavior | Current anchor |
|---|---|---|---|---|
| Unknown incompatible cache metadata | cache confusion / compatibility drift | fail-closed key rejection | allowed but included in canonical key payload | `fj-cache::{build_cache_key,build_cache_key_ref}` |
| Transform/evidence cardinality mismatch | transform-order abuse / provenance loss | reject before execution | reject before execution | `fj-core::verify_transform_composition` |
| Empty transform evidence string | provenance tamper / unverifiable ledger | reject before execution | reject before execution | `fj-core::verify_transform_composition` |
| Repeated `grad` or repeated `vmap` in stack | unsupported composition regime | reject as unsupported sequence | reject as unsupported sequence | `fj-core::verify_transform_composition` |
| Malformed Jaxpr def-use graph | malformed graph signature | fail validation | fail validation | `fj-core::Jaxpr::validate_well_formed` |
| Missing variable during evaluation | runtime graph corruption | fail interpreter | fail interpreter | `fj-interpreters::eval_jaxpr_with_consts` |
| `vmap` leading-axis mismatch across mapped args | batch-axis contract violation | fail transform execution | fail transform execution | `fj-dispatch::execute_vmap` |
| Durability scrub hash mismatch | artifact tamper/corruption | block artifact acceptance | block artifact acceptance | `fj-conformance::durability::scrub_sidecar` |
| Decode-proof failure under symbol loss | insufficient durability evidence | block artifact acceptance | block artifact acceptance | `fj-conformance::durability::generate_decode_proof` |
| Unknown symbol kind in sidecar | malformed sidecar payload | decode failure | decode failure | `fj-conformance::durability::decode_from_sidecar_records` |
| Nested subtrace left open at finalize | trace lifecycle ambiguity | fail-closed finalize | fail-closed finalize | `fj-trace::SimpleTraceContext::finalize` |

### 16.2 Explicit Undefined/Implementation-Defined Zones (Current Slice)

The following zones are intentionally bounded and must be treated as non-parity-complete until corresponding packet work closes:

1. Transform multiplicity scope is limited: current engine supports at most one `grad` and one `vmap` in a transform stack.
2. Nested-tail `grad` semantics currently use finite-difference fallback, not full symbolic composition in every transform-order scenario.
3. `vmap` semantics are presently constrained to leading-axis mapping requirements encoded in `execute_vmap`.
4. Primitive coverage is incomplete for the full legacy surface; unsupported LAX behaviors explicitly raise `EvalError::Unsupported`.
5. Partial-eval residual abstract values are currently coarse (`F64` scalar placeholders), so residual-type metadata is not yet a full fidelity mirror of all source values.
6. No stabilized top-level facade crate exists yet; workspace `pub` APIs are not final user-contract guarantees by default.

### 16.3 Hardened-Mode Rationale at Compatibility-Sensitive Boundaries

| Boundary | Hardened rationale | Constraint |
|---|---|---|
| Cache metadata compatibility | preserve forward progress for unknown feature fields while preserving deterministic key derivation | unknown fields must still be part of canonical hash payload |
| Dispatch/runtime error channel | add diagnostics context without silently changing transform semantics | no fail-open transform execution on invariant violations |
| Durability verification | preserve operational safety under hostile/partial artifact conditions | scrub/proof failures remain release blockers |

### 16.4 Mitigation Notes and Follow-On Enforcement Hooks

- Keep fail-closed behavior as the default for graph validity, transform proofing, and durability integrity.
- Treat hardened-path continuations as explicit policy exceptions, never silent defaults.
- Back every compatibility-sensitive exception with differential fixtures and ledger evidence.
- Route undefined-zone closure work through packet beads (`FJ-P2C-002/003/004/008`) and conformance gates (`G3`, `G4`, `G8`) before promoting surfaces to stable compatibility claims.

## 17. Pass-A Closure Crosswalk (DOC-PASS-10)

### 17.1 Gap-Matrix Coverage Mapping

Gap-matrix source: `artifacts/docs/bd-3dl.23.1_gap_matrix.v1.md`

| Pass-A objective | Coverage in this document | Evidence/anchor style |
|---|---|---|
| subsystem topology and boundary decomposition | sections `2`, `8`, `10`, `11` | legacy anchor tables + crate boundary matrix + layering constraints |
| ownership and dependency clarity | sections `8.1`, `8.2`, `8.3`, `9` | explicit crate owners, dependency direction map, hidden coupling register |
| execution/control-flow narratives | section `13` | ordered workflow traces with branch/failure tables |
| lifecycle and ordering semantics | section `14` | code-anchored lifecycle matrix + invariants + regression surfaces |
| failure/error/recovery semantics | section `15` | boundary taxonomy + trigger/impact/recovery matrix |
| security/compatibility edge-case treatment | section `16` | strict/hardened edge-case matrix + undefined-zone declarations |

### 17.2 Structural Claim Traceability Index

| Structural claim class | Primary anchor families |
|---|---|
| transform API and composition structure | `jax/_src/api.py`, `jax/_src/interpreters/{partial_eval,ad,batching}.py`, `fj-core`, `fj-dispatch` |
| IR/tracing structure | `jax/_src/core.py`, `fj-trace`, `fj-core` |
| cache/compile structure | `jax/_src/{cache_key,compiler,compilation_cache}.py`, `fj-cache`, `fj-dispatch` |
| runtime/backend boundary structure | `jax/_src/{dispatch,xla_bridge,ffi}.py`, `fj-runtime`, `fj-conformance::durability` |
| workspace topology structure | `Cargo.toml` workspace graph + crate-level public surfaces |

### 17.3 Pass-A Acceptance Readout

- Material expansion: structure coverage now spans topology, dependency direction, lifecycle ordering, error/recovery, and security/compatibility edge cases in one integrated draft.
- Topology/ownership/dependency clarity: crate ownership and dependency direction are explicit and cycle-checked.
- Source-anchored reviewability: every major structural claim is tied to concrete legacy or Rust anchor families above, with sectioned matrices suitable for independent review.

## 18. Unit/E2E Test Corpus and Logging Evidence Crosswalk (DOC-PASS-09)

### 18.1 Behavior -> Verification Asset Crosswalk

| Major behavior family | Unit/property anchors | Differential/E2E anchors | Logging/forensics anchors |
|---|---|---|---|
| transform composition and order semantics | `fj-core` transform-composition tests; `fj-dispatch` transform-order tests | `crates/fj-conformance/tests/e2e.rs` (`e2e_p2c001_transform_order_enforcement`, `e2e_p2c001_transform_stack_composition`); `golden_journeys` transform-composition scenario | `frankenjax.e2e.log.v1` logs with `intermediate_states` + replay command |
| strict vs hardened compatibility split | `fj-cache` strict unknown-feature rejection tests; `fj-dispatch` strict fail-closed cache-gate test | `crates/fj-conformance/tests/e2e.rs` (`e2e_p2c001_strict_vs_hardened_mode_split`) | E2E forensic log (`mode`, `details`, `artifact_refs`) + CI run manifest summaries |
| IR well-formedness and determinism | `fj-core::Jaxpr::validate_well_formed` tests; canonical fingerprint determinism tests | `crates/fj-conformance/tests/ir_core_oracle.rs`; `e2e_p2c001_trace_to_ir_roundtrip`; `e2e_p2c001_ir_determinism_under_replay` | `frankenjax.test-log.v1` (`fixture_id`, `artifact_refs`, `phase_timings`) |
| partial-eval/staging correctness | `fj-interpreters/src/partial_eval.rs` + `staging.rs` unit/property suites | `crates/fj-conformance/tests/pe_staging_oracle.rs`; `crates/fj-conformance/tests/e2e_p2c003.rs` full scenario set | `frankenjax.e2e.log.v1` traces for each P2C-003 scenario |
| primitive semantics and interpreter execution | inline unit tests in `fj-lax`, `fj-interpreters`, `fj-ad` | `crates/fj-conformance/tests/transforms.rs` parity bundle run; `golden_journeys` basic/grad/vmap scenarios | `frankenjax.test-log.v1` across crate-local logged tests |
| durability integrity and artifact acceptance | durability unit tests in `crates/fj-conformance/src/durability.rs`; schema validation in `artifact_schemas.rs` | CI/reliability flow via `scripts/enforce_quality_gates.sh` + failure forensics tests | `frankenjax.failure-diagnostic.v1`, `frankenjax.run-manifest.v1`, scrub/proof artifacts |
| CI gate replayability and failure diagnostics | gate-specific tests in `crates/fj-conformance/tests/failure_forensics.rs` | runtime/e2e gate execution via `scripts/enforce_quality_gates.sh` and `scripts/run_e2e.sh` | failure diagnostic schema + one-command replay command checks |

### 18.2 Replay/Forensics Logging Field Requirements

| Log schema | Required replay/forensics fields | Primary producers |
|---|---|---|
| `frankenjax.test-log.v1` | `test_id`, `fixture_id`, `seed`, `mode`, `env`, `artifact_refs`, `result`, `duration_ms`, `details`, `phase_timings` | `fj-test-utils::TestLogV1`; logged test harnesses in `fj-core`, `fj-trace`, `fj-interpreters` |
| `frankenjax.e2e.log.v1` | `scenario_id`, `packet_id`, `fixture_id`, `mode`, `ts_utc_unix_ms`, `env`, `replay_command`, `input_capture`, `intermediate_states`, `output_capture`, `artifact_refs`, `result`, `details` | `crates/fj-conformance/tests/e2e.rs`, `crates/fj-conformance/tests/e2e_p2c003.rs`; fallback synthesis in `scripts/run_e2e.sh` |
| `frankenjax.golden-journey.v1` | `scenario_id`, `scenario_category`, `replay_command`, `input_capture`, `assertions`, `output_capture`, `result`, `duration_ms` | `crates/fj-conformance/tests/golden_journeys.rs` |
| `frankenjax.failure-diagnostic.v1` | `gate`, `test`, `status`, `summary`, `detail_path`, `replay_cmd`, `timestamp_unix_ms` | `crates/fj-conformance/tests/failure_forensics.rs`; manifest generation scripts |
| `frankenjax.run-manifest.v1` | `run_id`, `summary`, `gate_results`, `failures`, `artifact_index`, `env` | `scripts/generate_run_manifest.sh`; CI artifact aggregation |

### 18.3 Coverage Gaps (Prioritized)

| Priority | Gap | Why it matters | Next enforcement hook |
|---|---|---|---|
| `P0` | No unified cross-crate “error taxonomy conformance” suite | error/fail-closed claims are currently distributed and could drift silently | add conformance meta-suite in `fj-conformance` consuming sections 15/16 matrices |
| `P0` | Packet E2E coverage is concentrated in `P2C-001` and `P2C-003` | structural docs now cover broader surfaces than current scenario corpus | add scenario packets for `P2C-004/005/008` with forensic logs |
| `P1` | Durability is strongly unit-tested but lacks dedicated scenario-style E2E journey | end-to-end operator workflow (generate->scrub->proof->gate) should be replayable as one scenario | add durability E2E test binary and include in `run_e2e.sh` discovery set |
| `P1` | Behavior-to-test mapping is documented but not machine-checked | future edits could invalidate crosswalk completeness | add CI check that verifies crosswalk rows map to existing tests/log schemas |
| `P2` | E2E `seed` is often `None` in current forensic logs | reduces deterministic replay depth for randomized scenarios | standardize deterministic seed capture for E2E harnesses |

### 18.4 Corpus Inventory Snapshot

- Conformance integration tests: `artifact_schemas`, `e2e`, `e2e_p2c003`, `failure_forensics`, `golden_journeys`, `ir_core_oracle`, `pe_staging_oracle`, `smoke`, `transforms`.
- E2E orchestration entrypoint: `scripts/run_e2e.sh` (packet/scenario filters, forensic log fallback generation).
- Reliability gate orchestration: `scripts/enforce_quality_gates.sh` (coverage, flake, runtime, crash, perf, consolidated gate report).
- Flake detector report schema path: `scripts/detect_flakes.sh` -> `artifacts/ci/flake_report.v1.json`.

## 19. Section-Level Confidence Annotations (DOC-PASS-14)

Confidence scale:
- `High`: directly validated against current repository sources with low interpretation risk.
- `Medium-High`: source-anchored synthesis with limited inference.
- `Medium`: source-anchored but more sensitive to fixture churn or future implementation changes.

| Section | Confidence | Basis | Revalidation trigger |
|---|---|---|---|
| `1. Legacy Oracle` | `High` | canonical upstream/local legacy paths are explicit and stable | path relocation or legacy snapshot refresh |
| `2. High-Value File/Function Anchors` | `High` | anchors point to concrete modules/functions in current legacy tree | legacy refactor in mapped files |
| `3. Semantic Hotspots` | `Medium-High` | hotspots are source-derived but prioritization is analytical | parity drift or transform-surface expansion |
| `4. Conformance Fixture Family Anchors` | `Medium` | fixture families are source-backed but corpus can evolve rapidly | fixture version update or new packet additions |
| `5. Compatibility-Critical Inputs for Cache Keying` | `Medium-High` | mapped from cache/compiler sources and current Rust cache surfaces | cache-key schema or compatibility policy change |
| `6. Security and Reliability Risk Areas` | `Medium-High` | threat surfaces tie to concrete modules and doctrine | new threat model or gate-topology changes |
| `7. Extraction Boundary (Current)` | `Medium-High` | boundary map reflects current crate roles and known missing slices | major crate decomposition or merge |
| `8. Workspace Topology (Current Rust Slice)` | `High` | directly aligned with workspace members and crate graph | workspace membership changes |
| `9. Hidden/Implicit Coupling Register` | `Medium-High` | couplings are source-grounded cross-crate inferences | major integration rewiring |
| `10. Conflict Check: Current vs Target Architecture` | `High` | corrected against current workspace (`fj-trace` present; other target crates absent) | introduction of `fj-transforms`/`fj-lowering`/backend facade crates |
| `11. Symbol/API Census and Surface Classification` | `Medium-High` | counts and surface classes are source-derived as of 2026-02-15 | any `pub` surface churn |
| `12. Data Model, State, and Invariant Mapping` | `Medium-High` | maps concrete types/invariants but includes synthesis across layers | IR/runtime model expansion |
| `13. Execution-Path Tracing and Control-Flow Narratives` | `Medium` | narratives are source-anchored but depend on evolving orchestration paths | dispatch/runtime/CI flow rewrites |
| `14. Concurrency/Lifecycle Semantics and Ordering Guarantees` | `Medium-High` | lifecycle/order claims tied to concrete tests and crate behavior | async/runtime integration enablement |
| `15. Error Taxonomy, Failure Modes, and Recovery Semantics` | `Medium-High` | error classes anchored to typed error enums and gate scripts | new error domains or policy shifts |
| `16. Security/Compatibility Edge Cases and Undefined Zones` | `Medium` | explicit undefined zones are documented but expected to shrink | packet closure of currently undefined surfaces |
| `17. Pass-A Closure Crosswalk` | `Medium-High` | crosswalk is internally consistent with documented sections and gap matrix | gap matrix revision |
| `18. Unit/E2E Test Corpus and Logging Evidence Crosswalk` | `Medium` | mapped to current test/log corpus, which is actively growing | new tests/schemas or renamed scenarios |
| `20. Complexity, Performance, and Memory Characterization` | `Medium-High` | complexity classes derived from code-anchored loop/structure analysis; memory growth from concrete data structure inspection | major algorithm changes, new primitive implementations, or parallelism introduction |

Pass-A structure specialist note:
- This confidence table should be refreshed whenever bead `bd-3dl.23.14` (final consistency/sign-off) runs, and at every major crate-boundary change.

## 20. Complexity, Performance, and Memory Characterization (DOC-PASS-05)

### 20.1 Rust Subsystem Complexity Classes (Code-Anchored)

| Subsystem | Function | Complexity | Key dimensions | Code anchor |
|---|---|---|---|---|
| IR validation | `Jaxpr::validate_well_formed` | O((V + E * K) * log V) | V = variables, E = equations, K = avg arity | `fj-core/src/lib.rs:650-705` |
| Canonical fingerprint | `Jaxpr::canonical_fingerprint` | O(E) amortized O(1) | E = equations (cached via OnceLock) | `fj-core/src/lib.rs:620-648` |
| Transform composition proof | `verify_transform_composition` | O(S) | S = transform stack depth (typically <= 3) | `fj-core/src/lib.rs:947-997` |
| Trace materialization | `SimpleTraceContext::process_primitive` | O(K * log T) | K = input arity, T = tracer count | `fj-trace/src/lib.rs:576-603` |
| Shape inference (elementwise) | `infer_primitive_output_avals` | O(max(R1, R2)) | R = tensor rank | `fj-trace/src/lib.rs:369-394` |
| Jaxpr building from trace | `build_closed_jaxpr` | O(E * K * log V) | E = equations, K = avg arity, V = tracers | `fj-trace/src/lib.rs:458-572` |
| Primitive eval (elementwise) | `eval_primitive` (Add, Sub, Mul, etc.) | O(N) | N = total tensor elements | `fj-lax/src/lib.rs:85-229` |
| Primitive eval (dot product) | `eval_primitive` (Dot) | O(N) | N = vector length | `fj-lax/src/lib.rs:277-347` |
| Primitive eval (transpose) | `eval_transpose` | O(N * R) | N = elements, R = rank | `fj-lax/src/lib.rs:548-636` |
| Primitive eval (broadcast) | `eval_broadcast_in_dim` | O(N * R) | N = output elements, R = output rank | `fj-lax/src/lib.rs:640-754` |
| Primitive eval (concatenate) | `eval_concatenate` | O(N * E_in) | N = output elements, E_in = input tensor count | `fj-lax/src/lib.rs:758-883` |
| Primitive eval (slice) | `eval_slice` | O(N + R) | N = output elements, R = rank | `fj-lax/src/lib.rs:887-971` |
| Primitive eval (reduce) | `eval_primitive` (ReduceSum, etc.) | O(N) | N = input elements | `fj-lax/src/lib.rs:350-392` |
| IR interpreter | `eval_jaxpr_with_consts` | O(E * K_max) | E = equations, K_max = max primitive cost | `fj-interpreters/src/lib.rs:65-132` |
| Partial evaluation | `partial_eval_jaxpr` | O(E * K) | E = equations, K = avg equation size | `fj-interpreters/src/partial_eval.rs:112-150` |
| AD forward pass | `forward_with_tape` | O(E * K_max) | E = equations, K_max = max primitive cost | `fj-ad/src/lib.rs:49-109` |
| AD backward pass | `backward` | O(E * K * log V) | E = equations, K = arity, V = variables | `fj-ad/src/lib.rs:111-143` |
| Dispatch orchestration | `dispatch` | O(E + S) | E = equations (hash), S = stack depth | `fj-dispatch/src/lib.rs:135-186` |
| Grad transform execution | `execute_grad` | O(E * K_max) | E = equations | `fj-dispatch/src/lib.rs:204-229` |
| Vmap transform execution | `execute_vmap` | O(L * E * K_max) | L = leading dim, E = equations | `fj-dispatch/src/lib.rs:262-352` |
| Cache key (streaming hash) | `hash_canonical_payload_ref` | O(E + S + C) | E = equations, S = stack, C = compile opts | `fj-cache/src/lib.rs:112-154` |
| Cache key (string build) | `canonical_payload` | O(E + S + C) | same as above | `fj-cache/src/lib.rs:156-183` |
| E-graph build | `jaxpr_to_egraph` | O(E * K) | E = equations, K = avg arity | `fj-egraph/src/lib.rs:82-160` |
| E-graph saturation | `egg::Runner::run` | O(M * (E + R)^2) worst case | M = iterations, R = rewrite rules | external `egg` library |
| Durability encode | `encode_artifact_to_sidecar` | O(S + T * log T) | S = artifact bytes, T = symbol count | `fj-conformance/src/durability.rs:126-200` |
| Durability scrub | `scrub_sidecar` | O(S + T * log T) | same dimensions | `fj-conformance/src/durability.rs` |
| Durability decode proof | `generate_decode_proof` | O(S + T * log T) | same dimensions | `fj-conformance/src/durability.rs` |

### 20.2 Rust Memory Growth Model

| Growth driver | Owner crate | Formula | Peak trigger | Current mitigation |
|---|---|---|---|---|
| Tensor element storage | `fj-core` (`TensorValue`) | O(N) where N = product of dims | large tensors (rank >= 3 with big dims) | dense `Vec<Literal>` storage; no lazy/view semantics |
| Tensor stack (vmap output) | `fj-core` (`stack_axis0`) | O(L * N) | vmap over large tensors | multiple clone/extend calls during stacking |
| Trace context accumulation | `fj-trace` (`tracer_avals`) | O(T) where T = total traced values | complex programs with many traced intermediates | BTreeMap; no pruning during trace |
| Trace frame stack | `fj-trace` (`frame_stack`) | O(D * E_frame) | deep nesting with large per-frame equation counts | Vec of TraceFrames |
| Interpreter environment | `fj-interpreters` (`eval_jaxpr`) | O(V) | V = total bound variables in program | FxHashMap grows with program size; no dead-var cleanup during eval |
| AD tape | `fj-ad` (`TapeEntry` list) | O(E * K) | large programs under grad | stores input values per equation; clones values at tape recording |
| Adjoint map | `fj-ad` (`backward`) | O(V) | all differentiable variables | BTreeMap accumulating gradients |
| Vmap intermediate outputs | `fj-dispatch` (`execute_vmap`) | O(L * O) | L = leading dim, O = output count | all L iteration outputs held simultaneously before stacking |
| Cache key string | `fj-cache` (`canonical_payload`) | O(E + S + C) | large jaxpr with many transforms/options | full string materialization (mitigated by streaming hash path) |
| E-graph node pool | `fj-egraph` (egg internals) | O(N_nodes) unbounded | complex programs with many rewrite opportunities | egg's internal union-find; saturation may grow exponentially |
| Durability symbol records | `fj-conformance` | O(T) where T = symbols | large artifacts with many source blocks | Vec of SidecarSymbolRecord with base64-encoded data |

### 20.3 Identified Hotspot Families (Rust Implementation)

**Family R-H1: Shape manipulation kernel inner loops**
- Components: `eval_transpose`, `eval_broadcast_in_dim`, `eval_concatenate` in `fj-lax`
- Pattern: per-element coordinate computation with O(R) work per element
- Transpose: O(N * R) for stride-based index remapping (`fj-lax/src/lib.rs:603-626`)
- BroadcastInDim: O(N * R) for coordinate-to-flat-index conversion (`fj-lax/src/lib.rs:724-745`)
- Concatenate: O(N * E_in) linear search to locate source tensor per element (`fj-lax/src/lib.rs:843-850`)
- Optimization opportunity: precompute stride tables, use block copy for contiguous regions

**Family R-H2: Vmap execution multiplier**
- Components: `execute_vmap` in `fj-dispatch`
- Pattern: full program evaluation repeated L times (leading dimension)
- Cost: O(L * E * K_max) where inner cost inherits from interpreter + primitives (`fj-dispatch/src/lib.rs:262-352`)
- Memory: O(L * V) intermediate values held until final stacking
- Optimization opportunity: batched primitive evaluation, lazy tensor slicing

**Family R-H3: AD tape and backward pass**
- Components: `forward_with_tape`, `backward` in `fj-ad`
- Pattern: forward records all intermediate values; backward reverses with adjoint accumulation
- Cost: forward O(E * K_max), backward O(E * K * log V) from BTreeMap operations (`fj-ad/src/lib.rs:49-143`)
- Memory: tape clones input values at every equation (`fj-ad/src/lib.rs:98`)
- Optimization opportunity: arena allocation for tape entries, COW value semantics

**Family R-H4: Tensor materialization and copying**
- Components: `TensorValue` construction and `stack_axis0` in `fj-core`
- Pattern: all operations produce fully materialized dense tensor outputs
- Cost: O(N) allocation per elementwise op; O(L * N) for vmap stacking (`fj-core/src/lib.rs:273-277, 364-409`)
- Memory: no sharing between input and output tensors; repeated clone/extend calls
- Optimization opportunity: view/slice semantics, copy-on-write tensor storage

**Family R-H5: Cache key and fingerprint computation**
- Components: `build_cache_key_ref`, `hash_canonical_payload_ref`, `canonical_fingerprint` in `fj-cache` and `fj-core`
- Pattern: string serialization of full IR + SHA-256 hashing
- Cost: O(E + S + C) per dispatch; amortized O(1) for fingerprint via OnceLock (`fj-cache/src/lib.rs:112-154`, `fj-core/src/lib.rs:620-648`)
- Memory: canonical payload string allocation on non-streaming path
- Current mitigation: streaming hash variant (`hash_canonical_payload_ref`) avoids full string materialization

**Family R-H6: E-graph saturation risk**
- Components: `optimize_jaxpr` in `fj-egraph`
- Pattern: egg equality saturation with algebraic rewrite rules
- Cost: O(M * (E + R)^2) worst case; M = saturation iterations, R = 8 rewrite rules
- Memory: e-graph node pool grows with rewrites; worst case unbounded
- Current mitigation: not yet wired into dispatch path; isolated optimization sandbox

### 20.4 Complexity Scaling Under Transform Composition

| Composition | Time scaling | Memory scaling | Dominant hotspot family |
|---|---|---|---|
| `eval(f)` | O(E * K_max) | O(V) env | R-H4 (tensor materialization) |
| `jit(f)` cache miss | O(E + S + C) hash + O(E * K_max) eval | O(V) + O(E + S + C) key | R-H5 (cache) + R-H4 |
| `jit(f)` cache hit | O(E + S + C) hash + O(1) | O(E + S + C) key | R-H5 |
| `grad(f)` | O(E * K_max) forward + O(E * K * log V) backward | O(E * K) tape + O(V) adjoints | R-H3 (tape/backward) |
| `vmap(f)` | O(L * E * K_max) | O(L * V) intermediates + O(L * N) stacking | R-H2 (vmap multiplier) |
| `vmap(grad(f))` | O(L * (E * K_max + E * K * log V)) | O(L * (E * K + V)) | R-H2 x R-H3 |
| `jit(vmap(grad(f)))` miss | O(E+S+C) + O(L * (2 * E * K_max)) | O(L * (E * K + V)) + key | R-H2 x R-H3 x R-H5 |

### 20.5 Legacy-to-Rust Complexity Correspondence

| Legacy pattern | Legacy complexity | Rust equivalent | Rust complexity | Parity note |
|---|---|---|---|---|
| `eval_jaxpr` with dead-var cleanup | O(E) with inline GC | `eval_jaxpr_with_consts` without dead-var cleanup | O(E * K_max) | Rust env grows monotonically; dead-var cleanup is a future optimization |
| `trace_to_jaxpr` tracer accumulation | O(ops) with all tracers stored | `build_closed_jaxpr` | O(E * K * log E) | BTreeMap overhead vs Python dict; similar growth pattern |
| `linearize_jaxpr` with weakref cache | O(ops) cached | `forward_with_tape` + `backward` | O(E * K_max + E * K * log V) | no linearization caching yet; tape approach is direct |
| `cache_key.get` with deepcopy | O(S + C) | `hash_canonical_payload_ref` streaming | O(E + S + C) | streaming hash avoids deepcopy; string fingerprint substitutes for IR serialization |
| `dce_jaxpr` reverse pass | O(E * K) | `fj-interpreters::dce_jaxpr` (in `partial_eval.rs:317`) | O(E * K) | similar reverse-pass pattern |
| `process_call` residual forwarding | O(n * m) with substitution | `partial_eval_jaxpr` single-pass | O(E * K) | no nested call splitting yet; simpler but less optimized |

### 20.6 Performance Budget Crosswalk

Hotspot families map to provisional budgets from EXHAUSTIVE_LEGACY_ANALYSIS.md section 8:

| Budget target | Hotspot families affected | Measurement method |
|---|---|---|
| Transform composition overhead <= +10% | R-H2 (vmap), R-H3 (AD) | `FJ-P2C-001..003` sentinel workloads |
| Cache hit path p95 <= +8% | R-H5 (cache key) | `FJ-P2C-005` cache-key churn workloads |
| p99 regression <= +10% | R-H1 (shape ops), R-H4 (tensor copy) | `FJ-P2C-008` lax primitive throughput |
| Peak RSS regression <= +10% | R-H2 (vmap intermediates), R-H3 (AD tape), R-H4 (tensor materialization) | peak RSS under sentinel composition workloads |

These budgets are pending empirical calibration against first benchmark cycle (see EXHAUSTIVE_LEGACY_ANALYSIS.md section 15 optimization governance).

## 21. DOC-PASS-13 Final Consistency Sweep and Sign-off

Date: 2026-02-20
Agent: CoralOwl (claude-code/opus-4.6)

### 21.1 Verification Passes Completed

| Pass | Result | Scope |
|---|---|---|
| DOC-PASS-05 (Complexity/Performance) | Section 20 added with code-anchored analysis | Both documents |
| DOC-PASS-12 (Red-Team Review) | Critical fixes: fj-trace added to sections 8.1-8.3, dead_code_eliminate corrected to dce_jaxpr, hardened mode behavior fixed, batching attribution corrected | This document |
| DOC-PASS-14 (Structure Specialist) | 12/12 crate symbol counts exact, 6/6 dependency checks pass, 0 layering violations | This document |
| DOC-PASS-15 (Behavior Specialist) | 5/5 behavioral claims confirmed against Rust source code | This document |
| DOC-PASS-16 (Risk/Perf/Test Specialist) | 6/6 test assets verified, 2/2 complexity claims confirmed, budgets consistent | This document |

### 21.2 Critical Fixes Applied

1. Added `fj-trace` to crate ownership matrix (section 8.1), dependency map (section 8.2), and layering constraints (section 8.3).
2. Corrected phantom `dead_code_eliminate` reference to actual `dce_jaxpr` function.
3. Fixed batching state attribution from `fj-dispatch + fj-interpreters` to `fj-dispatch` only.
4. Corrected `validate_well_formed` complexity to include K (arity) factor.
5. Fixed `build_closed_jaxpr` complexity to use V (tracers) not E (equations) in log term.

### 21.3 Sign-off

This document has been expanded through DOC-PASS-01 through DOC-PASS-16 with independent red-team review and three specialist verification passes. All structural claims (crate boundaries, dependency graph, symbol counts, layering), behavioral claims (5 code-verified assertions), and risk/test claims (test corpus existence, complexity accuracy, budget consistency) are verified against current repository state. The document is approved as the normative workspace structure reference for Phase-2C work.
