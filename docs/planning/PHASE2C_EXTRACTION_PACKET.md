# PHASE2C_EXTRACTION_PACKET.md — FrankenJAX

Date: 2026-02-13

Purpose: convert Phase-2 analysis into direct implementation tickets with concrete legacy anchors, target crates, and oracle tests.

## 1. Ticket Packets

| Ticket ID | Subsystem | Legacy anchors (classes/functions) | Target crates | Oracle tests |
|---|---|---|---|---|
| `FJ-P2C-001` | IR core (Jaxpr/Tracer) | `Jaxpr`, `ClosedJaxpr`, `Primitive`, `Trace`, `Tracer`, `eval_jaxpr`, `ensure_no_leaks` in `jax/_src/core.py` | `fj-core` | `tests/jaxpr_util_test.py`, `tests/jaxpr_effects_test.py` |
| `FJ-P2C-002` | API transform front-door | `jit`, `grad`, `value_and_grad`, `jacfwd`, `jacrev`, `vmap`, `make_jaxpr` in `jax/_src/api.py` | `fj-dispatch`, `fj-core` | `tests/api_test.py`, `tests/extend_test.py` |
| `FJ-P2C-003` | Partial evaluation and staging | `JaxprTrace`, `JaxprTracer`, `trace_to_jaxpr_nounits`, `partial_eval_jaxpr_nounits`, `DynamicJaxprTracer` in `jax/_src/interpreters/partial_eval.py` | `fj-interpreters` | staging and transform suites in `tests/*` |
| `FJ-P2C-004` | Dispatch/effects runtime | `apply_primitive`, `RuntimeTokenSet`, `wait_for_tokens`, `_device_put_impl` in `jax/_src/dispatch.py` | `fj-dispatch`, `fj-runtime` | `tests/xla_interpreter_test.py`, dispatch integration tests |
| `FJ-P2C-005` | Compilation cache/keying | `get_cache_key`, `get_executable_and_time`, `put_executable_and_time`, `reset_cache`; `LRUCache` class in `lru_cache.py` | `fj-cache` | `tests/compilation_cache_test.py`, `tests/cache_key_test.py`, `tests/lru_cache_test.py` |
| `FJ-P2C-006` | Backend bridge and platform routing | `register_backend_factory`, `get_backend`, `default_backend`, `device_count` in `jax/_src/xla_bridge.py` | `fj-runtime` | `tests/xla_bridge_test.py` |
| `FJ-P2C-007` | FFI call interface | `register_ffi_target`, `ffi_call`, `ffi_call_lowering`, `FfiEffect` in `jax/_src/ffi.py` | `fj-runtime`, `fj-dispatch` | ffi + lowering targeted tests |
| `FJ-P2C-008` | LAX primitive first wave | `add`, `mul`, `div`, `sqrt`, `exp`, `log`, `broadcast_shapes`, `convert_element_type` in `jax/_src/lax/lax.py` | `fj-lax`, `fj-interpreters` | `tests/lax_test.py`, `tests/lax_numpy_test.py`, `tests/lax_scipy_test.py` |

## 2. Packet Definition Template

For each ticket above, deliver all artifacts in the same PR:

1. `legacy_anchor_map.md`: path + line anchors + extracted behavior.
2. `contract_table.md`: input/output/error + tracing/effect semantics.
3. `fixture_manifest.json`: oracle mapping and fixture IDs.
4. `parity_gate.yaml`: strict + hardened pass criteria.
5. `risk_note.md`: boundary risks and mitigations.

## 3. Strict/Hardened Expectations per Packet

- Strict mode: exact scoped JAX observable behavior.
- Hardened mode: same outward contract with bounded defensive checks (trace validity, cache safety, backend sanity).
- Unknown incompatible backend/ffi/cache metadata: fail-closed.

## 4. Immediate Execution Order

1. `FJ-P2C-001`
2. `FJ-P2C-003`
3. `FJ-P2C-002`
4. `FJ-P2C-004`
5. `FJ-P2C-005`
6. `FJ-P2C-006`
7. `FJ-P2C-007`
8. `FJ-P2C-008`

## 5. Done Criteria (Phase-2C)

- All 8 packets have extracted anchor maps and contract tables.
- At least one runnable fixture family exists per packet in `fj-conformance`.
- Packet-level parity report schema is produced for every packet.
- RaptorQ sidecars are generated for fixture bundles and parity reports.

## 6. Per-Ticket Extraction Schema (Mandatory Fields)

Every `FJ-P2C-*` packet MUST include:
1. `packet_id`
2. `legacy_paths`
3. `legacy_symbols`
4. `ir_contract`
5. `transform_contract`
6. `dispatch_effect_contract`
7. `cache_backend_contract`
8. `error_contract`
9. `strict_mode_policy`
10. `hardened_mode_policy`
11. `excluded_scope`
12. `oracle_tests`
13. `performance_sentinels`
14. `compatibility_risks`
15. `raptorq_artifacts`

Missing fields => packet state `NOT READY`.

## 7. Risk Tiering and Gate Escalation

| Ticket | Risk tier | Why | Extra gate |
|---|---|---|---|
| `FJ-P2C-001` | Critical | Jaxpr/Tracer invariants are foundational | jaxpr invariant replay |
| `FJ-P2C-002` | High | API transform normalization is user-facing | argument normalization lock |
| `FJ-P2C-003` | Critical | partial-eval errors are subtle and severe | residual graph witness suite |
| `FJ-P2C-004` | Critical | effect/token sequencing controls correctness | token ordering trace gate |
| `FJ-P2C-005` | High | cache-key drift creates performance+correctness issues | cache key stability gate |
| `FJ-P2C-007` | High | FFI boundary has security/compatibility risk | ffi contract adversarial gate |

Critical tickets must pass strict drift `0`.

## 8. Packet Artifact Topology (Normative)

Directory template:
- `artifacts/phase2c/FJ-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FJ-P2C-00X/contract_table.md`
- `artifacts/phase2c/FJ-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FJ-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FJ-P2C-00X/risk_note.md`
- `artifacts/phase2c/FJ-P2C-00X/parity_report.json`
- `artifacts/phase2c/FJ-P2C-00X/parity_report.raptorq.json`
- `artifacts/phase2c/FJ-P2C-00X/parity_report.decode_proof.json`

## 9. Optimization and Isomorphism Proof Hooks

Optimization allowed only after strict parity baseline.

Required proof block:
- transform semantics preserved
- effect/token semantics preserved
- cache-key behavior preserved
- fixture checksum verification pass/fail

## 10. Packet Readiness Rubric

Packet is `READY_FOR_IMPL` only when:
1. extraction schema complete,
2. fixture manifest includes happy/edge/adversarial paths,
3. strict/hardened gates are machine-checkable,
4. risk note includes compatibility + security mitigations,
5. parity report has RaptorQ sidecar + decode proof.
