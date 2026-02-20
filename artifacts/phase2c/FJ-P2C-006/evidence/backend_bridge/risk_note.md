# FJ-P2C-006 Risk Note: Backend Bridge and Platform Routing

## Threat Analysis

| # | Threat | Residual Risk | Mitigation Evidence |
|---|--------|---------------|---------------------|
| 1 | Device spoofing | Negligible (V1) | Backend string is opaque metadata; eval_jaxpr executes identically regardless. Security threat matrix threat #1 |
| 2 | Buffer overflow via wrong layout | Negligible | TensorValue::new() validates shape/data match. `#![forbid(unsafe_code)]` on all crates. Security threat matrix threat #2 |
| 3 | Cross-device data leakage | None (V1) | Single-device architecture. All Values host-resident. Security threat matrix threat #3 |
| 4 | Resource exhaustion | Low | Host OOM → process abort (not exploitable panic). No device memory pool. Security threat matrix threat #4 |
| 5 | Backend priority manipulation | Negligible (V1) | No dynamic discovery. Backend is caller-specified. Security threat matrix threat #5 |

## Invariant Checklist

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | CPU backend always available | VERIFIED | cpu_backend_name, cpu_backend_default_device, e2e_cpu_backend_all_primitives |
| 2 | Backend trait uniform across platforms | VERIFIED | Backend trait definition in backend.rs; CpuBackend implements all methods |
| 3 | Value round-trip preserves data | VERIFIED | cpu_backend_buffer_roundtrip_preserves_data, prop_cpu_backend_transfer_preserves_data (256+ cases) |
| 4 | Backend string flows into cache key | VERIFIED | key_sensitivity_backend_change (fj-cache), prop_distinct_backends_produce_distinct_keys |
| 5 | Backend-agnostic semantics | VERIFIED | oracle_cpu_backend_matches_eval_jaxpr_add2, oracle_cpu_backend_matches_all_programs |
| 6 | Fallback routing works | VERIFIED | registry_resolve_with_fallback, adversarial_fallback_from_nonexistent_to_cpu, e2e_backend_fallback_to_cpu |
| 7 | Multi-device isolation | VERIFIED | two_cpu_backend_instances_are_independent, metamorphic_same_program_different_device_counts |
| 8 | Error propagation is structured | VERIFIED | adversarial_unsupported_backend_request, adversarial_allocate_nonexistent_device, adversarial_transfer_nonexistent_device |

## Performance Summary

| Benchmark | Latency | Target | Status |
|-----------|---------|--------|--------|
| backend_execute/add2 | 145 ns | < 5 µs | PASS |
| backend_execute/square | 136 ns | < 5 µs | PASS |
| backend_execute/dot3_10eqn | 148 ns | < 5 µs | PASS |
| backend_allocate/256_bytes | 28 ns | < 100 ns | PASS |
| backend_transfer/1kb | 26 ns | < 100 ns | PASS |
| backend_discovery/devices | 16 ns | < 100 ns | PASS |

## Test Count

- 37 unit tests (fj-backend-cpu) + 14 unit tests (fj-runtime backend/device/buffer)
- 5 oracle tests (backend_bridge_oracle)
- 4 metamorphic tests
- 6 adversarial tests
- 6 E2E tests (e2e_p2c006)
- **72 total**, all passing
