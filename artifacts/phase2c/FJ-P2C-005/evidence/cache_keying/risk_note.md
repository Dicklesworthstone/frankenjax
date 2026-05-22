# FJ-P2C-005 Risk Note: Compilation Cache/Keying

## Threat Analysis

| # | Threat | Residual Risk | Mitigation Evidence |
|---|--------|---------------|---------------------|
| 1 | Cache key collision | Negligible | SHA-256 collision resistance < 2^-128. 1540+ distinct keys generated with 0 collisions (oracle_collision_resistance_10k, e2e_cache_key_collision_resistance) |
| 2 | Cache poisoning | None (V1) | No persistent cache storage in V1 dispatch path. CacheManager exists but is not wired into dispatch. Future: integrity verification via SHA-256 on read |
| 3 | Delimiter injection | Low | Backend names containing `\|` produce different keys than clean names (adversarial_delimiter_in_backend_name). Future: length-prefix fields |
| 4 | Stale artifact serving | None (V1) | No persistent cache in dispatch. Namespace prefix `fjx-` provides coarse version isolation |
| 5 | Key drift across versions | Low | Golden key stability harness (golden_keys_are_internally_consistent) detects accidental drift. No version string in key yet |

## Invariant Checklist

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Same inputs → same key (deterministic) | VERIFIED | cache_key_is_stable_for_identical_inputs, oracle_cache_key_determinism_across_calls (100 iterations) |
| 2 | Different inputs → different keys (collision resistance) | VERIFIED | 7 key_sensitivity_* tests, oracle_collision_resistance_10k (1540+ keys, 0 collisions), prop_distinct_backends_produce_distinct_keys |
| 3 | Strict mode rejects unknown features | VERIFIED | strict_mode_rejects_unknown_features, e2e_strict_mode_rejection, adversarial_strict_rejects_unknown_features_uniformly |
| 4 | Hardened mode includes unknown features in key | VERIFIED | hardened_mode_accepts_unknown_features, e2e_hardened_mode_inclusion |
| 5 | Streaming and owned builders produce identical hashes | VERIFIED | streaming_hash_matches_owned_hash, adversarial_streaming_and_owned_agree_under_stress |
| 6 | Cache eviction does not affect correctness | VERIFIED | metamorphic_eviction_preserves_correctness, e2e_cache_eviction_under_pressure |
| 7 | Persistent cache survives process restart | VERIFIED | cache_manager_file_backed_survives_reopen, e2e_cache_key_stability_across_restart, adversarial_file_cache_persistence_survives_reopen |
| 8 | Cache key includes all semantically relevant fields | VERIFIED | 7 key_sensitivity tests cover mode, backend, transforms, compile_options, custom_hook, jaxpr |

## Performance Summary

| Benchmark | Latency | Target | Status |
|-----------|---------|--------|--------|
| cache_key/streaming/10eqn_program | 234 ns | < 1 µs | PASS |
| cache_key/owned/10eqn_program | 410 ns | < 1 µs | PASS |
| cache_lookup/miss/in_memory | 52 ns | < 200 ns | PASS |
| cache_lookup/hit/in_memory | 305 ns | < 2 µs | PASS |
| compatibility_matrix_row | 63 ns | — | PASS |

## Test Count

- 36 unit tests (fj-cache)
- 5 oracle tests (cache_keying_oracle)
- 4 metamorphic tests
- 6 adversarial tests
- 6 E2E tests (e2e_p2c005)
- **57 total**, all passing
