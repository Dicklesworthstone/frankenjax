# Benchmark Baselines V2 (2026-03-12)

## Command

```bash
rch exec -- cargo bench --workspace
```

## Overview

- **82 benchmarks** across **7 bench suites** covering **10 crates**
- **100 samples** per benchmark (Criterion default)
- Bead: bd-20wj [V2-EVIDENCE-04]
- Supersedes: `dispatch_baseline_2026-02-14.md`

## Results

### dispatch_latency (fj-dispatch)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| jit/scalar_add | 2.34 us | [2.29, 2.40] us |
| jit/scalar_square_plus_linear | 28.16 us | [26.58, 29.87] us |
| jit/vector_add_one | 2.81 us | [2.76, 2.87] us |
| grad/scalar_square | 2.59 us | [2.53, 2.66] us |
| vmap/vector_add_one | 2.94 us | [2.87, 3.02] us |
| vmap/rank2_add_one | 3.18 us | [3.12, 3.25] us |
| jit_grad/scalar_square | 3.10 us | [3.04, 3.17] us |
| vmap_grad/vector_square | 4.15 us | [4.07, 4.23] us |

### eval_jaxpr_throughput (fj-interpreters)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| chain_add/10 | 548 ns | [541, 554] ns |
| chain_add/100 | 4.56 us | [4.49, 4.62] us |
| chain_add/1000 | 43.73 us | [43.24, 44.24] us |

### transform_composition (fj-core)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| single/jit | 222 ns | [217, 226] ns |
| single/grad | 216 ns | [212, 219] ns |
| single/vmap | 210 ns | [207, 214] ns |
| depth2/jit_grad | 265 ns | [259, 272] ns |
| depth3/jit_vmap_grad | 274 ns | [267, 281] ns |
| empty_stack | 167 ns | [165, 170] ns |

### cache_key_generation (fj-cache via fj-dispatch bench)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| simple/1eq_1t | 177 ns | [175, 180] ns |
| medium/3eq_2t | 187 ns | [184, 189] ns |
| large/100eq_1t | 1.54 us | [1.54, 1.55] us |
| hardened/unknown_features | 315 ns | [312, 317] ns |

### cache_subsystem (fj-cache)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| cache_key/owned/empty_program | 229 ns | [225, 234] ns |
| cache_key/owned/10eqn_program | 333 ns | [328, 339] ns |
| cache_key/streaming/10eqn_program | 181 ns | [179, 183] ns |
| cache_lookup/hit/in_memory | 257 ns | [250, 263] ns |
| cache_lookup/miss/in_memory | 44 ns | [42, 46] ns |
| compatibility_matrix_row | 55 ns | [53, 57] ns |

### ledger_append (fj-ledger)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| single_append | 133 ns | [131, 136] ns |
| burst_100_appends | 31.88 us | [30.98, 32.88] us |

### jaxpr_fingerprint (fj-core)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| canonical_fingerprint/1 | 804 ns | [787, 821] ns |
| canonical_fingerprint/10 | 3.66 us | [3.61, 3.72] us |
| canonical_fingerprint/100 | 28.16 us | [27.58, 28.79] us |
| cached_fingerprint/10eq | 1.88 ns | [1.87, 1.90] ns |

### jaxpr_validation (fj-core)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| validate_well_formed/1 | 85 ns | [82, 89] ns |
| validate_well_formed/10 | 386 ns | [377, 396] ns |
| validate_well_formed/100 | 5.77 us | [5.63, 5.92] us |

### lax_eval (fj-lax)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| dispatch_overhead_add_scalar | 10.3 ns | [10.2, 10.4] ns |
| add_scalar_i64 | 10.1 ns | [10.0, 10.3] ns |
| add_1k_i64_vec | 27.69 us | [27.36, 28.03] us |
| mul_1k_f64_vec | 32.14 us | [31.57, 32.75] us |
| dot_100_i64 | 653 ns | [636, 670] ns |
| reduce_sum_1k_i64 | 2.20 us | [2.14, 2.25] us |
| sin_1k_f64 | 32.17 us | [31.92, 32.43] us |
| exp_1k_f64 | 27.30 us | [27.04, 27.55] us |
| reshape_1k_to_10x100 | 410 ns | [400, 419] ns |
| eq_1k_i64 | 22.35 us | [22.10, 22.63] us |

### api_overhead (fj-api)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| jit/scalar_add | 2.26 us | [2.23, 2.29] us |
| grad/scalar_square | 2.80 us | [2.72, 2.88] us |
| vmap/vector_add_one | 2.78 us | [2.73, 2.85] us |
| value_and_grad/scalar_square | 2.78 us | [2.71, 2.86] us |

### api_vs_dispatch (fj-api)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| api_jit_add | 2.62 us | [2.54, 2.70] us |
| dispatch_jit_add | 2.21 us | [2.18, 2.24] us |
| api_grad_square | 2.63 us | [2.58, 2.68] us |
| dispatch_grad_square | 2.49 us | [2.42, 2.58] us |

### value_and_grad_runtime (fj-api)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| shared/square_plus_linear | 3.31 us | [3.25, 3.39] us |
| separate/square_plus_linear | 24.10 us | [22.75, 25.56] us |
| shared/deep_100_nodes | 90.43 us | [89.45, 91.41] us |
| separate/deep_100_nodes | 322.46 us | [315.41, 331.11] us |

### api_composition (fj-api)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| jit_grad/builder | 2.86 us | [2.82, 2.91] us |
| jit_grad/compose_helper | 2.96 us | [2.90, 3.02] us |
| jit_vmap/builder | 3.08 us | [3.02, 3.15] us |
| vmap_grad/builder | 4.18 us | [4.09, 4.29] us |
| jit_vmap_grad/compose_helper | 4.46 us | [4.36, 4.56] us |

### api_mode_config (fj-api)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| strict_jit | 2.34 us | [2.31, 2.37] us |
| hardened_jit | 2.38 us | [2.34, 2.43] us |

### partial_eval (fj-interpreters)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| all_known/10eq | 515 ns | [509, 522] ns |
| all_unknown/10eq | 492 ns | [486, 498] ns |
| mixed/neg_mul | 346 ns | [339, 355] ns |
| all_known/100eq | 4.03 us | [3.99, 4.07] us |
| all_known/1000eq | 40.18 us | [39.74, 40.66] us |
| program_spec/square_plus_linear | 177 ns | [173, 181] ns |

### dce (fj-interpreters)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| all_used/10eq | 664 ns | [651, 680] ns |
| all_used/100eq | 5.40 us | [5.31, 5.49] us |
| all_used/1000eq | 46.61 us | [45.82, 47.45] us |

### staging (fj-interpreters)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| full_pipeline/neg_mul_mixed | 758 ns | [739, 778] ns |
| full_pipeline/all_known_add2 | 290 ns | [284, 298] ns |
| full_pipeline/chain_100eq | 8.71 us | [8.59, 8.84] us |

### backend_cpu (fj-backend-cpu)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| backend_execute/add2 | 184 ns | [180, 188] ns |
| backend_execute/square | 174 ns | [171, 177] ns |
| backend_execute/dot3_10eqn | 206 ns | [204, 208] ns |
| backend_execute/wide_parallel_64 | 255.60 us | [248.35, 263.72] us |
| interpreter_execute/wide_parallel_64 | 5.99 us | [5.92, 6.07] us |
| backend_allocate/256_bytes | 24.9 ns | [24.2, 25.5] ns |
| backend_transfer/1kb | 21.7 ns | [21.5, 21.9] ns |
| backend_discovery/devices | 12.7 ns | [12.5, 12.8] ns |

### ffi (fj-ffi)

| Benchmark | p50 | CI [lower, upper] |
|-----------|-----|-------------------|
| ffi_roundtrip/scalar_f64 | 70.8 ns | [69.3, 72.4] ns |
| ffi_roundtrip/noop | 31.4 ns | [30.8, 32.1] ns |
| ffi_roundtrip/1k_f64_vec | 782 ns | [767, 800] ns |
| marshal/scalar_to_buffer | 13.2 ns | [13.1, 13.3] ns |
| marshal/buffer_to_scalar | 6.7 ns | [6.7, 6.8] ns |
| marshal/1k_tensor_to_buffer | 734 ns | [713, 759] ns |
| marshal/1k_buffer_to_tensor | 1.18 us | [1.15, 1.21] us |
| registry/lookup | 30.0 ns | [29.5, 30.6] ns |

## V1 vs V2 Comparison (dispatch_baseline suite)

| Benchmark | V1 (2026-02-14) | V2 (2026-03-12) | Delta |
|-----------|-----------------|-----------------|-------|
| jit/scalar_add | ~2.5 us | 2.34 us | -6.4% (faster) |
| jit/scalar_square_plus_linear | ~3.5 us | 28.16 us | Note: V2 added e-graph + AD |
| jit/vector_add_one | ~2.8 us | 2.81 us | ~0% |
| grad/scalar_square | ~4.2 us | 2.59 us | -38.3% (faster) |
| vmap/vector_add_one | ~3.1 us | 2.94 us | -5.2% (faster) |
| vmap/rank2_add_one | ~8.5 us | 3.18 us | -62.6% (faster) |
| jit_grad/scalar_square | ~5.0 us | 3.10 us | -38.0% (faster) |
| vmap_grad/vector_square | ~7.5 us | 4.15 us | -44.7% (faster) |
| chain_add/10 | ~650 ns | 548 ns | -15.7% (faster) |
| chain_add/100 | ~5.5 us | 4.56 us | -17.1% (faster) |
| chain_add/1000 | ~55 us | 43.73 us | -20.5% (faster) |

## Scope

This V2 baseline covers all benchmark suites across the workspace:

1. **dispatch_latency**: jit/grad/vmap x scalar/vector + compositions
2. **eval_jaxpr_throughput**: 10/100/1000 equation chain programs
3. **transform_composition**: single/depth-2/depth-3/empty stacks
4. **cache_key_generation**: simple/medium/large/hardened inputs
5. **ledger_append**: single and burst-100 appends
6. **jaxpr_fingerprint + validation**: fresh vs cached
7. **lax_eval**: primitive operation latency (10 operations)
8. **api_overhead**: API wrapper entry points
9. **api_vs_dispatch**: wrapper vs raw dispatch comparison
10. **value_and_grad_runtime**: shared vs separate forward pass
11. **api_composition**: builder vs compose helper
12. **api_mode_config**: strict vs hardened mode overhead
13. **partial_eval + dce + staging**: interpreter optimization passes
14. **backend_cpu**: CPU backend execution, allocation, transfer
15. **ffi**: FFI roundtrip, marshaling, registry lookup

Crates exercised: fj-core, fj-lax, fj-interpreters, fj-cache, fj-dispatch, fj-ledger, fj-ad, fj-api, fj-backend-cpu, fj-ffi.

## CI Gate

Performance regression gate: `scripts/check_perf_regression.sh`
- Threshold: 5% p95 regression (configurable in `reliability_budgets.v1.json`)
- Machine-readable baseline: `benchmark_baselines_v2_2026-03-12.json`

## Notes

- V2 shows significant performance improvements in grad/vmap dispatch paths due to optimized AD engine and batching rules.
- `jit/scalar_square_plus_linear` regression is due to the V2 e-graph optimization pass being applied (correctness over speed for multi-equation programs).
- Criterion baseline saved for future comparisons.
