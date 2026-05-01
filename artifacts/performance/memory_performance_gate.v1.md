# Memory Performance Gate

- schema: `frankenjax.memory-performance-gate.v1`
- bead: `frankenjax-cstq.4`
- status: `pass`
- backend: `linux_procfs_status_vm_hwm`
- peak RSS budget: `1073741824` bytes

| Phase | Workload | Status | Iterations | Peak RSS bytes | Delta RSS bytes | Evidence |
|---|---|---:|---:|---:|---:|---|
| `trace` | `trace_canonical_fingerprint` | `pass` | `32` | `5677056` | `1044480` | `benchmark:jaxpr_fingerprint/canonical_fingerprint/10` |
| `compile_dispatch` | `dispatch_jit_scalar` | `pass` | `16` | `10985472` | `5111808` | `benchmark:dispatch_latency/jit/scalar_add` |
| `ad` | `ad_grad_scalar` | `pass` | `16` | `11587584` | `598016` | `benchmark:dispatch_latency/grad/scalar_square` |
| `vmap` | `vmap_vector_add_one` | `pass` | `16` | `11919360` | `327680` | `benchmark:dispatch_latency/vmap/vector_add_one` |
| `fft` | `fft_complex_vector` | `pass` | `16` | `12050432` | `131072` | `surface:lax_fft` |
| `linalg` | `linalg_cholesky_matrix` | `pass` | `16` | `12050432` | `0` | `surface:lax_linalg_cholesky` |
| `cache` | `cache_hit_miss` | `pass` | `32` | `12058624` | `8192` | `benchmark:cache_subsystem/cache_lookup/hit/in_memory, benchmark:cache_subsystem/cache_lookup/miss/in_memory` |
| `durability` | `durability_sidecar_round_trip` | `pass` | `1` | `12423168` | `364544` | `artifacts/performance/memory_durability_probe.v1.json, artifacts/performance/memory_durability_probe.v1.sidecar.json, artifacts/performance/memory_durability_probe.v1.scrub.json, artifacts/performance/memory_durability_probe.v1.proof.json` |

## Issues

No memory performance issues found.
