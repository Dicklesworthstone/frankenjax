# FrankenJAX Perf Release Readiness Scorecard

Updated: 2026-06-19

Scope: verify recent code-first `fj-lax` perf backlog against original JAX on
realistic warmed CPU workloads. This scorecard records measured readiness only;
unmeasured `code-first batch-test pending` entries remain outside the score.

## Environment

- Agent: cod-b / WildForge
- Host: AMD Ryzen Threadripper PRO 5975WX, 64 logical CPUs, Linux 6.17.0-35
- Rust: `rustc 1.98.0-nightly (f20a92ec0 2026-06-07)`
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-b`
- Rust bench command family: `cargo bench -p fj-lax --bench lax_baseline`
- JAX oracle: `uv run --with 'jax[cpu]==0.9.2' --with numpy python`
- JAX/JAXLIB: 0.9.2 / 0.9.2, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 12 batches x 20
  iterations per workload

Additional current clamp gauntlet environment:

- Agent: cod-a / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-a`
- Rust bench command: `cargo bench -p fj-lax --bench clamp_gauntlet`
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 50 runs x 100
  inner loops per workload

## Measured Workloads

| Bead | Workload | Rust timing | JAX timing | Rust/JAX | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| frankenjax-19wst | `tile_scalar_f32_1024x1024` | 51.435 us | 317.753 us | 0.162 | Rust 6.18x faster |
| frankenjax-19wst | `tile_scalar_complex128_1024x1024` | 412.679 us | 579.030 us | 0.713 | Rust 1.40x faster |
| frankenjax-1z7k9 | `complex_f32_tensor_scalar_1m` | 1.379 ms | 1.272 ms | 1.084 | Rust 1.08x slower |
| frankenjax-1z7k9 | `complex_f64_tensor_scalar_1m` | 0.914 ms | 3.730 ms | 0.245 | Rust 4.08x faster |
| frankenjax-mcqr.105 | `f32_mixed_scalar_tensor_1m` | 159.383 us mean | 115.540 us mean | 1.379 | Rust 1.38x slower |
| frankenjax-mcqr.105 | `f64_mixed_scalar_tensor_1m` | 996.940 us mean | 213.651 us mean | 4.666 | Rust 4.67x slower |
| frankenjax-mcqr.106 | `bf16_mixed_scalar_tensor_1m` | 15.571 ms mean | 121.313 us mean | 128.353 | Rust 128.35x slower |
| frankenjax-mcqr.106 | `f16_mixed_scalar_tensor_1m` | 19.859 ms mean | 371.729 us mean | 53.423 | Rust 53.42x slower |
| frankenjax-mcqr.107 | `bf16_tensor_tensor_tensor_1m` | 15.652 ms mean | 183.707 us mean | 85.200 | Rust 85.20x slower |
| frankenjax-mcqr.107 | `f16_tensor_tensor_tensor_1m` | 20.951 ms mean | 229.951 us mean | 91.110 | Rust 91.11x slower |

## Readiness

- JAX domination score for this measured set: 30/100.
- Basis: 3 of 10 measured realistic workloads beat warmed original JAX CPU.
- Release blockers for this set:
  - `complex_f32_tensor_scalar_1m` remains a near-parity JAX loss at Rust/JAX
    1.084.
  - Dense clamp verification is a Rust-internal keep but an external JAX loss
    on all six measured clamp workloads, with the worst gap in half-precision
    tensor clamp.
- Reverts: none. No measured workload showed that reverting the committed dense
  path would improve results; the mixed complex constructor cluster keeps the
  F64 win and routes the F32 loss to deeper representation work, and the clamp
  cluster keeps because every dense path beat the boxed Rust reference by at
  least 1.68x.
- Next measured gate: Complex64/F32 constructor must target packed Complex64
  output construction or a fused real-to-complex path, not another retry of the
  boxed-literal-elision lever.
- Next clamp gate: BF16/F16 clamp must remove per-lane `clamp_literal`
  widen/round overhead with a raw-bit proof harness; F32/F64 clamp must target
  SIMD/parallel throughput or output allocation/fusion rather than another
  boxed-literal-elision retry.
