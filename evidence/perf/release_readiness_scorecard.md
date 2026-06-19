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
- Rust bench command: `cargo bench -p fj-lax --bench clamp_gauntlet`; current
  bitcast rows use `cargo bench -p fj-lax --bench lax_baseline` with the
  `eval/bitcast_*` filter.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 50 runs x 100
  inner loops per clamp workload; 30 runs x 50 inner loops per bitcast workload

Additional current scalar broadcast environment:

- Agent: cod-b / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-b`
- Rust bench command: `cargo bench -p fj-lax --bench lax_baseline` with the
  `eval/broadcast_scalar` filter, local same-host execution.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 60 runs x 1000
  inner loops per scalar broadcast workload

## Measured Workloads

| Bead | Workload | Rust timing | JAX timing | Rust/JAX | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| frankenjax-19wst | `tile_scalar_f32_1024x1024` | 51.435 us | 317.753 us | 0.162 | Rust 6.18x faster |
| frankenjax-19wst | `tile_scalar_complex128_1024x1024` | 412.679 us | 579.030 us | 0.713 | Rust 1.40x faster |
| frankenjax-1z7k9 | `complex_f32_tensor_scalar_1m` | 1.379 ms | 1.272 ms | 1.084 | Rust 1.08x slower |
| frankenjax-1z7k9 | `complex_f64_tensor_scalar_1m` | 0.914 ms | 3.730 ms | 0.245 | Rust 4.08x faster |
| frankenjax-alc0j | `broadcast_scalar_u32_1024x1024` | 51.307 us mean | 79.979 us mean | 0.642 | Rust 1.56x faster (noisy CV) |
| frankenjax-alc0j | `broadcast_scalar_u64_1024x1024` | 104.911 us mean | 124.569 us mean | 0.842 | Rust 1.19x faster (noisy near-tie) |
| frankenjax-alc0j | `broadcast_scalar_complex128_1024x1024` | 283.150 us mean | 245.381 us mean | 1.154 | Rust 1.15x slower (noisy loss) |
| frankenjax-mcqr.105 | `f32_mixed_scalar_tensor_1m` | 159.383 us mean | 115.540 us mean | 1.379 | Rust 1.38x slower |
| frankenjax-mcqr.105 | `f64_mixed_scalar_tensor_1m` | 996.940 us mean | 213.651 us mean | 4.666 | Rust 4.67x slower |
| frankenjax-mcqr.106 | `bf16_mixed_scalar_tensor_1m` | 15.571 ms mean | 121.313 us mean | 128.353 | Rust 128.35x slower |
| frankenjax-mcqr.106 | `f16_mixed_scalar_tensor_1m` | 19.859 ms mean | 371.729 us mean | 53.423 | Rust 53.42x slower |
| frankenjax-mcqr.107 | `bf16_tensor_tensor_tensor_1m` | 15.652 ms mean | 183.707 us mean | 85.200 | Rust 85.20x slower |
| frankenjax-mcqr.107 | `f16_tensor_tensor_tensor_1m` | 20.951 ms mean | 229.951 us mean | 91.110 | Rust 91.11x slower |
| frankenjax-mcqr.101 | `bitcast_f32_i32_1m` | 430.783 us mean | 133.290 us mean | 3.232 | Rust 3.23x slower (45.35x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_i32_f32_1m` | 642.693 us mean | 93.945 us mean | 6.841 | Rust 6.84x slower (35.28x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_f64_u64_1m` | 270.723 us mean | 176.611 us mean | 1.533 | Rust 1.53x slower (91.41x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_u64_f64_1m` | 228.320 us mean | 175.404 us mean | 1.302 | Rust 1.30x slower (89.89x vs literal ref) |
| frankenjax-mcqr.100 | `bitcast_f32_bf16_1m` | 2.710 ms mean | 138.058 us mean | 19.630 | Rust 19.63x slower (21.29x vs literal ref) |
| frankenjax-mcqr.100 | `bitcast_bf16_f32_1m` | 669.626 us mean | 139.426 us mean | 4.803 | Rust 4.80x slower (47.11x vs literal ref) |
| frankenjax-e07uw | `fusion_arith8_f64_1m` (jit chain) | 3.320 ms fused | 272.7 us mean | 12.175 | Rust 12.18x slower |
| frankenjax-bjqfr | `fusion_bf16_broadcast_1m` | 10.776 ms | 146.9 us mean | 73.357 | Rust 73.36x slower (reverted) |
| frankenjax-f62hx | `transpose_attn_BSHD_f32` (block-copy) | 791.5 us | 186.7 us mean | 4.239 | Rust 4.24x slower (10.3x vs naive) |
| frankenjax-thnjs | `broadcast_bias_D768_to_4096x768_f32` (block-copy) | 283.75 us | 178.9 us mean | 1.586 | Rust 1.59x slower (21.8x vs naive) |
| frankenjax-hfq7o | `integer_pow2_f64_1m` (v*v fix) | 405.45 us | 184.1 us mean | 2.202 | Rust 2.20x slower (was 12.9x) |
| frankenjax-hfq7o | `integer_pow2_f32_1m` (v*v fix) | 169.61 us | 121.1 us mean | 1.400 | Rust 1.40x slower (was 21.8x) |
| frankenjax-idunl | `slice_crop_1024x1024_to_512x512_f32` (block-copy) | 45.97 us | 44.17 us mean | 1.041 | Rust 1.04x slower (TIE; 6.1x vs naive) |
| (dense contiguous gather) | `gather_embed_16384x768_take4096_f32` | 1.145 ms | 271.3 us mean | 4.220 | Rust 4.22x slower (random-read-bound; 4.05x vs naive) |
| frankenjax-7eqrs | `complex_ctor_re_im_to_c128_1m` (de-box) | 775.72 us | 497.34 us mean | 1.561 | Rust 1.56x slower (near-parity; 25.2x vs boxed) [rch] |
| frankenjax-dxqfj | `split_multi_1024x1024_f32_axis1` (lazy sections) | 96.651 us mean | 149.082 us bare / 136.983 us materialized mean | 0.648 / 0.706 | Rust 1.42-1.54x faster (noisy CV; keep) |
| elementwise | `add_f64_1m` (LOCAL same-host) | 415.00 us | 192.0 us mean | 2.162 | Rust 2.16x slower (alloc+AVX2) |
| elementwise | `add_f32_1m` (LOCAL same-host) | 135.98 us | 80.4 us mean | 1.691 | Rust 1.69x slower |
| elementwise | `mul_f64_1m` (LOCAL same-host) | 422.96 us | 161.7 us mean | 2.615 | Rust 2.61x slower |

## Readiness

- METHODOLOGY CORRECTION (2026-06-19): prior block-copy/de-box rows measured Rust
  on a REMOTE rch worker but JAX LOCALLY. Same-binary calibration: rch worker is
  ~1.45x SLOWER than local, so those rch rows are pessimistic by ~1.45x. Corrected
  same-host estimates FLIP several to wins/ties: slice ~0.72x (Rust FASTER),
  integer_pow2 f32 ~0.97x (~tie/win), broadcast ~1.10x, complex_ctor ~1.08x. Future
  vs-JAX rows MUST run the Rust bench LOCALLY (cargo bench, not rch).
- JAX domination score (same-host corrected estimate): ~35/100 — at least slice
  (0.72x) and integer_pow2 f32 (0.97x) beat or tie JAX same-host, with broadcast/
  complex_ctor within ~10%. The six new bitcast rows add internal wins but no
  new external JAX wins.
- The de-box category SPLITS: bandwidth-bound de-box (complex ctor, integer_pow
  f32) ties/beats JAX same-host; heavy-per-lane de-box (clamp half 53-128x) does
  not — there per-lane work, not boxing, dominates.
- Elementwise add/mul same-host 1.69-2.61x slower: structural (per-op output
  allocation vs XLA buffer reuse + AVX2-vs-AVX512), not the elementwise loop
  (native-vs-widen f32 tie proves the loop is bandwidth-bound). Sharpened pattern from 4 measured structural ops: the JAX gap
  is set by the READ access pattern, not the copy:
    - SEQUENTIAL read (slice 1.04x TIE, broadcast 1.59x) — near-parity.
    - RANDOM/STRIDED read (transpose 4.24x, gather/embedding 4.22x) — ~4x loss.
  All are 4-22x internal wins. Plus the de-box-fixed compute (integer_pow2 f32
  1.40x / f64 2.20x). Remaining JAX losses are: random/strided-read structural ops
  (~4x, memory-access-pattern/prefetch gap), compute-bound transcendentals/matmul
  (+fma gated), and the interpreter-vs-XLA gap on jit'd chains.
- BEST gauntlet result: the integer_pow x**2 fix (hfq7o) — measurement caught a
  runtime-`powi(2)` LIBCALL (~6.75 GB/s); replacing it with `v*v` (bit-identical)
  gave 5.8x (f64) / 15.6x (f32) and closed the JAX gap from 12.9-21.8x to
  1.4-2.2x. Pattern: any op taking a runtime small-int power must not call powi.
- Contiguous-block memcpy cluster (f62hx/thnjs + siblings) is the BEST-performing
  lever family measured this conversation: genuine algorithmic wins (transpose
  10.3x, broadcast 21.8x vs the per-element odometer) with a JAX gap of only
  1.6-4.24x (broadcast near-parity, write-bound; transpose 4.24x, strided) —
  categorically better than the de-box dense clusters (50-128x JAX loss). KEEP all.
  Residual JAX gaps are memory-store/layout, not algorithm: broadcast needs
  streaming stores; transpose needs layout-aware elision. NOT more block-copy.
- SplitMulti lazy section construction (dxqfj) is a measured same-host win against
  both JAX bare split and materialized split (Rust/JAX 0.648-0.706), but the row
  stays noisy (Rust CV 6.85%, JAX CV 10-12%). KEEP as a construction-path win;
  do not upgrade it to certification-grade evidence until a tighter run gets both
  sides under 5% CV.
- eval_jaxpr fusion cluster (e07uw/7g72q/rl9ha/bjqfr): Rust-internal win
  (fused vs unfused per-op: f64 5.06x, f32 6.70x, f32 broadcast 15.75x, i64
  broadcast 3.21x) but NOT JAX domination — the tree-walking interpreter (even
  fused) is ~12x slower than XLA-compiled jax.jit on the f64 chain. Closing this
  needs the compiled-jaxpr arena executor (z6o97/6dfew), not more fusion ops.
- Release blockers for this set:
  - `complex_f32_tensor_scalar_1m` remains a near-parity JAX loss at Rust/JAX
    1.084.
  - `broadcast_scalar_complex128_1024x1024` is a noisy JAX loss at Rust/JAX
    1.154, while the U32/U64 scalar broadcast siblings are noisy wins. Treat the
    scalar broadcast cluster as a mixed keep, not a release-grade pass, until a
    low-CV rerun confirms the split.
- Dense clamp verification is a Rust-internal keep but an external JAX loss
  on all six measured clamp workloads, with the worst gap in half-precision
  tensor clamp.
- Dense bitcast verification is a Rust-internal keep but an external JAX loss
  on all six measured bitcast workloads. The f64/u64 same-width rows are
  near the known memory-bandwidth residual (1.30-1.53x slower), while the BF16
  width-changing row is not release-ready at 4.80-19.63x slower.
- eval_jaxpr (interpreter) is ~12-72x slower than jax.jit on elementwise
  chains; the interpreter-vs-compiler gap is the dominant release blocker for
  jit'd workloads.
  - 8 fj-interpreters PLAN golden tests are RED (digest drift from earlier
    untested storage/serialization commits; plan==generic parity intact). Suite
    is RED until verified golden refresh; behavior correct.
- Reverts: bf16/f16 broadcast fusion (frankenjax-bjqfr) REVERTED — measured 1.02x
  (~0 gain); bf16 tensor fusion is bandwidth-bound and the per-lane f64
  decode/encode cancels the savings. i64 broadcast fusion (rl9ha, 3.21x) and
  f64/f32 fusion KEPT. (Prior dense clusters: no reverts.)
- Next measured gate: Complex64/F32 constructor must target packed Complex64
  output construction or a fused real-to-complex path, not another retry of the
  boxed-literal-elision lever.
- Next clamp gate: BF16/F16 clamp must remove per-lane `clamp_literal`
  widen/round overhead with a raw-bit proof harness; F32/F64 clamp must target
  SIMD/parallel throughput or output allocation/fusion rather than another
  boxed-literal-elision retry.
- Next bitcast gate: same-width f64/u64 should share the elementwise/store
  throughput plan; half-width F32/BF16 needs a raw packed chunk builder that
  avoids the current shape-changing construction overhead. Do not retry the
  boxed-literal-elision bitcast family without fresh profiler evidence.
