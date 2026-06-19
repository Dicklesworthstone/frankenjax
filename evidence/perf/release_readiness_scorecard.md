# FrankenJAX Perf Release Readiness Scorecard

Updated: 2026-06-19

Scope: verify recent code-first `fj-lax`/`fj-core` perf backlog against original JAX on
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

Additional current clamp and bitcast gauntlet environment:

- Agent: cod-a / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-a`
- Rust bench command: `cargo bench -p fj-lax --bench clamp_gauntlet`; current
  bitcast rows use `cargo bench -p fj-lax --bench lax_baseline` with the
  `eval/bitcast_*` filter.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 50 runs x 100
  inner loops per clamp workload; current `.96/.98` bitcast rows use 50 runs x
  500 inner loops
- `frankenjax-mcqr.108` half clamp rows additionally ran RCH same-worker
  before/after Criterion on `ovh-a` with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`; the
  scorecard's Rust/JAX ratios use the local same-host Criterion pass
  `frankenjax-mcqr-108-local-after`.
- `frankenjax-mcqr.109` one-pass half clamp was a rejected candidate, measured
  only as RCH same-worker before/after on `ovh-a`; it changed no shipped
  scorecard rows.

Additional current scalar broadcast environment:

- Agent: cod-b / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-b`
- Rust bench command: `cargo bench -p fj-lax --bench lax_baseline` with the
  `eval/broadcast_scalar` filter, local same-host execution.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 60 runs x 1000
  inner loops per scalar broadcast workload

Additional current fj-core stack/repeat/slice/extraction environment:

- Agent: cod-b / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-b`
- Rust bench command: `cargo bench -p fj-core --bench core_baseline` with the
  `core/tensor_stack_axis0`, `core/scalar_stack_axis0`, and
  `core/tensor_repeat_axis0` filters, plus the `core/scalar_repeat_axis0`
  scalar-repeat control filter and `core/tensor_slice_axis0` slice filter,
  plus the `core/tensor_to_i64_vec` extraction filter and
  `core/tensor_value_new` constructor/extraction filter, local same-host
  execution.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 80 runs x 200
  inner loops for `jax.jit(jnp.stack)` over 64 x 1024 F64 arrays and
  `jax.jit(jnp.repeat(v[None, :], 64, axis=0))` over a 1024-element F64 vector;
  200 runs x 1000 inner loops for `jax.jit(jnp.stack)` over 64 scalar F64
  inputs; 100 runs x 1000 inner loops for `jax.jit(jnp.repeat(x[None], 64,
  axis=0))` over one scalar F64 input; 100 runs x 1000 inner loops for
  `jax.jit(lambda x: x[31, :])` and `jax.jit(lambda x: x[31, :] + 0.0)` over a
  64 x 1024 F64 matrix; 100 runs x 1000 inner loops for
  `jax.jit(lambda x: x)(x).block_until_ready()` and NumPy host-copy extraction
  over a 4096-element I64 vector; 100 runs x 1000 inner loops for
  `jnp.asarray(values, dtype=jnp.float64).block_until_ready()` and NumPy
  host-copy extraction over a 1000-element F64 vector.

Additional cod-a repeat validation environment:

- Agent: cod-a / WildForge
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-a`
- Rust bench command: `cargo bench -p fj-core --bench core_baseline` with the
  `core/tensor_repeat_axis0` filter, local same-host execution.
- JAX oracle: `benchmarks/jax_comparison/.venv/bin/python`
- JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 50 runs x 500
  inner loops for `jax.jit(jnp.tile(x[None, :], (64, 1)) + 0.0)` over a
  1024-element F64 vector.

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
| frankenjax-cod-b-dense-tensor-stack-axis0-rw4k4 | `stack_axis0_f64_64x1k` | 3.3963 us mean | 41.0467 us mean | 0.083 | Rust 12.09x faster (1.04x vs literal control) |
| frankenjax-cod-b-dense-scalar-stack-axis0-tobpl | `scalar_stack_axis0_f64_64` | 137.53 ns mean | 25.1019 us mean | 0.0055 | Rust 182.21x faster (host-dispatch dominated) |
| frankenjax-cod-b-dense-scalar-repeat-axis0-zb2b7 | `scalar_repeat_axis0_f64_64` | 66.809 ns mean | 4.6720 us mean | 0.0143 | Rust 69.93x faster (2.71x vs materializing control; JAX CV 9.46%) |
| frankenjax-cod-b-dense-tensor-repeat-axis0-jk3ed | `repeat_axis0_f64_1k_x64` | 3.2461 us mean | 11.7639 us mean | 0.276 | Rust 3.62x faster (28.36x vs materializing control) |
| frankenjax-cod-b-dense-tensor-repeat-axis0-jk3ed | `repeat_axis0_f64_1k_x64` cod-a tile validation | 2.887 us slope | 16.606 us mean | 0.174 | Rust 5.75x faster; JAX CV 29.61%, directional only |
| frankenjax-cod-b-dense-slice-axis0-4bnj5 | `slice_axis0_f64_64x1k_row31` | 238.18 ns lazy / 513.73 ns +extract | 5.1409 us bare / 5.0677 us +0 mean | 0.0463 / 0.101 | Rust 21.58x faster lazy, 9.86x faster +extract (5.98x vs materializing control; JAX CV 7-11%) |
| frankenjax-cod-b-dense-to-i64-vec-7xbu9 | `to_i64_vec_i64_4k` | 552.47 ns mean | 6.5825 us identity / 9.0209 us NumPy copy mean | 0.0839 / 0.0612 | Rust 11.92x faster vs identity lower-bound, 16.33x faster vs host copy (12.05x vs literal fallback; JAX CV 6.6-19.5%) |
| frankenjax-mcqr.97 | `tensor_value_new_f64_1k` | 1.4152 us construct / 1.4131 us +extract | 48.5634 us ready / 55.9286 us host copy mean | 0.0291 / 0.0253 | External Rust win, internal mixed: 3.27x slower construction-only vs forced literal, 1.68x faster +extract |
| frankenjax-mcqr.105 | `f32_mixed_scalar_tensor_1m` | 159.383 us mean | 115.540 us mean | 1.379 | Rust 1.38x slower |
| frankenjax-mcqr.105 | `f64_mixed_scalar_tensor_1m` | 996.940 us mean | 213.651 us mean | 4.666 | Rust 4.67x slower |
| frankenjax-mcqr.108 | `bf16_mixed_scalar_tensor_1m` | 3.616 ms mean | 122.705 us mean | 29.466 | Rust 29.47x slower (8.60x RCH same-worker speedup; 12.47x vs boxed ref) |
| frankenjax-mcqr.108 | `f16_mixed_scalar_tensor_1m` | 3.521 ms mean | 319.088 us mean | 11.034 | Rust 11.03x slower (8.41x RCH same-worker speedup; 10.64x vs boxed ref) |
| frankenjax-mcqr.108 | `bf16_tensor_tensor_tensor_1m` | 2.993 ms mean | 148.870 us mean | 20.102 | Rust 20.10x slower (7.64x RCH same-worker speedup; 10.73x vs boxed ref) |
| frankenjax-mcqr.108 | `f16_tensor_tensor_tensor_1m` | 3.653 ms mean | 196.938 us mean | 18.549 | Rust 18.55x slower (7.37x RCH same-worker speedup; 8.81x vs boxed ref) |
| frankenjax-mcqr.104 | `i64_mixed_scalar_lo_tensor_hi_1m` | 689.991 us mean | 200.558 us mean | 3.440 | Rust 3.44x slower (34.28x vs boxed ref) |
| frankenjax-mcqr.104 | `i64_mixed_tensor_lo_scalar_hi_1m` | 611.377 us mean | 197.536 us mean | 3.095 | Rust 3.10x slower (42.12x vs boxed ref) |
| frankenjax-mcqr.103 | `i64_tensor_tensor_tensor_1m` | 1.046 ms mean | 287.593 us mean | 3.636 | Rust 3.64x slower (23.59x vs boxed ref) |
| frankenjax-mcqr.101 | `bitcast_f32_i32_1m` | 430.783 us mean | 133.290 us mean | 3.232 | Rust 3.23x slower (45.35x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_i32_f32_1m` | 642.693 us mean | 93.945 us mean | 6.841 | Rust 6.84x slower (35.28x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_f64_u64_1m` | 270.723 us mean | 176.611 us mean | 1.533 | Rust 1.53x slower (91.41x vs literal ref) |
| frankenjax-mcqr.101 | `bitcast_u64_f64_1m` | 228.320 us mean | 175.404 us mean | 1.302 | Rust 1.30x slower (89.89x vs literal ref) |
| frankenjax-mcqr.100 | `bitcast_f32_bf16_1m` | 2.710 ms mean | 138.058 us mean | 19.630 | Rust 19.63x slower (21.29x vs literal ref) |
| frankenjax-mcqr.100 | `bitcast_bf16_f32_1m` | 669.626 us mean | 139.426 us mean | 4.803 | Rust 4.80x slower (47.11x vs literal ref) |
| frankenjax-mcqr.96 | `bitcast_f32_u32_1m` | 97.423 us mean | 113.430 us mean | 0.859 | Rust 1.16x faster, noisy CV (201.42x vs literal ref) |
| frankenjax-mcqr.96 | `bitcast_f64_i64_1m` | 270.586 us mean | 183.715 us mean | 1.473 | Rust 1.47x slower (82.27x vs literal ref) |
| frankenjax-mcqr.98 | `bitcast_f64_u32_1m` | 1.272 ms mean | 186.831 us mean | 6.809 | Rust 6.81x slower (47.76x vs literal ref) |
| frankenjax-mcqr.98 | `bitcast_u32_f64_1m` | 919.708 us mean | 189.478 us mean | 4.854 | Rust 4.85x slower (43.99x vs literal ref) |
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
- JAX domination score (same-host corrected/measured estimate): ~43/100 — scalar
  stack_axis0 (0.0055x), scalar repeat_axis0 (0.0143x), tensor stack_axis0
  (0.083x), tensor repeat_axis0 (0.276x), dense to_i64_vec host extraction
  (0.061x-0.084x), TensorValue::new host construction/extraction
  (0.025x-0.029x external but mixed internally), slice (corrected ~0.72x), and
  integer_pow2 f32 (corrected ~0.97x) beat or tie JAX same-host, with
  broadcast/complex_ctor within ~10%. The 10 bitcast rows add internal wins,
  one noisy external JAX win, and nine external JAX losses.
- The fj-core `stack_axis0` tensor concat-storage row is a strong external
  Rust/JAX win (12.09x faster), but the actual lever is only a narrow 1.04x
  internal improvement over the literal-backed control. KEEP it for realistic
  vmap/loop-stack construction; do not use it as evidence that the whole
  dense-storage backlog is release-ready.
- The fj-core scalar `stack_axis0` row is an even larger external win (182.21x
  faster on 64 scalar F64 inputs), but it is a host-dispatch dominated
  micro-workload for JAX and has Rust/JAX CVs above 5%. KEEP the scalar dense
  constructor path; treat the exact multiplier as directional rather than a
  broad release-readiness proof for non-scalar stack workloads.
- The fj-core scalar `repeat_axis0` row is a measured keep: 66.809 ns versus a
  181.12 ns materializing Rust control (2.71x faster) and 4.6720 us JAX scalar
  repeat (Rust/JAX 0.0143x, 69.93x faster). Like scalar stack, the external row
  is host-dispatch dominated and JAX CV is 9.46%, so keep the construction path
  but treat the exact multiplier as directional.
- The fj-core `repeat_axis0` tensor concat-storage row is a stronger internal
  proof than stack: optimized repeat is 28.36x faster than the materializing
  control and 3.62x faster than JAX on the 1k x64 F64 repeat. The JAX row has
  CV 10.47%, so treat the external ratio as directional rather than
  certification-grade; the internal revert/control gap is large enough to KEEP.
  A cod-a tile-equivalent validation rerun also wins externally (Rust/JAX
  0.174x) but has JAX CV 29.61%, so it is corroborating evidence only.
- The fj-core `to_i64_vec` extraction row is a measured keep: dense extraction
  is 552.47 ns versus a 6.6566 us literal fallback (12.05x faster internally)
  and beats JAX host extraction directionally by 11.92x-16.33x. Because the JAX
  identity and NumPy-copy rows have CV above 5%, use the external ratio as
  directional and the dense/literal Rust split as the hard keep criterion.
- The fj-core `TensorValue::new` row is mixed: generic densification is a
  construction-only regression versus forced literal construction (1.4152 us vs
  432.80 ns, 3.27x slower), but it wins once the consumer extracts typed F64
  values (1.4131 us vs 2.3787 us, 1.68x faster) and beats JAX construction/host
  copy directionally by 34-40x. Keep it as a storage-enabling path, not as a
  construction-only optimization.
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
  on all nine measured clamp workloads. The I64 rows lose by 3.10-3.64x versus
  JAX despite 23.59-42.12x internal wins over the boxed Rust reference; the
  worst gap remains half-precision tensor clamp. The follow-up one-pass
  generic half-bound helper was rejected after same-worker regressions of
  1.18x-1.47x, so temp-vector removal alone is not a sufficient next lever.
- Dense bitcast verification is a Rust-internal keep with mixed external JAX
  evidence across 10 measured bitcast workloads. The F32->U32 row is a noisy
  same-host Rust win at Rust/JAX 0.859, but F64->I64 loses 1.47x and the
  width-changing F64->U32/U32->F64 rows lose 4.85-6.81x. The f64/u64 same-width
  rows remain near the known memory-bandwidth residual (1.30-1.53x slower),
  while BF16 width-changing remains not release-ready at 4.80-19.63x slower.
- eval_jaxpr (interpreter) is ~12-72x slower than jax.jit on elementwise
  chains; the interpreter-vs-compiler gap is the dominant release blocker for
  jit'd workloads.
  - 8 fj-interpreters PLAN golden tests are RED (digest drift from earlier
    untested storage/serialization commits; plan==generic parity intact). Suite
    is RED until verified golden refresh; behavior correct.
- Reverts: bf16/f16 broadcast fusion (frankenjax-bjqfr) REVERTED — measured 1.02x
  (~0 gain); bf16 tensor fusion is bandwidth-bound and the per-lane f64
  decode/encode cancels the savings. frankenjax-mcqr.109 one-pass half clamp
  REVERTED — measured 1.18x-1.47x slower than the two-pass raw-bit helper. i64
  broadcast fusion (rl9ha, 3.21x) and f64/f32 fusion KEPT.
- Next measured gate: Complex64/F32 constructor must target packed Complex64
  output construction or a fused real-to-complex path, not another retry of the
  boxed-literal-elision lever.
- Next clamp gate: BF16/F16 clamp must remove per-lane `clamp_literal`
  widen/round overhead with a raw-bit proof harness, but not by retrying generic
  one-pass temp-vector removal. F32/F64 and I64 clamp must target
  SIMD/parallel throughput, output allocation/fusion, or compiled-buffer reuse
  rather than another boxed-literal-elision retry.
- Next bitcast gate: same-width f32/u32 has only a noisy external win, and
  f64/i64/u64 stays near the memory/store gap; both should share the
  elementwise/store throughput plan. Width-changing F64/U32, U32/F64, and
  half-width F32/BF16 need raw packed chunk builders plus output-layout work
  that avoid the current shape-changing construction overhead. Do not retry the
  boxed-literal-elision bitcast family without fresh profiler evidence.
