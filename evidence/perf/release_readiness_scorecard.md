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
  `core/tensor_value_new` constructor/extraction filter, plus the
  `core/literal_buffer_to_vec` materialization filter, plus the
  `core/literal_buffer_eq` equality filter, local same-host execution.
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
  host-copy extraction over a 1000-element F64 vector; 100 runs x 1000 inner
  loops for `jax.jit(lambda x: x)(x).block_until_ready()` and NumPy host-copy
  extraction over a 65536-element F64 vector; 100 runs x 1000 inner loops for
  `jax.jit(lambda x, y: jnp.array_equal(x, y))` equal and tail-mismatch checks
  over 65536-element F64 vectors, measured as both device-ready lower bound and
  host-bool scalar transfer.

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
| frankenjax-mcqr.99 | `literal_buffer_to_vec_f64_64k` | 26.644 us dense / 33.306 us literal | 19.5312 us identity / 32.9368 us host copy mean | 1.364 / 0.809 | Rust 1.25x faster internally; 1.36x slower than JAX identity lower-bound, 1.24x faster than JAX host copy; directional JAX CV 8.7-9.1% |
| frankenjax-mcqr.102 | `literal_buffer_eq_f64_64k` | 27.484 us equal / 27.570 us mismatch / 53.559 us literal equal | 58.2421 us equal ready / 66.6275 us equal host bool / 56.2721 us mismatch ready / 68.9387 us mismatch host bool | 0.472 / 0.413 / 0.490 / 0.400 | Rust 1.95x faster internally; 2.04-2.50x faster than noisy JAX equality comparators |
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
| frankenjax-k8z8g | `einsum2_general_bqhd_bkhd_bhqk_f64` (permute-copy + GEMM) | 1.3123 ms mean | 317.774 us ready / 278.461 us host-copy mean | 4.130 / 4.713 | Rust loses externally; KEEP internally (489.99x vs old odometer, digest match) |
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
- The fj-core `LiteralBuffer::to_vec` dense row is a modest measured keep:
  direct dense F64 materialization is 26.644 us versus a 33.306 us literal
  fallback (1.25x faster internally). It is not a clean external domination row:
  Rust is 1.36x slower than the JAX identity-ready lower bound, but 1.24x faster
  than the JAX NumPy host-copy comparator. Because JAX CV is 8.7-9.1% and the
  host-copy row copies raw F64 values while Rust materializes `Literal` enums,
  keep this as storage-path evidence only; deeper work should avoid
  `Vec<Literal>` at consumer boundaries.
- The fj-core `LiteralBuffer` equality row is a stronger storage-path keep:
  dense F64 equality is 27.484 us on equal inputs and 27.570 us on a tail
  mismatch, versus 53.559 us for literal-backed equality (1.95x faster
  internally). It also beats JAX `array_equal` directionally by 2.04-2.50x, but
  the JAX rows have 11.6-18.0% CV, so use the external ratio as directional and
  the dense/literal split as the keep proof.
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
- Einsum general permute-copy (k8z8g) confirms the same pattern at the contraction
  boundary: the old odometer path is gone (489.99x faster, digest-identical), but
  the current Rust path is still 4.13-4.71x slower than JAX on the moderate
  attention-shaped comparator. This is an internal keep and an external loss;
  next work must target fused layout-aware einsum lowering, packed/tiled GEMM,
  AVX/FMA policy, or compiled-buffer reuse rather than another permute-copy retry.
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

## CobaltForge - Large-array threaded cheap binops: JAX WIN (2026-06-19)

- First measured JAX-DOMINATING elementwise result in this campaign. Threading the
  cheap same-shape f64/f32 binops (Add/Sub/Mul/Div/Max/Min) above an 8.39M-element
  gate flips four large-array workloads from JAX losses to wins:
  add_f64 16M/64M and add_f32 16M/64M now run at Rust/JAX 0.51-0.59 (1.69-1.97x
  FASTER than `jax.jit` x64, same host), versus 3.6-4.6x slower on the prior serial
  fresh-alloc path. Internal serial->parallel speedup 6.19-7.69x. Bit-identical
  (lane-independent), guarded by `cheap_binary_parallel_f64_bit_identical_to_serial`.
- Correction logged in the ledger: the benchmark host (Zen3 5975WX) has NO AVX-512,
  so prior "JAX wins via AVX-512 8-wide" attributions for elementwise losses are
  incorrect on this hardware; XLA also runs AVX2 and does NOT thread large CPU
  elementwise. The real gaps were the serial fresh-alloc page-fault cliff and
  single-core DRAM bandwidth.
- Still NOT release-grade and untouched by this change: the L3-resident regime
  (n<=~4M, incl. the 1M elementwise rows) where JAX wins on pure L3 streaming
  bandwidth (~200 vs ~70 GB/s). Threading regresses there and is correctly gated
  off. Closing the L3-resident gap needs compiled-jaxpr arena buffer reuse, not
  threading.

## CobaltForge - Bias-add (scalar-tensor) threading: JAX WIN (frankenjax-aazu6, 2026-06-19)

- Second JAX-dominating elementwise row. Threading the cheap scalar-tensor binops
  (bias-add `x+c`, scaling `x*c`, max/min) above the 8.39M-element gate:
  biasadd_f64 16M = Rust/JAX 0.56 (1.78x faster), 64M = 0.50 (2.00x faster), vs the
  prior serial scalar-broadcast path which LOST to JAX by ~5.2-5.3x (it craters to
  3.6-3.8 GB/s — broadcast overhead stacked on the fresh-alloc page-fault cliff).
  Internal serial->parallel 9.18-10.63x. Bit-identical, guarded by
  `cheap_scalar_tensor_parallel_f64_bit_identical_to_serial`. Closes aazu6.
- Same gating discipline: L3-resident sizes stay serial (no regression). The
  compiled-jaxpr arena buffer-reuse swing remains the only lever for the
  L3-resident regime where JAX still wins on pure cache bandwidth.

## CobaltForge - i64 same-shape threading: JAX WIN (2026-06-19)

- Completes the same-shape elementwise threading family (f64/f32/i64). add_i64 16M/64M
  now Rust/JAX 0.56 (1.78x faster than jax.jit int64), internal 6.47-6.77x, vs prior
  serial ~3.6-3.8x slower. Bit-identical via the exact int_op (wrapping/div-by-zero
  semantics preserved), guarded by same_shape_i64_parallel_bit_identical_to_serial.
  Required a non-breaking `+ Sync` on eval_binary_elementwise's int_op bound.

## CobaltForge - Caching allocator (mimalloc): highest-EV lever, MAINTAINER-GATED (2026-06-19)

- The ~6 GB/s serial fresh-alloc cliff behind every large-elementwise JAX loss is a
  glibc page-fault artifact, not inherent. mimalloc (warm-span reuse) gives 3.7-3.9x
  on serial large add (5.9->21.7 GB/s @16M, 6.0->23.4 @64M) = ~JAX PARITY with no
  threading and no value change, and would help EVERY allocating op + the L3-resident
  regime — broader than the 6 ops I hand-threaded.
- Substitute interaction: under mimalloc, the shipped cheap-binop threading becomes a
  no-op (1.02x @16M) and slightly regresses (0.93x @64M) because vec![0.0;n] re-zeroes
  the warm span serially. Adopting mimalloc should be paired with raising/removing the
  CHEAP_BINARY_PARALLEL_MIN gate.
- Gated because: workspace allocator policy + library #[global_allocator] conflicts
  with the PeakAlloc bench. Filed as a bead. Not committed unilaterally (cf. +fma).

## CobaltForge - Threaded cheap unary (JAX WIN) + mimalloc CORRECTION (2026-06-19)

- Cheap unary (Neg/Abs/Sign/Square/Floor/Ceil/Reciprocal) now threaded above the 8.39M
  gate: neg_f64 16M/64M = Rust/JAX 0.48/0.51 (2.09x/1.97x faster than jax.jit), internal
  8.03/7.98x (4.3->34.3 GB/s). Bit-identical.
- CORRECTION to the round-4 mimalloc note: on the real eval_primitive path (with threading
  shipped), threading BEATS mimalloc for memory-bound ops (parallel page-faulting ~34-38
  GB/s > mimalloc warm-span serial ~20-24). mimalloc and threading are COMPLEMENTARY, not
  substitutes; KEEP the threading gate regardless of allocator choice. mimalloc remains a
  modest complementary win only for large allocating ops that are hard to thread.

## CobaltForge - Threaded broadcast: JAX WIN (2026-06-19)

- BroadcastInDim f64/f32 large-output replicate now threaded (calloc'd output + parallel
  page-faulting): [1024]->[16K/64K,1024] = Rust/JAX 0.27 (3.65-3.76x faster than jax.jit
  broadcast_to), internal 9.1-9.3x (2.4->~22 GB/s). Bit-identical, guarded. Broadcast is
  ubiquitous (bias/feature materialization). Other dtypes keep the serial path (calloc needs
  a concrete element type); easy follow-on.

## CobaltForge - Threaded convert (f64<->f32): JAX WIN (2026-06-19)

- ConvertElementType hot casts threaded (calloc'd output + parallel page-faulting):
  f64->f32 = Rust/JAX 0.47-0.61 (1.64-2.14x faster), f32->f64 = 0.45-0.50 (1.99-2.21x faster);
  internal 6.0-8.5x (5.9/3.4 -> 28-38 GB/s). Bit-identical, guarded. Mixed-precision casts are
  everywhere. Other casts (int/half) are easy follow-ons (same calloc+thread helper).

## CobaltForge - Threaded transpose (rank-2 f64/f32): JAX WIN (2026-06-19)

- rank-2 [1,0] transpose threaded (calloc'd output + column-range cache-blocked sub-transposes):
  f64 4096x4096 / 8192x8192 = Rust/JAX 0.34/0.33 (2.93-3.0x faster than jax.jit), internal
  12.3-13.0x (2.2 -> 27.6-28.9 GB/s). Bit-identical (incl. non-square), guarded. "Tiling
  regresses" was about cache-blocking, not threading — this keeps tiling and adds threads.
  Other dtypes / N-D transpose are easy follow-ons.

## CobaltForge - Threaded gather (f64/f32 embedding lookup): JAX WIN (2026-06-19)

- Contiguous-row gather (embedding lookup) threaded (calloc'd output + parallel row memcpy):
  f32 [16384,1024] nidx 16384/65536 = Rust/JAX 0.57/0.48 (1.74-2.07x faster than jax.jit gather),
  internal 6.5-8.4x (2.2 -> 14-18 GB/s). Bit-identical (incl. OOB fill), guarded. Embedding
  lookup is ubiquitous in NLP. Other dtypes / strided gather are easy follow-ons.

## CobaltForge - Threaded concatenate (axis-0 f64/f32): JAX WIN, was parity (2026-06-19)

- Contiguous axis-0 concat threaded (calloc'd output + parallel chunked copy): ~16M/~64M f64 =
  Rust/JAX 0.11 (9.2-9.4x faster than jax.jit concatenate), internal 8.1x (2.1 -> 16.7-16.9 GB/s).
  Was at parity (both ~2 GB/s page-fault bound); now ~9x domination. Bit-identical (incl. uneven
  sources / chunk-crossing), guarded. Common in KV-cache / batch concat. axis>0 + other dtypes
  are follow-ons.

## CobaltForge - Threaded bf16/f16 broadcast + gather (training dtype): JAX WIN (2026-06-19)

- Extended the threaded broadcast/gather to bf16/f16 (dominant training dtype) via the existing
  generic _into helpers (calloc'd u16 output). bf16 broadcast 4.48-8.38x internal (2.5->11.8-20.9
  GB/s); bf16 gather ~6x (11-17 GB/s). Bit-identical, guarded. Bias broadcast + embedding gather
  are ubiquitous in training. Other dtypes (i64/u32/u64) + bf16 transpose/concat are follow-ons.

## CobaltForge - Threaded scalar-broadcast / full (f64/f32): JAX WIN (2026-06-19)

- jnp.full / scalar-const fill threaded (calloc'd output + parallel constant write): 16M/64M f64 =
  Rust/JAX 0.29/0.28 (3.42-3.53x faster than jax.jit full), internal 7.85-8.42x (2.5 -> 20.3-20.9
  GB/s). Bit-identical (incl. NaN), guarded. Common in init/masks. Other dtypes are follow-ons.

## CobaltForge - Threaded max/min reduce (f64/f32): JAX WIN (2026-06-19)

- Full ReduceMax/ReduceMin threaded (parallel partial SIMD reduce + associative combine): 16M/64M
  f64 = Rust/JAX 0.80/0.74 (1.25-1.35x faster than jax.jit max), internal 1.64-3.06x (21.6 ->
  35.3-59.9 GB/s). Bit-identical (NaN/±0/±inf, f64+f32), guarded. Was a 1.3-2.3x loss. ReduceSum
  stays a loss (10.9 GB/s) but is bit-exact-pinned (non-associative) -> needs XLA-order matching,
  not threading.

## CobaltForge - Threaded bf16/f16 transpose + concat (training dtype): JAX WIN (2026-06-19)

- bf16/f16 rank-2 transpose + axis-0 concat threaded (calloc'd u16 output + parallel copy via the
  generic _into helpers): bf16 transpose 13.6-20.2 GB/s, bf16 concat 13.1-17.0 GB/s (serial ~2-2.5
  GB/s cliff -> ~5-8x). Bit-identical (incl. non-square transpose), guarded. bf16 attention
  transpose + KV-cache concat are ubiquitous in LLM training/inference.

## CobaltForge - i64 broadcast/gather WIN + argmax verified + SATURATION (2026-06-19)

- i64 broadcast (~22 GB/s, ~9x) + i64 gather (~16 GB/s, ~7x) threaded (index/id tensors). argmax
  VERIFIED already 1.36x faster than JAX (no change). With this, all measured large memory-bound
  ops match/beat JAX across f64/f32/i64/bf16/f16 (fill/copy/reduce patterns). Remaining JAX losses
  are only the off-limits float non-associative reductions (sum/prod/cumsum) and the L3-resident
  regime — both need the multi-session compiled-jaxpr work, not threading.

## CobaltForge - Threaded select/where (f64/f32 masking): JAX WIN (2026-06-19)

- select/where dense f64/f32 fast paths threaded (both dense-Bool and bit-packed-BoolWords cond)
  via new reusable threaded_index_fill_into: 16M/64M = Rust/JAX 0.46/0.57 (2.16/1.75x faster than
  jax.jit where), internal 6.17-7.02x (1.9 -> 11.9-12.8 GB/s). Bit-identical (NaN/±0/±inf, both
  cond backings), guarded. Masking is ubiquitous (attention/dropout/clip). i64/bf16 arms + select_n
  are mechanical follow-ons.
