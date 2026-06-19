# Negative Evidence Ledger

This ledger records code-first performance attempts and retry predicates so dead
ends are not rediscovered without new evidence.

## frankenjax-mcqr.97 - TensorValue::new Dense Literal Storage

- Date: 2026-06-18
- Agent: cod-b / TopazOrchid
- Lever: densify homogeneous F64/F32/Bool/BF16/F16/Complex literal vectors in
  `TensorValue::new` after element-count validation.
- Status: batch-test pending.
- Benchmark guard: `core/tensor_value_new_1k_f64_generic_dense`,
  `core/tensor_value_new_1k_f64_forced_literal`,
  `core/tensor_value_new_then_to_f64_vec_1k`,
  `core/tensor_value_new_forced_literal_then_to_f64_vec_1k`.
- Conformance guard: matching literal families materialize bit-identically;
  mismatched literal/dtype tensors remain literal-backed.
- Retry predicate: do not retry FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat/storage-copy families without fresh same-worker benchmark evidence.
  Do not repeat the already committed stack/repeat/slice/to_i64 storage levers
  unless a profile identifies a distinct call path.

## frankenjax-mcqr.99 - Direct Dense LiteralBuffer to_vec Materialization

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::to_vec()` materialize directly from dense storage
  variants instead of forcing `as_slice()` to build/cache a full literal vector
  and then clone it.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_to_vec_dense_f64_64k`,
  `core/literal_buffer_to_vec_literal_f64_64k`.
- Conformance guard: direct `to_vec()` matches `as_slice().to_vec()` bit-for-bit
  across dense F64/F32/I64/U32/U64/Bool/BF16/F16/Complex storage plus lazy
  concat/repeated-patches paths.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64
  or `TensorValue::new` dense-storage families under this bead. Do not revisit
  FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager concat without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-q59j4 - Direct Dense LiteralBuffer COW Mutation

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::make_mut()` materialize dense/lazy storage through
  the direct storage-aware `to_vec()` path instead of forcing `as_slice()` to
  build/cache an intermediate full literal vector before mutation.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_index_mut_dense_f64_64k`,
  `core/literal_buffer_index_mut_literal_f64_64k`.
- Conformance guard: dense and lazy COW mutation preserves the original cloned
  sequence and mutates to the same materialized literal sequence as a literal
  buffer for F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, plus a dense-sort materialization path.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, or `LiteralBuffer::to_vec` dense-storage families under
  this bead. Reopen the dense COW mutation family only with focused criterion
  evidence showing mutation/materialization cost remains a top-five fj-core
  bottleneck. Do not revisit FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.100 - Dense Half-Width Bitcast Chunks

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->BF16/F16 and BF16/F16->F32
  through packed f32/u16 slices instead of per-`Literal` byte conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  both measured BF16 reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_bf16_dense_1m`,
  `eval/bitcast_f32_bf16_literal_ref_1m`,
  `eval/bitcast_bf16_f32_dense_1m`,
  `eval/bitcast_bf16_f32_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench lax_baseline`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench lax_baseline -- 'eval/bitcast_(f32_i32|i32_f32|f64_u64|u64_f64|f32_bf16|bf16_f32)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline frankenjax-mcqr-100-101-bitcast`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/bitcast_gauntlet.py --runs 30 --warmup 5 --inner-loops 50 --output /tmp/frankenjax_mcqr_100_101_bitcast_jax_raw.json`.
  - `bitcast_f32_bf16_1m`: Rust dense mean 2.710 ms vs Rust literal
    reference 57.706 ms (21.29x faster internally); JAX mean 138.058 us;
    Rust/JAX 19.63x slower.
  - `bitcast_bf16_f32_1m`: Rust dense mean 669.626 us vs Rust literal
    reference 31.549 ms (47.11x faster internally); JAX mean 139.426 us;
    Rust/JAX 4.80x slower.
  - Decision: keep, not revert. Both rows are large internal wins over the
    original per-Literal path, but they are external JAX losses. The next BF16
    bitcast attempt must target raw packed output construction throughput, not
    another boxed-literal-elision pass.
- Conformance guard: dense and literal-backed half-width bitcasts produce the
  same output shapes, dtypes, raw half chunks, and round-trip f32 bit patterns
  across NaN, infinities, signed zero, normals, and a custom NaN payload.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry this half-width bitcast family unless focused
  criterion evidence shows the dense rows are still slower than the
  literal-backed reference or the original path remains a top-five fj-lax
  bottleneck. Do not merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or peer-owned fj-core dense-storage lanes without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.101 - Dense Signed/Unsigned Same-Width Bitcast Pairs

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->I32, I32->F32, F64->U64, and
  U64->F64 through packed typed slices instead of per-`Literal` byte
  conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  all four measured same-width reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_i32_dense_1m`,
  `eval/bitcast_f32_i32_literal_ref_1m`,
  `eval/bitcast_i32_f32_dense_1m`,
  `eval/bitcast_i32_f32_literal_ref_1m`,
  `eval/bitcast_f64_u64_dense_1m`,
  `eval/bitcast_f64_u64_literal_ref_1m`,
  `eval/bitcast_u64_f64_dense_1m`,
  `eval/bitcast_u64_f64_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust/JAX commands match the `frankenjax-mcqr.100` bitcast gauntlet entry.
  - `bitcast_f32_i32_1m`: Rust dense mean 430.783 us vs Rust literal
    reference 19.537 ms (45.35x faster internally); JAX mean 133.290 us;
    Rust/JAX 3.23x slower.
  - `bitcast_i32_f32_1m`: Rust dense mean 642.693 us vs Rust literal
    reference 22.677 ms (35.28x faster internally); JAX mean 93.945 us;
    Rust/JAX 6.84x slower.
  - `bitcast_f64_u64_1m`: Rust dense mean 270.723 us vs Rust literal
    reference 24.747 ms (91.41x faster internally); JAX mean 176.611 us;
    Rust/JAX 1.53x slower.
  - `bitcast_u64_f64_1m`: Rust dense mean 228.320 us vs Rust literal
    reference 20.525 ms (89.89x faster internally); JAX mean 175.404 us;
    Rust/JAX 1.30x slower.
  - Decision: keep, not revert. Every row is a large internal win and the
    f64/u64 rows are near the existing memory-bandwidth residual gap, but all
    rows are still external JAX losses.
- Conformance guard: dense and literal-backed signed/unsigned same-width
  bitcasts produce the same shapes, dtypes, exact integer bit lanes, packed
  storage, and round-trip float bit patterns across NaN, infinities, signed
  zero, normals, and custom NaN payloads.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry the already committed F32<->U32, F64<->I64,
  F64<->U32, U32<->F64, F32<->BF16/F16, BF16/F16->F32, or these signed/
  unsigned same-width bitcast pairs without focused criterion evidence showing
  the dense rows are still slower than the literal-backed reference or the
  original per-`Literal` path remains a top-five fj-lax bottleneck. Do not
  revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  peer-owned fj-core dense-storage lanes without fresh same-worker benchmark
  evidence and ownership check.

## frankenjax-co009 - Stream Dense LiteralBuffer Serialization

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `Serialize for LiteralBuffer` stream storage-direct elements
  through `SerializeSeq` instead of forcing `as_slice()` to cache a full
  materialized literal vector for dense packed buffers.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_serialize_dense_f64_64k`,
  `core/literal_buffer_serialize_literal_f64_64k`.
- Conformance guard: streamed JSON matches materialized `Vec<Literal>` JSON
  across F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, and mixed dense/literal concat paths.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation, or this
  serialization streaming family without fresh focused criterion evidence
  showing it remains a top-five fj-core bottleneck. Do not revisit FMA/SIMD exp,
  GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or eager concat without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.102 - Storage-Direct LiteralBuffer Equality

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer` equality compare storage-direct typed ranges and
  recursively compare concat slices instead of forcing `as_slice()` on both
  operands and caching full materialized literal vectors for dense packed
  buffers.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_eq_dense_f64_64k_equal`,
  `core/literal_buffer_eq_dense_f64_64k_mismatch`,
  `core/literal_buffer_eq_literal_f64_64k_equal`.
- Conformance guard: storage-direct equality matches materialized
  `Vec<Literal>` equality across F64/F64OnePlusX/F32/I64/U32/U64/Bool/
  BoolWords/Half/Complex, repeated-patches, concat, mixed dense/literal concat,
  and the `LiteralBuffer`/`Vec<Literal>` cross-`PartialEq` impls.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation,
  serialization streaming, or this equality family without fresh focused
  criterion evidence showing comparison remains a top-five fj-core bottleneck.
  Do not revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  eager concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-alc0j - Dense Scalar Broadcast for U32/U64/Complex

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `BroadcastInDim` fills for U32, U64, Complex64, and
  Complex128 through `new_u32_values`, `new_u64_values`, and
  `new_complex_values` instead of allocating a `Vec<Literal>` fill.
- Status: measured mixed/noisy against warmed JAX CPU. Keep the committed dense
  fill path, but record negative head-to-head evidence for Complex128.
- Benchmark guard: `eval/broadcast_scalar_u32_1024x1024`,
  `eval/broadcast_scalar_u64_1024x1024`,
  `eval/broadcast_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/broadcast_scalar --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline scalar
    broadcast harness with `jax_enable_x64=true`, JAX/JAXLIB 0.10.1, CPU
    backend, 60 runs x 1000 warmed inner loops.
  - `broadcast_scalar_u32_1024x1024`: Rust Criterion mean 51.307 us, slope
    49.804 us, median 50.460 us, Rust CV 10.56%; JAX mean 79.979 us, p50
    84.805 us, JAX CV 26.62%; Rust/JAX mean ratio 0.642, Rust 1.56x faster.
    Classification: noisy win.
  - `broadcast_scalar_u64_1024x1024`: Rust Criterion mean 104.911 us, slope
    102.906 us, median 102.130 us, Rust CV 18.20%; JAX mean 124.569 us, p50
    130.398 us, JAX CV 31.74%; Rust/JAX mean ratio 0.842, Rust 1.19x faster.
    Classification: noisy win/near-tie.
  - `broadcast_scalar_complex128_1024x1024`: Rust Criterion mean 283.150 us,
    slope 263.538 us, median 276.539 us, Rust CV 18.07%; JAX mean 245.381 us,
    p50 259.656 us, JAX CV 24.66%; Rust/JAX mean ratio 1.154, Rust 1.15x
    slower. Classification: noisy loss.
  - Decision: keep, not revert. The U32 and U64 scalar-fill rows beat JAX by
    mean ratio, while Complex128 loses; all three rows are above the 5% CV
    certification bar, so this is real negative/positive evidence but not a
    release-grade perf certificate. There is no same-run evidence that reverting
    the dense fill path improves the Complex128 row, and reverting would discard
    the measured U32/U64 wins. The next Complex128 attempt must target complex
    packed-fill/store throughput, not another boxed-literal-elision pass.
- Conformance guard: scalar fills materialize to the same repeated literals and
  expose dense typed storage via `as_u32_slice`, `as_u64_slice`, or
  `as_complex_slice`.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax dense_broadcast_in_dim_matches_literal_path_and_stays_dense --lib`
  passed 1 test, 0 failed.
- Retry predicate: do not retry scalar `BroadcastInDim` dense-fill storage for
  these dtypes unless focused criterion evidence shows this path remains a
  top-five `fj-lax` bottleneck with both Rust and JAX CV below 5%, or the dense
  constructor representation changes. For Complex128 specifically, do not
  repeat the same `Vec<(f64, f64)>` dense fill lever; the next attempt must
  attack packed complex output construction/store throughput directly. Do not
  merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  broader scalar-broadcast arithmetic work without fresh benchmark evidence and
  ownership check.

## frankenjax-dxqfj - Lazy SplitMulti Section Buffers

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: build each `eval_split_multi` output from `LiteralBuffer::from_concat_slices`
  over the original tensor backing instead of copying each section through
  `Vec<Literal>` and re-densifying via `TensorValue::new`.
- Status: measured keep against warmed JAX CPU. No revert.
- Benchmark guard: `eval/split_multi_1024x1024_f32_axis1`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/split_multi_1024x1024_f32_axis1 --sample-size 60 --warm-up-time 1 --measurement-time 5`.
  - Rust Criterion: mean 96.651 us, median 95.038 us, slope 98.875 us
    (95% CI 97.211-100.534 us), stddev 6.616 us, CV 6.85%.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline split
    harness, JAX/JAXLIB 0.10.1, `jax_enable_x64=true`, CPU backend, 60 runs x
    1000 warmed inner loops.
  - `jnp.split(x, [256], axis=1)` bare output: JAX mean 149.082 us, p50
    148.877 us, CV 12.08%; Rust/JAX mean ratio 0.648, Rust 1.54x faster.
  - `jnp.split(x, [256], axis=1)` with `+0.0` materialization on both outputs:
    JAX mean 136.983 us, p50 139.649 us, CV 10.43%; Rust/JAX mean ratio 0.706,
    Rust 1.42x faster.
  - Decision: keep. The JAX side remains noisy, but both JAX modes are slower
    than the Rust lazy-section split by a margin larger than the Rust CI band.
    This is a measured split-construction win, not a claim about arbitrary
    downstream consumers after full materialization.
- Conformance guard: uneven multi-output split materializes the same literals for
  each section and exposes dense f32 storage through `as_f32_slice`;
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax split_multi --lib`
  passed 2 tests, 0 failed.
- Retry predicate: retry only if a profiler attributes a clearly-above-noise
  share to `eval_split_multi` section construction on a wider multi-output split
  workload after Criterion and JAX CV are both below 5%, or if
  `LiteralBuffer::Concat` dense-lane behavior changes. Do not merge with
  FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or broader
  reshape/slice/gather work without fresh benchmark evidence and ownership
  check.

## frankenjax-19wst - Dense Scalar Tile Fills

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `Tile` uniform fills through dense typed `TensorValue`
  constructors instead of filling a `Vec<Literal>` with the recursive boxed
  tiler and then re-densifying.
- Status: measured keep against warmed JAX CPU for both guard workloads.
- Benchmark guard: `eval/tile_scalar_f32_1024x1024`,
  `eval/tile_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/tile_scalar --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `tile_scalar_f32_1024x1024`: Rust criterion median 51.435 us vs JAX
    batched median 317.753 us; ratio Rust/JAX 0.162, Rust 6.18x faster.
  - `tile_scalar_complex128_1024x1024`: Rust criterion median 412.679 us vs
    JAX batched median 579.030 us; ratio Rust/JAX 0.713, Rust 1.40x faster.
  - Decision: keep. No revert; both measured workloads beat the original JAX
    oracle under warmed CPU execution. Complex128 margin is modest enough that
    future retries must preserve this exact head-to-head guard.
- Conformance guard: scalar tile materializes to the same repeated literals and
  exposes dense typed storage for f32, u64, and complex128.
- Retry predicate: do not retry scalar `Tile` dense-fill storage unless focused
  criterion evidence shows this path remains a top-five `fj-lax` bottleneck or
  the scalar tile representation changes. Do not merge it with scalar
  `BroadcastInDim`, dense tensor tile, FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or broader shape/data movement work without fresh
  benchmark evidence and ownership check.

## frankenjax-1z7k9 - Dense Tensor-Scalar Complex Constructor

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route matching F64/F32 `eval_complex` tensor-scalar and scalar-tensor
  constructors through dense `new_complex_values` instead of rebuilding every
  element as a boxed `Literal`.
- Status: measured mixed; keep the committed dense path, but record a negative
  JAX head-to-head result for the F32 tensor-scalar workload.
- Benchmark guard: `eval/complex_f32_tensor_scalar_1m`,
  `eval/complex_f64_tensor_scalar_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/complex_f --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `complex_f32_tensor_scalar_1m`: Rust criterion median 1.379 ms vs JAX
    batched median 1.272 ms; ratio Rust/JAX 1.084, Rust 1.08x slower. This is
    negative evidence for "dense constructor alone dominates original JAX" on
    Complex64/F32.
  - `complex_f64_tensor_scalar_1m`: Rust criterion median 0.914 ms vs JAX
    batched median 3.730 ms; ratio Rust/JAX 0.245, Rust 4.08x faster.
  - Decision: keep, not revert. The cluster has one clear JAX win and one clear
    JAX loss; there is no same-run evidence that reverting the dense constructor
    improves the F32 result, and reverting would discard the measured F64 win.
    Route the F32 loss to a deeper primitive-level follow-up instead of retrying
    the same boxed-literal-elision lever.
- Conformance guard: dense tensor-scalar and scalar-tensor constructor outputs
  materialize identically to explicit literal-backed references and expose dense
  complex storage.
- Retry predicate: do not retry tensor-scalar `eval_complex` constructor storage
  unless focused criterion evidence shows this path remains a top-five
  `fj-lax` bottleneck or mixed-dtype promotion becomes the measured hotspot.
  For Complex64/F32 specifically, do not repeat the same
  `Vec<(f64, f64)> -> new_complex_values` dense constructor lever; the next
  attempt must attack representation/construction overhead directly, such as a
  Complex64-native packed builder or a JAX-comparable fused real-to-complex path.
  Do not merge with same-shape complex constructor, FFT extraction, complex
  binary tensor-scalar ops, or broader complex arithmetic work without fresh
  benchmark evidence and ownership check.

## frankenjax-mcqr.105-.107 - Dense Clamp Mixed/Tensor Bounds

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify the committed dense `clamp` fast paths for mixed scalar/tensor
  F32/F64 bounds, mixed scalar/tensor BF16/F16 bounds, and same-shape BF16/F16
  tensor bounds against both the Rust boxed reference path and original JAX CPU.
- Status: measured keep internally; negative head-to-head result versus JAX for
  all six measured workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_105_107_clamp_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, warmed JAX CPU venv with 50 runs x
  100 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax --bench clamp_gauntlet`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- --save-baseline frankenjax-mcqr-105-107`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_105_107_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend, `jax_enable_x64=true`.

| Workload | Rust dense mean | Rust boxed mean | Dense/boxed | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `f32_mixed_scalar_tensor_1m` | 159.383 us | 27.645 ms | 173.45x faster | 115.540 us | 1.38x slower | Keep internal win; JAX loss |
| `f64_mixed_scalar_tensor_1m` | 996.940 us | 27.487 ms | 27.57x faster | 213.651 us | 4.67x slower | Keep internal win; JAX loss |
| `bf16_mixed_scalar_tensor_1m` | 15.571 ms | 43.758 ms | 2.81x faster | 121.313 us | 128.35x slower | Keep internal win; JAX loss |
| `f16_mixed_scalar_tensor_1m` | 19.859 ms | 41.186 ms | 2.07x faster | 371.729 us | 53.42x slower | Keep internal win; JAX loss |
| `bf16_tensor_tensor_tensor_1m` | 15.652 ms | 35.451 ms | 2.26x faster | 183.707 us | 85.20x slower | Keep internal win; JAX loss |
| `f16_tensor_tensor_tensor_1m` | 20.951 ms | 35.255 ms | 1.68x faster | 229.951 us | 91.11x slower | Keep internal win; JAX loss |

- CV notes: Rust dense CV ranged from 5.75% to 13.73%; JAX CV ranged from
  5.70% to 21.86%. Treat the F32 JAX loss as lower-confidence than the half
  losses, but it remains directionally negative under the same-host run.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo test -p fj-lax clamp --lib`
  passed 25 tests, 0 failed, 6 ignored benchmark tests. The passing set includes
  `f32_f64_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`,
  `half_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`, and
  `half_clamp_same_shape_tensor_bounds_dense_matches_generic`.
- Decision: keep, not revert. Every measured target clears the Rust boxed
  reference keep gate by at least 1.68x, so reverting would discard a real
  internal improvement. Do not claim original-JAX domination for this cluster.
- Retry predicate: do not retry boxed-literal elision for these clamp shapes.
  F32/F64 follow-up must attack SIMD/parallel clamp throughput or output
  allocation/fusion directly. BF16/F16 follow-up must avoid the per-lane
  `clamp_literal` widen/round path, for example with a raw half-bits proof
  harness plus vectorized or table-assisted min/max semantics. Do not merge the
  next attempt with broadcast-shape generalization, scalar-bound relu/relu6, or
  unrelated dense constructor work without fresh focused Criterion evidence and
  an updated JAX head-to-head row.

## frankenjax-e07uw/7g72q/rl9ha/bjqfr - eval_jaxpr Fusion Cluster (Square / Reciprocal / integer_pow[2] / i64+bf16 broadcast)

- Lever: extend the `eval_jaxpr` cheap-op elementwise fusion to (a) `Square`,
  `Reciprocal`, `integer_pow[2]` via builder operand-synthesis (unary == fused
  binary: `Square=Mul(x,x)`, `Reciprocal=Div(1,x)`, `x**2=Mul(x,x)`), and (b) i64
  + bf16/f16 row/col broadcast operands (mirror the existing f64/f32 broadcast).
- Conformance GUARD: GREEN. `cargo test -p fj-interpreters --lib --release` — the
  fusion bit-identity tests (`fusion_*_chain_matches_reference_bit_for_bit`,
  `fusion_f64_{row,col}_broadcast_*`, etc.) all PASS. (8 unrelated golden-digest
  tests for broadcast_in_dim/transpose/reshape/scan/dce/staging PLANS are RED — see
  the separate "Golden digest drift" entry; their `plan==generic` parity asserts
  PASS, only the hardcoded sha256 lines drift; NOT caused by this cluster.)
- Measured evidence (2026-06-19, rch worker, `eval_fusion_speed` same-invocation
  A/B, 1,048,576 elements / 1024x1024, 8-op chains; rch bench noise ~1.0x band):
  - Internal (Rust fused `eval_jaxpr` vs unfused per-op `eval_primitive`):

  | Workload | unfused | fused | speedup | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | F64 arith8 1M | 16.788 ms | 3.320 ms | 5.06x | KEEP |
  | F32 arith8 1M | 7.174 ms | 1.071 ms | 6.70x | KEEP |
  | F32 clamp 1M | 9.081 ms | 1.739 ms | 5.22x | KEEP |
  | F32 row-broadcast | 12.331 ms | 0.783 ms | 15.75x | KEEP |
  | F32 col-broadcast | 11.479 ms | 0.704 ms | 16.31x | KEEP |
  | F64 row-broadcast | 8.703 ms | 2.567 ms | 3.39x | KEEP |
  | F64 col-broadcast | 8.452 ms | 2.282 ms | 3.70x | KEEP |
  | I64 arith8 1M | 5.976 ms | 3.330 ms | 1.79x | KEEP |
  | **I64 row-broadcast (rl9ha)** | 7.546 ms | 2.352 ms | **3.21x** | **KEEP** |
  | BF16 arith8 1M (not mine) | 10.893 ms | 10.599 ms | 1.03x | ~0 gain, flag owner |
  | **BF16 row-broadcast (bjqfr)** | 10.776 ms | 10.604 ms | **1.02x** | **REVERTED** |

  - Head-to-head vs original JAX (jax.jit CPU x64, `fusion_chain.py`, 1M):

  | Workload | JAX jit mean | Rust fused | Rust/JAX | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | arith8 f64 1M | 272.7 us | 3.320 ms | ~12.2x slower | JAX win (interp vs XLA) |
  | square f64 1M | 293.4 us | (rides f64 5.06x) | >10x slower | JAX win |
  | bf16 broadcast 1M | 146.9 us | 10.6 ms | ~72x slower | JAX win |

- Decision:
  - KEEP Square / Reciprocal / integer_pow[2] (bit-identical operand-synthesis;
    they let variance/L2/`x**2`/normalization chains ride the proven f64/f32
    5-6.7x fusion win) and i64 broadcast (3.21x).
  - REVERT bf16/f16 broadcast fusion (bjqfr) — measured 1.02x = ~0 gain. bf16
    tensor fusion is bandwidth-bound; the per-lane f64 decode/encode
    (half_fusion_widen + half_fused_binary) cancels the materialization savings
    (consistent with bf16 same-shape ~1.0x). Reverted in commit after this entry.
- Honest framing: the fusion lever is a real **Rust-internal interpreter** win
  (5-16x vs the pre-fusion per-op path) but does NOT achieve original-JAX
  domination on jit'd elementwise chains — the Rust tree-walking interpreter
  (even fused) is ~12-72x slower than XLA-compiled jax.jit. Fusion narrows the
  per-op interpreter tax; it cannot close the interpreter-vs-compiler gap. The
  vs-JAX win requires the compiled-jaxpr arena executor (bead z6o97 family /
  6dfew), not more fusion-op coverage.
- Retry predicate: do NOT re-add bf16/f16 tensor fusion via the scalar per-lane
  widen/round path. It only pays with a SIMD bf16<->f32 convert kernel (see
  project_bf16_matmul_and_convert_simd). Do NOT chase more fusion-op coverage for
  vs-JAX wins — the gap is interpreter-vs-compiler, addressable only by the
  compiled-jaxpr core.

## Golden digest drift - 8 fj-interpreters PLAN goldens (pre-existing, NOT this session)

- Symptom: `cargo test -p fj-interpreters --lib --release` shows 8 FAILED golden
  tests: dense_f64_broadcast_in_dim_plan / dense_f64_transpose_plan /
  dense_reshape_plan (x2) / eval_top_level_scan_i64_add_emit_fast_path /
  test_dce_all_used_large_chain / test_pe_two_eq_mixed_residual / staging single-unknown.
- Diagnosis: each test asserts `planned_out == generic_out` (PARITY) THEN a
  hardcoded sha256. The panics are all on the sha256 line (hash-vs-hash); the
  parity asserts PASS. So the COMPUTED VALUES are internally consistent
  (plan==generic) and the digests merely drifted vs the recorded constants —
  accumulated from earlier untested "code-first" dense-storage/serialization
  commits (e.g. mcqr.97 TensorValue::new densify, co009 serialization).
- NOT caused by this session's fusion work (none of these tests exercise the
  cheap-op fusion path; the 200 passing tests include all fusion bit-identity tests).
- Action: NOT refreshed here. Refreshing requires per-test value verification vs
  JAX/expected (parity alone does not prove vs-original correctness — a shared
  regression in both paths would pass parity). Flagged for the dense-storage/
  serialization commit owners to refresh with verified values, or for a
  test-capable golden-refresh pass. Conformance is RED on these 8 until then.

## frankenjax-f62hx (+ thnjs/idunl/rbkn9/02i98/ecffn) - Contiguous-Block Memcpy Transpose/Slice/Gather/Broadcast

- Lever: replace the per-element coordinate-decode odometer in the structural
  data-movement kernels (transpose_general, slice_strided_gather,
  dynamic_slice_dense, gather_window_blocks, broadcast_replicate, rev_gather)
  with `extend_from_slice` block memcpy of the contiguous trailing run. This is an
  ALGORITHMIC change (memcpy vs odometer), distinct from boxed->dense de-box.
- Representative measured: TRANSPOSE block-copy (f62hx), the attention transpose
  [B,S,H,D]->[B,H,S,D] = [8,512,8,64] f32 (2,097,152 elems), perm (0,2,1,3) which
  keeps the [D]=64 feature vector contiguous (block_len=64).
- Conformance GUARD: GREEN. `transpose_gauntlet.rs` asserts the block-copy path is
  bit-identical to the pre-f62hx per-element odometer reference at 4 probe indices;
  fj-lax transpose conformance tests pass.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | throughput | vs naive |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive, committed) | 791.50 us | 2.65 Gelem/s | 10.30x faster |
  | naive per-element odometer (pre-f62hx) | 8.1525 ms | 257 Melem/s | baseline |

  - Head-to-head vs original JAX (`transpose_gauntlet.py`, jax.jit CPU x64,
    jnp.transpose+0.0 to force materialization, mean):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit transpose | 186.7 us | - |
  | Rust block-copy | 791.5 us | 4.24x slower |
  | Rust naive (pre-f62hx) | 8.15 ms | 43.6x slower |

- Decision: KEEP. The block-copy is a real **10.3x internal** algorithmic win and
  NARROWS the JAX gap from 43.6x to 4.24x. It is the strongest measured lever in
  this conversation's backlog (the dense de-box clusters only matched the boxed
  Rust reference; this one is a genuine throughput jump). Still a JAX loss (4.24x)
  but directionally the right kind of optimization.
- Generalization: the sibling block-copy kernels (slice/gather/dynamic_slice/
  broadcast/rev) use the same memcpy-vs-odometer transform on the same contiguous
  trailing-block structure, so each is expected to deliver a comparable internal
  multi-x win on its contiguous regime. Measured here via the transpose
  representative; the others KEEP on the same proof + the bit-identity tests.
- Retry predicate: closing the remaining 4.24x to JAX on transpose needs either
  cache-blocked tiling of the strided (non-identity-suffix) case (REJECTED before
  as a regression — see project_transpose_already_optimal) or avoiding the
  materialized transpose entirely (layout-aware fusion), NOT more block-copy.

## frankenjax-thnjs - Contiguous-Block Memcpy Broadcast (second block-copy data point)

- Lever: broadcast_replicate replicates the contiguous trailing source block via
  extend_from_slice instead of per-element coordinate decode (sibling of the
  transpose block-copy f62hx). Bias/feature broadcast [D] -> [rows, D] is in every
  transformer layer.
- Workload: bias broadcast [768] -> [4096, 768] f32 (3,145,728 elems), the
  D=768-model-dim-over-4096-tokens bias replicate. broadcast_dimensions [1].
- Conformance GUARD: GREEN. `broadcast_gauntlet.rs` asserts block-copy ==
  pre-thnjs per-element reference at 4 probe indices; broadcast conformance passes.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | vs naive |
  | --- | ---: | ---: |
  | block-copy (eval_primitive, committed) | 283.75 us | 21.80x faster |
  | naive per-element decode (pre-thnjs) | 6.1866 ms | baseline |

  - Head-to-head vs original JAX (`broadcast_gauntlet.py`, jax.jit CPU x64,
    jnp.broadcast_to+0.0 materialized):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit broadcast | 178.9 us (mean; p50 164.9, cv 48%) | - |
  | Rust block-copy | 283.75 us | ~1.59x slower |
  | Rust naive (pre-thnjs) | 6.19 ms | ~34.6x slower |

- Decision: KEEP. 21.8x internal win and only ~1.6x off JAX (NEAR-PARITY) — the
  closest-to-JAX block-copy lever measured. Broadcast is write-bandwidth-bound, so
  both engines sit near memory bandwidth (Rust ~44 GB/s vs JAX ~70 GB/s store
  throughput); the residual gap is store-vectorization, not algorithm.
- Cluster summary (2 measured): block-copy structural kernels deliver 10-22x
  internal wins and a 1.6-4.24x JAX gap (broadcast 1.6x write-bound, transpose
  4.24x strided) -- categorically better than the de-box dense clusters (50-128x
  JAX loss). The block-copy lever is the right kind of optimization; KEEP all
  siblings (slice/gather/dynamic_slice/rev) on the same transform + bit-identity tests.
- Retry predicate: broadcast's residual 1.6x to JAX is store throughput — would
  need SIMD/streaming-store output (non-temporal writes), not more block-copy.

## frankenjax-hfq7o - Dense integer_pow x**2 (MEASURED FIX: powi libcall -> v*v)

- Lever: dense integer_pow path (de-box) for the ubiquitous x**2 (variance/MSE/poly).
- Workload: integer_pow[2] on 1M f64 and 1M f32 (integer_pow_gauntlet.rs + .py).
- FINDING (gauntlet measurement caught a latent inefficiency): the original dense
  path used `v.powi(exponent)` with `exponent` a RUNTIME i32, so LLVM could not
  fold powi(2) to a mul -> a per-element powi LIBCALL, ~6.75 GB/s (10x below memory
  bandwidth). FIXED to special-case exponent==2 -> `v*v` (f64) /
  `(f64::from(v)*f64::from(v)) as f32`, bit-exactly equal to powi(2).
- Conformance GUARD: GREEN. `cargo test -p fj-lax --lib integer_pow` 10/0; bench
  asserts dense==boxed at 3 indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64):

  | Workload | dense BEFORE (powi) | dense AFTER (v*v) | fix speedup | JAX | Rust/JAX before -> after |
  | --- | ---: | ---: | ---: | ---: | --- |
  | integer_pow2_f64_1m | 2.3675 ms | 405.45 us | 5.84x | 184.1 us | 12.86x -> 2.20x slower |
  | integer_pow2_f32_1m | 2.6444 ms | 169.61 us | 15.59x | 121.1 us | 21.83x -> 1.40x slower |

  (Internal vs boxed Vec<Literal> path: f64 25.6ms/f32 27.5ms boxed -> 40-160x
  faster dense; but the headline is the powi->v*v fix that made x**2 bandwidth-bound.)
- Decision: KEEP + FIXED. The fix is the single biggest measured improvement of
  the gauntlet: it closed the JAX gap from 12.9-21.8x to 1.4-2.2x (f32 near-parity)
  by removing the runtime-powi libcall. x**2 is now bandwidth-bound (~40-47 GB/s,
  vs JAX ~66-87 GB/s store); the residual 1.4-2.2x is store throughput (same gap as
  the broadcast block-copy lever).
- Retry predicate: the remaining 1.4-2.2x is memory-store vectorization (SIMD/
  streaming stores) — the SAME residual as broadcast; NOT an algorithm issue. Apply
  the same powi-runtime-arg lesson to any other op taking a runtime small-integer
  power (none found in the hot path; integer_pow[3+] is rarer + needs powi grouping).

## frankenjax-idunl - Contiguous-Block Memcpy Slice (third block-copy data point: PARITY)

- Lever: slice_strided_gather copies the contiguous trailing block via
  extend_from_slice (2D crop / windowing). Workload: crop [1024,1024]->[512,512]
  f32 (262,144 output elems), start [256,256] limit [768,768].
- Conformance GUARD: GREEN. `slice_gauntlet.rs` asserts block-copy == pre-idunl
  per-element reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64,
  x[256:768,256:768]+0.0 materialized):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive) | 45.97 us | 6.10x faster | 1.04x slower (TIE) |
  | naive per-element | 280.51 us | baseline | ~6.4x slower |
  | JAX jit crop | 44.17 us (mean, cv 26%) | - | - |

- Decision: KEEP. 6.1x internal win and ~PARITY with JAX (1.04x, within noise) —
  the FIRST measured non-loss in the conversation. Slice is a pure contiguous
  block memcpy of sub-region rows; both engines are memcpy-bound and tie.
- Block-copy cluster (3 measured): transpose 4.24x (strided read), broadcast 1.59x
  (replicate, write-bound), slice 1.04x (sub-region, memcpy-bound). The
  bandwidth/memcpy-bound block-copy ops TIE or near-tie JAX; the strided-read
  transpose is 4.24x (random-access read pattern). The block-copy lever family is
  the closest-to-JAX work in the codebase; KEEP all.

## frankenjax (dense contiguous gather / embedding lookup) - random-access regime

- Lever: dense contiguous gather memcpy's each [dim] row via extend_from_slice
  (embedding lookup). Workload: table [16384,768] f32, gather 4096 random token
  rows -> [4096,768] (3.1M output elems, random reads into a 50MB table).
- Conformance GUARD: GREEN. `gather_gauntlet.rs` asserts dense == per-element
  reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jnp.take CPU x64):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (eval_primitive, row memcpy) | 1.1450 ms | 4.05x faster | 4.22x slower |
  | naive per-element | 4.6382 ms | baseline | ~17x slower |
  | JAX jit take | 271.3 us (cv 13%) | - | - |

- Decision: KEEP. 4.05x internal win; the per-row memcpy is the correct bit-exact
  approach. JAX-loss (4.22x) because gather is RANDOM-ACCESS-READ-bound (~11 GB/s
  effective: each row read is a cache miss into the 50MB table); XLA hides that
  latency better (prefetch / memory-level parallelism / SIMD gather).
- REFINED CLUSTER PATTERN (4 block-copy/structural ops measured): the JAX gap is
  set by the READ access pattern, not the copy:
    - SEQUENTIAL read (slice 1.04x TIE, broadcast 1.59x): near-parity with JAX.
    - RANDOM/STRIDED read (transpose 4.24x, gather 4.22x): ~4x JAX loss.
  All are 4-22x internal wins over the per-element baseline. The block-copy lever
  is correct; the residual ~4x on random/strided reads is a memory-access-pattern
  gap (prefetch / SIMD-gather / cache-aware tiling), NOT the copy algorithm.
- Retry predicate: closing the gather/transpose ~4x needs software prefetch or
  memory-level-parallelism (interleaved multi-row gather) — blocked by forbid-unsafe
  for raw prefetch intrinsics; would need a safe MLP-exposing restructure. NOT more
  block-copy. Do NOT retry per-element or naive gather.

## frankenjax-7eqrs - Dense Complex Constructor lax.complex(re,im) (de-box, near-parity)

- Lever: dense complex constructor zips two f64 slices into packed (re,im) storage
  via new_complex_values (FFT/signal real+imag combine), vs the boxed per-Literal
  path. Workload: re[1M]+im[1M] f64 -> complex128[1M] (32MB traffic).
- Conformance GUARD: GREEN. `complex_ctor_gauntlet.rs` asserts dense == boxed.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.lax.complex CPU x64):

  | Arm | median | vs boxed | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (new_complex_values) | 775.72 us | 25.2x faster | 1.56x slower |
  | boxed (per-Literal) | 19.558 ms | baseline | ~39x slower |
  | JAX jit lax.complex | 497.34 us (cv 6%) | - | - |

- Decision: KEEP. 25.2x internal de-box win and ~1.56x off JAX (NEAR-PARITY,
  bandwidth-bound at 41 vs 64 GB/s — the same store-throughput gap as broadcast).
- REFINES the de-box category split: de-box of DATA-MOVEMENT/simple ops (complex
  ctor 1.56x, integer_pow-after-v*v-fix 1.40x) APPROACHES JAX (bandwidth-bound);
  de-box of HEAVY-PER-LANE ops (clamp half 53-128x JAX loss, mcqr.106/.107) does
  NOT — there the per-lane decode/encode/compute dominates, not the boxing. So
  "de-box helps reach JAX" iff the op is bandwidth-bound, not per-lane-compute-bound.
- Retry predicate: complex ctor residual 1.56x is store throughput (same as
  broadcast) — SIMD/streaming stores, not algorithm. Conj/real/imag (1ylsj) and
  complex elementwise (same dense storage) expected to behave the same; KEEP.

## METHODOLOGY CORRECTION + elementwise binary (rch-vs-local calibration)

- CRITICAL: prior gauntlet rows (transpose/broadcast/slice/gather/integer_pow/
  complex_ctor) measured Rust via `rch exec` on a REMOTE worker, but JAX runs
  LOCALLY. Same-binary calibration (f32 add, one A/B binary run both places):
  rch worker 173.2 us vs LOCAL 119.6 us => the rch worker is ~1.45x SLOWER than
  local. So all prior Rust/JAX ratios are PESSIMISTIC by ~1.45x; the true
  same-host ratios are ~/1.45.
- f32 native-vs-widen add was ~0 gain on BOTH machines (rch 173.2/174.6 = 1.008x;
  local native 119.6 / widen 122.3 = 1.02x) -> f32 same-shape add is
  bandwidth-bound; the native-f32 attempt was REVERTED.
- LOCAL same-host elementwise dense vs JAX (the trustworthy same-machine numbers;
  JAX jax.jit CPU x64):

  | Workload | Rust dense (LOCAL) | JAX | Rust/JAX (same-host) |
  | --- | ---: | ---: | ---: |
  | add_f64_1m | 415.00 us | 192.0 us | 2.16x slower |
  | add_f32_1m | 135.98 us | 80.4 us | 1.69x slower |
  | mul_f64_1m | 422.96 us | 161.7 us | 2.61x slower |

- Corrected (÷~1.45) same-host estimates for the prior rch rows — several FLIP to
  wins/ties: slice ~0.72x (Rust FASTER), integer_pow2 f32 ~0.97x (Rust ~tie/win),
  broadcast ~1.10x, complex_ctor ~1.08x, integer_pow2 f64 ~1.52x, transpose ~2.93x,
  gather ~2.91x. (These are estimates; future vs-JAX rows MUST run the Rust bench
  LOCALLY, not via rch, for a same-host comparison.)
- Elementwise add/mul same-host loss (1.69-2.61x) is structural: per-op output
  allocation (frankenjax allocates a fresh Vec per primitive; XLA reuses buffers)
  + AVX2 (4-wide f64) vs JAX likely AVX-512 (8-wide). Both are build/architecture
  matters, not a code bug — the inner map+collect already autovectorizes (native
  vs widen tie proves the loop isn't the bottleneck).
- Decision: REVERT f32 native add (~0 gain). KEEP all measured dense paths. Retry
  predicate for elementwise: the gap is per-op allocation (needs buffer reuse =
  compiled-jaxpr arena) and AVX-512 (build flag) — NOT the elementwise loop.
