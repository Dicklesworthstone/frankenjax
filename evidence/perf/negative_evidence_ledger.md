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
- Status: batch-test pending.
- Benchmark guard: `eval/bitcast_f32_bf16_dense_1m`,
  `eval/bitcast_f32_bf16_literal_ref_1m`,
  `eval/bitcast_bf16_f32_dense_1m`,
  `eval/bitcast_bf16_f32_literal_ref_1m`.
- Conformance guard: dense and literal-backed half-width bitcasts produce the
  same output shapes, dtypes, raw half chunks, and round-trip f32 bit patterns
  across NaN, infinities, signed zero, normals, and a custom NaN payload.
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
- Status: batch-test pending.
- Benchmark guard: `eval/bitcast_f32_i32_dense_1m`,
  `eval/bitcast_f32_i32_literal_ref_1m`,
  `eval/bitcast_i32_f32_dense_1m`,
  `eval/bitcast_i32_f32_literal_ref_1m`,
  `eval/bitcast_f64_u64_dense_1m`,
  `eval/bitcast_f64_u64_literal_ref_1m`,
  `eval/bitcast_u64_f64_dense_1m`,
  `eval/bitcast_u64_f64_literal_ref_1m`.
- Conformance guard: dense and literal-backed signed/unsigned same-width
  bitcasts produce the same shapes, dtypes, exact integer bit lanes, packed
  storage, and round-trip float bit patterns across NaN, infinities, signed
  zero, normals, and custom NaN payloads.
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

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: route scalar `BroadcastInDim` fills for U32, U64, Complex64, and
  Complex128 through `new_u32_values`, `new_u64_values`, and
  `new_complex_values` instead of allocating a `Vec<Literal>` fill.
- Status: batch-test pending.
- Benchmark guard: `eval/broadcast_scalar_u32_1024x1024`,
  `eval/broadcast_scalar_u64_1024x1024`,
  `eval/broadcast_scalar_complex128_1024x1024`.
- Conformance guard: scalar fills materialize to the same repeated literals and
  expose dense typed storage via `as_u32_slice`, `as_u64_slice`, or
  `as_complex_slice`.
- Retry predicate: do not retry scalar `BroadcastInDim` dense-fill storage for
  these dtypes unless focused criterion evidence shows this path remains a
  top-five `fj-lax` bottleneck or the dense constructor representation changes.
  Do not merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota,
  or broader scalar-broadcast arithmetic work without fresh benchmark evidence
  and ownership check.

## frankenjax-dxqfj - Lazy SplitMulti Section Buffers

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: build each `eval_split_multi` output from `LiteralBuffer::from_concat_slices`
  over the original tensor backing instead of copying each section through
  `Vec<Literal>` and re-densifying via `TensorValue::new`.
- Status: batch-test pending.
- Benchmark guard: `eval/split_multi_1024x1024_f32_axis1`.
- Conformance guard: uneven multi-output split materializes the same literals for
  each section and exposes dense f32 storage through `as_f32_slice`.
- Retry predicate: do not retry split section materialization unless focused
  criterion evidence shows `eval_split_multi` remains a top-five shape/data
  movement bottleneck or `LiteralBuffer::Concat` dense-lane behavior changes.
  Do not merge with FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota,
  or broader reshape/slice/gather work without fresh benchmark evidence and
  ownership check.

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
