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
