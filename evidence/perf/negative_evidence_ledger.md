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
