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
