# frankenjax-cod-b-dense-tensor-stack-axis0-rw4k4

## Lever

Route `TensorValue::stack_axis0` for tensor inputs through
`LiteralBuffer::from_concat_slices` plus `TensorValue::new_with_literal_buffer`
after preserving the existing dtype and shape validation. This keeps dense
tensor lanes packed through loop/vmap-style tensor stacks instead of eagerly
materializing a flat `Vec<Literal>`.

## Benchmark Target

- `core/tensor_stack_axis0_dense_f64_64x1k`
- `core/tensor_stack_axis0_literal_f64_64x1k`

Expected win: dense inputs avoid 65,536 `Literal` constructions during the
stack call and defer materialization until a literal slice is explicitly
requested.

## Conformance Guard

- Added `stack_axis0_tensor_uses_concat_dense_storage` to pin dense F64, F32,
  I64, U32, Bool, BF16, and Complex64 storage lanes after tensor stack.
- Existing stack dtype, shape-mismatch, mixed-kind, and rank-restoration checks
  continue to exercise externally visible semantics.

## Negative-Evidence Ledger

- Kept rejected/no-gate FMA, SIMD exp, GEMM, QR, SVD, and cumsum families out of
  this batch; those require fresh same-worker benchmark evidence or maintainer
  policy decisions before retry.
- Did not touch peer-owned fj-lax arithmetic or lax benchmark files despite
  local dirt.
- Did not run criterion, rch, or tests in this code-first batch per explicit
  user constraint; the committed criterion targets are ready for the batch
  benchmark lane.

## Validation

Passed:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core
```

Result: `Finished dev profile ... target(s) in 0.30s`.
