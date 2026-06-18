# frankenjax-cod-b-dense-to-i64-vec-7xbu9

## Change

Route `TensorValue::to_i64_vec` through `LiteralBuffer::as_i64_slice` before
falling back to per-`Literal` conversion. This preserves the existing fallback
for literal-backed buffers while avoiding typed-to-`Literal` materialization on
packed I64/I32-backed tensors and packed I64 concat slices.

## Recommendation Contract

- Hotspot evidence: dense storage remains the routed fj-core frontier from prior
  packed `LiteralBuffer` work; realistic interpreter and dispatch paths extract
  index tensors with `to_i64_vec`.
- Mapped graveyard sections: cache-aware flat layout, constants-kill-you
  benchmarking discipline, and evidence ledger requirements from the canonical
  alien graveyard corpus.
- EV score: Impact 3 * Confidence 4 * Reuse 4 / (Effort 1 * Friction 1) = 48.
- Priority: A for code-first batch; S only after same-worker criterion evidence.
- Baseline comparator: old `to_i64_vec` iterated `self.elements`, forcing
  dense buffers to materialize `Literal::I64` values before collecting `i64`.
- Fallback trigger: if criterion shows no same-worker win or any conformance
  guard fails, revert this single commit and mark the attempt rejected.
- Budgeted mode: O(n) copy remains the only required work; no new allocation
  beyond the returned `Vec<i64>`.
- Primary risk: accidentally changing malformed/literal fallback behavior.
  Countermeasure: focused guards cover dense, concat-dense, literal fallback,
  and wrong-dtype `None`.

## Negative-Evidence Ledger

| Attempt | Status | Evidence | Retry Rule |
| --- | --- | --- | --- |
| FMA/SIMD exp and FMA-GEMM | Blocked | `frankenjax-cntiy` requires maintainer policy for global `+fma` or audited unsafe target features. | Do not retry in cargo-check-only batch. |
| GEMM/QR/SVD/cumsum/concat eager families | Rejected/deferred | Existing beads and memory route these to benchmark-capable sessions. | Reopen only with fresh same-worker criterion evidence. |
| Dense `to_i64_vec` typed-slice fast path | Pending | This commit adds code and guards; benchmark execution is batch-test pending by directive. | Keep only if `core/tensor_to_i64_vec_dense_4k` beats the literal path and no conformance guard regresses. |

## Verification Plan

- Added criterion targets:
  - `core/tensor_to_i64_vec_dense_4k`
  - `core/tensor_to_i64_vec_literal_4k`
- Added conformance guards:
  - dense I64 path preserves values and exposes packed storage
  - literal fallback preserves values when no packed slice exists
  - concat dense I64 path preserves values through the new fast path
  - wrong dtype still returns `None`
- Validation allowed in this batch:
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core`

## Batch Status

Code-first batch-test pending: no tests, rch, or criterion execution were run in
this turn by user directive.
