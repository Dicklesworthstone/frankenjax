# frankenjax-cod-b-dense-tensor-repeat-axis0-jk3ed

## Change

Route `TensorValue::repeat_axis0` for tensor inputs through
`LiteralBuffer::from_concat_slices` and `TensorValue::new_with_literal_buffer`.
Dense tensor repeats now store repeated blocks as packed concat slices instead of
eagerly materializing `Vec<Literal>` and then reconstructing typed storage.

## Recommendation Contract

- Hotspot evidence: prior fj-core dense-storage wins made `LiteralBuffer`
  preservation the active route for vmap/broadcast-style workloads.
- Mapped graveyard sections: flat/cache-aware layout, vectorized batch
  execution, constants-kill-you benchmark discipline, and evidence-ledger
  contracts from the canonical alien graveyard corpus.
- EV score: Impact 4 * Confidence 4 * Reuse 4 / (Effort 2 * Friction 1) = 32.
- Priority: A for code-first batch; S only after same-worker criterion evidence.
- Baseline comparator: old tensor repeat allocated and filled a repeated
  `Vec<Literal>`, forcing dense lanes through literal materialization.
- Fallback trigger: revert this commit if `core/tensor_repeat_axis0_dense_f64_1k_x64`
  does not beat the literal/materializing baseline or any conformance guard
  regresses under the batch suite.
- Budgeted mode: storage metadata is O(repeat_count), returned tensor length is
  unchanged, and empty tensors avoid repeat-count-sized slice metadata.
- Primary risk: changing repeated tensor shape/equality or dtype-family storage.
  Countermeasure: guards cover equality with repeated stack plus dense borrowable
  lanes across representative numeric, bool, half, and complex dtypes.

## Negative-Evidence Ledger

| Attempt | Status | Evidence | Retry Rule |
| --- | --- | --- | --- |
| FMA/SIMD exp and FMA-GEMM | Blocked | `frankenjax-cntiy` is maintainer-gated on global `+fma` vs audited unsafe target features. | Do not retry in cargo-check-only batch. |
| GEMM/QR/SVD/cumsum eager families | Rejected/deferred | Existing memories and beads route these to benchmark-capable same-worker sessions. | Reopen only with fresh focused and broad criterion evidence. |
| Scalar repeat dense packing | Shipped separately | `frankenjax-cod-b-dense-scalar-repeat-axis0-zb2b7` covers scalar input repeats. | Keep tensor repeat attribution separate. |
| Tensor repeat concat storage | Pending | This commit adds code and guards; benchmark execution is batch-test pending by directive. | Keep only if dense repeat construction wins and conformance stays green. |

## Verification Plan

- Added criterion targets:
  - `core/tensor_repeat_axis0_dense_f64_1k_x64`
  - `core/tensor_repeat_axis0_literal_f64_1k_x64`
- Added conformance guards:
  - repeated tensor still equals repeated stack
  - dense tensor repeats preserve borrowable packed lanes for F64, F32, I64,
    U32, Bool, BF16, and Complex64
  - empty/overflow repeat behavior remains fail-closed through the existing
    leading-dimension guard
- Validation allowed in this batch:
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core`

## Batch Status

Code-first batch-test pending: no tests, rch, or criterion execution were run in
this turn by user directive.
