# frankenjax-cod-b-dense-slice-axis0-4bnj5

Status: code-first batch-test pending
Agent: TopazOrchid
Date: 2026-06-18

## Lever

Preserve dense `LiteralBuffer` storage through `TensorValue::slice_axis0` for rank > 1 tensors by routing the contiguous row slice through `LiteralBuffer::from_concat_slices` and `TensorValue::new_with_literal_buffer`.

This targets realistic `vmap` and mapped-dispatch workloads where axis-0 extraction feeds many small tensor slices into downstream dense fast paths. The old path used `self.elements[start..end].to_vec()`, which forced dense F64/F32/Bool/Half/Complex storage through lazy `Literal` materialization before rebuilding the tensor.

## Expected Impact

- Hot workload: `vmap` mapped argument extraction and interpreter/dispatch tensor slicing.
- Bench target: Criterion rows that repeatedly slice mapped tensor inputs before dense scalar/tensor kernels.
- Comparator: current Rust port vs legacy JAX original where available, plus same-worker before/after Rust rows in the pending batch-test phase.

## Behavior Preservation

- Ordering preserved: yes, the slice remains the same contiguous row-major `[start, start + slice_len)` interval.
- Tie-breaking unchanged: N/A.
- Floating point: bit-identical; dense F32/F64 materialization uses the same stored bits and `Literal::from_f32/from_f64` paths.
- Half/complex: bit-identical; the existing dense half/complex concat accessors enforce uniform dtype before exposing packed slices.
- RNG seeds: unchanged/N/A.
- Error surface: unchanged for empty stacks, rank-zero slices, out-of-range indices, shape overflow, and element-count mismatch.

## Guard

Added `tensor_slice_axis0_preserves_dense_storage` covering F64 NaN payloads, F32 signed zero/NaN bits, U32, Bool, BF16, and Complex64. The guard checks both dense accessors and materialized literal equality where bit preservation is most fragile.

## Negative-Evidence Ledger

| Candidate | Verdict | Retry Predicate |
| --- | --- | --- |
| Touch peer-owned `crates/fj-lax/src/arithmetic.rs` U32/U64 sort work | Rejected this batch | Only retry when the peer-owned dirty diff is integrated or explicitly handed off. |
| Touch peer-owned `crates/fj-lax/benches/lax_baseline.rs` bench edits | Rejected this batch | Only retry after the current bench diff is committed by its owner or the user directs ownership transfer. |
| FMA/global SIMD-exp policy | Blocked | Only retry after `frankenjax-cntiy` maintainer decision resolves global `+fma` vs audited per-function target features. |
| GEMM/QR/SVD/cumsum eager micro-levers | Rejected by prior campaign evidence | Only retry with a fresh profile naming the exact primitive and same-worker Criterion baseline. |
| Dense scalar stack/repeat packing | Already shipped | Do not repeat; extend to a distinct storage boundary only. |

## Validation

Per user directive, this batch only runs:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core
```

No tests, RCH, Criterion, or conformance suite were run in this code-first pass.
