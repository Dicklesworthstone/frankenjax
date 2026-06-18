# frankenjax-mcqr: dense SelectN index-decode pass

Date: 2026-06-18
Agent: cod-a / RedBeaver
Status: in_progress, code-first batch-test pending

## Lever

Extend the existing dense tensor-output `SelectN` path to decode dense index
buffers directly:

- I64: existing inline path retained.
- U32: decode contiguous table IDs without materializing `Literal::U32`.
- U64: decode contiguous table IDs while preserving the existing
  `Literal::U64(...).as_i64()` `i64::MAX` guard.
- Bool: decode dense boolean indices directly for two-way select.
- BoolWords: bit-test packed boolean indices directly for two-way select.

This is a cache-layout lever, not a semantic rewrite. `SelectN` remains
"decode index, bounds-check, copy case[index][i]". The generic per-`Literal`
index loop remains the fallback for unsupported storage layouts.

## Alien-source mapping

- `/alien-graveyard` section 8.2 vectorized execution: process dense batches
  instead of tuple-at-a-time interpretation. Here the tuple-at-a-time cost was
  the index `Literal` materialization inside a dense output gather.
- `/alien-graveyard` section 7.2 succinct bitvectors: keep packed predicates in
  `Vec<u64>` and query bits directly. The BoolWords arm follows that model.
- `/alien-graveyard` section 7.7 Swiss Tables: separate compact metadata from
  payload and scan the compact representation in cache-friendly groups. The
  index buffer is the metadata stream; dense case tensors are the payload.
- `/alien-artifact-coding`: preserve the isomorphism witness by comparing dense
  index outputs to the boxed literal path for every new index storage class.
- `/extreme-software-optimization`: one lever per commit, local behavioral guard,
  and criterion rows for the next batch run.

## Correctness guard

Updated `dense_select_n_matches_literal_path_and_stays_dense` in
`crates/fj-lax/src/arithmetic.rs` to compare:

- Dense U32 index vs boxed U32 index on dense/boxed U32 cases.
- Dense U64 index vs boxed U64 index on dense/boxed U64 cases.
- Dense U64 index greater than `i64::MAX` preserves the existing
  `TypeMismatch` behavior.
- Packed BoolWords index vs boxed Bool index on two-way dense/boxed U32 cases.
- Packed BoolWords index with three cases still rejects boolean indices with the
  existing error.

## Benchmark guard

Added criterion rows in `crates/fj-lax/benches/lax_baseline.rs`:

- `eval/select_n_64k_u32_boolwords_index_vec`
- `eval/select_n_64k_u32_bool_index_literal_ref`

These quantify the direct packed-mask index path against the literal reference
path in the next batch run.

## Negative-evidence ledger

- U32/U64/Bool/Complex case-tensor dense output was already shipped in the prior
  SelectN wide-storage pass. This pass is only the index decode lever; do not
  retry the same case-output expansion without new profile evidence.
- `tile` U32/U64/Complex was previously inspected and rejected as an edit target:
  `eval_tile` already has dense U32/U64/Bool/Complex arms. No retry without a
  fresh profile showing a different tile bottleneck.
- FMA/SIMD-exp/GEMM remains maintainer-gated under `frankenjax-cntiy`; no retry
  until the policy decision changes.
- Cumsum axis specialization remains a prior rejected/non-comparable family; no
  retry without fresh same-worker evidence.
- No speedup is claimed in this commit. Benchmarks were intentionally not run in
  this code-first pass; the criterion rows are the batch-test target.

## Local validation

Passed:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax
```
