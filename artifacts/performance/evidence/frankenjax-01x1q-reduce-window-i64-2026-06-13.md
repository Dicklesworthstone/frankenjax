# frankenjax-01x1q: dense i64 reduce_window fast path (integer pooling)

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax

## Lever

`reduce_window` (max/min/sum pooling + windowed reductions) had a dense `f64`
fast path (`eval_reduce_window_dense_float`, any rank, F64 + dense-F32) but
**integer input had none** — I64 tensors ran the generic per-`Literal` gather +
string-dispatched `reduce_window_accumulate_literal` loop, the slowest path.

Add `eval_reduce_window_dense_i64`, a direct mirror of the float path: read the
dense `i64` buffer, hoist the op once (`i64::max` / `i64::min` / `wrapping_add`),
and run it over the same precomputed row-major tap-offset stencil with the same
interior-vs-border split. Wired in `eval_reduce_window` after the dense-float
block, gated `no_base_dilation && no_window_dilation && dtype==I64 &&
(max|min|sum)`.

I32 deliberately stays on the generic path (i32 is stored boxed → `as_i64_slice`
returns None) and is narrowed mod-2^32 by the `eval_primitive` chokepoint
regardless.

## Baseline + Result (same worker, same invocation A/B)

Command:

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_dense_i64_vs_generic --release -- --ignored --nocapture
```

```text
BENCH reduce_window i64 sum([512,512],win=3x3,stride=1): generic=13.7145ms dense=3.0953ms speedup=4.43x
```

The A/B is fair: both sides build an output buffer; the generic reference calls
the SAME `reduce_window_initial_accumulator` / `reduce_window_accumulate_literal`
/ `reduce_window_accumulator_literal` helpers the production generic loop uses,
gathering each tap as `t.elements[flat]`. Checksums match
(`assert_eq!(d_gen, d_dense)`).

Keep: **4.43x**. Score: Impact 4.43 x Confidence 0.95 / Effort 1 = 4.2.

## Isomorphism Proof

- Same row-major tap order (identical odometer / tap-offset construction as the
  float path and the generic loop).
- Same i64 init: max=`i64::MIN`, min=`i64::MAX`, sum=`0` — matching
  `reduce_window_initial_accumulator` for the I64 dtype.
- Same ops: `i64::max` / `i64::min` / `wrapping_add` — exactly the I64 arm of
  `reduce_window_accumulate_literal`.
- OOB/border taps contribute `init`, which is the identity for every op
  (`wrapping_add(0)`, `max(_, i64::MIN)`, `min(_, i64::MAX)` are no-ops) — same
  as the generic `pad_literal` contribution.
- Output is `Literal::I64`, matching `reduce_window_accumulator_literal` for I64.
- Sum order is irrelevant (wrapping add is associative/commutative mod 2^64);
  max/min are idempotent — so the result is bit-identical regardless of the
  interior/border traversal.

Behavior proof: all 37 `reduce_window` lib tests pass unchanged, including the
i64 paths (`reduce_window_sum_preserves_i64_literal_dtype_and_wraps`,
`reduce_window_max_1d`, `reduce_window_min_1d`, `reduce_window_max_1d_stride2`)
that now route through the dense i64 path.

```text
rch exec -- cargo test -p fj-lax --lib reduce_window --release
=> test result: ok. 37 passed; 0 failed
```
