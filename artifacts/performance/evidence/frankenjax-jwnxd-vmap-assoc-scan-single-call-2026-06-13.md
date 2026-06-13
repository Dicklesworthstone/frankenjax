# frankenjax-jwnxd: single-call vmap rule for AssociativeScan

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule` routed `AssociativeScan` through `batch_passthrough_leading`
— B independent per-slice scans + a stack. `associative_scan` has NO axis param
(it always prefix-scans the operand's leading axis), so a plain axis-shift can't
work. `batch_associative_scan` instead:

1. moves the batch dim to front → `[B, T, …]` (scan/time axis now at 1),
2. swaps axes 0↔1 (self-inverse permutation) → `[T, B, …]`,
3. runs ONE `associative_scan` — it prefix-combines the `[B, …]` slices
   elementwise along T, which is exactly the per-slice scan for every batch lane,
4. swaps back → `[B, T, …]`.

The whole-batch call also engages the dense scan fast paths (vs B small per-slice
scans). The rank-0 per-element case (batched scalar; scan is identity) defers to
the per-slice path.

## Parity

associative_scan's combine is associative + deterministic and applied per
`[B, …]` lane independently, so the transposed single scan equals the per-slice
stack; `body_op`/`reverse` pass through unchanged.
`batch_associative_scan_matches_per_slice_fallback` asserts identical
batch_dim+shape+f64 bits vs `batch_passthrough_leading` across body_op
`add`/`mul`/`max`, reverse `false`/`true`, ranks `[B,T]` and `[B,T,X]`, and
batch_dim 0 and 1.

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_associative_scan_single_call_vs_per_slice --release -- --ignored --nocapture

BENCH vmap(associative_scan) [65536,16]: per-slice=106.1158ms single-call=36.5139ms speedup=2.91x
```

Keep: **2.91x**. Score: 2.91 × 0.95 / 1 = 2.76.

Behavior proof: 296 fj-dispatch lib tests pass (incl. the new parity test); the
bench asserts identical output element count before timing.
