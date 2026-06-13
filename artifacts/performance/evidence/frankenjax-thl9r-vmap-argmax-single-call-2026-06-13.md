# frankenjax-thl9r: single-call vmap rule for Argmin/Argmax

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule` routed `Argmin`/`Argmax` through `batch_passthrough_leading`
— the per-slice fallback: for a batch of size B it slices the batched operand B
times, calls `eval_primitive(argmax)` on each `[…]` slice, then `stack_axis0`s B
results (B dispatches + B result `Value` allocs + a stack alloc).

argmax/argmin are **single-axis reductions** (eval reads the `"axis"` param,
default last axis, exactly one axis reduced — no flatten mode). So vmap of an
argmax over original axis `a` is just an argmax over axis `a+1` of the
batch-front tensor, in ONE `eval_primitive` call — and that single call already
evaluates B contiguous slices via the SIMD/threaded contiguous-argmax path.

`batch_argmax_argmin` mirrors `batch_reduce`: move batch dim to front, normalize
the original axis against the per-element rank (negative → end-relative, absent →
`rank-1`) EXACTLY as `parse_axis_param` does, then shift `+1` and re-emit the
`"axis"` param. The rank-0 per-element case (batched scalar) defers to the
per-slice path.

## Parity (project_vmap_param_key_mismatch)

eval reads `"axis"`, so the rule shifts `"axis"` (not "dimension"); the shift is
applied to the NORMALIZED non-negative axis (so a negative/end-relative axis is
not naively `+1`'d). The shifted axis is always ≥ 1, so the batch dim (now 0) is
never reduced.

`batch_argmax_argmin_matches_per_slice_fallback` asserts the single-call output
(batch_dim + shape + i64 indices) is element-identical to
`batch_passthrough_leading` across: axis `0`, `1`, `-1`, `-2`, absent-default;
ranks 2 and 3; batch_dim at 0 and 1; both Argmax and Argmin.

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_argmax_single_call_vs_per_slice --release -- --ignored --nocapture
```

| shape `[B,N]` axis=-1 | per-slice | single-call | speedup |
|---|---|---|---|
| `[524288, 8]`  (classification argmax) | 291.73ms | 90.63ms | **3.22x** |
| `[262144, 16]` | 120.55ms | 64.27ms | 1.88x |
| `[8192, 2048]` | 284.50ms | 263.72ms | 1.08x |

The win scales with dispatch-dominance: large-N argmax is memory-bound (the
single call still reads the same bytes, so ~1.08x), but the realistic
small-class-dim shape (vmap argmax over a class axis with a large batch) is
dominated by the B dispatches + B per-result allocs the single call eliminates.
Keep row: **`[524288,8]` 3.22x**. Score: 3.22 × 0.95 / 1 = 3.06.

Behavior proof: 292 fj-dispatch lib tests pass (incl. the new parity test); the
bench also asserts identical indices before timing.
