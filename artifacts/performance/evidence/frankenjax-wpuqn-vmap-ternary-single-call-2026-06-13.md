# frankenjax-wpuqn: single-call vmap rule for Fma/Betainc

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule` routed `Fma` (`a*b+c`) and `Betainc` (`I_x(a,b)`) through
`batch_passthrough_leading` — per-slice eval+stack (B dispatches + B allocs +
stack). Both are pure 3-input **elementwise** ops, identical in structure to
`Clamp`/`Select`, which already batch via `harmonize_ternary` + one eval.

`batch_ternary_elementwise(primitive, …)` harmonizes the three operands to a
common batch-front shape (`harmonize_ternary` moves/broadcasts each + aligns
logical ranks) and evals ONCE. The op is elementwise+deterministic, so the single
call's per-element results equal the per-slice stack bit-for-bit. For Betainc the
single call also lets the eval's own threaded path (`eval_ternary_elementwise`
fans out at its work threshold) run across the whole batch, where per-slice ran B
serial slices.

## Parity

`batch_ternary_elementwise_matches_per_slice_fallback` asserts the single-call
output (batch_dim + shape + f64 bits) equals `batch_passthrough_leading` for both
Fma and Betainc, with (1) all three operands batched and (2) a shared unbatched
scalar third operand (broadcast).

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_betainc_single_call_vs_per_slice --release -- --ignored --nocapture

BENCH vmap(betainc) [65536,8]: per-slice=178.9019ms single-call=72.1572ms speedup=2.48x
```

Betainc is compute-bound (continued fraction + lgamma per element); the per-slice
path ran 65536 serial betaincs, the single call fans the whole batch across
threads. Keep: **2.48x**. Score: 2.48 × 0.95 / 1 = 2.36.

Behavior proof: 294 fj-dispatch lib tests pass (incl. the new parity test); the
bench also asserts identical output element count before timing.
