# frankenjax-zm88o: dense bf16/f16 reduce_window (half-float pooling)

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever

bf16/f16 `reduce_window` (sum/max/min, **any** window incl. the common 3×3 pool)
ran the fully-generic per-`Literal` path: `reduce_window_dense_f64_view` only
covered F64/F32, so half-float never reached the dense f64-accumulator fast path
(`eval_reduce_window_dense_float`). bf16/f16 is JAX's common ML pooling dtype, so
this is a common-case gap (not just large windows).

Fix: (1) extend `reduce_window_dense_f64_view` to widen bf16/f16 → f64 EXACTLY as
`Literal::as_f64` does (the same conversion the generic path applies per tap);
(2) add a BF16/F16 output arm to `eval_reduce_window_dense_float` that rounds each
f64 accumulator via the SAME `reduce_window_literal_from_f64` and emits dense half
storage; (3) widen the dense-float wiring gate to include BF16/F16.

## Parity (bit-identical)

The generic half path uses an F64 accumulator (the `_` arm of
`reduce_window_initial_accumulator`), accumulates `as_f64`-widened taps in window
order, and rounds once via `reduce_window_literal_from_f64`. The dense path does
the identical widen, identical f64 accumulate in the same row-major window order,
and the identical final round → bit-for-bit identical (sum: f64 accumulate then
one round; max/min: selects an exactly-widened input value, round-trip exact).

`bench_reduce_window_half_dense_vs_generic` asserts the dense (HalfFloat storage)
output's half bits equal the boxed-`Literal` generic path's for BF16 and F16,
sum and max. Existing `reduce_window_max_preserves_bf16_literal_dtype` etc. now
route through the dense path and still pass (40 reduce_window tests green).

## Result (same-invocation A/B, dense vs generic per-Literal, 3×3 pool)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_half_dense_vs_generic --release -- --ignored --nocapture

BENCH reduce_window BF16 sum([512,512],win=3x3): generic=47.5927ms dense=5.7906ms speedup=8.22x
BENCH reduce_window BF16 max([512,512],win=3x3): generic=56.2025ms dense=8.9149ms speedup=6.30x
BENCH reduce_window F16  sum([512,512],win=3x3): generic=60.6528ms dense=7.5548ms speedup=8.03x
BENCH reduce_window F16  max([512,512],win=3x3): generic=57.1276ms dense=8.8694ms speedup=6.44x
```

Keep: **6.3–8.2x** on the common 3×3 half-float pool. Score: 8.22 × 0.95 / 1 = 7.8.
