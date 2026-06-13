# frankenjax-xek59: dense complex reduce_window (sum/max/min)

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever

Complex (Complex64/Complex128) `reduce_window` was the LAST dtype on the generic
per-`Literal` path — every other dtype (f64/f32/i64/bool/bf16/f16) already had a
dense fast path. `eval_reduce_window_dense_complex` reads the dense `(re, im)`
slice (`as_complex_slice`) and runs the hoisted op over the same row-major
tap-offset stencil + interior/border split as the i64 dense path, emitting dense
complex storage (`new_complex_values`).

## Parity (bit-identical)

Sum accumulates the `(re, im)` components in window order (so the f64 component
sums round identically to the generic odometer); max/min use the SAME
lexicographic `reduce_window_complex_ge` and the SAME init
(`reduce_window_complex_initial`: max=(−∞,−∞), min=(+∞,+∞), sum=(0,0)); OOB/border
taps contribute that init, a no-op for every op.
`reduce_window_complex_dense_matches_generic` asserts the dense (Complex storage)
output's `(re,im)` bits equal the boxed-`Literal` generic path's for sum/max/min
across VALID/SAME padding and strides {1,2}. 41 reduce_window tests pass.

## Result (same-invocation A/B, dense vs generic per-Literal)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_complex_dense_vs_generic --release -- --ignored --nocapture

BENCH reduce_window Complex128 sum([384,384],win=3x3): generic=28.5450ms dense=2.3986ms speedup=11.90x
```

Keep: **11.90x**. Score: 11.90 × 0.95 / 1 = 11.3.

## reduce_window dtype family — now COMPLETE

Every dtype has a dense (and where applicable window-independent) path:
- f64/f32: dense f64-accumulator (+ separable deque max/min, rank-2 stencils).
- i64: dense + summed-area-table sum (rank 1–6) + monotonic-deque max/min.
- bf16/f16: dense f64-accumulator (zm88o) + separable deque max/min (m348z).
- bool: dense logical and/or.
- **complex: dense (this)**.
