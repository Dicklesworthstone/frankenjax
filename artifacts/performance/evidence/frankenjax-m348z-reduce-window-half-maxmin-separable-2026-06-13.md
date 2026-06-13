# frankenjax-m348z: separable deque bf16/f16 max/min reduce_window

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever

Today's bf16/f16 reduce_window work (frankenjax-zm88o) routes half-float max/min
through the dense f64-accumulator path — per-window O(output·∏window). The
window-independent monotonic-deque separable max/min path
(`reduce_window_separable_maxmin`) was still gated F64/F32. Since `zm88o` already
taught `reduce_window_dense_f64_view` to widen bf16/f16→f64, and max/min SELECT
an exactly-widened input value (so rounding back via the same
`reduce_window_literal_from_f64` is a round-trip identity), extending the
separable gate + output to bf16/f16 is bit-identical and makes large-window
half-float max/min pooling window-independent.

Gated `max/min && ∏window > 2·∑window` (same as float; small windows stay on the
dense_float per-window path).

## Parity (bit-identical)

max/min select an exactly-widened input value (or the ∓∞/NaN init for
all-pad/NaN windows), rounded back via the SAME `reduce_window_literal_from_f64`
the generic/dense path uses. `reduce_window_half_maxmin_separable_matches_generic`
asserts the dense (HalfFloat storage, separable deque) half bits equal the
boxed-`Literal` generic path's for BF16 and F16, max and min, on a 15×15 window
(∏window=225 > 2·∑window=60). 40 reduce_window tests pass.

## Result (same-invocation A/B, separable deque vs per-window)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_half_maxmin_separable_vs_dense --release -- --ignored --nocapture

BENCH reduce_window BF16 max([512,512],win=15x15,s=1): per-window=199.7572ms separable=4.4755ms speedup=44.63x
```

Keep: **44.63x** at 15×15 (grows with window — separable is window-independent,
per-window is O(∏window)). Score: 44.63 × 0.95 / 1 = 42.

## reduce_window family status (window-independent for large windows)

- sum: i64 summed-area-table (wjty3 rank-2, rmsv6 rank 1/3-6); float stays
  per-window (FP non-associative).
- max/min: float deque (pre-existing), i64 deque (mv6p6), **bf16/f16 deque (this)**.
- Small-window + half-float small-window: dense f64-accumulator path (zm88o).
