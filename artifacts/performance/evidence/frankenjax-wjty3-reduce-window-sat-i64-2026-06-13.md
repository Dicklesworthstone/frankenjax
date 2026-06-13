# frankenjax-wjty3: summed-area-table i64 sum reduce_window

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever (alien-graveyard: integral image / summed-area table)

Rank-2 i64 sum-pooling (windowed reduction) summed each output window's taps:
O(output · ∏window). Replace with a **summed-area table** (integral image): build
the prefix-sum table once (O(input)), then every output window sum is a 4-corner
inclusion-exclusion (O(1) per output, **window-independent**). Total
O(input + output) vs O(output · ∏window) — a different complexity class, so the
win grows without bound as the window grows.

`eval_reduce_window_rank2_i64_sum_sat`: integral image of shape
`(rows+1)×(cols+1)` with a zero first row/col; each output =
`sat[r1][c1] − sat[r0][c1] − sat[r1][c0] + sat[r0][c0]` over the clamped
in-bounds rectangle. Gated `rank==2 && sum && i64 && no dilation && ∏window ≥ 16`
(small 2×2/3×3 windows stay on the per-window dense path, which has no
integral-image build cost).

## Parity (bit-exact for integers)

Integer addition is associative AND commutative — i64 `wrapping_add` is exact in
the ring ℤ/2⁶⁴ and `wrapping_sub` inverts it — so the 4-corner rectangle sum
equals the per-window ascending wrapping tap sum **bit-for-bit**, even under
overflow. OOB/padding taps contribute 0 in both (the clamped in-bounds rectangle
is exactly the set of non-pad taps).

`reduce_window_i64_sum_sat_matches_per_window` asserts equality vs an independent
per-window in-bounds wrapping-sum reference across VALID/SAME padding, strides
{1,2,3}, and windows {4×4, 5×5, 7×3, 4×5}. (Float sum-pooling stays on the
per-window path — FP add is NOT associative, so SAT would not be bit-identical.)

## Result (same-invocation A/B, SAT vs per-window dense)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_i64_sum_sat_vs_dense --release -- --ignored --nocapture

BENCH reduce_window i64 sum([512,512],win=15x15,s=1): per-window=29.8232ms SAT=1.8089ms speedup=16.49x
```

Keep: **16.49x** at 15×15, and the ratio grows with window size (SAT is
window-independent; per-window is O(∏window)). Score: 16.49 × 0.95 / 1 = 15.7.

Behavior proof: 38 reduce_window lib tests pass (incl. the new parity test); the
bench asserts identical checksum vs the per-window reference before timing.
