# frankenjax-43tr8 SIMD contiguous argmax/argmin evidence

Bead: `frankenjax-43tr8`
Agent: `BeigeMouse`
Date: 2026-06-13

## Lever

Add safe-Rust portable SIMD reducers for dense F32/F64 `argmin`/`argmax` when the reduced axis is contiguous (`axis_stride == 1`).

The generic scalar reducer remains the source of truth for:

- Strided rows.
- Rows shorter than one SIMD vector.
- Any row containing NaN.

The SIMD path uses strict `>` / `<` replacement masks, so ties keep the earliest index. Horizontal reduction explicitly chooses the lower index on equal values. If any SIMD chunk or scalar tail contains NaN, the helper falls back to the existing `arg_extreme_float` reducer for exact first-NaN stickiness.

## Same-worker benchmark

Command:

```bash
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib bench_argmax_f32_dense_vs_boxed \
  --release -- --ignored --nocapture
```

Worker: `hz1`
Benchmark: `argmax f32 [256,32768] axis1`

| Version | Dense time | Boxed time | Dense vs boxed |
| --- | ---: | ---: | ---: |
| Baseline `13440494` | 6.3343 ms | 54.9051 ms | 8.67x |
| SIMD final | 3.8376 ms | 52.5448 ms | 13.69x |

Dense path delta: `6.3343 / 3.8376 = 1.65x`.

Score: `Impact 1.65 * Confidence 0.95 / Effort 0.70 = 2.24`, keep.

## Behavior proof

Golden output digest:

```text
9d58e890fff3ceeba4962617ae365536954e31e42aab7ba4d05f5160e9698e2f
```

Focused proof command:

```bash
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib \
  argmax_argmin_contiguous_simd_float_matches_generic_and_golden -- --nocapture
```

Result: passed, 1 test.

Broader argmin/argmax command:

```bash
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib argmax_argmin -- --nocapture
```

Result: passed, 4 tests.

Covered parity cases:

- F32 and F64 dense outputs match boxed generic outputs.
- Argmax and argmin both covered.
- Non-vector-tail row lengths (`73`, `67`) exercise vector chunks plus scalar tail.
- Duplicate extrema preserve first occurrence.
- `+0.0` / `-0.0` ties preserve first occurrence.
- Signed NaNs and later competing finite/NaN values preserve first-NaN stickiness via scalar fallback.

Isomorphism notes:

- Ordering: unchanged for strided rows and all non-F32/F64 dtypes; contiguous F32/F64 uses strict comparison masks matching scalar replacement.
- Tie-breaking: earliest index preserved in vector lanes, horizontal reduction, and tail handling.
- Floating-point: no arithmetic introduced; only IEEE comparisons and `is_nan` checks. F32 values are not widened in the SIMD comparison path, but comparisons are order-equivalent to the previous exact `f64::from(f32)` widening for all non-NaN F32 values; NaN rows fall back to scalar.
- RNG: none.

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-lax/src/tensor_ops.rs
git diff --check -- crates/fj-lax/src/tensor_ops.rs
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-lax --lib
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fj-lax --lib -- -D warnings -A clippy::doc_lazy_continuation -A clippy::needless_range_loop
```

`cargo clippy -j 1 -p fj-lax --lib -- -D warnings` fails before and after this change on pre-existing `crates/fj-lax/src/linalg.rs` lints:

- `clippy::doc_lazy_continuation` at `linalg.rs:590..603`.
- `clippy::needless_range_loop` at `linalg.rs:637`, `689`, `3473`, `6139`, `6439`.

The same strict clippy failure was reproduced from the pre-change baseline worktree at `13440494`.

UBS:

```bash
ubs crates/fj-lax/src/tensor_ops.rs
```

Result: nonzero due existing broad inventory in this large file. UBS subchecks reported formatting clean, no clippy warnings/errors, cargo check clean, and tests build clean.
