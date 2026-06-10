# frankenjax-lu4yw pass214: F64 row-broadcast compare fast path

## Target

Ready perf bead: `frankenjax-lu4yw` (`[perf][no-gaps] fj-core/fj-lax word-native Bool mask pipeline`).

Profile-backed rows came from the existing `fj-lax` Criterion suite on RCH worker
`vmi1227854`. The direct word-backed Bool storage route was tested first and rejected;
the kept lever attacks the same comparison bottleneck with a different structural
primitive: a row-major rank-2-by-rank-1 F64 broadcast compare loop that bypasses the
generic broadcast odometer.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-lu4yw-baseline1 cargo bench -j 1 -p fj-lax --bench lax_baseline -- 'eval/(lt_same_shape_1024x1024_f64|lt_broadcast_row_1024x1024_f64|select_1024x1024_f64|reduce_and_64k_bool_vec)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Rows:

- `eval/select_1024x1024_f64`: `[645.58 us 687.53 us 724.45 us]`
- `eval/lt_same_shape_1024x1024_f64`: `[451.87 us 515.03 us 595.37 us]`
- `eval/lt_broadcast_row_1024x1024_f64`: `[1.5794 ms 1.6987 ms 1.8095 ms]`
- `eval/reduce_and_64k_bool_vec`: `[986.21 ns 1.0048 us 1.0212 us]`

## Rejected Subroutes

1. Direct word-backed Bool storage (`LiteralBufferStorage::BoolWords`) was proof-clean
   but rejected. Best same-worker row improved broadcast compare only to
   `[1.4390 ms 1.5058 ms 1.5672 ms]`, barely moved same-shape/reduce, and regressed
   `select` to `[802.96 us 875.23 us 918.96 us]`.
2. Branchless F64 `select` bit-copy was proof-clean but rejected below the score gate:
   `eval/select_1024x1024_f64` `[645.58 us 687.53 us 724.45 us]` to
   `[630.77 us 645.98 us 660.50 us]`.

Both source candidates were removed before this kept lever.

## Lever Kept

`crates/fj-lax/src/comparison.rs` now recognizes dense F64 broadcast comparisons for:

- rank-2 matrix `[rows, cols]` compared with rank-1 row `[cols]`
- rank-1 row `[cols]` compared with rank-2 matrix `[rows, cols]`

The fast path iterates row-major contiguous matrix chunks and zips each row with the
row vector. Other shapes fall through to the existing generic broadcast path.

## After Benchmark

Pre-rebase candidate command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-lu4yw-bcast-candidate2 cargo bench -j 1 -p fj-lax --bench lax_baseline -- 'eval/lt_broadcast_row_1024x1024_f64' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Result on the same worker:

- `eval/lt_broadcast_row_1024x1024_f64`: `[1.0595 ms 1.0980 ms 1.1349 ms]`

After rebasing onto `origin/main` (`07625f5c`), the exact-source command was rerun:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-lu4yw-bcast-postrebase cargo bench -j 1 -p fj-lax --bench lax_baseline -- 'eval/lt_broadcast_row_1024x1024_f64' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Final exact-source result on the same worker:

- `eval/lt_broadcast_row_1024x1024_f64`: `[1.1557 ms 1.2140 ms 1.2622 ms]`

Delta:

- Median: `1.6987 ms -> 1.2140 ms` = `1.40x`
- Conservative interval: `1.5794 ms -> 1.2622 ms` = `1.25x`
- Score: `Impact 1.40 * Confidence 0.95 / Effort 0.6 = 2.22`

## Isomorphism

- Ordering: output remains the same row-major linear order as the generic broadcast
  visitor.
- Floating point: every element still calls the same `float_cmp(a, b)` closure with
  the same operand pair. No arithmetic, reassociation, mixed precision, FMA, or
  tolerance change was introduced.
- Ties and NaNs: IEEE comparison behavior is delegated unchanged to the existing
  primitive-specific closure. NaN, infinities, and signed zero are covered by the
  golden test.
- RNG: no RNG path is touched.
- Fallbacks: non-dense, non-F64, non-rank2-row-broadcast, and other broadcast shapes
  use the existing generic path. Zero-column shapes return the same empty Bool tensor
  instead of entering `chunks_exact(0)`.

## Golden Proof

Focused RCH proof:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- cargo test -j 1 -p fj-lax --lib f64_row_broadcast_compare_bit_identical_to_literal_path -- --nocapture
```

Result: `1 passed; 0 failed`.

Golden SHA-256:

```text
fd7293300699f850c5fd274a548f8ee215b9cc4f3b75e86cfbea2fa0e02e00c6
```

The test compares dense fast-path output to the existing Literal-backed generic
broadcast path for matrix-row and row-matrix orientations across `Eq`, `Ne`, `Lt`,
`Le`, `Gt`, and `Ge`.

## Validation

- `rustfmt --edition 2024 --check crates/fj-lax/src/comparison.rs`: passed
- `git diff --check`: passed
- RCH `cargo check -j 1 -p fj-lax --all-targets`: passed on `vmi1227854`
- RCH `cargo clippy -j 1 -p fj-lax --all-targets --no-deps -- -D warnings`: blocked by
  pre-existing fj-lax lint debt outside this patch (`cz0g0_f32accum_evidence.rs`,
  `arithmetic.rs`, `simd_exp.rs`, `threefry.rs`, `lib.rs`)
- `cargo fmt -p fj-lax --check`: blocked by pre-existing broad formatting drift outside
  this patch; touched `comparison.rs` passes direct rustfmt check

## Next Primitive

Do not repeat storage-only Bool packing or branchless select-only tweaks. After this
commit, reprofile and route deeper toward compare-to-consumer fusion where select or
reductions consume predicate structure without materializing the full Bool vector.
