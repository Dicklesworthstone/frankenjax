# frankenjax-prbo6 rejection: generic bit-packed Bool comparison output

Date: 2026-06-10
Agent: BoldFalcon
Bead: frankenjax-prbo6
Scope: `fj-core` Bool storage plus `fj-lax` comparison output

## Profile-backed target

The bead targets Bool-producing comparison/predicate output. Prior campaign notes identified Bool output as the remaining floor after dense i64/f64 comparison input paths: comparison still emits one bool value per element and later bool consumers must scan that output.

## Lever tested

Add a `LiteralBufferStorage::BoolBits` backing (`Vec<u64>`, one bit per bool) plus an exact-size iterator constructor, then route dense same-shape/scalar comparison fast paths through the packed constructor.

This was one lever: generic packed Bool output storage for comparison fast paths. Broadcast visitor output was intentionally left dense because the current visitor is push-based, not exact-size iterator based.

## RCH baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
CARGO_TARGET_DIR=target/rch-boldfalcon-prbo6-compare-baseline \
rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  'eval/(lt_64k_i64_vec|lt_same_shape_1024x1024_f64|lt_broadcast_row_1024x1024_f64)' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Worker: `vmi1227854`

Rows:

- `eval/lt_same_shape_1024x1024_f64`: `[396.55 us 435.75 us 484.85 us]`
- `eval/lt_broadcast_row_1024x1024_f64`: `[1.6145 ms 1.6522 ms 1.6969 ms]`
- `eval/lt_64k_i64_vec`: `[36.891 us 37.991 us 39.877 us]`

## Candidate re-benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
CARGO_TARGET_DIR=target/rch-boldfalcon-prbo6-compare-candidate \
rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  'eval/(lt_64k_i64_vec|lt_same_shape_1024x1024_f64|lt_broadcast_row_1024x1024_f64)' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Worker: `vmi1227854`

Rows:

- `eval/lt_same_shape_1024x1024_f64`: `[1.1145 ms 1.1784 ms 1.2508 ms]`
- `eval/lt_broadcast_row_1024x1024_f64`: `[1.4521 ms 1.5885 ms 1.7603 ms]`
- `eval/lt_64k_i64_vec`: `[26.039 us 29.164 us 34.924 us]`

The same-shape f64 row regressed from `435.75 us` to `1.1784 ms` (0.37x throughput, 2.70x slower). The smaller i64 row improved from `37.991 us` to `29.164 us` (1.30x), but the broad comparison/mask target is dominated by the large contiguous f64 path and the regression fails the keep gate.

## Proof run before rejection

Commands:

```bash
RCH_REQUIRE_REMOTE=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
CARGO_TARGET_DIR=target/rch-boldfalcon-prbo6-core-test \
rch exec -- cargo test -j 1 -p fj-core packed_bool_literal_buffer_preserves_value_api_and_cow -- --nocapture
```

Result: passed, 1 test.

```bash
RCH_REQUIRE_REMOTE=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
CARGO_TARGET_DIR=target/rch-boldfalcon-prbo6-lax-test \
rch exec -- cargo test -j 1 -p fj-lax comparison::tests:: -- --nocapture
```

Result: passed, 16 tests, 1 ignored.

## Isomorphism proof

- Ordering preserved: candidate applied the same comparison closure in the same row-major element order.
- Tie-breaking unchanged: bool output carries only true/false values; no tie-breaking semantics were introduced.
- Floating-point: candidate used the same `f64`/widened-`f32` comparisons and did not reorder FP operations.
- RNG: not applicable; comparison paths use no RNG.
- Golden output after rejection restore: product source diff is empty.

Golden SHA-256:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  git diff -- crates/fj-core/src/lib.rs crates/fj-lax/src/comparison.rs
```

## Decision

Rejected. Score: 0.0. The candidate is proof-clean but fails the performance gate due the large same-shape f64 regression. No product source was retained.

The failure means the next primitive should not be another generic bool iterator packer. Attack a word-native predicate pipeline instead: compare kernels that emit `u64` mask words directly via SIMD `to_bitmask`/chunk builders, plus bool reductions using `popcount` and select/where paths that consume bit words without materializing dense bools. Target ratio: 4x+ for compare-to-reduce and predicate mask chains.
