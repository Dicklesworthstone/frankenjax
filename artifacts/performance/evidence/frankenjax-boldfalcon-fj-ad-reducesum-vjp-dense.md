# fj-ad ReduceSum VJP Dense Scalar Cotangent

Date: 2026-06-09
Agent: BoldFalcon
Surface: `crates/fj-ad/src/lib.rs`

## Target

Profile-backed target: `cargo bench -p fj-ad --bench ad_baseline -- ad/grad_sum_x2_plus_x_1k --noplot`

The benchmarked graph is:

`x -> reduce_sum((x * x) + x)` for a dense F64 vector of length 1024.

## Lever

`ReduceSum` VJP with a scalar F64/F32 cotangent now builds the broadcast cotangent through dense `TensorValue` constructors instead of materializing `Vec<Literal>`.

This changes storage only. The logical gradient values, element order, and raw scalar bits are preserved.

## Baseline

Command:

`rch exec -- cargo bench -p fj-ad --bench ad_baseline -- --noplot`

RCH result: failed open locally because no worker was admissible.

Criterion row:

`ad/grad_sum_x2_plus_x_1k time: [7.9510 us 8.0242 us 8.1091 us]`

## After

Command:

`rch exec -- cargo bench -p fj-ad --bench ad_baseline -- ad/grad_sum_x2_plus_x_1k --noplot`

RCH result: failed open locally because no worker was admissible.

Criterion row:

`ad/grad_sum_x2_plus_x_1k time: [4.6457 us 4.6833 us 4.7279 us]`

Criterion delta:

`change: [-41.996% -41.039% -40.121%] (p = 0.00 < 0.05)`

Speedup from medians: `8.0242 / 4.6833 = 1.71x`.

Score: `Impact 1.71 * Confidence 0.95 / Effort 0.5 = 3.25`, keep.

## Isomorphism Proof

- Ordering: output tensor length and element order are unchanged; every element is still the repeated scalar cotangent in row-major storage.
- Tie-breaking: not applicable; `ReduceSum` VJP has no comparisons or tie routing.
- Floating point: no arithmetic is introduced. F64/F32 values are rebuilt with `from_bits` into dense storage, and `LiteralBuffer` materialization maps back through the same `Literal::from_f64` / `Literal::from_f32` bit path.
- RNG: not applicable.

Golden-output SHA-256:

`grad_sum_x2_plus_x_1k_golden_sha256 = 5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`

## Validation

- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`
- `cargo fmt -p fj-ad -- --check`
- `rch exec -- cargo check -p fj-ad --all-targets`
- `rch exec -- cargo test -p fj-ad reduce_sum_vjp -- --nocapture`
- `rch exec -- cargo test -p fj-ad grad_sum_x2_plus_x_1k_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fj-ad --lib`

`cargo clippy -p fj-ad --all-targets -- -D warnings` is currently blocked before `fj-ad` by peer/pre-existing dependency failures:

- `crates/fj-trace/src/lib.rs:1808` unused `num_spatial`
- `crates/fj-lax/src/arithmetic.rs` peer-owned batched matmul helpers trip `clippy::too_many_arguments`
- `crates/fj-lax/src/reduction.rs:136` requires `std::simd::num::SimdUint` for `Simd::cast`

UBS was run on `crates/fj-ad/src/lib.rs`; it exits nonzero on broad pre-existing test unwrap/assert/indexing inventory, while its fmt/clippy/build sections passed in the shadow workspace.
