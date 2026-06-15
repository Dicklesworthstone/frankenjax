# frankenjax-9zwwb - f32 GEMM production flat-packed route

Date: 2026-06-15
Agent: SilverMaple
Crate: `fj-lax`

## Lever

The production native-f32 GEMM path now routes large single-matrix f32 matmul
through the flat packed-B kernel with the existing row-split compute threading. The KC-blocked
macro-kernel remains available as the explicit `kcblocked` bench/proof mode, but
its production gate is empty until a fresh size band proves it wins.

This is one routing lever: keep the already bit-proven flat packed path on the
production hot route instead of selecting the slower/noisier KC-blocked branch.

## Profile And Baseline

Same-worker RCH helper profile on `vmi1149989`:

- Command: `rch exec -- cargo bench -p fj-lax --bench lax_baseline -- f32_gemm_2048 --warm-up-time 1 --measurement-time 3 --sample-size 10`
- `linalg/f32_gemm_2048_packed`: `214.56..222.48..229.89 ms`
- `linalg/f32_gemm_2048_register`: `656.63..751.78..845.00 ms`
- `linalg/f32_gemm_2048_kcblocked`: `228.42..269.36..312.97 ms`

Earlier same-worker production baseline before the route change:

- Command: `rch exec -- cargo bench -p fj-lax --bench lax_baseline -- f32_gemm_2048_production --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Worker: `vmi1149989`
- `linalg/f32_gemm_2048_production`: `110.43..112.92..115.19 ms`

Exact-source same-worker A/B after RCH restored the narrower final source:

- Old KC production gate command: `rch exec -- cargo bench -p fj-lax --bench lax_baseline -- f32_gemm_2048_production --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Worker: `vmi1227854`
- Temporary old gate: `linalg/f32_gemm_2048_production` `172.34..193.71..216.68 ms`

## Re-Benchmark

Final-source production candidate:

- Command: `rch exec -- cargo bench -p fj-lax --bench lax_baseline -- f32_gemm_2048_production --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Worker: `vmi1227854`
- `linalg/f32_gemm_2048_production`: `52.966..53.720..54.329 ms`

Result:

- Midpoint speedup: `193.71 / 53.720 = 3.61x`
- Conservative speedup: `172.34 / 54.329 = 3.17x`
- Score: Impact 5 x Confidence 4 / Effort 2 = 10.0

## Isomorphism Proof

- Ordering: each output element still accumulates `l` in strictly ascending order
  inside `matmul_2d_packed_row_block_f32`; row chunks are disjoint output slices.
- B packing: the final source does not change packed-B layout or copy order.
- Tie-breaking: not applicable.
- Floating point: no reassociation, no FMA substitution, no mixed precision, and no
  RNG. The route selects an existing f32-accumulation kernel with the same per-cell
  multiply/add sequence as the serial packed proof path.
- Golden sha256: `02399fb13d6e0643dc9d8ade2c1dd2ce7cb985e38dcd41513cf80e438c0e54c8`
  from `batched_matmul_2d_f32_native_golden_digest`.

## Validation

- `RCH_WORKER=vmi1149989 rch exec -- cargo test -p fj-lax batched_matmul_2d_f32 --lib -- --nocapture`
  - Passed 5 tests, including `batched_matmul_2d_f32_native_golden_digest` and
    `batched_matmul_2d_f32_in_matches_native_f32_bits`.
- `RCH_WORKER=vmi1149989 rch exec -- cargo check -p fj-lax --lib`
  - Passed.
- `RCH_WORKER=vmi1149989 rch exec -- cargo clippy -p fj-lax --lib -- -D warnings`
  - Passed on RCH worker `vmi1153651`.
- `RCH_WORKER=vmi1149989 rch exec -- cargo check -p fj-lax --benches`
  - Passed on RCH worker `vmi1149989`.
- `RCH_WORKER=vmi1149989 rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings`
  - Passed on RCH worker `vmi1149989`.
- `rustfmt --edition 2024 --check crates/fj-lax/src/tensor_contraction.rs crates/fj-lax/benches/lax_baseline.rs`
  - Passed.
- `git diff --check -- crates/fj-lax/src/tensor_contraction.rs crates/fj-lax/benches/lax_baseline.rs`
  - Passed.

Known workspace hygiene note: `cargo fmt -p fj-lax -- --check` still reports a
pre-existing formatting drift in `crates/fj-lax/src/lib.rs`, which is outside this
bead and was not changed here. `ubs` exits nonzero on broad pre-existing inventory
warnings/false positives in the touched files; its cargo check/clippy/test-build
subchecks were clean.
