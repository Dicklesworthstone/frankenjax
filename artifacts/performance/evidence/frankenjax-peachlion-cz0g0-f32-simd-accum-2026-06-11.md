# frankenjax-cz0g0: f32 dot_general native-SIMD accumulation

Date: 2026-06-11
Agent: PeachLion
Bead: frankenjax-cz0g0
Crate: fj-lax

## Target

Profile-backed target: dense f32 `dot_general` matmul through `general_real_tensordot`.

Baseline command:

```bash
rch exec -- cargo test -p fj-lax --release bench_f32_matmul_dot_general -- --ignored --nocapture
```

Baseline worker: `vmi1227854`

| Case | Baseline | Rejected scalar native f32 | Kept SIMD native f32 |
| --- | ---: | ---: | ---: |
| `[256,256] * [256,256]` | 2.1537 ms | 2.3538 ms | 1.7611 ms |
| `[512,128] * [128,512]` | 3.5652 ms | 4.3105 ms | 2.3797 ms |

Result: kept SIMD candidate is 1.22x and 1.50x faster than the same-worker baseline. The scalar native-f32 attempt was rejected because it regressed both benchmark rows.

Score: Impact 3.0 x Confidence 0.9 / Effort 1.0 = 2.7.

## Lever

`batched_matmul_2d_f32_in` now accumulates dense f32 matmul outputs in native f32 using safe Rust portable SIMD:

- full output columns use 16-wide `Simd<f32, 16>` accumulators;
- tail columns use the same scalar native-f32 order;
- batch and row partitioning are unchanged;
- each output lane folds `k` in ascending order.

This wires the cz0g0 native-f32 policy path into production instead of leaving it as evidence-only code.

## Behavior Proof

Ordering: output order remains row-major `[batch, row, col]`; thread splitting still partitions whole output rows.

Floating point: this is the intentional cz0g0 semantic change from f64 accumulation to native f32 accumulation for dense f32 matmul. Within the native-f32 path, every output element still uses ascending-k multiply-add order. Tests prove native f32 bit identity and bound the old f64-accum delta below `1e-3` relative error.

Tie-breaking and RNG: not applicable.

Golden output digest:

```text
02399fb13d6e0643dc9d8ade2c1dd2ce7cb985e38dcd41513cf80e438c0e54c8
```

Focused proof command:

```bash
rch exec -- env CARGO_TARGET_DIR=/tmp/frankenjax-peachlion-cz0g0-test-target cargo test -j 1 -p fj-lax f32 -- --nocapture
```

Remote worker: `vmi1227854`
Result: 41 passed, 0 failed, 33 ignored.

## Validation Notes

Passing:

- `rch exec -- cargo test -j 1 -p fj-lax --release bench_f32_matmul_dot_general -- --ignored --nocapture`
- `rch exec -- env CARGO_TARGET_DIR=/tmp/frankenjax-peachlion-cz0g0-test-target cargo test -j 1 -p fj-lax f32 -- --nocapture`
- `git diff --check`

Caveats:

- `cargo fmt -p fj-lax -- --check` fails on broad pre-existing formatting drift in `fj-lax` files outside this patch. The touched diff itself is whitespace-clean.
- `rch exec -- env CARGO_TARGET_DIR=/tmp/frankenjax-peachlion-cz0g0-clippy-target cargo clippy -j 1 -p fj-lax --all-targets -- -D warnings` ran remotely and failed on pre-existing lint debt outside this lever, including too-many-arguments helpers, `simd_exp` doc/precision lints, `reduction` manual `Option::zip`, `threefry` range-loop lints, and `lib.rs` `manual_is_multiple_of`.
- `ubs` on the three touched files exits nonzero because it inventories broad pre-existing panic/unwrap/direct-index patterns. Its embedded fmt, clippy, cargo check, test-build, audit, and deny subchecks were clean.
