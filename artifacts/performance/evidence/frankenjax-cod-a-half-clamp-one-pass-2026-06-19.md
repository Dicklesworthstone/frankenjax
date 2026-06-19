# frankenjax-mcqr.109 - One-Pass Half Clamp Rejection

Date: 2026-06-19
Agent: cod-a / WildForge

## Lever

Attempted to replace the committed two-pass raw BF16/F16 clamp composition with
a one-pass fused helper:

- one output allocation, no `tmp` vector,
- generic scalar/tensor bound abstraction for low/high bounds,
- same JAX argument order: `max(lo, x)` then `min(hi, maxed)`,
- same signed-zero tie fixup and scalar fallback for NaN/F16 edge chunks.

The candidate was reverted after measurement. No arithmetic code from this
attempt is shipped.

## Commands

- Baseline:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-before`
- Candidate conformance:
  `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`
- Candidate bench:
  `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-after`
- Post-revert production conformance:
  `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`

RCH selected `ovh-a` for the baseline and the after run was pinned to
`RCH_WORKER=ovh-a`.

## Same-Worker Result

| Workload | Before two-pass mean | Candidate one-pass mean | Candidate / before | Outcome |
| --- | ---: | ---: | ---: | --- |
| `bf16_mixed_scalar_tensor_1m` | 2.3918 ms | 3.2624 ms | 1.36x slower | REJECT |
| `f16_mixed_scalar_tensor_1m` | 3.0263 ms | 4.4595 ms | 1.47x slower | REJECT |
| `bf16_tensor_tensor_tensor_1m` | 2.7423 ms | 3.2236 ms | 1.18x slower | REJECT |
| `f16_tensor_tensor_tensor_1m` | 3.4745 ms | 4.4271 ms | 1.27x slower | REJECT |

The candidate passed the narrow half-clamp tests while it existed:
4 passed, 0 failed, 2 ignored benchmark tests. The extra edge-matrix test was
part of the rejected candidate and was removed with the rejected code.

After reverting the candidate, the production half-clamp gate passed again:
3 passed, 0 failed, 2 ignored benchmark tests.

## JAX Context

Using the current mcqr.108 JAX CPU oracle means for context only
(`jax[cpu]` 0.10.1, 50 runs x 100 inner loops), the rejected candidate would
still be far slower than JAX:

| Workload | Candidate one-pass mean | JAX mean | Candidate/JAX |
| --- | ---: | ---: | ---: |
| `bf16_mixed_scalar_tensor_1m` | 3.2624 ms | 122.705 us | 26.59x slower |
| `f16_mixed_scalar_tensor_1m` | 4.4595 ms | 319.088 us | 13.98x slower |
| `bf16_tensor_tensor_tensor_1m` | 3.2236 ms | 148.870 us | 21.65x slower |
| `f16_tensor_tensor_tensor_1m` | 4.4271 ms | 196.938 us | 22.48x slower |

These ratios mix remote Rust with local JAX and are not release scorecard rows;
the same-worker Rust before/after regression is the hard reject criterion.

## Decision

Reject and revert. The temp vector and second pass are not the current dominant
cost in this implementation. The generic one-pass abstraction added enough
bound dispatch, extra vector setup, and duplicated half widen/round work to lose
all four measured rows.

Do not retry this generic `HalfClampBound` fused-helper shape. The next half
clamp attempt should either:

- compare raw half bits with a proven dtype-specific total-order/classification
  table that avoids f64 widen/round on normal finite lanes,
- specialize a single hot shape without enum-bound abstraction and prove it
  beats the current two-pass helper in same-worker Criterion, or
- fuse clamp with a producer/consumer so the intermediate result is eliminated
  across primitive boundaries rather than inside this already-optimized helper.
