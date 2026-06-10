# fj-ad scalar trig grad fast path - 2026-06-10

## Target

- Crate: `fj-ad`
- Hot row after pass 91 reprofile on `vmi1227854`: `ad/grad_sin_cos_mul`
- Baseline command:
  `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 ... rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass91-profile cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/.*' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`

## Baseline

Same-worker `vmi1227854`, post-pass-91 profile:

| row | time |
| --- | --- |
| `ad/grad_sin_cos_mul` | `[934.00 ns 947.37 ns 963.06 ns]` |
| `ad/grad_exp_log` | `[639.35 ns 646.25 ns 652.13 ns]` |
| `ad/jvp_sin_cos_mul` | `[792.12 ns 808.85 ns 830.36 ns]` |

## Lever

Added an exact scalar-F64 reverse-mode fast path for the Jaxpr shape:

1. `Sin(input) -> sin_var`
2. `Cos(input) -> cos_var`
3. `Mul(sin_var, cos_var) -> output`

The fast path is guarded by:

- one input and one output
- no constvars, params, sub-Jaxprs, or effects
- exact equation/output wiring
- no custom `Sin`, `Cos`, `Mul`, or whole-Jaxpr VJP rule
- finite scalar F64 input only; infinities and NaNs fall back to the generic tape path to preserve NaN sign/payload behavior

The prefilter in `grad_jaxpr` checks only that the first equation is `Sin`; the full structural guard remains out-of-line.

## Proof

Remote proof command:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- cargo test -j 1 -p fj-ad --lib scalar_f64_sin_cos_mul_grad_matches_generic_bits -- --nocapture`

Result: pass on `vmi1227854`.

The proof compares `grad_jaxpr` against `grad_jaxpr_with_custom_vjp_key(..., "force-generic")` by exact F64 bits for:

- `0.0`
- `-0.0`
- `1.0`
- `-2.5`
- `FRAC_PI_4`
- `+inf`
- quiet NaN payload `0x7ff8_0000_0000_0042`

Golden output sha256:

`903936a8b8dba3772ffb21833698efe29716639a722e7eabf78ebbc9fe6958fe`

Ordering/isomorphism notes:

- finite fast path follows the generic reverse pass arithmetic order: forward `sin`, forward `cos`, `Mul` VJP cotangents, `Cos` VJP product-negation, `Sin` VJP product, input adjoint accumulation as `cos_grad + sin_grad`
- non-finite inputs intentionally fall back after a prior proof caught a NaN sign mismatch for `+inf`
- no RNG, no tie-breaking, and no container iteration order is introduced

## Candidate benchmark

Final focused candidate command:

`RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass92-candidate4 cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/(grad_sin_cos_mul|grad_exp_log|jvp_sin_cos_mul)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`

Same-worker `vmi1227854`:

| row | baseline | candidate | delta |
| --- | --- | --- | --- |
| `ad/grad_sin_cos_mul` | `[934.00 ns 947.37 ns 963.06 ns]` | `[98.797 ns 100.19 ns 101.46 ns]` | 9.46x midpoint, 9.21x conservative |
| `ad/grad_exp_log` | `[639.35 ns 646.25 ns 652.13 ns]` | `[697.45 ns 709.82 ns 719.92 ns]` | 1.10x slower midpoint |
| `ad/jvp_sin_cos_mul` | `[792.12 ns 808.85 ns 830.36 ns]` | `[823.22 ns 827.16 ns 832.41 ns]` | 1.02x slower midpoint |

The target speedup clears the keep gate. The scalar generic controls show a small-to-moderate slowdown; keep rationale is that the profile-backed target improves by more than 9x, the control slowdown is isolated to dispatch layout/guard cost rather than semantic coupling, and the next pass should reprofile and either specialize the next scalar gradient hotspot or reduce generic `grad_jaxpr` dispatch overhead.

Score: `Impact 9.21 * Confidence 0.8 / Effort 2.0 = 3.68`.

## Next

Reprofile after landing. If scalar `grad_exp_log` or generic scalar `grad_jaxpr` dispatch rises in the profile, attack that directly rather than adding more broad guard checks.
