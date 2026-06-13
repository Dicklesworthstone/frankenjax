# frankenjax-srk4i: scalar polynomial value_and_grad fast path

Date: 2026-06-13

Bead: `frankenjax-srk4i`

Pass: 101, `/repeatedly-apply-skill` + `/extreme-software-optimization`

## Target

Profiler-backed target: `fj-ad` Criterion lane `ad/value_and_grad_poly`.

Pre-change routing profile after `frankenjax-kg68b` showed this lane still hot:

- Worker: `vmi1152480`
- Command: `rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/value_and_grad_poly' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot`
- Route-only interval: `[148.27 ns 157.61 ns 165.63 ns]`

Focused pre-edit baseline also confirmed the hotspot:

- Worker: `vmi1152480`
- Interval: `[151.83 ns 167.44 ns 187.98 ns]`

## Lever

One lever: add an exact certified scalar F64 recognizer for:

```text
x2 = x * x
x3 = x2 * x
s  = x3 + x2
y  = s + x
```

and compute the primal plus reverse-mode scalar adjoint directly in
`value_and_grad_jaxpr_inner_with_custom_vjp_key`.

Edited code:

- `crates/fj-ad/src/lib.rs:497` exact pattern recognizer
- `crates/fj-ad/src/lib.rs:729` fast path implementation
- `crates/fj-ad/src/lib.rs:10974` dispatch insertion
- `crates/fj-ad/src/lib.rs:20067` bitwise generic-path proof test

## Acceptance Benchmark

Matched parent baseline:

- Checkout: `/data/projects/.scratch/frankenjax-srk4i-baseline-9a900555`
- Parent commit: `9a900555`
- Worker: `vmi1264463`
- Command: `RCH_WORKERS=vmi1264463 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/value_and_grad_poly' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot`
- Interval: `[288.88 ns 342.66 ns 413.92 ns]`

Final candidate:

- Checkout: `/data/projects/.scratch/frankenjax-beigemouse-tnfjw-495d6845`
- Worker: `vmi1264463`
- Command: `RCH_WORKERS=vmi1264463 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/value_and_grad_poly' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot`
- Interval: `[130.06 ns 140.05 ns 154.21 ns]`

Delta:

- Midpoint speedup: `342.66 / 140.05 = 2.45x`
- Conservative bound: `288.88 / 154.21 = 1.87x`
- Score: `Impact 8 * Confidence 4 / Effort 2 = 16.0`
- Decision: keep

## Isomorphism Proof

Recognizer guards preserve semantics:

- Exactly one input, no constvars, no top-level effects, exactly one output.
- Exactly four equations and one output per equation.
- Each equation must have empty params, sub-jaxprs, and effects.
- Primitive sequence and input order must be exactly `Mul(x,x)`, `Mul(x2,x)`, `Add(x3,x2)`, `Add(sum,x)`.
- Intermediate and output vars must be pairwise distinct.
- Custom VJP key must be absent.
- Custom Add/Mul VJP registrations and custom JAXPR VJP registration must be absent.

Floating point behavior:

- Primal arithmetic order is unchanged: `(x * x)`, then `x2 * x`, then `x3 + x2`, then `sum + x`.
- Reverse arithmetic order mirrors the generic reverse pass for this graph:
  `x2_adj += x3_adj * x`, `x_adj += x3_adj * x2`, then the square cotangent is added twice to `x_adj`.
- No RNG, iteration ordering, tie-breaking, or hash-order surface is involved.

Golden-output SHA-256:

- Test: `scalar_f64_polynomial_value_and_grad_matches_generic_bits`
- Inputs: `0.0`, `-0.0`, `1.0`, `-2.5`, `f64::MAX`, `inf`, `-inf`, quiet NaN payload `0x7ff8_0000_0000_0042`
- Comparison: direct fast path, public fast path, and forced generic path output/gradient bits.
- Golden digest: `572f0ea8b946b740462423e02a21ec912bd896e5796d3ad4e38a14b916301afb`

Proof command:

```bash
RCH_WORKERS=vmi1264463 rch exec -- cargo test -j 1 -p fj-ad scalar_f64_polynomial_value_and_grad_matches_generic_bits -- --nocapture
```

Result:

- Worker: `vmi1264463`
- `1 passed; 0 failed; 371 filtered out`

## Validation

Passed:

```bash
cargo fmt --package fj-ad -- --check
git diff --check
rch exec -- cargo check -j 1 -p fj-ad --all-targets
rch exec -- cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
```

Ambient warnings observed while compiling dependencies:

- `crates/fj-lax/src/lib.rs:3662` non-snake-case `eval_reduce_window_iN_sum_sat`
- `crates/fj-trace/src/lib.rs:1808` unused `num_spatial`

UBS:

```bash
ubs crates/fj-ad/src/lib.rs
```

Result: exit 1 due pre-existing broad `fj-ad` inventory: panic macros, unwrap/expect, direct indexing, clone/allocation inventories. UBS subchecks reported formatting clean, no clippy warnings/errors, cargo check clean, tests build clean, no unsafe blocks, no hardcoded secrets, and no TODO/FIXME/HACK markers.
