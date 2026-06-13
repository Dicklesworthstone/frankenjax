# frankenjax-gbtzl: scalar F64 exp-log gradient fast path

## Target

- Bead: `frankenjax-gbtzl`
- Surface: `crates/fj-ad/src/lib.rs`
- Profile source: RCH Criterion on `ovh-a`, 2026-06-13
- Hot row: `ad/grad_exp_log`

The focused pre-change baseline on `ovh-a` showed `ad/grad_exp_log` at
`[914.47 ns 995.82 ns 1.0546 us]`. Neighboring scalar fast-path rows from the
same profiling pass were much lower (`grad_sin_cos_mul` around `65 ns`), so the
remaining cost was the generic reverse-mode tape/env path for the two-equation
scalar `Exp -> Log` Jaxpr.

## Lever

One source lever was kept: detect the exact scalar F64 `Exp(x) -> Log(exp_x)`
shape in `value_and_grad_jaxpr_inner_with_custom_vjp_key`, require built-in
`Exp`/`Log` VJP rules and no custom Jaxpr VJP, and compute the same forward and
reverse scalar operations without building the tape.

The fast path preserves the generic reverse control flow:

1. forward `exp(x)`
2. forward `log(exp_x)`
3. backward `1.0 / exp_x`
4. if that cotangent is exactly zero, skip the upstream `Exp` VJP just like
   `backward` does
5. otherwise recompute `exp(x)` for the `Exp` VJP and multiply

All non-matching shapes, non-F64 values, params, effects, sub-Jaxprs, custom
primitive VJPs, and custom Jaxpr VJPs fall back to the generic path.

## Benchmark

RCH worker: `ovh-a`

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE AGENT_NAME=BeigeMouse \
  rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_exp_log' \
  --sample-size 40 --measurement-time 3 --warm-up-time 1 --noplot
```

Results:

- Baseline: `ad/grad_exp_log [914.47 ns 995.82 ns 1.0546 us]`
- Candidate: `ad/grad_exp_log [94.222 ns 94.367 ns 94.549 ns]`
- Midpoint speedup: `10.55x`
- Conservative interval speedup: `9.67x`
- Score: `25.0` (`Impact 5 * Confidence 5 / Effort 1`)

## Isomorphism

- Ordering: exact two-equation pattern only; no equation reordering.
- Floating point: output and gradient bits are compared against
  `value_and_grad_jaxpr_with_custom_vjp_key(..., "force-generic")`.
- Edge cases: `+0.0`, `-0.0`, finite positives/negatives, `MIN_POSITIVE`,
  overflow-adjacent `709.0`, `+inf`, `-inf`, and a payloaded quiet NaN.
- Zero-cotangent branch: `+inf` preserves the generic backward skip, returning
  gradient `0.0` rather than evaluating `0.0 * inf`.
- RNG/tie-breaking: not present.
- Golden SHA-256:
  `e0b86a6503bf24506cc007fb3d75260ec216581ff9ccdfea749d7f0e6e59e909`

## Validation

- `cargo fmt --package fj-ad -- --check`: pass
- `git diff --check`: pass
- RCH `cargo test -j 1 -p fj-ad`: pass, `370 passed`, doctests pass
- RCH `cargo check -j 1 -p fj-ad --all-targets`: pass
  - Note: path dependency `fj-trace` still emits a pre-existing unused-variable
    warning.
- RCH `cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings`: blocked by
  pre-existing `fj-lax` clippy debt in `linalg.rs`, `reduction.rs`, and
  `tensor_ops.rs`.
- RCH `cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings`:
  pass for the touched `fj-ad` package.
- `ubs crates/fj-ad/src/lib.rs ...`: nonzero because UBS inventories the whole
  large `fj-ad` file and reports pre-existing test panic/unwrap/assert surfaces;
  its fmt, clippy, check, test-build, audit, and deny subchecks were clean.

## Decision

Keep. The lever is narrow, profile-backed, behavior-bit-golden, and clears the
Score >= 2.0 gate with a same-worker `10.55x` midpoint win.
