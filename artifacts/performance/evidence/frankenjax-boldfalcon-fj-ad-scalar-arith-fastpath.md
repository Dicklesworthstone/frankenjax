# fj-ad Scalar F64 AD Arithmetic Fast Path

Pass: BoldFalcon profile-backed optimization pass 2
Commit base: f020aac6
Target: `value_and_grad_runtime/shared/deep_100_nodes`

## Baseline

Command:

```bash
rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad_runtime/shared/deep_100_nodes --noplot
```

Clean worktree: `/data/projects/.scratch/frankenjax-boldfalcon-whilef32-final-20260609`
Worker: `vmi1227854`

Result:

```text
time: [82.183 us 83.279 us 84.363 us]
```

## Candidate

Command:

```bash
rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad_runtime/shared/deep_100_nodes --noplot
```

Worktree: `/data/projects/frankenjax`
Worker: `vmi1227854`

Result:

```text
time: [76.719 us 78.086 us 79.465 us]
```

Median speedup: `83.279 / 78.086 = 1.066x`.

Score: `Impact 1.066 * Confidence 0.95 / Effort 0.50 = 2.03`.

## Lever

`fj-ad::value_add` and `fj-ad::value_mul` now bypass `fj-lax::eval_primitive`
for scalar `Value::Scalar(Literal::F64Bits(_))` pairs. All tensor, mixed-dtype,
integer, half, f32, and complex cases still use the existing evaluator.

## Isomorphism Proof

- Ordering: unchanged. The helpers replace one scalar primitive application
  with the same single scalar `+` or `*` at the same call site.
- Tie-breaking: not applicable to add/mul.
- Floating-point: scalar F64 inputs use `f64::from_bits(lhs) op
  f64::from_bits(rhs)` and `Literal::from_f64`, exactly matching the generic
  evaluator's F64 scalar path.
- RNG: no RNG surface.
- Golden SHA: `grad_sum_x2_plus_x_1k_golden_sha256 =
  5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`.

## Validation

```bash
rch exec -- cargo test -p fj-ad scalar_f64_ad_arithmetic_matches_eval_primitive_bits -- --nocapture
rch exec -- cargo test -p fj-ad scalar_non_f64_ad_arithmetic_stays_on_eval_path -- --nocapture
rch exec -- cargo test -p fj-ad grad_sum_x2_plus_x_1k_golden_sha256 -- --nocapture
cargo fmt -p fj-ad -- --check
git diff --check -- crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-boldfalcon-fj-ad-scalar-arith-fastpath.md
rch exec -- cargo check -p fj-ad --all-targets
rch exec -- cargo test -p fj-ad --lib
rch exec -- cargo clippy -p fj-ad --all-targets -- -D warnings
rch exec -- cargo clippy -p fj-ad --all-targets -- -D warnings -A unused-variables -A clippy::too_many_arguments -A clippy::manual_is_multiple_of -A clippy::needless_range_loop -A clippy::useless_vec
ubs crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-boldfalcon-fj-ad-scalar-arith-fastpath.md
```

- Focused tests: passed.
- Formatting and whitespace checks: passed.
- `cargo check -p fj-ad --all-targets`: passed on `vmi1227854` with the
  pre-existing `fj-trace` unused-variable warning.
- `cargo test -p fj-ad --lib`: passed locally via RCH fail-open, 361 tests.
- Strict clippy failed before this patch's new code due pre-existing dependency
  and crate lint debt:
  `fj-trace::num_spatial`, `fj-lax` `too_many_arguments`,
  `fj-interpreters` `manual_is_multiple_of`, and older `fj-ad`
  `manual_is_multiple_of` / `needless_range_loop` / `useless_vec`.
- Adjusted clippy with only those known pre-existing lint classes allowed:
  passed on `vmi1227854`.
- UBS exited nonzero on the longstanding whole-file inventory
  (unwrap/expect/panic/assert/indexing counts in tests and existing code).
  UBS section 12 reported formatting clean and no clippy warnings/errors;
  section 13 reported cargo check and test builds clean.
