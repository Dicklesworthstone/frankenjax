# frankenjax-wtrks - staging single-equation execute shortcut

Date: 2026-06-13
Agent: BeigeMouse
Scope: `fj-interpreters`

## Target

Post-pass243 RCH profile on `vmi1152480` ranked the staging full pipeline row as the largest
remaining `fj-interpreters` row:

```text
staging/full_pipeline/neg_mul_mixed [2.2198 us 2.2971 us 2.3796 us]
```

Focused pre-edit baseline via RCH on `vmi1227854`:

```text
staging/full_pipeline/neg_mul_mixed [1.1811 us 1.2176 us 1.2547 us]
```

## Lever

Add one execution shortcut in `execute_staged` for the exact single-unknown-output,
single-unknown-equation staging shape produced by the mixed `neg_mul` pipeline. The shortcut
resolves residual and dynamic inputs in the same `jaxpr_unknown.invars` prefix order and delegates
the equation body to `eval_equation_outputs_from_resolved`.

All other staged programs fall through to the existing generic `eval_jaxpr_with_consts` path.

## Benchmark

Final exact-source benchmark via RCH on same worker as the ranked profile row:

```text
worker: vmi1152480
before: staging/full_pipeline/neg_mul_mixed [2.2198 us 2.2971 us 2.3796 us]
after:  staging/full_pipeline/neg_mul_mixed [892.73 ns 919.79 ns 953.87 ns]
ratio:  2.50x by mean
score:  4.75 = 2.50 impact * 0.95 confidence / 0.50 effort
```

After rebasing onto `origin/main` tip `bc929dd1`, the focused RCH confirmation row on
`vmi1152480` remained in the same band:

```text
after rebase: staging/full_pipeline/neg_mul_mixed [916.81 ns 937.70 ns 960.57 ns]
ratio:        2.45x by mean against the pre-edit same-worker profile row
```

Final push base was then rebased once more over `5735c6d2`, whose touched files were
`.beads/issues.jsonl` plus `fj-lax` cleanup files. No `fj-interpreters` source or benchmark file
overlapped this lever.

The focused pre-edit RCH baseline on `vmi1227854` and an intermediate same-worker after row on
`vmi1227854` also showed a win:

```text
before: [1.1811 us 1.2176 us 1.2547 us]
after:  [833.23 ns 856.20 ns 880.96 ns]
```

## Isomorphism Proof

- Ordering: residuals and dynamic args are read only by matching each input variable against
  `jaxpr_unknown.invars`; the residual prefix and dynamic suffix ordering is unchanged.
- Equation semantics: the shortcut calls the existing `eval_equation_outputs_from_resolved`
  evaluator, so primitive semantics and error translation stay anchored to the generic interpreter.
- Fallback surface: effects, constvars, multi-output programs, known outputs, mismatched arity, and
  non-single-equation unknown programs all fall through or return the same arity/missing-variable
  error class.
- Floating point: no new arithmetic is introduced; any FP operation remains inside the existing
  primitive evaluator.
- Tie-breaking: no sorting, hashing, or ordering choice is introduced.
- RNG: no RNG is used or reordered.

Golden output SHA-256 from focused RCH test:

```text
a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
```

The golden fixture covers staged `neg_mul` with known `5`, dynamic `3`, compares the fast path
result against full `eval_jaxpr`, and hashes `(staged.jaxpr_unknown, staged.residuals, result)`.

## Validation

Passed:

```text
rustfmt --edition 2024 --check crates/fj-interpreters/src/staging.rs
git diff --check
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- staging/full_pipeline/neg_mul_mixed
rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

Post-rebase focused checks also passed:

```text
rustfmt --edition 2024 --check crates/fj-interpreters/src/staging.rs
git diff --check HEAD~1..HEAD
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- staging/full_pipeline/neg_mul_mixed
```

Known unrelated blockers:

```text
rch exec -- cargo clippy -j 1 -p fj-interpreters --lib -- -D warnings
blocked before target crate by crates/fj-trace/src/lib.rs:1808 unused variable num_spatial

rch exec -- cargo clippy -j 1 -p fj-interpreters --lib --no-deps -- -D warnings
blocked by pre-existing crates/fj-interpreters/src/lib.rs:5466 clippy::question_mark

cargo fmt --check
blocked by pre-existing formatting drift outside this lever

ubs crates/fj-interpreters/src/staging.rs
nonzero from pre-existing file-wide inventory and a false positive on Instant::now in test logging;
built-in fmt/clippy/check/test-build/audit/deny sections were clean after the patch
```

## Decision

Keep and close `frankenjax-wtrks`.

Next primitive after landing: reprofile `fj-interpreters` with the full `pe_baseline` bench. Do not
add another exact staging-shape shortcut unless the fresh profile ranks it; the deeper route is a
small staged-program executor/cache that compiles reusable one- and two-equation residual programs
without re-entering the generic environment path.
