# frankenjax-aip01 -- exact staged-program cache

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-aip01
Crate: fj-interpreters
Lever: one-entry thread-local cache for successful effect-free staged programs.

## Profile-backed target

The target was selected from the post-`frankenjax-14w36` RCH profile on worker
`vmi1149989`:

```text
staging/full_pipeline/neg_mul_mixed [746.70 ns 785.49 ns 822.42 ns]
staging/full_pipeline/all_known_add2 [494.52 ns 516.91 ns 539.78 ns]
```

This was the largest remaining `fj-interpreters` `pe_baseline` row after the DCE
cache pass landed.

## Baseline

Same-worker acceptance baseline from the post-`frankenjax-14w36` profile:

```text
RCH worker: vmi1149989
staging/full_pipeline/neg_mul_mixed [746.70 ns 785.49 ns 822.42 ns]
```

Focused pre-edit routing baseline on another worker:

```text
RCH worker: vmi1152480
staging/full_pipeline/neg_mul_mixed [893.67 ns 914.18 ns 935.13 ns]
```

The acceptance ratio uses only the comparable `vmi1149989` row.

## Change

`stage_jaxpr_with_consts` now probes a thread-local cache before running partial
evaluation and residual program construction. The cache stores only successful
staging results for effect-free Jaxprs and is keyed by all observable staging
inputs:

- `Jaxpr::canonical_fingerprint()`
- exact structural `Jaxpr` equality
- exact const values
- exact unknown mask
- exact known values

The hit path returns a cloned `StagedProgram`; the miss path is the original
staging algorithm followed by insertion of the successful result.

Alien primitive: incremental/self-adjusting computation. The repeated structural
staging query is maintained as an exact memoized artifact, instead of rebuilding
the same residual program on every identical call.

## After

Final safety-patched code on the same worker as the profile-backed baseline:

```text
RCH worker: vmi1149989
staging/full_pipeline/neg_mul_mixed [179.06 ns 181.92 ns 185.24 ns]
```

Median ratio: `785.49 / 181.92 = 4.32x`

Score: `Impact 4.32 * Confidence 0.95 / Effort 0.50 = 8.20`

## Isomorphism proof

- Error behavior is unchanged: only successful `stage_jaxpr_with_consts` results
  are cached. Failed staging calls still run the original path and return the
  original error.
- Effect behavior is unchanged: Jaxprs with root effects, equation effects, or
  nested sub-Jaxpr effects are never stored. Effectful programs always miss.
- Cache identity is exact: a hit requires matching fingerprint, structural
  Jaxpr equality, const values, unknown mask, and known values.
- Public-field mutation cannot create a stale-fingerprint hit because the
  structural `Jaxpr` equality guard is checked after the fingerprint compare.
- Ordering is unchanged: cached `StagedProgram` is a clone of the prior miss-path
  output, so known Jaxpr, unknown Jaxpr, residuals, output mask, known outputs,
  and dynamic input order are identical.
- Numeric behavior is unchanged: staging does not execute primitive arithmetic,
  and `execute_staged` remains the execution path. FP operation order, tie
  behavior, NaN behavior, and RNG absence are unchanged.
- Mutation/alias behavior is unchanged: callers receive an owned cloned
  `StagedProgram`; the cached entry is not exposed for mutation.

## Golden proof

Command:

```text
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
```

Result: passed on RCH `vmi1149989`.

Golden SHA:

```text
a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
```

The focused test now exercises both the miss path and the cached second call,
then executes the cached staged program and verifies the same result.

## Validation

Passed:

```text
rustfmt --edition 2024 --check crates/fj-interpreters/src/staging.rs
git diff --check
rch exec -- cargo check -j 1 -p fj-interpreters --lib
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
```

Post-rebase confirmation:

```text
RCH vmi1149989: focused golden test passed, SHA a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
RCH vmi1152480: cargo check -j 1 -p fj-interpreters --lib passed
```

Blocked outside this lever:

```text
rch exec -- cargo clippy -j 1 -p fj-interpreters --lib -- -D warnings
```

On the rebased tree, the clippy command failed before reaching `fj-interpreters`
because dependency crate `fj-lax` has pre-existing lint debt in
`crates/fj-lax/src/linalg.rs`: `doc_lazy_continuation` at lines 680-693 and
`needless_range_loop` at lines 727, 779, 3563, 6229, and 6529.

UBS:

```text
ubs crates/fj-interpreters/src/staging.rs
```

UBS exited nonzero from broad file-wide heuristic inventory in this existing
test-heavy module, including a false positive treating `Instant::now()` timing
as security-token generation. Its built-in fmt, clippy, cargo check, test-build,
audit, deny, and resource-lifecycle sections were clean.
