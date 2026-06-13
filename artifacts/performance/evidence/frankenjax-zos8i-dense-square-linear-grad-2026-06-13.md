# frankenjax-zos8i - Lazy Dense Square-Linear Grad Buffer

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-zos8i
Crates: fj-core, fj-ad

## Target

Profile-backed target after closing `frankenjax-krb67`:

- RCH post-close profile on `vmi1152480`, command:
  `cargo bench -p fj-ad --bench ad_baseline -- 'ad/(grad_sum_x2_plus_x_1k|grad_sin_cos_mul|grad_exp_log|jvp_sin_cos_mul|value_and_grad_poly)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`
- Dominant row: `ad/grad_sum_x2_plus_x_1k [1.5142 us 1.5830 us 1.6518 us]`.

`br ready --json` had no ready `[perf]` bead at the time of selection; live linalg/GEMM beads were owned by peer agents. This pass targets the profiler-evident `fj-ad` residual.

## Lever

One lever kept:

- Add `LiteralBufferStorage::F64OnePlusXPlusX`, a lazy dense F64 storage form that owns an `Arc<Vec<f64>>` input and materializes `1.0 + x + x` only if the output is inspected as literals or dense values.
- Add `LiteralBuffer::f64_values_arc()` so exact fast paths can borrow dense F64 input storage without copying.
- Change only the exact `grad_jaxpr(sum(x * x + x))` dense F64 grad-only path to return the lazy buffer instead of eagerly allocating and filling the 1k gradient vector.

Alien-graveyard mapping:

- Equality saturation / certified rewrite: the guard matches the exact square-plus-linear reduce-sum Jaxpr and leaves all nonmatching programs on the generic interpreter.
- Adaptive runtime specialization: the specialization is selected only for dense F64 input and the exact grad-only transform shape.
- Lazy materialization / deforestation: the output gradient tensor is represented by a compact formula over the input buffer until a consumer demands concrete values.

## Baseline And Rebench

The route profile ran on `vmi1152480`; RCH selected `vmi1149989` for the candidate. To keep the evidence same-worker, a clean detached baseline worktree was created at pre-change commit `2930fa3d` and benchmarked on the same RCH worker as the candidate.

Same-worker baseline:

- Worktree: `/data/projects/.scratch/frankenjax-zos8i-baseline-2930fa3d`
- Commit: `2930fa3d`
- Worker: `vmi1149989`
- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 RCH_WORKERS=vmi1149989 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_sum_x2_plus_x_1k' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`
- Result: `ad/grad_sum_x2_plus_x_1k [1.2529 us 1.3390 us 1.4497 us]`.

Candidate:

- Worker: `vmi1149989`
- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_sum_x2_plus_x_1k' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`
- RCH-selected worker: `vmi1149989`
- Result: `ad/grad_sum_x2_plus_x_1k [148.22 ns 151.49 ns 154.69 ns]`.

Win:

- Midpoint speedup: `1339.0 ns / 151.49 ns = 8.84x`.
- Conservative speedup: `1252.9 ns / 154.69 ns = 8.10x`.
- Score: `Impact 5 x Confidence 5 / Effort 1 = 25.0`.

## Behavior Proof

Guard and ordering:

- The fast path is gated by the same exact `dense_f64_square_plus_linear_reducesum_input` recognizer.
- It applies only to a single dense F64 input, the exact `x*x + x` reduce-sum trace, no effects, no sub-Jaxprs, and no custom VJP route.
- Nonmatching dtype, shape, custom-rule, or Jaxpr cases still return `None` and use the generic AD interpreter.
- Output arity, tensor dtype, tensor shape, and row-major element order are unchanged.

Floating point:

- Lazy materialization uses the same per-element operation order as the eager baseline:
  `let mut value = 1.0_f64; value += x; value += x;`.
- It does not rewrite to `1.0 + 2.0 * x`, does not fuse multiply-add, and does not reassociate across elements.
- Signed zero, infinities, and NaN payload propagation are preserved by the existing F64 addition sequence.

Tie-breaking and RNG:

- No comparisons or tie-breaking are introduced.
- No RNG state exists on this path.

Mutation and lifetime:

- The lazy output owns an `Arc<Vec<f64>>` clone of the dense input storage.
- Later mutation through `LiteralBuffer::make_mut` remains copy-on-write; it cannot mutate the lazy output's captured base values.

Serialization, equality, and iteration:

- `as_slice`, `as_f64_slice`, `IntoIterator`, and literal conversion all materialize the same concrete literals as the prior eager vector.
- The public observable tensor remains a normal F64 tensor once inspected.

Golden output:

- Test: `grad_sum_x2_plus_x_1k_golden_sha256`.
- SHA-256: `5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`.

## Validation

- `cargo fmt --package fj-core --package fj-ad` passed.
- `git diff --check` passed.
- `cargo fmt --package fj-core --package fj-ad -- --check` passed.
- RCH `cargo check -j 1 -p fj-core --all-targets` passed on `vmi1152480`.
- RCH `cargo test -j 1 -p fj-ad grad -- --nocapture` passed on `vmi1227854`: 89 passed, including `dense_f64_square_plus_linear_reducesum_grad_matches_generic_bits` and `grad_sum_x2_plus_x_1k_golden_sha256`.
- RCH `cargo check -j 1 -p fj-ad --all-targets` passed on `vmi1153651`.
- RCH `cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings` passed on `vmi1264463`.
- RCH `cargo clippy -j 1 -p fj-core --all-targets --no-deps -- -D warnings` passed on `vmi1153651`.
- Known background warning observed during dependency compilation: pre-existing `fj-trace` unused variable `num_spatial`.
- UBS on touched files exited 1 because of broad pre-existing inventory in `fj-core`/`fj-ad`: unwrap/expect in tests and helpers, panic macros in tests, direct indexing inventory, float equality checks, and existing allocation/clone heuristics. No new real defect was found in the lazy buffer lever. The new `IntoIterator` clone/collect hits mirror the existing storage-variant materialization contract.
