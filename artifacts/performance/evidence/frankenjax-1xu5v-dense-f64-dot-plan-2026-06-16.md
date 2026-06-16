# frankenjax-1xu5v: dense F64 rank-1 dot typed-slot plan

## Target

`frankenjax-1xu5v` continues the compiled interpreter typed-slot frontier after
`frankenjax-0x3pu` shipped dense F64 full `reduce_sum`. A one-equation
`Primitive::Dot` body over dense rank-1 F64 tensors still fell through
`run_dense_env_into` and paid generic primitive dispatch on every repeated
evaluation.

## Baseline

RCH same-worker baseline on `vmi1293453`, before the dot plan existed and after
adding only the ignored benchmark harness:

```text
cargo test -p fj-interpreters bench_dense_f64_dot_plan_overhead -- --ignored --nocapture
BENCH dense-f64 dot [64] 1000000 evals: GENERIC 3256.5ns/eval -> PLANNED 3366.6ns/eval = 0.97x sha256=7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa
```

The "PLANNED" arm was the current `run_dense_plan_into` fallback route, not a
dot fast path.

## Lever

Add a narrow `DenseF64DotPlan` for exactly one effect-free, no-param,
sub-jaxpr-free `Primitive::Dot` equation whose inputs are variables and whose
single output is the Jaxpr output.

Runtime selection accepts only:

- two tensor inputs,
- both `DType::F64`,
- both dense typed storage,
- both rank 1,
- equal lengths.

Every other case falls through to the existing generic interpreter.

## After

RCH same-worker after benchmark on `vmi1293453`:

```text
cargo test -p fj-interpreters bench_dense_f64_dot_plan_overhead -- --ignored --nocapture
BENCH dense-f64 dot [64] 1000000 evals: GENERIC 3211.5ns/eval -> PLANNED 703.8ns/eval = 4.56x sha256=7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa
```

Accepted comparison: old planned fallback `3366.6ns/eval` to new planned route
`703.8ns/eval` = `4.78x`.

Score: `3.6 = Impact 4.0 x Confidence 0.90 / Effort 1.0`.

Post-rebase sanity on the same worker after `origin/main` advanced with the F32
reduce_sum plan:

```text
cargo test -p fj-interpreters bench_dense_f64_dot_plan_overhead -- --ignored --nocapture
BENCH dense-f64 dot [64] 1000000 evals: GENERIC 3297.4ns/eval -> PLANNED 676.5ns/eval = 4.87x sha256=7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa
```

Final post-rebase sanity after `origin/main` advanced again with dense
prod/max/min reduce planning, on RCH worker `vmi1149989`:

```text
cargo test -p fj-interpreters bench_dense_f64_dot_plan_overhead -- --ignored --nocapture
BENCH dense-f64 dot [64] 1000000 evals: GENERIC 2404.9ns/eval -> PLANNED 574.4ns/eval = 4.19x sha256=7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa
```

## Behavior Proof

Focused RCH proof:

```text
cargo test -p fj-interpreters dense_f64_dot_plan_matches_generic_sha256 -- --nocapture
test tests::dense_f64_dot_plan_matches_generic_sha256 ... ok
```

The focused proof also passed after the rebase.
After the final rebase, `rch` could not lease an admissible worker for the
focused proof and failed open locally; the same focused proof passed, and the
remote benchmark above reconfirmed the unchanged golden SHA.

Golden output SHA-256:

```text
7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa
```

Isomorphism checklist:

- Equation order unchanged: the plan only accepts a single dot equation.
- Output ordering unchanged: rank-1 dot produces one scalar output.
- Floating-point order unchanged: seed `0.0`, then `sum += lhs[i] * rhs[i]` for
  ascending `i`, matching `fj-lax` `dot_accumulate_contiguous`.
- No FMA or reassociation introduced.
- NaN and signed-zero behavior covered by the pinned golden fixture.
- Dtype, shape, arity, scalar, higher-rank, and non-dense cases fall through to
  the generic interpreter.
- No RNG or tie-breaking surface exists.
- Safe Rust only; no `unsafe`.
