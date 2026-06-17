# frankenjax-mcqr.71: exact Hessian for product of separable sums

## Target

After the square/self-mul exact Hessian slices, the next profile-backed fallback
shape is a product of two distinct scalar reductions:

```text
f(x) = reduce_sum(g(x)) * reduce_sum(h(x))
```

where `g` and `h` are independent unary elementwise chains over one finite dense
F64 tensor input. Before this pass, this shape fell through to central
differences.

## Baseline

Local release benchmark while ts1/rch remote was offline:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result before the production recognizer change:

```text
hessian general n=192: SERIAL 41.036ms  PARALLEL 6.998ms  speedup 5.86x
```

Accepted baseline for `hessian_jaxpr`: `6.998ms`.

## Candidate

The new recognizer traces two independent unary chains backward from the final
`Mul` inputs through their `ReduceSum` producers. It accepts only one dense F64
input, single-output/effect-free/param-free unary equations with known first and
second derivatives, two distinct reduction branches, no custom JVP/VJP rules,
and no unused or shared producer equations.

Formula:

```text
H_ij = g'_i h'_j + h'_i g'_j + [i == j] * (sum(h) g''_i + sum(g) h''_i)
```

Same benchmark after the change:

```text
hessian general n=192: SERIAL 38.511ms  PARALLEL 0.173ms  speedup 222.84x
```

Accepted comparison: `6.998ms -> 0.173ms` = `40.5x`.

Warmed hyperfine wrapper for the candidate command:

```text
Time (mean +/- sigma): 443.5 ms +/- 10.8 ms
Range: 431.3 ms ... 451.8 ms
```

Score: `4.5 = Impact 5.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused Hessian release proof:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad hessian --lib --release -- --nocapture
```

Result:

```text
7 passed; 0 failed; 2 ignored
```

New product golden SHA:

```text
dde9f0a7db4485d9dc9b6aaee09b7d179772c978ebbde7b024feff64deef46ad
```

Existing goldens preserved:

```text
square/self-mul exact: 8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0
separable diagonal:   7a42b4e6a4b18cf77a7efcf248f694db80fe7b76ea40488f73210b4920e12764
```

## Isomorphism

- The accepted pattern uses the exact analytic Hessian for a product of two
  separable sums.
- Branch tracing rejects shared producers, unused equations, distinct-variable
  self-mul ambiguity, params/effects/sub-jaxprs, unsupported primitives, custom
  JVP/VJP rules, non-F64 inputs, non-tensor inputs, and nonfinite intermediates.
- Output shape and order remain row-major `[input_dim, input_dim]`.
- Existing square/self-mul and central-difference fallback behavior remains
  covered by tests.
- No tie-breaking surface, no RNG, no unsafe code, and no C BLAS/LAPACK/XLA
  linkage.

## Validation

Passed:

```text
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
cargo test -j 1 -p fj-ad hessian --lib --release -- --nocapture
cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
cargo check -j 1 -p fj-ad --all-targets
cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings
hyperfine --warmup 1 --runs 3 'CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture'
```

UBS on `crates/fj-ad/src/lib.rs` remained nonzero from the existing large-file
heuristic inventory. Its embedded formatter, clippy, cargo check, test-build,
audit, and deny sections were clean and it reported no unsafe blocks.
