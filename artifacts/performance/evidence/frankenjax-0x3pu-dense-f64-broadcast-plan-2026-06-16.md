# frankenjax-0x3pu: dense F64 broadcast typed-slot plan

Date: 2026-06-16
Bead: `frankenjax-0x3pu`
Crate: `fj-interpreters`
Touched source: `crates/fj-interpreters/src/lib.rs`

## Profile-backed target

After the reshape and transpose typed-slot passes, one-equation dense F64
`Primitive::BroadcastInDim` bodies still fell through to `run_dense_env_into`
and `eval_primitive`. The repeated-body cost included string parsing for
`shape` and `broadcast_dimensions`, generic primitive dispatch, and fj-lax's
per-element broadcast odometer setup.

Baseline was local because the 2026-06-16 user override marked `ts1` offline and
forbade waiting on remote RCH:

```text
cargo test -j 1 -p fj-interpreters bench_dense_f64_broadcast_in_dim_plan_overhead -- --ignored --nocapture
BENCH dense-f64 broadcast_in_dim [8]->[8,8] 1000000 evals:
GENERIC 11263.3ns/eval -> PLANNED 10935.9ns/eval = 1.03x
sha256=0633011b20168960838076a4296d2de2081b15e686b9af3b45f7ee835b8e7778
```

The "PLANNED" arm above was the current dense-plan fallback route.

## One lever

Add `DenseF64BroadcastInDimPlan` for exactly one effect-free, sub-jaxpr-free
`Primitive::BroadcastInDim` equation over one variable input whose output is the
Jaxpr output.

The plan pre-parses target `shape` and optional `broadcast_dimensions` once.
Runtime handles the common bias-style dense F64 rank-1 -> rank-2 trailing-axis
broadcast (`broadcast_dimensions=[1]`, or the default rank-1-to-rank-2 mapping).
When the input length equals the target column count, each output row is emitted
with `extend_from_slice(src)`. If the input dimension is `1`, the scalar element
is repeated across the full output.

Unsupported cases fall through to the existing generic interpreter:

- scalar input,
- non-F64 tensors,
- non-rank-1 input or non-rank-2 output,
- empty output,
- non-trailing broadcast dimensions,
- incompatible dimensions,
- malformed shape or dimension params,
- effectful or multi-equation Jaxprs.

## Re-benchmark

Focused local post-lever benchmark:

```text
cargo test -j 1 -p fj-interpreters bench_dense_f64_broadcast_in_dim_plan_overhead -- --ignored --nocapture
BENCH dense-f64 broadcast_in_dim [8]->[8,8] 1000000 evals:
GENERIC 10867.5ns/eval -> PLANNED 2889.1ns/eval = 3.76x
sha256=0633011b20168960838076a4296d2de2081b15e686b9af3b45f7ee835b8e7778
```

Accepted comparison: old planned fallback `10935.9ns/eval` to new planned route
`2889.1ns/eval` = `3.79x`.

Hyperfine wrapper timing for the focused local benchmark:

```text
hyperfine --warmup 0 --runs 3 \
  'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-broadcast cargo test -j 1 -p fj-interpreters bench_dense_f64_broadcast_in_dim_plan_overhead -- --ignored --nocapture'

Time (mean +- sigma): 81.609 s +- 2.789 s
Range: 79.464 s ... 84.762 s
```

Score: `3.6 = Impact 4.0 x Confidence 0.90 / Effort 1.0`.

## Isomorphism proof

Focused proof:

```text
cargo test -j 1 -p fj-interpreters dense_f64_broadcast_in_dim_plan_matches_generic_and_golden -- --nocapture
test tests::dense_f64_broadcast_in_dim_plan_matches_generic_and_golden ... ok
```

Golden output SHA-256:

```text
0633011b20168960838076a4296d2de2081b15e686b9af3b45f7ee835b8e7778
```

Checklist:

- Equation order unchanged: the plan only accepts one broadcast equation.
- Output ordering unchanged: the single equation output must equal the Jaxpr
  output list.
- Floating-point behavior unchanged: broadcast performs no arithmetic, no
  reassociation, no FMA, and no rounding.
- Element order unchanged: for the `[8]->[8,8]` proof case, row-major output is
  eight consecutive copies of the input row, exactly matching fj-lax's
  `broadcast_replicate` mapping for `broadcast_dimensions=[1]`.
- Tie-breaking unchanged: no comparisons or tie choices exist.
- RNG unchanged: no randomness.
- Error behavior preserved: only valid dense F64 rank-1 -> rank-2 trailing-axis
  broadcasts take the planned route; unsupported shapes, dtypes, ranks, empty
  output, dimensions, and malformed cases fall back to `eval_primitive`.
- Safety unchanged: safe Rust only; no `unsafe`.

## Validation

Passed locally with crate-scoped commands:

```text
cargo fmt -p fj-interpreters
cargo fmt --check -p fj-interpreters
cargo test -j 1 -p fj-interpreters dense_f64_broadcast_in_dim_plan_matches_generic_and_golden -- --nocapture
cargo test -j 1 -p fj-interpreters bench_dense_f64_broadcast_in_dim_plan_overhead -- --ignored --nocapture
hyperfine --warmup 0 --runs 3 'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-broadcast cargo test -j 1 -p fj-interpreters bench_dense_f64_broadcast_in_dim_plan_overhead -- --ignored --nocapture'
cargo check -j 1 -p fj-interpreters --all-targets
cargo clippy -j 1 -p fj-interpreters --all-targets -- -D warnings
git diff --check -- crates/fj-interpreters/src/lib.rs
```

`ubs crates/fj-interpreters/src/lib.rs` remained nonzero from the existing
file-wide heuristic inventory, including existing unwrap/panic/index warnings
and a false positive around a local variable named `decode`. Its embedded
fmt/clippy/check/test-build/audit/deny sections were clean, and it reported no
unsafe blocks.
