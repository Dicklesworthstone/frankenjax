# frankenjax-1d6cp: dense unary exp thread threshold

Date: 2026-06-13
Agent: BeigeMouse
Pass: pass239
Scope: `fj-lax` dense unary elementwise path

## Target

No ready unclaimed `[perf]` bead was available. The previous progress entry routed
Winograd conv2d work into the f32 GEMM panel-packing surface, which overlaps
another agent's active GEMM claim. This pass instead used the profiler-evident
`eval/exp_512k_f64` dense unary scheduling hotspot.

One lever shipped: replace the old `1 << 18` dense unary parallel fan-out gate
with a shared `dense_unary_threads` grain of one million elements per thread.
This keeps medium tensors serial while preserving parallel execution for larger
dense unary transcendental tensors.

## Benchmarks

Same-worker RCH Criterion worker: `vmi1152480`.

Commands:

```text
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- eval/exp_512k_f64

RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- eval/exp_4m_f64
```

Baseline:

```text
eval/exp_512k_f64 time [3.5972 ms 4.2104 ms 4.8607 ms]
eval/exp_4m_f64   time [40.083 ms 44.808 ms 49.729 ms]
```

Candidate:

```text
eval/exp_512k_f64 time [3.4091 ms 3.6304 ms 3.8547 ms]
eval/exp_4m_f64   time [16.780 ms 17.732 ms 18.835 ms]
```

Median ratios:

```text
512k: 1.16x faster
4m:   2.53x faster
```

Score: `2.53 impact * 0.95 confidence / 0.50 effort = 4.81`.

## Isomorphism Proof

- Ordering: output indices are still written to the same positions and joined in
  chunk order by the same scoped-thread implementation.
- Floating point: every element still applies exactly one `f64::exp`, `f32::exp`,
  `f64::sin`, etc. No reductions, reassociation, or fused arithmetic are added.
- Tie-breaking: no comparisons or tie surfaces are introduced.
- RNG: no RNG is used.
- Fallbacks: scalar and boxed literal paths are unchanged. The lever only changes
  when the existing dense unary parallel fast path is used.

Golden proof test:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib \
  dense_f64_exp_unary_grain_preserves_output_bits_and_golden -- --nocapture
```

Result:

```text
running 1 test
dense f64 exp unary-grain golden digest: e35daabbc391dc11d594a4c87f3bad3c885ceaeb9bb9c1a06c12985f6b8c09f3
test arithmetic::tests::dense_f64_exp_unary_grain_preserves_output_bits_and_golden ... ok
test result: ok. 1 passed; 0 failed
```

Pinned golden SHA:

```text
e35daabbc391dc11d594a4c87f3bad3c885ceaeb9bb9c1a06c12985f6b8c09f3
```

The proof compares dense F64 output bits against the boxed `Literal::F64` path
for normal values, signed zero, infinities, and a pinned NaN payload.

## Validation

Passed:

```text
git diff --check
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-lax --lib
ubs crates/fj-lax/src/arithmetic.rs
```

`ubs` exits nonzero from pre-existing file-wide panic/unwrap/direct-indexing
inventory, but its built-in formatting, clippy, cargo check, test-build,
cargo-audit, and cargo-deny sections were clean.

Blocked by pre-existing lint debt outside this lever:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo clippy -j 1 -p fj-lax --lib --no-deps -- -D warnings
```

Failures:

```text
crates/fj-lax/src/linalg.rs:680-693 clippy::doc_lazy_continuation
crates/fj-lax/src/linalg.rs:727,779,3563,6229,6529 clippy::needless_range_loop
crates/fj-lax/src/reduction.rs:1139 clippy::too_many_arguments
crates/fj-lax/src/tensor_ops.rs:5939 clippy::ptr_arg
```

## Decision

Ship. The large dense unary target improves by `2.53x` with a small positive
`512k` result and exact golden-bit parity. Reprofile after landing; do not repeat
thread-threshold tuning unless a fresh profile shows a new scheduling floor.
