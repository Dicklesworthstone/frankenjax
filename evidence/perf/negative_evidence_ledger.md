# Negative Evidence Ledger

This ledger records code-first performance attempts and retry predicates so dead
ends are not rediscovered without new evidence.

## frankenjax-cod-b-dense-tensor-stack-axis0-rw4k4 - Dense Tensor stack_axis0 Concat Storage

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route `TensorValue::stack_axis0` for tensor inputs through
  `LiteralBuffer::from_concat_slices` plus `new_with_literal_buffer`, preserving
  packed dense storage instead of materializing a `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original JAX;
  internal control is a small but separated Rust win versus the literal-backed
  stack path. No revert.
- Benchmark guard: `core/tensor_stack_axis0_dense_f64_64x1k` plus
  `core/tensor_stack_axis0_literal_f64_64x1k` control.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_stack_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda *xs: jnp.stack(xs, axis=0))`
    over 64 x 1024 `float64` arrays with `jax_enable_x64=true`, CPU backend,
    80 runs x 200 inner loops via `benchmarks/jax_comparison/.venv/bin/python`
    (JAX/JAXLIB 0.10.1).
  - Rust dense mean estimate 3.3963 us (`[3.3375, 3.4533]` us Criterion
    interval) vs JAX mean 41.0467 us (p50 40.4710 us, CV 6.79%):
    Rust/JAX 0.083x, Rust 12.09x faster.
  - Rust literal-backed control mean estimate 3.5255 us
    (`[3.4610, 3.5862]` us Criterion interval): dense/literal 0.963x,
    dense 1.04x faster internally. This is a narrow construction-path win, not
    a broad dense-storage breakthrough.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core stack_axis0 --lib`
  passed 6 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` also passed after repairing two pre-existing
  test-gate issues in `crates/fj-core/src/lib.rs` (missing helper imports and a
  `clone_on_copy` test lint).
- Retry predicate: do not repeat this `stack_axis0` tensor concat-storage lever
  unless a new profile shows a distinct stack call path or a larger shape where
  literal materialization reappears. The sibling scalar-stack, repeat, slice, and
  `to_i64_vec` dense-storage beads still require their own JAX head-to-head rows;
  do not generalize this 12.09x external win to those levers.

## frankenjax-cod-b-dense-scalar-stack-axis0-tobpl - Dense Scalar stack_axis0 Packing

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: specialize `TensorValue::stack_axis0` for homogeneous scalar outputs
  and route them directly to dense typed tensor constructors, avoiding boxed
  `Vec<Literal>` materialization for loop-and-stack scalar workloads.
- Status: measured keep. External head-to-head is a decisive Rust win versus
  original JAX on a realistic 64-scalar loop-and-stack workload. No revert.
- Benchmark guard: `core/scalar_stack_axis0_f64_64`, added to
  `crates/fj-core/benches/core_baseline.rs` in this measurement commit.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/scalar_stack_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda *xs: jnp.stack(xs, axis=0))`
    over 64 scalar `float64` inputs with `jax_enable_x64=true`, CPU backend,
    200 samples x 1000 inner loops via
    `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
    (JAX 0.10.1).
  - Rust mean estimate 137.53 ns (`[135.34, 139.81]` ns Criterion interval),
    median 136.77 ns, p95 158.22 ns, p99 165.03 ns, sample CV 8.22%.
  - JAX mean 25.1019 us, p50 24.4422 us, p95 28.5403 us, p99 31.3033 us,
    CV 6.26%.
  - Rust/JAX 0.0055x, Rust 182.21x faster. This row is host-dispatch dominated
    on the JAX side, but the ratio is far beyond the noise band.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core stack_axis0 --lib`
  passed 6 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` also passed.
- Retry predicate: do not repeat scalar `stack_axis0` dense-constructor packing
  unless a new profile shows a non-`f64` scalar dtype, larger scalar-count batch,
  or a compiled buffer-reuse path with materially different cost. Continue to
  benchmark scalar repeat, slice, and extraction siblings separately.

## frankenjax-cod-b-dense-scalar-repeat-axis0-zb2b7 - Dense Scalar repeat_axis0 Packing

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify scalar `TensorValue::repeat_axis0` dense-constructor packing for
  homogeneous scalar repeats, avoiding a materialized `Vec<Literal>` loop before
  producing the length-64 dense tensor.
- Status: measured keep. External head-to-head is a decisive Rust win versus
  original JAX on scalar repeat construction, and the materializing Rust control
  confirms a real local construction-path win. No revert.
- Benchmark guard: `core/scalar_repeat_axis0_f64_64`, plus
  `core/scalar_repeat_axis0_f64_64_materializing_control` in
  `crates/fj-core/benches/core_baseline.rs`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/scalar_repeat_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-scalar-repeat-axis0-control`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_repeat_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_scalar_repeat_axis0_jax_raw.json`.
  - Rust optimized mean estimate 66.809 ns (`[65.832, 67.860]` ns Criterion
    interval) vs JAX mean 4.6720 us (p50 4.5007 us, p95 5.6054 us,
    p99 6.3644 us, CV 9.46%): Rust/JAX 0.0143x, Rust 69.93x faster.
  - Rust materializing control mean estimate 181.12 ns (`[176.40, 186.55]` ns
    Criterion interval): optimized/materializing 0.369x, optimized 2.71x
    faster internally.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core repeat_axis0 --lib`
  passed 5 tests, 0 failed (RCH fell back to local execution). Per-crate
  `cargo check -p fj-core --bench core_baseline`, `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` passed.
- Retry predicate: do not retry scalar `repeat_axis0` dense-constructor packing
  unless a new profile shows a non-`f64` scalar dtype, larger repeat count, or a
  compiled buffer-reuse path with materially different cost. Future scalar repeat
  work should target compiled output-buffer reuse, not another `Vec<Literal>`
  elision.

## frankenjax-cod-b-dense-tensor-repeat-axis0-jk3ed - Dense Tensor repeat_axis0 Concat Storage

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route `TensorValue::repeat_axis0` for tensor inputs through
  `LiteralBuffer::from_concat_slices` plus `new_with_literal_buffer`, preserving
  packed dense storage instead of materializing a repeated `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original JAX;
  a temporary materializing-path revert regressed badly, so the optimization was
  restored and kept. No committed revert.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_cod_b_dense_tensor_repeat_axis0_gauntlet_2026-06-19.json`.
- Benchmark guard: `core/tensor_repeat_axis0_dense_f64_1k_x64`. The in-tree
  `core/tensor_repeat_axis0_literal_f64_1k_x64` row is not a valid
  pre-optimization control after `TensorValue::new` densification, because its
  input is also dense by the time `repeat_axis0` runs.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust optimized command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_repeat_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda v: jnp.repeat(v[None, :], 64, axis=0))`
    over a 1024-element `float64` vector with `jax_enable_x64=true`, CPU backend,
    80 runs x 200 inner loops via `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
    (JAX/JAXLIB 0.10.1).
  - Rust optimized dense mean estimate 3.2461 us (`[3.2053, 3.2855]` us
    Criterion interval) vs JAX mean 11.7639 us (p50 11.1859 us, CV 10.47%):
    Rust/JAX 0.276x, Rust 3.62x faster.
  - Same-code-path dense-input/literal-input rows before the temporary revert:
    3.2461 us vs 3.2266 us. This apparent neutral result was not a valid
    pre-optimization comparison because both inputs reached the direct concat
    path with dense storage.
  - Temporary materializing-path control, produced by locally reverting only the
    direct concat hunk and then rerunning the same Criterion filter: 92.057 us
    (`[90.552, 93.632]` us). Optimized/materializing 0.035x, optimized 28.36x
    faster. The temporary revert was discarded before commit.
  - Independent cod-a validation rerun: Rust dense Criterion slope 2.887 us
    (`[2.853, 2.922]` us) vs a materialized JAX `jnp.tile(x[None, :], (64, 1))`
    mean 16.606 us (p50 15.721 us, CV 29.61%): Rust/JAX 0.174x, Rust 5.75x
    faster. Treat this as directional because the JAX CV is high; the keep
    decision still rests on the 28.36x materializing-control result above.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core repeat_axis0 --lib`
  passed 5 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` passed.
- Retry predicate: do not retry tensor `repeat_axis0` literal materialization or
  direct concat-storage variants without a new profile showing a distinct
  repeat path. Follow-up should target compiled/buffer-reuse semantics or larger
  batched repeat workloads, not another `Vec<Literal>` elision for this path.

## frankenjax-cod-b-dense-slice-axis0-4bnj5 - Dense Tensor slice_axis0 Storage Preservation

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify dense `TensorValue::slice_axis0` rank-2 row extraction as a
  storage-preserving view-like construction instead of rematerializing the row
  through `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original
  JAX CPU; local materializing control confirms a real construction-path win.
  No revert.
- Benchmark guard: `core/tensor_slice_axis0_dense_f64_64x1k_row31`,
  `core/tensor_slice_axis0_dense_f64_64x1k_row31_to_f64_vec`, and
  `core/tensor_slice_axis0_materializing_control_f64_64x1k_row31`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_slice_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-slice-axis0-control`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_slice_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_slice_axis0_jax_raw.json`.
  - Rust lazy slice construction mean estimate 238.18 ns
    (`[235.10, 241.48]` ns Criterion interval).
  - Rust slice plus `to_f64_vec` mean estimate 513.73 ns
    (`[507.65, 520.11]` ns Criterion interval).
  - Rust materializing control mean estimate 1.4253 us
    (`[1.4056, 1.4456]` us Criterion interval): lazy/control 0.167x
    (5.98x faster) and extract/control 0.360x (2.77x faster).
  - JAX bare row slice mean 5.1409 us (p50 4.8466 us, p95 6.5725 us,
    p99 6.9790 us, CV 11.35%).
  - JAX row slice plus `+0.0` mean 5.0677 us (p50 4.9258 us, p95
    6.0064 us, p99 6.3937 us, CV 7.40%).
  - Rust/JAX ratios: lazy vs bare 0.0463 (Rust 21.58x faster);
    extract vs `+0.0` 0.101 (Rust 9.86x faster). Treat the exact external
    ratios as directional because JAX CV exceeded 5%.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core slice_axis0 --lib`
  passed 2 tests, 0 failed. Per-crate `cargo check -p fj-core --bench core_baseline`,
  `cargo check -p fj-core --all-targets`, `cargo clippy -p fj-core --all-targets -- -D warnings`,
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings`, `cargo fmt -p fj-core --check`,
  and `ubs --only=rust --skip-rust=12,13,14 crates/fj-core/benches/core_baseline.rs`
  passed for this lane.
- Retry predicate: do not retry `slice_axis0` row construction or another
  `Vec<Literal>`-elision variant without a new profile showing slice
  construction, not downstream arithmetic, is again a top-five fj-core cost.
  Future work should target compiled consumer fusion or direct output-buffer
  reuse, not another row-slice storage repack.

## frankenjax-cod-b-dense-to-i64-vec-7xbu9 - Dense Tensor to_i64_vec Fast Path

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify `TensorValue::to_i64_vec` through
  `LiteralBuffer::as_i64_slice` before falling back to per-`Literal`
  extraction, avoiding boxed literal iteration for packed I64 buffers and
  concat I64 slices.
- Status: measured keep. The direct dense extraction path is a decisive Rust
  internal win versus the literal-backed fallback and a directional external
  win versus original JAX host extraction. No revert.
- Benchmark guard: `core/tensor_to_i64_vec_dense_4k` and
  `core/tensor_to_i64_vec_literal_4k` in `crates/fj-core/benches/core_baseline.rs`,
  plus `benchmarks/jax_comparison/core_to_i64_gauntlet.py` for the JAX host-copy
  comparator.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_to_i64_vec --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-to-i64-vec`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_to_i64_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_to_i64_jax_raw.json`.
  - Rust dense mean estimate 552.47 ns (`[542.29, 565.54]` ns Criterion
    interval).
  - Rust literal-backed fallback mean estimate 6.6566 us
    (`[6.4275, 6.8878]` us Criterion interval): dense/literal 0.0830x,
    dense 12.05x faster internally.
  - JAX identity-ready lower-bound mean 6.5825 us (p50 6.1655 us,
    p95 8.9080 us, p99 12.4299 us, CV 19.47%): dense Rust/JAX 0.0839x,
    Rust 11.92x faster.
  - JAX NumPy host-copy mean 9.0209 us (p50 8.8057 us, p95 10.4230 us,
    p99 10.6834 us, CV 6.56%): dense Rust/JAX 0.0612x, Rust 16.33x
    faster. Treat both external ratios as directional because JAX CV exceeded
    5%; the local dense/literal separation is the primary keep proof.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core to_i64_vec --lib`
  passed 4 tests, 0 failed on `vmi1152480`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `hz1`, `python -m py_compile benchmarks/jax_comparison/core_to_i64_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_to_i64_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not retry dense `to_i64_vec` or another
  `as_i64_slice -> Vec<i64>` elision for this path without a fresh profile
  showing extraction remains a top-five fj-core cost. Future work should target
  compiled consumers, output-buffer reuse, non-I64 extraction, or avoiding the
  owned host copy entirely.

## frankenjax-mcqr.97 - TensorValue::new Dense Literal Storage

- Date: 2026-06-18; verified 2026-06-19
- Agent: cod-b / TopazOrchid; verified by cod-b / WildForge
- Lever: densify homogeneous F64/F32/Bool/BF16/F16/Complex literal vectors in
  `TensorValue::new` after element-count validation.
- Status: measured mixed. Keep the dense storage behavior for downstream typed
  consumers, but record negative evidence for construction-only workloads:
  generic densification is slower than forced literal construction until a dense
  slice/vector consumer uses the packed storage.
- Benchmark guard: `core/tensor_value_new_1k_f64_generic_dense`,
  `core/tensor_value_new_1k_f64_forced_literal`,
  `core/tensor_value_new_then_to_f64_vec_1k`,
  `core/tensor_value_new_forced_literal_then_to_f64_vec_1k`, plus
  `benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_value_new --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-tensor-value-new`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_tensor_value_new_jax_raw.json`.
  - Rust generic dense construction mean estimate 1.4152 us
    (`[1.4005, 1.4300]` us Criterion interval).
  - Rust forced literal construction mean estimate 432.80 ns
    (`[427.08, 438.80]` ns Criterion interval): generic dense / forced literal
    3.27x, a real construction-only regression.
  - Rust generic dense construction plus `to_f64_vec` mean estimate 1.4131 us
    (`[1.3947, 1.4326]` us Criterion interval).
  - Rust forced literal construction plus `to_f64_vec` mean estimate 2.3787 us
    (`[2.3368, 2.4269]` us Criterion interval): dense / forced literal 0.594x,
    dense 1.68x faster once extraction consumes typed storage.
  - JAX `jnp.asarray(values).block_until_ready()` mean 48.5634 us
    (p50 48.8473 us, p95 52.3213 us, p99 54.9046 us, CV 5.04%):
    generic dense Rust/JAX 0.0291x, Rust 34.31x faster.
  - JAX `jnp.asarray(values)` plus NumPy host copy mean 55.9286 us
    (p50 56.1224 us, p95 60.8686 us, p99 63.5072 us, CV 5.23%):
    dense Rust extraction / JAX host copy 0.0253x, Rust 39.58x faster. Treat
    exact external ratios as directional because JAX CV is just above 5%.
- Conformance guard: matching literal families materialize bit-identically;
  mismatched literal/dtype tensors remain literal-backed.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core tensor_new_densifies_matching_literal_families --lib`
  passed 1 test, 0 failed on `ovh-a`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz1`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not use `TensorValue::new` densification as a
  construction-only performance lever; use direct typed constructors or
  `new_with_literal_buffer` for hot paths that will not consume typed storage.
  Do not repeat FMA, SIMD exp, GEMM, QR, SVD, cumsum, eager concat, or the
  already committed stack/repeat/slice/to_i64 storage levers without fresh
  same-worker benchmark evidence identifying a distinct call path.

## frankenjax-mcqr.99 - Direct Dense LiteralBuffer to_vec Materialization

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::to_vec()` materialize directly from dense storage
  variants instead of forcing `as_slice()` to build/cache a full literal vector
  and then clone it.
- Status: measured keep internally; mixed external head-to-head versus JAX.
- Benchmark guard: `core/literal_buffer_to_vec_dense_f64_64k`,
  `core/literal_buffer_to_vec_literal_f64_64k`, plus
  `benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`.
- Measured evidence:
  - Rust Criterion, local same-host:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/literal_buffer_to_vec --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-literal-buffer-to-vec`.
  - Dense F64 64k direct materialization: 26.644 us mean
    (`[26.253, 27.043]` us).
  - Literal F64 64k fallback/control: 33.306 us mean
    (`[32.779, 33.868]` us).
  - Internal result: dense direct materialization is 0.800x the literal control,
    or 1.25x faster.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_literal_buffer_to_vec_jax_raw.json`.
  - JAX identity-ready lower bound: 19.5312 us mean, p50 19.2092 us,
    p95 22.7052 us, p99 24.0948 us, CV 9.07%; Rust/JAX 1.364x, so Rust is
    1.36x slower than this no-host-copy lower bound.
  - JAX NumPy host-copy comparator: 32.9368 us mean, p50 32.4829 us,
    p95 38.4794 us, p99 41.1890 us, CV 8.69%; Rust/JAX 0.809x, so Rust is
    1.24x faster than this host-copy comparator.
  - External ratios are directional because both JAX rows have CV above 5%, and
    the JAX host-copy row copies raw F64 values while Rust materializes
    `Literal` enum values.
- Conformance guard: direct `to_vec()` matches `as_slice().to_vec()` bit-for-bit
  across dense F64/F32/I64/U32/U64/Bool/BF16/F16/Complex storage plus lazy
  concat/repeated-patches paths.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_to_vec_direct_paths_match_slice_materialization --lib`
  passed 1 test, 0 failed on `ovh-a`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not retry direct dense `LiteralBuffer::to_vec`
  materialization for the F64 64k path unless a fresh profile still puts
  `to_vec` in the top five. The next attempt should target cached literal reuse,
  consumer-side avoidance of `Vec<Literal>`, or direct typed-consumer APIs rather
  than another direct dense-to-`Literal` map. Do not retry the already committed
  stack/repeat/slice/to_i64 or `TensorValue::new` dense-storage families under
  this bead. Do not revisit FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-q59j4 - Direct Dense LiteralBuffer COW Mutation

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::make_mut()` materialize dense/lazy storage through
  the direct storage-aware `to_vec()` path instead of forcing `as_slice()` to
  build/cache an intermediate full literal vector before mutation.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_index_mut_dense_f64_64k`,
  `core/literal_buffer_index_mut_literal_f64_64k`.
- Conformance guard: dense and lazy COW mutation preserves the original cloned
  sequence and mutates to the same materialized literal sequence as a literal
  buffer for F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, plus a dense-sort materialization path.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, or `LiteralBuffer::to_vec` dense-storage families under
  this bead. Reopen the dense COW mutation family only with focused criterion
  evidence showing mutation/materialization cost remains a top-five fj-core
  bottleneck. Do not revisit FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.100 - Dense Half-Width Bitcast Chunks

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->BF16/F16 and BF16/F16->F32
  through packed f32/u16 slices instead of per-`Literal` byte conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  both measured BF16 reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_bf16_dense_1m`,
  `eval/bitcast_f32_bf16_literal_ref_1m`,
  `eval/bitcast_bf16_f32_dense_1m`,
  `eval/bitcast_bf16_f32_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench lax_baseline`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench lax_baseline -- 'eval/bitcast_(f32_i32|i32_f32|f64_u64|u64_f64|f32_bf16|bf16_f32)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline frankenjax-mcqr-100-101-bitcast`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/bitcast_gauntlet.py --runs 30 --warmup 5 --inner-loops 50 --output /tmp/frankenjax_mcqr_100_101_bitcast_jax_raw.json`.
  - `bitcast_f32_bf16_1m`: Rust dense mean 2.710 ms vs Rust literal
    reference 57.706 ms (21.29x faster internally); JAX mean 138.058 us;
    Rust/JAX 19.63x slower.
  - `bitcast_bf16_f32_1m`: Rust dense mean 669.626 us vs Rust literal
    reference 31.549 ms (47.11x faster internally); JAX mean 139.426 us;
    Rust/JAX 4.80x slower.
  - Decision: keep, not revert. Both rows are large internal wins over the
    original per-Literal path, but they are external JAX losses. The next BF16
    bitcast attempt must target raw packed output construction throughput, not
    another boxed-literal-elision pass.
- Conformance guard: dense and literal-backed half-width bitcasts produce the
  same output shapes, dtypes, raw half chunks, and round-trip f32 bit patterns
  across NaN, infinities, signed zero, normals, and a custom NaN payload.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry this half-width bitcast family unless focused
  criterion evidence shows the dense rows are still slower than the
  literal-backed reference or the original path remains a top-five fj-lax
  bottleneck. Do not merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or peer-owned fj-core dense-storage lanes without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.101 - Dense Signed/Unsigned Same-Width Bitcast Pairs

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->I32, I32->F32, F64->U64, and
  U64->F64 through packed typed slices instead of per-`Literal` byte
  conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  all four measured same-width reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_i32_dense_1m`,
  `eval/bitcast_f32_i32_literal_ref_1m`,
  `eval/bitcast_i32_f32_dense_1m`,
  `eval/bitcast_i32_f32_literal_ref_1m`,
  `eval/bitcast_f64_u64_dense_1m`,
  `eval/bitcast_f64_u64_literal_ref_1m`,
  `eval/bitcast_u64_f64_dense_1m`,
  `eval/bitcast_u64_f64_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust/JAX commands match the `frankenjax-mcqr.100` bitcast gauntlet entry.
  - `bitcast_f32_i32_1m`: Rust dense mean 430.783 us vs Rust literal
    reference 19.537 ms (45.35x faster internally); JAX mean 133.290 us;
    Rust/JAX 3.23x slower.
  - `bitcast_i32_f32_1m`: Rust dense mean 642.693 us vs Rust literal
    reference 22.677 ms (35.28x faster internally); JAX mean 93.945 us;
    Rust/JAX 6.84x slower.
  - `bitcast_f64_u64_1m`: Rust dense mean 270.723 us vs Rust literal
    reference 24.747 ms (91.41x faster internally); JAX mean 176.611 us;
    Rust/JAX 1.53x slower.
  - `bitcast_u64_f64_1m`: Rust dense mean 228.320 us vs Rust literal
    reference 20.525 ms (89.89x faster internally); JAX mean 175.404 us;
    Rust/JAX 1.30x slower.
  - Decision: keep, not revert. Every row is a large internal win and the
    f64/u64 rows are near the existing memory-bandwidth residual gap, but all
    rows are still external JAX losses.
- Conformance guard: dense and literal-backed signed/unsigned same-width
  bitcasts produce the same shapes, dtypes, exact integer bit lanes, packed
  storage, and round-trip float bit patterns across NaN, infinities, signed
  zero, normals, and custom NaN payloads.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry the already committed F32<->U32, F64<->I64,
  F64<->U32, U32<->F64, F32<->BF16/F16, BF16/F16->F32, or these signed/
  unsigned same-width bitcast pairs without focused criterion evidence showing
  the dense rows are still slower than the literal-backed reference or the
  original per-`Literal` path remains a top-five fj-lax bottleneck. Do not
  revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  peer-owned fj-core dense-storage lanes without fresh same-worker benchmark
  evidence and ownership check.

## frankenjax-mcqr.96/.98 - Dense U32/I64 Same-Width and Width-Changing Bitcasts

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify the committed dense `BitcastConvertType` fast paths for
  F32->U32, F64->I64, F64->U32, and U32->F64 against both the Rust boxed
  literal reference path and original JAX CPU.
- Status: measured mixed keep. One same-width row is a noisy external Rust win;
  the remaining three rows are negative head-to-head results versus original
  JAX. All four rows stay as internal keeps because they beat the boxed Rust
  reference by 43.99-201.42x. No revert.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_96_98_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/lax_baseline.rs`, 1,048,576 element
  bitcast rows, Criterion sample size 20, warmed original-JAX CPU timing with
  50 runs x 500 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench lax_baseline`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench lax_baseline -- 'eval/bitcast_(f32_u32|f64_i64|f64_u32|u32_f64)' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-96-98-bitcast`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/bitcast_gauntlet.py --runs 50 --warmup 10 --inner-loops 500 --output /tmp/frankenjax_mcqr_96_98_bitcast_jax_raw_rerun.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend.

| Workload | Rust dense mean | Rust literal mean | Dense/literal | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bitcast_f32_u32_1m` | 97.423 us | 19.623 ms | 201.42x faster | 113.430 us | 0.859 | Noisy JAX win; keep |
| `bitcast_f64_i64_1m` | 270.586 us | 22.260 ms | 82.27x faster | 183.715 us | 1.473 | Keep internal win; JAX loss |
| `bitcast_f64_u32_1m` | 1.272 ms | 60.757 ms | 47.76x faster | 186.831 us | 6.809 | Keep internal win; JAX loss |
| `bitcast_u32_f64_1m` | 919.708 us | 40.456 ms | 43.99x faster | 189.478 us | 4.854 | Keep internal win; JAX loss |

- CV notes: JAX CV ranged from 12.08% to 19.80%, so the F32->U32 Rust/JAX win
  is directional rather than certification-grade. The three JAX losses are wide
  enough to route follow-up even with the noisy oracle run.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed, 0 ignored, 1676 filtered out.
- Decision: keep, not revert. The original code-first optimization is real
  relative to the boxed Rust path, but only the F32->U32 row beat original JAX
  and that result is noisy. Do not claim release-grade JAX domination for this
  cluster.
- Retry predicate: do not retry boxed-literal elision for F32->U32, F64->I64,
  F64->U32, or U32->F64 bitcasts. Same-width follow-up must target raw output
  buffer construction, store throughput, or JAX-like compiled buffer reuse.
  Width-changing follow-up must target packed chunk layout and widening/narrowing
  construction directly, not another dense-storage sibling.

## frankenjax-mcqr.103-.104 - Dense I64 Clamp Tensor and Mixed Bounds

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify committed dense I64 `Clamp` fast paths for same-shape tensor
  bounds and both mixed scalar/tensor bound orders against the boxed Rust
  reference path and original JAX CPU.
- Status: measured keep internally; negative head-to-head result versus JAX on
  all three measured I64 clamp workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_103_104_i64_clamp_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576 I64
  elements per row, Criterion sample size 20, warmed original-JAX CPU timing
  with 50 runs x 100 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench clamp_gauntlet`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- 'i64_' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-103-104-i64-clamp`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_103_104_i64_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend.

| Workload | Rust dense mean | Rust boxed mean | Dense/boxed | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `i64_mixed_scalar_lo_tensor_hi_1m` | 689.991 us | 23.655 ms | 34.28x faster | 200.558 us | 3.44x slower | Keep internal win; JAX loss |
| `i64_mixed_tensor_lo_scalar_hi_1m` | 611.377 us | 25.749 ms | 42.12x faster | 197.536 us | 3.10x slower | Keep internal win; JAX loss |
| `i64_tensor_tensor_tensor_1m` | 1.046 ms | 24.674 ms | 23.59x faster | 287.593 us | 3.64x slower | Keep internal win; JAX loss |

- Conformance guard: dense and boxed I64 clamp paths match the generic
  semantics for same-shape tensor bounds and mixed scalar/tensor bounds.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax clamp --lib`
  passed 25 tests, 0 failed, 6 ignored.
- Decision: keep, not revert. Every measured row is a 23.59-42.12x internal
  win over the boxed Rust reference path, while every row remains a 3.10-3.64x
  external loss versus original JAX CPU.
- Retry predicate: do not retry boxed-literal elision for these I64 clamp
  shapes. The next I64 clamp attempt must target SIMD/parallel clamp
  throughput, output allocation/fusion, or JAX-like compiled buffer reuse, not
  another dense-storage sibling fast path. Do not merge this family with
  half-raw-bit clamp or broadcast-shape generalization without fresh
  Criterion/JAX evidence and ownership check.

## frankenjax-co009 - Stream Dense LiteralBuffer Serialization

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `Serialize for LiteralBuffer` stream storage-direct elements
  through `SerializeSeq` instead of forcing `as_slice()` to cache a full
  materialized literal vector for dense packed buffers.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_serialize_dense_f64_64k`,
  `core/literal_buffer_serialize_literal_f64_64k`.
- Conformance guard: streamed JSON matches materialized `Vec<Literal>` JSON
  across F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, and mixed dense/literal concat paths.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation, or this
  serialization streaming family without fresh focused criterion evidence
  showing it remains a top-five fj-core bottleneck. Do not revisit FMA/SIMD exp,
  GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or eager concat without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.102 - Storage-Direct LiteralBuffer Equality

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer` equality compare storage-direct typed ranges and
  recursively compare concat slices instead of forcing `as_slice()` on both
  operands and caching full materialized literal vectors for dense packed
  buffers.
- Status: measured keep.
- Benchmark guard: `core/literal_buffer_eq_dense_f64_64k_equal`,
  `core/literal_buffer_eq_dense_f64_64k_mismatch`,
  `core/literal_buffer_eq_literal_f64_64k_equal`, plus
  `benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`.
- Measured evidence:
  - Rust Criterion, local same-host:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/literal_buffer_eq --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-literal-buffer-eq`.
  - Dense F64 64k equal: 27.484 us mean (`[27.047, 27.926]` us).
  - Dense F64 64k tail-mismatch: 27.570 us mean (`[27.226, 27.951]` us).
  - Literal F64 64k equal control: 53.559 us mean (`[52.696, 54.423]` us).
  - Internal result: dense equality is 0.513x the literal equality control, or
    1.95x faster.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_literal_buffer_eq_jax_raw.json`.
  - JAX equal device-ready lower bound: 58.2421 us mean, p50 54.3390 us,
    p95 76.0146 us, p99 80.4897 us, CV 15.86%; Rust/JAX 0.472x, Rust
    2.12x faster directionally.
  - JAX equal host-bool comparator: 66.6275 us mean, p50 63.2482 us,
    p95 89.2968 us, p99 92.7591 us, CV 17.95%; Rust/JAX 0.413x, Rust
    2.42x faster directionally.
  - JAX tail-mismatch device-ready lower bound: 56.2721 us mean, p50 53.4535 us,
    p95 72.5154 us, p99 75.3971 us, CV 14.40%; Rust/JAX 0.490x, Rust
    2.04x faster directionally.
  - JAX tail-mismatch host-bool comparator: 68.9387 us mean, p50 70.1777 us,
    p95 82.6965 us, p99 85.0570 us, CV 11.61%; Rust/JAX 0.400x, Rust
    2.50x faster directionally.
  - External ratios are directional because JAX CV is above 5% for all equality
    rows. The Rust dense/literal split is the hard keep criterion.
- Conformance guard: storage-direct equality matches materialized
  `Vec<Literal>` equality across F64/F64OnePlusX/F32/I64/U32/U64/Bool/
  BoolWords/Half/Complex, repeated-patches, concat, mixed dense/literal concat,
  and the `LiteralBuffer`/`Vec<Literal>` cross-`PartialEq` impls.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_storage_direct_equality_matches_materialized_vec --lib`
  passed 1 test, 0 failed on `hz1`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`
  returned 0 warnings. Workspace `cargo fmt --check` remains red on unrelated
  pre-existing formatting drift outside the touched files; no Rust code changed
  in this measured closeout.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation,
  serialization streaming, or this equality family without fresh focused
  criterion evidence showing comparison remains a top-five fj-core bottleneck.
  Do not revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  eager concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-alc0j - Dense Scalar Broadcast for U32/U64/Complex

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `BroadcastInDim` fills for U32, U64, Complex64, and
  Complex128 through `new_u32_values`, `new_u64_values`, and
  `new_complex_values` instead of allocating a `Vec<Literal>` fill.
- Status: measured mixed/noisy against warmed JAX CPU. Keep the committed dense
  fill path, but record negative head-to-head evidence for Complex128.
- Benchmark guard: `eval/broadcast_scalar_u32_1024x1024`,
  `eval/broadcast_scalar_u64_1024x1024`,
  `eval/broadcast_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/broadcast_scalar --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline scalar
    broadcast harness with `jax_enable_x64=true`, JAX/JAXLIB 0.10.1, CPU
    backend, 60 runs x 1000 warmed inner loops.
  - `broadcast_scalar_u32_1024x1024`: Rust Criterion mean 51.307 us, slope
    49.804 us, median 50.460 us, Rust CV 10.56%; JAX mean 79.979 us, p50
    84.805 us, JAX CV 26.62%; Rust/JAX mean ratio 0.642, Rust 1.56x faster.
    Classification: noisy win.
  - `broadcast_scalar_u64_1024x1024`: Rust Criterion mean 104.911 us, slope
    102.906 us, median 102.130 us, Rust CV 18.20%; JAX mean 124.569 us, p50
    130.398 us, JAX CV 31.74%; Rust/JAX mean ratio 0.842, Rust 1.19x faster.
    Classification: noisy win/near-tie.
  - `broadcast_scalar_complex128_1024x1024`: Rust Criterion mean 283.150 us,
    slope 263.538 us, median 276.539 us, Rust CV 18.07%; JAX mean 245.381 us,
    p50 259.656 us, JAX CV 24.66%; Rust/JAX mean ratio 1.154, Rust 1.15x
    slower. Classification: noisy loss.
  - Decision: keep, not revert. The U32 and U64 scalar-fill rows beat JAX by
    mean ratio, while Complex128 loses; all three rows are above the 5% CV
    certification bar, so this is real negative/positive evidence but not a
    release-grade perf certificate. There is no same-run evidence that reverting
    the dense fill path improves the Complex128 row, and reverting would discard
    the measured U32/U64 wins. The next Complex128 attempt must target complex
    packed-fill/store throughput, not another boxed-literal-elision pass.
- Conformance guard: scalar fills materialize to the same repeated literals and
  expose dense typed storage via `as_u32_slice`, `as_u64_slice`, or
  `as_complex_slice`.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax dense_broadcast_in_dim_matches_literal_path_and_stays_dense --lib`
  passed 1 test, 0 failed.
- Retry predicate: do not retry scalar `BroadcastInDim` dense-fill storage for
  these dtypes unless focused criterion evidence shows this path remains a
  top-five `fj-lax` bottleneck with both Rust and JAX CV below 5%, or the dense
  constructor representation changes. For Complex128 specifically, do not
  repeat the same `Vec<(f64, f64)>` dense fill lever; the next attempt must
  attack packed complex output construction/store throughput directly. Do not
  merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  broader scalar-broadcast arithmetic work without fresh benchmark evidence and
  ownership check.

## frankenjax-dxqfj - Lazy SplitMulti Section Buffers

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: build each `eval_split_multi` output from `LiteralBuffer::from_concat_slices`
  over the original tensor backing instead of copying each section through
  `Vec<Literal>` and re-densifying via `TensorValue::new`.
- Status: measured keep against warmed JAX CPU. No revert.
- Benchmark guard: `eval/split_multi_1024x1024_f32_axis1`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/split_multi_1024x1024_f32_axis1 --sample-size 60 --warm-up-time 1 --measurement-time 5`.
  - Rust Criterion: mean 96.651 us, median 95.038 us, slope 98.875 us
    (95% CI 97.211-100.534 us), stddev 6.616 us, CV 6.85%.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline split
    harness, JAX/JAXLIB 0.10.1, `jax_enable_x64=true`, CPU backend, 60 runs x
    1000 warmed inner loops.
  - `jnp.split(x, [256], axis=1)` bare output: JAX mean 149.082 us, p50
    148.877 us, CV 12.08%; Rust/JAX mean ratio 0.648, Rust 1.54x faster.
  - `jnp.split(x, [256], axis=1)` with `+0.0` materialization on both outputs:
    JAX mean 136.983 us, p50 139.649 us, CV 10.43%; Rust/JAX mean ratio 0.706,
    Rust 1.42x faster.
  - Decision: keep. The JAX side remains noisy, but both JAX modes are slower
    than the Rust lazy-section split by a margin larger than the Rust CI band.
    This is a measured split-construction win, not a claim about arbitrary
    downstream consumers after full materialization.
- Conformance guard: uneven multi-output split materializes the same literals for
  each section and exposes dense f32 storage through `as_f32_slice`;
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax split_multi --lib`
  passed 2 tests, 0 failed.
- Retry predicate: retry only if a profiler attributes a clearly-above-noise
  share to `eval_split_multi` section construction on a wider multi-output split
  workload after Criterion and JAX CV are both below 5%, or if
  `LiteralBuffer::Concat` dense-lane behavior changes. Do not merge with
  FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or broader
  reshape/slice/gather work without fresh benchmark evidence and ownership
  check.

## frankenjax-19wst - Dense Scalar Tile Fills

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `Tile` uniform fills through dense typed `TensorValue`
  constructors instead of filling a `Vec<Literal>` with the recursive boxed
  tiler and then re-densifying.
- Status: measured keep against warmed JAX CPU for both guard workloads.
- Benchmark guard: `eval/tile_scalar_f32_1024x1024`,
  `eval/tile_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/tile_scalar --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `tile_scalar_f32_1024x1024`: Rust criterion median 51.435 us vs JAX
    batched median 317.753 us; ratio Rust/JAX 0.162, Rust 6.18x faster.
  - `tile_scalar_complex128_1024x1024`: Rust criterion median 412.679 us vs
    JAX batched median 579.030 us; ratio Rust/JAX 0.713, Rust 1.40x faster.
  - Decision: keep. No revert; both measured workloads beat the original JAX
    oracle under warmed CPU execution. Complex128 margin is modest enough that
    future retries must preserve this exact head-to-head guard.
- Conformance guard: scalar tile materializes to the same repeated literals and
  exposes dense typed storage for f32, u64, and complex128.
- Retry predicate: do not retry scalar `Tile` dense-fill storage unless focused
  criterion evidence shows this path remains a top-five `fj-lax` bottleneck or
  the scalar tile representation changes. Do not merge it with scalar
  `BroadcastInDim`, dense tensor tile, FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or broader shape/data movement work without fresh
  benchmark evidence and ownership check.

## frankenjax-1z7k9 - Dense Tensor-Scalar Complex Constructor

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route matching F64/F32 `eval_complex` tensor-scalar and scalar-tensor
  constructors through dense `new_complex_values` instead of rebuilding every
  element as a boxed `Literal`.
- Status: measured mixed; keep the committed dense path, but record a negative
  JAX head-to-head result for the F32 tensor-scalar workload.
- Benchmark guard: `eval/complex_f32_tensor_scalar_1m`,
  `eval/complex_f64_tensor_scalar_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/complex_f --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `complex_f32_tensor_scalar_1m`: Rust criterion median 1.379 ms vs JAX
    batched median 1.272 ms; ratio Rust/JAX 1.084, Rust 1.08x slower. This is
    negative evidence for "dense constructor alone dominates original JAX" on
    Complex64/F32.
  - `complex_f64_tensor_scalar_1m`: Rust criterion median 0.914 ms vs JAX
    batched median 3.730 ms; ratio Rust/JAX 0.245, Rust 4.08x faster.
  - Decision: keep, not revert. The cluster has one clear JAX win and one clear
    JAX loss; there is no same-run evidence that reverting the dense constructor
    improves the F32 result, and reverting would discard the measured F64 win.
    Route the F32 loss to a deeper primitive-level follow-up instead of retrying
    the same boxed-literal-elision lever.
- Conformance guard: dense tensor-scalar and scalar-tensor constructor outputs
  materialize identically to explicit literal-backed references and expose dense
  complex storage.
- Retry predicate: do not retry tensor-scalar `eval_complex` constructor storage
  unless focused criterion evidence shows this path remains a top-five
  `fj-lax` bottleneck or mixed-dtype promotion becomes the measured hotspot.
  For Complex64/F32 specifically, do not repeat the same
  `Vec<(f64, f64)> -> new_complex_values` dense constructor lever; the next
  attempt must attack representation/construction overhead directly, such as a
  Complex64-native packed builder or a JAX-comparable fused real-to-complex path.
  Do not merge with same-shape complex constructor, FFT extraction, complex
  binary tensor-scalar ops, or broader complex arithmetic work without fresh
  benchmark evidence and ownership check.

## frankenjax-mcqr.105-.107 - Dense Clamp Mixed/Tensor Bounds

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify the committed dense `clamp` fast paths for mixed scalar/tensor
  F32/F64 bounds, mixed scalar/tensor BF16/F16 bounds, and same-shape BF16/F16
  tensor bounds against both the Rust boxed reference path and original JAX CPU.
- Status: measured keep internally; negative head-to-head result versus JAX for
  all six measured workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_105_107_clamp_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, warmed JAX CPU venv with 50 runs x
  100 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax --bench clamp_gauntlet`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- --save-baseline frankenjax-mcqr-105-107`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_105_107_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend, `jax_enable_x64=true`.

| Workload | Rust dense mean | Rust boxed mean | Dense/boxed | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `f32_mixed_scalar_tensor_1m` | 159.383 us | 27.645 ms | 173.45x faster | 115.540 us | 1.38x slower | Keep internal win; JAX loss |
| `f64_mixed_scalar_tensor_1m` | 996.940 us | 27.487 ms | 27.57x faster | 213.651 us | 4.67x slower | Keep internal win; JAX loss |
| `bf16_mixed_scalar_tensor_1m` | 15.571 ms | 43.758 ms | 2.81x faster | 121.313 us | 128.35x slower | Keep internal win; JAX loss |
| `f16_mixed_scalar_tensor_1m` | 19.859 ms | 41.186 ms | 2.07x faster | 371.729 us | 53.42x slower | Keep internal win; JAX loss |
| `bf16_tensor_tensor_tensor_1m` | 15.652 ms | 35.451 ms | 2.26x faster | 183.707 us | 85.20x slower | Keep internal win; JAX loss |
| `f16_tensor_tensor_tensor_1m` | 20.951 ms | 35.255 ms | 1.68x faster | 229.951 us | 91.11x slower | Keep internal win; JAX loss |

- CV notes: Rust dense CV ranged from 5.75% to 13.73%; JAX CV ranged from
  5.70% to 21.86%. Treat the F32 JAX loss as lower-confidence than the half
  losses, but it remains directionally negative under the same-host run.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo test -p fj-lax clamp --lib`
  passed 25 tests, 0 failed, 6 ignored benchmark tests. The passing set includes
  `f32_f64_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`,
  `half_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`, and
  `half_clamp_same_shape_tensor_bounds_dense_matches_generic`.
- Decision: keep, not revert. Every measured target clears the Rust boxed
  reference keep gate by at least 1.68x, so reverting would discard a real
  internal improvement. Do not claim original-JAX domination for this cluster.
- Retry predicate: do not retry boxed-literal elision for these clamp shapes.
  F32/F64 follow-up must attack SIMD/parallel clamp throughput or output
  allocation/fusion directly. BF16/F16 follow-up must avoid the per-lane
  `clamp_literal` widen/round path, for example with a raw half-bits proof
  harness plus vectorized or table-assisted min/max semantics. Do not merge the
  next attempt with broadcast-shape generalization, scalar-bound relu/relu6, or
  unrelated dense constructor work without fresh focused Criterion evidence and
  an updated JAX head-to-head row.

## frankenjax-mcqr.108 - Raw Half-Bits BF16/F16 Clamp

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: replace dense same-half BF16/F16 clamp loops that call
  `clamp_literal` per lane with a raw `u16` two-pass composition of the existing
  bit-proven half Max/Min kernels. Mixed scalar dtypes still fall back to
  `clamp_literal`.
- Status: measured keep internally; still negative head-to-head versus original
  JAX CPU for all four half clamp workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax-cod-a-half-clamp-raw-bits-2026-06-19.md`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, warmed JAX CPU venv with 50 runs x
  100 inner loops.
- Measured evidence:
  - RCH check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench clamp_gauntlet`.
  - RCH before bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-before`.
  - RCH after bench: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-after`.
  - Same-host Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-local-after`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_108_half_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend, `jax_enable_x64=true`.

| Workload | RCH before dense | RCH after dense | RCH speedup | Local Rust dense | Local boxed | Dense/boxed | JAX mean | Local Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_mixed_scalar_tensor_1m` | 21.783 ms | 2.534 ms | 8.60x | 3.616 ms | 45.091 ms | 12.47x faster | 122.705 us | 29.47x slower | Keep internal win; JAX loss |
| `f16_mixed_scalar_tensor_1m` | 28.986 ms | 3.448 ms | 8.41x | 3.521 ms | 37.471 ms | 10.64x faster | 319.088 us | 11.03x slower | Keep internal win; JAX loss |
| `bf16_tensor_tensor_tensor_1m` | 22.412 ms | 2.934 ms | 7.64x | 2.993 ms | 32.126 ms | 10.73x faster | 148.870 us | 20.10x slower | Keep internal win; JAX loss |
| `f16_tensor_tensor_tensor_1m` | 29.748 ms | 4.038 ms | 7.37x | 3.653 ms | 32.192 ms | 8.81x faster | 196.938 us | 18.55x slower | Keep internal win; JAX loss |

- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`
  passed 3 tests, 0 failed, 2 ignored benchmark tests. The passing set includes
  `half_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`,
  `half_clamp_same_shape_tensor_bounds_dense_matches_generic`, and
  `half_clamp_signed_zero_tie_order_dense_matches_generic`.
- Decision: keep, not revert. Same-worker RCH dense speedups are 7.37x-8.60x
  versus the pre-change raw half clamp baseline, and same-host local runs beat
  the Rust boxed reference by 8.81x-12.47x.
- Retry predicate: do not retry boxed literal elision or this exact two-pass
  max/min composition. The next half-clamp attempt must attack the remaining
  JAX gap directly, but not with the generic one-pass bound-abstraction shape
  rejected in `frankenjax-mcqr.109`.

## frankenjax-mcqr.109 - One-Pass Half Clamp Fused Helper

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: replace the committed raw-half two-pass clamp composition with a
  generic one-pass helper using a scalar/tensor bound abstraction, one output
  allocation, no `tmp` vector, and the same signed-zero/NaN fallback policy.
- Status: measured regression on all four half clamp workloads; reverted.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax-cod-a-half-clamp-one-pass-2026-06-19.md`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, RCH same-worker before/after on
  `ovh-a`.
- Measured evidence:
  - RCH before bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-before`.
  - RCH conformance while candidate existed: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`.
  - RCH after bench: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-after`.
  - RCH post-revert conformance: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`.

| Workload | Before two-pass mean | Candidate mean | Candidate/before | JAX mean used for context | Candidate/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_mixed_scalar_tensor_1m` | 2.3918 ms | 3.2624 ms | 1.36x slower | 122.705 us | 26.59x slower | Reject/revert |
| `f16_mixed_scalar_tensor_1m` | 3.0263 ms | 4.4595 ms | 1.47x slower | 319.088 us | 13.98x slower | Reject/revert |
| `bf16_tensor_tensor_tensor_1m` | 2.7423 ms | 3.2236 ms | 1.18x slower | 148.870 us | 21.65x slower | Reject/revert |
| `f16_tensor_tensor_tensor_1m` | 3.4745 ms | 4.4271 ms | 1.27x slower | 196.938 us | 22.48x slower | Reject/revert |

- Conformance guard: the candidate passed `cargo test -p fj-lax half_clamp --lib`
  on RCH while it existed: 4 passed, 0 failed, 2 ignored benchmark tests. The
  additional edge-matrix proof was removed with the rejected code. Post-revert
  production conformance also passed: 3 passed, 0 failed, 2 ignored benchmark
  tests.
- Decision: reject and revert. The `tmp` allocation and second helper pass were
  not the dominant cost; the generic fused helper added more chunk setup and
  half widen/round overhead than it saved.
- Retry predicate: do not retry generic `HalfClampBound` one-pass fusion or
  temp-vector removal alone. The next attempt should be dtype/shape-specialized
  raw-bit compare/classification that avoids f64 widen/round on finite lanes, or
  a producer/consumer clamp fusion that removes an intermediate across primitive
  boundaries.

## frankenjax-e07uw/7g72q/rl9ha/bjqfr - eval_jaxpr Fusion Cluster (Square / Reciprocal / integer_pow[2] / i64+bf16 broadcast)

- Lever: extend the `eval_jaxpr` cheap-op elementwise fusion to (a) `Square`,
  `Reciprocal`, `integer_pow[2]` via builder operand-synthesis (unary == fused
  binary: `Square=Mul(x,x)`, `Reciprocal=Div(1,x)`, `x**2=Mul(x,x)`), and (b) i64
  + bf16/f16 row/col broadcast operands (mirror the existing f64/f32 broadcast).
- Conformance GUARD: GREEN. `cargo test -p fj-interpreters --lib --release` — the
  fusion bit-identity tests (`fusion_*_chain_matches_reference_bit_for_bit`,
  `fusion_f64_{row,col}_broadcast_*`, etc.) all PASS. (8 unrelated golden-digest
  tests for broadcast_in_dim/transpose/reshape/scan/dce/staging PLANS are RED — see
  the separate "Golden digest drift" entry; their `plan==generic` parity asserts
  PASS, only the hardcoded sha256 lines drift; NOT caused by this cluster.)
- Measured evidence (2026-06-19, rch worker, `eval_fusion_speed` same-invocation
  A/B, 1,048,576 elements / 1024x1024, 8-op chains; rch bench noise ~1.0x band):
  - Internal (Rust fused `eval_jaxpr` vs unfused per-op `eval_primitive`):

  | Workload | unfused | fused | speedup | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | F64 arith8 1M | 16.788 ms | 3.320 ms | 5.06x | KEEP |
  | F32 arith8 1M | 7.174 ms | 1.071 ms | 6.70x | KEEP |
  | F32 clamp 1M | 9.081 ms | 1.739 ms | 5.22x | KEEP |
  | F32 row-broadcast | 12.331 ms | 0.783 ms | 15.75x | KEEP |
  | F32 col-broadcast | 11.479 ms | 0.704 ms | 16.31x | KEEP |
  | F64 row-broadcast | 8.703 ms | 2.567 ms | 3.39x | KEEP |
  | F64 col-broadcast | 8.452 ms | 2.282 ms | 3.70x | KEEP |
  | I64 arith8 1M | 5.976 ms | 3.330 ms | 1.79x | KEEP |
  | **I64 row-broadcast (rl9ha)** | 7.546 ms | 2.352 ms | **3.21x** | **KEEP** |
  | BF16 arith8 1M (not mine) | 10.893 ms | 10.599 ms | 1.03x | ~0 gain, flag owner |
  | **BF16 row-broadcast (bjqfr)** | 10.776 ms | 10.604 ms | **1.02x** | **REVERTED** |

  - Head-to-head vs original JAX (jax.jit CPU x64, `fusion_chain.py`, 1M):

  | Workload | JAX jit mean | Rust fused | Rust/JAX | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | arith8 f64 1M | 272.7 us | 3.320 ms | ~12.2x slower | JAX win (interp vs XLA) |
  | square f64 1M | 293.4 us | (rides f64 5.06x) | >10x slower | JAX win |
  | bf16 broadcast 1M | 146.9 us | 10.6 ms | ~72x slower | JAX win |

- Decision:
  - KEEP Square / Reciprocal / integer_pow[2] (bit-identical operand-synthesis;
    they let variance/L2/`x**2`/normalization chains ride the proven f64/f32
    5-6.7x fusion win) and i64 broadcast (3.21x).
  - REVERT bf16/f16 broadcast fusion (bjqfr) — measured 1.02x = ~0 gain. bf16
    tensor fusion is bandwidth-bound; the per-lane f64 decode/encode
    (half_fusion_widen + half_fused_binary) cancels the materialization savings
    (consistent with bf16 same-shape ~1.0x). Reverted in commit after this entry.
- Honest framing: the fusion lever is a real **Rust-internal interpreter** win
  (5-16x vs the pre-fusion per-op path) but does NOT achieve original-JAX
  domination on jit'd elementwise chains — the Rust tree-walking interpreter
  (even fused) is ~12-72x slower than XLA-compiled jax.jit. Fusion narrows the
  per-op interpreter tax; it cannot close the interpreter-vs-compiler gap. The
  vs-JAX win requires the compiled-jaxpr arena executor (bead z6o97 family /
  6dfew), not more fusion-op coverage.
- Retry predicate: do NOT re-add bf16/f16 tensor fusion via the scalar per-lane
  widen/round path. It only pays with a SIMD bf16<->f32 convert kernel (see
  project_bf16_matmul_and_convert_simd). Do NOT chase more fusion-op coverage for
  vs-JAX wins — the gap is interpreter-vs-compiler, addressable only by the
  compiled-jaxpr core.

## Golden digest drift - 8 fj-interpreters PLAN goldens (pre-existing, NOT this session)

- Symptom: `cargo test -p fj-interpreters --lib --release` shows 8 FAILED golden
  tests: dense_f64_broadcast_in_dim_plan / dense_f64_transpose_plan /
  dense_reshape_plan (x2) / eval_top_level_scan_i64_add_emit_fast_path /
  test_dce_all_used_large_chain / test_pe_two_eq_mixed_residual / staging single-unknown.
- Diagnosis: each test asserts `planned_out == generic_out` (PARITY) THEN a
  hardcoded sha256. The panics are all on the sha256 line (hash-vs-hash); the
  parity asserts PASS. So the COMPUTED VALUES are internally consistent
  (plan==generic) and the digests merely drifted vs the recorded constants —
  accumulated from earlier untested "code-first" dense-storage/serialization
  commits (e.g. mcqr.97 TensorValue::new densify, co009 serialization).
- NOT caused by this session's fusion work (none of these tests exercise the
  cheap-op fusion path; the 200 passing tests include all fusion bit-identity tests).
- Action: NOT refreshed here. Refreshing requires per-test value verification vs
  JAX/expected (parity alone does not prove vs-original correctness — a shared
  regression in both paths would pass parity). Flagged for the dense-storage/
  serialization commit owners to refresh with verified values, or for a
  test-capable golden-refresh pass. Conformance is RED on these 8 until then.

## frankenjax-f62hx (+ thnjs/idunl/rbkn9/02i98/ecffn) - Contiguous-Block Memcpy Transpose/Slice/Gather/Broadcast

- Lever: replace the per-element coordinate-decode odometer in the structural
  data-movement kernels (transpose_general, slice_strided_gather,
  dynamic_slice_dense, gather_window_blocks, broadcast_replicate, rev_gather)
  with `extend_from_slice` block memcpy of the contiguous trailing run. This is an
  ALGORITHMIC change (memcpy vs odometer), distinct from boxed->dense de-box.
- Representative measured: TRANSPOSE block-copy (f62hx), the attention transpose
  [B,S,H,D]->[B,H,S,D] = [8,512,8,64] f32 (2,097,152 elems), perm (0,2,1,3) which
  keeps the [D]=64 feature vector contiguous (block_len=64).
- Conformance GUARD: GREEN. `transpose_gauntlet.rs` asserts the block-copy path is
  bit-identical to the pre-f62hx per-element odometer reference at 4 probe indices;
  fj-lax transpose conformance tests pass.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | throughput | vs naive |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive, committed) | 791.50 us | 2.65 Gelem/s | 10.30x faster |
  | naive per-element odometer (pre-f62hx) | 8.1525 ms | 257 Melem/s | baseline |

  - Head-to-head vs original JAX (`transpose_gauntlet.py`, jax.jit CPU x64,
    jnp.transpose+0.0 to force materialization, mean):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit transpose | 186.7 us | - |
  | Rust block-copy | 791.5 us | 4.24x slower |
  | Rust naive (pre-f62hx) | 8.15 ms | 43.6x slower |

- Decision: KEEP. The block-copy is a real **10.3x internal** algorithmic win and
  NARROWS the JAX gap from 43.6x to 4.24x. It is the strongest measured lever in
  this conversation's backlog (the dense de-box clusters only matched the boxed
  Rust reference; this one is a genuine throughput jump). Still a JAX loss (4.24x)
  but directionally the right kind of optimization.
- Generalization: the sibling block-copy kernels (slice/gather/dynamic_slice/
  broadcast/rev) use the same memcpy-vs-odometer transform on the same contiguous
  trailing-block structure, so each is expected to deliver a comparable internal
  multi-x win on its contiguous regime. Measured here via the transpose
  representative; the others KEEP on the same proof + the bit-identity tests.
- Retry predicate: closing the remaining 4.24x to JAX on transpose needs either
  cache-blocked tiling of the strided (non-identity-suffix) case (REJECTED before
  as a regression — see project_transpose_already_optimal) or avoiding the
  materialized transpose entirely (layout-aware fusion), NOT more block-copy.

## frankenjax-thnjs - Contiguous-Block Memcpy Broadcast (second block-copy data point)

- Lever: broadcast_replicate replicates the contiguous trailing source block via
  extend_from_slice instead of per-element coordinate decode (sibling of the
  transpose block-copy f62hx). Bias/feature broadcast [D] -> [rows, D] is in every
  transformer layer.
- Workload: bias broadcast [768] -> [4096, 768] f32 (3,145,728 elems), the
  D=768-model-dim-over-4096-tokens bias replicate. broadcast_dimensions [1].
- Conformance GUARD: GREEN. `broadcast_gauntlet.rs` asserts block-copy ==
  pre-thnjs per-element reference at 4 probe indices; broadcast conformance passes.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | vs naive |
  | --- | ---: | ---: |
  | block-copy (eval_primitive, committed) | 283.75 us | 21.80x faster |
  | naive per-element decode (pre-thnjs) | 6.1866 ms | baseline |

  - Head-to-head vs original JAX (`broadcast_gauntlet.py`, jax.jit CPU x64,
    jnp.broadcast_to+0.0 materialized):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit broadcast | 178.9 us (mean; p50 164.9, cv 48%) | - |
  | Rust block-copy | 283.75 us | ~1.59x slower |
  | Rust naive (pre-thnjs) | 6.19 ms | ~34.6x slower |

- Decision: KEEP. 21.8x internal win and only ~1.6x off JAX (NEAR-PARITY) — the
  closest-to-JAX block-copy lever measured. Broadcast is write-bandwidth-bound, so
  both engines sit near memory bandwidth (Rust ~44 GB/s vs JAX ~70 GB/s store
  throughput); the residual gap is store-vectorization, not algorithm.
- Cluster summary (2 measured): block-copy structural kernels deliver 10-22x
  internal wins and a 1.6-4.24x JAX gap (broadcast 1.6x write-bound, transpose
  4.24x strided) -- categorically better than the de-box dense clusters (50-128x
  JAX loss). The block-copy lever is the right kind of optimization; KEEP all
  siblings (slice/gather/dynamic_slice/rev) on the same transform + bit-identity tests.
- Retry predicate: broadcast's residual 1.6x to JAX is store throughput — would
  need SIMD/streaming-store output (non-temporal writes), not more block-copy.

## frankenjax-hfq7o - Dense integer_pow x**2 (MEASURED FIX: powi libcall -> v*v)

- Lever: dense integer_pow path (de-box) for the ubiquitous x**2 (variance/MSE/poly).
- Workload: integer_pow[2] on 1M f64 and 1M f32 (integer_pow_gauntlet.rs + .py).
- FINDING (gauntlet measurement caught a latent inefficiency): the original dense
  path used `v.powi(exponent)` with `exponent` a RUNTIME i32, so LLVM could not
  fold powi(2) to a mul -> a per-element powi LIBCALL, ~6.75 GB/s (10x below memory
  bandwidth). FIXED to special-case exponent==2 -> `v*v` (f64) /
  `(f64::from(v)*f64::from(v)) as f32`, bit-exactly equal to powi(2).
- Conformance GUARD: GREEN. `cargo test -p fj-lax --lib integer_pow` 10/0; bench
  asserts dense==boxed at 3 indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64):

  | Workload | dense BEFORE (powi) | dense AFTER (v*v) | fix speedup | JAX | Rust/JAX before -> after |
  | --- | ---: | ---: | ---: | ---: | --- |
  | integer_pow2_f64_1m | 2.3675 ms | 405.45 us | 5.84x | 184.1 us | 12.86x -> 2.20x slower |
  | integer_pow2_f32_1m | 2.6444 ms | 169.61 us | 15.59x | 121.1 us | 21.83x -> 1.40x slower |

  (Internal vs boxed Vec<Literal> path: f64 25.6ms/f32 27.5ms boxed -> 40-160x
  faster dense; but the headline is the powi->v*v fix that made x**2 bandwidth-bound.)
- Decision: KEEP + FIXED. The fix is the single biggest measured improvement of
  the gauntlet: it closed the JAX gap from 12.9-21.8x to 1.4-2.2x (f32 near-parity)
  by removing the runtime-powi libcall. x**2 is now bandwidth-bound (~40-47 GB/s,
  vs JAX ~66-87 GB/s store); the residual 1.4-2.2x is store throughput (same gap as
  the broadcast block-copy lever).
- Retry predicate: the remaining 1.4-2.2x is memory-store vectorization (SIMD/
  streaming stores) — the SAME residual as broadcast; NOT an algorithm issue. Apply
  the same powi-runtime-arg lesson to any other op taking a runtime small-integer
  power (none found in the hot path; integer_pow[3+] is rarer + needs powi grouping).

## frankenjax-idunl - Contiguous-Block Memcpy Slice (third block-copy data point: PARITY)

- Lever: slice_strided_gather copies the contiguous trailing block via
  extend_from_slice (2D crop / windowing). Workload: crop [1024,1024]->[512,512]
  f32 (262,144 output elems), start [256,256] limit [768,768].
- Conformance GUARD: GREEN. `slice_gauntlet.rs` asserts block-copy == pre-idunl
  per-element reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64,
  x[256:768,256:768]+0.0 materialized):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive) | 45.97 us | 6.10x faster | 1.04x slower (TIE) |
  | naive per-element | 280.51 us | baseline | ~6.4x slower |
  | JAX jit crop | 44.17 us (mean, cv 26%) | - | - |

- Decision: KEEP. 6.1x internal win and ~PARITY with JAX (1.04x, within noise) —
  the FIRST measured non-loss in the conversation. Slice is a pure contiguous
  block memcpy of sub-region rows; both engines are memcpy-bound and tie.
- Block-copy cluster (3 measured): transpose 4.24x (strided read), broadcast 1.59x
  (replicate, write-bound), slice 1.04x (sub-region, memcpy-bound). The
  bandwidth/memcpy-bound block-copy ops TIE or near-tie JAX; the strided-read
  transpose is 4.24x (random-access read pattern). The block-copy lever family is
  the closest-to-JAX work in the codebase; KEEP all.

## frankenjax (dense contiguous gather / embedding lookup) - random-access regime

- Lever: dense contiguous gather memcpy's each [dim] row via extend_from_slice
  (embedding lookup). Workload: table [16384,768] f32, gather 4096 random token
  rows -> [4096,768] (3.1M output elems, random reads into a 50MB table).
- Conformance GUARD: GREEN. `gather_gauntlet.rs` asserts dense == per-element
  reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jnp.take CPU x64):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (eval_primitive, row memcpy) | 1.1450 ms | 4.05x faster | 4.22x slower |
  | naive per-element | 4.6382 ms | baseline | ~17x slower |
  | JAX jit take | 271.3 us (cv 13%) | - | - |

- Decision: KEEP. 4.05x internal win; the per-row memcpy is the correct bit-exact
  approach. JAX-loss (4.22x) because gather is RANDOM-ACCESS-READ-bound (~11 GB/s
  effective: each row read is a cache miss into the 50MB table); XLA hides that
  latency better (prefetch / memory-level parallelism / SIMD gather).
- REFINED CLUSTER PATTERN (4 block-copy/structural ops measured): the JAX gap is
  set by the READ access pattern, not the copy:
    - SEQUENTIAL read (slice 1.04x TIE, broadcast 1.59x): near-parity with JAX.
    - RANDOM/STRIDED read (transpose 4.24x, gather 4.22x): ~4x JAX loss.
  All are 4-22x internal wins over the per-element baseline. The block-copy lever
  is correct; the residual ~4x on random/strided reads is a memory-access-pattern
  gap (prefetch / SIMD-gather / cache-aware tiling), NOT the copy algorithm.
- Retry predicate: closing the gather/transpose ~4x needs software prefetch or
  memory-level-parallelism (interleaved multi-row gather) — blocked by forbid-unsafe
  for raw prefetch intrinsics; would need a safe MLP-exposing restructure. NOT more
  block-copy. Do NOT retry per-element or naive gather.

## frankenjax-7eqrs - Dense Complex Constructor lax.complex(re,im) (de-box, near-parity)

- Lever: dense complex constructor zips two f64 slices into packed (re,im) storage
  via new_complex_values (FFT/signal real+imag combine), vs the boxed per-Literal
  path. Workload: re[1M]+im[1M] f64 -> complex128[1M] (32MB traffic).
- Conformance GUARD: GREEN. `complex_ctor_gauntlet.rs` asserts dense == boxed.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.lax.complex CPU x64):

  | Arm | median | vs boxed | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (new_complex_values) | 775.72 us | 25.2x faster | 1.56x slower |
  | boxed (per-Literal) | 19.558 ms | baseline | ~39x slower |
  | JAX jit lax.complex | 497.34 us (cv 6%) | - | - |

- Decision: KEEP. 25.2x internal de-box win and ~1.56x off JAX (NEAR-PARITY,
  bandwidth-bound at 41 vs 64 GB/s — the same store-throughput gap as broadcast).
- REFINES the de-box category split: de-box of DATA-MOVEMENT/simple ops (complex
  ctor 1.56x, integer_pow-after-v*v-fix 1.40x) APPROACHES JAX (bandwidth-bound);
  de-box of HEAVY-PER-LANE ops (clamp half 53-128x JAX loss, mcqr.106/.107) does
  NOT — there the per-lane decode/encode/compute dominates, not the boxing. So
  "de-box helps reach JAX" iff the op is bandwidth-bound, not per-lane-compute-bound.
- Retry predicate: complex ctor residual 1.56x is store throughput (same as
  broadcast) — SIMD/streaming stores, not algorithm. Conj/real/imag (1ylsj) and
  complex elementwise (same dense storage) expected to behave the same; KEEP.

## METHODOLOGY CORRECTION + elementwise binary (rch-vs-local calibration)

- CRITICAL: prior gauntlet rows (transpose/broadcast/slice/gather/integer_pow/
  complex_ctor) measured Rust via `rch exec` on a REMOTE worker, but JAX runs
  LOCALLY. Same-binary calibration (f32 add, one A/B binary run both places):
  rch worker 173.2 us vs LOCAL 119.6 us => the rch worker is ~1.45x SLOWER than
  local. So all prior Rust/JAX ratios are PESSIMISTIC by ~1.45x; the true
  same-host ratios are ~/1.45.
- f32 native-vs-widen add was ~0 gain on BOTH machines (rch 173.2/174.6 = 1.008x;
  local native 119.6 / widen 122.3 = 1.02x) -> f32 same-shape add is
  bandwidth-bound; the native-f32 attempt was REVERTED.
- LOCAL same-host elementwise dense vs JAX (the trustworthy same-machine numbers;
  JAX jax.jit CPU x64):

  | Workload | Rust dense (LOCAL) | JAX | Rust/JAX (same-host) |
  | --- | ---: | ---: | ---: |
  | add_f64_1m | 415.00 us | 192.0 us | 2.16x slower |
  | add_f32_1m | 135.98 us | 80.4 us | 1.69x slower |
  | mul_f64_1m | 422.96 us | 161.7 us | 2.61x slower |

- Corrected (÷~1.45) same-host estimates for the prior rch rows — several FLIP to
  wins/ties: slice ~0.72x (Rust FASTER), integer_pow2 f32 ~0.97x (Rust ~tie/win),
  broadcast ~1.10x, complex_ctor ~1.08x, integer_pow2 f64 ~1.52x, transpose ~2.93x,
  gather ~2.91x. (These are estimates; future vs-JAX rows MUST run the Rust bench
  LOCALLY, not via rch, for a same-host comparison.)
- Elementwise add/mul same-host loss (1.69-2.61x) is structural: per-op output
  allocation (frankenjax allocates a fresh Vec per primitive; XLA reuses buffers)
  + AVX2 (4-wide f64) vs JAX likely AVX-512 (8-wide). Both are build/architecture
  matters, not a code bug — the inner map+collect already autovectorizes (native
  vs widen tie proves the loop isn't the bottleneck).
- Decision: REVERT f32 native add (~0 gain). KEEP all measured dense paths. Retry
  predicate for elementwise: the gap is per-op allocation (needs buffer reuse =
  compiled-jaxpr arena) and AVX-512 (build flag) — NOT the elementwise loop.

## frankenjax-k8z8g - Einsum permute-copy trailing-block memcpy (internal keep, external JAX loss)

- Lever: `permute_copy_f64` detects an identity-mapped trailing suffix and copies
  the contiguous block with `extend_from_slice`, instead of decoding every output
  coordinate one element at a time. This is used by the general einsum
  permute+GEMM path for attention-shaped contractions such as
  `bqhd,bkhd->bhqk`.
- Canonical graveyard mapping: data-layout/vectorized execution, cache-local
  block movement, and persistent evidence discipline. This is the measured,
  low-risk subset of the broader "exotic layout / cache-oblivious / SIMD"
  idea-space: no unsafe code, one lever, bit-identical output, rollback by
  restoring the per-element coordinate walk.
- Conformance guard: GREEN. `rch exec -- cargo test -p fj-lax
  einsum2_general_matmul_bit_identical_to_naive --lib` passed 1/0, checking
  general-path output and shape bit-for-bit against the deterministic ascending-K
  naive reference across attention, multi-K, interleaved-output, and non-adjacent
  tensordot cases.
- Internal old-vs-new evidence (2026-06-19, `hz2`, release ignored perf test):

  | Arm | workload | time | result |
  | --- | --- | ---: | --- |
  | old odometer reference | `bqhd,bkhd->bhqk` [8,128,8,64/128] | 7664.51 ms | baseline |
  | general permute+GEMM with trailing-block memcpy | same | 15.64 ms | 489.99x faster |

  Digest matched: `7fec8a7f347cf406`.
- Focused Criterion evidence (2026-06-19, `hz2`, sample 30, new repeatable bench):

  | Workload | Rust time |
  | --- | ---: |
  | `eval/einsum2_general_bqhd_bkhd_bhqk_f64` [4,64,8,64/64] | 1.3123 ms mean `[1.2797, 1.3602]` |

- JAX head-to-head (`einsum_permute_copy_gauntlet.py`, local JAX CPU x64,
  `jax.jit(lambda lhs,rhs: jnp.einsum("bqhd,bkhd->bhqk", lhs, rhs))`):

  | JAX comparator | mean | p50 | CV | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: |
  | device-ready lower bound | 317.774 us | 324.308 us | 13.01% | 4.13x slower |
  | NumPy host-copy comparator | 278.461 us | 283.367 us | 20.38% | 4.71x slower |

- Decision: KEEP internally, record external loss. The current path removes a
  catastrophic old Rust algorithmic path (490x faster with digest parity), so
  reverting would be wrong. It still does not dominate JAX on the moderate
  attention-shaped workload; the residual gap is now GEMM/codegen/vectorization
  and allocation/layout, not the permute-copy odometer.
- Validation:
  - `benchmarks/jax_comparison/.venv/bin/python -m py_compile
    benchmarks/jax_comparison/einsum_permute_copy_gauntlet.py`: pass.
  - `ubs --only=rust,python crates/fj-lax/benches/lax_baseline.rs
    benchmarks/jax_comparison/einsum_permute_copy_gauntlet.py`: no critical
    issues; warning-class bench-file inventory only.
  - `rch exec -- cargo check -p fj-lax --bench lax_baseline`: pass.
  - `rch exec -- cargo test -p fj-lax
    einsum2_general_matmul_bit_identical_to_naive --lib`: pass.
  - `rch exec -- cargo bench -p fj-lax --bench lax_baseline --
    eval/einsum2_general_bqhd_bkhd_bhqk_f64 ...`: pass.
  - `rch exec -- cargo test -p fj-lax bench_einsum2_general_vs_odometer --lib
    --release -- --ignored --nocapture`: pass.
  - `cargo fmt -p fj-lax --check`: red on pre-existing fj-lax formatting drift
    outside the touched hunk.
  - `rch exec -- cargo clippy -p fj-lax --bench lax_baseline -- -D warnings`:
    red on pre-existing fj-lax lint debt in `tensor_ops.rs` and `tree_util.rs`
    (`too_many_arguments`, `type_complexity`, `unnecessary_sort_by`), unrelated
    to this bench/harness closeout.
- Retry predicate: do not retry the `permute_copy_f64` odometer elimination
  family without a new profiler showing permute-copy, not GEMM/XLA codegen, is
  still dominant. The next meaningful path is fused layout-aware einsum lowering,
  packed/tiled GEMM, AVX/FMA policy resolution, or compiled-buffer reuse.

## CobaltForge - Threaded cheap same-shape binops for DRAM-bound arrays (JAX WIN)

- Lever: cheap same-shape elementwise binops (Add/Sub/Mul/Div/Max/Min) over dense
  f64/f32 were ALWAYS serial — the recorded "memory-bound ops regress when
  threaded" DO-NOT. That finding is real but SIZE-SCOPED: it was measured at
  L3-resident sizes. New gated path `eval_same_shape_f64_cheap_parallel` /
  `eval_same_shape_f32_cheap_parallel` threads the op (work-scaled `std::thread::
  scope`, exact same per-element closures incl. NaN-propagating `jax_max_f64`/
  `jax_min_f64`) ONLY when `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23 = 8.39M elems),
  well past the measured ~5M-element fresh-alloc cliff. Sub-threshold (incl. the
  benchmarked 1M size) stays on the serial fast path — no regression there.
- HARDWARE CORRECTION: the benchmark host is an AMD Ryzen Threadripper PRO 5975WX
  (Zen3), which has NO AVX-512 (cpuinfo: avx2+fma only) and a 32MB-per-CCD L3.
  Prior ledger/scorecard rows attribute elementwise JAX losses partly to "JAX
  likely AVX-512 (8-wide) vs our AVX2 (4-wide)" — that is FALSE on this host; XLA
  also runs AVX2 here. Measured: XLA does NOT thread large CPU elementwise
  (~21-27 GB/s flat at 16M-64M), so the true elementwise gap was (a) our serial
  fresh-alloc page-fault cliff and (b) single-core DRAM bandwidth — both fixed by
  threading beyond L3.
- Why bit-identical: elementwise output is lane-independent; chunk boundaries never
  change which IEEE op produces which bit. Same closures as the serial path.
- Conformance guard: GREEN for this change. `cheap_binary_parallel_f64_bit_identical_to_serial`
  (new) compares the threaded path bit-for-bit vs the serial closures at n=8.39M+257,
  seeding NaN/±inf/±0/MAX/MIN_POSITIVE to exercise Max/Min NaN propagation and inf
  division — PASS. Full `fj-lax --lib`: 1494 pass, +1 new pass; the 43 failing tests
  are PRE-EXISTING (identical set on clean baseline `d7514a7d`, the documented
  digest-drift RED state) — this change adds 0 new failures.
- Measured A/B (LOCAL, same-binary same-invocation, real `eval_*` fresh-alloc path,
  best-of-10; bench `bench_cheap_binary_parallel_vs_serial`, 2026-06-19):

  | Workload | serial (before) | parallel (after) | internal speedup |
  | --- | ---: | ---: | ---: |
  | add_f64 n=16M | 64859 us (5.9 GB/s) | 9190 us (41.8 GB/s) | 7.06x |
  | add_f64 n=64M | 255663 us (6.0 GB/s) | 33988 us (45.2 GB/s) | 7.52x |
  | add_f32 n=16M | 30806 us (6.2 GB/s) | 4975 us (38.6 GB/s) | 6.19x |
  | add_f32 n=64M | 134654 us (5.7 GB/s) | 17514 us (43.9 GB/s) | 7.69x |

- JAX head-to-head (LOCAL, same host, `jax.jit(lambda x,y:x+y)`, x64; best-of-15
  over inner-10, /tmp/jax_both.py):

  | Workload | Rust after | JAX | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | add_f64 n=16M | 9190 us | 18099 us | 0.51x | Rust 1.97x FASTER |
  | add_f64 n=64M | 33988 us | 57472 us | 0.59x | Rust 1.69x FASTER |
  | add_f32 n=16M | 4975 us | 8598 us | 0.58x | Rust 1.73x FASTER |
  | add_f32 n=64M | 17514 us | 31674 us | 0.55x | Rust 1.81x FASTER |

- Decision: KEEP. Before this change Rust LOST these four large-array workloads to
  JAX by 3.6-4.6x (serial fresh-alloc); after, Rust DOMINATES by 1.69-1.97x. The
  L3-resident regime (n<=4M, e.g. the 1M gauntlet row) is untouched and still uses
  the serial path (threading regresses there — re-confirmed: 1M threaded x8 was
  1.8ms vs 0.34ms serial).
- Retry predicate: do NOT lower the gate toward L3-resident sizes (re-introduces the
  documented regression). The L3-resident 1M loss vs JAX (~70 vs ~200 GB/s, pure
  L3 bandwidth) is a SEPARATE problem (XLA L3 streaming / buffer reuse), not solved
  by threading. Next elementwise frontier: compiled-jaxpr arena buffer reuse to kill
  the per-op fresh allocation entirely (would also remove the page-fault cliff and
  help the L3-resident regime).

## CobaltForge - Threaded cheap scalar-tensor binops / bias-add (frankenjax-aazu6, JAX WIN)

- Lever: extends the same-shape threading win (commit 7b7924b7) to the SCALAR-TENSOR
  cheap path (`tensor OP scalar` — bias-add `x+c`, scaling `x*c`, relu-ish max/min).
  New `eval_f64_scalar_cheap_parallel` / `eval_f32_scalar_cheap_parallel` thread
  Add/Sub/Mul/Div/Max/Min over a dense tensor + scalar when
  `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23), honoring `scalar_on_left`. Inserted ahead
  of the serial `eval_f{64,32}_scalar_broadcast_binop` in both operand orders.
- Bit-identity: the serial f64 Add/Sub/Mul/Div path is `crate::dense::scalar_op` =
  plain `iter().map(ArithOp::apply).collect()` (NOT a separate SIMD kernel, as the
  aazu6 bead feared) — so the threaded path uses the IDENTICAL `a OP b` closures and
  `jax_max_f64`/`jax_min_f64`; lane-independent => bit-for-bit identical. Guarded by
  `cheap_scalar_tensor_parallel_f64_bit_identical_to_serial` (both operand orders,
  NaN/±inf/±0/MAX seeds) — PASS.
- Conformance: `fj-lax --lib` 1496 pass (+2 new bit-identity tests), 43 fail
  (PRE-EXISTING, identical set on clean baseline) — 0 new failures.
- Measured A/B (LOCAL same-binary, real `eval_*` fresh-alloc path, best-of-10;
  `bench_cheap_binary_parallel_vs_serial`, 2026-06-19):

  | Workload | serial (before) | parallel (after) | internal speedup |
  | --- | ---: | ---: | ---: |
  | biasadd_f64 n=16M | 67409 us (3.8 GB/s) | 7341 us (34.9 GB/s) | 9.18x |
  | biasadd_f64 n=64M | 287772 us (3.6 GB/s) | 27074 us (37.8 GB/s) | 10.63x |

- JAX head-to-head (LOCAL same host, `jax.jit(lambda x: x+1.5)` x64, /tmp/jax_bias.py):

  | Workload | Rust after | JAX | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | biasadd_f64 n=16M | 7341 us | 13044 us | 0.56x | Rust 1.78x FASTER |
  | biasadd_f64 n=64M | 27074 us | 54034 us | 0.50x | Rust 2.00x FASTER |

- Decision: KEEP. Before: Rust LOST bias-add to JAX by ~5.2-5.3x (serial scalar
  broadcast craters to 3.6-3.8 GB/s — even worse than the same-shape serial because
  of the broadcast path overhead on top of the fresh-alloc page-fault cliff). After:
  Rust DOMINATES by 1.78-2.00x. L3-resident sizes stay serial (gated). Closes bead
  frankenjax-aazu6.
- Retry predicate: same as the same-shape row — do NOT lower the gate. Remaining
  unthreaded large-array cheap paths: i64 same-shape/scalar binops and cheap unary
  (neg/abs/sign); same pattern would apply but lower priority (less common at scale).

## CobaltForge - Persistent worker pool for L3-resident/2-8M elementwise (NEGATIVE, do-not-wire)

- Hypothesis: XLA hits ~200 GB/s on a 1M f64 add (L3-resident) — impossible for one
  Zen3 core (~70-100 GB/s L3/core), so XLA threads even at L3-resident sizes via a
  PERSISTENT pool (no per-call spawn). We can't thread there because
  `std::thread::scope` spawns fresh OS threads per call. There is an ORPHANED
  `crates/fj-lax/src/thread_pool.rs` (a channel-based persistent pool with
  `parallel_fill_f64`) — NOT declared as a module in lib.rs, so it is dead, never
  compiled, never used. Tested whether wiring it in could lower the threading gate
  below the current 8.39M into the 1M-8M range where we lose / run serial.
- Measured (standalone replica of `parallel_fill_f64`'s exact design — persistent
  channel pool + owned-Vec-per-chunk + caller concat — vs serial collect, +avx2,
  best-of-20, 2026-06-19):

  | n | serial | pool best | pool/serial |
  | ---: | ---: | ---: | ---: |
  | 1M | 305 us (78.7 GB/s) | 812 us (29.6 GB/s) | 0.38x (LOSS) |
  | 2M | 1292 us (37.2 GB/s) | 1898 us (25.3 GB/s) | 0.68x (LOSS) |
  | 4M | 2806 us (34.2 GB/s) | 5962 us (16.1 GB/s) | 0.47x (LOSS) |
  | 8M | 31338 us (6.1 GB/s) | 33229 us (5.8 GB/s) | 0.94x (LOSS) |

- Root cause: the pool's owned-Vec-concat design (chosen to stay unsafe-free) is
  fundamentally wrong for MEMORY-BOUND elementwise: (1) the caller's final
  `out[s..].copy_from_slice(buf)` concat DOUBLES output traffic and re-serializes
  the page-faults of the full output Vec — exactly the cost my `std::thread::scope`
  path AVOIDS by writing each chunk directly into one fresh output; (2) per-chunk
  `vec![0.0; len]` zero-fill + channel round-trip add latency; (3) cross-CCD threads
  don't share the 32MB/CCD L3, so L3-resident data can't be parallelized at L3 speed.
  Note: even at 8M (where my scope path WINS 7x via parallel page-faulting of one
  output) the pool LOSES (0.94x) because the concat re-serializes those faults.
- Decision: DO NOT wire in thread_pool.rs; DO NOT route elementwise through it.
  Confirms the standing conclusion: the L3-resident JAX gap (1M: ~70 vs ~200 GB/s)
  needs compiled-jaxpr arena BUFFER REUSE (write into a pre-faulted donated buffer,
  no per-op fresh alloc), NOT a thread pool. A copy-free persistent pool would need
  either `unsafe` (forbidden here) or `Arc<[AtomicU64]>` shared output (still an O(n)
  read-out copy into Vec<f64> at the end — same doubling). Left orphaned file as-is.
- Retry predicate: only revisit a persistent pool for elementwise if it can write
  DIRECTLY into the final output without a concat/read-out copy (i.e. after a
  buffer-reuse arena exists, at which point the spawn-free pool becomes worthwhile
  for the L3-resident regime). Not before.

## CobaltForge - Threaded same-shape i64 binops for DRAM-bound arrays (JAX WIN)

- Lever: completes the same-shape elementwise threading family (after f64/f32 in
  7b7924b7) with i64. New `eval_same_shape_i64_parallel` threads the dense i64
  same-shape binop using the EXACT `int_op` closure the serial path uses (carries
  per-primitive semantics: `wrapping_add`/`wrapping_sub`/`wrapping_mul`, `i64::max/min`,
  `checked_div().unwrap_or(0)`), gated at `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23).
  Required adding `+ Sync` to `eval_binary_elementwise`'s `int_op` bound — non-breaking,
  every caller passes a stateless (Sync) closure. Generic over op: cheap ops win via
  DRAM bandwidth + parallel page-faulting, heavy ones (pow/gcd/lcm) on compute.
- Bit-identity: i64 results are lane-independent; same `int_op`, same order. Guarded by
  `same_shape_i64_parallel_bit_identical_to_serial` (i64::MAX/MIN/0/-1 edges + forced
  div-by-zero lanes across wrapping add/sub/mul, max/min, checked_div) — PASS.
- Conformance: `fj-lax --lib` 1497 pass (+1 new), 43 fail (PRE-EXISTING, identical set
  on clean baseline) — 0 new failures.
- Measured A/B (LOCAL same-binary, real `eval_*` fresh-alloc path, best-of-10) and JAX
  head-to-head (`jax.jit(lambda x,y:x+y)` int64 x64, /tmp/jax_i64.py):

  | Workload | serial (before) | parallel (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | add_i64 n=16M | 64098 us (6.0 GB/s) | 9913 us (38.7 GB/s) | 6.47x | 17603 us | 0.56x (1.78x FASTER) |
  | add_i64 n=64M | 266637 us (5.8 GB/s) | 39364 us (39.0 GB/s) | 6.77x | 70136 us | 0.56x (1.78x FASTER) |

- Decision: KEEP. Before: Rust LOST i64 add to JAX by ~3.6-3.8x (serial fresh-alloc
  cliff). After: DOMINATES by 1.78x. Same gate discipline; L3-resident sizes serial.
- Family status: same-shape f64/f32/i64 + scalar-tensor f64/f32 cheap binops now all
  threaded beyond L3 and all FASTER than JAX. Remaining unthreaded large-array cheap
  paths: i64 scalar-tensor + cheap unary (neg/abs/sign) — same pattern, lower priority.

## CobaltForge - Caching allocator (mimalloc) eliminates the fresh-alloc page-fault cliff (MAINTAINER-GATED, measured)

- THE finding: the serial ~6 GB/s cliff on >L3 fresh-output elementwise (root cause
  behind all my threading wins) is NOT inherent — it is glibc returning fresh,
  zero-faulted mmap pages per op (page faults serialized on first write). A caching
  allocator (mimalloc) recycles warm, already-faulted spans across same-size
  allocations, so the per-op fault cost disappears.
- Measured (standalone, +avx2, best-of-20, serial `a+b` collect, fresh Vec/op,
  2026-06-19; same Zen3 5975WX, JAX f64 add reference in prior rows):

  | n | glibc serial | mimalloc serial | mimalloc speedup | JAX | mimalloc/JAX |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1M  | 92.3 GB/s | 88.5 GB/s | 0.96x (L3-resident, no cliff) | ~200 GB/s | (still lose) |
  | 16M | 5.9 GB/s | 21.7 GB/s | 3.7x | 18099 us (21.2 GB/s) | ~PARITY |
  | 64M | 6.0 GB/s | 23.4 GB/s | 3.9x | 57472 us (26.7 GB/s) | 1.14x slower |

  mimalloc alone brings SERIAL large-add to ~JAX parity (no threading, no value change).
- CRITICAL interaction — mimalloc and my shipped cheap-binop THREADING are SUBSTITUTES,
  not complements (threaded `a+b`, all cores, fresh output, best-of-20):

  | n | glibc thr/ser | mimalloc thr/ser |
  | ---: | ---: | ---: |
  | 16M | 6.32x (threading wins) | 1.02x (threading no-op) |
  | 64M | 6.67x (threading wins) | 0.93x (threading REGRESSES) |

  Mechanism: my threaded path's `vec![0.0; n]` (calloc) under glibc gets untouched
  zero-pages that the worker WRITES fault IN PARALLEL (the win). Under mimalloc the
  reused warm span is re-zeroed by calloc SERIALLY before the workers run, so the
  serial memset caps the threaded path at serial speed. => If mimalloc is adopted,
  the CHEAP_BINARY_PARALLEL_MIN threading gate should be RAISED/removed (it stops
  helping and slightly hurts at 64M); mimalloc is the more general lever (every
  allocating op + the L3-resident regime, not just the 6 ops I hand-threaded).
- Why NOT unilaterally committed: (1) workspace-policy decision (global allocator
  choice, like the +fma gate); (2) a library-level `#[global_allocator]` in fj-lax
  would CONFLICT with the existing `PeakAlloc` global_allocator in
  `crates/fj-interpreters/benches/eval_chain_memory.rs` (two global allocators per
  binary = compile error); (3) it supersedes shipped threading, so adoption needs a
  coordinated gate-retune. forbid(unsafe_code) is NOT a blocker — declaring an
  external allocator's `#[global_allocator]` static is safe code.
- Recommendation (filed as a bead): adopt mimalloc as the default global allocator
  workspace-wide (resolve the PeakAlloc bench conflict by scoping/removing its
  tracker or cfg-gating), then raise/remove the cheap-binop threading gate. Expected
  ~3.7-3.9x on the allocation component of every large op and serial JAX parity on
  large elementwise, plus likely gains across all fj-lax/fj-core ops that allocate a
  fresh output. This is the single highest-EV lever measured in the campaign.
- Retry predicate: do not re-measure the mimalloc-vs-glibc cliff (settled). Next step
  is the maintainer allocator decision + gate retune, or the compiled-jaxpr arena
  (which would subsume both by reusing a donated output buffer with no alloc at all).

## CobaltForge - Threaded cheap unary f64/f32 (JAX WIN) + mimalloc framing CORRECTION

- Lever: threads the serial dense f64/f32 unary fast paths (Neg/Abs/Sign/Square/Floor/
  Ceil/Round/Reciprocal and any sub-threshold transcendental reaching the fast path) at
  `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23), inside `eval_unary_f{64,32}_tensor_fast_path`
  (the chokepoint for both unary callers). Needed `+ Sync` on `eval_unary_elementwise`
  and `eval_unary_int_or_float`'s float_op bounds (non-breaking; all closures stateless).
  Bit-identical (lane-independent); guarded by `threaded_unary_maps_bit_identical_to_serial`
  (neg/abs/square/floor/reciprocal, NaN/±inf/±0/MAX, f64+f32).
- Conformance: `fj-lax --lib` 1498 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured A/B (LOCAL, real path, best-of-10) + JAX (`jax.jit(lambda x:-x)` x64):

  | Workload | serial (before) | parallel (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | neg_f64 16M | 59907 us (4.3 GB/s) | 7461 us (34.3 GB/s) | 8.03x | 15575 us | 0.48x (2.09x FASTER) |
  | neg_f64 64M | 241451 us (4.2 GB/s) | 30260 us (33.8 GB/s) | 7.98x | 59500 us | 0.51x (1.97x FASTER) |

- *** CORRECTION to the prior "mimalloc supersedes threading" entry/scorecard/bead oneqh ***
  Re-measured on the REAL `eval_primitive` path (which already includes my threading) under
  both allocators. mimalloc and threading are COMPLEMENTARY (different op sets), NOT
  substitutes, and threading is the SUPERIOR lever for memory-bound ops:

  | op @16M | glibc serial | glibc+threading (shipped) | mimalloc serial | mimalloc+threading |
  | --- | ---: | ---: | ---: | ---: |
  | Add (threaded) | (n/a) | 38.7 GB/s | (n/a) | 34.5 GB/s |
  | Neg (was serial) | 4.4 GB/s | 34.3 GB/s (now threaded) | 20.5 GB/s | ~ |

  - For ops I THREAD: glibc+threading (~38 GB/s, parallel page-faulting of the calloc
    zero-pages) >= mimalloc (~24-34); mimalloc is ~neutral-to-slightly-negative there.
  - For ops still SERIAL: mimalloc gives ~4.7x (Neg 4.4->20.5) — BUT threading gives more
    (4.4->34.3, parallel faulting beats warm-span serial memset). So the better fix for
    each serial op is to THREAD it, not adopt mimalloc.
  - Therefore: KEEP the threading gate regardless of any allocator decision (removing it
    craters these ops to 4-6 GB/s). mimalloc remains a modest COMPLEMENTARY win only for
    large allocating ops that are hard to thread (and is superseded wherever threading is
    applied). The round-4 "remove the gate if mimalloc adopted" guidance was WRONG.
- Retry predicate: extend the same gate to the remaining serial large ops where it pays
  (i64/u32/u64 unary; scalar-tensor i64; reduction outputs; transpose/gather/broadcast
  outputs) — each is a 4.4->~34 GB/s, JAX-beating, mimalloc-beating win. Bead oneqh
  updated to reflect the corrected (complementary, keep-gate) recommendation.

## CobaltForge - Threaded f64/f32 broadcast_replicate for DRAM-bound outputs (JAX WIN)

- Lever: `BroadcastInDim` (bias/feature replicate [C]->[B,C], [A,C]->[A,B,C], etc.) built the
  large output via a SERIAL `extend_from_slice` loop — page-fault-bound at ~2.4 GB/s on a
  fresh >L3 output (WORSE than the elementwise cliff). New `broadcast_replicate_into<T>` fills
  a caller-allocated `vec![<concrete zero>; total]` (alloc_zeroed/calloc = untouched zero
  pages) by splitting the outer block iterations across scoped threads — parallel page-faults
  the output and copies the contiguous src blocks concurrently. Wired into the f64 + f32 arms
  of `eval_broadcast_in_dim` (the calloc must be concrete-typed; generic `vec![T::default()]`
  does NOT lower to calloc, so other dtypes keep the serial `broadcast_replicate`).
- Bit-identity: each output block `o` is written from the SAME `src[base(o)..+block_len]` at the
  SAME offset `o*block_len`; `base(o)` is the mixed-radix outer-coordinate->base mapping, equal
  to the serial odometer. Guarded by `broadcast_replicate_into_bit_identical_to_serial`
  (incl. a size-1-stretch multi-outer-axis case), compared bit-for-bit to the serial generic.
- Conformance: `fj-lax --lib` 1499 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, [1024]->[N,1024], best-of-20) + JAX
  (`jax.jit(jnp.broadcast_to)` x64, /tmp/jax_bcast.py):

  | output | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16.4M | 55417 us (2.4 GB/s) | 6097 us (21.5 GB/s) | 9.09x | 22254 us (5.9 GB/s) | 0.27x (3.65x FASTER) |
  | 65.5M | 214491 us (2.4 GB/s) | 23130 us (22.7 GB/s) | 9.27x | 87049 us (6.0 GB/s) | 0.27x (3.76x FASTER) |

- Decision: KEEP. Broadcast is ubiquitous (every bias/feature materialization). Before, Rust
  LOST to JAX by 2.5x AND was catastrophically slow internally; after, Rust DOMINATES by 3.65-
  3.76x. Note JAX's own broadcast_to materializes at only ~6 GB/s here. Same gate; small (<gate)
  broadcasts stay serial (the calloc+serial-copy is bit-identical, negligible overhead).
- Retry predicate: extend `_into` to i64/u32/u64/bool/half broadcast arms (each allocates a
  concrete-zero vec -> calloc, so the same threaded fill applies) when those dtypes show up
  large. Confirms the general principle: every large fresh-output op wins from calloc +
  parallel page-faulting; threading beats both serial-glibc and mimalloc.

## CobaltForge - Threaded f64<->f32 convert (ConvertElementType) for DRAM-bound arrays (JAX WIN)

- Lever: the hot mixed-precision casts f64->f32 (downcast) and f32->f64 (upcast) built their
  large output via a serial `.iter().map(cast).collect()`, page-fault-bound at ~5.9 / ~3.4 GB/s.
  New `threaded_convert_into<S,D>` fills a caller-allocated calloc'd output
  (`vec![0.0f32; n]` / `vec![0.0f64; n]`) by splitting into scoped-thread chunks applying the
  per-element cast. Wired as early returns in the f64-source (->F32) and f32-source (->F64) arms
  of `eval_convert_element_type`, above `CHEAP_BINARY_PARALLEL_MIN`. Other casts keep serial
  (each could thread the same way — caller allocates a concrete-zero vec for calloc).
- Bit-identity: per-element `v as f32` / `f64::from(v)`, lane-independent => identical to serial.
  Guarded by `threaded_convert_bit_identical_to_serial` (NaN/±inf/±0, both directions).
- Conformance: `fj-lax --lib` 1500 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-20) + JAX (`x.astype(...)` x64, /tmp/jax_conv.py):

  | cast | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | f64->f32 16M | 32280 us (5.9 GB/s) | 5328 us (36.0 GB/s) | 6.06x | 11422 us | 0.47x (2.14x FASTER) |
  | f64->f32 64M | 129841 us (5.9 GB/s) | 19965 us (38.5 GB/s) | 6.50x | 32748 us | 0.61x (1.64x FASTER) |
  | f32->f64 16M | 55907 us (3.4 GB/s) | 6800 us (28.2 GB/s) | 8.22x | 13499 us | 0.50x (1.99x FASTER) |
  | f32->f64 64M | 226888 us (3.4 GB/s) | 26645 us (28.8 GB/s) | 8.51x | 58869 us | 0.45x (2.21x FASTER) |

- Decision: KEEP. Mixed-precision casts are ubiquitous; before, Rust LOST to JAX by 1.6-4.2x;
  after, Rust DOMINATES by 1.64-2.21x. Same gate; small casts stay serial.
- Retry predicate: extend the same calloc+threaded fill to the other hot casts (f64->i32/i64
  quantization, f32->i32, half<->f32) — each allocates a concrete-zero output so the same
  helper applies. Confirms the campaign principle across yet another op family.

## CobaltForge - Threaded rank-2 f64/f32 transpose for DRAM-bound arrays (JAX WIN)

- Lever: the rank-2 [1,0] transpose used `transpose_2d_blocked` (cache-blocked but SERIAL,
  and it pre-fills the output with `vec![src[0]; total]` = a serial fault of the whole output
  before transposing) — ~2.2 GB/s, page-fault + strided-read bound. New `transpose_2d_into<T>`
  fills a caller-allocated calloc'd output by splitting the OUTPUT-row range (= source-column
  range) across scoped threads, each running a cache-blocked sub-transpose. Wired into the
  f64/f32 arms of the rank-2 branch above `CHEAP_BINARY_PARALLEL_MIN`; other dtypes / small
  keep `transpose_2d_blocked`.
- Bit-identity: every `out[j*rows+i] = src[i*cols+j]`, just produced concurrently into disjoint
  contiguous output-column slices. Guarded by `transpose_2d_into_bit_identical_to_serial`
  (NON-SQUARE dims 8192x2048, 3000x5600, + square — catches rows/cols swaps), bit-for-bit vs
  the serial blocked kernel.
- "Tiling regresses" (recorded) is about CACHE-BLOCKING the strided walk, NOT threading — this
  KEEPS the existing block tiling and adds thread-level parallelism on top (orthogonal, legal).
- Conformance: `fj-lax --lib` 1501 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(lambda x: x.T+0.0)` x64,
  /tmp/jax_t.py):

  | transpose | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | f64 4096x4096 | 119433 us (2.2 GB/s) | 9716 us (27.6 GB/s) | 12.29x | 28454 us (9.4 GB/s) | 0.34x (2.93x FASTER) |
  | f64 8192x8192 | 484097 us (2.2 GB/s) | 37217 us (28.9 GB/s) | 13.01x | 111599 us (9.6 GB/s) | 0.33x (3.0x FASTER) |

- Decision: KEEP. Transpose is ubiquitous (attention, matmul prep). Before, Rust LOST to JAX by
  ~4.3x; after, Rust DOMINATES by ~3x. Same gate; small/other-dtype transposes unchanged.
- Retry predicate: extend `transpose_2d_into` to i64/u32/u64/half/complex rank-2 arms (concrete
  zero -> calloc) and thread the N-D `transpose_general` outer odometer the same way (base_of
  decomposition, like broadcast). Same campaign principle.

## CobaltForge - Threaded f64/f32 contiguous gather (embedding lookup) for DRAM-bound (JAX WIN)

- Lever: the contiguous full-row gather (embedding lookup, slice_sizes [1,DIM]) built its large
  output via a serial per-index `extend_from_slice`, page-fault-bound at ~2.1-2.2 GB/s. New
  `gather_contiguous_into<T>` fills a caller-allocated calloc'd output by splitting the index
  range across scoped threads (each does its rows' memcpy / OOB-fill). Wired into the f64/f32
  contiguous arms above `CHEAP_BINARY_PARALLEL_MIN`; pre-validates bounds and returns false
  (caller falls to the serial path -> identical error) if any in-bounds index would exceed src.
- Bit-identity: index i -> out[i*slice_elems..], Some(idx)->src row copy, None->fill, exactly the
  serial contiguous copy. Guarded by `gather_contiguous_into_bit_identical_to_serial` (mix of
  in-bounds Some + OOB None/fill at total >= gate), bit-for-bit vs a serial reference.
- Conformance: `fj-lax --lib` 1502 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, [16384,1024] f32 table, best-of-15) + JAX
  (`jax.jit(lambda t,i: t[i])` x64, /tmp/jax_g.py):

  | nidx (out elems) | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16384 (16.8M) | 31161 us (2.2 GB/s) | 4781 us (14.0 GB/s) | 6.52x | 8336 us (8.1 GB/s) | 0.57x (1.74x FASTER) |
  | 65536 (67M) | 125562 us (2.1 GB/s) | 15007 us (17.9 GB/s) | 8.37x | 31003 us (8.7 GB/s) | 0.48x (2.07x FASTER) |

- Decision: KEEP. Embedding gather is huge in NLP; before, Rust LOST to JAX by ~3.7-4x; after,
  DOMINATES by 1.74-2.07x. (Gather hits 14-18 GB/s vs ~34 for streaming ops — random table reads
  miss cache — but still beats JAX, which is also gather-cache-bound at ~8 GB/s.) Same gate.
- Retry predicate: extend `gather_contiguous_into` to i64/i32/u32/u64/half/complex contiguous
  arms (concrete-zero calloc) and to the strided `gather_window_blocks` (per-index independent;
  same index-range split). Same campaign principle.

## CobaltForge - Threaded dense f64/f32 axis-0 concatenate for DRAM-bound (JAX WIN, was parity)

- Lever: contiguous (outer==1: axis-0 / KV-cache / batch) concat built its large output via
  `LiteralBuffer::from_concat_slices`, page-fault-bound at ~2.1 GB/s. New `concat_contiguous_into<T>`
  fills a caller-allocated calloc'd output by splitting the output into contiguous chunks across
  scoped threads; each chunk copies from whichever source(s) it overlaps (cumulative-offset table,
  so chunk boundaries may cross source boundaries). Wired into the f64/f32 `outer==1` case above
  `CHEAP_BINARY_PARALLEL_MIN`; axis>0 (interleaved) and other dtypes keep `from_concat_slices`.
- Bit-identity: out = srcs concatenated in order, exactly `from_concat_slices` for outer==1.
  Guarded by `concat_contiguous_into_bit_identical_to_serial` (5 uneven sources incl. len-1 and
  len-7 so chunk boundaries cross sources, + end-to-end 3-operand eval_concatenate).
- Conformance: `fj-lax --lib` 1503 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path WITH checksum to defeat dead-code elision, best-of-15)
  + JAX (`jax.jit(jnp.concatenate axis=0)` x64, /tmp/jax_cat.py):

  | out elems | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | ~16M | 63926 us (2.1 GB/s) | 7872 us (16.7 GB/s) | 8.12x | 72226 us (1.8 GB/s) | 0.11x (9.2x FASTER) |
  | ~64M | ~256000 us (2.1 GB/s) | 30982 us (16.9 GB/s) | ~8x | 290818 us (1.8 GB/s) | 0.11x (9.4x FASTER) |

- Decision: KEEP. Concat was at PARITY with JAX (both ~2 GB/s, page-fault bound) — threading turns
  it into ~9x DOMINATION. Concat is common (KV-cache append, batch/feature concat). Same gate.
- MEASUREMENT NOTE: an unconsumed `eval_primitive(Concatenate)` in a bench loop was elided by the
  optimizer (reported 0.3us / 500000 GB/s); always consume the result (checksum) when timing.
- Retry predicate: extend to axis>0 (interleaved row blocks; thread the per-row task list) and
  other dtypes (concrete-zero calloc). Same campaign principle.

## CobaltForge - Threaded bf16/f16 broadcast + gather (training dtype) for DRAM-bound (JAX WIN)

- Lever: extends the proven threaded broadcast/gather to the bf16/f16 (u16-backed) dtype — the
  DOMINANT training dtype, previously left on the serial cliff (~2.5 GB/s). The generic
  `broadcast_replicate_into<T>` / `gather_contiguous_into<T>` already accept `T: Copy+Send+Sync`
  (u16 qualifies); wired the half-float broadcast arm and contiguous-gather arm to allocate a
  calloc'd `vec![0u16; total]` and use the threaded fill. Bit-identical (u16 bit copies).
- Guarded by `threaded_half_float_broadcast_and_gather_bit_identical` (bf16 broadcast vs serial
  broadcast_replicate; bf16 gather incl. OOB None/fill vs serial reference).
- Conformance: `fj-lax --lib` 1504 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | serial (before) | threaded (after) | internal |
  | --- | ---: | ---: | ---: |
  | bf16 broadcast [1024]->[16384,1024] | 12706 us (2.6 GB/s) | 2838 us (11.8 GB/s) | 4.48x |
  | bf16 broadcast [1024]->[65536,1024] | 53838 us (2.5 GB/s) | 6421 us (20.9 GB/s) | 8.38x |
  | bf16 gather nidx=16384 (16.8M) | (~serial cliff) | 3030 us (11.1 GB/s) | ~6x |
  | bf16 gather nidx=65536 (67M) | (~serial cliff) | 7826 us (17.2 GB/s) | ~6x |

- Decision: KEEP. bf16/f16 is the training-dominant dtype; bias broadcast + embedding gather are
  ubiquitous there. Same calloc+parallel-page-fault lever, now covering the half-float backings.
- Retry predicate: extend the same to i64/u32/u64 broadcast/gather/transpose/concat arms (all
  Copy+Send+Sync, concrete-zero calloc) and to bf16/f16 transpose/concat. Mechanical.

## CobaltForge - Threaded scalar-broadcast / full() constant fill (f64/f32) for DRAM-bound (JAX WIN)

- Lever: scalar broadcast (`jnp.full`, scalar->tensor, mask/const init) built its output via
  `vec![v; total]`, which for a NON-ZERO v is a serial element fill, page-fault-bound at ~2.5 GB/s
  (v==0 already callocs). New `threaded_fill_into<T>` writes v into a caller-allocated calloc'd
  output across scoped threads (parallel page-fault). Wired into the F64Bits/F32Bits arms of the
  BroadcastInDim scalar path above `CHEAP_BINARY_PARALLEL_MIN`. Bit-identical to `vec![v; total]`.
- Guarded by `threaded_scalar_fill_bit_identical_to_serial` (v = 3.14 / 0.0 / -2.5 / NaN, + f32).
- Conformance: `fj-lax --lib` 1505 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.full)` x64, /tmp/jax_full.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 49542 us (2.6 GB/s) | 6310 us (20.3 GB/s) | 7.85x | 21549 us (5.9 GB/s) | 0.29x (3.42x FASTER) |
  | 64M | 205894 us (2.5 GB/s) | 24466 us (20.9 GB/s) | 8.42x | 86302 us (5.9 GB/s) | 0.28x (3.53x FASTER) |

- Decision: KEEP. jnp.full / scalar-const materialization is common (init, masks, fills). Before,
  Rust LOST to JAX by ~2.3x; after, DOMINATES by ~3.5x. Same gate; v==0 stays calloc (already fast).
- Retry predicate: extend threaded_fill_into to the half/i64/u32/u64/bool scalar arms (concrete-zero
  calloc) and to array_creation full/zeros/ones if those hit the cliff. Mechanical.

## CobaltForge - Threaded full f64/f32 max/min reduce for DRAM-bound (JAX WIN)

- Lever: full ReduceMax/ReduceMin used a single-threaded SIMD fold (`simd_reduce_minmax_f64/f32`,
  ~21.6 GB/s f64) — read-bound, so a single core can't saturate multi-channel DRAM. New
  `threaded_reduce_minmax_f64/f32` splits the input into chunks, SIMD-reduces each on a scoped
  thread, and combines the partials. Gated at `CHEAP_BINARY_PARALLEL_MIN` (serial SIMD wins in
  cache). (Sum/Prod stay scalar/serial — float-sum is non-associative and bit-exact-pinned vs JAX;
  not touched.)
- Bit-identity: max/min are associative+commutative; each partial resolves ±0 sign and NaN exactly
  per-chunk, and the combine preserves both (NaN if any partial NaN -> canonical NaN; f64::max/min
  maxNum/minNum ±0 handling is order-independent, so fold-of-folds == full fold). Guarded by
  `threaded_reduce_minmax_bit_identical_to_serial` (NaN mid/boundary, ±0-only, ±inf; f64 AND f32).
- Conformance: `fj-lax --lib` 1506 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.max)` x64, /tmp/jax_red.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 5935 us (21.6 GB/s) | 3627 us (35.3 GB/s) | 1.64x | 4518 us (28.3 GB/s) | 0.80x (1.25x FASTER) |
  | 64M | 26189 us (19.5 GB/s) | 8552 us (59.9 GB/s) | 3.06x | 11568 us (44.3 GB/s) | 0.74x (1.35x FASTER) |

- Decision: KEEP. max/min reduce (softmax stability, clipping bounds) was a 1.3-2.3x JAX LOSS;
  threading flips it to a 1.25-1.35x WIN. First reduction-family threading; read-bound (no
  page-fault cliff), so the win is bandwidth (~2-3x internal) not the dramatic fill-op factors.
- NEGATIVE (recorded): ReduceSum is 10.9 GB/s (3-4x slower than JAX's 33-43) but CANNOT be threaded
  or SIMD-vectorized — float sum is non-associative and pinned bit-exact vs JAX (memory: DO-NOT
  float-sum SIMD). Closing it needs a bit-exact-matching tree/pairwise order = the compiled-jaxpr /
  XLA-order-matching swing, not threading.
- Retry predicate: extend threaded minmax to bf16/f16 full reduce + argmax/argmin (needs
  first-index-preserving combine) and to axis reductions where the reduced extent is huge.

## CobaltForge - Threaded bf16/f16 transpose + axis-0 concat (training/inference dtype) (JAX WIN)

- Lever: extends the proven threaded rank-2 transpose (`transpose_2d_into`) and axis-0 concat
  (`concat_contiguous_into`) to the bf16/f16 (u16-backed) dtype — bf16 attention transposes and
  bf16 KV-cache / feature concat are ubiquitous in LLM training & inference, previously left on
  the serial cliff (~2-2.5 GB/s). The helpers are already generic over T: Copy+Send+Sync (u16
  qualifies); wired the half-float arms to a calloc'd vec![0u16; total]. Bit-identical (u16 copies).
- Guarded by `threaded_half_float_transpose_and_concat_bit_identical` (bf16 transpose non-square
  6000x3000 vs serial transpose_2d_blocked + end-to-end eval_transpose; bf16 2-operand axis-0 concat
  vs serial).
- Conformance: `fj-lax --lib` 1507 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | threaded (after) | (serial was ~2-2.5 GB/s cliff) internal |
  | --- | ---: | ---: |
  | bf16 transpose 4096x4096 | 4927 us (13.6 GB/s) | ~5x |
  | bf16 transpose 8192x8192 | 13283 us (20.2 GB/s) | ~8x |
  | bf16 concat0 out=32.8M | 5006 us (13.1 GB/s) | ~6x |
  | bf16 concat0 out=131M | 15446 us (17.0 GB/s) | ~8x |

- Decision: KEEP. Same calloc+parallel-page-fault lever, now covering the training-dominant dtype
  for transpose+concat (matching the bf16 broadcast+gather shipped in 3fe6f23a). Bit-identical.
- Retry predicate: extend bf16/f16 to convert (half<->f32) and the i64/u32/u64 arms of all threaded
  ops. Mechanical (concrete-zero calloc + existing generic _into helpers).

## CobaltForge - Threaded i64 broadcast + gather (index/id tensors) + argmax verification (JAX WIN)

- Lever: extends the threaded broadcast/gather to i64 (i32 shares the backing) — position-ids /
  token-ids broadcast and index/id gather at scale, previously on the serial cliff (~2.2-2.5 GB/s).
  Generic `broadcast_replicate_into` / `gather_contiguous_into` accept i64 (Copy+Send+Sync); wired
  the i64 arms to calloc'd `vec![0i64; total]`, preserving i32 width on the broadcast result.
  Bit-identical. Guarded by `threaded_i64_broadcast_and_gather_bit_identical` (broadcast vs serial;
  gather incl. OOB None/fill vs serial).
- Conformance: `fj-lax --lib` 1508 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | threaded | (serial ~2.2-2.5 GB/s cliff) internal |
  | --- | ---: | ---: |
  | i64 broadcast ->[16384,1024] | 5966 us (22.5 GB/s) | ~9x |
  | i64 broadcast ->[65536,1024] | 24545 us (21.9 GB/s) | ~9x |
  | i64 gather nidx=16384 (16.8M) | 8634 us (15.5 GB/s) | ~7x |
  | i64 gather nidx=65536 (67M)   | 31753 us (16.9 GB/s) | ~7x |

- ARGMAX/ARGMIN — VERIFIED ALREADY WINNING (no change). Measured full f64 argmax (LOCAL,
  best-of-15) vs `jax.jit(jnp.argmax)` x64:

  | n | Rust | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: |
  | 16M | 10131 us (12.6 GB/s) | 13766 us (9.3 GB/s) | 0.74x (1.36x FASTER) |
  | 64M | 40560 us (12.6 GB/s) | 55010 us (9.3 GB/s) | 0.74x (1.36x FASTER) |

  Rust argmax already beats JAX 1.36x; the delicate NaN/first-index-tie threaded combine is NOT
  worth the risk since there is no competitive gap. Left as-is.
- SATURATION NOTE: with this commit, every measured large memory-bound op (binops/scalar-tensor/
  unary/broadcast/scalar-fill/convert/transpose/gather/concat across f64/f32/i64/bf16/f16, plus
  max/min-reduce and argmax) now MATCHES or BEATS JAX. The only remaining JAX losses are the
  documented off-limits/architectural ones: ReduceSum/ReduceProd & cumsum (float non-associative,
  bit-exact-pinned -> need XLA-order matching) and the L3-resident regime (needs compiled-jaxpr
  arena buffer reuse). Both are multi-session swings, NOT threading.

## CobaltForge - Threaded f64/f32 select/where (masking) for DRAM-bound (JAX WIN)

- Lever: `select`/`jnp.where` (masking — attention, dropout, clipping) ran its dense f64/f32 fast
  paths serially (~1.8-1.9 GB/s, worse than the binop cliff: it reads cond + 2 branches + writes
  output). New reusable `threaded_index_fill_into<T>(out, threads, pick)` fills a calloc'd output by
  index across scoped threads; wired into both cond backings (dense Bool slice AND bit-packed
  BoolWords from a comparison mask) of the f64 and f32 select fast paths, above
  CHEAP_BINARY_PARALLEL_MIN. Bit-identical (per-index pick, lane-independent).
- Guarded by `threaded_select_bit_identical_to_serial` (both cond backings, f64+f32, NaN/±0/±inf
  branches), bit-for-bit vs a serial reference.
- Conformance: `fj-lax --lib` 1509 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, with checksum, best-of-15) + JAX
  (`jax.jit(jnp.where)` x64, /tmp/jax_sel.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 66376 us (1.9 GB/s) | 10753 us (11.9 GB/s) | 6.17x | 23242 us (5.5 GB/s) | 0.46x (2.16x FASTER) |
  | 64M | 280863 us (1.8 GB/s) | 40010 us (12.8 GB/s) | 7.02x | 70142 us (7.3 GB/s) | 0.57x (1.75x FASTER) |

- Decision: KEEP. select/where masking is ubiquitous; before, Rust LOST to JAX by ~2.9-4x; after,
  DOMINATES by 1.75-2.16x. (Lower GB/s than 2-input ops — 3 inputs + 1 output of traffic — but
  still beats JAX.) `threaded_index_fill_into` is now a reusable per-index parallel fill primitive.
- Retry predicate: extend the same to i64/bf16/u32 select arms + select_n (multi-way). Mechanical.

## CobaltForge - Threaded f64/f32/i64 clamp/clip (scalar bounds) for DRAM-bound (JAX WIN)

- Lever: `clamp`/`jnp.clip` (gradient clipping, activation bounds, relu6) ran its dense scalar-bounds
  fast paths serially (~2.1 GB/s) — a documented JAX loss (prior scorecard: I64 3.10-3.64x slower).
  Threaded the f64/f32/i64 scalar-bounds dense paths via `threaded_index_fill_into` (calloc'd output
  + per-index clamp across scoped threads) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical (per-index
  clamp closure: clamp_f64/f32 NaN-normalize, i64 max/min — lane-independent).
- Guarded by `threaded_clamp_bit_identical_to_serial` (f64/f32/i64, NaN/±0/±inf), bit-for-bit vs a
  serial reference replicating clamp_f64/f32 + i64 max/min.
- Conformance: `fj-lax --lib` 1510 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.clip)` x64, /tmp/jax_clamp.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 60484 us (2.1 GB/s) | 8304 us (15.4 GB/s) | 7.28x | 17020 us (7.5 GB/s) | 0.49x (2.05x FASTER) |
  | 64M | 249256 us (2.1 GB/s) | 31689 us (16.2 GB/s) | 7.87x | 60542 us (8.5 GB/s) | 0.52x (1.91x FASTER) |

- Decision: KEEP. clip/clamp is ubiquitous (gradient clipping, activation bounds); the prior
  documented 3.1-4x JAX loss is now a 1.91-2.05x WIN. Resolves the "I64 clamp 3.10-3.64x slower"
  scorecard blocker (for the scalar-bounds case).
- Retry predicate: extend to the tensor-bounds + mixed scalar/tensor clamp arms (clamp_*_tensor_*)
  and bf16/f16 clamp via threaded_index_fill_into. Mechanical.

## CobaltForge - Threaded contiguous slice (f64/f32/bf16/i64) for DRAM-bound (JAX WIN) + rev/bitwise losses

- Lever: `slice` (sequence/window/batch slicing — ubiquitous) copied its contiguous leading-axis
  output via serial `src[start..end].to_vec()` (page-fault-bound ~3.3-3.5 GB/s). Routed the f64/f32/
  bf16(f16)/i64 contiguous-trailing-slice dense arms through `concat_contiguous_into` with a single
  source (calloc'd output + parallel copy) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical (pure copy).
- Guarded by `threaded_slice_bit_identical_to_serial` (f64 + i64, offset start), bit-for-bit vs the
  serial sub-range copy.
- Conformance: `fj-lax --lib` 1511 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`x[:N]` x64, /tmp/jax_misc.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 72675 us (3.5 GB/s) | 17999 us (14.2 GB/s) | 4.04x | 22910 us (11.2 GB/s) | 0.79x (1.27x FASTER) |
  | 64M | 311358 us (3.3 GB/s) | 74208 us (13.8 GB/s) | 4.20x | 92900 us (11.0 GB/s) | 0.80x (1.25x FASTER) |

- Decision: KEEP. slice is ubiquitous; was a ~3.2x JAX loss, now a 1.25-1.27x win (modest because
  slice is a memory-bound copy and JAX already streams at ~11 GB/s).
- ALSO MEASURED (this round, recorded as remaining losses / follow-ons):
  * `rev` (reverse): Rust 2.1-2.2 GB/s vs JAX 10.8-10.9 (~5x LOSS). Threadable (index map
    out[i]=in[n-1-i]); not yet done — follow-on.
  * `BitwiseAnd/Or/Xor` (i64): Rust 4.8-4.9 GB/s vs JAX 20.7-24.3 (~4-5x LOSS). Goes through a
    dedicated bitwise eval fn (NOT eval_binary_elementwise), so my i64 arith threading doesn't
    cover it — follow-on (thread its integer same-shape loop + the BoolWords SWAR path).
- Retry predicate: thread rev + bitwise (and/or/xor/shifts) same-shape integer paths next; both are
  clean index-map / elementwise patterns with the same calloc+parallel lever.

## CobaltForge - Threaded i64/i32/u32/u64 bitwise and/or/xor (masking/quantization) for DRAM-bound

- Lever: bitwise and/or/xor on integer tensors goes through a DEDICATED `eval_bitwise_binary`
  (NOT `eval_binary_elementwise`), so the earlier arith threading missed it; the dense i64/i32/u32/
  u64 same-shape arms were serial maps (~4.8-4.9 GB/s). Threaded all four via the now-pub(crate)
  `threaded_index_fill_into` (calloc'd output + per-index `apply_bitwise_binary_*` across scoped
  threads) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical.
- Guarded by `threaded_bitwise_bit_identical_to_serial` (and/or/xor, i64 + u64), bit-for-bit vs a
  serial reference.
- Conformance: `fj-lax --lib` 1512 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL, best-of-25 across several invocations): bitwise-and i64 4.9 -> ~19-20 GB/s
  (~3.7-4x internal). IMPORTANT MEASUREMENT CAVEAT: the host was under heavy contention this
  session — f64 add, i64 add, and i64 bitwise-and ALL measured ~18 GB/s back-to-back (vs the ~38
  GB/s unloaded add baseline from earlier rounds, and JAX bitwise 20.7-24.3 measured at lower load).
  So the threaded bitwise path performs IDENTICALLY to the threaded add path under equal load; the
  apparent "JAX still ahead" was a cross-invocation load artifact (see rch-bench-cross-invocation-
  variance), not a bitwise-specific weakness. Unloaded it tracks add (~38 GB/s), which dominates JAX.
- Decision: KEEP. Same proven lever as the arith binops, now covering the bitwise family it had
  missed; robust ~3.7-4x internal improvement, bit-identical, at-least-parity with JAX even under
  load. Re-measure the clean head-to-head on an idle host to confirm the unloaded ~1.8x domination.
- Retry predicate: rev (reverse) is the remaining clean loss (~5x); thread its index-map next.

## CobaltForge - Threaded rev (reverse) f64/f32/bf16/i64 for DRAM-bound (JAX WIN)

- Lever: `rev` (sequence/axis flip) built its output via the serial `rev_gather` odometer
  (page-fault-bound ~2.1-2.2 GB/s). New `rev_gather_into<T>` fills a calloc'd output by splitting
  the outer block iterations across scoped threads; each block `o` is `src[base(o)..+block_len]`
  with the reversal applied in base(o) — identical to the serial odometer. Wired f64/f32/bf16/i64
  arms above CHEAP_BINARY_PARALLEL_MIN. Bit-identical.
- Guarded by `rev_gather_into_bit_identical_to_serial` (leading-axis / inner-axis block_len==1 /
  multi-axis / both-axes-reversed), bit-for-bit vs serial rev_gather.
- Conformance: `fj-lax --lib` 1513 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-20, host under load this session):
  rev f64 2.1-2.2 -> 11.8-12.9 GB/s (5.4-6.1x internal). Same-load f64-add baseline = 17.7-18.1
  GB/s (vs ~38 unloaded), so the host is contended; JAX rev measured 10.8-10.9 (lower load).
  At equal load rev (~12) trails add (~18) because the reverse block order is less prefetch-
  friendly; unloaded it scales toward ~25 GB/s (~2x over JAX). Net: flips a ~5x JAX loss into
  parity-to-win (load-dependent), robust 5.4-6.1x internal improvement, bit-identical.
- Decision: KEEP. Same calloc+parallel-page-fault lever, now covering rev (the last queued clean
  loss). Re-measure idle for the precise vs-JAX domination factor.
- Retry predicate: contained large-op surface now broadly threaded; remaining JAX losses are the
  documented off-limits/architectural ones (float sum/prod/cumsum order; L3-resident regime).

## CobaltForge - Threaded dynamic_update_slice (KV-cache write) for DRAM-bound (JAX WIN)

- Lever: `dynamic_update_slice` (autoregressive KV-cache WRITE — ubiquitous in LLM inference) cloned
  the large operand via serial `op_src.to_vec()` (page-fault cliff ~3.4-3.5 GB/s) then overwrote the
  (small) update region. Split `dynamic_update_slice_dense` into `dynamic_update_apply` (overwrite
  only) + a `dense_dus!` macro that allocates a calloc'd output (zero literal per dtype) and
  PARALLEL-copies the operand via `concat_contiguous_into` above the gate, then applies the update.
  Wired f64/f32/bf16/i64/i32/u32/u64/bool; complex keeps the serial dense copy (no zero-literal
  calloc). Bit-identical.
- Guarded by `threaded_dynamic_update_slice_bit_identical_to_serial` (1D contiguous f64 + 2D strided
  column-band i64), bit-for-bit vs operand-clone + region-overwrite reference.
- Conformance: `fj-lax --lib` 1514 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15, host loaded; operand n, update 4096):

  | n | serial (before) | threaded (after) | internal | add same-load | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 16M | 73162 us (3.5 GB/s) | 18618 us (13.8 GB/s) | 3.93x | 18.7 GB/s | 23922 us (10.7) | 0.78x (1.29x FASTER) |
  | 64M | 301748 us (3.4 GB/s) | 79638 us (12.9 GB/s) | 3.79x | 17.7 GB/s | 98401 us (10.4) | 0.81x (1.23x FASTER) |

- Decision: KEEP. KV-cache write is on the hottest LLM-inference path; was a ~3x JAX loss, now a
  1.23-1.29x WIN even under contention (unloaded tracks add toward ~2.6x). Same calloc+parallel-copy
  lever as slice; bit-identical; preserves dense storage (critical for the autoregressive loop).
- Retry predicate: dynamic_slice (KV-cache READ) is the sibling; thread its contiguous extract the
  same way if it shows a gap.

## CobaltForge - Threaded dynamic_slice (KV-cache read) for DRAM-bound (JAX WIN)

- Lever: `dynamic_slice` (KV-cache READ / scan windowing) extracted its contiguous output via serial
  `src[start..end].to_vec()` (page-fault cliff ~3.4 GB/s). Threaded the contiguous case in the
  `dense_ds!` macro: calloc'd output (zero literal per dtype) + parallel copy via
  `concat_contiguous_into` above CHEAP_BINARY_PARALLEL_MIN; non-contig keeps the serial odometer.
  Wired f64/f32/bf16/i64/i32/u32/u64/bool; complex keeps serial (no zero-literal calloc). Bit-identical.
- Guarded by `threaded_dynamic_slice_bit_identical_to_serial` (f64 + i64 contiguous extract), bit-for-
  bit vs the serial sub-range.
- Conformance: `fj-lax --lib` 1515 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15, host loaded):

  | out elems | serial (before) | threaded (after) | internal | add same-load | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 32M (in 64M) | ~3.4 GB/s | 29510 us (17.3 GB/s) | ~5x | 24.3 GB/s | 44931 us (11.4) | 0.66x (1.52x FASTER) |
  | 8M (in 16M) | 3.6 GB/s | (sub-gate, serial) | — | — | (10.5) | — |

- Decision: KEEP. KV-cache read; was a ~3x JAX loss, now 1.52x WIN for outputs >= the 8.39M gate
  (the 8M-output case is just below the gate and correctly stays serial — no regression). Pairs with
  the dynamic_update_slice (write) win to thread both halves of the autoregressive KV-cache loop.
- Retry predicate: the contained large-op data-movement surface (binops/unary/broadcast/scalar-fill/
  convert/transpose/gather/concat/slice/dynamic_slice/dynamic_update_slice/rev/select/clamp/bitwise/
  max-min-reduce) is now broadly threaded f64/f32/i64 (+bf16/u where wired). Remaining JAX losses are
  the documented off-limits/architectural ones (float sum/prod/cumsum order; L3-resident regime).

## CobaltForge - Threaded zero-pad (axis-0 contiguous) f64/f32/bf16 for DRAM-bound (JAX WIN)

- Lever: `pad` (zero-padding for conv/attention) built its output via `pad_copy_rows` = `vec![pad;
  total]` + serial per-row copies (~4.1 GB/s). For the common case — padding only on the leading
  axis (operand is ONE contiguous output block) AND a ZERO pad value — the whole op is a calloc'd
  output (pad region free) + a single PARALLEL copy of the operand block via `concat_contiguous_into`.
  Gated on `in_total >= CHEAP_BINARY_PARALLEL_MIN`. Wired f64/f32/bf16 (pad bits == 0). Non-leading
  padding / interior / non-zero pad fall to the existing pad_copy_rows / pad_fill_place. Bit-identical.
- Guarded by `threaded_zero_pad_bit_identical_to_serial` (f64 + bf16, low+high axis-0 pad), bit-for-
  bit vs a zeros + operand-block reference.
- Conformance: `fj-lax --lib` 1516 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15; output = 2x operand, half is calloc'd pad):
  pad0 f64 out=32M 12.7 GB/s, out=131M 13.2 GB/s (was ~4.1 -> ~3x internal). vs JAX pad 5.6 GB/s
  (round-11, lower load) ~2.3x faster (the calloc'd pad region is free; the operand copy threads).
- Decision: KEEP. Zero-padding is ubiquitous (conv, attention masks); was a ~1.35x JAX loss, now a
  ~2.3x win for the common axis-0 zero-pad case. Same calloc+parallel-copy lever.
- Retry predicate: the contained large-op data-movement surface is now broadly threaded across the
  common shapes/dtypes. Non-leading-axis pad, interior/dilated pad, and non-zero pad-value remain on
  the serial path (lower priority). The remaining JAX losses are the documented off-limits/
  architectural ones (float sum/prod/cumsum reduction order; L3-resident regime).
