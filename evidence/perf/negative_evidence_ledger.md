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

- Date: 2026-06-18
- Agent: cod-b / TopazOrchid
- Lever: densify homogeneous F64/F32/Bool/BF16/F16/Complex literal vectors in
  `TensorValue::new` after element-count validation.
- Status: batch-test pending.
- Benchmark guard: `core/tensor_value_new_1k_f64_generic_dense`,
  `core/tensor_value_new_1k_f64_forced_literal`,
  `core/tensor_value_new_then_to_f64_vec_1k`,
  `core/tensor_value_new_forced_literal_then_to_f64_vec_1k`.
- Conformance guard: matching literal families materialize bit-identically;
  mismatched literal/dtype tensors remain literal-backed.
- Retry predicate: do not retry FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat/storage-copy families without fresh same-worker benchmark evidence.
  Do not repeat the already committed stack/repeat/slice/to_i64 storage levers
  unless a profile identifies a distinct call path.

## frankenjax-mcqr.99 - Direct Dense LiteralBuffer to_vec Materialization

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::to_vec()` materialize directly from dense storage
  variants instead of forcing `as_slice()` to build/cache a full literal vector
  and then clone it.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_to_vec_dense_f64_64k`,
  `core/literal_buffer_to_vec_literal_f64_64k`.
- Conformance guard: direct `to_vec()` matches `as_slice().to_vec()` bit-for-bit
  across dense F64/F32/I64/U32/U64/Bool/BF16/F16/Complex storage plus lazy
  concat/repeated-patches paths.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64
  or `TensorValue::new` dense-storage families under this bead. Do not revisit
  FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager concat without fresh
  same-worker benchmark evidence and ownership check.

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

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer` equality compare storage-direct typed ranges and
  recursively compare concat slices instead of forcing `as_slice()` on both
  operands and caching full materialized literal vectors for dense packed
  buffers.
- Status: batch-test pending.
- Benchmark guard: `core/literal_buffer_eq_dense_f64_64k_equal`,
  `core/literal_buffer_eq_dense_f64_64k_mismatch`,
  `core/literal_buffer_eq_literal_f64_64k_equal`.
- Conformance guard: storage-direct equality matches materialized
  `Vec<Literal>` equality across F64/F64OnePlusX/F32/I64/U32/U64/Bool/
  BoolWords/Half/Complex, repeated-patches, concat, mixed dense/literal concat,
  and the `LiteralBuffer`/`Vec<Literal>` cross-`PartialEq` impls.
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
  JAX gap directly, such as one-pass raw half clamp with the same signed-zero
  and NaN proof, temp allocation elimination, or fused producer/consumer clamp.

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
