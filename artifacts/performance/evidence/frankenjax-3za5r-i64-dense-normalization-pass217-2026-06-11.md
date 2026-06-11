# frankenjax-3za5r pass217: I64 dense constructor normalization

## Target

- Bead: `frankenjax-3za5r`
- Profile source: RCH `vmi1227854` dispatch profile on 2026-06-11.
- Hot row: `vmap_dot_i64/paired_vectors`
- Baseline row: `71.722..71.992..72.266 us`
- Rejected predecessor: `frankenjax-hlnfl` tried a `batching.rs` dense-slice accumulator, but the profiled benchmark uses `execute_vmap_paired_i64_dot_direct` in `fj-dispatch/src/lib.rs`, not `batch_paired_i64_vector_dot`. That hunk was removed before the kept lever.

## Lever

`TensorValue::new` now stores all-`Literal::I64` tensors as the existing dense I64 `LiteralBuffer` when the logical dtype is `DType::I64`, matching the already-shipped `DType::I32` constructor normalization.

This unlocks the existing `as_i64_slice()` fast path in paired-I64 vmap dot for tensors constructed through the public literal constructor.

## Benchmark

Command:

```text
rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- vmap_dot_i64/paired_vectors --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot
```

Same worker: `vmi1227854`

| State | Low | Median | High |
| --- | ---: | ---: | ---: |
| Baseline profile | 71.722 us | 71.992 us | 72.266 us |
| Candidate | 5.9647 us | 6.3800 us | 6.7419 us |

Median speedup: `11.28x`

Conservative bound: `71.722 / 6.7419 = 10.64x`

Score: `9.5` (`Impact 10.6 x Confidence 0.9 / Effort 1.0`)

## Isomorphism proof

- Dtype and shape are unchanged: only the internal `LiteralBuffer` storage variant changes.
- Element order is unchanged: the constructor drains the original literal vector in order into the dense I64 vector.
- Public literal materialization is unchanged: `as_slice()` returns the same ordered `Literal::I64` sequence as before.
- Dispatch arithmetic is unchanged: the existing direct vmap-dot fast path keeps the same batch-major row order and the same width-major `wrapping_mul` then `wrapping_add` order.
- No floating-point, RNG, tie-breaking, or nondeterministic surfaces are involved.
- Non-I64 literal mixes still fall back to literal-backed storage.

Golden checks:

- `i32_tensor_constructor_uses_dense_i64_storage`: `a5749ee53dedc45fde6e86c9ec1b6fa9bc13cde391b6eac6a6c1f5d0a8d54daa`
- `i64_tensor_constructor_uses_dense_i64_storage`: `bf3d1069d10c35e03e75849746c04f0066b8f7879af1a6b92a05efc336f6cbce`
- `dispatch_vmap_dot_i64_paired_vectors_wraps_like_lax`: output remains ordered I64 vector `[2, 39]` with strict dispatch evidence.

## Validation

- `cargo fmt -p fj-core --check`: pass
- RCH `cargo test -j 1 -p fj-core tensor_constructor_uses_dense_i64_storage -- --nocapture`: pass on `vmi1227854` (`2 passed`)
- RCH `cargo test -j 1 -p fj-dispatch --lib dispatch_vmap_dot_i64_paired_vectors_wraps_like_lax -- --nocapture`: pass on `vmi1227854` (`1 passed`)
- RCH `cargo check -j 1 -p fj-core -p fj-dispatch --all-targets`: pass on `vmi1227854`; dependency build still emits the pre-existing `fj-trace` `num_spatial` warning.
- `cargo clippy -j 1 --no-deps -p fj-core -p fj-dispatch --all-targets -- -D warnings`: pass; RCH fell back local for this retry.
- Broad RCH clippy without `--no-deps` failed on pre-existing `fj-lax` clippy debt before this package surface.
- `ubs crates/fj-core/src/lib.rs`: nonzero on broad pre-existing inventory; its fmt, clippy, check, test-build, audit, and deny sections were clean.

## Next route

Reprofile after landing. Do not repeat the rejected `batching.rs` accumulator route for this benchmark; the productive primitive was public-constructor dense storage enabling an already-present dispatch kernel.
