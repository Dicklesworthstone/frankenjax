# frankenjax-dr67k - lazy concat dense typed-accessor cache

Date: 2026-06-15
Agent: SilverMaple
Bead: `frankenjax-dr67k`
Target: `fj-core` lazy `LiteralBufferStorage::Concat` typed accessors feeding dense `fj-lax` readers.

## Profile-Backed Target

The bead identified a structural performance gap in `concat -> dense reader` pipelines:
`eval_concatenate` returns lazy `LiteralBufferStorage::Concat`, but typed dense accessors
previously returned `None` for concat storage. Downstream dense ops such as `Primitive::Add`
therefore fell back to per-`Literal` generic loops after concat.

Option (A), eager dense concat, was already rejected by prior cc evidence because it only reached
`1.67x` on a realistic concat-then-read row and broke the pinned lazy-concat contract. This pass
ships option (B): keep lazy concat construction and add cached typed dense materialization only
when a dense reader asks for it and every concat part already exposes the same dense type.

## Benchmark

Command:

```bash
RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- eval/concat_axis0_2x512x1024_then_add_f64 --quick --noplot
```

Same-worker RCH benchmark worker: `ovh-a`.

| Build | Criterion interval |
| --- | --- |
| Baseline, benchmark row only | `[4.1059 ms 4.1292 ms 4.2224 ms]` |
| Candidate | `[1.1719 ms 1.1784 ms 1.2041 ms]` |

Midpoint ratio: `3.50x`.
Conservative interval ratio: `3.41x`.
Score: `3.3` (`Impact 3.50 * Confidence 0.95 / Effort 1.0`).

## Lever

`LiteralBufferStorage::Concat` now carries separate `OnceLock<Option<Arc<Vec<T>>>>` caches for
`f64`, `f32`, `i64`, and half-float raw `u16` dense accessors. The typed accessor returns a
contiguous dense slice only when every concat part already exposes the matching typed dense slice.
If any part is literal-backed, mixed, or incompatible, the accessor caches and returns `None`.

The construction path remains lazy and O(parts). Existing literal materialization via `as_slice`,
indexing, iteration, and mutation semantics are unchanged.

## Isomorphism Proof

- Ordering: typed materialization walks the existing concat parts in order and uses each part's
  `[start, start + len)` slice range, matching `materialize_concat_slices`.
- Floating point: values are copied with `extend_from_slice`; no arithmetic, reassociation,
  comparison, canonicalization, or NaN normalization is introduced.
- Tie-breaking/RNG: concat has no tie surface and no RNG.
- Dtype/shape/error behavior: the accessor is purely a storage view. Tensor dtype/shape and
  concat parameter validation remain in `fj-lax` unchanged.
- Fallback behavior: mixed dense/literal concat parts still report `None` for typed dense access,
  so unsupported cases remain on the previous generic path.
- Mutation: mutating a concat buffer still materializes literal storage before writes; the focused
  tests assert the mutated buffer no longer exposes `as_f64_slice()`.
- Sharing: clones preserve the lazy concat storage and share the same caches; a mutation replaces
  storage on the mutated buffer just as before.

Golden SHA-256: `2865e0a49dc4739b8bc41f080918ed9114f39d6b453bef575e86becc9772a3ce`

The golden payload covers:

- f64 bit-exact slice order including `-0.0` and a NaN payload
- f32 bit-exact slice order including `-0.0`
- i64 slice order
- f16 raw half bits
- mixed dense + literal fallback to `None`

## Validation

```bash
cargo fmt --check --package fj-core --package fj-lax
git diff --check -- crates/fj-core/src/lib.rs crates/fj-lax/src/tensor_ops.rs crates/fj-lax/benches/lax_baseline.rs
RCH_WORKER=ovh-a rch exec -- cargo test -j 1 -p fj-core concat_dense_typed_slices_materialize_in_slice_order -- --nocapture
RCH_WORKER=ovh-a rch exec -- cargo test -j 1 -p fj-lax concatenate_pass65_lazy_output_preserves_tensor_contract -- --nocapture
RCH_WORKER=ovh-a rch exec -- cargo check -j 1 -p fj-core -p fj-lax --all-targets
RCH_WORKER=ovh-a rch exec -- cargo clippy -j 1 -p fj-core -p fj-lax --all-targets -- -D warnings
ubs crates/fj-core/src/lib.rs crates/fj-lax/src/tensor_ops.rs crates/fj-lax/benches/lax_baseline.rs artifacts/performance/evidence/frankenjax-dr67k-concat-dense-typed-accessor-cache-2026-06-15.md .skill-loop-progress.md .beads/issues.jsonl
```

Results:

- `cargo fmt --check` passed.
- `git diff --check` passed.
- Focused `fj-core` golden proof passed on RCH.
- Focused `fj-lax` concat contract proof passed on RCH.
- Crate-scoped `cargo check` passed on RCH; RCH selected `vmi1293453` despite the worker request.
- Crate-scoped `cargo clippy` passed on RCH; RCH selected `vmi1152480` despite the worker request.
- UBS exited nonzero from existing file-wide `fj-core`/`fj-lax`/bench heuristic inventory
  (panic/unwrap/direct-indexing/security false positives). Its built-in formatter, clippy,
  cargo check, test-build, cargo-audit, and cargo-deny sections were clean.

## Next Primitive

Reprofile ready perf beads after landing. If concat remains hot, attack a different structure such
as axis-aware dense concat copy scheduling or downstream fused concat-reader kernels, not eager
concat construction without fresh evidence.
