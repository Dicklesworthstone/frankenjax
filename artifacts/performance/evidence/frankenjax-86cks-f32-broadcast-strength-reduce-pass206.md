# frankenjax-86cks pass206: F32 broadcast gather strength reduction

## Target

- Bead: `frankenjax-86cks`
- Crate: `fj-interpreters`
- Hot row: F32 eval_jaxpr fusion for row/column broadcast chains, after passes 202 and 203 still spent about 8ms in fused broadcast gather loops.
- One lever: remove per-element row modulo and column division from the existing F32 broadcast fusion chunk evaluator.

## Baseline

Command:

```text
RCH_WORKER=ovh-a rch exec -- cargo bench -p fj-interpreters --bench eval_fusion_speed
```

Worker: `ovh-a`

Rows:

```text
EVAL_FUSION_SPEED_F64 n=1048576 ops=8 unfused=32.688ms fused=1.832ms speedup=17.84x
EVAL_FUSION_SPEED_F32 n=1048576 ops=8 unfused=8.198ms fused=0.781ms speedup=10.49x
EVAL_FUSION_SPEED_F32_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=10.125ms fused=8.364ms speedup=1.21x
EVAL_FUSION_SPEED_F32_COL_BROADCAST rows=1024 cols=1024 ops=8 unfused=9.724ms fused=8.359ms speedup=1.16x
```

## Change

- Added row-broadcast helpers that seed/apply contiguous row slices rather than computing `(base + offset) % row_len` for every output element.
- Added column-broadcast helpers that seed/apply row-major spans rather than computing `(base + offset) / cols` for every output element.
- Preserved the existing unsupported-shape fallback and all fused-tape guards.
- Added an empty-output early return in the row helper to preserve the old no-op behavior before any modulo operation.

## Isomorphism Proof

- Equation order is unchanged.
- Per-element primitive order is unchanged.
- F32 arithmetic is unchanged: every primitive still widens operands to F64, applies the same operation, and rounds the result back to F32 after each step.
- Row-broadcast indexing is unchanged. The helper partitions the same linear index stream into contiguous row spans and copies/applies the same `row[(base + offset) % row.len()]` values.
- Column-broadcast indexing is unchanged. The helper partitions the same row-major stream into row spans and applies the same `col[(base + offset) / cols]` values.
- No RNG, tie-breaking, dtype, shape, or error-surface behavior changes.
- Empty output chunks remain no-ops, matching the previous iterator behavior.

Golden output SHA-256 checks:

```text
F32 same-shape:       fef28624a52e5647abc35f0d388072b443cf081e5941243c6c58a8bd91f40a84
F32 row broadcast:   1f742aad15797ada82394f8d78c5b2d488ac650c272e8a81330a694621a64494
F32 column broadcast:5762f3ec4614f491d21407cbb09c5cd92915840f65d145070f8d8b5e8c7c5e3a
```

Focused proof command:

```text
rch exec -- cargo test -p fj-interpreters fusion_f32_ -- --nocapture
```

Result: passed on `ovh-a`; focused F32 same-shape, row-broadcast, and column-broadcast bit-for-bit tests all passed.

## Re-benchmark

Final exact-source command:

```text
RCH_WORKER=ovh-a rch exec -- cargo bench -p fj-interpreters --bench eval_fusion_speed
```

Worker: `ovh-a`

Rows:

```text
EVAL_FUSION_SPEED_F64 n=1048576 ops=8 unfused=35.411ms fused=2.281ms speedup=15.52x
EVAL_FUSION_SPEED_F32 n=1048576 ops=8 unfused=7.649ms fused=0.973ms speedup=7.86x
EVAL_FUSION_SPEED_F32_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=10.221ms fused=0.975ms speedup=10.49x
EVAL_FUSION_SPEED_F32_COL_BROADCAST rows=1024 cols=1024 ops=8 unfused=10.569ms fused=0.799ms speedup=13.23x
```

Scored target deltas:

```text
F32 row-broadcast fused path:    8.364ms -> 0.975ms, 8.58x faster
F32 column-broadcast fused path: 8.359ms -> 0.799ms, 10.46x faster
```

Score: `12.5` (`Impact=5`, `Confidence=5`, `Effort=2`).

Decision: keep and close `frankenjax-86cks`.

## Validation

```text
cargo fmt -p fj-interpreters --check
git diff --check -- crates/fj-interpreters/src/lib.rs
rch exec -- cargo check -p fj-interpreters --all-targets
rch exec -- cargo test -p fj-interpreters
rch exec -- cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings
ubs crates/fj-interpreters/src/lib.rs
```

Results:

- `cargo fmt -p fj-interpreters --check`: passed.
- `git diff --check -- crates/fj-interpreters/src/lib.rs`: passed.
- `cargo check -p fj-interpreters --all-targets`: passed on `ovh-a`.
- `cargo test -p fj-interpreters`: passed, `149 passed`, doc tests `0`.
- `cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings`: passed on `ovh-a`.
- `ubs crates/fj-interpreters/src/lib.rs`: nonzero due broad pre-existing file inventory, while its formatting, clippy, cargo check, test-build, cargo-audit, and cargo-deny sections were clean.

Known unrelated warnings printed by dependency crates during RCH gates:

- `fj-trace`: unused variable `num_spatial` at `crates/fj-trace/src/lib.rs:1808`.
- `fj-lax`: unused `cell_f64_reference` at `crates/fj-lax/src/cz0g0_f32accum_evidence.rs:104`.
