# frankenjax-hdqy9: dense i64 batched-operand gather

Date: 2026-06-14
Agent: SilverMaple
Crate: fj-dispatch
Target: `vmap_gather/batched_operand_shared_indices`, `vmap_gather/batched_operand_batched_indices`
Worker: `vmi1156319`

## Lever

Add a direct safe-Rust path for dense `i64` rank-2 batched operands in `batch_gather_batched_operand_direct`.
The path preserves the existing fallback for all other dtypes, ranks, sparse/non-dense storage, empty gather dimensions, and unsupported index layouts.

## Baseline

Current `origin/main` at `aafebd3a`:

| benchmark | low | midpoint | high |
| --- | ---: | ---: | ---: |
| `vmap_gather/batched_operand_shared_indices` | 63.184 us | 71.397 us | 78.440 us |
| `vmap_gather/batched_operand_batched_indices` | 66.757 us | 74.326 us | 81.656 us |

Command shape:

```text
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-silvermaple-gather-baseline-current \
  cargo bench -j 1 -p fj-dispatch --bench dispatch_overhead -- vmap_gather/batched_operand
```

## Candidate

Same worker after the one-lever change:

| benchmark | low | midpoint | high | midpoint ratio | conservative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vmap_gather/batched_operand_shared_indices` | 16.163 us | 16.988 us | 17.791 us | 4.20x | 3.55x |
| `vmap_gather/batched_operand_batched_indices` | 16.261 us | 17.277 us | 18.367 us | 4.30x | 3.63x |

Command shape:

```text
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-silvermaple-gather-candidate-current \
  cargo bench -j 1 -p fj-dispatch --bench dispatch_overhead -- vmap_gather/batched_operand
```

Score: keep. Impact 4.20, confidence 0.90, effort 0.25, score 15.1.

## Isomorphism

Ordering and tie-breaking: gather keeps row-major batch iteration and index order.

Floating point and RNG: no floating-point arithmetic and no RNG.

Mode semantics: `promise_in_bounds`, `clip`, `fill`, and `drop` continue to route through `resolve_gather_index`; fill/drop emit the existing `i64::MIN` sentinel.

Errors: negative index values return the same `BatchError::EvalError("negative index ...")` family used by the generic gather index path.

Golden output:

```text
shape/data digest: f72901b3b772939e862aa47330a10fb8b89a5b732d9840e262cd22c5509d65be
output: [10, 30, 80, 60]
```

## Validation

Passed:

```text
git diff --check
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-silvermaple-gather-proof-current \
  cargo test -j 1 -p fj-dispatch test_batch_trace_gather --lib -- --nocapture
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-silvermaple-gather-check-current \
  cargo check -j 1 -p fj-dispatch --lib
```

The focused test run passed 8 tests, including `test_batch_trace_gather_batched_operand_i64_dense_golden_sha256`.

Blocked by pre-existing debt:

```text
cargo fmt -p fj-dispatch --check
```

`batching.rs` has broad pre-existing rustfmt drift outside this lever; the new helper hunk was manually formatted and `git diff --check` passes.

```text
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-silvermaple-gather-clippy-current \
  cargo clippy -j 1 -p fj-dispatch --lib -- -D warnings
```

Clippy stops in pre-existing `fj-lax/src/linalg.rs` lint debt (`doc_lazy_continuation`, `needless_range_loop`) before validating this changed code. Existing tracker: `frankenjax-p7ri2`.

```text
ubs crates/fj-dispatch/src/batching.rs
```

UBS exits nonzero on broad pre-existing file inventory in `batching.rs` (old panic/unwrap/assert/indexing surfaces); its internal fmt, clippy, check, and test-build sub-gates are clean in the shadow scan.
