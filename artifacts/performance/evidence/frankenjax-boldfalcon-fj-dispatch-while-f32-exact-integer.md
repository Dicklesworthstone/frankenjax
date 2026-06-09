# fj-dispatch f32 while exact-integer closed form

Pass: 192
Agent: BoldFalcon
Commit base: origin/main f020aac6
Crate: fj-dispatch
Hotspot: `vmap_while/batched_scalar_f32_add_128`

## Profile And Baseline

Profile-backed target: `dispatch_baseline` ranked `vmap_while/batched_scalar_f32_add_128` as a residual dispatch hot row after prior f32 while direct-path work.

Keep-gate baseline, same RCH worker `vmi1227854`:

- Before: `9.9293 us` midpoint, interval `[9.0930, 10.966] us`
- Candidate: `5.9885 us` midpoint, interval `[5.8302, 6.1917] us`
- Same-worker speedup: `1.658x`

Final current-base corroboration via `rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- vmap_while/batched_scalar_f32_add_128` on f020aac6 plus this patch:

- RCH fell open locally because no admissible workers were available.
- Final current-base candidate: `5.8766 us` midpoint, interval `[5.8207, 5.9384] us`

Score: `Impact 1.66 * Confidence 0.95 / Effort 0.5 = 3.15`, kept because `>= 2.0`.

## Change

`batch_while_f32_add_lt_batch0` now tries an exact-integer algebraic solver before the existing repeated-add lane loop. The solver returns `Ok(None)` for every case outside the proof envelope, so non-certified inputs use the original implementation.

The proof envelope is intentionally narrow:

- `body_op=add`, `cond_op=lt`, and the existing f32 batch-axis-0 while route.
- Finite f32 values that are exact integers in `[-16_777_216, 16_777_216]`.
- Positive exact-integer step.
- Final result remains exactly representable as f32.
- Iteration count is strictly below `max_iter`.

## Isomorphism Proof

- Ordering preserved: per-lane output order is unchanged; one output is emitted per input lane in batch order.
- Tie-breaking unchanged: no tie-breaking logic is introduced.
- Floating point unchanged for accepted lanes: all accepted values are exact f32 integers, so the closed form computes the same final f32 bit pattern as repeated f32 addition. Fractional, NaN, infinite, non-positive-step, and rounding-sensitive lanes fall back.
- Error boundary unchanged: `max_iter == 0` and `iterations >= max_iter` return the same while max-iteration error as the original loop.
- RNG unchanged: no RNG path involved.
- Golden output: `[0x42c00000,0x42c60000,0x42de0000,0x41200000,0x40000000,0x41c00000]`
- Golden sha256: `e3de78209337533170c435223a5d2ddccd83a3f299933f379ceebccfa64409e8`

## Validation

- `git diff --check -- crates/fj-dispatch/src/batching.rs`: passed.
- `rch exec -- cargo test -p fj-dispatch while_f32_add_lt -- --nocapture`: passed, 2 focused tests.
- `rch exec -- cargo check -p fj-dispatch --all-targets`: passed with pre-existing `fj-trace` warning for `num_spatial`.
- `rch exec -- cargo clippy -p fj-dispatch --all-targets -- -D warnings`: failed on pre-existing upstream lints outside this patch:
  - `crates/fj-trace/src/lib.rs:1808` unused `num_spatial`
  - `crates/fj-lax/src/arithmetic.rs:6417` `canonical_batched_matmul_dims` has too many arguments
  - `crates/fj-lax/src/arithmetic.rs:6465` `batched_standard_i64_matmul` has too many arguments
  - `crates/fj-lax/src/arithmetic.rs:6506` `batched_standard_complex_matmul` has too many arguments
- `cargo fmt --package fj-dispatch --check`: failed on pre-existing file-wide formatting drift in `crates/fj-dispatch/src/batching.rs`; new lines were manually aligned with the rustfmt suggestions for this hunk.
- `ubs crates/fj-dispatch/src/batching.rs`: no critical findings; scanner reported existing file-wide warning inventory.

## Coordination

No unclaimed ready perf bead existed for this exact hotspot. `br ready --json` exposed only the parity-policy blocker bead, and active perf surfaces were already assigned to IcyGlacier. This pass used the allowed profiler-evident hotspot fallback and avoided `.beads` changes.
