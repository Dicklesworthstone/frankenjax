# frankenjax-zc8bn: BF16 matmul A-tile decode cache

Date: 2026-06-16
Bead: `frankenjax-zc8bn`
Crate: `fj-lax`
Touched source: `crates/fj-lax/src/tensor_contraction.rs`

## Profile-backed target

Post-`frankenjax-dr67k` linalg routing profile identified
`linalg/bf16_matmul_1024_blocked` as a high-latency non-overlapping target,
with `frankenjax-mcqr.30` dense storage still assigned to another agent.

Baseline command:

```bash
RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  "linalg/(cholesky_1024x1024_f64|qr_1024x1024_f64|lu_1024x1024_f64|matmul_2d_1024x1024x1024_f64|bf16_matmul_1024_blocked|f16_matmul_512_f32accum|conv2d_gemm_f32_native)" \
  --quick --noplot
```

RCH selected worker `vmi1152480`.

Baseline row:

```text
linalg/bf16_matmul_1024_blocked [65.460 ms 65.810 ms 65.897 ms]
```

## One lever

`batched_matmul_row_block_bf16_in` previously decoded the same four BF16 A-row
values once per output column panel. For a 1024-wide output with `F32_NR = 16`,
each A scalar was decoded across 64 panels.

The shipped lever widens one `F32_MR x K` A row tile into a reusable f32 scratch
buffer once per row tile, then reuses that tile across all full column panels
and scalar column remainders.

No B packing, output rounding, dtype dispatch, shape validation, or fallback
route changed.

## Re-benchmark

Candidate command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 \
  RCH_ENV_ALLOWLIST=RCH_WORKER,RCH_WORKERS,CARGO_TARGET_DIR \
  rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  linalg/bf16_matmul_1024_blocked --quick --noplot
```

RCH selected worker `vmi1152480`.

Candidate row:

```text
linalg/bf16_matmul_1024_blocked [47.199 ms 48.701 ms 49.077 ms]
```

Speedup:

- midpoint: `65.810 / 48.701 = 1.35x`
- conservative interval: `65.460 / 49.077 = 1.33x`

Opportunity score:

```text
Impact 3.0 * Confidence 4.4 / Effort 4.0 = 3.3
```

Keep decision: `Score >= 2.0`, same-worker win, source change is one lever.

## Isomorphism proof

- Visible output ordering preserved: yes. The same output buffer cells are
  written exactly once; only traversal over independent cells changes.
- Per-output arithmetic order preserved: yes. Each output element still folds
  `l = 0..k` in ascending order, then applies `round_f32_to_bf16` through the
  existing vector or scalar rounding helper.
- Tie-breaking unchanged: N/A, no comparisons or tie choices changed.
- Floating-point behavior: bit-identical for the pinned BF16 golden route.
  Widening BF16 A inputs once per row tile uses the same `bf16_bits_to_f32`
  conversion, and accumulation order per element is unchanged.
- RNG: N/A, no randomness.
- Error and fallback behavior: unchanged. Row remainders, dtype dispatch, shape
  checks, and unsupported paths use the existing code.

Golden proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 \
  RCH_ENV_ALLOWLIST=RCH_WORKER,RCH_WORKERS,CARGO_TARGET_DIR \
  rch exec -- cargo test -j 1 -p fj-lax bf16 --lib -- --nocapture
```

Result: 15 passed, 0 failed, 12 ignored, 1552 filtered.

Pinned golden SHA-256:

```text
ff880f79bfc352fc4c4943db02e7f7e34b4c6bcb0989c960bc5f88cd6cee7bb9
```

Covered tests include:

- `tensor_contraction::tests::batched_matmul_2d_bf16_native_golden_digest`
- `tensor_contraction::tests::bf16_in_matches_f32_accum`
- `tensor_contraction::tests::round_f32xn_to_bf16_matches_scalar_edges`
- `tensor_contraction::tests::bf16_register_blocked_remainders_match_reference`
- `tensor_contraction::tests::batched_matmul_2d_bf16_batch1_packed_route_matches_register_kernel`

## Validation

Passed:

```bash
cargo fmt --check --package fj-lax
git diff --check -- crates/fj-lax/src/tensor_contraction.rs .beads/issues.jsonl
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-lax --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fj-lax --all-targets -- -D warnings
```

`ubs crates/fj-lax/src/tensor_contraction.rs` exited nonzero from existing
file-wide heuristic inventory. The reported "critical" entries are scanner
false positives on a numeric `decode` closure and pre-existing test/helper
code, not JWT handling. UBS built-in formatter, clippy, cargo check,
test-build, audit, and deny sections were clean.

## Rollback

Revert the commit that introduces the A-tile f32 scratch buffer in
`batched_matmul_row_block_bf16_in`.

## Next route

Reprofile after landing. If BF16 GEMM remains visible, the next distinct
primitive is B-panel reuse/packing or a register-block autotuning pass, not
more A decode micro-tuning.
