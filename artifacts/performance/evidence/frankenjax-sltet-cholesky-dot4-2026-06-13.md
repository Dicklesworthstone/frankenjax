# frankenjax-sltet: Cholesky Schur dot4 row reuse

Date: 2026-06-13
Agent: BeigeMouse
Base commit: `8886b116508294070de3f54bf3dbf82a37f2dd71`
Worker: `vmi1293453`

## Target

`frankenjax-sltet` targeted linalg trailing-update GEMM work. The literal FMA
route remains a no-ship in the current repo policy: `.cargo/config.toml` enables
`+avx2` but deliberately excludes `+fma`, and `cz0g0_fma_evidence` documents that
`mul_add` is only meaningful under a separate global target-feature decision.

This kept lever stays within the same Cholesky trailing-update surface without a
global build-policy change. The lower-triangle Schur update now computes four
adjacent `q` dots at a time, reusing the same `p_row` SIMD load while each dot
keeps the same vector-lane accumulation, horizontal lane sum, and scalar tail
order as the old single-dot helper.

## Opportunity Score

| Lever | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Four-wide Cholesky Schur dot batching | 3 | 4 | 2 | 6.0 |

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/cholesky_1024x1024_f64
```

Result:

```text
linalg/cholesky_1024x1024_f64
time: [38.152 ms 39.726 ms 41.394 ms]
```

## Candidate

Same command, same worker.

First candidate:

```text
linalg/cholesky_1024x1024_f64
time: [32.834 ms 36.137 ms 40.101 ms]
```

Confirmation candidate:

```text
linalg/cholesky_1024x1024_f64
time: [30.448 ms 31.565 ms 32.775 ms]
```

Median speedup using the confirmation row: `39.726 / 31.565 = 1.259x`.
Conservative interval speedup: `38.152 / 32.775 = 1.164x`.

## Isomorphism Proof

- Ordering preserved: yes. Rows and lower-triangle `q` entries are still visited
  in ascending order; four adjacent independent output entries are updated as a
  batch.
- Tie-breaking unchanged: N/A; no comparisons or selection behavior changed.
- Floating-point unchanged for each dot: yes. For each individual `p,q` dot, the
  SIMD chunk order, lane-local sums, horizontal lane sum order, and scalar tail
  accumulation are the same as the previous `cholesky_schur_dot` helper.
- RNG unchanged: N/A.
- Golden output SHA: `cholesky_blocked_path_golden_output_digest` passed with
  existing expected digest
  `cae3c6a0fcc965880d1379765d0b7886deb1ca3d1c9dc1036ca705e60306ff0a`.

## Validation

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo test -j 1 -p fj-lax cholesky_ -- --nocapture
```

Passed: 14 tests, including
`cholesky_lower_schur_update_matches_full_gemm_on_consumed_triangle` and
`cholesky_blocked_path_golden_output_digest`.

```bash
git diff --check
```

Passed.

`rustfmt --edition 2024 --check crates/fj-lax/src/linalg.rs` still fails only on
pre-existing formatting drift outside this touched block. The known fj-lax clippy
blockers are tracked separately in `frankenjax-p7ri2`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo clippy -j 1 -p fj-lax --lib -- -D warnings
```

Failed on the known pre-existing `frankenjax-p7ri2` lints in `linalg.rs`
(`doc_lazy_continuation` and `needless_range_loop`); no diagnostic pointed at the
new dot4 helper.

```bash
ubs crates/fj-lax/src/linalg.rs
```

Returned nonzero due the existing file-wide panic/unwrap/direct-indexing
inventory. UBS reported its format, clippy, cargo check, and test-build phases
clean in the shadow workspace.
