# frankenjax-lu86j: threaded 2D im2col gather

Date: 2026-06-13
Agent: BeigeMouse
Base commit: `1ce83d7ae25d754055d05afd2cb09603d0b5618c`
Worker: `vmi1293453`

## Target

`frankenjax-lu86j` identified 3x3 stride-1 conv2d as the CNN hot path. The prior
Winograd F(2,3) attempt preserved tolerance parity but regressed because it traded
one large im2col GEMM for transform overhead and sixteen smaller GEMMs. This lever
attacks the transform side directly: split the 2D im2col row fill into independent
contiguous row chunks before the existing GEMM.

The GEMM implementation and every per-output accumulation order are unchanged.

## Opportunity Score

| Lever | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Thread 2D im2col gather rows | 3 | 4 | 2 | 6.0 |

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo bench -j 1 -p fj-lax --bench lax_baseline -- eval/conv2d_64x64x32_3x3x64_f64
```

Result:

```text
eval/conv2d_64x64x32_3x3x64_f64
time: [26.079 ms 27.205 ms 28.459 ms]
```

## Candidate

Same command, same worker.

```text
eval/conv2d_64x64x32_3x3x64_f64
time: [21.415 ms 22.539 ms 23.786 ms]
```

Median speedup: `27.205 / 22.539 = 1.207x`.
Conservative interval speedup: `26.079 / 23.786 = 1.096x`.

## Isomorphism Proof

- Ordering preserved: yes. Each im2col row is still filled from the same input
  coordinates in ascending `kh, kw` order; row chunks are independent and rejoin at
  the same row offsets.
- Tie-breaking unchanged: N/A; no comparisons or selection behavior changed.
- Floating-point unchanged: identical. The new helper only copies `f32`/`f64`
  elements into the zero-initialized im2col buffer. The existing f32/f64 GEMM calls
  and their accumulation order are unchanged.
- RNG unchanged: N/A.
- Golden output SHA: `conv2d_f32_native_accum_golden_sha256` passed with existing
  expected digest `4642006de6ba3f3a608d30fb5a7904647f37a9a8d0277894fb7c45b1c8491490`.

## Validation

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo test -j 1 -p fj-lax conv_2d_im2col_dense_matches_literal_bits -- --nocapture
```

Passed: 1 test. Pre-existing warning: duplicate test attribute in `linalg.rs`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- \
  cargo test -j 1 -p fj-lax conv2d_f32_native_accum_golden_sha256 -- --nocapture
```

Passed: 1 test. Pre-existing warning: duplicate test attribute in `linalg.rs`.

```bash
rustfmt --edition 2024 --check crates/fj-lax/src/tensor_ops.rs
git diff --check
```

Both passed.

`cargo fmt -p fj-lax --check` still fails on pre-existing formatting drift in
`lax_baseline.rs`, `arithmetic.rs`, `linalg.rs`, and `tensor_contraction.rs`.
`ubs crates/fj-lax/src/tensor_ops.rs` returns nonzero due the pre-existing file-wide
inventory of test panics/unwraps/direct indexing; its cargo check, clippy, and test
build phases were clean.
