# frankenjax-i3t3h grouped/depthwise conv native-f32 evidence

Date: 2026-06-11
Bead: frankenjax-i3t3h
Lever: grouped/depthwise conv1d and conv2d for F32/BF16/F16 now decode operands to f32, accumulate in f32, and round once to out dtype.

## Baseline and Rebench

A clean detached `HEAD` worktree was used for the pre-change baseline with only
the benchmark helper inserted. RCH admitted the remote baseline to
`vmi1227854`, but both candidate retries fell open locally because no worker was
admissible (`critical_pressure`, `insufficient_slots`). The keep decision uses
the same-machine local-fallback before/after pair below; the remote baseline is
recorded separately and is not mixed into the speedup calculation.

Command:

```text
CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenjax-i3t3h-... rch exec -- cargo test -p fj-lax --release --lib bench_f32_grouped_conv_accum -- --ignored --nocapture
```

Baseline (`HEAD` 12b4afb0, clean detached worktree):

```text
f32 depthwise conv1d [1,1024,256]*[5,1,256] g=256: 3.1542 ms
f32 grouped conv1d   [1,512,256]*[3,8,256] g=32:   1.5924 ms
f32 depthwise conv2d [1,56,56,128]*[3,3,1,128] g=128: 1.9353 ms
f32 grouped conv2d   [1,28,28,256]*[3,3,8,256] g=32:  5.4451 ms
```

Candidate local-fallback retries:

```text
retry a:
f32 depthwise conv1d [1,1024,256]*[5,1,256] g=256:     0.1714 ms
f32 grouped conv1d   [1,512,256]*[3,8,256] g=32:       0.9305 ms
f32 depthwise conv2d [1,56,56,128]*[3,3,1,128] g=128:  0.4554 ms
f32 grouped conv2d   [1,28,28,256]*[3,3,8,256] g=32:   3.9614 ms

retry b:
f32 depthwise conv1d [1,1024,256]*[5,1,256] g=256:     0.1726 ms
f32 grouped conv1d   [1,512,256]*[3,8,256] g=32:       0.9355 ms
f32 depthwise conv2d [1,56,56,128]*[3,3,1,128] g=128:  0.4734 ms
f32 grouped conv2d   [1,28,28,256]*[3,3,8,256] g=32:   4.0801 ms
```

Conservative local speedup range versus baseline:

```text
f32 depthwise conv1d: 18.27x-18.40x
f32 grouped conv1d:    1.70x-1.71x
f32 depthwise conv2d:  4.09x-4.25x
f32 grouped conv2d:    1.33x-1.37x
```

Remote baseline only (`vmi1227854`, candidate not admitted remotely):

```text
f32 depthwise conv1d [1,1024,256]*[5,1,256] g=256:     2.7149 ms
f32 grouped conv1d   [1,512,256]*[3,8,256] g=32:       1.9276 ms
f32 depthwise conv2d [1,56,56,128]*[3,3,1,128] g=128:  1.6964 ms
f32 grouped conv2d   [1,28,28,256]*[3,3,8,256] g=32:   4.6208 ms
```

Score: Impact 4 x Confidence 3 / Effort 3 = 4.0. Confidence is capped at 3
because the scheduler did not admit a remote candidate run, despite two
crate-scoped RCH retries.

## Isomorphism Proof

- Ordering preserved: yes. Output order remains row-major `[batch, spatial, channel]`.
- Tie-breaking unchanged: N/A. Convolution has no tie-breaking.
- Floating-point: intentional parity change from f64 accumulation to native f32 accumulation for F32/BF16/F16, matching XLA policy. Each output still accumulates in ascending tap/channel order.
- RNG seeds: N/A.
- OOB behavior: preserved as `0.0_f32 * rhs`, including infinities and signed-zero-sensitive padding behavior.
- Golden output: grouped-conv digest `dfe98ccb7192d4ce86072dccafae4f8717d7c5abec0ccd16367070c05811f8b3`.

## Verification

```text
rch exec -- cargo test -p fj-lax grouped_conv_native_f32_accum_matches_reference_and_golden_sha256 --lib -- --nocapture
PASS

rch exec -- cargo test -p fj-lax conv2d_grouped_native_f32_accum_matches_reference --lib -- --nocapture
PASS
```

The two commands above also ran through RCH local fallback because remote admission was unavailable.

```text
rch exec -- cargo check -p fj-lax --lib
PASS on vmi1227854

rch exec -- cargo test -p fj-lax --lib
PASS, 1339 passed, 53 ignored
```
