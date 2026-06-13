# frankenjax-9ghcl: Winograd F(4x4,3x3) conv2d rejection

## Target

`frankenjax-9ghcl` followed the closed `frankenjax-lu86j` conv2d line after
two rejected shallow routes:

- F(2,3) Winograd regressed because it split one large optimized im2col GEMM
  into sixteen small, unpacked channel contractions plus serial transforms.
- Direct-threaded scalar output loops regressed because they discarded the
  packed im2col/GEMM path.

The deeper candidate was the standard safe-Rust F(4x4,3x3) Winograd transform:
six-by-six input/filter transforms, four-by-four output tiles, direct fallback
only for border outputs, and tolerance proof against the current native-f32
im2col path.

## Alien primitive selection

The local no-gaps directive requires a deeper primitive rather than repeating
micro-levers. The alien-graveyard scan did not name Winograd directly; the
applicable primitives were polyhedral loop tiling, packed/tiled numeric kernels,
and vectorized execution. F(4,3) is the advanced-math artifact for this pass:
it changes the arithmetic schedule and multiply count, while preserving safe
Rust and a tolerance/golden proof surface.

## Baseline

Worker: `ovh-a`

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a RCH_WORKERS=ovh-a RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib bench_f32_conv2d --release -- --ignored --nocapture
```

Output:

```text
BENCH f32 conv2d [8,32,32,16]*[3,3,16,32]: 0.9000ms
BENCH f32 conv2d [4,28,28,32]*[3,3,32,64]: 0.8694ms
```

## Candidate gate

The temporary candidate harness was test-only and removed before this commit.
It implemented F(4,3) full tiles and a direct f32 border fallback, then compared
against the current im2col output before timing.

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a RCH_WORKERS=ovh-a RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib bench_f32_conv2d_winograd_f43_candidate --release -- --ignored --nocapture
```

Output:

```text
BENCH f32 conv2d F43 candidate [8,32,32,16]*[3,3,16,32]: im2col=0.7437ms f43=12.3420ms speedup=0.06x max_abs=4.5776367e-5 direct_max=0e0
BENCH f32 conv2d F43 candidate [4,28,28,32]*[3,3,32,64]: im2col=0.7963ms f43=14.9799ms speedup=0.05x max_abs=6.1035156e-5 direct_max=0e0
```

Correctness signal: proof-clean for the benchmark rows under the tolerance
contract (`max_abs <= 1e-2`), and direct f32 fallback matched im2col exactly.

Performance signal: decisive regression. Score `0.0`; no source hunk kept.

## Decision

Rejected/deferred. Plain safe-Rust F(4,3), even with larger tiles, repeats the
same structural failure as F(2,3): transform overhead and unfused per-position
channel contractions overwhelm the multiply-count reduction.

Next route: do not repeat standalone Winograd transforms. The next viable
conv2d primitive needs fused transformed-domain contractions:

- pack all 36 F(4,3) transformed filter planes into a persistent position-major
  layout;
- compute input transforms into a tile-major scratch once per worker chunk;
- fuse the 36 position contractions into a shared packed microkernel or
  persistent tile scheduler, so the path behaves like one packed GEMM family
  rather than 36 tiny GEMMs/loops;
- only then compare against im2col on the same profile rows.

Target ratio for that follow-up: recover at least `2.0x` over the current im2col
path on `[8,32,32,16]*[3,3,16,32]` or `[4,28,28,32]*[3,3,32,64]`.
