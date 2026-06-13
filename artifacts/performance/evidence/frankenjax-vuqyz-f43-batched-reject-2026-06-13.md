# frankenjax-vuqyz: F(4x4,3x3) batched transformed contractions rejection

## Target

Follow-up to `frankenjax-9ghcl`. Standalone F(4,3) Winograd was proof-clean
but regressed because it ran the transformed-domain channel contractions as
unfused loops. This pass tested the next natural fusion level without touching
the SageAnchor-owned f32 GEMM microkernel work:

- pack all 36 F(4,3) filter transform planes in position-major order;
- compute tile-major input transforms for all full tiles;
- call the existing `batched_matmul_2d_f32_in` once with `batch=36`,
  `m=tiles`, `k=c_in`, `n=c_out`;
- output-transform full 4x4 tiles and direct-fold only border outputs.

## Baseline

Worker: `ovh-a`

Earlier unchanged baseline in this pass family:

```text
BENCH f32 conv2d [8,32,32,16]*[3,3,16,32]: 0.9000ms
BENCH f32 conv2d [4,28,28,32]*[3,3,32,64]: 0.8694ms
```

The candidate command also printed same-invocation im2col rows:

```text
im2col=0.8179ms
im2col=0.7588ms
```

## Candidate gate

Temporary test-only candidate command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a RCH_WORKERS=ovh-a RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib bench_f32_conv2d_winograd_f43_batched_candidate --release -- --ignored --nocapture
```

Output:

```text
BENCH f32 conv2d F43 batched candidate [8,32,32,16]*[3,3,16,32]: im2col=0.8179ms f43_batched=10.0231ms speedup=0.08x max_abs=4.5776367e-5
BENCH f32 conv2d F43 batched candidate [4,28,28,32]*[3,3,32,64]: im2col=0.7588ms f43_batched=19.9470ms speedup=0.04x max_abs=6.1035156e-5
```

Correctness signal: proof-clean for the benchmark rows under the same
`max_abs <= 1e-2` tolerance envelope as `frankenjax-9ghcl`.

Performance signal: decisive regression. Score `0.0`; no source hunk kept.

## Decision

Rejected/deferred. The existing batched f32 GEMM path does fuse the 36
transformed positions into one API call, but `batch > 1` intentionally skips
B-panel packing because each batch has a different transformed filter plane.
The result still behaves like many small `k=c_in` contractions plus transform
overhead, so it does not compete with the current large im2col GEMM.

Next route: stop pursuing conv2d Winograd inside `tensor_ops.rs` alone. The
needed primitive crosses into the f32 GEMM microkernel/panel-packing surface:
position-major packed-B support or a dedicated transformed-domain contraction
kernel that treats the 36 planes as one packed superproblem. That overlaps the
currently owned `frankenjax-9zwwb` f32 GEMM work, so route this evidence to that
owner instead of taking the file surface.
