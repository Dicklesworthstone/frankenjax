# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

## 2026-06-20 - frankenjax-cntiy softmax +fma/devirtualization no-ship

`frankenjax-cntiy` remains maintainer-gated, but this pass rejects two narrower
agent-actionable interpretations of the gate.

Same-host JAX oracle: `softmax_2d_65536x16` mean 1.0524 ms, p50 1.0765 ms,
CV 9.39%. Local Rust Criterion used
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b-local-20260620`;
RCH validation used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`.

| Probe | Rust mean | Rust/JAX | Verdict |
| --- | ---: | ---: | --- |
| default `nn/softmax_2d_65536x16_fused` | 2.2163 ms | 2.106 | Loss: JAX 2.11x faster |
| local `RUSTFLAGS="-C target-feature=+fma"` | 2.2096 ms | 2.100 | Neutral as a lever; still JAX loss |
| generic row-call devirtualization patch | 2.3303 ms | 2.214 | Regression; reverted |

RCH `cargo test -p fj-lax softmax_2d --lib` passed 10/0. An initial RCH
default-vs-`+fma` row on `vmi1149989` looked like a 4.7x `+fma` win, but a
rerun default row on `vmi1152480` and the same-host run both measured ~2.2 ms,
so that first default row is recorded only as an anomaly.

Decision: no source or build-policy win shipped. Global `+fma` alone does not
create the missing SIMD/fast-exp softmax kernel. Ratio scorecard: 0 wins / 3
losses / 0 neutral vs JAX.

## 2026-06-20 - frankenjax-xljoh compiled-dispatch f64 mid-cache keep

`frankenjax-xljoh` shipped only the measured narrow fallback: reusable
`CompiledJaxprRunner` routes one-tensor f64 linear chains in the
65,536..=262,144 element band through owned compiled eval. It did not ship a
broad f32 route or an upper-band f64 route.

Directional candidate Rust/JAX ratios from the new comparator artifact
`../artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`:

| Row | Rust/JAX | Verdict |
| --- | ---: | --- |
| f64 4K x8 | 0.541 | Rust faster; guard off |
| f64 65K x8 | 1.516 | JAX faster; kept internal mid-cache fallback |
| f64 262K x8 | 3.195 | JAX faster; remaining gap |
| f64 1M x8 | 20.36 | JAX faster; fallback correctly off |
| f64 16M x8 | 4.55 | JAX faster; no safe win in this lever |
| f32 4K x8 | 0.163 | Rust faster; no f32 route |
| f32 65K x8 | 0.637 | Rust faster; no f32 route |

Retry predicate: do not reattempt broad f32 routing or f64 fallback expansion
without same-host or same-worker proof. The remaining f64 large-chain loss needs
deeper output reuse, vector codegen, or backend specialization.

## 2026-06-20 - frankenjax-q59j4/co009 LiteralBuffer internal keeps

`frankenjax-q59j4` and `frankenjax-co009` were closed as measured internal
keeps after RCH validation. These rows do not claim a JAX-vs-Rust win because
they cover host-internal `LiteralBuffer` mutation and conformance/fixture
serialization paths with no direct JAX API-equivalent comparator.

Validation used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`.
The Criterion pass ran on RCH worker `vmi1149989`; `fj-conformance` ran on
`hz2` and passed 45 tests, 0 failed.

| Row | Dense Rust | Literal control | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `literal_buffer_index_mut_f64_64k` | 24.003 us | 33.278 us | 0.721 | Keep: 1.39x faster internally; JAX N/A |
| `literal_buffer_serialize_f64_64k` | 1.3443 ms | 1.6493 ms | 0.815 | Keep: 1.23x faster internally; JAX N/A |

Decision: keep both committed direct paths. They reduce internal ledger,
fixture, and mutation overhead; the next JAX-facing work remains output/arena
reuse, non-temporal stores/prefetch/NUMA, or a specific typed-path external
loss with same-host proof.

## 2026-06-20 - frankenjax-oneqh allocator default no-ship

`frankenjax-oneqh` was closed as an evidence-only no-ship. The proposed default
allocator change was not committed, and `CHEAP_BINARY_PARALLEL_MIN` was not
retuned or removed.

Local same-host allocator preload verification used
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b-local-20260620`
with `fj-lax` `elementwise_gauntlet`. JAX used 0.10.1 CPU x64 via
`benchmarks/jax_comparison/.venv/bin/python`. Jemalloc was available at
`/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`; mimalloc was not installed.

| Row | glibc Rust | jemalloc Rust | JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `add_f64_16m/dense` | 24.502 ms, Rust/JAX 0.870 | 33.095 ms, Rust/JAX 1.174 | 28.179 ms | Reject: allocator preload regresses |
| `add_f32_16m/dense` | 15.945 ms, Rust/JAX 1.128 | 16.784 ms, Rust/JAX 1.188 | 14.130 ms | Reject/neutral: no allocator win |
| `mul_f64_16m/dense` | 29.210 ms, Rust/JAX 1.024 | 27.790 ms, Rust/JAX 0.974 | 28.525 ms | Neutral/slight win only |

Decision: no production allocator change. The next credible levers are
compiled-jaxpr output/arena reuse, non-temporal stores/prefetch/NUMA work, or a
specific unowned typed-path gap with same-host proof.
