# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

## 2026-06-20 - frankenjax-ligu5 vmap gather dense-float no-ship

`frankenjax-ligu5` was reopened against the dispatch-layer batched gather/scatter
de-box vein. Prior passes had already shipped direct I64 batched-operand gather
and lazy rank-1 overwrite scatter, so this pass added f64/f32 batched-operand
gather benchmark rows and measured them before touching production code.

RCH Rust timing used `fj-dispatch` on worker `vmi1152480` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`. JAX timing used
the local CPU `jax.jit(vmap(take))` comparator in
`artifacts/performance/evidence/frankenjax-ligu5-jax-vmap-gather-20260620T1325Z.json`.

| Row | Rust midpoint | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| I64 batched operand + batched indices | 8.6722 us | 31.266 us | 0.277 | Rust 3.61x faster |
| F64 batched operand + batched indices | 25.251 us | 33.224 us | 0.760 | Rust 1.32x faster |
| F32 batched operand + batched indices | 27.257 us | 31.369 us | 0.869 | Rust 1.15x faster |

Decision: keep the new benchmark coverage, but ship no production gather/scatter
change. JAX CV was high (17-25%), so these are routing rows rather than
certification-grade ratios; they are still enough to reject the suspected
dense-float batched gather JAX loss. Retry only with a fresh profile showing a
non-dense, higher-rank, partial-slice, or different dtype subcase.

## 2026-06-20 - frankenjax-n75xr f64 scalar-add SIMD medium-band keep

`frankenjax-n75xr` kept a narrow exact-pattern `std::simd` specialization for
f64 `tensor + scalar + ...` compiled-dispatch chains only in the medium band
`1,048,576 <= n < FUSION_THREAD_MIN_ELEMS`. It preserves the scalar-add order per
lane and rejects operand-reversed first adds so edge-case bit semantics remain on
the generic fusion path.

Same-worker RCH proof on `vmi1227854`:

| Row | Baseline Rust | Candidate Rust | JAX mean | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| f64 1M x8 | 1.3412 ms | 821.23 us | 83.299 us | 9.859 | Keep: 1.63x internal win, still JAX loss |
| f64 16M x8, ungated branch | 78.933 ms | 104.63 ms | 27.610 ms | 3.790 | Reject: +32.56% regression, gated off before commit |

Small rows were not claimed: 65K and 262K are outside the kept gate and remain
JAX losses. Follow-up reruns after the final gate landed on `ovh-a`; `RCH_WORKER`
did not pin back to `vmi1227854`, so those rows are non-comparable worker noise.
Ratio scorecard after the kept gate: 0 wins / 4 losses / 0 neutral vs JAX for
the f64 large-chain production row set.

## 2026-06-20 - frankenjax-cntiy FMA primitive policy no-ship

`frankenjax-cntiy` tested the direct ternary FMA primitive row that the softmax
`+fma` probe could not isolate. The benchmark harness now records `fma_f64_1m`
and `fma_f32_1m` with dense and boxed controls, but no production compiler flag
or target-feature policy was shipped.

JAX comparator caveat: JAX 0.10.1 in the local oracle environment does not
expose a public `jax.lax.fma`, so the comparator is warmed
`jax.jit(lambda a, b, c: a * b + c)`.

| Probe | Rust mean | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| default dense `fma_f64_1m` | 2.6124 ms | 273.448 us | 9.553 | Loss: JAX 9.55x faster |
| default dense `fma_f32_1m` | 2.7622 ms | 111.281 us | 24.822 | Loss: JAX 24.82x faster |
| local `RUSTFLAGS="-C target-feature=+fma"` f64 | 925.02 us | 273.448 us | 3.383 | Improved but still JAX loss; no policy ship |
| local `RUSTFLAGS="-C target-feature=+fma"` f32 | 207.98 us | 111.281 us | 1.869 | Improved but still JAX loss; no policy ship |

RCH build guard passed for `cargo build --release -p fj-lax --benches` on
`vmi1227854`. RCH timing on `hz1` showed the dense primitive path is still a
real internal de-box win over boxed controls: f64 3.5198 ms vs 29.267 ms
(8.31x), f32 3.7757 ms vs 30.825 ms (8.16x). Local same-host default rows were
2.6124 ms vs 24.592 ms boxed (9.41x) and 2.7622 ms vs 26.151 ms boxed (9.47x).

Decision: keep the benchmark coverage, but do not ship a global `+fma` policy.
Ratio scorecard: 0 wins / 4 losses / 0 neutral vs JAX. Retry only with a
semantics-approved per-kernel target feature, generated/vectorized code path, or
output-reuse strategy that can be tested against this row directly.

## 2026-06-20 - frankenjax-xljoh.1 f64 large-chain register/tile no-ships

`frankenjax-xljoh.1` tested two deeper f64 compiled-dispatch levers against
the remaining large-chain JAX losses. Both were reverted before commit.

Baseline on RCH worker `vmi1227854` for the unchanged runner:
65K 46.896 us, 262K 214.91 us, 1M 1.3315 ms, 16M 116.11 ms.
JAX comparator means are from the existing
`../artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`.

| Probe | Worker | Best relevant rows | Rust/JAX | Verdict |
| --- | --- | --- | ---: | --- |
| ordered register pass for 8 scalar adds | `vmi1153651` | 65K 52.746 us; 1M 3.5382 ms; 16M 142.56 ms | 1.55 / 42.48 / 5.16 | Reject: failed to vectorize; worse than existing runner and still JAX loss |
| `FUSION_CHUNK` 8K -> 64K | `vmi1149989` | 65K 52.828 us; 1M 1.9764 ms; 16M 128.78 ms | 1.55 / 23.73 / 4.66 | Reject: Criterion reported regressions (+7.33%, +21.35%, +4.66%) |

Retry predicate: do not retry per-element register scalar-add fusion or larger
fusion tiles without disassembly/profile evidence showing SIMD vectorization and
a same-worker before/after win. The remaining f64 chain gap is not a simple
chunk-size or step-order loop issue.

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
