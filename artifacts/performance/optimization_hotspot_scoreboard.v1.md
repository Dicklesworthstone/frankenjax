# Optimization Hotspot Scoreboard

- Schema: `frankenjax.optimization-hotspot-scoreboard.v1`
- Bead: `frankenjax-cstq.11`
- Status: `pass`
- Follow-up threshold: `2.0`
- Rows: `8`

| Rank | Family | Hotspot | p95 ns | p99 ns | Peak RSS | Score | Follow-up |
|---:|---|---|---:|---:|---:|---:|---|
| `1` | `egraph_saturation` | `hotspot-egraph-saturation-001` | `5047567` | `5244470` | `10043392` | `2.65` | `frankenjax-cstq.11.2` |
| `2` | `vmap_multiplier` | `hotspot-vmap-multiplier-001` | `60936` | `62358` | `8904704` | `2.15` | `frankenjax-cstq.11.1` |
| `3` | `ad_tape_backward_map` | `hotspot-ad-tape-001` | `108866` | `111902` | `10653696` | `1.50` | `monitor` |
| `4` | `fft_linalg_reduction_mixes` | `hotspot-fft-linalg-reduction-001` | `9197` | `11923` | `10850304` | `1.50` | `monitor` |
| `5` | `tensor_materialization` | `hotspot-tensor-materialization-001` | `13645` | `26470` | `10850304` | `1.45` | `monitor` |
| `6` | `shape_kernels` | `hotspot-shape-kernels-001` | `1924` | `3697` | `11382784` | `1.28` | `monitor` |
| `7` | `cache_key_hashing` | `hotspot-cache-key-hashing-001` | `17644` | `19908` | `11382784` | `1.18` | `monitor` |
| `8` | `durability_encode_decode` | `hotspot-durability-encode-decode-001` | `5839176` | `6198125` | `11251712` | `1.11` | `monitor` |

## One-Lever Queue

- `hotspot-egraph-saturation-001`: Profile shape-adjacent tensor programs and choose one rewrite or extraction lever with semantics-preservation proof. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-egraph rch exec -- cargo test -p fj-conformance --test egraph_preserves_semantics -- --nocapture`.
- `hotspot-vmap-multiplier-001`: Profile batch-size/rank scaling and choose one BatchTrace vectorization lever only after same-worker evidence. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-vmap rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- vmap`.
- `hotspot-ad-tape-001`: Monitor reverse-mode tape lookup and only open a bead if p99 or allocation evidence crosses the threshold. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-ad rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad`.
- `hotspot-fft-linalg-reduction-001`: Keep FFT/linalg/reduction mixed workloads in the scoreboard until higher-rank oracle fixtures justify one primitive-specific optimization. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-lax rch exec -- cargo bench -p fj-lax --bench lax_baseline -- lax_eval`.
- `hotspot-tensor-materialization-001`: Monitor tensor output materialization after recent preallocation wins; require new allocation proof before another bead. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-tensor rch exec -- cargo bench -p fj-lax --bench lax_baseline -- add_1k`.
- `hotspot-shape-kernels-001`: Keep high-rank shape inference under guardrail measurement; do not optimize without a failing scale profile. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-shape rch exec -- cargo test -p fj-trace prop_shape_inference_deterministic -- --nocapture`.
- `hotspot-cache-key-hashing-001`: Keep streaming cache-key hasher as a regression sentinel; avoid new work unless cache-key p99 regresses. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-cache rch exec -- cargo bench -p fj-cache --bench cache_baseline -- cache_key`.
- `hotspot-durability-encode-decode-001`: Track sidecar encode/decode RSS and only optimize after larger artifact profiles show a single bottleneck. Proof template `artifacts/performance/isomorphism_proof_template.v1.md`. Replay `CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-durability rch exec -- cargo run -p fj-conformance --bin fj_durability -- pipeline --artifact artifacts/performance/benchmark_baselines_v2_2026-03-12.json --sidecar /data/tmp/frankenjax-hotspot-durability/sidecar.json --report /data/tmp/frankenjax-hotspot-durability/scrub.json --proof /data/tmp/frankenjax-hotspot-durability/proof.json`.

No optimization hotspot issues found.
