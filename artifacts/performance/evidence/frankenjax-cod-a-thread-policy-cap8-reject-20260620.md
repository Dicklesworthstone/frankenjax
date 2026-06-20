# Dense 16M Elementwise Thread Policy Cap8 Rejection

Agent: WildForge / cod-a
Date: 2026-06-20
Target: `fj-lax` dense same-shape multi-input elementwise compute

## Hypothesis

The same-load ledger showed large multi-input compute ops still losing to JAX by roughly 1.2-1.3x.
The shipped Rust implementation already threads these rows, but it uses all available cores with
64K-element chunks. I tested whether fewer, larger chunks could reduce thread/cache pressure and
improve sustained read bandwidth.

## Candidate

Temporarily changed the f64/f32 cheap same-shape thread policy to:

```text
threads = (elements / 1_048_576).clamp(1, min(available_cores, 8))
```

The production candidate was reverted after measurement. No cap8 production change remains.

## Same-Binary A/B

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet -- \
  'thread_policy_f64_add_16m' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --save-baseline cod-a-thread-policy-cap8-ab-final
```

Worker: `vmi1152480`

| policy | threads | mean | result |
| --- | ---: | ---: | --- |
| shipped `work_scaled_threads` | 10 | 17.344 ms | baseline |
| cap8 / 1M chunks | 8 | 20.199 ms | 1.16x slower |

Decision: reject and revert cap8.

## Current Rust Rows

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet -- \
  'add_f64_16m|add_f32_16m|mul_f64_16m' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2 \
  --save-baseline cod-a-thread-policy-current
```

Worker: `ovh-a`

| row | mean |
| --- | ---: |
| add_f64_16m/dense | 29.214 ms |
| add_f32_16m/dense | 13.845 ms |
| mul_f64_16m/dense | 28.999 ms |

## JAX Comparator

Command:

```bash
uv run --no-project --with 'jax[cpu]' \
  python benchmarks/jax_comparison/elementwise_gauntlet.py \
  --runs 10 --warmup 3 --inner-loops 5 \
  --output artifacts/performance/evidence/frankenjax-cod-a-thread-policy-jax-20260620T0205Z.json
```

JAX: `0.10.2`

| row | JAX mean | Rust/JAX |
| --- | ---: | ---: |
| add_f64_16m | 27.798 ms | 1.051 |
| add_f32_16m | 13.668 ms | 1.013 |
| mul_f64_16m | 27.673 ms | 1.048 |

Cross-host ratio scorecard, 5% neutral band: 0 wins / 1 loss / 2 neutral.

## Retry Predicate

Do not retry a smaller `std::thread` worker count for this family. The next credible levers are
NUMA/affinity pinning, explicit prefetch, non-temporal stores, or output reuse in a compiled runner.
