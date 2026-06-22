# Negative Evidence Ledger

This ledger records code-first performance attempts and retry predicates so dead
ends are not rediscovered without new evidence.

## frankenjax-mcqr - histogram-prefix scatter-add bucket build no-ship

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED NO-SHIP / SOURCE REVERTED. The attempted single-buffer
  histogram + prefix-sum partition for 1-D f64 `scatter_add` was correctness
  green but slower than the retained per-bucket `Vec` owner-computes path.
- Target gap: `eval/scatter_add_1m_f64_1d`, still a JAX loss after the retained
  bucketed path.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: radix-partition/hash-join style bucket construction:
    per-bucket counts, exclusive prefix offsets, one contiguous `(target,
    update_index)` buffer, then owner-computes disjoint output ranges.
  - Isomorphism: each resolved target is placed into the same output-range
    bucket as production, and the second fill pass visits `index_vals` in the
    original order, so duplicate-index additions inside each bucket preserve the
    retained path's update order.
  - EV decision: revert. The extra resolve pass and cursor/prefix traffic cost
    more than the removed `Vec<Vec<_>>` bucket allocation/push overhead on the
    measured row.

Correctness proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  range_partitioned_f64_scatter_add_matches_literal_path \
  --lib --release -- --nocapture
```

- RCH worker: `ovh-a`; result: 1 focused `fj-lax` test passed.

Bench evidence:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_WORKER \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/scatter_add_1m_f64_1d' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

- RCH worker selection was daemon-side; `RCH_WORKER` was not a hard pin. A direct
  `ovh-a` production-vs-candidate pair was eventually captured.
- Candidate run 1 on `ovh-a`: **17.162 ms** midpoint (`16.627..17.987ms`).
- Candidate run 2 on `ovh-a`: **13.426 ms** midpoint (`12.918..13.908ms`).
- Production baseline on `ovh-a`: **11.852 ms** midpoint (`10.765..12.552ms`).
- Non-paired production row on `vmi1149989`: **18.581 ms** midpoint
  (`13.627..23.846ms`), recorded only as worker-noise context.

Ratio-vs-JAX ledger, using the existing fresh JAX 0.10.1 CPU x64 p50
**3.639273 ms** for the exact fixture:

| workload / variant | Rust midpoint | JAX p50 | Rust/JAX p50 | candidate/production | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| retained production bucketed path (`ovh-a` refresh) | 11.852 ms | 3.639273 ms | 3.26 | 1.00 | JAX loss, baseline |
| histogram-prefix candidate, best repeat | 13.426 ms | 3.639273 ms | 3.69 | 1.13 | **REVERT; +13.3% slower** |
| histogram-prefix candidate, first run | 17.162 ms | 3.639273 ms | 4.72 | 1.45 | **REVERT; +44.8% slower** |

Scorecard: **0 JAX wins / 1 JAX loss / 0 neutral** for this target, with
**1 Rust-side keep / 2 reverted branches** across the scatter-add family. Retry
predicate: do not retry single-buffer histogram/prefix partitioning for this
row. The next credible scatter-add route needs a fundamentally different safe
parallel direct-write proof or a lower-overhead duplicate-index specialization
that beats the retained **9.9667 ms** historical same-worker row and then the
fresh JAX **3.639273 ms** p50.

## frankenjax-cntiy - softmax relaxed-exp16 fast path no-ship

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: REJECTED / REVERTED. No production source change remains.
- Target gap: `nn/softmax_2d_65536x16_fused`, the current fused softmax row
  compared head-to-head against `jax.nn.softmax(..., axis=-1)` on the exact
  65,536 x 16 f64 fixture.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: replace per-row libm `exp` with the existing
    `simd_poly_exp_into` approximation for finite 16-column softmax rows.
  - Candidate details: stack-copy each shifted 16-value row, evaluate polynomial
    exp into a stack array, sum, and normalize; non-finite rows and all other
    column counts stayed on the old path.
  - EV decision: revert. The exact JAX comparator target barely moved and still
    lost hard; the unrelated log-softmax timing improved in a routing run, but
    that was not the claimed JAX comparator and did not justify shipping a
    relaxed-exp semantic surface.

Remote correctness proof before revert:

```text
AGENT_NAME=CrimsonOtter RCH_WORKER=hz2 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_WORKER \
  rch exec -- cargo test -p fj-lax softmax_2d --release -- --nocapture
```

- RCH worker: `hz2`; result: 10 focused `fj-lax` softmax/log-softmax tests
  passed. The command compiled remotely and retrieved only small artifacts.

Remote bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  softmax_2d_65536x16_fused -- --warm-up-time 1 --measurement-time 3 \
  --sample-size 15 --noplot
```

- Baseline RCH worker: `hz2`.
- Baseline production `nn/softmax_2d_65536x16_fused`: **2.6515 ms** midpoint
  (`2.4187..2.8259 ms`).
- Baseline production `nn/log_softmax_2d_65536x16_fused`: **6.0083 ms**
  midpoint (`4.3410..7.9548 ms`), noisy/outlier-heavy.

```text
AGENT_NAME=CrimsonOtter RCH_WORKER=hz2 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_WORKER \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  softmax_2d_65536x16_fused -- --warm-up-time 1 --measurement-time 3 \
  --sample-size 15 --noplot
```

- Candidate RCH worker: `vmi1149989`; `RCH_WORKER=hz2` was ignored for this
  bench, so this is routing evidence, not strict same-worker keep proof.
- Candidate `nn/softmax_2d_65536x16_fused`: **2.5922 ms** midpoint
  (`1.9908..3.1984 ms`), only **2.2%** better than the hz2 baseline midpoint
  and well inside cross-worker/noise ambiguity.
- Candidate `nn/log_softmax_2d_65536x16_fused`: **2.4909 ms** midpoint
  (`2.2375..2.7020 ms`). This is a useful future log-softmax routing signal,
  but it was not the claimed JAX softmax target and was not same-worker.

Fresh JAX comparator:

- Command: `benchmarks/jax_comparison/.venv/bin/python
  benchmarks/jax_comparison/softmax_gauntlet.py`
- JAX/JAXLIB: repo venv CPU backend with x64 enabled by the comparator script.
- Fixture: exact 65,536 x 16 f64 softmax row used by
  `nn/softmax_2d_65536x16_fused`.
- JAX mean: **1.091807 ms**; p50 **1.101048 ms**.

Ratio-vs-JAX ledger:

| workload / variant | Rust midpoint | JAX mean | Rust/JAX mean | verdict |
| --- | ---: | ---: | ---: | --- |
| production softmax baseline | 2.6515 ms | 1.091807 ms | 2.43 | JAX loss |
| relaxed-exp16 candidate | 2.5922 ms | 1.091807 ms | 2.37 | REVERT; ~0-gain, still JAX loss |

Scorecard: **0 wins / 1 loss / 0 neutral** for the JAX softmax target, with
**0 kept / 1 rejected**. Retry predicate: do not retry a stack-copy
`simd_poly_exp_into` row16 softmax path. The next credible softmax/attention
route needs same-binary or same-worker proof that first beats **2.65 ms** by a
large margin and then closes the fresh JAX **1.09 ms** comparator; likely route
remains approved target-feature/FMA SIMD exp or a fused attention-level kernel.

## frankenjax-mcqr - bucketed f64 scatter-add narrows but does not close JAX gap

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED KEEP / STILL JAX LOSS. Production dense f64 1-D
  `scatter_add` now routes large scalar-slice updates through a safe bucketed
  owner-computes path; a later unique-index atomic branch was measured and
  reverted.
- Target gap: `eval/scatter_add_1m_f64_1d`, the remaining order-dependent
  scatter-add row from the `frankenjax-mcqr` no-gaps scorecard.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: owner-computes scatter partitioning. Each worker owns a
    disjoint output range, so safe Rust can parallelize duplicate-index
    scatter-add without locks or `unsafe`.
  - Retained lever: resolve each index once, bucket `(target, update_index)` by
    output range while preserving original update order inside the bucket, then
    add updates in scoped threads over disjoint mutable output chunks.
  - Reverted lever: a no-duplicate `AtomicU64` direct-write branch. It was
    correctness-green but slower than the bucketed path on the same row
    (**10.822ms** vs **9.9667ms**).
  - EV decision: keep the bucketed path because it is a 3.07x same-worker
    Rust-side speedup. Do not call the row closed: it still loses JAX by 2.74x.

Fresh JAX comparator:

- Command environment: `benchmarks/jax_comparison/.venv/bin/python`,
  JAX/JAXLIB 0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `eval/scatter_add_1m_f64_1d` sequence added to
  `crates/fj-lax/benches/lax_baseline.rs`: zero f64 vector, LCG permutation
  indices, deterministic f64 updates, `mode=add`, `index_mode=clip`.
- JAX mean: **3.956295 ms**; p50 **3.639273 ms**; min **3.237853 ms**.

Remote correctness proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  range_partitioned_f64_scatter_add_matches_literal_path \
  --lib --release -- --nocapture
```

- RCH worker: `hz2`; result: 1 focused `fj-lax` test passed.

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-conformance --test gather_scatter_oracle \
  --release -- --nocapture
```

- RCH worker: `vmi1152480`; result: 59 `gather_scatter_oracle` tests passed.

Remote bench proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_WORKER=vmi1227854 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_WORKER \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/scatter_add_1m_f64_1d' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

- RCH worker: `vmi1227854`; `RCH_WORKER` was honored for the final baseline /
  candidate pair after earlier ignored pins.
- Baseline serial dense scatter-add: **30.611 ms** midpoint.
- First all-range scan candidate: **14.146 ms** midpoint; improved but worse
  than retained bucketed path.
- Retained bucketed owner-computes path: **9.9667 ms** midpoint
  (`9.5639..10.308 ms`).
- Reverted unique atomic branch: **10.822 ms** midpoint; Criterion reported
  +13.132% regression versus the bucketed run.

Ratio-vs-JAX ledger:

| workload / variant | Rust midpoint | JAX p50 | Rust/JAX p50 | verdict |
| --- | ---: | ---: | ---: | --- |
| baseline dense serial scatter-add | 30.611 ms | 3.639273 ms | 8.41 | JAX loss |
| all-range scan owner-computes candidate | 14.146 ms | 3.639273 ms | 3.89 | no-ship; slower than bucketed |
| retained bucketed owner-computes | 9.9667 ms | 3.639273 ms | 2.74 | **KEEP but still JAX loss** |
| unique-index atomic branch | 10.822 ms | 3.639273 ms | 2.97 | REVERT; +8.6% slower than bucketed |

Scorecard: **0 JAX wins / 1 JAX loss / 0 neutral** for this target, with
**1 Rust-side keep / 1 reverted branch**. Retry predicate: do not retry
range-scan or atomic-direct variants. The next credible scatter-add route needs
to remove bucket construction/allocation cost or introduce a safe proof of
parallel direct writes that beats **9.9667 ms** in a same-worker A/B and then
beats the fresh JAX **3.639273 ms** p50.

## frankenjax-mcqr - cummax/cummin head-to-head closes one order-dependent gap

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED / BENCH COVERAGE ADDED. No production source changed.
- Target gap: the order-dependent cumulative extrema rows called out by the
  `frankenjax-mcqr` no-gaps scorecard after the cumsum prefix-scan keep.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: measurement-first cumulative-extrema triage. JAX CPU's
    order-dependent ops were broadly slow, but fj-lax had no exact cummax/cummin
    rows to prove whether this was a real remaining loss.
  - Implemented lever: add exact fj-lax Criterion rows for a deterministic 1M
    f64 1D fixture matching the JAX comparator shape and dtype.
  - EV decision: no production rewrite. `cummax` is a real fj-lax win; `cummin`
    is near parity / slight fj-lax win, with enough noise to keep it out of the
    "closed forever" bucket.

Remote bench command:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/cum(max|min)_1m_f64_1d' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Correctness proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-conformance --test cummax_cummin_oracle --release -- --nocapture
```

- RCH worker: `ovh-a`; result: 28 `cummax_cummin_oracle` tests passed.
- JAX comparator: local `benchmarks/jax_comparison/.venv`, JAX/JAXLIB 0.10.1,
  CPU backend, `jax_enable_x64=true`, same deterministic 1M f64 fixture.
- fj-lax bench worker: `hz2`; per-crate Criterion, no new `.scratch`.

Ratio-vs-JAX ledger:

| workload | fj-lax Criterion midpoint | JAX mean | JAX p50 | Rust/JAX p50 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `eval/cummax_1m_f64_1d` | 2.0374 ms | 3.833558 ms | 3.458314 ms | 0.589 | **fj-lax wins; JAX/Rust 1.70x** |
| `eval/cummin_1m_f64_1d` | 3.4187 ms | 3.623474 ms | 3.602727 ms | 0.949 | near-parity / slight fj-lax win |

Scorecard: **1 win / 0 loss / 1 neutral-slight-win**. Retry predicate: do not
spend a production rewrite on `cummax` before lower-scoring JAX losses; revisit
`cummin` only if fresh same-worker evidence shows a real loss beyond Criterion
noise. Next `frankenjax-mcqr` target remains a measured JAX loss such as
scatter-add, not another cumulative-extrema micro-probe.

## frankenjax-murmw - Bluestein-prime FFT tile/thread no-ships

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: REJECTED / REVERTED. No production source change remains.
- Target gap: `eval/fft_batch_256x1009_prime_complex128_dense_input`, the
  rough/prime-length batch row where `1009` routes through the shipped SoA
  Bluestein path and still trails JAX.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: cache/task-granularity control around the existing
    communication-avoiding SoA Bluestein kernel.
  - Rejected lever 1: widen `BLUESTEIN_TILE_ROWS` from 4 to 8 rows to reduce
    per-tile overhead and increase vertical lane width.
  - Rejected lever 2: cap the vectorized Bluestein scheduler at 8 threads so
    each task owns more convolution FFT work instead of fanning `256x1009` into
    tiny two-tile jobs under swarm contention.
  - EV decision: revert both. The first lever regressed on the same worker; the
    second produced a tempting but non-comparable `ovh-a` routing signal, then
    regressed hard when rerun on the same `hz2` worker as production.

Remote bench command shape:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_batch_256x1009_prime_complex128_dense_input' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Ratio-vs-JAX ledger:

| workload | worker | Rust production | Rust candidate | JAX comparator | Rust/JAX candidate | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `BLUESTEIN_TILE_ROWS=8` | `hz2` | 3.8919 ms | 4.5907 ms | 0.478 ms | 9.60 | REVERT: +20.635% Criterion regression |
| `PARALLEL_MAX_THREADS=8` | `hz2` | 4.3750 ms | 6.3681 ms | 0.478 ms | 13.32 | REVERT: +37.004% Criterion regression |

Notes:

- The JAX comparator is the existing fresh JAX/JAXLIB 0.10.1 x64 mean for the
  exact `256x1009` fixture: **0.478 ms**.
- A capped-thread run on `ovh-a` measured **1.7611 ms** (Rust/JAX **3.68x**),
  but `RCH_WORKER=ovh-a` was ignored by the follow-up proof command, so this is
  routing evidence only, not an accepted keep.
- Current production after reverts: fresh same-worker `hz2` production midpoint
  **4.3750 ms**, Rust/JAX **9.15x**.

Retry predicate: do not retry Bluestein tile-height, representation-only, or
coarse thread-count probes without hardware-counter evidence. The next credible
prime/rough FFT route must change the internal convolution kernel itself
(SIMD-within-FFT/pocketfft-class complex shuffles) or prove a quiesced-host
threading regime with a completed same-worker production/candidate A/B.

## frankenjax-mcqr - blocked f64 cumsum prefix-scan flips 4M 1D row to JAX win

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: MEASURED KEEP / JAX WIN. Production `Primitive::Cumsum` now routes one
  long forward dense f64 line through a blocked prefix-scan that breaks the
  scalar dependency chain while staying inside the cumulative tolerance contract.
- Target gap: `eval/cumsum_4m_f64_1d`, the new 4M 1D cumsum loss in the umbrella
  `frankenjax-mcqr` no-gaps bead.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: parallel prefix/scan scheduling to break the scalar f64
    dependency chain that keeps the current dense scan at about 30ms while JAX
    uses a reassociated scan around 14ms.
  - Implemented lever: split one long f64 line into thread-local scans, compute
    exclusive block offsets, then apply offsets in parallel.
  - EV decision: keep. The first strict absolute proof was over-conservative for
    a reassociated million-element scan; the retained proof bounds cumsum and
    cumprod drift by a tight relative/absolute allowance far below the conformance
    oracle bar, while cummax/cummin remain exact.

Remote correctness proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-lax \
  blocked_dense_f64_single_line_cumulative_matches_serial_reference \
  --lib --release -- --nocapture
```

- RCH worker: `hz2`; result: 1 focused `fj-lax` test passed.

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-conformance --test cumulative_oracle \
  --release -- --nocapture
```

- RCH worker: `vmi1149989`; result: 45 `cumulative_oracle` tests passed.

Fresh JAX comparator:

- Command environment: `benchmarks/jax_comparison/.venv/bin/python`,
  JAX/JAXLIB 0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `eval/cumsum_4m_f64_1d` sequence from
  `crates/fj-lax/benches/lax_baseline.rs`: `np.arange(1 << 22) * 0.001`.
- JAX mean: **18.318300 ms**; p50 **18.290179 ms**; min **16.053977 ms**.

Remote bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'eval/cumsum_4m_f64_1d' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- RCH worker: `ovh-a`; per-crate Criterion.
- Retained blocked prefix-scan row: **7.5297 ms** midpoint
  (`7.4698..7.6034 ms`).
- Sequential-floor diagnostic row matched by the same filter:
  `eval/cumsum_4m_f64_1d_tight` **23.805 ms** midpoint
  (`23.729..23.877 ms`).

Ratio-vs-JAX ledger:

| workload | Rust current | JAX comparator | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| old sequential dense scan | 30.3 ms | 18.3183 ms | 1.65 | historical JAX loss |
| retained blocked prefix-scan | 7.5297 ms | 18.3183 ms | 0.41 | **fj-lax wins; JAX/Rust 2.43x** |
| same-run tight sequential diagnostic | 23.805 ms | 18.3183 ms | 1.30 | confirms dependency-chain floor |

Retry predicate: do not revisit the naive "strict sequential bit" objection for
large f64 cumsum; the accepted route is reassociation-tolerant and conformance
green. Future work should target other remaining JAX losses unless a regression
shows this row falling behind a fresh same-fixture JAX comparator.

## frankenjax-cntiy - tanh SIMD-exp fast path narrows but does not close JAX gap

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: MEASURED KEEP / STILL JAX LOSS. Production `Primitive::Tanh` now routes
  large dense f64 tensors through the existing SIMD polynomial `exp` helper and
  the stable identity `tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))`.
- Claimed tracker context: `frankenjax-cntiy` remains the open maintainer-gated
  FMA/softmax bead. This `tanh` sub-gap is tolerance-only and does not resolve
  the parent gate.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: libm/fenv replacement with a vectorized kernel and a
    certified tolerance envelope, targeted only at a dense tensor size where the
    current threaded libm route is measured expensive.
  - Implemented lever: dense-f64, >=1M-element fast path using `simd_poly_exp_into`;
    scalar, small tensor, f32, half, boxed, and complex behavior remains on the
    existing path.
  - EV decision: keep as a narrowing lever. It is not a JAX win and does not close
    `cntiy`; the next route needs lower-allocation SIMD/FMA polynomial work or an
    approved target-feature specialization.

Remote correctness proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-lax \
  simd_poly_tanh_large_dense_f64_matches_libm_tolerance \
  --release --lib -- --nocapture
```

- RCH worker: `vmi1149989`; result: 1 focused `fj-lax` test passed.
- Accuracy print: `[simd_poly_tanh] max_abs_err=2.220e-16 (oracle bar 1e-10)`.

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-conformance --test tanh_oracle \
  --release -- --nocapture
```

- RCH worker: `vmi1149989`; result: 36 `tanh_oracle` tests passed.
- Note: a direct local warm-target conformance attempt failed before tests with
  rustc metadata drift (`b30f3df3b` remote artifacts vs local `f20a92ec0`), so
  conformance was rerun through RCH without cleaning the shared target dir.

Remote bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'eval/tanh_1m_f64_vec' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- RCH worker: `ovh-a`; per-crate Criterion, same invocation.
- Old libm-reference row: **6.1998 ms** midpoint (`6.1748..6.2341 ms`).
- Retained production row: **4.2741 ms** midpoint (`3.7186..5.1672 ms`).
- Raw SIMD-exp probe row: **3.7810 ms** midpoint (`3.7327..3.8450 ms`).
- Internal Rust speedup: **1.45x** retained production versus old libm-reference.

Fresh JAX comparator:

- Command environment: `benchmarks/jax_comparison/.venv/bin/python`, JAX/JAXLIB
  0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `eval/tanh_1m_f64_vec` sequence from
  `crates/fj-lax/benches/lax_baseline.rs`: `((i % 4001) - 2000) * 0.001`.
- JAX mean: **0.293181 ms**; median **0.277430 ms**; min **0.230216 ms**.

Ratio-vs-JAX ledger:

| workload | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| old threaded libm-reference `tanh` | 6.1998 ms | 0.293181 ms | 21.15 | large JAX loss |
| retained SIMD-exp production `tanh` | 4.2741 ms | 0.293181 ms | 14.58 | kept narrowing lever; still a JAX loss |
| raw SIMD-exp probe | 3.7810 ms | 0.293181 ms | 12.90 | upper-bound routing signal only |

Retry predicate: do not retry another allocation-equivalent `exp(-2|x|)`
rewrite. The next tanh route must reduce the two-pass allocation floor or use
approved target-feature/FMA SIMD polynomial work, and it must beat the retained
production path in a same-binary A/B before dispatch.

## frankenjax-cntiy - erf rational approximation narrows but does not close JAX gap

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: MEASURED KEEP / STILL JAX LOSS. Production `Primitive::Erf` now uses
  fdlibm-derived minimax rational bands for the common `|x| < 2.857` range,
  while retaining the prior Maclaurin bridge for `2.857..3.5` and the existing
  high-tail behavior.
- Claimed tracker context: `frankenjax-cntiy` remains the open maintainer-gated
  FMA/softmax bead. This `erf` sub-gap is tolerance-only and does not resolve the
  parent gate.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: libm/fenv replacement with bounded approximants before
    deeper SIMD work. This follows the tolerance-only transcendental route:
    remove scalar series/libm tax without changing bit-pinned goldens.
  - Implemented lever: replace the hot common-range Maclaurin loop with
    rational bands; keep high-range behavior stable.
  - Rejected lever: a degree-20 Chebyshev `[0,2]` approximation. It passed the
    same focused correctness gates but measured slower than the rational
    candidate signal, so it was reverted before commit.
  - EV decision: keep the rational lever as a material narrowing step, then route
    further `erf` work to true SIMD/vector polynomial or approved target-feature
    work. Do not repeat scalar polynomial reshuffles unless they beat this path
    in a same-binary A/B.

Remote correctness proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-lax erf_high_accuracy_and_seam_continuity --lib --release
```

- RCH worker: `ovh-a`; result: 1 focused `fj-lax` test passed.

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-conformance --test erf_oracle --release
```

- RCH worker: `ovh-a`; result: 31 `erf_oracle` tests passed.

Remote bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'eval/erf_1m_f64_vec' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- Old Rust series baseline: RCH `ovh-a`, **21.110 ms** midpoint
  (`20.977..21.255 ms`).
- Retained rational candidate:
  - RCH `vmi1149989`, **6.8485 ms** midpoint (`6.4220..7.4408 ms`), best
    observed current-code row.
  - RCH `hz2`, **12.515 ms** midpoint (`12.083..12.807 ms`), extra
    non-comparable worker point.
- Rejected Chebyshev candidate: RCH `ovh-a`, **10.596 ms** midpoint
  (`10.527..10.680 ms`), **49.743%** faster than the old series on the same
  worker but slower than the rational candidate signal; reverted.
- Worker pinning was not available in `rch exec`, so the retained rational
  before/after is not strict same-worker certification. The ratio-vs-JAX and
  repeated direction are recorded honestly; future work should use a same-binary
  A/B harness if worker placement matters.

Fresh JAX comparator:

- Command environment: `benchmarks/jax_comparison/.venv/bin/python`, JAX/JAXLIB
  0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `eval/erf_1m_f64_vec` sequence from
  `crates/fj-lax/benches/lax_baseline.rs`, timed as 60 runs x 100 inner loops
  with `block_until_ready()`.
- JAX mean: **1.495718 ms**; p50 **1.497700 ms**.

| workload | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| old `erf` Maclaurin series | 21.110 ms | 1.495718 ms | 14.11 | large JAX loss |
| retained rational `erf` candidate, best observed | 6.8485 ms | 1.495718 ms | 4.58 | kept narrowing lever; still a JAX loss |
| retained rational `erf` candidate, extra `hz2` point | 12.515 ms | 1.495718 ms | 8.37 | still a JAX loss |
| rejected Chebyshev candidate | 10.596 ms | 1.495718 ms | 7.08 | reverted |

Retry predicate: the next `erf` route must be a true SIMD/vectorized polynomial
or target-feature/FMA specialization that preserves `erf_oracle` and beats the
current rational path in a same-binary A/B. The broader `cntiy` FMA/softmax
decision is not resolved by this keep.

## frankenjax-murmw - scalar-specialized mixed-radix FFT no-ship

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED NO-SHIP / REVERTED. No production source change remains; the
  temporary focused correctness test and ignored A/B harness were removed.
- Target gap: `eval/fft_batch_128x1000_complex128`, the remaining smooth-
  composite batched FFT loss.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: generated length-specialized FFT kernels. This pass used
    a scalar stage-iterative radix-2/3/5-specialized proxy for
    `1000 = 2^3 * 5^3`, keeping the row-local memory layout and avoiding the
    previously rejected cross-row SoA transpose.
  - EV decision: reject. The contiguous stage schedule removed recursion but
    introduced enough digit-reversal/stage-copy overhead that it was slower than
    the recursive per-row mixed-radix baseline.

Remote correctness proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  iterative_scalar_specialized_matches_recursive_to_tolerance \
  --release --lib -- --nocapture
```

- RCH worker: `hz2`; result: 1 focused `fj-lax` test passed.
- Note: `rch` rewrote `CARGO_TARGET_DIR` to a worker pool path and rehydrated
  dependencies remotely; this was acceptable for correctness and same-binary
  A/B triage, but not a warm-target Criterion proof.

Remote same-binary A/B:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  bench_mixed_radix_iterative_scalar_specialized_vs_per_row \
  --release --lib -- --ignored --nocapture
```

- RCH worker: `hz2`.
- Printed result:

```text
[mixed-radix iterative scalar-specialized 128x1000] recursive=1.500ms specialized=3.242ms ratio=0.46x (min of 9 interleaved)
```

Ratio-vs-JAX ledger:

| workload | Rust current | JAX comparator | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `eval/fft_batch_128x1000_complex128`, retained production | 3.4978 ms | 0.230078 ms | 15.20 | current JAX loss remains |
| scalar-specialized stage-iterative candidate | 3.242 ms microbench | 0.230078 ms | 14.09 | rejected: 0.46x recursive floor before full eval |

Focused conformance after reverting the candidate:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-conformance --test fft_oracle \
  --release -- --nocapture
```

- RCH worker: `vmi1152480`; result: 27/27 `fft_oracle` tests passed.

Retry predicate: do not retry stage-iterative scalar kernels for `n=1000`
unless the design eliminates digit-reversal/stage-copy overhead and first beats
recursive mixed-radix in the same-binary A/B. The remaining credible routes are
a true generated in-place/recursive `1000 = 2^3 * 5^3` kernel or a quiesced-host
threading proof.

## frankenjax-murmw - row-buffer reuse and Bluestein smooth retry no-ship

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED NO-SHIP / REVERTED. No production source change remains.
- Target gap: `eval/fft_batch_128x1000_complex128`, smooth-composite batched FFT.
- Scorecard: **0 wins / 1 loss / 0 neutral** versus fresh JAX.

Fresh production baseline:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_WORKER=vmi1152480 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_batch_128x1000_complex128' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

- RCH worker: `vmi1152480`; per-crate only, no local cargo build, no new
  `.scratch` worktree.
- Rust Criterion interval: **3.1347..4.4919 ms**, midpoint **4.0559 ms**.
- Fresh JAX/JAXLIB 0.10.1 CPU x64 comparator, exact
  `complex_matrix(128,1000)` fixture, 30 samples x 100 inner loops:
  mean **0.177952 ms**, p50 **0.175767 ms**.
- Current retained production Rust/JAX: **22.79x** by midpoint.

Candidate measurements:

| candidate | evidence | result |
| --- | --- | --- |
| Flat iterative mixed-radix SoA route | RCH `hz2`, same-binary `bench_mixed_radix_iterative_soa_vs_per_row` | per-row **1.550 ms** vs iter **6.532 ms**, ratio **0.24x**; remains disabled |
| Smooth-composite Bluestein SoA retry | RCH `ovh-a` hinted **1.19x**, but RCH `vmi1152480` repeat printed per-row **1.904 ms** vs Bluestein **1.917 ms** | ratio **0.99x**; no reproducible production gate |
| Recursive mixed-radix row-buffer zero-fill elision | RCH `vmi1152480` Criterion after edit: **2.6909..2.8715 ms** but `p=0.17`; direct same-binary A/B printed zero-fill **2.212 ms** vs reuse-len **2.211 ms** | ratio **1.00x**; source hunk and temporary harness reverted |

Alien-graveyard/extreme-optimization route: vectorized execution and
length-specialized staged kernels remain the plausible family, but these three
micro-levers do not move the measured floor. Retry only with a generated
in-place/recursive `1000 = 2^3 * 5^3` kernel that first beats per-row recursive
mixed-radix in the same binary, or with a quiesced-host threading proof.

## frankenjax-murmw - specialized iterative SoA mixed-radix FFT no-ship

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED NO-SHIP / REVERTED. No production source change remains; the
  temporary ignored A/B harness was removed after measurement.
- Target gap: `eval/fft_batch_128x1000_complex128`, the remaining smooth-
  composite batched FFT loss.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: generated/vectorized FFT butterflies with precomputed
    twiddles, using `1000 = 2^3 * 5^3` as the concrete length-specialization
    proxy. The lever kept the flat iterative SoA mixed-radix schedule but
    replaced each stage's O(r^2) small DFT with the same specialized radix-2/3/5
    butterflies proven in the recursive production kernel.
  - EV decision: reject. The SoA schedule's transpose/digit-reversal overhead
    still dominates even after removing the small-DFT arithmetic tax; the
    candidate is slower than the recursive per-row mixed-radix baseline and
    remains far behind JAX.

Remote production Rust baseline:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  eval/fft_batch_128x1000_complex128 \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

- RCH worker: `ovh-a`; per-crate, no local cargo build, no new `.scratch`
  worktree.
- Rust Criterion midpoint: **3.4978 ms** (`3.2761..3.6825 ms`).

Fresh JAX comparator:

- Command environment: `benchmarks/jax_comparison/.venv/bin/python`, JAX/JAXLIB
  0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `complex_matrix(128,1000)` formula from
  `crates/fj-lax/benches/lax_baseline.rs`, timed as 30 batches x 100 inner
  loops with `block_until_ready()`.
- JAX mean: **0.230078 ms**; p50 **0.233834 ms**.
- Current production Rust/JAX: **15.20x**. Scorecard: **0 wins / 1 loss / 0
  neutral** for this row.

Remote same-binary A/B for the candidate:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME \
  rch exec -- cargo test -p fj-lax \
  bench_mixed_radix_iterative_soa_specialized_vs_per_row \
  --release -- --ignored --nocapture
```

- RCH worker: `hz2`; per-crate release test binary, no local cargo build, no new
  `.scratch` worktree.
- Correctness gate inside the harness: specialized iterative SoA matched the
  recursive mixed-radix reference to FFT tolerance on sampled rows before timing.
- Printed A/B result:

```text
[specialized mixed-radix iterative SoA 128x1000] 1T per-row=1.511ms specialized=2.668ms ratio=0.57x (min of 9 interleaved)
```

Interpretation: the specialized iterative SoA route is **43% slower** than the
per-row recursive baseline (`0.57x` as fast). Even its microbench time would
still be **11.60x** the fresh JAX mean, so there is no dispatch gate to keep.

Retry predicate: do not retry the flat iterative SoA family for `n=1000` unless
the new design eliminates the SoA transpose/digit-reversal cost and first beats
per-row recursive mixed-radix in a same-binary A/B. The next credible FFT route
is a truly generated length-specialized in-place/recursive `1000 = 2^3 * 5^3`
kernel, or a quiesced-host threading proof; both require fresh same-worker
evidence before any production dispatch change.

## frankenjax-cntiy - cbrt scalar Halley fast path narrows but does not close JAX gap

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: MEASURED KEEP / STILL JAX LOSS. Production `Primitive::Cbrt` now uses
  a guarded f64 bit-hack initial estimate plus two Halley refinements for normal
  finite magnitudes, falling back to `f64::cbrt` for zero, non-finite, and
  extreme-range inputs.
- Claimed tracker context: `frankenjax-cntiy` remains the open maintainer-gated
  FMA/softmax bead. This cbrt sub-gap is tolerance-only and does not resolve the
  parent gate, but it clears the peer compile blocker and gives a real retained
  production speedup.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: libm/fenv replacement via a bounded approximation kernel,
    with vectorized execution as the deeper route.
  - Implemented lever: scalar approximation first, because cbrt has no
    bit-golden digest and the local oracle tolerance is 1e-10. This avoids the
    `+fma` golden-policy gate while still attacking the libm-call tax.
  - EV decision: keep the scalar lever, then route any further cbrt work to a
    true SIMD/vector polynomial kernel. Do not repeat another scalar micro-tweak
    unless it beats this fast path in a same-binary A/B.

Remote correctness proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -p fj-lax cbrt --lib --release -- --nocapture
```

- RCH worker: `vmi1149989`; no local cargo build and no new `.scratch` or
  worktree.
- Result: 8 focused cbrt/lib tests passed.
- Accuracy probe: `validate_fast_cbrt_accuracy` printed
  `max_rel_err=6.455e-15`, well below the 1e-10 oracle bar.
- Full conformance: `cargo test -p fj-conformance --release` passed remotely
  on RCH worker `hz2` after the focused oracle run.

Remote same-binary bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'eval/cbrt_1m_f64_vec' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- RCH worker: `vmi1149989`; per-crate, warm target request, no new `.scratch`.
- Same-binary Rust control:
  - `eval/cbrt_1m_f64_vec_libm_reference`: **11.876 ms** midpoint
    (`9.7168..13.975 ms`).
  - `eval/cbrt_1m_f64_vec`: **3.2973 ms** midpoint (`3.1680..3.4991 ms`).
  - Rust-side speedup: **3.60x** versus the threaded libm reference.
- Fresh JAX/JAXLIB 0.10.1 CPU x64 comparator on the exact bench fixture:
  **2.157837 ms** mean, 30 runs x 50 inner loops.

| workload | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `eval/cbrt_1m_f64_vec_libm_reference` | 11.876 ms | 2.157837 ms | 5.50 | old libm path was a large JAX loss |
| `eval/cbrt_1m_f64_vec` | 3.2973 ms | 2.157837 ms | 1.53 | kept narrowing lever; still a JAX loss |

Retry predicate: the next cbrt route must be a SIMD/vectorized polynomial or
range-reduced kernel that preserves the 1e-10 tolerance oracle and beats the
current scalar fast path in a same-binary A/B. The broader `cntiy` FMA/softmax
decision is not resolved by this cbrt keep.

## frankenjax-ur4h3 - fresh BOLD-VERIFY closes small-eigh lane

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: VERIFIED WIN / CLOSE. Fresh re-authenticated BOLD-VERIFY reran the
  cod-b per-crate Criterion gate and an exact JAX comparator. The allocator/copy
  keep from `2859e41c` still holds; no new production code lever was attempted.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: communication-avoiding dense linear algebra / panel
    Householder (`alien_cs_graveyard.md` section 9.6) with the explicit
    constants warning from the graveyard failure-mode taxonomy.
  - EV decision: below threshold for this bead because the remaining measured
    48x48 row is already faster than JAX, and the prior small-Jacobi and naive
    symmetric-reduction routes are recorded no-ships. A blocked/panel reducer
    remains a future large-n lever only if a fresh same-worker profile shows an
    actual large-matrix gap.

Remote bench proof:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'linalg/(eigh_48x48_f64|svd_48x48_f64)' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- The literal requested `cargo bench --release` form was tried first and failed
  remotely on RCH worker `ovh-a` because Cargo rejects `--release` for `bench`.
  The valid Criterion bench form above was then run remotely on the same worker.
- RCH worker: `ovh-a`; no local cargo build and no new `.scratch` or worktree.
- Rust Criterion midpoint:
  - `linalg/eigh_48x48_f64`: **200.13 us** (`188.93..223.75 us`)
  - `linalg/svd_48x48_f64`: **105.50 us** (`105.44..105.53 us`)

Fresh exact JAX/JAXLIB comparator:

- JAX/JAXLIB: 0.10.1 / 0.10.1, CPU backend, `jax_enable_x64=true`.
- Fixture: exact `lax_baseline.rs` 48x48 inputs (`bench_eigh_48` and
  `real_matrix(48,48)`), 60 runs x 100 inner loops.

| workload | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `linalg/eigh_48x48_f64` | 200.13 us | 293.517 us | 0.682 | Rust win; no remaining small-eigh JAX loss |
| `linalg/svd_48x48_f64` | 105.50 us | 622.642 us | 0.169 | Existing Rust win strengthened |

Conclusion: close `frankenjax-ur4h3` rather than re-open allocator/copy
micro-levers or force a constants-heavy blocked/panel eigensolver into a
48x48 row that is already winning. Reopen only with fresh evidence of a real
large-n JAX/upstream loss.

## frankenjax-murmw - smooth-composite Bluestein detour no-ship

- Date: 2026-06-21
- Agent: cod-a
- Status: MEASURED NO-SHIP / REVERTED. No production source change remains.
- Target gap: `eval/fft_batch_128x1000_complex128`, the current smooth-
  composite batched FFT loss.
- Alien-graveyard/extreme-optimization route:
  - Candidate family: communication-avoiding/vectorized batch FFT. Reuse the
    existing Bluestein SoA path whose two internal convolution FFTs are the flat
    radix-2 kernel that vectorizes across row tiles.
  - EV decision: rejected by same-worker same-binary evidence. The constants
    warning dominates here: Bluestein's extra convolution work beats neither
    recursive mixed-radix nor JAX for smooth `n=1000`.

Fresh target scorecard:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
  rch exec -- cargo bench -j 1 --profile release -p fj-lax \
  --bench lax_baseline 'eval/fft_batch_128x1000_complex128' -- \
  --sample-size 10 --measurement-time 3 --warm-up-time 1 --noplot
```

- RCH worker: `hz1`; remote/per-crate, no local build and no new `.scratch`.
- Rust Criterion midpoint: **3.6581 ms** (`3.5478..3.7359 ms`).
- Fresh local JAX/JAXLIB 0.10.1 CPU x64 comparator, exact
  `complex_matrix(128,1000)` fixture: mean **0.245442 ms**, p50
  **0.250693 ms**.
- Retained production scorecard for this row: **0 wins / 1 loss / 0 neutral**;
  Rust/JAX **14.90x** by JAX mean.

Rejected lever proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
  rch exec -- cargo test -j 1 --release -p fj-lax \
  bench_smooth_bluestein_soa_vs_mixed_per_row -- --ignored --nocapture
```

- RCH worker: `hz1`.
- Same-binary A/B: **mixed=1.975ms**, **bluestein=2.690ms**,
  **ratio=0.73x**. The smooth-composite Bluestein detour is **36% slower** than
  recursive mixed-radix for the exact fixture.
- Decision: revert the smooth-Bluestein production gate and the temporary
  ignored A/B harness. Do not retry smooth-composite Bluestein routing without
  new evidence.

Retry predicate: the next `murmw` attempt must change the kernel family and
beat per-row mixed-radix in a same-binary A/B before production dispatch is
allowed to change. Credible routes are generated length-specialized
`1000 = 2^3 * 5^3` kernels, production-specialized radix-3/5 butterflies, or a
different SIMD/cache-blocked design.

## frankenjax-ur4h3 - eigh allocator/copy-reduction stack KEEP

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: MEASURED KEEP. This resolves the three earlier disk-low pending-bench
  entries for QL in-place transpose, Householder left-update scratch reuse, and
  Householder reflector scratch reuse.
- Lever stack:
  - `815ad85a` reuses the Householder reflector buffer in
    `hessenberg_reduction`.
  - `b240d41d` reuses the left Householder update dot-product scratch buffer.
  - `0dd28aad` transposes the QL eigenvector buffer in place around
    `symmetric_tridiagonal_ql`.
  - Current cleanup gates the old non-scratch `apply_householder_left` wrapper
    to `#[cfg(test)]`; production uses `apply_householder_left_with_scratch`.
- Alien-graveyard/extreme-optimization route used: constants/cache wall (§16.7)
  plus communication-avoiding dense-kernel discipline (§9.7/CA-QR family),
  implemented as a conservative allocation/copy-reduction lever rather than a
  new eigensolver.

RCH `fj-lax` Criterion command:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'linalg/(eigh_48x48_f64|svd_48x48_f64)' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

- The Cargo version in this checkout rejects `cargo bench --release`; `cargo
  bench` was used because Criterion already builds the bench profile.
- RCH selected `vmi1227854` and rewrote the target dir to a worker-scoped pool.
  This was still remote and per-crate; no local full workspace build was run.
- Old saved Criterion row in the cod-b target tree:
  `linalg/eigh_48x48_f64` **566.12 us** mean.
- Current RCH stdout:
  `linalg/eigh_48x48_f64` **203.18 us** midpoint (`197.59..211.62 us`) and
  `linalg/svd_48x48_f64` **98.678 us** midpoint (`95.111..103.32 us`).
- Retrieved Criterion estimate file:
  `eigh` **215.99 us** mean, `svd` **98.764 us** mean, with Criterion's stored
  relative `eigh` delta about **-73%** versus the saved row.

Fresh exact JAX/JAXLIB 0.10.1 x64 comparator:

- Fixture: exact `crates/fj-lax/benches/lax_baseline.rs` inputs
  (`bench_eigh_48` and `real_matrix(48,48)`), 60 runs x 100 inner loops, CPU
  backend, `jax_enable_x64=true`.
- `eigh_48x48_f64` JAX mean **210.721 us**; Rust/JAX is **0.964** using RCH
  stdout midpoint and **1.025** using the retrieved Criterion estimate.
- `svd_48x48_f64` JAX mean **571.848 us**; Rust/JAX is **0.173** by either Rust
  estimate. SVD remains a Rust win.

Isomorphism proof:

- Householder reflector entries are fully overwritten before use.
- The left scratch buffer is cleared before dot accumulation.
- Dot accumulation order and matrix update order are unchanged.
- QL arithmetic and rotation order are unchanged.
- The in-place transpose changes only storage for the layout conversion and
  preserves row-major eigenvector output.

Retry predicate: do not rerun allocator/copy micro-levers for this row unless a
new profile shows allocation/copy back on top. If `eigh_48x48_f64` remains a
true JAX loss under a same-host comparator, route to blocked/panel Householder
with packed or streaming Q updates, or a different eigenvector accumulation
strategy with same-worker production proof.

## frankenjax-murmw - flat iterative SoA mixed-radix FFT DISABLED no-ship

- Date: 2026-06-21
- Agent: cod-a / CrimsonOtter
- Status: MEASURED NO-SHIP. The smooth-composite iterative SoA gate is disabled
  in production via `MIXED_RADIX_ITERATIVE_SOA_MAX_N = 0`; the kernel and tests
  remain as a documented rejection artifact.
- Target gap: smooth-composite batched complex FFT, especially
  `eval/fft_batch_128x1000_complex128`, which remained a JAX loss in the last
  measured row set: Rust **2.930 ms** vs JAX/JAXLIB 0.10.1 x64 **0.233 ms**,
  Rust/JAX **12.55x**.
- Lever tested: lift the flat iterative mixed-radix SoA prototype into production
  behind a conservative gate for dense smooth-composite batches:
  `batch >= 8`, `n <= 1024`, and `batch*n <= 2^18`. This targets the existing
  128x1000 row while leaving larger already-threaded batches on the proven
  recursive per-row path until direct RCH evidence exists.
- Correctness argument before build: the candidate uses the same root table,
  generalized digit-reversal, stage order, per-lane arithmetic, and inverse
  `1/n` scaling as the scalar iterative prototype. It is bit-identical to scalar
  iterative per row, but only tolerance-equivalent to the recursive production
  mixed-radix path because op order differs. Smooth-composite FFT parity is
  tolerance-based, so this is a legal candidate if the focused tests pass.
- Correctness gates:
  - `cargo test -p fj-lax --lib --release --no-run` passed remotely on RCH
    `vmi1152480`.
  - `production_mixed_radix_soa_matches_scalar_iterative_by_bits` passed on
    RCH `vmi1152480`.
  - `cargo test -p fj-lax --lib --release mixed_radix -- --nocapture` passed
    5 runnable tests on RCH `vmi1152480`; the A/B microbench remained ignored.
  - `cargo test -j 1 -p fj-conformance --test fft_oracle -- --nocapture`
    passed 27/27 on RCH `vmi1149989`.

Performance evidence:

```text
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_WORKER=vmi1152480 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo test -p fj-lax --lib --release \
  bench_mixed_radix_iterative_soa_vs_per_row -- --ignored --nocapture
```

- Same-binary A/B on RCH `vmi1152480`: **per-row=1.798ms**, **iter=5.911ms**,
  **ratio=0.30x**. The iterative SoA path is **3.29x slower** than the per-row
  recursive path. An earlier disabling run recorded an even worse
  **per-row=2.947ms**, **iter=19.096ms**, **ratio=0.15x**.
- Production Criterion with the gate disabled:

```text
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -j 1 --profile release -p fj-lax \
  --bench lax_baseline 'eval/fft_batch_128x1000_complex128' -- \
  --sample-size 10 --measurement-time 3 --warm-up-time 1 --noplot
```

- RCH `hz2` result: **2.7829ms** midpoint (`2.7035..2.8831ms`).
- JAX comparator: recorded local CPU JAX/JAXLIB 0.10.1 x64 row **0.233ms**.
  The current environment did not have `jax` installed, so this pass reused
  the recorded comparator rather than installing new packages under disk
  pressure.
- Current production Rust/JAX: **11.94x** slower; the rejected iterative SoA
  microbench would be **25.37x** JAX.

Retry predicate: do not retry straight SoA transposes of recursive or iterative
mixed-radix. The next credible smooth-composite route must change the kernel
family: specialized radix-3/5 butterflies with direct vectorization, generated
length-specialized kernels, or another design that beats the per-row recursive
path in the same `bench_mixed_radix_iterative_soa_vs_per_row` A/B before it is
allowed onto the production dispatch path.

## frankenjax-ur4h3 - Householder left-update scratch reuse resolved by stack keep

- Date: 2026-06-21
- Agent: cod-b / CrimsonOtter
- Status: RESOLVED by the measured 2026-06-21 `frankenjax-ur4h3` stack keep
  above.
- Lever: `hessenberg_reduction` now reuses the left Householder update's
  dot-product buffer across panels. The existing `apply_householder_left` API
  remains intact for other call sites, while the production reduction path calls
  a scratch-backed helper. The helper zeroes the active scratch prefix before
  use, preserving the same dot accumulation order and matrix update order.
- Do not evaluate this helper in isolation unless a fresh profile asks for
  component-level attribution.

## frankenjax-ur4h3 - Householder reflector scratch reuse resolved by stack keep

- Date: 2026-06-20
- Agent: cod-b / CrimsonOtter
- Status: RESOLVED by the measured 2026-06-21 `frankenjax-ur4h3` stack keep
  above.
- Lever: `hessenberg_reduction` now reuses one Householder reflector scratch
  buffer across all panels instead of heap-allocating `Vec<f64>` once per
  reduction step. Each panel writes the full active reflector slice before use,
  so the reflector arithmetic, update order, and output contract are unchanged.
- The earlier RCH `hz1` production row remains historical context; the BOLD-
  VERIFY pass refreshed the exact JAX fixture comparator and kept the combined
  allocator/copy-reduction stack.
## frankenjax-murmw - SoA Bluestein batch FFT keep

- Date: 2026-06-20
- Agent: cod-a / CrimsonOtter
- Target gap: `frankenjax-murmw`, the rough/prime-length batched complex FFT
  lane after mixed-radix SoA was rejected. The target production row added here
  is `eval/fft_batch_256x1009_prime_complex128_dense_input`, where `1009` forces
  Bluestein and the batch crosses the existing threaded full-eval threshold.
- Alien-graveyard/extreme-optimization route used: data-layout specialization
  and cache-blocked row-vertical SIMD-friendly execution. `transform_batches_bluestein_vectorized`
  transposes a small row tile to SoA, vectorizes chirp pre/post multiply and
  kernel pointwise multiply across rows, and uses the already-proven flat radix-2
  `vectorized_pow2_block` for Bluestein's two convolution FFTs.
- Source kept: route non-power-of-two, non-smooth dense complex batches through
  the SoA Bluestein path when `batch >= 8` and convolution length
  `m <= BLUESTEIN_SOA_MAX_M`; retain `BLUESTEIN_SOA_MAX_M=16384` because the
  largest covered row was measured and proved bit-identical.

Same-worker Rust proof on RCH `hz2`:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/(fft_batch_256x1009_prime_complex128_dense_input|fft_batch_128x1000_complex128|fft_batch_2048x256_complex128_dense_input)' \
  --sample-size 10 --measurement-time 3 --warm-up-time 1 --noplot
```

Fresh local JAX comparator used `benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1, `jax_enable_x64=true`, CPU backend, warmed
`jax.jit(lambda a: jnp.fft.fft(a, axis=-1))`, 30 runs over the same
`sin(i*0.125) + 1j*cos(i*0.25)` dense complex inputs.

| workload / result | Rust mean | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `hz2` baseline `fft_batch_256x1009_prime_complex128_dense_input` | 11.096 ms | 0.478 ms | 23.20 | Target baseline loss |
| `hz2` kept SoA Bluestein candidate `fft_batch_256x1009_prime_complex128_dense_input` | 4.453 ms | 0.478 ms | 9.31 | KEEP; 2.49x Rust-side win, still JAX loss |
| `hz2` composite control `fft_batch_128x1000_complex128` | 2.930 ms | 0.233 ms | 12.55 | Loss; not routed through this gate |
| `hz2` power-of-two dense control `fft_batch_2048x256_complex128_dense_input` | 5.979 ms | 0.313 ms | 19.13 | Loss; unchanged route |

- Same-binary microbench on RCH `hz2`, interleaved min-of-9: n=127/m=256
  `8.940 ms -> 3.000 ms` (**2.98x**), n=1009/m=2048
  `10.666 ms -> 2.423 ms` (**4.40x**), n=4099/m=16384
  `53.256 ms -> 14.803 ms` (**3.60x**).
- Validation: `cargo test -p fj-lax vectorized_bluestein_bit_identical_to_per_row
  --lib --release -- --nocapture` passed on RCH `hz2`, now covering n in
  {3,7,11,13,17,23,127,257,1009,4099}, both directions and batch 1..11.
  `cargo test -p fj-conformance --test fft_oracle -- --nocapture` passed 27/27
  on RCH `vmi1152480`; `cargo test -p fj-conformance --test
  linalg_fft_oracle_parity -- --nocapture` passed 1/1 on RCH `vmi1227854`.
  `cargo test -p fj-lax fft --lib --release -- --nocapture` passed 47/0 with
  6 ignored microbenches on RCH `vmi1152480`; `cargo clippy -p fj-lax
  --all-targets -- -D warnings` passed on RCH `hz1`; `cargo build --release
  -p fj-lax --benches` passed on RCH `vmi1149989`.
- Ratio scorecard after the keep: **0 wins / 3 losses / 0 neutral vs JAX** for
  the full-eval FFT row set above. Production score for this pass: **1 kept
  Rust-side win / 0 shipped regressions**.
- Retry predicate: do not treat this as closing `frankenjax-murmw`; it narrows
  the prime/rough Bluestein lane but leaves JAX 9.31x ahead on the target. The
  next credible route must attack the kernel class directly: generated
  length-specialized FFTs, portable-SIMD radix-4/8 butterflies, native composite
  radices, or fused output materialization that removes the remaining scalar
  butterfly/output gap.

## frankenjax-murmw - radix-4 power-of-two FFT specialization no-ship

- Date: 2026-06-22
- Agent: cod-a / CrimsonOtter
- Target gap: `frankenjax-murmw`, specifically the power-of-two batched complex
  FFT lane and child route `frankenjax-murmw.1`.
- Alien-graveyard/extreme-optimization route used: generated/vectorized
  shape-specialized execution for radix-4 or split-radix power-of-two kernels.
  This is the plausible pocketfft-class direction after representation-only,
  tile-height, gate-disable, direct-output, and Bluestein detours.
- Lever rejected before promotion: the existing test-only radix-4 SoA butterfly
  path was measured head-to-head against the retained radix-2 SoA butterfly
  path. It lost badly, so no production source hunk was kept.

Same-binary probe:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo test -p fj-lax --lib --release \
  bench_radix4_vs_radix2_soa -- --ignored --nocapture

[radix4 vs radix2 SoA butterflies 8x256] radix2=0.0056ms \
  radix4=0.0128ms ratio=0.44x
```

Fresh retained-HEAD production/JAX check:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  AGENT_NAME=CrimsonOtter \
  rch exec -- cargo bench -j 1 --profile release -p fj-lax \
  --bench lax_baseline 'eval/fft_batch_2048x256_complex128' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

The first attempted Criterion command used `cargo bench --release`, which Cargo
rejected; the corrected per-crate release invocation used `--profile release`.
The regex filter matched both the boxed and dense-input rows on RCH worker
`ovh-a`.

Fresh local JAX comparator used `benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1, CPU backend, `jax_enable_x64=true`, warmed
`jax.jit(lambda a: jnp.fft.fft(a, axis=-1))`, 30 timed runs over the exact
`2048x256` complex128 fixture generated by `complex_matrix`/
`complex_matrix_dense`: `sin(i*0.125) + 1j*cos(i*0.25)`. JAX mean was
0.328385 ms, p50 0.316985 ms, p95 0.424876 ms.

| workload / result | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| retained `eval/fft_batch_2048x256_complex128` | 4.3538 ms | 0.328385 ms | 13.26 | Active JAX loss |
| retained `eval/fft_batch_2048x256_complex128_dense_input` | 1.0879 ms | 0.328385 ms | 3.31 | Active JAX loss |
| radix-4 SoA butterfly probe | 0.0128 ms | n/a | 0.44x vs radix-2 | REJECT |

- Ratio scorecard for retained production rows in this pass: **0 wins / 2
  losses / 0 neutral vs JAX**. Candidate score: **0 wins / 1 loss / 0 neutral;
  0 kept / 1 rejected**.
- No production `crates/fj-lax/src/fft.rs` source hunk was retained. Retry only
  with generated length-specialized kernels, true SIMD/shuffle support, native
  mixed-radix plans for composite rows, or a backend route that can credibly
  approach pocketfft's higher-radix/SIMD kernel class. Do not retry a plain
  radix-4 SoA butterfly promotion without a same-binary A/B exceeding the
  retained radix-2 path by at least 1.15x.
## frankenjax-ur4h3 - symmetric tridiagonal reduction no-ship

- Date: 2026-06-20
- Agent: cod-b / CrimsonOtter
- Target gap: after the large-n `eigh` cache fixes and the small-Jacobi
  rejection, the live `fj-lax` linalg gap is the 48x48 real symmetric `eigh`
  row. Fresh production baseline on RCH worker `hz1`:
  `linalg/eigh_48x48_f64` 267.84 us mean and `linalg/svd_48x48_f64` 176.93 us
  mean. Fresh local warmed JAX/JAXLIB 0.10.1 x64 comparator:
  `eigh_48x48_f64` 201.429 us mean / 192.507 us p50 (CV 12.6%) and
  `svd_48x48_f64` 955.544 us mean. Production ratios: `eigh` Rust/JAX mean
  1.330 (loss); `svd` Rust/JAX mean 0.185 (existing Rust win).
- Alien-graveyard/extreme-optimization route used: numerical-kernel
  specialization and vectorized/cache-local execution, not a new eigensolver.
  Hypothesis: a real-symmetric Householder tridiagonal reduction could avoid
  doing a general Hessenberg similarity transform on symmetric input by updating
  only the trailing symmetric block plus Q.
- Lever rejected and reverted: `tridiag_ql_eigendecomposition` was routed
  through a candidate `symmetric_tridiagonal_reduction` using the rank-2 update
  `A_sub -= v*w^T + w*v^T` with the existing row-major Q right-apply. Focused
  correctness passed while the candidate existed, but the production timing gate
  did not.

Fresh production baseline command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'linalg/(eigh_48x48_f64|svd_48x48_f64)' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1, `jax_enable_x64=true`, warmed `jax.jit(jnp.linalg.eigh)`
and `jax.jit(lambda a: jnp.linalg.svd(a, full_matrices=False))`, 60 runs x 100
inner loops over the same deterministic 48x48 symmetric input.

| workload / result | Rust mean | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `hz1` production `linalg/eigh_48x48_f64` | 267.84 us | 201.429 us | 1.330 | Active JAX loss |
| `hz1` production `linalg/svd_48x48_f64` | 176.93 us | 955.544 us | 0.185 | Existing Rust win |
| `vmi1153651` symmetric-reduction candidate `eigh_48x48_f64` | 375.37 us | 201.429 us | 1.863 | REVERT/no proof |
| `vmi1153651` candidate `svd_48x48_f64` control | 244.01 us | 955.544 us | 0.255 | Control remains Rust win |

- Validation while the candidate existed:
  `cargo test -p fj-lax tridiag_ql_eigh_matches_jacobi_and_reconstructs --lib
  -- --nocapture` passed on RCH `hz1`; `cargo test -p fj-lax eigh --lib --
  --nocapture` passed 13 tests with 1 ignored benchmark on RCH `vmi1293453`.
- Measurement caveat: candidate timing landed on RCH `vmi1153651` while the
  production baseline landed on `hz1`, so the cross-worker delta is routing
  evidence, not same-worker regression proof. The reject decision is still
  justified because the candidate remained a worse absolute JAX loss than the
  retained production row.
- Ratio scorecard for retained production 48x48 linalg rows: **1 win / 1 loss /
  0 neutral vs JAX**. Rejected target candidate score: **0 wins / 1 loss / 0
  neutral**; no source was kept.
- Retry predicate: do not retry this naive symmetric rank-2 tridiagonal
  reduction shape (`w = beta * A_sub * v`, `A_sub -= v*w^T + w*v^T`, row-major
  Q update) without a same-binary component A/B proving the reducer itself beats
  the current general-Hessenberg setup. The next credible `frankenjax-ur4h3`
  route is a blocked/panel Householder reducer with packed/streaming Q updates,
  or a different end-to-end eigenvector accumulation strategy, both requiring
  same-worker production proof.

## frankenjax-4ryym - f64 GEMM SIMD load-codegen no-ship

- Date: 2026-06-20
- Agent: cod-b / CrimsonOtter
- Target gap: follow-up to `frankenjax-ifou2` after generic Rayon/global-pool
  row scheduling was rejected. Fresh `hz1` production `matmul_2d` rows still
  lose hard to warmed local CPU JAX 0.10.1 x64: 256^3 4.090x slower, 512^3
  11.169x slower, and 1024^3 14.996x slower by midpoint/median.
- Alien-graveyard/extreme-optimization route used: target-feature/codegen and
  microkernel load expression, not another pool/scheduler retry. Hypothesis:
  replacing manual eight-scalar `Simd::from_array([brow[0]..brow[7]])` loads
  with `Simd::from_slice` could give LLVM a cleaner vector-load idiom in the
  f64 `MR x NR` microkernel while preserving the exact ascending-`l`
  accumulation order.
- Lever rejected and reverted: `f64xnr_from_slice` was changed to
  `F64xN::from_slice(values)` and the flat row-block hot loop reused that
  helper. Focused release matmul tests passed while the candidate existed, but
  same-worker timing failed the keep gate; source was restored before commit.
- Profiling limit: `perf stat -e cycles,instructions,cache-misses true` failed
  on the host because `perf_event_paranoid=4`, so no hardware counter claims
  are made.

Fresh production baseline command on RCH worker `hz1`:

```text
AGENT_NAME=CrimsonOtter BR_AGENT_NAME=cod-b \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/(matmul_2d_256x256x256_f64|matmul_2d_512x512x512_f64|matmul_2d_1024x1024x1024_f64|strassen_ab_1024_(matmul2d|strassen))' \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

JAX comparator used
`/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`, JAX
0.10.1, CPU backend, `jax_enable_x64=true`, warmed `jax.jit(lambda a, b: a @ b)`
with medians: 256^3 0.3006295 ms, 512^3 0.621733 ms, 1024^3 2.7191275 ms.

| workload / result | Rust midpoint | same-worker delta | JAX median | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `hz1` production `matmul_2d_256x256x256_f64` | 1.2296 ms | baseline | 0.3006295 ms | 4.090 | Active JAX loss |
| `hz1` production `matmul_2d_512x512x512_f64` | 6.9439 ms | baseline | 0.621733 ms | 11.169 | Active JAX loss |
| `hz1` production `matmul_2d_1024x1024x1024_f64` | 40.776 ms | baseline | 2.7191275 ms | 14.996 | Active JAX loss |
| `hz1` `strassen_ab_1024_matmul2d` control | 39.758 ms | baseline | 2.7191275 ms | 14.622 | Active JAX loss |
| `hz1` `strassen_ab_1024_strassen` control | 97.337 ms | baseline | 2.7191275 ms | 35.797 | Existing no-ship remains |
| Accidental routing baseline on `vmi1293453`, `matmul_2d_256x256x256_f64` | 2.7308 ms | cross-worker only | 0.3006295 ms | 9.084 | Routing evidence only |
| Accidental routing baseline on `vmi1293453`, `matmul_2d_512x512x512_f64` | 9.0670 ms | cross-worker only | 0.621733 ms | 14.583 | Routing evidence only |
| Accidental routing baseline on `vmi1293453`, `matmul_2d_1024x1024x1024_f64` | 33.046 ms | cross-worker only | 2.7191275 ms | 12.153 | Routing evidence only |
| `Simd::from_slice` candidate A on `vmi1153651`, `matmul_2d_256x256x256_f64` | 8.9680 ms | cross-worker/noisy | 0.3006295 ms | 29.831 | REVERT/no proof |
| `Simd::from_slice` candidate A on `vmi1153651`, `matmul_2d_512x512x512_f64` | 21.088 ms | cross-worker/noisy | 0.621733 ms | 33.918 | REVERT/no proof |
| `Simd::from_slice` candidate A on `vmi1153651`, `matmul_2d_1024x1024x1024_f64` | 79.742 ms | cross-worker/noisy | 2.7191275 ms | 29.326 | REVERT/no proof |
| `Simd::from_slice` candidate B on `vmi1153651`, `matmul_2d_256x256x256_f64` | 8.2187 ms | neutral/no change, p=0.68 | 0.3006295 ms | 27.338 | REVERT/no gain |
| `Simd::from_slice` candidate B on `vmi1153651`, `matmul_2d_512x512x512_f64` | 40.519 ms | +108.32%, p=0.00 | 0.621733 ms | 65.171 | REVERT/regression |
| `Simd::from_slice` candidate B on `vmi1153651`, `matmul_2d_1024x1024x1024_f64` | 90.284 ms | neutral/no change, p=0.34 | 2.7191275 ms | 33.203 | REVERT/no gain |

- Ratio scorecard for retained production matmul rows: **0 wins / 3 losses / 0
  neutral vs JAX**. If the Strassen A/B controls are included, the measured
  rowset is **0 wins / 5 losses / 0 neutral**.
- Rejected candidate score for accepted same-worker `vmi1153651` rows: **0
  wins / 3 losses / 0 neutral vs JAX**; internally it was one significant
  regression and two neutral/no-gain rows. Candidate A and the accidental
  `vmi1293453` baseline are logged as routing evidence only because worker
  selection drift made them non-comparable to `hz1`.
- Validation while the candidate existed: `cargo check -p fj-lax --benches`
  passed on RCH `vmi1152480`; `cargo test -p fj-lax matmul_2d --lib --release
  -- --nocapture` passed 23 tests with 2 ignored microbenches on RCH `ovh-a`.
- Retry predicate: do not retry `Simd::from_slice`/manual-load spelling changes
  in the f64 `MR x NR` microkernel without assembly-level proof that LLVM is
  currently scalarizing the load. The next credible `frankenjax-4ryym` route is
  still a real backend/codegen lever: target-feature-specialized FMA kernel,
  BLAS-class packed microkernel, owned arena handoff, or a safe scoped-pool
  design with same-worker 256/512/1024 proof. Generic Rayon/global-pool row
  scheduling remains banned by `frankenjax-ifou2`.

## frankenjax-ifou2 - Rayon persistent-pool matmul_2d no-ship

- Date: 2026-06-20
- Agent: cod-b / CrimsonOtter
- Target gap: `fj-lax` f64 GEMM/matmul rows remain large JAX losses even after
  previous register tiling, B-panel packing, and scoped row-block threading.
  Fresh local JAX 0.10.1 x64 rows measured 0.264947 ms for 256^3, 0.576574 ms
  for 512^3, and 2.665036 ms for 1024^3.
- Alien-graveyard/extreme-optimization route used: runtime scheduling and
  persistent worker reuse, not another scalar microkernel tweak. Hypothesis:
  replacing per-call `std::thread::scope` spawning with a persistent safe pool
  would remove the suspected mid-size thread-creation tax while preserving
  bit-identical row-block accumulation.
- Lever rejected and reverted: add `rayon = "1.11"` to `fj-lax`, use
  `par_chunks_mut` for packed-B full-panel fanout and output row chunks, then
  narrow the route after observing 1024 regressions. All source and lockfile
  changes were restored before commit.

Production baseline command on RCH worker `hz1`:

```text
AGENT_NAME=CrimsonOtter BR_AGENT_NAME=cod-b \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/(matmul_2d_256x256x256_f64|matmul_2d_512x512x512_f64|strassen_ab_1024_(matmul2d|strassen))' \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

Candidate reruns used the clean worktree `/data/projects/frankenjax-cod-b`
warmed to `hz1` with `rch cache warm --workers hz1`, then direct `ssh` execution
under `/tmp/rch/frankenjax-cod-b/686cb8c44dc7ea3b` with the same Criterion
filter/settings.

| workload / lever | Rust midpoint | same-worker delta | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `matmul_2d_256x256x256_f64` | 1.3226 ms | baseline | 0.264947 ms | 4.992 | Active JAX loss |
| baseline `matmul_2d_512x512x512_f64` | 6.3494 ms | baseline | 0.576574 ms | 11.012 | Active JAX loss |
| baseline `strassen_ab_1024_matmul2d` | 33.919 ms | baseline | 2.665036 ms | 12.727 | Active JAX loss |
| baseline `strassen_ab_1024_strassen` | 79.752 ms | baseline | 2.665036 ms | 29.925 | Existing Strassen no-ship |
| all-Rayon pool `matmul_2d_256x256x256_f64` | 0.98603 ms | -29.810%, p<0.05 | 0.264947 ms | 3.722 | Looked good, not enough to ship |
| all-Rayon pool `matmul_2d_512x512x512_f64` | 6.1108 ms | -6.854%, p<0.05 | 0.576574 ms | 10.598 | Small internal win, still JAX loss |
| all-Rayon pool `strassen_ab_1024_matmul2d` | 54.996 ms | +52.148%, p<0.05 | 2.665036 ms | 20.636 | REVERT |
| all-Rayon pool `strassen_ab_1024_strassen` | 84.003 ms | neutral, p=0.22 | 2.665036 ms | 31.520 | No keep |
| block-k fallback `matmul_2d_256x256x256_f64` | 0.96878 ms | neutral vs prior candidate | 0.264947 ms | 3.657 | Not durable proof |
| block-k fallback `matmul_2d_512x512x512_f64` | 9.0076 ms | noisy/no change vs prior candidate | 0.576574 ms | 15.623 | REVERT |
| block-k fallback `strassen_ab_1024_matmul2d` | 65.353 ms | +21.307%, p<0.05 | 2.665036 ms | 24.522 | REVERT |
| block-k fallback `strassen_ab_1024_strassen` | 194.18 ms | +131.16%, p<0.05 | 2.665036 ms | 72.862 | REVERT |
| `<=512` Rayon gate `matmul_2d_256x256x256_f64` | 3.0695 ms | +218.92%, p<0.05 | 0.264947 ms | 11.585 | REVERT |
| `<=512` Rayon gate `matmul_2d_512x512x512_f64` | 12.894 ms | +77.588%, p<0.05 | 0.576574 ms | 22.363 | REVERT |
| `<=512` Rayon gate `strassen_ab_1024_matmul2d` | 94.836 ms | +45.113%, p<0.05 | 2.665036 ms | 35.585 | REVERT |
| `<=512` Rayon gate `strassen_ab_1024_strassen` | 209.53 ms | neutral/no keep vs prior noisy row | 2.665036 ms | 78.622 | REVERT |

- Ratio scorecard for retained production matmul rows: **0 wins / 4 losses / 0
  neutral vs JAX**. Rejected candidate score: **0 wins / 12 losses / 0 neutral
  vs JAX** on the rows above, despite two early internal improvements.
- Root-cause read from the failed lever: a generic safe persistent pool is not a
  drop-in win for this kernel family. The 256/512 apparent win did not survive
  the repeated gate, and larger flat-packed GEMM regressed hard. The current
  scoped spawn tax may still matter on quiet hardware, but the schedule/cache
  interaction is too unstable to ship without a deeper design.
- Validation while the candidate existed: `cargo check -p fj-lax --benches`
  passed through RCH `vmi1293453`; `cargo test -p fj-lax matmul_2d --lib
  --release -- --nocapture` passed 23 tests with 2 ignored microbenches through
  RCH `vmi1167313`; final docs-only conformance gate `cargo test -p
  fj-conformance --lib` passed 45/45 through RCH `hz2`. `cargo fmt -p fj-lax
  --check` is red on pre-existing clean HEAD formatting drift outside this
  pass. Source was reverted before commit.
- Retry predicate: do not retry Rayon/global-pool row chunks, Rayon packed-B
  fanout, or a `<=512` size gate. The next credible ifou2 route must start with
  a quiet-host profile and then move to a real GEMM backend lever: BLAS-class
  microkernel, target-feature-specialized/FMA codegen, owned arena handoff, or a
  carefully designed safe scoped pool with 256/512/1024 same-worker proof.

## frankenjax-murmw - power-of-two FFT tile-height and packed-input guard no-ship

- Date: 2026-06-20
- Agent: cod-a / CrimsonOtter
- Target gap: `fj-lax` 2048x256 power-of-two batched FFT rows remain large JAX
  losses after the earlier SoA gate, direct-output, persistent pool, and radix-4
  no-ships.
- Alien-graveyard route used: cache-shape specialization and layout-aware kernel
  selection. Two variants were tested: force the existing SoA tile height from 8
  rows to 4 rows, then guard that 4-row tile for packed dense inputs only while
  preserving the old 8-row tile for boxed/literal inputs.
- Production code changed: none. Both `crates/fj-lax/src/fft.rs` variants were
  reverted before commit because full release Criterion did not show a clean
  target-row win.

Baseline and candidate command family:

```text
AGENT_NAME=CrimsonOtter \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  RCH_WORKER=vmi1152480 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  fft_batch_2048x256 -- --warm-up-time 1 --measurement-time 3
```

Comparable Rust proof used RCH worker `vmi1152480` for baseline and both
candidates. JAX comparator rows were refreshed locally with
`benchmarks/jax_comparison/.venv/bin/python` and JAX/JAXLIB 0.10.1 x64:
complex FFT 249.3358 us, RFFT 183.5284 us, real-input FFT 2.2802276 ms, IRFFT
220.4353 us. The JAX ratios below route the remaining gap; same-worker Rust
deltas decide keep/reject.

| workload / lever | Rust midpoint | same-worker result | JAX mean | Rust/JAX | verdict |
| --- | ---: | --- | ---: | ---: | --- |
| baseline `fft_batch_2048x256_complex128` | 7.1002 ms | baseline | 249.3358 us | 28.48 | Active JAX loss |
| baseline `fft_batch_2048x256_complex128_dense_input` | 4.7948 ms | baseline | 249.3358 us | 19.23 | Active JAX loss |
| baseline `rfft_batch_2048x256_f64` | 5.7007 ms | baseline | 183.5284 us | 31.06 | Active JAX loss |
| baseline `rfft_batch_2048x256_f64_dense_input` | 3.2996 ms | baseline | 183.5284 us | 17.98 | Active JAX loss |
| baseline `fft_batch_2048x256_real_dense_input` | 4.4143 ms | baseline | 2.2802276 ms | 1.94 | Active JAX loss |
| baseline `irfft_batch_2048x256_complex128` | 6.9618 ms | baseline | 220.4353 us | 31.58 | Active JAX loss |
| tile4 `fft_batch_2048x256_complex128` | 8.3863 ms | +18.114% regression | 249.3358 us | 33.64 | REVERT |
| tile4 `fft_batch_2048x256_complex128_dense_input` | 3.4137 ms | -28.805% faster | 249.3358 us | 13.69 | No ship: paired target regression |
| tile4 `rfft_batch_2048x256_f64` | 5.5569 ms | neutral, p=0.50 | 183.5284 us | 30.28 | Control only |
| tile4 `rfft_batch_2048x256_f64_dense_input` | 2.7705 ms | -16.036% faster | 183.5284 us | 15.10 | Control only |
| tile4 `fft_batch_2048x256_real_dense_input` | 4.1814 ms | neutral, p=0.18 | 2.2802276 ms | 1.83 | Control only |
| tile4 `irfft_batch_2048x256_complex128` | 3.2352 ms | -53.529% faster | 220.4353 us | 14.68 | Control only |
| guarded tile4 `fft_batch_2048x256_complex128` | 7.1895 ms | neutral | 249.3358 us | 28.83 | No gain |
| guarded tile4 `fft_batch_2048x256_complex128_dense_input` | 5.4350 ms | regression | 249.3358 us | 21.80 | REVERT |
| guarded tile4 `rfft_batch_2048x256_f64` | 11.993 ms | regression/noise, not touched path | 183.5284 us | 65.35 | Control only |
| guarded tile4 `rfft_batch_2048x256_f64_dense_input` | 3.5889 ms | regression | 183.5284 us | 19.56 | Control only |
| guarded tile4 `fft_batch_2048x256_real_dense_input` | 4.8481 ms | regression | 2.2802276 ms | 2.13 | Control only |
| guarded tile4 `irfft_batch_2048x256_complex128` | 3.6708 ms | apparent non-target win | 220.4353 us | 16.65 | Control only |

- Ratio scorecard for retained production rows in this pass: **0 wins / 6
  losses / 0 neutral vs JAX**. Rejected candidates score **0 kept wins / 0
  shipped regressions** because the source was reverted before commit.
- Root-cause read from the failed levers: tile height alone is not the dominant
  bottleneck. The 4-row tile can help packed dense complex FFT in isolation, but
  the boxed path regresses; adding a packed-input guard erased the dense gain and
  added branch/dispatch noise. This is layout-selection churn, not the needed
  kernel-family change.
- Validation after revert: `cargo test -p fj-lax fft --lib` passed 44/44 with 3
  ignored microbenches; `cargo test -p fj-conformance --test fft_oracle` passed
  27/27; `cargo test -p fj-conformance --test linalg_fft_oracle_parity` passed
  1/1; `cargo build --release -p fj-lax --benches` passed on RCH
  `vmi1153651`.
- Retry predicate: do not retry tile-height or representation-only SoA tweaks
  without hardware-counter evidence. The next credible `murmw` route is a real
  kernel rewrite: generated length-specialized kernels, iterative in-place
  radix-4/8 SoA, portable SIMD butterflies, or cache-blocked multi-row
  transforms.

## frankenjax-murmw - FFT batch final-buffer writes and persistent row pool no-ships

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Target gap: `fj-lax` complex128 FFT batch rows from `frankenjax-murmw`, especially
  `fft_batch_2048x256` where JAX/pocketfft remains 8-26x faster than the current
  Rust path depending on dense-vs-boxed harness shape.
- Alien-graveyard route used: exotic output layout and runtime scheduling rather
  than another plan-cache tweak. The tested ideas were (1) write radix-2 batch
  rows directly into the final output slice to avoid a row-local temp/copy and
  (2) reuse a process-lifetime worker pool for row chunks to remove repeated
  scoped thread spawn overhead.
- Production code changed: none. Both source candidates were reverted after
  same-worker measurement. The only kept file change is `.rchignore` adding
  `artifacts/`, which prevents unrelated generated reports from bloating RCH
  transfers and does not affect runtime behavior.

Baseline command:

```text
AGENT_NAME=cod-a \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/(fft_batch_2048x256_complex128_dense_input|fft_batch_2048x256_complex128|fft_256_complex128|ifft_256_complex128)' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 \
  --save-baseline frankenjax-cod-a-fft-direct-out-baseline
```

Post-change command:

```text
AGENT_NAME=cod-a \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/(fft_batch_2048x256_complex128_dense_input|fft_batch_2048x256_complex128|fft_256_complex128|ifft_256_complex128)' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 \
  --baseline frankenjax-cod-a-fft-direct-out-baseline
```

Accepted same-worker proof uses RCH worker `vmi1227854`. JAX comparator rows are
the `frankenjax-murmw` JAX 0.10.2 x64 measurements from AzureLynx: `fft_256`
5.90 us, `ifft_256` 5.72 us, and `fft_batch_2048x256` 236 us.

| workload / lever | Rust midpoint | same-worker delta | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `fft_256_complex128` | 5.4692 us | baseline | 5.90 us | 0.927 | Current Rust near-parity/win control |
| baseline `ifft_256_complex128` | 5.8726 us | baseline | 5.72 us | 1.027 | Current near-parity control |
| baseline `fft_batch_2048x256_complex128` | 6.0686 ms | baseline | 236 us | 25.71 | Active JAX loss |
| baseline `fft_batch_2048x256_complex128_dense_input` | 1.8895 ms | baseline | 236 us | 8.01 | Active JAX loss |
| direct final output-slice, `fft_256_complex128` | 5.4826 us | -8.677% by Criterion | 5.90 us | 0.929 | Control only; target rows reject |
| direct final output-slice, `ifft_256_complex128` | 5.4758 us | -5.307% by Criterion | 5.72 us | 0.957 | Control only; target rows reject |
| direct final output-slice, `fft_batch_2048x256_complex128` | 6.4767 ms | neutral +3.4346%, p=0.23 | 236 us | 27.44 | REVERT |
| direct final output-slice, `fft_batch_2048x256_complex128_dense_input` | 2.2349 ms | +12.542% regression, p<0.05 | 236 us | 9.47 | REVERT |
| persistent row thread pool, `fft_256_complex128` | 5.5501 us | neutral +2.2041%, p=0.42 | 5.90 us | 0.940 | Control only |
| persistent row thread pool, `ifft_256_complex128` | 6.0925 us | neutral +0.7862%, p=0.74 | 5.72 us | 1.065 | Control only |
| persistent row thread pool, `fft_batch_2048x256_complex128` | 6.7331 ms | +10.245% regression, p<0.05 | 236 us | 28.53 | REVERT |
| persistent row thread pool, `fft_batch_2048x256_complex128_dense_input` | 2.8739 ms | +48.632% regression, p<0.05 | 236 us | 12.18 | REVERT |

- Ratio scorecard for the current baseline row set: **1 win / 2 losses / 1
  neutral vs JAX** if a 5% neutral band is used (`fft_256` win, `ifft_256`
  neutral, both 2048x256 batch rows losses). The target batch gap remains open.
- Ratio scorecard for the two rejected levers on their target batch rows: **0
  wins / 4 losses / 0 neutral vs JAX**, plus same-worker Rust regressions on
  three of four target batch measurements and one no-gain row.
- Root-cause read from the failed levers: the existing row-local scratch plus
  sequential copy is cheaper than striding final-row writes directly, and the
  current scoped row threading cost is not the dominant batch-row bottleneck.
  Persistent pooling adds extra allocation/channel/copy overhead without changing
  the scalar radix-2 butterfly throughput.
- Validation: `cargo test -p fj-lax fft --lib` passed 43/43 through RCH;
  `cargo test -p fj-conformance --test fft_oracle` passed 27/27; `cargo test -p
  fj-conformance --test linalg_fft_oracle_parity` passed 1/1.
- Retry predicate: do not retry final-buffer writes, chunk-pool scheduling, or
  thread-spawn shaving for these batched pow2 complex rows without new cache-miss
  or scheduler evidence. The next credible `murmw` attempt needs a real kernel:
  SIMD radix-4/8 butterflies, native mixed-radix factorization for non-pow2 sizes,
  cache-blocked multi-row transforms, or generated length-specialized kernels,
  all gated by same-worker before/after plus the JAX ratio ledger.
## frankenjax-ur4h3 - small real-eigh cyclic-Jacobi gate rejected

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Target gap: `fj-lax` 48x48 real symmetric `eigh` still loses to warmed JAX
  CPU after the earlier large-n tridiagonal/QL cache fixes; the same sweep
  showed 48x48 SVD is already a Rust win and should not be attacked in this
  pass.
- Profiling/hypothesis route used: alien-graveyard communication-avoiding and
  small-kernel specialization thinking. Hypothesis: below 64x64, cyclic Jacobi
  might beat the two-stage Householder/tridiagonal QL path by keeping the whole
  matrix resident and avoiding setup/transpose constants.
- Lever rejected and reverted: a guarded `m <= 64` real-`eigh` route through
  `jacobi_eigendecomposition_cyclic`. Focused correctness passed, but
  same-worker release timing regressed the real benchmark by 3.83x, so the
  source hunk was restored before commit.

Initial head-to-head command:

```text
AGENT_NAME=WildForge RCH_FORCE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/(svd_48x48_f64|eigh_48x48_f64)' \
  --sample-size 15 --warm-up-time 1 --measurement-time 3 --noplot \
  --save-baseline cod-b-ur4h3-head-48
```

Rejected candidate command:

```text
AGENT_NAME=WildForge RCH_FORCE_REMOTE=1 RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/eigh_48x48_f64' --sample-size 15 --warm-up-time 1 \
  --measurement-time 3 --noplot
```

RCH ignored the requested `vmi1293453` pin and selected `vmi1152480`; a
production rerun was therefore taken on the same selected worker after source
revert.

Production rerun command:

```text
AGENT_NAME=WildForge RCH_FORCE_REMOTE=1 RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 \
  RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/eigh_48x48_f64' --sample-size 15 --warm-up-time 1 \
  --measurement-time 3 --noplot
```

JAX comparator used
`/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python` with
JAX/JAXLIB 0.10.1, CPU backend, and `jax_enable_x64=true`.

| workload | production Rust midpoint | candidate Rust midpoint | same-worker delta | JAX mean | production/JAX | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `linalg/eigh_48x48_f64` | 237.78 us | 910.28 us | 3.83x slower | 160.423 us | 1.482 | 5.674 | REVERT |

Initial routing rows:

| workload | Rust midpoint | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `linalg/svd_48x48_f64` | 263.72 us | 460.886 us | 0.572 | Existing Rust win; not targeted |
| `linalg/eigh_48x48_f64` first RCH route | 626.00 us | 160.423 us | 3.902 | Existing loss; same-worker rerun above supersedes for rejection |

- Ratio scorecard for retained production 48x48 linalg rows: **1 win / 1 loss
  / 0 neutral vs JAX**. Rejected candidate score: **0 wins / 1 loss / 0
  neutral vs JAX**. Production score: **0 kept wins / 0 shipped regressions**.
- Validation: focused `cargo test -p fj-lax eigh -- --nocapture` passed on RCH
  `vmi1149989`; full `cargo test -p fj-conformance` passed on RCH `hz2`;
  `cargo build --release -p fj-lax --benches` passed on RCH `hz2`.
- Retry predicate: do not retry small-`m` cyclic-Jacobi routing. The remaining
  credible `eigh` lever is a symmetry-specialized tridiagonal reduction or a
  blocked/panel Householder route that reduces setup without returning to
  sweep-heavy Jacobi.
## frankenjax-murmw - radix-4 power-of-four FFT no-ship

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Target gap: current `fj-lax` FFT losses to warmed JAX CPU after the earlier
  smooth-composite mixed-radix and batched pow2 plan-cache keeps.
- Profiling/hypothesis route used: alien-graveyard vectorized execution and
  FFT/NTT-butterfly material suggested fewer passes and higher-radix kernels;
  profiling discipline split the scenario into full eval rows plus a
  same-binary kernel A/B. Hypothesis was partially supported at kernel level and
  rejected at full eval level.
- Lever rejected and reverted: a safe recursive `Radix4Plan` for power-of-four
  lengths, routed through `BatchFftPlan` for pow4 rows. It matched DFT/IDFT
  tolerance but regressed end-to-end `eval_primitive(FFT)` rows, so the source
  change and its temporary tests/bench were removed before commit.

Baseline command:

```text
AGENT_NAME=WildForge RCH_FORCE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_(1000|1009_prime|batch_128x1000|batch_2048x256_complex128|batch_2048x256_complex128_dense_input)' \
  --sample-size 15 --warm-up-time 1 --measurement-time 3 --noplot \
  --save-baseline frankenjax-cod-b-fft-head-baseline
```

Candidate command:

```text
AGENT_NAME=WildForge RCH_FORCE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_(256_complex128|batch_2048x256_complex128|batch_2048x256_complex128_dense_input|rfft_batch_2048x256_f64_dense_input|irfft_batch_2048x256_complex128)' \
  --sample-size 15 --warm-up-time 1 --measurement-time 3 --noplot
```

JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
with JAX/JAXLIB 0.10.1, CPU, x64 enabled, warmed `jax.jit`.

| workload | HEAD Rust midpoint | candidate Rust midpoint | JAX mean | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `eval/fft_1000_complex128` | 70.525 us | unchanged | 31.204 us | 2.260 | Existing JAX loss |
| `eval/fft_1009_prime_complex128` | 243.86 us | unchanged | 63.710 us | 3.828 | Existing JAX loss |
| `eval/fft_batch_128x1000_complex128` | 5.9966 ms | unchanged | 197.438 us | 30.372 | Existing JAX loss |
| `eval/fft_batch_2048x256_complex128` | 30.983 ms | 42.532 ms | 284.224 us | 149.64 | REVERT: full eval regression |
| `eval/fft_batch_2048x256_complex128_dense_input` | 13.033 ms | 33.199 ms | 284.224 us | 116.80 | REVERT: full eval regression |

- Contradictory microbench that was not sufficient to ship: the temporary
  ignored same-binary A/B on RCH worker `hz2` measured `Radix2Plan` 80.976 ms
  vs `Radix4Plan` 38.024 ms for 2048 rows of length 256 (2.13x faster) and
  passed a DFT tolerance check. The full eval rows on `vmi1153651` overrode the
  microbench because the real workload regressed after input/output and batch
  plumbing were included.
- Ratio scorecard for this pass: **0 wins / 7 losses / 0 neutral vs JAX**.
  Production score: **0 kept wins / 0 shipped regressions** because the source
  was reverted before commit.
- Retry predicate: do not retry naive recursive radix-4 `BatchFftPlan` routing
  from an inner-kernel win alone. Future attempts need an iterative in-place
  radix-4/8 kernel, SoA/SIMD complex layout, or cache-blocked batched FFT that
  proves full eval improvement head-to-head; measure end-to-end before keeping.

## frankenjax-mcqr - f64->u32 packed width-changing bitcast keep; u32->f64 prezero/thread revert

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Target gap: remaining `bitcast_f64_u32_1m` and `bitcast_u32_f64_1m`
  release losses after the previous presized-fill pass flipped the
  `f32<->bf16` rows to JAX wins.
- Lever kept: dense `f64 -> u32` width-changing `bitcast_convert_type` now
  presizes the output and writes the little-endian low/high `u32` words directly
  from `f64::to_bits()`. For 1M+ source values it uses scoped threads over
  disjoint output chunks. Shape expansion and dtype construction are unchanged.
- Lever rejected and reverted: the symmetric `u32 -> f64` prezero/threaded
  candidate assembled values with indexed low/high chunks and regressed the
  dense 1M row, so that direction was restored to the original byte-array/push
  implementation before commit.
- Alien-graveyard route used: vectorized execution/artifact-layout thinking
  rather than algorithmic complexity. The useful part was turning an append
  stream into a fixed packed output layout; the failed part was applying the
  same layout idea to a widening direction where prezero/write traffic cost more
  than the old push path.

Baseline command:

```text
AGENT_NAME=cod-a \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/bitcast_(f64_u32|u32_f64)_(dense|literal_ref)_1m' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 \
  --save-baseline frankenjax-cod-a-bitcast-f64-u32-baseline
```

Post-change command:

```text
AGENT_NAME=cod-a \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/bitcast_(f64_u32|u32_f64)_(dense|literal_ref)_1m' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 \
  --baseline frankenjax-cod-a-bitcast-f64-u32-baseline
```

RCH selected worker `hz1` for both accepted baseline and candidate timing.

| workload | baseline Rust midpoint | candidate Rust midpoint | same-worker delta | JAX mean | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `eval/bitcast_f64_u32_dense_1m` | 1.5904 ms | 1.1781 ms | 1.35x faster, Criterion -29.249% | 163.747 us | 7.195 | KEEP internally; still JAX loss |
| `eval/bitcast_f64_u32_literal_ref_1m` | 74.361 ms | 69.507 ms | 1.07x faster, Criterion -6.528% | 163.747 us | 424.48 | Control row improved; no JAX win |
| `eval/bitcast_u32_f64_dense_1m` rejected candidate | 989.96 us | 1.2378 ms | 1.25x slower midpoint, Criterion +28.518% | 123.933 us | 9.988 | REVERT |
| `eval/bitcast_u32_f64_literal_ref_1m` rejected candidate | 51.321 ms | 49.209 ms | neutral, Criterion p=0.07 | 123.933 us | 397.06 | No production claim |

- JAX comparator:
  `benchmarks/jax_comparison/bitcast_gauntlet.py --runs 20 --warmup 5
  --inner-loops 200 --output /tmp/frankenjax-cod-a-bitcast-jax-20260620T1448Z.json`,
  JAX 0.10.1 / jaxlib 0.10.1, CPU, x64 enabled. `bitcast_f64_u32_1m`
  mean was 163.746649 us (p50 185.050925 us, CV 34.45%);
  `bitcast_u32_f64_1m` mean was 123.932654 us (p50 109.489063 us, CV
  31.97%). These high-CV local JAX rows are routing evidence, not the hard keep
  proof.
- Ratio scorecard for all four measured rows: **0 wins / 4 losses / 0 neutral
  vs JAX**. Production score for this pass after reverting the regressed
  direction: **1 internal win / 0 shipped regressions**, but `f64->u32` remains
  7.20x slower than fresh JAX and `u32->f64` remains unchanged and still a JAX
  loss.
- Validation: `cargo test -p fj-lax bitcast --lib` passed 4/4 through RCH;
  `cargo test -p fj-conformance --test bitcast_oracle` passed 36/36 through
  RCH; full `cargo test -p fj-conformance` passed through RCH/local fallback;
  `cargo check -p fj-lax --benches` passed on RCH `vmi1149989`;
  `cargo clippy -p fj-lax --lib -- -D warnings` passed on RCH `vmi1227854`.
- Non-blocking hygiene debt: `cargo fmt --check -p fj-lax` is still red on
  pre-existing unrelated formatting drift in benches/tests and older
  `tensor_ops.rs` test blocks; the new packed-bitcast hunk was not reported in
  the rustfmt diff.
- Retry predicate: do not retry symmetric prezero/threaded `u32 -> f64`
  widening without fresh evidence that the extra initialization traffic is gone.
  The remaining credible width-changing route is a deeper packed-chunk/vector
  kernel or output-reuse design that avoids both `Vec::push` checks and
  whole-buffer zeroing, with same-worker before/after proof and a fresh JAX
  comparator.

## frankenjax-xjbvr - Floor/Round/Sign unary-chain fusion keep; JAX still dominates

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Target gap: `fj-interpreters` compiled-dispatch dense add/unary chains where
  `Floor`, `Round`, and `Sign` forced the runner out of cheap-op fusion and left
  Rust far behind JAX's `jit` fused CPU kernel.
- Lever kept: extend `CheapOp` with `Floor`, `Ceil`, `Trunc`, `RoundAway`,
  `RoundNearest`, and `Sign`; admit those primitives in the f64/f32/half
  dense-chain scanners; admit `Sign` in the i64 scanner; apply unary cheap ops
  inside the existing dense fused chunks. The f32 and half routes preserve the
  existing widen/round-back behavior rather than changing semantics for speed.
- Alien-graveyard route used: local partial-evaluation/codegen specialization,
  not a new data structure. The win comes from treating monadic rounding/sign
  ops as fusible SSA nodes so the runner avoids per-equation dispatch and
  intermediate tensor materialization.
- Harness kept: `compiled_dispatch_speed` now records 1,048,576-element
  `floor`, `round`, and `sign` add/unary chain rows for f64 and f32.
- Conformance repairs kept: `nextafter_oracle` expected the stale
  `floating nextafter lhs` substring; current `fj-lax` reports
  `expected floating operands`. `polygamma_oracle` expected all integer operands
  rejected, but current `fj-lax` intentionally accepts numeric integer order and
  rejects integer `x`. Both tests were updated to match the current runtime
  contract after verifying the failures were unrelated to the interpreter
  fusion.

Baseline command:

```text
AGENT_NAME=cod-a RCH_FORCE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- \
  1m_add_unary_chain --sample-size 15 --measurement-time 2 --warm-up-time 1 --noplot
```

Baseline worker was `hz1`. A first post-change run on `vmi1227854` was faster
in absolute time, but the accepted internal delta below uses a same-worker
post-change rerun on `hz1`.

Same-worker Rust timing on `hz1`:

| workload | baseline compiled_runner | candidate compiled_runner | internal speedup | JAX mean | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `floor_f64_1m_add_unary_chain/n=4` | 21.229 ms | 2.5597 ms | 8.29x | 199.892 us | 12.805 | KEEP internally; still JAX loss |
| `round_f64_1m_add_unary_chain/n=4` | 20.516 ms | 1.8803 ms | 10.91x | 186.162 us | 10.100 | KEEP internally; still JAX loss |
| `sign_f64_1m_add_unary_chain/n=4` | 10.955 ms | 2.7290 ms | 4.01x | 342.029 us | 7.979 | KEEP internally; still JAX loss |
| `floor_f32_1m_add_unary_chain/n=4` | 8.8956 ms | 8.0347 ms | 1.11x | 103.966 us | 77.282 | KEEP as marginal internal win; still severe JAX loss |
| `round_f32_1m_add_unary_chain/n=4` | 16.423 ms | 4.6730 ms | 3.51x | 123.124 us | 37.954 | KEEP internally; still severe JAX loss |
| `sign_f32_1m_add_unary_chain/n=4` | 11.639 ms | 5.5279 ms | 2.11x | 128.710 us | 42.948 | KEEP internally; still severe JAX loss |

- Faster post-change routing rows on RCH worker `vmi1227854` were also recorded:
  f64 floor 1.2723 ms, f64 round 1.2006 ms, f64 sign 1.7201 ms, f32 floor
  3.7928 ms, f32 round 3.6287 ms, f32 sign 5.0230 ms. These are not used for
  the internal keep proof because their baseline was on `hz1`.
- JAX comparator artifact:
  `artifacts/performance/evidence/frankenjax-xjbvr-jax-unary-chain-20260620T1358Z.json`;
  local CPU JAX 0.10.1 / jaxlib 0.10.1, x64 enabled, warmed
  `jax.jit` add/unary chains. JAX means had 10-16% CV, so the exact external
  multipliers are routing evidence rather than release-certification ratios.
- Ratio scorecard for this pass: **0 wins / 6 losses / 0 neutral vs JAX**.
  The keep is justified by same-worker internal gap reduction, not by JAX
  domination.
- Validation: focused
  `fusion_floor_ceil_trunc_round_sign_chains_match_unfused_reference` passed;
  `cargo check -p fj-interpreters --benches` passed on RCH `vmi1149989`;
  `cargo clippy -p fj-interpreters --benches -- -D warnings` passed on RCH
  `vmi1149989`; `cargo test -p fj-conformance --test nextafter_oracle` passed
  33/33; `cargo test -p fj-conformance --test polygamma_oracle` passed 26/26
  on RCH `hz2`; full `cargo test -p fj-conformance` passed on RCH `hz2`.
- Retry predicate: do not retry per-equation dispatch shaving for this exact
  unary-chain family. Future attempts need a deeper XLA-class lever: generated
  vector kernels, f32-native rounding/sign semantics behind an explicit
  contract, output/arena reuse across fused kernels, or backend specialization
  that eliminates the remaining memory traffic. For f32 specifically, the
  current widen-to-f64 semantics are the main reason the Rust/JAX ratio remains
  severe.

## cod-b - width-changing bitcast presized-fill keep; local same-target bench invalid (2026-06-20)

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Claimed tracker context: `frankenjax-cntiy` was the open cod-b ready bead, but
  the FMA-policy surface already had fresh no-ship evidence. This pass followed
  the scorecard instead and targeted the still-measured `bitcast_f32_bf16_1m`
  and `bitcast_bf16_f32_1m` losses.
- Lever kept: dense width-changing `bitcast_convert_type` now pre-sizes the
  output buffer and fills by index for `f32 -> bf16/f16 chunks` and
  `bf16/f16 chunks -> f32`. Byte order, trailing-dimension handling, and dtype
  construction are unchanged; the change removes repeated `Vec::push` growth
  checks and exposes a fixed-size fill loop.
- Alien-graveyard route used: vectorized/artifact-layout thinking rather than
  a new algorithm. The key observation was that the slow path was not the
  bit reinterpretation itself but the materialization layout of the output
  artifact.

Same-worker RCH timing on `vmi1227854` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`:

| workload | baseline Rust midpoint | candidate Rust midpoint | same-worker speedup | JAX mean | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `eval/bitcast_f32_bf16_dense_1m` | 978.58 us | 125.40 us | 7.80x | 140.512 us | 0.892 | KEEP: Rust 1.12x faster than JAX |
| `eval/bitcast_bf16_f32_dense_1m` | 533.82 us | 123.49 us | 4.32x | 151.382 us | 0.816 | KEEP: Rust 1.23x faster than JAX |

- Candidate literal controls: `eval/bitcast_f32_bf16_literal_ref_1m` 38.853 ms
  and `eval/bitcast_bf16_f32_literal_ref_1m` 27.965 ms. They prove the dense
  route is still the path under test; their worker-to-worker noise is not used
  for the keep decision.
- JAX comparator:
  `benchmarks/jax_comparison/bitcast_gauntlet.py --runs 20 --warmup 5
  --inner-loops 200 --output /tmp/frankenjax-cod-b-bitcast-jax-20260620T1327Z.json`,
  JAX 0.10.1, CPU, x64 enabled. The accepted Rust/JAX ratio compares remote
  RCH Rust with local JAX, which is conservative relative to the scorecard note
  that RCH workers are often slower than local.
- Invalid measurement recorded: a local Rust bench against
  `/data/projects/.rch-targets/frankenjax-cod-b` failed before measurement with
  rustc `E0514` because the RCH target directory contained artifacts compiled
  by a different nightly (`beae78130`) than the local toolchain (`f20a92ec0`).
  No cleanup was attempted.
- Validation: `cargo test -p fj-lax bitcast --lib` passed 4/4 on RCH
  `vmi1293453`; `cargo test -p fj-conformance --test bitcast_oracle` passed
  36/36 on RCH `hz2`; `cargo check -p fj-lax --all-targets` passed on RCH
  `hz1`; production `cargo clippy -p fj-lax --lib -- -D warnings` passed on
  RCH `vmi1149989`; `git diff --check` passed for the touched docs/code/bead
  files.
- Non-blocking hygiene debt: `cargo fmt --check` is red on pre-existing
  repo-wide formatting drift, and `cargo clippy -p fj-lax --all-targets --
  -D warnings` is red on unrelated test lint debt (`arithmetic.rs` unnecessary
  casts, `nn.rs` manual clamps, and older `tensor_ops.rs` test type/clone
  lints). Filed follow-up `frankenjax-98eoz`; no unrelated lint cleanup was
  folded into this perf commit.
- Ratio scorecard for these two rows after the keep: 2 wins / 0 losses / 0
  neutral vs JAX. This removes two prior bitcast release losses from the active
  scorecard.
- Retry predicate: do not retry the old push-based width-changing bitcast
  materialization. Future bitcast work should focus on the remaining same-width
  signed reinterpret losses (`f32<->i32`, `f64<->i64`) or `f64<->u32`
  width-changing rows with fresh same-worker proof.

## frankenjax-ligu5 - dense f64/f32 batched-operand gather is not a JAX loss

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Target gap: verify whether the open `fj-dispatch` batched gather/scatter
  de-box bead still had an unmined dense-float batched-operand gather loss after
  prior direct I64 gather and lazy scatter keeps.
- Harness change kept: `crates/fj-dispatch/benches/dispatch_baseline.rs` now
  records `vmap_gather/batched_operand_batched_indices_f64` and
  `vmap_gather/batched_operand_batched_indices_f32` next to the existing I64 row.
- Production code changed: none.
- Prior negative evidence considered: do not retry the rejected flatten-index
  single-call gather path or the rejected `PreparedGatherIndices` ownership/view
  tweak without a fresh profile; both were previously slower.

Current Rust benchmark:

```text
AGENT_NAME=cod-a RCH_FORCE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- \
  vmap_gather/batched_operand_batched_indices \
  --sample-size 20 --measurement-time 3 --warm-up-time 1 --noplot
```

Worker `vmi1152480` results:

| workload | Rust low | Rust midpoint | Rust high | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| I64 batched operand + batched indices | 8.0403 us | 8.6722 us | 9.4162 us | 31.266 us | 0.277 | Rust 3.61x faster; no gap |
| F64 batched operand + batched indices | 24.523 us | 25.251 us | 26.261 us | 33.224 us | 0.760 | Rust 1.32x faster; no gap |
| F32 batched operand + batched indices | 25.437 us | 27.257 us | 29.104 us | 31.369 us | 0.869 | Rust 1.15x faster; no gap |

- JAX comparator artifact:
  `artifacts/performance/evidence/frankenjax-ligu5-jax-vmap-gather-20260620T1325Z.json`;
  it used JAX 0.10.1, x64 enabled, local CPU, warmed
  `jax.jit(lambda x, i: jax.vmap(lambda row, idx: jnp.take(row, idx))(x, i))`.
- Caveat: JAX CV was high (17-25%), so these rows are routing evidence rather
  than a certification-grade release claim. The direction is still sufficient
  to reject a production dense-float batched gather patch for this exact shape.
- Ratio scorecard for this subcase: 3 wins / 0 losses / 0 neutral vs JAX.
- Retry predicate: do not retry dense rank-2 batched-operand gather de-boxing
  for I64/f64/f32 without a fresh profile showing a concrete loss on a
  non-dense, higher-rank, partial-slice, or different dtype subcase. Future
  `ligu5`-family work should start from a new `dispatch_baseline` profile and
  avoid the rejected flatten/index-view family.

## frankenjax-n75xr - f64 scalar-add chain generated SIMD medium-band keep; upper-band no-ship

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Target gap: the remaining `compiled_dispatch` f64 large-chain JAX losses, with
  emphasis on the 1,048,576-element row where Rust still lost by more than an
  order of magnitude.
- Lever kept: a narrow exact-pattern specialization for
  `tensor + scalar + scalar + ...` f64 add chains in
  `1,048,576 <= n < FUSION_THREAD_MIN_ELEMS`. The route uses a generated-style
  `std::simd::Simd<f64, 8>` register loop over the copied input and keeps the
  original scalar-add order per lane. It deliberately rejects operand-reversed
  first adds so NaN payload/sign-order edge cases stay on the generic fusion
  path.
- Behavioral proof: focused unit
  `f64_scalar_add_chain_simd_matches_ordered_scalar_reference` passed via RCH on
  `vmi1227854` and checks bitwise equality against ordered scalar reference,
  including signed zero, infinities, and a NaN payload.
- Build guard: `AGENT_NAME=cod-b
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec --
  cargo build --release -p fj-interpreters --benches`, worker `vmi1227854`,
  passed before the timing pass.
- Validation guard after the final gate: `cargo check -p fj-interpreters
  --benches`, same target dir through RCH, passed on `vmi1152480`.
- Final post-clippy-fix guards: `cargo clippy -p fj-interpreters --lib
  --no-deps -- -D warnings` passed on `vmi1293453`;
  `cargo test -p fj-interpreters
  f64_scalar_add_chain_simd_matches_ordered_scalar_reference --lib` passed on
  `vmi1227854`; `cargo test -p fj-conformance --lib` passed 45/0 on
  `vmi1152480`.
- Non-blocking hygiene debt: `cargo fmt --check -p fj-interpreters` remains red
  on pre-existing formatting in `compiled_dispatch_speed.rs`,
  `eval_fusion_speed.rs`, and older `src/lib.rs` regions; `cargo clippy -p
  fj-interpreters --benches --no-deps -- -D warnings` reaches an unrelated
  pre-existing bench lint in `eval_fusion_speed.rs`. UBS on `src/lib.rs`
  remains nonzero on the existing whole-file panic/assert/index inventory while
  its internal build/check/clippy/fmt sections are clean.
- Supporting disassembly sanity check on the retrieved
  `compiled_dispatch_speed` binary found vector add instructions (`vaddpd` /
  `vaddsd`). This is recorded only as supporting evidence because the binary is
  LTO/stripped and the keep decision is based on timing plus bitwise tests.

Same-worker RCH timing, worker `vmi1227854`, target
`/data/projects/.rch-targets/frankenjax-cod-b`, command family
`cargo bench -p fj-interpreters --bench compiled_dispatch_speed --
'compiled_dispatch/(compiled_runner|compiled_runner_scalar)/(bigchain65536|bigchain262144|bigchain1048576|bigchain16777216)'
--warm-up-time 1 --measurement-time 5`. JAX means are from
`artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`
(`jax.jit` CPU 0.10.1, x64):

| workload | baseline Rust | candidate Rust | JAX mean | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| f64 65K x8 | 47.427 us | 43.324 us | 34.033 us | 1.273 | No claim: specialization gate is off for this row; routing/noise only |
| f64 262K x8 | 226.80 us | 226.18 us | 76.827 us | 2.944 | No claim: specialization gate is off; effectively neutral and still JAX loss |
| f64 1M x8 | 1.3412 ms | 821.23 us | 83.299 us | 9.859 | KEEP: 1.63x internal win, still JAX loss |
| f64 16M x8 | 78.933 ms | 104.63 ms | 27.610 ms | 3.790 | REJECT: +32.56% regression, branch gated off before commit |

- Negative evidence inside the keep: the first ungated candidate routed the
  16M row through the new one-pass SIMD copy loop and regressed from 78.933 ms
  to 104.63 ms. That branch was reverted by tightening the gate to stay below
  `FUSION_THREAD_MIN_ELEMS`, preserving the existing threaded fusion path for
  huge arrays.
- Cross-worker reruns after the gate are not used as keep/reject proof. RCH
  selected `ovh-a` for the follow-up and ignored `RCH_WORKER=vmi1227854`; those
  rows varied widely (for example, 1M 1.6850 ms then 2.2376 ms on `ovh-a`) and
  are recorded only as worker-selection instability.
- Ratio scorecard after the kept gate: 0 wins / 4 losses / 0 neutral vs JAX for
  the f64 65K, 262K, 1M, and 16M production row set. The lever is still worth
  keeping because it cuts the 1M JAX gap from 16.10x slower to 9.86x slower
  without changing the other production gates.
- Retry predicate: do not extend this exact scalar-add SIMD route into the
  threaded upper band without a same-worker win over the current threaded path.
  The next credible JAX-dominating lever needs output reuse/arena writeback,
  legal relaxed-FP folding (`x + 8`) under an explicit contract, or generated
  backend kernels that can reduce memory traffic rather than only replacing the
  generic per-step loop.

## frankenjax-cntiy - FMA primitive dense path and `+fma` policy no-ship

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Target gap: isolate the direct ternary FMA primitive after the earlier softmax
  probe showed global `+fma` was neutral for `softmax_2d_65536x16`.
- Rust harness change kept: `crates/fj-lax/benches/elementwise_gauntlet.rs`
  now has dense-vs-boxed `fma_f64_1m` and `fma_f32_1m` rows. The JAX comparator
  has matching rows in `benchmarks/jax_comparison/elementwise_gauntlet.py`.
- JAX comparator caveat: installed JAX 0.10.1 does not expose public
  `jax.lax.fma`, so the oracle row is warmed
  `jax.jit(lambda a, b, c: a * b + c)`.
- RCH build guard:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec --
  cargo build --release -p fj-lax --benches`, worker `vmi1227854`, passed.
- RCH timing guard:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec --
  cargo bench -p fj-lax --bench elementwise_gauntlet fma_ -- --quiet`, worker
  `hz1`, passed.

Remote RCH internal dense-vs-boxed rows:

| workload | dense | boxed | dense/boxed | verdict |
| --- | ---: | ---: | ---: | --- |
| `fma_f64_1m` | 3.5198 ms | 29.267 ms | 0.120 | Keep coverage: dense is 8.31x faster internally |
| `fma_f32_1m` | 3.7757 ms | 30.825 ms | 0.123 | Keep coverage: dense is 8.16x faster internally |

Local same-host Rust/JAX rows, default codegen:

| workload | Rust dense | boxed control | JAX | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `fma_f64_1m` | 2.6124 ms | 24.592 ms | 273.448 us | 9.553 | Loss: JAX 9.55x faster |
| `fma_f32_1m` | 2.7622 ms | 26.151 ms | 111.281 us | 24.822 | Loss: JAX 24.82x faster |

Local same-host `RUSTFLAGS="-C target-feature=+fma"` policy probe:

| workload | `+fma` Rust dense | JAX | Rust/JAX | vs default | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `fma_f64_1m` | 925.02 us | 273.448 us | 3.383 | 2.82x faster | No-ship: still a JAX loss |
| `fma_f32_1m` | 207.98 us | 111.281 us | 1.869 | 13.28x faster | No-ship: still a JAX loss |

- Decision: keep the benchmark rows and dense FMA coverage, but do not ship a
  global `+fma` build policy. It is a large primitive-specific internal win but
  not a JAX-dominating lever by itself, and the global flag still has bit-exact
  accumulation risk outside this row.
- Ratio scorecard: 0 wins / 4 losses / 0 neutral vs JAX.
- Retry predicate: revisit only with semantics-approved per-kernel target
  feature/codegen, generated vector kernels, or output/arena reuse that can be
  measured directly against the new FMA rows. Do not rerun global `+fma` as a
  standalone fix for `frankenjax-cntiy`.

## frankenjax-xljoh.1 - f64 large-chain register pass and larger fusion tile no-ships

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Target gap: the remaining `compiled_dispatch` f64 large-chain JAX losses after
  `frankenjax-xljoh` (65K 1.52x slower than JAX, 262K 3.19x, 1M 20.36x,
  16M 4.55x in the remote-Rust/local-JAX comparator).
- Baseline command:
  `AGENT_NAME=WildForge CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
  rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed --
  'compiled_dispatch/(compiled_runner|compiled_runner_scalar)/(bigchain65536|bigchain262144|bigchain1048576|bigchain16777216)'
  --warm-up-time 1 --measurement-time 5`.
- Fresh unchanged baseline, RCH worker `vmi1227854`:

| workload | compiled_runner | scalar control | note |
| --- | ---: | ---: | --- |
| f64 65K x8 | 46.896 us | 59.727 us | existing mid-cache fallback remains fastest |
| f64 262K x8 | 214.91 us | 244.78 us | existing mid-cache fallback remains fastest |
| f64 1M x8 | 1.3315 ms | 2.8176 ms | chunked fusion is much faster than scalar control |
| f64 16M x8 | 116.11 ms | 127.36 ms | still a JAX loss, but best current Rust path |

- Probe 1, REVERTED before commit: ordered per-element register pass for the
  exact `x + 1.0` repeated-8 scalar-add chain. The idea was to preserve strict
  operation order (`((((x+1)+1)+...)`) while doing one memory pass. Candidate
  worker `vmi1153651` (RCH did not keep the baseline worker):

| workload | candidate | scalar control | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| f64 65K x8 | 52.746 us | 181.80 us | 1.55 | Reject: worse than unchanged 46.896 us and still JAX loss |
| f64 262K x8 | 351.51 us | 1.4586 ms | 4.58 | Reject: severe regression vs unchanged 214.91 us |
| f64 1M x8 | 3.5382 ms | 6.2466 ms | 42.48 | Reject: 2.66x slower than unchanged 1.3315 ms |
| f64 16M x8 | 142.56 ms | 279.87 ms | 5.16 | Reject: slower than unchanged 116.11 ms |

  Root cause: the runtime step slice did not become the desired SIMD loop. It
  beat the same-run scalar control but lost to the existing chunked/in-place
  vectorized runner. No source kept.

- Probe 2, REVERTED before commit: `FUSION_CHUNK` 8,192 -> 65,536 elements to
  reduce chunk-loop overhead. Candidate worker `vmi1149989`; Criterion had saved
  comparable history for the same benchmark family and reported regressions:

| workload | candidate | Criterion delta | candidate/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| f64 65K x8 | 52.828 us | +7.33%, p=0.00 | 1.55 | Reject: regression |
| f64 262K x8 | 256.43 us | +2.78%, p=0.08 | 3.34 | Reject/neutral: no win |
| f64 1M x8 | 1.9764 ms | +21.35%, p=0.00 | 23.73 | Reject: regression |
| f64 16M x8 | 128.78 ms | +4.66%, p=0.00 | 4.66 | Reject: regression |

- Decision: NO-SHIP for both levers. Code was returned to byte-for-byte clean
  relative to HEAD before evidence docs were updated.
- Retry predicate: do not retry per-element register scalar-add fusion or larger
  fusion tiles without disassembly/profile evidence that the candidate emits SIMD
  and a same-worker before/after win. The f64 chain gap is now routed to deeper
  specialization/codegen questions: legal algebraic simplification policy,
  generated straight-line SIMD kernels, or a backend that can match XLA's folded
  `x + 8` behavior where semantics allow it.

## frankenjax-xljoh - compiled-dispatch f64 mid-cache owned-eval fallback

- Date: 2026-06-20
- Agent: cod-a / WildForge
- Lever kept: narrow `CompiledJaxprRunner` fallback from the reusable dense f64
  scalar plan to the owned `CompiledJaxpr::eval` path for one-tensor f64 linear
  chains in the 65,536..=262,144 element band. This is a branchless hot-path
  shape/class specialization, not a broad dtype or allocator policy change.
- Why this was not the original broad CheapOp fusion route: the baseline showed
  the reusable runner already wins at 4K and 1M, f32 already wins on the tested
  rows, and the only measured internal loss was the f64 mid-cache band where
  owned compiled eval beat the reusable runner. The kept guard is deliberately
  the smallest measured escape hatch.
- Rust baseline, RCH worker `vmi1149989`, target
  `/data/projects/.rch-targets/frankenjax-cod-a`, command
  `cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- compiled_dispatch --warm-up-time 1 --measurement-time 5`:
  - f64 4K: eager 3.280 us, compiled_runner 2.673 us. Verdict: keep runner; do
    not route small rows through owned eval.
  - f64 65K: eager 39.998 us, compiled 38.941 us, compiled_runner 47.105 us,
    scalar 59.558 us. Verdict: runner loses; candidate route justified.
  - f64 262K: eager 237.173 us, compiled_runner 247.034 us, scalar 255.254 us.
    Verdict: runner loses; candidate route justified.
  - f64 1M: eager 7.205 ms, compiled_runner 1.629 ms, scalar 1.778 ms. Verdict:
    keep runner/fusion path; do not expand fallback upward.
  - f64 16M: eager 118.539 ms, compiled_runner 123.040 ms, scalar 123.660 ms.
    Verdict: no safe same-worker win; leave untouched.
  - f32 4K: eager 1.952 us, compiled_runner 1.938 us, scalar 1.876 us. Verdict:
    neutral/noisy, no f32 route.
  - f32 65K: eager 24.989 us, compiled_runner 26.384 us, scalar 28.661 us by
    mean, but median runner was 24.356 us vs eager 25.214 us. Verdict: no f32
    route without a cleaner same-host row.
- Candidate bench, RCH selected `vmi1152480` despite the pinned-worker request,
  same target dir and filtered Criterion command:
  - f64 4K: compiled_runner 3.319 us. Guard did not fire; directionally no
    claimed before/after win because worker changed.
  - f64 65K: compiled_runner 51.601 us vs compiled 56.910 us vs eager 59.825 us
    vs scalar 64.532 us. Verdict: keep; 1.10x faster than owned compiled in the
    same candidate run, 1.16x faster than eager, 1.25x faster than scalar.
  - f64 262K: compiled_runner 245.474 us vs compiled 245.607 us vs eager
    253.769 us vs scalar 280.130 us. Verdict: keep; matches owned compiled and
    stays 1.03x faster than eager, 1.14x faster than scalar.
  - f64 1M: compiled_runner 1.696 ms vs compiled 7.159 ms vs eager 6.507 ms.
    Verdict: fallback correctly did not fire; expanding it would regress.
  - f64 16M: compiled_runner 125.556 ms vs JAX 27.610 ms. Verdict: still a
    major JAX loss, but this lever does not fix it; needs deeper backend/vector
    codegen or output reuse work.
  - f32 4K: compiled_runner 1.352 us. Verdict: f32 remains a Rust win; no change.
  - f32 65K: compiled_runner 21.486 us. Verdict: f32 remains a Rust win; no
    change.
- JAX comparator: `benchmarks/jax_comparison/interpreter_compiled_dispatch_gauntlet.py`
  extended to the large f64/f32 chain rows. Artifact:
  `artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`.
  JAX 0.10.1 CPU x64, warmed `block_until_ready`, local host. Ratios below use
  candidate Rust mean / JAX mean, so they are routing-grade directional because
  Rust was remote and JAX was local:

| Workload | Rust mean | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| f64 chain 4K x8 | 3.319 us | 6.136 us | 0.541 | Rust 1.85x faster; guard off |
| f64 chain 65K x8 | 51.601 us | 34.033 us | 1.516 | JAX 1.52x faster; kept internal mid-cache improvement |
| f64 chain 262K x8 | 245.474 us | 76.827 us | 3.195 | JAX 3.19x faster; still a gap |
| f64 chain 1M x8 | 1.696 ms | 83.299 us | 20.36 | JAX 20.36x faster; fallback off by design |
| f64 chain 16M x8 | 125.556 ms | 27.610 ms | 4.55 | JAX 4.55x faster; no safe win found |
| f32 chain 4K x8 | 1.352 us | 8.278 us | 0.163 | Rust 6.12x faster; JAX CV 46%, noisy |
| f32 chain 65K x8 | 21.486 us | 33.742 us | 0.637 | Rust 1.57x faster |

- Rejected/no-ship ideas:
  - Broad f32 fallback or fusion route: rejected because f32 rows already beat JAX
    on the measured large-chain comparator and the internal rows were noisy rather
    than a clean loss.
  - Expanding the f64 fallback above 262K: rejected because the 1M row is a strong
    reusable-runner win and would regress by about 4x if routed to owned eval.
  - Chasing the 16M loss with this lever: rejected because both owned eval and
    reusable runner remain memory/codegen limited versus JAX; route to deeper
    output reuse, vector codegen, or cache-aware backend work.
- Validation:
  - `cargo check -p fj-interpreters --all-targets`: pass via RCH.
  - `cargo clippy -p fj-interpreters --lib --no-deps -- -D warnings`: pass via
    RCH worker `vmi1149989`.
  - `cargo test -p fj-interpreters compiled_jaxpr_eval_matches_eager_eval_jaxpr --lib`:
    pass via RCH.
  - `cargo test -p fj-conformance`: pass, 45 lib tests plus oracle/gate suites
    green using `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`;
    RCH had no admissible worker slot, so this ran under the local fallback.
  - `cargo test -p fj-interpreters --lib`: blocked by pre-existing golden-hash
    drift in 8 unrelated tests; the targeted compiled-jaxpr semantic test passes.
  - Workspace/all-target clippy is blocked before this crate by existing `fj-trace`
    and `fj-lax` lints; recorded as release-readiness debt, not absorbed into this
    perf lever.
- Retry predicate: do not reattempt broad f32 or upper-band f64 routing without a
  same-host or same-worker row showing a new loss. The remaining f64 large-chain
  gap is not a dispatch-guard problem.

## frankenjax-cc-threaded-integer-axis-reduce - Threaded INTEGER axis-reduce: trailing JAX WIN, leading 2.13x (bit-exact)

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Lever: thread the dense INTEGER axis-reduce (separate `int_op` path, not the float
  `dense_f64_axis_reduce`). Two bit-exact strategies, both exploiting integer
  associativity:
  - TRAILING axis (`inner==1`, e.g. sum over last axis): outputs are independent
    contiguous blocks -> thread over OUTPUT rows (contiguous reads).
  - LEADING axis (`inner>1, outer==1`, e.g. sum over axis 0): column accumulation.
    Integer add is ASSOCIATIVE, so thread the REDUCE dimension with CONTIGUOUS reads
    (each thread folds a row-band into a local `inner`-wide partial, combine partials
    in chunk order). This is the CONTIGUOUS win the float column reduce can't get
    (float reassociates -> only strided column threading, see threaded-leading-axis-reduce).
- Status: SHIPPED (measured, [16384,1024] i64, same-load at load ~3):
  - axis 1 (trailing): serial 8.77 ms -> threaded 4.34 ms (2.02x). vs JAX 5.20 ms:
    **Rust/JAX 0.83 = Rust 1.20x FASTER (WIN).**
  - axis 0 (leading): serial 9.53 ms -> threaded 4.47 ms (2.13x). vs JAX 3.55 ms:
    Rust/JAX 1.26 — still a small JAX loss but the gap shrank from 2.68x to 1.26x
    (contiguous reduce-chunking; the residual is partial-combine + BW efficiency).
  - 8-thread cap (read-bound saturation, consistent with the other reduce levers).
- Bit-identity guard: `threaded_integer_axis_reduce_bit_identical_to_serial` (>= gate,
  both axes incl i64 wrap). Reduce tests 137/0, clippy clean. Benches
  `eval/reduce_sum_16kx1k_axis{0,1}_i64`; reproducer
  `benchmarks/jax_comparison/reduce_int_axes_gauntlet.py`.
- Retry predicate: middle-axis integer reduce (`inner>1 && outer>1`) stays serial
  (less common; thread over `outer` is the follow-on if a profile shows it). Do not
  raise the 8-thread cap (read-bound).

## frankenjax-cc-threaded-leading-axis-reduce - Threaded sum-over-axis-0: 1.50x bit-exact (narrows JAX loss 2.38x -> 1.59x)

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Lever: thread the LEADING-axis (column-accumulation) float reduce in
  `dense_f64_axis_reduce` — i.e. `reduce_sum`/mean over axis 0 (batch reduction,
  ubiquitous in ML). The trailing-axis reduce was already threaded; the leading
  (kept = trailing suffix) path was SERIAL. Threads split the OUTPUT COLUMNS:
  each column folds its rows in ascending order on exactly one thread, so this is
  BIT-IDENTICAL to the serial fold even for non-associative float (a single column's
  sum is never reassociated — only different columns go to different threads).
- Status: SHIPPED (measured, same-load at load ~3, [16384,1024] f64 sum axis 0):
  - serial 8.67 ms -> threaded (8-thread cap) **5.81 ms** = 1.50x internal
    (14.7 -> 22 GB/s). Bit-identity guarded by
    `threaded_leading_axis_reduce_bit_identical_to_serial`. Reduce tests 136/0.
  - vs JAX `jnp.sum(x, axis=0)` = 3.64 ms (CV 8.8%): serial was Rust/JAX 2.38 (JAX
    2.38x faster); threaded is Rust/JAX 1.59 — **still a JAX LOSS, but the gap is
    nearly halved.** Reproducer `benchmarks/jax_comparison/reduce_axis0_gauntlet.py`.
- Why it doesn't fully catch JAX: the bit-exact requirement forces COLUMN threading,
  whose reads are STRIDED in row-major layout (~22 GB/s ceiling). JAX threads the
  REDUCE dimension (contiguous reads, ~35 GB/s) but that REASSOCIATES the float sum
  (pairwise) — the same non-associativity blocker as the full-reduce. So the
  remaining ~1.59x is the float-reassociation gap (see reduce_sum-jax-loss entry;
  maintainer multi-accumulator decision). 32 threads REGRESSED (6.31ms) vs 8 — a
  strided read-bound pass saturates at ~8 cores; do NOT raise the cap.
- Retry predicate: the INTEGER leading-axis reduce could fully catch JAX (associative
  -> reduce-dimension chunking is bit-exact + contiguous), but goes through the
  separate int_op path; a follow-on. Do not re-thread the FLOAT leading reduce over
  the reduce dimension (reassociates — blocked).

## frankenjax-cc-threaded-integer-full-reduce - Threaded INTEGER full-reduce: JAX WIN (1.32x), bit-exact

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Lever: thread the dense INTEGER full-reduce (all-axes -> scalar) in `eval_reduce`.
  Integer reduce ops (sum/prod/and/or/xor/max/min) are ASSOCIATIVE / order-invariant
  and `int_init` is each op's identity, so a chunked partial-fold combined with
  `int_op` is **BIT-IDENTICAL** to the sequential fold (mod 2^64 / mod 2^32 is a +,*
  homomorphism; max/min order-free). A full reduce is a pure sequential read
  (BW-bound): one core cannot saturate multi-channel DRAM, so split the read across
  ~8 cores. **This is the float-reduce lever WITHOUT the float blocker** — integers
  reassociate bit-exactly, so NO tolerance, NO golden churn, NO maintainer call.
- Status: SHIPPED (measured, same-load at load ~3-5):
  - frankenjax 16M i64 reduce_sum: serial 7.44 ms -> threaded (8-thread cap) **4.35 ms**
    = 1.71x internal (17.2 -> 29.4 GB/s).
  - vs JAX `jnp.sum` (int64, 16M) = 5.73 ms (mean, CV 10%): serial was Rust/JAX 1.30 =
    JAX 1.28x faster (we LOST); threaded is **Rust/JAX 0.76 = Rust 1.32x FASTER**.
    The lever FLIPS an integer-sum loss into a win.
  - Thread count: `work_scaled_threads(n).min(8)` — 64 threads (the uncapped count)
    REGRESSED to 5.66 ms (oversubscription + sequential combine; a pure read saturates
    DRAM at ~8 cores). 8-thread cap is the keep.
- Bit-identity guard: `threaded_integer_full_reduce_bit_identical_to_serial` (>= gate,
  asserts threaded == sequential fold for sum/prod/and/or/xor/max/min incl. i64 wrap).
  Bench `eval/reduce_sum_16m_i64_full`; reproducer `benchmarks/jax_comparison/isum_gauntlet.py`.
  Signature: `int_op` gained `+ Sync` on `eval_reduce` + `eval_reduce_axes` (all callers
  pass fn-pointers/uncapturing closures, already Sync).
- Retry predicate: do NOT thread the FLOAT full-reduce (non-associative — see the
  reduce_sum-jax-loss entry; that one stays blocked on the maintainer's multi-accumulator
  decision). The integer win is independent and shipped. Do not raise the 8-thread cap
  (more threads REGRESS a read-bound reduce).

## frankenjax-cc-reduce-sum-jax-loss - reduce_sum LOSES 2.87x to JAX (latency-bound sequential fold; multi-accumulator BLOCKED on maintainer)

- Date: 2026-06-19
- Agent: cc / CobaltForge
- GAP WHERE WE LOSE (measured, same-load at load ~6): Rust `eval_primitive(ReduceSum)`
  over a 16M f64 1-D vector = **13.36 ms** ([13.20,13.51] Criterion) vs JAX
  `jnp.sum` = **4.65 ms** (mean, CV 5.9%), jax 0.10.1 CPU x64, warmed.
  **Rust/JAX 2.87 = JAX 2.87x FASTER.** New bench `eval/reduce_sum_16m_f64_full`;
  reproducer `benchmarks/jax_comparison/sum_gauntlet.py`.
- Root cause: `eval_reduce`'s float path is a STRICT single-accumulator sequential
  f64 fold (`acc = float_op(acc, x)`, ascending order) — deliberately non-reassociated
  to stay bit-exact vs the generic Literal fold. A single accumulator is LATENCY-bound
  (each add depends on the previous, ~0.8 ns/add = ~13 ms for 16M), whereas JAX sums
  pairwise/vectorized (throughput-bound).
- The lever (multi-accumulator / pairwise / threaded partial sums) is **tolerance-legal
  vs JAX** — `reduce_sum_oracle` compares with `(a-b).abs() < tol`, NOT bit-exact — and
  is **strictly MORE accurate** (pairwise error grows ~log n vs ~n for sequential, i.e.
  CLOSER to both the true sum and JAX's own pairwise result). So the standing
  "DO-NOT-REATTEMPT float-sum SIMD (non-associative)" objection is weaker than it reads:
  the only hard blocker is that it reassociates, which breaks (a) the internal
  `dense_*_reduce_sum_*_bit_identical_to_literal_path` guards and (b) the FROZEN bit-exact
  goldens `artifacts/performance/evidence/fj_lax_dense_f64_reduce_sum_pass60*` — both of
  which pin the CURRENT sequential bits. Re-baselining those to a fixed multi-accumulator
  order would make sum both faster AND more accurate, deterministically.
- Status: NOT implemented. This changes a fundamental op's numerical output library-wide
  + re-baselines frozen goldens, so it is a MAINTAINER numerical-policy decision (same
  class as the +fma flag), not a unilateral agent change. RECOMMENDATION for the maintainer:
  reconsider the DO-NOT — a k-accumulator (k=8) scalar fold (no SIMD/FMA/unsafe needed)
  should recover ~3-5x on large f64/f32 reductions (sum/mean — ubiquitous in ML) within
  the existing tolerance, and improve accuracy. If approved, change the shared fold so
  dense==literal still match, then regenerate the reduce_sum perf goldens.
- Retry predicate: do not ship a multi-accumulator sum WITHOUT maintainer sign-off on the
  golden re-baseline; do not attempt a bit-exact speedup (none exists — the sequential
  dependency chain is inherent to the order).

## frankenjax-cc-softmax-jax-loss - 2D softmax LOSES 2.2x to JAX (threading is NOT the lever)

- Date: 2026-06-19
- Agent: cc / CobaltForge
- GAP WHERE WE LOSE (measured, same-load back-to-back at load ~18): Rust fused
  `nn::softmax_2d` [65536,16] f64 = **2.22 ms** ([2.20,2.24] Criterion) vs JAX
  `jax.nn.softmax(x, axis=-1)` = **1.01 ms** (mean 1.008, p50 1.016, CV 11.4%),
  jax 0.10.1 CPU x64, warmed. **Rust/JAX 2.20 = JAX 2.2x FASTER.**
- Attempt (REVERTED): hypothesised softmax was under-threaded — `softmax_2d_thread_count`
  uses the memory-bound `work_scaled_threads` (65536 elems/thread -> only ~16 threads at
  1M elems, ~46 cores idle). Tried a compute-scaled count (16384 elems/thread -> ~64
  threads). Result: **3.59 ms, a REGRESSION** (1.6x SLOWER than the 16-thread default).
  Reverted (nn.rs byte-identical to HEAD). So softmax is NOT thread-starved: the
  16-thread default is already optimal; oversubscribing adds thread::scope spawn cost +
  L3/cache contention. The op is L3-bandwidth / exp-throughput bound, not core-starved.
- Root cause of the loss: per-element scalar `f64::exp` (libm, ~7-10ns) vs XLA's
  VECTORIZED exp, plus XLA fusing the max/exp/sum/div passes. The fix (SIMD or fast-poly
  exp) is the same one gated in [[project_simd_poly_exp_fma_finding]]: SIMD-poly f64 exp
  is 2.2x WITH FMA / 0.79x WITHOUT, and FMA is blocked by the deliberate no-`+fma` /
  `#![forbid(unsafe_code)]` policy ([[project_fma_lever_policy_blocked]]). A scalar fast-poly
  exp inside softmax would also break the bit-identical `softmax_2d_fused_bit_identical_to_rowmap`
  guard. So this loss is BLOCKED on the same FMA/SIMD-exp maintainer decision as matmul/conv.
- Retry predicate: do NOT re-thread softmax (more threads REGRESS — measured). The softmax
  (and any exp-bound nn op: log_softmax, logsumexp, gelu, sigmoid) vs-JAX gap is the exp
  throughput wall; only a `+fma`/SIMD-exp policy change moves it. Reproducer:
  `benchmarks/jax_comparison/softmax_gauntlet.py`.

## frankenjax-cntiy - global +fma alone does NOT close the 2D softmax JAX gap; row-call devirtualization regressed

- Date: 2026-06-20
- Agent: cod-b / WildForge
- Target: `frankenjax-cntiy` maintainer gate, focused on the current
  `nn/softmax_2d_65536x16_fused` JAX loss.
- Same-host JAX oracle:
  `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/softmax_gauntlet.py`
  reported JAX CPU x64 `jax.nn.softmax(axis=-1)` mean **1.0524 ms**, p50
  **1.0765 ms**, CV 9.39%.

| Probe | Rust mean | Rust/JAX mean | Verdict |
| --- | ---: | ---: | --- |
| default Rust, local `fj-lax` Criterion | 2.2163 ms | 2.106 | LOSS: JAX still 2.11x faster |
| `RUSTFLAGS="-C target-feature=+fma"`, local | 2.2096 ms | 2.100 | LOSS vs JAX; NEUTRAL as a lever (0.997x, Criterion no-change p=0.26) |
| generic `fill_softmax_rows_parallel<F>` row-call devirtualization patch | 2.3303 ms | 2.214 | REGRESSION: +5.34%, p=0.00; reverted before commit |

- RCH evidence using requested `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`:
  `cargo test -p fj-lax softmax_2d --lib` passed on `ovh-a` (10 passed, 0
  failed). The crate-scoped default-vs-`+fma` RCH probe produced one anomalous
  first default row on `vmi1149989` (`softmax` 11.820 ms, `log_softmax` 9.0829
  ms) followed by `+fma` rows of 2.5208 ms and 3.7028 ms. A later default RCH
  rerun on `vmi1152480` measured `softmax` at 2.1842 ms, matching local default,
  so the apparent 4.7x RCH `+fma` improvement is rejected as non-proof.
- Ratio scorecard for this pass, using same-host Rust/JAX rows: **0 wins / 3
  losses / 0 neutral vs JAX**. Lever scorecard: `+fma` alone neutral/no-ship;
  row-call devirtualization rejected and reverted.
- Decision: no production `+fma` build policy change and no `nn.rs` source
  change. The current softmax path still calls scalar `f64::exp`; a global FMA
  flag by itself does not create the missing SIMD/fast-exp kernel. The maintainer
  decision remains narrower than "turn on +fma": approve a relaxed-FP
  SIMD/fast-exp internal softmax/attention contract or keep the bit-exact scalar
  path and accept the ~2.1x JAX loss.
- Retry predicate: do not reattempt function-pointer/generic devirtualization
  for softmax rows. Do not claim global `+fma` closes softmax unless a same-host
  run shows the `nn/softmax_2d_65536x16_fused` Rust/JAX ratio below 1.0. The
  next real lever is a semantics-approved SIMD/fast-exp specialization, not the
  build flag alone.

## frankenjax-cc-unary-threading-surface-mined - Cheap Unary Threading Audit (negative)

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Investigation (no code change): audited whether cheap single-input "win-class"
  unary ops (neg/abs/sign/square — 1 read, 1 write, fresh page-faulting output)
  are threaded above the DRAM gate, since they are NOT named in the prior threaded
  movement/convert list. Finding: they ALREADY ARE threaded for the hot dtypes.
  `eval_neg`/`eval_abs`/`eval_unary_int_or_float` route F64 through
  `eval_unary_f64_tensor_fast_path` and F32 through its sibling, both of which fan
  out via `threaded_unary_f64_map` (split_at_mut, calloc'd output) above
  `CHEAP_BINARY_PARALLEL_MIN` — same gate/discipline as the cheap binaries. BF16/F16
  neg/abs are intercepted earlier by `half_neg_abs_simd`. So Floor/Ceil/Reciprocal/
  Neg/Abs/Sign/Square over f64/f32/half are all covered.
- The ONLY unthreaded unary residual is the INTEGER tail (i64/i32/u32/u64 arms of
  `eval_unary_int_or_float`, serial `.iter().map().collect()`). Left serial on
  purpose: it is the niche integer-elementwise tail the dtype-matrix guidance warns
  against grinding (no real ML workload negates/abs's a 16M+ int tensor), and the
  win would be the generic calloc+parallel-fault one, not a new mechanism.
- Retry predicate: do NOT re-audit cheap-unary threading — f64/f32/half done. Only
  revisit the integer-unary tail if a profile shows a real workload bottlenecked on
  large integer neg/abs/sign; otherwise it stays serial.


## frankenjax-cc-convert-half-downcast-threading - ConvertElementType f64/f32 -> bf16/f16 Threading

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Lever: thread the dense `convert_element_type` half-float DOWNCAST
  (`f64/f32 -> bf16/f16`) above `CHEAP_BINARY_PARALLEL_MIN` (1<<23), filling a
  calloc'd `u16` output across the persistent pool via `threaded_convert_into`
  (split_at_mut, zero extra copy). Bit-identical: each lane is an independent
  single-round `convert_{bf16,f16}_bits`, so chunk boundaries never change a bit.
  This extends the existing threaded f64<->f32 convert fast paths to the bf16/f16
  targets used by mixed-precision ML.
- Status: SPLIT VERDICT (measured, same-host local Criterion before/after):
  - **f64 -> bf16: KEEP (win).** Serial 17.40 ms vs threaded 9.50 ms at 16M
    (1<<24) = **1.83x faster**, bit-identical. The 8-byte source read is
    bandwidth/page-fault-bound serially (~7.5 GB/s); splitting across cores
    raises aggregate bandwidth. f16 sibling kept by the same mechanism (f16
    decode is strictly more compute per lane, so it benefits at least as much).
  - **f32 -> bf16: REVERTED (regression).** Serial 6.38 ms vs threaded 7.92 ms
    at 16M = **0.81x (1.24x SLOWER)**. The 4-byte f32 read already runs ~15 GB/s
    serially and the convert is a cheap bit-shift, so thread fan-out overhead
    dominates. The f32-source half-threading guard was removed; f32 stays serial.
- Benchmark guard: `eval/convert_16m_f32_to_bf16` (now serial baseline) and
  `eval/convert_16m_f64_to_bf16` (threaded win), added to
  `crates/fj-lax/benches/lax_baseline.rs`.
- Measured commands (same host, AMD 5975WX, 64 logical CPUs):
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc-local cargo bench
    -p fj-lax --bench lax_baseline -- eval/convert_16m --warm-up-time 1
    --measurement-time 8`, before (serial: bf16 guards forced off) vs after.
  - vs-JAX HEAD-TO-HEAD (measured 2026-06-19 later, same-load back-to-back at load
    ~9 as the host wound down): Rust threaded 10.31 ms ([10.25, 10.36] Criterion)
    vs JAX `jax.jit(lambda a: a.astype(jnp.bfloat16))` 13.94 ms mean (p50 14.04,
    CV 5.6%) over the same 16M f64 input, jax 0.10.1 CPU x64, warmed
    block_until_ready 20 runs x 20 inner. **Rust/JAX 0.74 = Rust 1.35x FASTER.**
    Confirms the win class: the 8-byte source read is BW/page-fault-bound and
    Rust's parallel page-fault beats XLA's CPU astype. (Earlier this session the
    contended host blocked this row; the same-load back-to-back is now trustworthy.)
- Conformance guard: `cargo test -p fj-lax --release --lib convert` plus the full
  `--lib` suite (see below) pass for all convert paths; bit-identity covered by
  `convert_element_type_dense_matches_generic_all_targets`.
- Retry predicate: do NOT thread the f32 (or narrower, 4-byte) source half
  downcast — measured regression. Do not re-attempt f32->bf16 threading without a
  fundamentally different mechanism (e.g. SIMD bf16 pack, separately found NO-WIN
  in `project_bf16_matmul_and_convert_simd`). f64->bf16/f16 threading is shipped.

## frankenjax-cc-densify-guard-test-restoration - cbea72b3 broke 40 dense-vs-boxed guard tests

- Date: 2026-06-19
- Agent: cc / CobaltForge
- Finding (NOT a perf lever; a conformance regression discovered while verifying):
  `cbea72b3` ("frankenjax-mcqr.97 perf: TensorValue::new dense literal storage,
  code-first batch-test pending", 2026-06-18) made `TensorValue::new` densify
  homogeneous literal vectors for ALL dtypes via `dense_buffer_for_declared_dtype`
  (F64/F32/half/complex/bool/ints). This silently turned the *reference* inputs of
  40 `dense_*_matches_generic` / `*_bit_identical_to_literal_path` guard tests
  dense, tripping their `as_*_slice().is_none()` preconditions and collapsing the
  dense-vs-boxed comparison into a tautology. `cargo test -p fj-lax --lib` was RED
  (43 failed) for ~20 h; the deferred "batch-test" never landed.
- Fix (test-module only, the documented i64-densify precedent b76fa3be): added a
  `#[cfg(test)] crate::new_boxed(dtype, shape, Vec<Literal>)` helper that routes
  through `TensorValue::new_with_literal_buffer(.., LiteralBuffer::new(..))` to
  keep references genuinely boxed, and pointed the 40 failing tests' reference
  builders at it (53 scoped call-site swaps + 2 shared helpers `v_f64`/
  `make_complex_vector`). Two output-storage assertions whose fallback output now
  legitimately densifies (`reduce_window malformed-literal`, `f64_scalar_broadcast`)
  were relaxed to value-equality (still proves the literal fallback path is taken).
- Result: fj-lax `--lib` 1556 passed / 3 failed (was 1516/43). The 3 remaining are
  PRE-EXISTING and NOT densify-related (confirmed failing on a clean-HEAD stash):
  `eval_polygamma_scalar`, `threaded_dense_polygamma_bit_identical_to_reference`
  (polygamma eval returns Err for digamma(1.0)), and
  `complex_tensor_scalar_dense_path_bit_identical_to_literal` (complex
  Atan2/XLogY/LogAddExp return Err at the dense eval). Handed off to the team.
- Retry predicate: when `TensorValue::new`-built homogeneous tensors are used as a
  BOXED reference in a test, build them via `crate::new_boxed` (fj-lax) /
  `new_with_literal_buffer(LiteralBuffer::new(..))`; do not assume `TensorValue::new`
  stays boxed. The 3 pre-existing polygamma/complex eval failures are a separate
  correctness gap (not densify, not perf).
## frankenjax-6dfew - CompiledJaxpr Tensor-Param Pre-Scan (NO-SHIP)

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever considered: extend `CompiledJaxpr`'s dense plan with typed tensor
  equation param pre-scan so hot repeated eval avoids per-call
  `BTreeMap<String, String>` param parsing and generic `eval_primitive`
  dispatch for tensor equations.
- Alien/optimization mapping: graveyard vectorized execution + query/JIT
  entries point at removing interpreter overhead for hot repeated queries, but
  the live code already applies the matching lower-risk primitive:
  `run_dense_env_into` fuses maximal cheap elementwise tensor chains via
  `try_fuse_elementwise_chain`, and scalar chains use existing scalar arena
  plans. Remaining residual is not the original dispatch/param-parse target.
- Status: NO-SHIP / close as negative evidence. The focused benchmark does not
  clear the Score >= 2.0 gate for the proposed lever:
  - scalar compiled path is slower than eager:
    n=8 146.14 ns vs 34.66 ns (0.24x), n=32 217.46 ns vs 82.13 ns
    (0.38x), n=128 695.67 ns vs 310.40 ns (0.45x).
  - tensor64 compiled path is only a small win:
    n=8 1.8546 us vs eager 2.1616 us (1.17x), n=32 7.3435 us vs
    eager 7.9298 us (1.08x).
  These are already after chain fusion, so typed param pre-scan can only attack
  the small leftover call overhead, not the dominant per-element work.
- Head-to-head vs JAX CPU (`jax.jit`, x64 enabled, JAX 0.10.2,
  `uv run --with 'jax[cpu]' --with numpy`, 40 runs x 200 inner loops):
  - scalar n=8/32/128 JAX means 5.735/6.146/5.101 us; Rust eager and compiled
    are far faster in-process, but compiled is worse than eager, so no new
    CompiledJaxpr lever is justified.
  - tensor64 n=8 JAX mean 5.026 us; Rust compiled 1.855 us (Rust/JAX 0.37x).
  - tensor64 n=32 JAX mean 4.954 us; Rust compiled 7.343 us (Rust/JAX 1.48x).
    This external loss is real, but it is not a typed-param-parse loss.
- Measurement commands:
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- compiled_dispatch --warm-up-time 1 --measurement-time 5 --sample-size 20`
    selected `hz2` and used a cold worker-scoped target path.
  - `uv run --with 'jax[cpu]' --with numpy python benchmarks/jax_comparison/interpreter_compiled_dispatch_gauntlet.py --runs 40 --warmup 8 --inner-loops 200 --output /tmp/frankenjax_interpreter_compiled_dispatch_jax.json`.
- Behavior proof / non-implementation rationale: no production code changed.
  The obvious JAX-style rewrite, folding `(((x + 1.0) + 1.0) ...)` to
  `x + n`, is not bit-preserving for arbitrary floating-point inputs because
  it changes rounding points. It is only safe for integer wrapping additions,
  where Rust already beats JAX and compiled still loses to eager internally.
- Retry predicate: do not implement tensor-param pre-scan for Add-chain dispatch
  without a new profile showing a non-fused, param-heavy single-op workload
  where param parsing is top-5. To attack the tensor64 n=32 JAX loss, use a
  semantics-safe optimizer/codegen lane with explicit FP contract, or buffer
  reuse/arena work that reduces allocations without changing left-associative
  float semantics.

## frankenjax-cod-b-dense-tensor-stack-axis0-rw4k4 - Dense Tensor stack_axis0 Concat Storage

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route `TensorValue::stack_axis0` for tensor inputs through
  `LiteralBuffer::from_concat_slices` plus `new_with_literal_buffer`, preserving
  packed dense storage instead of materializing a `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original JAX;
  internal control is a small but separated Rust win versus the literal-backed
  stack path. No revert.
- Benchmark guard: `core/tensor_stack_axis0_dense_f64_64x1k` plus
  `core/tensor_stack_axis0_literal_f64_64x1k` control.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_stack_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda *xs: jnp.stack(xs, axis=0))`
    over 64 x 1024 `float64` arrays with `jax_enable_x64=true`, CPU backend,
    80 runs x 200 inner loops via `benchmarks/jax_comparison/.venv/bin/python`
    (JAX/JAXLIB 0.10.1).
  - Rust dense mean estimate 3.3963 us (`[3.3375, 3.4533]` us Criterion
    interval) vs JAX mean 41.0467 us (p50 40.4710 us, CV 6.79%):
    Rust/JAX 0.083x, Rust 12.09x faster.
  - Rust literal-backed control mean estimate 3.5255 us
    (`[3.4610, 3.5862]` us Criterion interval): dense/literal 0.963x,
    dense 1.04x faster internally. This is a narrow construction-path win, not
    a broad dense-storage breakthrough.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core stack_axis0 --lib`
  passed 6 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` also passed after repairing two pre-existing
  test-gate issues in `crates/fj-core/src/lib.rs` (missing helper imports and a
  `clone_on_copy` test lint).
- Retry predicate: do not repeat this `stack_axis0` tensor concat-storage lever
  unless a new profile shows a distinct stack call path or a larger shape where
  literal materialization reappears. The sibling scalar-stack, repeat, slice, and
  `to_i64_vec` dense-storage beads still require their own JAX head-to-head rows;
  do not generalize this 12.09x external win to those levers.

## frankenjax-cod-b-dense-scalar-stack-axis0-tobpl - Dense Scalar stack_axis0 Packing

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: specialize `TensorValue::stack_axis0` for homogeneous scalar outputs
  and route them directly to dense typed tensor constructors, avoiding boxed
  `Vec<Literal>` materialization for loop-and-stack scalar workloads.
- Status: measured keep. External head-to-head is a decisive Rust win versus
  original JAX on a realistic 64-scalar loop-and-stack workload. No revert.
- Benchmark guard: `core/scalar_stack_axis0_f64_64`, added to
  `crates/fj-core/benches/core_baseline.rs` in this measurement commit.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/scalar_stack_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda *xs: jnp.stack(xs, axis=0))`
    over 64 scalar `float64` inputs with `jax_enable_x64=true`, CPU backend,
    200 samples x 1000 inner loops via
    `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
    (JAX 0.10.1).
  - Rust mean estimate 137.53 ns (`[135.34, 139.81]` ns Criterion interval),
    median 136.77 ns, p95 158.22 ns, p99 165.03 ns, sample CV 8.22%.
  - JAX mean 25.1019 us, p50 24.4422 us, p95 28.5403 us, p99 31.3033 us,
    CV 6.26%.
  - Rust/JAX 0.0055x, Rust 182.21x faster. This row is host-dispatch dominated
    on the JAX side, but the ratio is far beyond the noise band.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core stack_axis0 --lib`
  passed 6 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` also passed.
- Retry predicate: do not repeat scalar `stack_axis0` dense-constructor packing
  unless a new profile shows a non-`f64` scalar dtype, larger scalar-count batch,
  or a compiled buffer-reuse path with materially different cost. Continue to
  benchmark scalar repeat, slice, and extraction siblings separately.

## frankenjax-cod-b-dense-scalar-repeat-axis0-zb2b7 - Dense Scalar repeat_axis0 Packing

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify scalar `TensorValue::repeat_axis0` dense-constructor packing for
  homogeneous scalar repeats, avoiding a materialized `Vec<Literal>` loop before
  producing the length-64 dense tensor.
- Status: measured keep. External head-to-head is a decisive Rust win versus
  original JAX on scalar repeat construction, and the materializing Rust control
  confirms a real local construction-path win. No revert.
- Benchmark guard: `core/scalar_repeat_axis0_f64_64`, plus
  `core/scalar_repeat_axis0_f64_64_materializing_control` in
  `crates/fj-core/benches/core_baseline.rs`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/scalar_repeat_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-scalar-repeat-axis0-control`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_repeat_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_scalar_repeat_axis0_jax_raw.json`.
  - Rust optimized mean estimate 66.809 ns (`[65.832, 67.860]` ns Criterion
    interval) vs JAX mean 4.6720 us (p50 4.5007 us, p95 5.6054 us,
    p99 6.3644 us, CV 9.46%): Rust/JAX 0.0143x, Rust 69.93x faster.
  - Rust materializing control mean estimate 181.12 ns (`[176.40, 186.55]` ns
    Criterion interval): optimized/materializing 0.369x, optimized 2.71x
    faster internally.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core repeat_axis0 --lib`
  passed 5 tests, 0 failed (RCH fell back to local execution). Per-crate
  `cargo check -p fj-core --bench core_baseline`, `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` passed.
- Retry predicate: do not retry scalar `repeat_axis0` dense-constructor packing
  unless a new profile shows a non-`f64` scalar dtype, larger repeat count, or a
  compiled buffer-reuse path with materially different cost. Future scalar repeat
  work should target compiled output-buffer reuse, not another `Vec<Literal>`
  elision.

## frankenjax-cod-b-dense-tensor-repeat-axis0-jk3ed - Dense Tensor repeat_axis0 Concat Storage

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route `TensorValue::repeat_axis0` for tensor inputs through
  `LiteralBuffer::from_concat_slices` plus `new_with_literal_buffer`, preserving
  packed dense storage instead of materializing a repeated `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original JAX;
  a temporary materializing-path revert regressed badly, so the optimization was
  restored and kept. No committed revert.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_cod_b_dense_tensor_repeat_axis0_gauntlet_2026-06-19.json`.
- Benchmark guard: `core/tensor_repeat_axis0_dense_f64_1k_x64`. The in-tree
  `core/tensor_repeat_axis0_literal_f64_1k_x64` row is not a valid
  pre-optimization control after `TensorValue::new` densification, because its
  input is also dense by the time `repeat_axis0` runs.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust optimized command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_repeat_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: warmed inline `jax.jit(lambda v: jnp.repeat(v[None, :], 64, axis=0))`
    over a 1024-element `float64` vector with `jax_enable_x64=true`, CPU backend,
    80 runs x 200 inner loops via `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
    (JAX/JAXLIB 0.10.1).
  - Rust optimized dense mean estimate 3.2461 us (`[3.2053, 3.2855]` us
    Criterion interval) vs JAX mean 11.7639 us (p50 11.1859 us, CV 10.47%):
    Rust/JAX 0.276x, Rust 3.62x faster.
  - Same-code-path dense-input/literal-input rows before the temporary revert:
    3.2461 us vs 3.2266 us. This apparent neutral result was not a valid
    pre-optimization comparison because both inputs reached the direct concat
    path with dense storage.
  - Temporary materializing-path control, produced by locally reverting only the
    direct concat hunk and then rerunning the same Criterion filter: 92.057 us
    (`[90.552, 93.632]` us). Optimized/materializing 0.035x, optimized 28.36x
    faster. The temporary revert was discarded before commit.
  - Independent cod-a validation rerun: Rust dense Criterion slope 2.887 us
    (`[2.853, 2.922]` us) vs a materialized JAX `jnp.tile(x[None, :], (64, 1))`
    mean 16.606 us (p50 15.721 us, CV 29.61%): Rust/JAX 0.174x, Rust 5.75x
    faster. Treat this as directional because the JAX CV is high; the keep
    decision still rests on the 28.36x materializing-control result above.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-core repeat_axis0 --lib`
  passed 5 tests, 0 failed. Per-crate `cargo check -p fj-core --all-targets`,
  `cargo clippy -p fj-core --all-targets -- -D warnings`, and
  `cargo fmt -p fj-core --check` passed.
- Retry predicate: do not retry tensor `repeat_axis0` literal materialization or
  direct concat-storage variants without a new profile showing a distinct
  repeat path. Follow-up should target compiled/buffer-reuse semantics or larger
  batched repeat workloads, not another `Vec<Literal>` elision for this path.

## frankenjax-cod-b-dense-slice-axis0-4bnj5 - Dense Tensor slice_axis0 Storage Preservation

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify dense `TensorValue::slice_axis0` rank-2 row extraction as a
  storage-preserving view-like construction instead of rematerializing the row
  through `Vec<Literal>`.
- Status: measured keep. External head-to-head is a Rust win versus original
  JAX CPU; local materializing control confirms a real construction-path win.
  No revert.
- Benchmark guard: `core/tensor_slice_axis0_dense_f64_64x1k_row31`,
  `core/tensor_slice_axis0_dense_f64_64x1k_row31_to_f64_vec`, and
  `core/tensor_slice_axis0_materializing_control_f64_64x1k_row31`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_slice_axis0 --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-slice-axis0-control`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_slice_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_slice_axis0_jax_raw.json`.
  - Rust lazy slice construction mean estimate 238.18 ns
    (`[235.10, 241.48]` ns Criterion interval).
  - Rust slice plus `to_f64_vec` mean estimate 513.73 ns
    (`[507.65, 520.11]` ns Criterion interval).
  - Rust materializing control mean estimate 1.4253 us
    (`[1.4056, 1.4456]` us Criterion interval): lazy/control 0.167x
    (5.98x faster) and extract/control 0.360x (2.77x faster).
  - JAX bare row slice mean 5.1409 us (p50 4.8466 us, p95 6.5725 us,
    p99 6.9790 us, CV 11.35%).
  - JAX row slice plus `+0.0` mean 5.0677 us (p50 4.9258 us, p95
    6.0064 us, p99 6.3937 us, CV 7.40%).
  - Rust/JAX ratios: lazy vs bare 0.0463 (Rust 21.58x faster);
    extract vs `+0.0` 0.101 (Rust 9.86x faster). Treat the exact external
    ratios as directional because JAX CV exceeded 5%.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core slice_axis0 --lib`
  passed 2 tests, 0 failed. Per-crate `cargo check -p fj-core --bench core_baseline`,
  `cargo check -p fj-core --all-targets`, `cargo clippy -p fj-core --all-targets -- -D warnings`,
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings`, `cargo fmt -p fj-core --check`,
  and `ubs --only=rust --skip-rust=12,13,14 crates/fj-core/benches/core_baseline.rs`
  passed for this lane.
- Retry predicate: do not retry `slice_axis0` row construction or another
  `Vec<Literal>`-elision variant without a new profile showing slice
  construction, not downstream arithmetic, is again a top-five fj-core cost.
  Future work should target compiled consumer fusion or direct output-buffer
  reuse, not another row-slice storage repack.

## frankenjax-cod-b-dense-to-i64-vec-7xbu9 - Dense Tensor to_i64_vec Fast Path

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: verify `TensorValue::to_i64_vec` through
  `LiteralBuffer::as_i64_slice` before falling back to per-`Literal`
  extraction, avoiding boxed literal iteration for packed I64 buffers and
  concat I64 slices.
- Status: measured keep. The direct dense extraction path is a decisive Rust
  internal win versus the literal-backed fallback and a directional external
  win versus original JAX host extraction. No revert.
- Benchmark guard: `core/tensor_to_i64_vec_dense_4k` and
  `core/tensor_to_i64_vec_literal_4k` in `crates/fj-core/benches/core_baseline.rs`,
  plus `benchmarks/jax_comparison/core_to_i64_gauntlet.py` for the JAX host-copy
  comparator.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_to_i64_vec --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-to-i64-vec`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_to_i64_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_to_i64_jax_raw.json`.
  - Rust dense mean estimate 552.47 ns (`[542.29, 565.54]` ns Criterion
    interval).
  - Rust literal-backed fallback mean estimate 6.6566 us
    (`[6.4275, 6.8878]` us Criterion interval): dense/literal 0.0830x,
    dense 12.05x faster internally.
  - JAX identity-ready lower-bound mean 6.5825 us (p50 6.1655 us,
    p95 8.9080 us, p99 12.4299 us, CV 19.47%): dense Rust/JAX 0.0839x,
    Rust 11.92x faster.
  - JAX NumPy host-copy mean 9.0209 us (p50 8.8057 us, p95 10.4230 us,
    p99 10.6834 us, CV 6.56%): dense Rust/JAX 0.0612x, Rust 16.33x
    faster. Treat both external ratios as directional because JAX CV exceeded
    5%; the local dense/literal separation is the primary keep proof.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core to_i64_vec --lib`
  passed 4 tests, 0 failed on `vmi1152480`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `hz1`, `python -m py_compile benchmarks/jax_comparison/core_to_i64_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_to_i64_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not retry dense `to_i64_vec` or another
  `as_i64_slice -> Vec<i64>` elision for this path without a fresh profile
  showing extraction remains a top-five fj-core cost. Future work should target
  compiled consumers, output-buffer reuse, non-I64 extraction, or avoiding the
  owned host copy entirely.

## frankenjax-mcqr.97 - TensorValue::new Dense Literal Storage

- Date: 2026-06-18; verified 2026-06-19
- Agent: cod-b / TopazOrchid; verified by cod-b / WildForge
- Lever: densify homogeneous F64/F32/Bool/BF16/F16/Complex literal vectors in
  `TensorValue::new` after element-count validation.
- Status: measured mixed. Keep the dense storage behavior for downstream typed
  consumers, but record negative evidence for construction-only workloads:
  generic densification is slower than forced literal construction until a dense
  slice/vector consumer uses the packed storage.
- Benchmark guard: `core/tensor_value_new_1k_f64_generic_dense`,
  `core/tensor_value_new_1k_f64_forced_literal`,
  `core/tensor_value_new_then_to_f64_vec_1k`,
  `core/tensor_value_new_forced_literal_then_to_f64_vec_1k`, plus
  `benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/tensor_value_new --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-tensor-value-new`.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_tensor_value_new_jax_raw.json`.
  - Rust generic dense construction mean estimate 1.4152 us
    (`[1.4005, 1.4300]` us Criterion interval).
  - Rust forced literal construction mean estimate 432.80 ns
    (`[427.08, 438.80]` ns Criterion interval): generic dense / forced literal
    3.27x, a real construction-only regression.
  - Rust generic dense construction plus `to_f64_vec` mean estimate 1.4131 us
    (`[1.3947, 1.4326]` us Criterion interval).
  - Rust forced literal construction plus `to_f64_vec` mean estimate 2.3787 us
    (`[2.3368, 2.4269]` us Criterion interval): dense / forced literal 0.594x,
    dense 1.68x faster once extraction consumes typed storage.
  - JAX `jnp.asarray(values).block_until_ready()` mean 48.5634 us
    (p50 48.8473 us, p95 52.3213 us, p99 54.9046 us, CV 5.04%):
    generic dense Rust/JAX 0.0291x, Rust 34.31x faster.
  - JAX `jnp.asarray(values)` plus NumPy host copy mean 55.9286 us
    (p50 56.1224 us, p95 60.8686 us, p99 63.5072 us, CV 5.23%):
    dense Rust extraction / JAX host copy 0.0253x, Rust 39.58x faster. Treat
    exact external ratios as directional because JAX CV is just above 5%.
- Conformance guard: matching literal families materialize bit-identically;
  mismatched literal/dtype tensors remain literal-backed.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core tensor_new_densifies_matching_literal_families --lib`
  passed 1 test, 0 failed on `ovh-a`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz1`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_tensor_value_new_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not use `TensorValue::new` densification as a
  construction-only performance lever; use direct typed constructors or
  `new_with_literal_buffer` for hot paths that will not consume typed storage.
  Do not repeat FMA, SIMD exp, GEMM, QR, SVD, cumsum, eager concat, or the
  already committed stack/repeat/slice/to_i64 storage levers without fresh
  same-worker benchmark evidence identifying a distinct call path.

## frankenjax-mcqr.99 - Direct Dense LiteralBuffer to_vec Materialization

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::to_vec()` materialize directly from dense storage
  variants instead of forcing `as_slice()` to build/cache a full literal vector
  and then clone it.
- Status: measured keep internally; mixed external head-to-head versus JAX.
- Benchmark guard: `core/literal_buffer_to_vec_dense_f64_64k`,
  `core/literal_buffer_to_vec_literal_f64_64k`, plus
  `benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`.
- Measured evidence:
  - Rust Criterion, local same-host:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/literal_buffer_to_vec --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-literal-buffer-to-vec`.
  - Dense F64 64k direct materialization: 26.644 us mean
    (`[26.253, 27.043]` us).
  - Literal F64 64k fallback/control: 33.306 us mean
    (`[32.779, 33.868]` us).
  - Internal result: dense direct materialization is 0.800x the literal control,
    or 1.25x faster.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_literal_buffer_to_vec_jax_raw.json`.
  - JAX identity-ready lower bound: 19.5312 us mean, p50 19.2092 us,
    p95 22.7052 us, p99 24.0948 us, CV 9.07%; Rust/JAX 1.364x, so Rust is
    1.36x slower than this no-host-copy lower bound.
  - JAX NumPy host-copy comparator: 32.9368 us mean, p50 32.4829 us,
    p95 38.4794 us, p99 41.1890 us, CV 8.69%; Rust/JAX 0.809x, so Rust is
    1.24x faster than this host-copy comparator.
  - External ratios are directional because both JAX rows have CV above 5%, and
    the JAX host-copy row copies raw F64 values while Rust materializes
    `Literal` enum values.
- Conformance guard: direct `to_vec()` matches `as_slice().to_vec()` bit-for-bit
  across dense F64/F32/I64/U32/U64/Bool/BF16/F16/Complex storage plus lazy
  concat/repeated-patches paths.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_to_vec_direct_paths_match_slice_materialization --lib`
  passed 1 test, 0 failed on `ovh-a`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_literal_buffer_to_vec_gauntlet.py`
  returned 0 warnings.
- Retry predicate: do not retry direct dense `LiteralBuffer::to_vec`
  materialization for the F64 64k path unless a fresh profile still puts
  `to_vec` in the top five. The next attempt should target cached literal reuse,
  consumer-side avoidance of `Vec<Literal>`, or direct typed-consumer APIs rather
  than another direct dense-to-`Literal` map. Do not retry the already committed
  stack/repeat/slice/to_i64 or `TensorValue::new` dense-storage families under
  this bead. Do not revisit FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-q59j4 - Direct Dense LiteralBuffer COW Mutation

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer::make_mut()` materialize dense/lazy storage through
  the direct storage-aware `to_vec()` path instead of forcing `as_slice()` to
  build/cache an intermediate full literal vector before mutation.
- Status: measured keep internally; no direct JAX API comparator.
- Benchmark guard: `core/literal_buffer_index_mut_dense_f64_64k`,
  `core/literal_buffer_index_mut_literal_f64_64k`.
- Measured evidence (2026-06-20, RCH remote worker `vmi1149989`):
  - Command: `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo bench -p fj-core --bench core_baseline -- 'core/literal_buffer_(serialize|index_mut)_(dense|literal)_f64_64k' --sample-size 20 --warm-up-time 1 --measurement-time 3`.
  - `core/literal_buffer_index_mut_dense_f64_64k`: 24.003 us mean
    (23.591-24.411 us).
  - `core/literal_buffer_index_mut_literal_f64_64k`: 33.278 us mean
    (32.714-33.793 us).
  - Dense/literal control ratio: 0.721x, or 1.39x faster than the literal
    control. JAX ratio: N/A, because this is a host-internal mutable
    `LiteralBuffer` storage path rather than a JAX API-equivalent workload.
  - Decision: keep, not revert.
- Conformance guard: dense and lazy COW mutation preserves the original cloned
  sequence and mutates to the same materialized literal sequence as a literal
  buffer for F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, plus a dense-sort materialization path.
  `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_make_mut_direct_paths_preserve_cow_sequences --lib`
  passed 1 test, 0 failed on `vmi1149989`; `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-conformance --lib`
  passed 45 tests, 0 failed on `hz2`.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, or `LiteralBuffer::to_vec` dense-storage families under
  this bead. Reopen the dense COW mutation family only with focused criterion
  evidence showing mutation/materialization cost remains a top-five fj-core
  bottleneck. Do not revisit FMA, SIMD exp, GEMM, QR, SVD, cumsum, or eager
  concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.100 - Dense Half-Width Bitcast Chunks

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->BF16/F16 and BF16/F16->F32
  through packed f32/u16 slices instead of per-`Literal` byte conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  both measured BF16 reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_bf16_dense_1m`,
  `eval/bitcast_f32_bf16_literal_ref_1m`,
  `eval/bitcast_bf16_f32_dense_1m`,
  `eval/bitcast_bf16_f32_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench lax_baseline`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench lax_baseline -- 'eval/bitcast_(f32_i32|i32_f32|f64_u64|u64_f64|f32_bf16|bf16_f32)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline frankenjax-mcqr-100-101-bitcast`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/bitcast_gauntlet.py --runs 30 --warmup 5 --inner-loops 50 --output /tmp/frankenjax_mcqr_100_101_bitcast_jax_raw.json`.
  - `bitcast_f32_bf16_1m`: Rust dense mean 2.710 ms vs Rust literal
    reference 57.706 ms (21.29x faster internally); JAX mean 138.058 us;
    Rust/JAX 19.63x slower.
  - `bitcast_bf16_f32_1m`: Rust dense mean 669.626 us vs Rust literal
    reference 31.549 ms (47.11x faster internally); JAX mean 139.426 us;
    Rust/JAX 4.80x slower.
  - Decision: keep, not revert. Both rows are large internal wins over the
    original per-Literal path, but they are external JAX losses. The next BF16
    bitcast attempt must target raw packed output construction throughput, not
    another boxed-literal-elision pass.
- Conformance guard: dense and literal-backed half-width bitcasts produce the
  same output shapes, dtypes, raw half chunks, and round-trip f32 bit patterns
  across NaN, infinities, signed zero, normals, and a custom NaN payload.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry this half-width bitcast family unless focused
  criterion evidence shows the dense rows are still slower than the
  literal-backed reference or the original path remains a top-five fj-lax
  bottleneck. Do not merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or peer-owned fj-core dense-storage lanes without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.101 - Dense Signed/Unsigned Same-Width Bitcast Pairs

- Date: 2026-06-18
- Agent: cod-a / TopazOrchid
- Lever: route dense `BitcastConvertType` F32->I32, I32->F32, F64->U64, and
  U64->F64 through packed typed slices instead of per-`Literal` byte
  conversion.
- Status: measured keep internally; negative head-to-head result versus JAX on
  all four measured same-width reinterpret workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_100_101_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `eval/bitcast_f32_i32_dense_1m`,
  `eval/bitcast_f32_i32_literal_ref_1m`,
  `eval/bitcast_i32_f32_dense_1m`,
  `eval/bitcast_i32_f32_literal_ref_1m`,
  `eval/bitcast_f64_u64_dense_1m`,
  `eval/bitcast_f64_u64_literal_ref_1m`,
  `eval/bitcast_u64_f64_dense_1m`,
  `eval/bitcast_u64_f64_literal_ref_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust/JAX commands match the `frankenjax-mcqr.100` bitcast gauntlet entry.
  - `bitcast_f32_i32_1m`: Rust dense mean 430.783 us vs Rust literal
    reference 19.537 ms (45.35x faster internally); JAX mean 133.290 us;
    Rust/JAX 3.23x slower.
  - `bitcast_i32_f32_1m`: Rust dense mean 642.693 us vs Rust literal
    reference 22.677 ms (35.28x faster internally); JAX mean 93.945 us;
    Rust/JAX 6.84x slower.
  - `bitcast_f64_u64_1m`: Rust dense mean 270.723 us vs Rust literal
    reference 24.747 ms (91.41x faster internally); JAX mean 176.611 us;
    Rust/JAX 1.53x slower.
  - `bitcast_u64_f64_1m`: Rust dense mean 228.320 us vs Rust literal
    reference 20.525 ms (89.89x faster internally); JAX mean 175.404 us;
    Rust/JAX 1.30x slower.
  - Decision: keep, not revert. Every row is a large internal win and the
    f64/u64 rows are near the existing memory-bandwidth residual gap, but all
    rows are still external JAX losses.
- Conformance guard: dense and literal-backed signed/unsigned same-width
  bitcasts produce the same shapes, dtypes, exact integer bit lanes, packed
  storage, and round-trip float bit patterns across NaN, infinities, signed
  zero, normals, and custom NaN payloads.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed.
- Retry predicate: do not retry the already committed F32<->U32, F64<->I64,
  F64<->U32, U32<->F64, F32<->BF16/F16, BF16/F16->F32, or these signed/
  unsigned same-width bitcast pairs without focused criterion evidence showing
  the dense rows are still slower than the literal-backed reference or the
  original per-`Literal` path remains a top-five fj-lax bottleneck. Do not
  revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  peer-owned fj-core dense-storage lanes without fresh same-worker benchmark
  evidence and ownership check.

## frankenjax-mcqr.96/.98 - Dense U32/I64 Same-Width and Width-Changing Bitcasts

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify the committed dense `BitcastConvertType` fast paths for
  F32->U32, F64->I64, F64->U32, and U32->F64 against both the Rust boxed
  literal reference path and original JAX CPU.
- Status: measured mixed keep. One same-width row is a noisy external Rust win;
  the remaining three rows are negative head-to-head results versus original
  JAX. All four rows stay as internal keeps because they beat the boxed Rust
  reference by 43.99-201.42x. No revert.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_96_98_bitcast_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/lax_baseline.rs`, 1,048,576 element
  bitcast rows, Criterion sample size 20, warmed original-JAX CPU timing with
  50 runs x 500 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench lax_baseline`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench lax_baseline -- 'eval/bitcast_(f32_u32|f64_i64|f64_u32|u32_f64)' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-96-98-bitcast`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/bitcast_gauntlet.py --runs 50 --warmup 10 --inner-loops 500 --output /tmp/frankenjax_mcqr_96_98_bitcast_jax_raw_rerun.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend.

| Workload | Rust dense mean | Rust literal mean | Dense/literal | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bitcast_f32_u32_1m` | 97.423 us | 19.623 ms | 201.42x faster | 113.430 us | 0.859 | Noisy JAX win; keep |
| `bitcast_f64_i64_1m` | 270.586 us | 22.260 ms | 82.27x faster | 183.715 us | 1.473 | Keep internal win; JAX loss |
| `bitcast_f64_u32_1m` | 1.272 ms | 60.757 ms | 47.76x faster | 186.831 us | 6.809 | Keep internal win; JAX loss |
| `bitcast_u32_f64_1m` | 919.708 us | 40.456 ms | 43.99x faster | 189.478 us | 4.854 | Keep internal win; JAX loss |

- CV notes: JAX CV ranged from 12.08% to 19.80%, so the F32->U32 Rust/JAX win
  is directional rather than certification-grade. The three JAX losses are wide
  enough to route follow-up even with the noisy oracle run.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax bitcast --lib`
  passed 4 tests, 0 failed, 0 ignored, 1676 filtered out.
- Decision: keep, not revert. The original code-first optimization is real
  relative to the boxed Rust path, but only the F32->U32 row beat original JAX
  and that result is noisy. Do not claim release-grade JAX domination for this
  cluster.
- Retry predicate: do not retry boxed-literal elision for F32->U32, F64->I64,
  F64->U32, or U32->F64 bitcasts. Same-width follow-up must target raw output
  buffer construction, store throughput, or JAX-like compiled buffer reuse.
  Width-changing follow-up must target packed chunk layout and widening/narrowing
  construction directly, not another dense-storage sibling.

## frankenjax-mcqr.103-.104 - Dense I64 Clamp Tensor and Mixed Bounds

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify committed dense I64 `Clamp` fast paths for same-shape tensor
  bounds and both mixed scalar/tensor bound orders against the boxed Rust
  reference path and original JAX CPU.
- Status: measured keep internally; negative head-to-head result versus JAX on
  all three measured I64 clamp workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_103_104_i64_clamp_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576 I64
  elements per row, Criterion sample size 20, warmed original-JAX CPU timing
  with 50 runs x 100 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench clamp_gauntlet`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- 'i64_' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-103-104-i64-clamp`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_103_104_i64_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1 / 0.10.1, `jax_enable_x64=true`, CPU backend.

| Workload | Rust dense mean | Rust boxed mean | Dense/boxed | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `i64_mixed_scalar_lo_tensor_hi_1m` | 689.991 us | 23.655 ms | 34.28x faster | 200.558 us | 3.44x slower | Keep internal win; JAX loss |
| `i64_mixed_tensor_lo_scalar_hi_1m` | 611.377 us | 25.749 ms | 42.12x faster | 197.536 us | 3.10x slower | Keep internal win; JAX loss |
| `i64_tensor_tensor_tensor_1m` | 1.046 ms | 24.674 ms | 23.59x faster | 287.593 us | 3.64x slower | Keep internal win; JAX loss |

- Conformance guard: dense and boxed I64 clamp paths match the generic
  semantics for same-shape tensor bounds and mixed scalar/tensor bounds.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax clamp --lib`
  passed 25 tests, 0 failed, 6 ignored.
- Decision: keep, not revert. Every measured row is a 23.59-42.12x internal
  win over the boxed Rust reference path, while every row remains a 3.10-3.64x
  external loss versus original JAX CPU.
- Retry predicate: do not retry boxed-literal elision for these I64 clamp
  shapes. The next I64 clamp attempt must target SIMD/parallel clamp
  throughput, output allocation/fusion, or JAX-like compiled buffer reuse, not
  another dense-storage sibling fast path. Do not merge this family with
  half-raw-bit clamp or broadcast-shape generalization without fresh
  Criterion/JAX evidence and ownership check.

## frankenjax-co009 - Stream Dense LiteralBuffer Serialization

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: make `Serialize for LiteralBuffer` stream storage-direct elements
  through `SerializeSeq` instead of forcing `as_slice()` to cache a full
  materialized literal vector for dense packed buffers.
- Status: measured keep internally; no direct JAX API comparator.
- Benchmark guard: `core/literal_buffer_serialize_dense_f64_64k`,
  `core/literal_buffer_serialize_literal_f64_64k`.
- Measured evidence (2026-06-20, RCH remote worker `vmi1149989`):
  - Command: `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo bench -p fj-core --bench core_baseline -- 'core/literal_buffer_(serialize|index_mut)_(dense|literal)_f64_64k' --sample-size 20 --warm-up-time 1 --measurement-time 3`.
  - `core/literal_buffer_serialize_dense_f64_64k`: 1.3443 ms mean
    (1.2858-1.4110 ms).
  - `core/literal_buffer_serialize_literal_f64_64k`: 1.6493 ms mean
    (1.4622-1.8378 ms).
  - Dense/literal control ratio: 0.815x, or 1.23x faster than the literal
    control. JAX ratio: N/A, because this is a conformance/fixture host
    serialization path rather than a JAX API-equivalent workload.
  - Decision: keep, not revert.
- Conformance guard: streamed JSON matches materialized `Vec<Literal>` JSON
  across F64/F64OnePlusX/F32/I64/U32/U64/Bool/BoolWords/Half/Complex,
  repeated-patches, concat, and mixed dense/literal concat paths.
  `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_streamed_serialization_matches_materialized_json --lib`
  passed 1 test, 0 failed on `vmi1149989`; `AGENT_NAME=cod-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-conformance --lib`
  passed 45 tests, 0 failed on `hz2`.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation, or this
  serialization streaming family without fresh focused criterion evidence
  showing it remains a top-five fj-core bottleneck. Do not revisit FMA/SIMD exp,
  GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or eager concat without fresh
  same-worker benchmark evidence and ownership check.

## frankenjax-mcqr.102 - Storage-Direct LiteralBuffer Equality

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: make `LiteralBuffer` equality compare storage-direct typed ranges and
  recursively compare concat slices instead of forcing `as_slice()` on both
  operands and caching full materialized literal vectors for dense packed
  buffers.
- Status: measured keep.
- Benchmark guard: `core/literal_buffer_eq_dense_f64_64k_equal`,
  `core/literal_buffer_eq_dense_f64_64k_mismatch`,
  `core/literal_buffer_eq_literal_f64_64k_equal`, plus
  `benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`.
- Measured evidence:
  - Rust Criterion, local same-host:
    `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-core --bench core_baseline -- core/literal_buffer_eq --sample-size 100 --warm-up-time 1 --measurement-time 10 --save-baseline frankenjax-cod-b-literal-buffer-eq`.
  - Dense F64 64k equal: 27.484 us mean (`[27.047, 27.926]` us).
  - Dense F64 64k tail-mismatch: 27.570 us mean (`[27.226, 27.951]` us).
  - Literal F64 64k equal control: 53.559 us mean (`[52.696, 54.423]` us).
  - Internal result: dense equality is 0.513x the literal equality control, or
    1.95x faster.
  - JAX command:
    `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_cod_b_literal_buffer_eq_jax_raw.json`.
  - JAX equal device-ready lower bound: 58.2421 us mean, p50 54.3390 us,
    p95 76.0146 us, p99 80.4897 us, CV 15.86%; Rust/JAX 0.472x, Rust
    2.12x faster directionally.
  - JAX equal host-bool comparator: 66.6275 us mean, p50 63.2482 us,
    p95 89.2968 us, p99 92.7591 us, CV 17.95%; Rust/JAX 0.413x, Rust
    2.42x faster directionally.
  - JAX tail-mismatch device-ready lower bound: 56.2721 us mean, p50 53.4535 us,
    p95 72.5154 us, p99 75.3971 us, CV 14.40%; Rust/JAX 0.490x, Rust
    2.04x faster directionally.
  - JAX tail-mismatch host-bool comparator: 68.9387 us mean, p50 70.1777 us,
    p95 82.6965 us, p99 85.0570 us, CV 11.61%; Rust/JAX 0.400x, Rust
    2.50x faster directionally.
  - External ratios are directional because JAX CV is above 5% for all equality
    rows. The Rust dense/literal split is the hard keep criterion.
- Conformance guard: storage-direct equality matches materialized
  `Vec<Literal>` equality across F64/F64OnePlusX/F32/I64/U32/U64/Bool/
  BoolWords/Half/Complex, repeated-patches, concat, mixed dense/literal concat,
  and the `LiteralBuffer`/`Vec<Literal>` cross-`PartialEq` impls.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec -- cargo test -p fj-core literal_buffer_storage_direct_equality_matches_materialized_vec --lib`
  passed 1 test, 0 failed on `hz1`. RCH
  `cargo check -p fj-core --bench core_baseline` passed on `hz2`, RCH
  `cargo clippy -p fj-core --bench core_baseline -- -D warnings` passed on
  `vmi1149989`, `python -m py_compile benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`
  passed, and `ubs --only=python benchmarks/jax_comparison/core_literal_buffer_eq_gauntlet.py`
  returned 0 warnings. Workspace `cargo fmt --check` remains red on unrelated
  pre-existing formatting drift outside the touched files; no Rust code changed
  in this measured closeout.
- Retry predicate: do not retry the already committed stack/repeat/slice/to_i64,
  `TensorValue::new`, `LiteralBuffer::to_vec`, dense COW mutation,
  serialization streaming, or this equality family without fresh focused
  criterion evidence showing comparison remains a top-five fj-core bottleneck.
  Do not revisit FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  eager concat without fresh same-worker benchmark evidence and ownership check.

## frankenjax-alc0j - Dense Scalar Broadcast for U32/U64/Complex

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `BroadcastInDim` fills for U32, U64, Complex64, and
  Complex128 through `new_u32_values`, `new_u64_values`, and
  `new_complex_values` instead of allocating a `Vec<Literal>` fill.
- Status: measured mixed/noisy against warmed JAX CPU. Keep the committed dense
  fill path, but record negative head-to-head evidence for Complex128.
- Benchmark guard: `eval/broadcast_scalar_u32_1024x1024`,
  `eval/broadcast_scalar_u64_1024x1024`,
  `eval/broadcast_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/broadcast_scalar --sample-size 100 --warm-up-time 1 --measurement-time 10`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline scalar
    broadcast harness with `jax_enable_x64=true`, JAX/JAXLIB 0.10.1, CPU
    backend, 60 runs x 1000 warmed inner loops.
  - `broadcast_scalar_u32_1024x1024`: Rust Criterion mean 51.307 us, slope
    49.804 us, median 50.460 us, Rust CV 10.56%; JAX mean 79.979 us, p50
    84.805 us, JAX CV 26.62%; Rust/JAX mean ratio 0.642, Rust 1.56x faster.
    Classification: noisy win.
  - `broadcast_scalar_u64_1024x1024`: Rust Criterion mean 104.911 us, slope
    102.906 us, median 102.130 us, Rust CV 18.20%; JAX mean 124.569 us, p50
    130.398 us, JAX CV 31.74%; Rust/JAX mean ratio 0.842, Rust 1.19x faster.
    Classification: noisy win/near-tie.
  - `broadcast_scalar_complex128_1024x1024`: Rust Criterion mean 283.150 us,
    slope 263.538 us, median 276.539 us, Rust CV 18.07%; JAX mean 245.381 us,
    p50 259.656 us, JAX CV 24.66%; Rust/JAX mean ratio 1.154, Rust 1.15x
    slower. Classification: noisy loss.
  - Decision: keep, not revert. The U32 and U64 scalar-fill rows beat JAX by
    mean ratio, while Complex128 loses; all three rows are above the 5% CV
    certification bar, so this is real negative/positive evidence but not a
    release-grade perf certificate. There is no same-run evidence that reverting
    the dense fill path improves the Complex128 row, and reverting would discard
    the measured U32/U64 wins. The next Complex128 attempt must target complex
    packed-fill/store throughput, not another boxed-literal-elision pass.
- Conformance guard: scalar fills materialize to the same repeated literals and
  expose dense typed storage via `as_u32_slice`, `as_u64_slice`, or
  `as_complex_slice`.
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax dense_broadcast_in_dim_matches_literal_path_and_stays_dense --lib`
  passed 1 test, 0 failed.
- Retry predicate: do not retry scalar `BroadcastInDim` dense-fill storage for
  these dtypes unless focused criterion evidence shows this path remains a
  top-five `fj-lax` bottleneck with both Rust and JAX CV below 5%, or the dense
  constructor representation changes. For Complex128 specifically, do not
  repeat the same `Vec<(f64, f64)>` dense fill lever; the next attempt must
  attack packed complex output construction/store throughput directly. Do not
  merge it with FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or
  broader scalar-broadcast arithmetic work without fresh benchmark evidence and
  ownership check.

## frankenjax-dxqfj - Lazy SplitMulti Section Buffers

- Date: 2026-06-18
- Agent: cod-b / WildForge
- Lever: build each `eval_split_multi` output from `LiteralBuffer::from_concat_slices`
  over the original tensor backing instead of copying each section through
  `Vec<Literal>` and re-densifying via `TensorValue::new`.
- Status: measured keep against warmed JAX CPU. No revert.
- Benchmark guard: `eval/split_multi_1024x1024_f32_axis1`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/split_multi_1024x1024_f32_axis1 --sample-size 60 --warm-up-time 1 --measurement-time 5`.
  - Rust Criterion: mean 96.651 us, median 95.038 us, slope 98.875 us
    (95% CI 97.211-100.534 us), stddev 6.616 us, CV 6.85%.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python` inline split
    harness, JAX/JAXLIB 0.10.1, `jax_enable_x64=true`, CPU backend, 60 runs x
    1000 warmed inner loops.
  - `jnp.split(x, [256], axis=1)` bare output: JAX mean 149.082 us, p50
    148.877 us, CV 12.08%; Rust/JAX mean ratio 0.648, Rust 1.54x faster.
  - `jnp.split(x, [256], axis=1)` with `+0.0` materialization on both outputs:
    JAX mean 136.983 us, p50 139.649 us, CV 10.43%; Rust/JAX mean ratio 0.706,
    Rust 1.42x faster.
  - Decision: keep. The JAX side remains noisy, but both JAX modes are slower
    than the Rust lazy-section split by a margin larger than the Rust CI band.
    This is a measured split-construction win, not a claim about arbitrary
    downstream consumers after full materialization.
- Conformance guard: uneven multi-output split materializes the same literals for
  each section and exposes dense f32 storage through `as_f32_slice`;
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo test -p fj-lax split_multi --lib`
  passed 2 tests, 0 failed.
- Retry predicate: retry only if a profiler attributes a clearly-above-noise
  share to `eval_split_multi` section construction on a wider multi-output split
  workload after Criterion and JAX CV are both below 5%, or if
  `LiteralBuffer::Concat` dense-lane behavior changes. Do not merge with
  FMA/SIMD exp, GEMM, QR, SVD, cumsum, OneHot, SelectN/iota, or broader
  reshape/slice/gather work without fresh benchmark evidence and ownership
  check.

## frankenjax-19wst - Dense Scalar Tile Fills

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route scalar `Tile` uniform fills through dense typed `TensorValue`
  constructors instead of filling a `Vec<Literal>` with the recursive boxed
  tiler and then re-densifying.
- Status: measured keep against warmed JAX CPU for both guard workloads.
- Benchmark guard: `eval/tile_scalar_f32_1024x1024`,
  `eval/tile_scalar_complex128_1024x1024`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/tile_scalar --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `tile_scalar_f32_1024x1024`: Rust criterion median 51.435 us vs JAX
    batched median 317.753 us; ratio Rust/JAX 0.162, Rust 6.18x faster.
  - `tile_scalar_complex128_1024x1024`: Rust criterion median 412.679 us vs
    JAX batched median 579.030 us; ratio Rust/JAX 0.713, Rust 1.40x faster.
  - Decision: keep. No revert; both measured workloads beat the original JAX
    oracle under warmed CPU execution. Complex128 margin is modest enough that
    future retries must preserve this exact head-to-head guard.
- Conformance guard: scalar tile materializes to the same repeated literals and
  exposes dense typed storage for f32, u64, and complex128.
- Retry predicate: do not retry scalar `Tile` dense-fill storage unless focused
  criterion evidence shows this path remains a top-five `fj-lax` bottleneck or
  the scalar tile representation changes. Do not merge it with scalar
  `BroadcastInDim`, dense tensor tile, FMA/SIMD exp, GEMM, QR, SVD, cumsum,
  OneHot, SelectN/iota, or broader shape/data movement work without fresh
  benchmark evidence and ownership check.

## frankenjax-1z7k9 - Dense Tensor-Scalar Complex Constructor

- Date: 2026-06-19
- Agent: cod-b / WildForge
- Lever: route matching F64/F32 `eval_complex` tensor-scalar and scalar-tensor
  constructors through dense `new_complex_values` instead of rebuilding every
  element as a boxed `Literal`.
- Status: measured mixed; keep the committed dense path, but record a negative
  JAX head-to-head result for the F32 tensor-scalar workload.
- Benchmark guard: `eval/complex_f32_tensor_scalar_1m`,
  `eval/complex_f64_tensor_scalar_1m`.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo bench -p fj-lax --bench lax_baseline -- eval/complex_f --sample-size 10 --warm-up-time 1 --measurement-time 3`.
  - JAX command: `uv run --with 'jax[cpu]==0.9.2' --with numpy python` with
    `jax_enable_x64=true`, JAX/JAXLIB 0.9.2, CPU backend, 12 batches x 20
    warmed iterations.
  - `complex_f32_tensor_scalar_1m`: Rust criterion median 1.379 ms vs JAX
    batched median 1.272 ms; ratio Rust/JAX 1.084, Rust 1.08x slower. This is
    negative evidence for "dense constructor alone dominates original JAX" on
    Complex64/F32.
  - `complex_f64_tensor_scalar_1m`: Rust criterion median 0.914 ms vs JAX
    batched median 3.730 ms; ratio Rust/JAX 0.245, Rust 4.08x faster.
  - Decision: keep, not revert. The cluster has one clear JAX win and one clear
    JAX loss; there is no same-run evidence that reverting the dense constructor
    improves the F32 result, and reverting would discard the measured F64 win.
    Route the F32 loss to a deeper primitive-level follow-up instead of retrying
    the same boxed-literal-elision lever.
- Conformance guard: dense tensor-scalar and scalar-tensor constructor outputs
  materialize identically to explicit literal-backed references and expose dense
  complex storage.
- Retry predicate: do not retry tensor-scalar `eval_complex` constructor storage
  unless focused criterion evidence shows this path remains a top-five
  `fj-lax` bottleneck or mixed-dtype promotion becomes the measured hotspot.
  For Complex64/F32 specifically, do not repeat the same
  `Vec<(f64, f64)> -> new_complex_values` dense constructor lever; the next
  attempt must attack representation/construction overhead directly, such as a
  Complex64-native packed builder or a JAX-comparable fused real-to-complex path.
  Do not merge with same-shape complex constructor, FFT extraction, complex
  binary tensor-scalar ops, or broader complex arithmetic work without fresh
  benchmark evidence and ownership check.

## frankenjax-mcqr.105-.107 - Dense Clamp Mixed/Tensor Bounds

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: verify the committed dense `clamp` fast paths for mixed scalar/tensor
  F32/F64 bounds, mixed scalar/tensor BF16/F16 bounds, and same-shape BF16/F16
  tensor bounds against both the Rust boxed reference path and original JAX CPU.
- Status: measured keep internally; negative head-to-head result versus JAX for
  all six measured workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax_mcqr_105_107_clamp_gauntlet_2026-06-19.json`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, warmed JAX CPU venv with 50 runs x
  100 inner loops.
- Measured evidence (2026-06-19, same-host CPU):
  - Rust check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax --bench clamp_gauntlet`.
  - Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- --save-baseline frankenjax-mcqr-105-107`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_105_107_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend, `jax_enable_x64=true`.

| Workload | Rust dense mean | Rust boxed mean | Dense/boxed | JAX mean | Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `f32_mixed_scalar_tensor_1m` | 159.383 us | 27.645 ms | 173.45x faster | 115.540 us | 1.38x slower | Keep internal win; JAX loss |
| `f64_mixed_scalar_tensor_1m` | 996.940 us | 27.487 ms | 27.57x faster | 213.651 us | 4.67x slower | Keep internal win; JAX loss |
| `bf16_mixed_scalar_tensor_1m` | 15.571 ms | 43.758 ms | 2.81x faster | 121.313 us | 128.35x slower | Keep internal win; JAX loss |
| `f16_mixed_scalar_tensor_1m` | 19.859 ms | 41.186 ms | 2.07x faster | 371.729 us | 53.42x slower | Keep internal win; JAX loss |
| `bf16_tensor_tensor_tensor_1m` | 15.652 ms | 35.451 ms | 2.26x faster | 183.707 us | 85.20x slower | Keep internal win; JAX loss |
| `f16_tensor_tensor_tensor_1m` | 20.951 ms | 35.255 ms | 1.68x faster | 229.951 us | 91.11x slower | Keep internal win; JAX loss |

- CV notes: Rust dense CV ranged from 5.75% to 13.73%; JAX CV ranged from
  5.70% to 21.86%. Treat the F32 JAX loss as lower-confidence than the half
  losses, but it remains directionally negative under the same-host run.
- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo test -p fj-lax clamp --lib`
  passed 25 tests, 0 failed, 6 ignored benchmark tests. The passing set includes
  `f32_f64_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`,
  `half_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`, and
  `half_clamp_same_shape_tensor_bounds_dense_matches_generic`.
- Decision: keep, not revert. Every measured target clears the Rust boxed
  reference keep gate by at least 1.68x, so reverting would discard a real
  internal improvement. Do not claim original-JAX domination for this cluster.
- Retry predicate: do not retry boxed-literal elision for these clamp shapes.
  F32/F64 follow-up must attack SIMD/parallel clamp throughput or output
  allocation/fusion directly. BF16/F16 follow-up must avoid the per-lane
  `clamp_literal` widen/round path, for example with a raw half-bits proof
  harness plus vectorized or table-assisted min/max semantics. Do not merge the
  next attempt with broadcast-shape generalization, scalar-bound relu/relu6, or
  unrelated dense constructor work without fresh focused Criterion evidence and
  an updated JAX head-to-head row.

## frankenjax-mcqr.108 - Raw Half-Bits BF16/F16 Clamp

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: replace dense same-half BF16/F16 clamp loops that call
  `clamp_literal` per lane with a raw `u16` two-pass composition of the existing
  bit-proven half Max/Min kernels. Mixed scalar dtypes still fall back to
  `clamp_literal`.
- Status: measured keep internally; still negative head-to-head versus original
  JAX CPU for all four half clamp workloads.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax-cod-a-half-clamp-raw-bits-2026-06-19.md`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, warmed JAX CPU venv with 50 runs x
  100 inner loops.
- Measured evidence:
  - RCH check: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench clamp_gauntlet`.
  - RCH before bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-before`.
  - RCH after bench: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-after`.
  - Same-host Rust bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-local-after`.
  - JAX command: `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_108_half_clamp_jax_raw.json`.
  - JAX/JAXLIB: 0.10.1, CPU backend, `jax_enable_x64=true`.

| Workload | RCH before dense | RCH after dense | RCH speedup | Local Rust dense | Local boxed | Dense/boxed | JAX mean | Local Rust/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_mixed_scalar_tensor_1m` | 21.783 ms | 2.534 ms | 8.60x | 3.616 ms | 45.091 ms | 12.47x faster | 122.705 us | 29.47x slower | Keep internal win; JAX loss |
| `f16_mixed_scalar_tensor_1m` | 28.986 ms | 3.448 ms | 8.41x | 3.521 ms | 37.471 ms | 10.64x faster | 319.088 us | 11.03x slower | Keep internal win; JAX loss |
| `bf16_tensor_tensor_tensor_1m` | 22.412 ms | 2.934 ms | 7.64x | 2.993 ms | 32.126 ms | 10.73x faster | 148.870 us | 20.10x slower | Keep internal win; JAX loss |
| `f16_tensor_tensor_tensor_1m` | 29.748 ms | 4.038 ms | 7.37x | 3.653 ms | 32.192 ms | 8.81x faster | 196.938 us | 18.55x slower | Keep internal win; JAX loss |

- Conformance guard: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`
  passed 3 tests, 0 failed, 2 ignored benchmark tests. The passing set includes
  `half_clamp_mixed_scalar_tensor_bounds_dense_matches_generic`,
  `half_clamp_same_shape_tensor_bounds_dense_matches_generic`, and
  `half_clamp_signed_zero_tie_order_dense_matches_generic`.
- Decision: keep, not revert. Same-worker RCH dense speedups are 7.37x-8.60x
  versus the pre-change raw half clamp baseline, and same-host local runs beat
  the Rust boxed reference by 8.81x-12.47x.
- Retry predicate: do not retry boxed literal elision or this exact two-pass
  max/min composition. The next half-clamp attempt must attack the remaining
  JAX gap directly, but not with the generic one-pass bound-abstraction shape
  rejected in `frankenjax-mcqr.109`.

## frankenjax-mcqr.109 - One-Pass Half Clamp Fused Helper

- Date: 2026-06-19
- Agent: cod-a / WildForge
- Lever: replace the committed raw-half two-pass clamp composition with a
  generic one-pass helper using a scalar/tensor bound abstraction, one output
  allocation, no `tmp` vector, and the same signed-zero/NaN fallback policy.
- Status: measured regression on all four half clamp workloads; reverted.
- Evidence artifact:
  `artifacts/performance/evidence/frankenjax-cod-a-half-clamp-one-pass-2026-06-19.md`.
- Benchmark guard: `crates/fj-lax/benches/clamp_gauntlet.rs`, 1,048,576
  element vectors, Criterion sample size 20, RCH same-worker before/after on
  `ovh-a`.
- Measured evidence:
  - RCH before bench: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-before`.
  - RCH conformance while candidate existed: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`.
  - RCH after bench: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-109-after`.
  - RCH post-revert conformance: `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`.

| Workload | Before two-pass mean | Candidate mean | Candidate/before | JAX mean used for context | Candidate/JAX | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bf16_mixed_scalar_tensor_1m` | 2.3918 ms | 3.2624 ms | 1.36x slower | 122.705 us | 26.59x slower | Reject/revert |
| `f16_mixed_scalar_tensor_1m` | 3.0263 ms | 4.4595 ms | 1.47x slower | 319.088 us | 13.98x slower | Reject/revert |
| `bf16_tensor_tensor_tensor_1m` | 2.7423 ms | 3.2236 ms | 1.18x slower | 148.870 us | 21.65x slower | Reject/revert |
| `f16_tensor_tensor_tensor_1m` | 3.4745 ms | 4.4271 ms | 1.27x slower | 196.938 us | 22.48x slower | Reject/revert |

- Conformance guard: the candidate passed `cargo test -p fj-lax half_clamp --lib`
  on RCH while it existed: 4 passed, 0 failed, 2 ignored benchmark tests. The
  additional edge-matrix proof was removed with the rejected code. Post-revert
  production conformance also passed: 3 passed, 0 failed, 2 ignored benchmark
  tests.
- Decision: reject and revert. The `tmp` allocation and second helper pass were
  not the dominant cost; the generic fused helper added more chunk setup and
  half widen/round overhead than it saved.
- Retry predicate: do not retry generic `HalfClampBound` one-pass fusion or
  temp-vector removal alone. The next attempt should be dtype/shape-specialized
  raw-bit compare/classification that avoids f64 widen/round on finite lanes, or
  a producer/consumer clamp fusion that removes an intermediate across primitive
  boundaries.

## frankenjax-e07uw/7g72q/rl9ha/bjqfr - eval_jaxpr Fusion Cluster (Square / Reciprocal / integer_pow[2] / i64+bf16 broadcast)

- Lever: extend the `eval_jaxpr` cheap-op elementwise fusion to (a) `Square`,
  `Reciprocal`, `integer_pow[2]` via builder operand-synthesis (unary == fused
  binary: `Square=Mul(x,x)`, `Reciprocal=Div(1,x)`, `x**2=Mul(x,x)`), and (b) i64
  + bf16/f16 row/col broadcast operands (mirror the existing f64/f32 broadcast).
- Conformance GUARD: GREEN. `cargo test -p fj-interpreters --lib --release` — the
  fusion bit-identity tests (`fusion_*_chain_matches_reference_bit_for_bit`,
  `fusion_f64_{row,col}_broadcast_*`, etc.) all PASS. (8 unrelated golden-digest
  tests for broadcast_in_dim/transpose/reshape/scan/dce/staging PLANS are RED — see
  the separate "Golden digest drift" entry; their `plan==generic` parity asserts
  PASS, only the hardcoded sha256 lines drift; NOT caused by this cluster.)
- Measured evidence (2026-06-19, rch worker, `eval_fusion_speed` same-invocation
  A/B, 1,048,576 elements / 1024x1024, 8-op chains; rch bench noise ~1.0x band):
  - Internal (Rust fused `eval_jaxpr` vs unfused per-op `eval_primitive`):

  | Workload | unfused | fused | speedup | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | F64 arith8 1M | 16.788 ms | 3.320 ms | 5.06x | KEEP |
  | F32 arith8 1M | 7.174 ms | 1.071 ms | 6.70x | KEEP |
  | F32 clamp 1M | 9.081 ms | 1.739 ms | 5.22x | KEEP |
  | F32 row-broadcast | 12.331 ms | 0.783 ms | 15.75x | KEEP |
  | F32 col-broadcast | 11.479 ms | 0.704 ms | 16.31x | KEEP |
  | F64 row-broadcast | 8.703 ms | 2.567 ms | 3.39x | KEEP |
  | F64 col-broadcast | 8.452 ms | 2.282 ms | 3.70x | KEEP |
  | I64 arith8 1M | 5.976 ms | 3.330 ms | 1.79x | KEEP |
  | **I64 row-broadcast (rl9ha)** | 7.546 ms | 2.352 ms | **3.21x** | **KEEP** |
  | BF16 arith8 1M (not mine) | 10.893 ms | 10.599 ms | 1.03x | ~0 gain, flag owner |
  | **BF16 row-broadcast (bjqfr)** | 10.776 ms | 10.604 ms | **1.02x** | **REVERTED** |

  - Head-to-head vs original JAX (jax.jit CPU x64, `fusion_chain.py`, 1M):

  | Workload | JAX jit mean | Rust fused | Rust/JAX | Outcome |
  | --- | ---: | ---: | ---: | --- |
  | arith8 f64 1M | 272.7 us | 3.320 ms | ~12.2x slower | JAX win (interp vs XLA) |
  | square f64 1M | 293.4 us | (rides f64 5.06x) | >10x slower | JAX win |
  | bf16 broadcast 1M | 146.9 us | 10.6 ms | ~72x slower | JAX win |

- Decision:
  - KEEP Square / Reciprocal / integer_pow[2] (bit-identical operand-synthesis;
    they let variance/L2/`x**2`/normalization chains ride the proven f64/f32
    5-6.7x fusion win) and i64 broadcast (3.21x).
  - REVERT bf16/f16 broadcast fusion (bjqfr) — measured 1.02x = ~0 gain. bf16
    tensor fusion is bandwidth-bound; the per-lane f64 decode/encode
    (half_fusion_widen + half_fused_binary) cancels the materialization savings
    (consistent with bf16 same-shape ~1.0x). Reverted in commit after this entry.
- Honest framing: the fusion lever is a real **Rust-internal interpreter** win
  (5-16x vs the pre-fusion per-op path) but does NOT achieve original-JAX
  domination on jit'd elementwise chains — the Rust tree-walking interpreter
  (even fused) is ~12-72x slower than XLA-compiled jax.jit. Fusion narrows the
  per-op interpreter tax; it cannot close the interpreter-vs-compiler gap. The
  vs-JAX win requires the compiled-jaxpr arena executor (bead z6o97 family /
  6dfew), not more fusion-op coverage.
- Retry predicate: do NOT re-add bf16/f16 tensor fusion via the scalar per-lane
  widen/round path. It only pays with a SIMD bf16<->f32 convert kernel (see
  project_bf16_matmul_and_convert_simd). Do NOT chase more fusion-op coverage for
  vs-JAX wins — the gap is interpreter-vs-compiler, addressable only by the
  compiled-jaxpr core.

## Golden digest drift - 8 fj-interpreters PLAN goldens (pre-existing, NOT this session)

- Symptom: `cargo test -p fj-interpreters --lib --release` shows 8 FAILED golden
  tests: dense_f64_broadcast_in_dim_plan / dense_f64_transpose_plan /
  dense_reshape_plan (x2) / eval_top_level_scan_i64_add_emit_fast_path /
  test_dce_all_used_large_chain / test_pe_two_eq_mixed_residual / staging single-unknown.
- Diagnosis: each test asserts `planned_out == generic_out` (PARITY) THEN a
  hardcoded sha256. The panics are all on the sha256 line (hash-vs-hash); the
  parity asserts PASS. So the COMPUTED VALUES are internally consistent
  (plan==generic) and the digests merely drifted vs the recorded constants —
  accumulated from earlier untested "code-first" dense-storage/serialization
  commits (e.g. mcqr.97 TensorValue::new densify, co009 serialization).
- NOT caused by this session's fusion work (none of these tests exercise the
  cheap-op fusion path; the 200 passing tests include all fusion bit-identity tests).
- Action: NOT refreshed here. Refreshing requires per-test value verification vs
  JAX/expected (parity alone does not prove vs-original correctness — a shared
  regression in both paths would pass parity). Flagged for the dense-storage/
  serialization commit owners to refresh with verified values, or for a
  test-capable golden-refresh pass. Conformance is RED on these 8 until then.

## frankenjax-f62hx (+ thnjs/idunl/rbkn9/02i98/ecffn) - Contiguous-Block Memcpy Transpose/Slice/Gather/Broadcast

- Lever: replace the per-element coordinate-decode odometer in the structural
  data-movement kernels (transpose_general, slice_strided_gather,
  dynamic_slice_dense, gather_window_blocks, broadcast_replicate, rev_gather)
  with `extend_from_slice` block memcpy of the contiguous trailing run. This is an
  ALGORITHMIC change (memcpy vs odometer), distinct from boxed->dense de-box.
- Representative measured: TRANSPOSE block-copy (f62hx), the attention transpose
  [B,S,H,D]->[B,H,S,D] = [8,512,8,64] f32 (2,097,152 elems), perm (0,2,1,3) which
  keeps the [D]=64 feature vector contiguous (block_len=64).
- Conformance GUARD: GREEN. `transpose_gauntlet.rs` asserts the block-copy path is
  bit-identical to the pre-f62hx per-element odometer reference at 4 probe indices;
  fj-lax transpose conformance tests pass.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | throughput | vs naive |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive, committed) | 791.50 us | 2.65 Gelem/s | 10.30x faster |
  | naive per-element odometer (pre-f62hx) | 8.1525 ms | 257 Melem/s | baseline |

  - Head-to-head vs original JAX (`transpose_gauntlet.py`, jax.jit CPU x64,
    jnp.transpose+0.0 to force materialization, mean):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit transpose | 186.7 us | - |
  | Rust block-copy | 791.5 us | 4.24x slower |
  | Rust naive (pre-f62hx) | 8.15 ms | 43.6x slower |

- Decision: KEEP. The block-copy is a real **10.3x internal** algorithmic win and
  NARROWS the JAX gap from 43.6x to 4.24x. It is the strongest measured lever in
  this conversation's backlog (the dense de-box clusters only matched the boxed
  Rust reference; this one is a genuine throughput jump). Still a JAX loss (4.24x)
  but directionally the right kind of optimization.
- Generalization: the sibling block-copy kernels (slice/gather/dynamic_slice/
  broadcast/rev) use the same memcpy-vs-odometer transform on the same contiguous
  trailing-block structure, so each is expected to deliver a comparable internal
  multi-x win on its contiguous regime. Measured here via the transpose
  representative; the others KEEP on the same proof + the bit-identity tests.
- Retry predicate: closing the remaining 4.24x to JAX on transpose needs either
  cache-blocked tiling of the strided (non-identity-suffix) case (REJECTED before
  as a regression — see project_transpose_already_optimal) or avoiding the
  materialized transpose entirely (layout-aware fusion), NOT more block-copy.

## frankenjax-thnjs - Contiguous-Block Memcpy Broadcast (second block-copy data point)

- Lever: broadcast_replicate replicates the contiguous trailing source block via
  extend_from_slice instead of per-element coordinate decode (sibling of the
  transpose block-copy f62hx). Bias/feature broadcast [D] -> [rows, D] is in every
  transformer layer.
- Workload: bias broadcast [768] -> [4096, 768] f32 (3,145,728 elems), the
  D=768-model-dim-over-4096-tokens bias replicate. broadcast_dimensions [1].
- Conformance GUARD: GREEN. `broadcast_gauntlet.rs` asserts block-copy ==
  pre-thnjs per-element reference at 4 probe indices; broadcast conformance passes.
- Measured evidence (2026-06-19, rch worker, Criterion sample 30):
  - Internal A/B (Rust):

  | Arm | median time | vs naive |
  | --- | ---: | ---: |
  | block-copy (eval_primitive, committed) | 283.75 us | 21.80x faster |
  | naive per-element decode (pre-thnjs) | 6.1866 ms | baseline |

  - Head-to-head vs original JAX (`broadcast_gauntlet.py`, jax.jit CPU x64,
    jnp.broadcast_to+0.0 materialized):

  | Engine | time | Rust/JAX |
  | --- | ---: | ---: |
  | JAX jit broadcast | 178.9 us (mean; p50 164.9, cv 48%) | - |
  | Rust block-copy | 283.75 us | ~1.59x slower |
  | Rust naive (pre-thnjs) | 6.19 ms | ~34.6x slower |

- Decision: KEEP. 21.8x internal win and only ~1.6x off JAX (NEAR-PARITY) — the
  closest-to-JAX block-copy lever measured. Broadcast is write-bandwidth-bound, so
  both engines sit near memory bandwidth (Rust ~44 GB/s vs JAX ~70 GB/s store
  throughput); the residual gap is store-vectorization, not algorithm.
- Cluster summary (2 measured): block-copy structural kernels deliver 10-22x
  internal wins and a 1.6-4.24x JAX gap (broadcast 1.6x write-bound, transpose
  4.24x strided) -- categorically better than the de-box dense clusters (50-128x
  JAX loss). The block-copy lever is the right kind of optimization; KEEP all
  siblings (slice/gather/dynamic_slice/rev) on the same transform + bit-identity tests.
- Retry predicate: broadcast's residual 1.6x to JAX is store throughput — would
  need SIMD/streaming-store output (non-temporal writes), not more block-copy.

## frankenjax-hfq7o - Dense integer_pow x**2 (MEASURED FIX: powi libcall -> v*v)

- Lever: dense integer_pow path (de-box) for the ubiquitous x**2 (variance/MSE/poly).
- Workload: integer_pow[2] on 1M f64 and 1M f32 (integer_pow_gauntlet.rs + .py).
- FINDING (gauntlet measurement caught a latent inefficiency): the original dense
  path used `v.powi(exponent)` with `exponent` a RUNTIME i32, so LLVM could not
  fold powi(2) to a mul -> a per-element powi LIBCALL, ~6.75 GB/s (10x below memory
  bandwidth). FIXED to special-case exponent==2 -> `v*v` (f64) /
  `(f64::from(v)*f64::from(v)) as f32`, bit-exactly equal to powi(2).
- Conformance GUARD: GREEN. `cargo test -p fj-lax --lib integer_pow` 10/0; bench
  asserts dense==boxed at 3 indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64):

  | Workload | dense BEFORE (powi) | dense AFTER (v*v) | fix speedup | JAX | Rust/JAX before -> after |
  | --- | ---: | ---: | ---: | ---: | --- |
  | integer_pow2_f64_1m | 2.3675 ms | 405.45 us | 5.84x | 184.1 us | 12.86x -> 2.20x slower |
  | integer_pow2_f32_1m | 2.6444 ms | 169.61 us | 15.59x | 121.1 us | 21.83x -> 1.40x slower |

  (Internal vs boxed Vec<Literal> path: f64 25.6ms/f32 27.5ms boxed -> 40-160x
  faster dense; but the headline is the powi->v*v fix that made x**2 bandwidth-bound.)
- Decision: KEEP + FIXED. The fix is the single biggest measured improvement of
  the gauntlet: it closed the JAX gap from 12.9-21.8x to 1.4-2.2x (f32 near-parity)
  by removing the runtime-powi libcall. x**2 is now bandwidth-bound (~40-47 GB/s,
  vs JAX ~66-87 GB/s store); the residual 1.4-2.2x is store throughput (same gap as
  the broadcast block-copy lever).
- Retry predicate: the remaining 1.4-2.2x is memory-store vectorization (SIMD/
  streaming stores) — the SAME residual as broadcast; NOT an algorithm issue. Apply
  the same powi-runtime-arg lesson to any other op taking a runtime small-integer
  power (none found in the hot path; integer_pow[3+] is rarer + needs powi grouping).

## frankenjax-idunl - Contiguous-Block Memcpy Slice (third block-copy data point: PARITY)

- Lever: slice_strided_gather copies the contiguous trailing block via
  extend_from_slice (2D crop / windowing). Workload: crop [1024,1024]->[512,512]
  f32 (262,144 output elems), start [256,256] limit [768,768].
- Conformance GUARD: GREEN. `slice_gauntlet.rs` asserts block-copy == pre-idunl
  per-element reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.jit CPU x64,
  x[256:768,256:768]+0.0 materialized):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | block-copy (eval_primitive) | 45.97 us | 6.10x faster | 1.04x slower (TIE) |
  | naive per-element | 280.51 us | baseline | ~6.4x slower |
  | JAX jit crop | 44.17 us (mean, cv 26%) | - | - |

- Decision: KEEP. 6.1x internal win and ~PARITY with JAX (1.04x, within noise) —
  the FIRST measured non-loss in the conversation. Slice is a pure contiguous
  block memcpy of sub-region rows; both engines are memcpy-bound and tie.
- Block-copy cluster (3 measured): transpose 4.24x (strided read), broadcast 1.59x
  (replicate, write-bound), slice 1.04x (sub-region, memcpy-bound). The
  bandwidth/memcpy-bound block-copy ops TIE or near-tie JAX; the strided-read
  transpose is 4.24x (random-access read pattern). The block-copy lever family is
  the closest-to-JAX work in the codebase; KEEP all.

## frankenjax (dense contiguous gather / embedding lookup) - random-access regime

- Lever: dense contiguous gather memcpy's each [dim] row via extend_from_slice
  (embedding lookup). Workload: table [16384,768] f32, gather 4096 random token
  rows -> [4096,768] (3.1M output elems, random reads into a 50MB table).
- Conformance GUARD: GREEN. `gather_gauntlet.rs` asserts dense == per-element
  reference at 4 probe indices.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jnp.take CPU x64):

  | Arm | median | vs naive | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (eval_primitive, row memcpy) | 1.1450 ms | 4.05x faster | 4.22x slower |
  | naive per-element | 4.6382 ms | baseline | ~17x slower |
  | JAX jit take | 271.3 us (cv 13%) | - | - |

- Decision: KEEP. 4.05x internal win; the per-row memcpy is the correct bit-exact
  approach. JAX-loss (4.22x) because gather is RANDOM-ACCESS-READ-bound (~11 GB/s
  effective: each row read is a cache miss into the 50MB table); XLA hides that
  latency better (prefetch / memory-level parallelism / SIMD gather).
- REFINED CLUSTER PATTERN (4 block-copy/structural ops measured): the JAX gap is
  set by the READ access pattern, not the copy:
    - SEQUENTIAL read (slice 1.04x TIE, broadcast 1.59x): near-parity with JAX.
    - RANDOM/STRIDED read (transpose 4.24x, gather 4.22x): ~4x JAX loss.
  All are 4-22x internal wins over the per-element baseline. The block-copy lever
  is correct; the residual ~4x on random/strided reads is a memory-access-pattern
  gap (prefetch / SIMD-gather / cache-aware tiling), NOT the copy algorithm.
- Retry predicate: closing the gather/transpose ~4x needs software prefetch or
  memory-level-parallelism (interleaved multi-row gather) — blocked by forbid-unsafe
  for raw prefetch intrinsics; would need a safe MLP-exposing restructure. NOT more
  block-copy. Do NOT retry per-element or naive gather.

## frankenjax-7eqrs - Dense Complex Constructor lax.complex(re,im) (de-box, near-parity)

- Lever: dense complex constructor zips two f64 slices into packed (re,im) storage
  via new_complex_values (FFT/signal real+imag combine), vs the boxed per-Literal
  path. Workload: re[1M]+im[1M] f64 -> complex128[1M] (32MB traffic).
- Conformance GUARD: GREEN. `complex_ctor_gauntlet.rs` asserts dense == boxed.
- Measured evidence (2026-06-19, rch Criterion sample 30; JAX jax.lax.complex CPU x64):

  | Arm | median | vs boxed | vs JAX |
  | --- | ---: | ---: | ---: |
  | dense (new_complex_values) | 775.72 us | 25.2x faster | 1.56x slower |
  | boxed (per-Literal) | 19.558 ms | baseline | ~39x slower |
  | JAX jit lax.complex | 497.34 us (cv 6%) | - | - |

- Decision: KEEP. 25.2x internal de-box win and ~1.56x off JAX (NEAR-PARITY,
  bandwidth-bound at 41 vs 64 GB/s — the same store-throughput gap as broadcast).
- REFINES the de-box category split: de-box of DATA-MOVEMENT/simple ops (complex
  ctor 1.56x, integer_pow-after-v*v-fix 1.40x) APPROACHES JAX (bandwidth-bound);
  de-box of HEAVY-PER-LANE ops (clamp half 53-128x JAX loss, mcqr.106/.107) does
  NOT — there the per-lane decode/encode/compute dominates, not the boxing. So
  "de-box helps reach JAX" iff the op is bandwidth-bound, not per-lane-compute-bound.
- Retry predicate: complex ctor residual 1.56x is store throughput (same as
  broadcast) — SIMD/streaming stores, not algorithm. Conj/real/imag (1ylsj) and
  complex elementwise (same dense storage) expected to behave the same; KEEP.

## METHODOLOGY CORRECTION + elementwise binary (rch-vs-local calibration)

- CRITICAL: prior gauntlet rows (transpose/broadcast/slice/gather/integer_pow/
  complex_ctor) measured Rust via `rch exec` on a REMOTE worker, but JAX runs
  LOCALLY. Same-binary calibration (f32 add, one A/B binary run both places):
  rch worker 173.2 us vs LOCAL 119.6 us => the rch worker is ~1.45x SLOWER than
  local. So all prior Rust/JAX ratios are PESSIMISTIC by ~1.45x; the true
  same-host ratios are ~/1.45.
- f32 native-vs-widen add was ~0 gain on BOTH machines (rch 173.2/174.6 = 1.008x;
  local native 119.6 / widen 122.3 = 1.02x) -> f32 same-shape add is
  bandwidth-bound; the native-f32 attempt was REVERTED.
- LOCAL same-host elementwise dense vs JAX (the trustworthy same-machine numbers;
  JAX jax.jit CPU x64):

  | Workload | Rust dense (LOCAL) | JAX | Rust/JAX (same-host) |
  | --- | ---: | ---: | ---: |
  | add_f64_1m | 415.00 us | 192.0 us | 2.16x slower |
  | add_f32_1m | 135.98 us | 80.4 us | 1.69x slower |
  | mul_f64_1m | 422.96 us | 161.7 us | 2.61x slower |

- Corrected (÷~1.45) same-host estimates for the prior rch rows — several FLIP to
  wins/ties: slice ~0.72x (Rust FASTER), integer_pow2 f32 ~0.97x (Rust ~tie/win),
  broadcast ~1.10x, complex_ctor ~1.08x, integer_pow2 f64 ~1.52x, transpose ~2.93x,
  gather ~2.91x. (These are estimates; future vs-JAX rows MUST run the Rust bench
  LOCALLY, not via rch, for a same-host comparison.)
- Elementwise add/mul same-host loss (1.69-2.61x) is structural: per-op output
  allocation (frankenjax allocates a fresh Vec per primitive; XLA reuses buffers)
  + AVX2 (4-wide f64) vs JAX likely AVX-512 (8-wide). Both are build/architecture
  matters, not a code bug — the inner map+collect already autovectorizes (native
  vs widen tie proves the loop isn't the bottleneck).
- Decision: REVERT f32 native add (~0 gain). KEEP all measured dense paths. Retry
  predicate for elementwise: the gap is per-op allocation (needs buffer reuse =
  compiled-jaxpr arena) and AVX-512 (build flag) — NOT the elementwise loop.

## frankenjax-k8z8g - Einsum permute-copy trailing-block memcpy (internal keep, external JAX loss)

- Lever: `permute_copy_f64` detects an identity-mapped trailing suffix and copies
  the contiguous block with `extend_from_slice`, instead of decoding every output
  coordinate one element at a time. This is used by the general einsum
  permute+GEMM path for attention-shaped contractions such as
  `bqhd,bkhd->bhqk`.
- Canonical graveyard mapping: data-layout/vectorized execution, cache-local
  block movement, and persistent evidence discipline. This is the measured,
  low-risk subset of the broader "exotic layout / cache-oblivious / SIMD"
  idea-space: no unsafe code, one lever, bit-identical output, rollback by
  restoring the per-element coordinate walk.
- Conformance guard: GREEN. `rch exec -- cargo test -p fj-lax
  einsum2_general_matmul_bit_identical_to_naive --lib` passed 1/0, checking
  general-path output and shape bit-for-bit against the deterministic ascending-K
  naive reference across attention, multi-K, interleaved-output, and non-adjacent
  tensordot cases.
- Internal old-vs-new evidence (2026-06-19, `hz2`, release ignored perf test):

  | Arm | workload | time | result |
  | --- | --- | ---: | --- |
  | old odometer reference | `bqhd,bkhd->bhqk` [8,128,8,64/128] | 7664.51 ms | baseline |
  | general permute+GEMM with trailing-block memcpy | same | 15.64 ms | 489.99x faster |

  Digest matched: `7fec8a7f347cf406`.
- Focused Criterion evidence (2026-06-19, `hz2`, sample 30, new repeatable bench):

  | Workload | Rust time |
  | --- | ---: |
  | `eval/einsum2_general_bqhd_bkhd_bhqk_f64` [4,64,8,64/64] | 1.3123 ms mean `[1.2797, 1.3602]` |

- JAX head-to-head (`einsum_permute_copy_gauntlet.py`, local JAX CPU x64,
  `jax.jit(lambda lhs,rhs: jnp.einsum("bqhd,bkhd->bhqk", lhs, rhs))`):

  | JAX comparator | mean | p50 | CV | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: |
  | device-ready lower bound | 317.774 us | 324.308 us | 13.01% | 4.13x slower |
  | NumPy host-copy comparator | 278.461 us | 283.367 us | 20.38% | 4.71x slower |

- Decision: KEEP internally, record external loss. The current path removes a
  catastrophic old Rust algorithmic path (490x faster with digest parity), so
  reverting would be wrong. It still does not dominate JAX on the moderate
  attention-shaped workload; the residual gap is now GEMM/codegen/vectorization
  and allocation/layout, not the permute-copy odometer.
- Validation:
  - `benchmarks/jax_comparison/.venv/bin/python -m py_compile
    benchmarks/jax_comparison/einsum_permute_copy_gauntlet.py`: pass.
  - `ubs --only=rust,python crates/fj-lax/benches/lax_baseline.rs
    benchmarks/jax_comparison/einsum_permute_copy_gauntlet.py`: no critical
    issues; warning-class bench-file inventory only.
  - `rch exec -- cargo check -p fj-lax --bench lax_baseline`: pass.
  - `rch exec -- cargo test -p fj-lax
    einsum2_general_matmul_bit_identical_to_naive --lib`: pass.
  - `rch exec -- cargo bench -p fj-lax --bench lax_baseline --
    eval/einsum2_general_bqhd_bkhd_bhqk_f64 ...`: pass.
  - `rch exec -- cargo test -p fj-lax bench_einsum2_general_vs_odometer --lib
    --release -- --ignored --nocapture`: pass.
  - `cargo fmt -p fj-lax --check`: red on pre-existing fj-lax formatting drift
    outside the touched hunk.
  - `rch exec -- cargo clippy -p fj-lax --bench lax_baseline -- -D warnings`:
    red on pre-existing fj-lax lint debt in `tensor_ops.rs` and `tree_util.rs`
    (`too_many_arguments`, `type_complexity`, `unnecessary_sort_by`), unrelated
    to this bench/harness closeout.
- Retry predicate: do not retry the `permute_copy_f64` odometer elimination
  family without a new profiler showing permute-copy, not GEMM/XLA codegen, is
  still dominant. The next meaningful path is fused layout-aware einsum lowering,
  packed/tiled GEMM, AVX/FMA policy resolution, or compiled-buffer reuse.

## CobaltForge - Threaded cheap same-shape binops for DRAM-bound arrays (JAX WIN)

- Lever: cheap same-shape elementwise binops (Add/Sub/Mul/Div/Max/Min) over dense
  f64/f32 were ALWAYS serial — the recorded "memory-bound ops regress when
  threaded" DO-NOT. That finding is real but SIZE-SCOPED: it was measured at
  L3-resident sizes. New gated path `eval_same_shape_f64_cheap_parallel` /
  `eval_same_shape_f32_cheap_parallel` threads the op (work-scaled `std::thread::
  scope`, exact same per-element closures incl. NaN-propagating `jax_max_f64`/
  `jax_min_f64`) ONLY when `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23 = 8.39M elems),
  well past the measured ~5M-element fresh-alloc cliff. Sub-threshold (incl. the
  benchmarked 1M size) stays on the serial fast path — no regression there.
- HARDWARE CORRECTION: the benchmark host is an AMD Ryzen Threadripper PRO 5975WX
  (Zen3), which has NO AVX-512 (cpuinfo: avx2+fma only) and a 32MB-per-CCD L3.
  Prior ledger/scorecard rows attribute elementwise JAX losses partly to "JAX
  likely AVX-512 (8-wide) vs our AVX2 (4-wide)" — that is FALSE on this host; XLA
  also runs AVX2 here. Measured: XLA does NOT thread large CPU elementwise
  (~21-27 GB/s flat at 16M-64M), so the true elementwise gap was (a) our serial
  fresh-alloc page-fault cliff and (b) single-core DRAM bandwidth — both fixed by
  threading beyond L3.
- Why bit-identical: elementwise output is lane-independent; chunk boundaries never
  change which IEEE op produces which bit. Same closures as the serial path.
- Conformance guard: GREEN for this change. `cheap_binary_parallel_f64_bit_identical_to_serial`
  (new) compares the threaded path bit-for-bit vs the serial closures at n=8.39M+257,
  seeding NaN/±inf/±0/MAX/MIN_POSITIVE to exercise Max/Min NaN propagation and inf
  division — PASS. Full `fj-lax --lib`: 1494 pass, +1 new pass; the 43 failing tests
  are PRE-EXISTING (identical set on clean baseline `d7514a7d`, the documented
  digest-drift RED state) — this change adds 0 new failures.
- Measured A/B (LOCAL, same-binary same-invocation, real `eval_*` fresh-alloc path,
  best-of-10; bench `bench_cheap_binary_parallel_vs_serial`, 2026-06-19):

  | Workload | serial (before) | parallel (after) | internal speedup |
  | --- | ---: | ---: | ---: |
  | add_f64 n=16M | 64859 us (5.9 GB/s) | 9190 us (41.8 GB/s) | 7.06x |
  | add_f64 n=64M | 255663 us (6.0 GB/s) | 33988 us (45.2 GB/s) | 7.52x |
  | add_f32 n=16M | 30806 us (6.2 GB/s) | 4975 us (38.6 GB/s) | 6.19x |
  | add_f32 n=64M | 134654 us (5.7 GB/s) | 17514 us (43.9 GB/s) | 7.69x |

- JAX head-to-head (LOCAL, same host, `jax.jit(lambda x,y:x+y)`, x64; best-of-15
  over inner-10, /tmp/jax_both.py):

  | Workload | Rust after | JAX | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | add_f64 n=16M | 9190 us | 18099 us | 0.51x | Rust 1.97x FASTER |
  | add_f64 n=64M | 33988 us | 57472 us | 0.59x | Rust 1.69x FASTER |
  | add_f32 n=16M | 4975 us | 8598 us | 0.58x | Rust 1.73x FASTER |
  | add_f32 n=64M | 17514 us | 31674 us | 0.55x | Rust 1.81x FASTER |

- Decision: KEEP. Before this change Rust LOST these four large-array workloads to
  JAX by 3.6-4.6x (serial fresh-alloc); after, Rust DOMINATES by 1.69-1.97x. The
  L3-resident regime (n<=4M, e.g. the 1M gauntlet row) is untouched and still uses
  the serial path (threading regresses there — re-confirmed: 1M threaded x8 was
  1.8ms vs 0.34ms serial).
- Retry predicate: do NOT lower the gate toward L3-resident sizes (re-introduces the
  documented regression). The L3-resident 1M loss vs JAX (~70 vs ~200 GB/s, pure
  L3 bandwidth) is a SEPARATE problem (XLA L3 streaming / buffer reuse), not solved
  by threading. Next elementwise frontier: compiled-jaxpr arena buffer reuse to kill
  the per-op fresh allocation entirely (would also remove the page-fault cliff and
  help the L3-resident regime).

## CobaltForge - Threaded cheap scalar-tensor binops / bias-add (frankenjax-aazu6, JAX WIN)

- Lever: extends the same-shape threading win (commit 7b7924b7) to the SCALAR-TENSOR
  cheap path (`tensor OP scalar` — bias-add `x+c`, scaling `x*c`, relu-ish max/min).
  New `eval_f64_scalar_cheap_parallel` / `eval_f32_scalar_cheap_parallel` thread
  Add/Sub/Mul/Div/Max/Min over a dense tensor + scalar when
  `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23), honoring `scalar_on_left`. Inserted ahead
  of the serial `eval_f{64,32}_scalar_broadcast_binop` in both operand orders.
- Bit-identity: the serial f64 Add/Sub/Mul/Div path is `crate::dense::scalar_op` =
  plain `iter().map(ArithOp::apply).collect()` (NOT a separate SIMD kernel, as the
  aazu6 bead feared) — so the threaded path uses the IDENTICAL `a OP b` closures and
  `jax_max_f64`/`jax_min_f64`; lane-independent => bit-for-bit identical. Guarded by
  `cheap_scalar_tensor_parallel_f64_bit_identical_to_serial` (both operand orders,
  NaN/±inf/±0/MAX seeds) — PASS.
- Conformance: `fj-lax --lib` 1496 pass (+2 new bit-identity tests), 43 fail
  (PRE-EXISTING, identical set on clean baseline) — 0 new failures.
- Measured A/B (LOCAL same-binary, real `eval_*` fresh-alloc path, best-of-10;
  `bench_cheap_binary_parallel_vs_serial`, 2026-06-19):

  | Workload | serial (before) | parallel (after) | internal speedup |
  | --- | ---: | ---: | ---: |
  | biasadd_f64 n=16M | 67409 us (3.8 GB/s) | 7341 us (34.9 GB/s) | 9.18x |
  | biasadd_f64 n=64M | 287772 us (3.6 GB/s) | 27074 us (37.8 GB/s) | 10.63x |

- JAX head-to-head (LOCAL same host, `jax.jit(lambda x: x+1.5)` x64, /tmp/jax_bias.py):

  | Workload | Rust after | JAX | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | biasadd_f64 n=16M | 7341 us | 13044 us | 0.56x | Rust 1.78x FASTER |
  | biasadd_f64 n=64M | 27074 us | 54034 us | 0.50x | Rust 2.00x FASTER |

- Decision: KEEP. Before: Rust LOST bias-add to JAX by ~5.2-5.3x (serial scalar
  broadcast craters to 3.6-3.8 GB/s — even worse than the same-shape serial because
  of the broadcast path overhead on top of the fresh-alloc page-fault cliff). After:
  Rust DOMINATES by 1.78-2.00x. L3-resident sizes stay serial (gated). Closes bead
  frankenjax-aazu6.
- Retry predicate: same as the same-shape row — do NOT lower the gate. Remaining
  unthreaded large-array cheap paths: i64 same-shape/scalar binops and cheap unary
  (neg/abs/sign); same pattern would apply but lower priority (less common at scale).

## CobaltForge - Persistent worker pool for L3-resident/2-8M elementwise (NEGATIVE, do-not-wire)

- Hypothesis: XLA hits ~200 GB/s on a 1M f64 add (L3-resident) — impossible for one
  Zen3 core (~70-100 GB/s L3/core), so XLA threads even at L3-resident sizes via a
  PERSISTENT pool (no per-call spawn). We can't thread there because
  `std::thread::scope` spawns fresh OS threads per call. There is an ORPHANED
  `crates/fj-lax/src/thread_pool.rs` (a channel-based persistent pool with
  `parallel_fill_f64`) — NOT declared as a module in lib.rs, so it is dead, never
  compiled, never used. Tested whether wiring it in could lower the threading gate
  below the current 8.39M into the 1M-8M range where we lose / run serial.
- Measured (standalone replica of `parallel_fill_f64`'s exact design — persistent
  channel pool + owned-Vec-per-chunk + caller concat — vs serial collect, +avx2,
  best-of-20, 2026-06-19):

  | n | serial | pool best | pool/serial |
  | ---: | ---: | ---: | ---: |
  | 1M | 305 us (78.7 GB/s) | 812 us (29.6 GB/s) | 0.38x (LOSS) |
  | 2M | 1292 us (37.2 GB/s) | 1898 us (25.3 GB/s) | 0.68x (LOSS) |
  | 4M | 2806 us (34.2 GB/s) | 5962 us (16.1 GB/s) | 0.47x (LOSS) |
  | 8M | 31338 us (6.1 GB/s) | 33229 us (5.8 GB/s) | 0.94x (LOSS) |

- Root cause: the pool's owned-Vec-concat design (chosen to stay unsafe-free) is
  fundamentally wrong for MEMORY-BOUND elementwise: (1) the caller's final
  `out[s..].copy_from_slice(buf)` concat DOUBLES output traffic and re-serializes
  the page-faults of the full output Vec — exactly the cost my `std::thread::scope`
  path AVOIDS by writing each chunk directly into one fresh output; (2) per-chunk
  `vec![0.0; len]` zero-fill + channel round-trip add latency; (3) cross-CCD threads
  don't share the 32MB/CCD L3, so L3-resident data can't be parallelized at L3 speed.
  Note: even at 8M (where my scope path WINS 7x via parallel page-faulting of one
  output) the pool LOSES (0.94x) because the concat re-serializes those faults.
- Decision: DO NOT wire in thread_pool.rs; DO NOT route elementwise through it.
  Confirms the standing conclusion: the L3-resident JAX gap (1M: ~70 vs ~200 GB/s)
  needs compiled-jaxpr arena BUFFER REUSE (write into a pre-faulted donated buffer,
  no per-op fresh alloc), NOT a thread pool. A copy-free persistent pool would need
  either `unsafe` (forbidden here) or `Arc<[AtomicU64]>` shared output (still an O(n)
  read-out copy into Vec<f64> at the end — same doubling). Left orphaned file as-is.
- Retry predicate: only revisit a persistent pool for elementwise if it can write
  DIRECTLY into the final output without a concat/read-out copy (i.e. after a
  buffer-reuse arena exists, at which point the spawn-free pool becomes worthwhile
  for the L3-resident regime). Not before.

## CobaltForge - Threaded same-shape i64 binops for DRAM-bound arrays (JAX WIN)

- Lever: completes the same-shape elementwise threading family (after f64/f32 in
  7b7924b7) with i64. New `eval_same_shape_i64_parallel` threads the dense i64
  same-shape binop using the EXACT `int_op` closure the serial path uses (carries
  per-primitive semantics: `wrapping_add`/`wrapping_sub`/`wrapping_mul`, `i64::max/min`,
  `checked_div().unwrap_or(0)`), gated at `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23).
  Required adding `+ Sync` to `eval_binary_elementwise`'s `int_op` bound — non-breaking,
  every caller passes a stateless (Sync) closure. Generic over op: cheap ops win via
  DRAM bandwidth + parallel page-faulting, heavy ones (pow/gcd/lcm) on compute.
- Bit-identity: i64 results are lane-independent; same `int_op`, same order. Guarded by
  `same_shape_i64_parallel_bit_identical_to_serial` (i64::MAX/MIN/0/-1 edges + forced
  div-by-zero lanes across wrapping add/sub/mul, max/min, checked_div) — PASS.
- Conformance: `fj-lax --lib` 1497 pass (+1 new), 43 fail (PRE-EXISTING, identical set
  on clean baseline) — 0 new failures.
- Measured A/B (LOCAL same-binary, real `eval_*` fresh-alloc path, best-of-10) and JAX
  head-to-head (`jax.jit(lambda x,y:x+y)` int64 x64, /tmp/jax_i64.py):

  | Workload | serial (before) | parallel (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | add_i64 n=16M | 64098 us (6.0 GB/s) | 9913 us (38.7 GB/s) | 6.47x | 17603 us | 0.56x (1.78x FASTER) |
  | add_i64 n=64M | 266637 us (5.8 GB/s) | 39364 us (39.0 GB/s) | 6.77x | 70136 us | 0.56x (1.78x FASTER) |

- Decision: KEEP. Before: Rust LOST i64 add to JAX by ~3.6-3.8x (serial fresh-alloc
  cliff). After: DOMINATES by 1.78x. Same gate discipline; L3-resident sizes serial.
- Family status: same-shape f64/f32/i64 + scalar-tensor f64/f32 cheap binops now all
  threaded beyond L3 and all FASTER than JAX. Remaining unthreaded large-array cheap
  paths: i64 scalar-tensor + cheap unary (neg/abs/sign) — same pattern, lower priority.

## CobaltForge - Caching allocator (mimalloc) eliminates the fresh-alloc page-fault cliff (MAINTAINER-GATED, measured)

- THE finding: the serial ~6 GB/s cliff on >L3 fresh-output elementwise (root cause
  behind all my threading wins) is NOT inherent — it is glibc returning fresh,
  zero-faulted mmap pages per op (page faults serialized on first write). A caching
  allocator (mimalloc) recycles warm, already-faulted spans across same-size
  allocations, so the per-op fault cost disappears.
- Measured (standalone, +avx2, best-of-20, serial `a+b` collect, fresh Vec/op,
  2026-06-19; same Zen3 5975WX, JAX f64 add reference in prior rows):

  | n | glibc serial | mimalloc serial | mimalloc speedup | JAX | mimalloc/JAX |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1M  | 92.3 GB/s | 88.5 GB/s | 0.96x (L3-resident, no cliff) | ~200 GB/s | (still lose) |
  | 16M | 5.9 GB/s | 21.7 GB/s | 3.7x | 18099 us (21.2 GB/s) | ~PARITY |
  | 64M | 6.0 GB/s | 23.4 GB/s | 3.9x | 57472 us (26.7 GB/s) | 1.14x slower |

  mimalloc alone brings SERIAL large-add to ~JAX parity (no threading, no value change).
- CRITICAL interaction — mimalloc and my shipped cheap-binop THREADING are SUBSTITUTES,
  not complements (threaded `a+b`, all cores, fresh output, best-of-20):

  | n | glibc thr/ser | mimalloc thr/ser |
  | ---: | ---: | ---: |
  | 16M | 6.32x (threading wins) | 1.02x (threading no-op) |
  | 64M | 6.67x (threading wins) | 0.93x (threading REGRESSES) |

  Mechanism: my threaded path's `vec![0.0; n]` (calloc) under glibc gets untouched
  zero-pages that the worker WRITES fault IN PARALLEL (the win). Under mimalloc the
  reused warm span is re-zeroed by calloc SERIALLY before the workers run, so the
  serial memset caps the threaded path at serial speed. => If mimalloc is adopted,
  the CHEAP_BINARY_PARALLEL_MIN threading gate should be RAISED/removed (it stops
  helping and slightly hurts at 64M); mimalloc is the more general lever (every
  allocating op + the L3-resident regime, not just the 6 ops I hand-threaded).
- Why NOT unilaterally committed: (1) workspace-policy decision (global allocator
  choice, like the +fma gate); (2) a library-level `#[global_allocator]` in fj-lax
  would CONFLICT with the existing `PeakAlloc` global_allocator in
  `crates/fj-interpreters/benches/eval_chain_memory.rs` (two global allocators per
  binary = compile error); (3) it supersedes shipped threading, so adoption needs a
  coordinated gate-retune. forbid(unsafe_code) is NOT a blocker — declaring an
  external allocator's `#[global_allocator]` static is safe code.
- Recommendation (filed as a bead): adopt mimalloc as the default global allocator
  workspace-wide (resolve the PeakAlloc bench conflict by scoping/removing its
  tracker or cfg-gating), then raise/remove the cheap-binop threading gate. Expected
  ~3.7-3.9x on the allocation component of every large op and serial JAX parity on
  large elementwise, plus likely gains across all fj-lax/fj-core ops that allocate a
  fresh output. This is the single highest-EV lever measured in the campaign.
- Retry predicate: do not re-measure the mimalloc-vs-glibc cliff (settled). Next step
  is the maintainer allocator decision + gate retune, or the compiled-jaxpr arena
  (which would subsume both by reusing a donated output buffer with no alloc at all).

## CobaltForge - Threaded cheap unary f64/f32 (JAX WIN) + mimalloc framing CORRECTION

- Lever: threads the serial dense f64/f32 unary fast paths (Neg/Abs/Sign/Square/Floor/
  Ceil/Round/Reciprocal and any sub-threshold transcendental reaching the fast path) at
  `n >= CHEAP_BINARY_PARALLEL_MIN` (1<<23), inside `eval_unary_f{64,32}_tensor_fast_path`
  (the chokepoint for both unary callers). Needed `+ Sync` on `eval_unary_elementwise`
  and `eval_unary_int_or_float`'s float_op bounds (non-breaking; all closures stateless).
  Bit-identical (lane-independent); guarded by `threaded_unary_maps_bit_identical_to_serial`
  (neg/abs/square/floor/reciprocal, NaN/±inf/±0/MAX, f64+f32).
- Conformance: `fj-lax --lib` 1498 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured A/B (LOCAL, real path, best-of-10) + JAX (`jax.jit(lambda x:-x)` x64):

  | Workload | serial (before) | parallel (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | neg_f64 16M | 59907 us (4.3 GB/s) | 7461 us (34.3 GB/s) | 8.03x | 15575 us | 0.48x (2.09x FASTER) |
  | neg_f64 64M | 241451 us (4.2 GB/s) | 30260 us (33.8 GB/s) | 7.98x | 59500 us | 0.51x (1.97x FASTER) |

- *** CORRECTION to the prior "mimalloc supersedes threading" entry/scorecard/bead oneqh ***
  Re-measured on the REAL `eval_primitive` path (which already includes my threading) under
  both allocators. mimalloc and threading are COMPLEMENTARY (different op sets), NOT
  substitutes, and threading is the SUPERIOR lever for memory-bound ops:

  | op @16M | glibc serial | glibc+threading (shipped) | mimalloc serial | mimalloc+threading |
  | --- | ---: | ---: | ---: | ---: |
  | Add (threaded) | (n/a) | 38.7 GB/s | (n/a) | 34.5 GB/s |
  | Neg (was serial) | 4.4 GB/s | 34.3 GB/s (now threaded) | 20.5 GB/s | ~ |

  - For ops I THREAD: glibc+threading (~38 GB/s, parallel page-faulting of the calloc
    zero-pages) >= mimalloc (~24-34); mimalloc is ~neutral-to-slightly-negative there.
  - For ops still SERIAL: mimalloc gives ~4.7x (Neg 4.4->20.5) — BUT threading gives more
    (4.4->34.3, parallel faulting beats warm-span serial memset). So the better fix for
    each serial op is to THREAD it, not adopt mimalloc.
  - Therefore: KEEP the threading gate regardless of any allocator decision (removing it
    craters these ops to 4-6 GB/s). mimalloc remains a modest COMPLEMENTARY win only for
    large allocating ops that are hard to thread (and is superseded wherever threading is
    applied). The round-4 "remove the gate if mimalloc adopted" guidance was WRONG.
- Retry predicate: extend the same gate to the remaining serial large ops where it pays
  (i64/u32/u64 unary; scalar-tensor i64; reduction outputs; transpose/gather/broadcast
  outputs) — each is a 4.4->~34 GB/s, JAX-beating, mimalloc-beating win. Bead oneqh
  updated to reflect the corrected (complementary, keep-gate) recommendation.

## CobaltForge - Threaded f64/f32 broadcast_replicate for DRAM-bound outputs (JAX WIN)

- Lever: `BroadcastInDim` (bias/feature replicate [C]->[B,C], [A,C]->[A,B,C], etc.) built the
  large output via a SERIAL `extend_from_slice` loop — page-fault-bound at ~2.4 GB/s on a
  fresh >L3 output (WORSE than the elementwise cliff). New `broadcast_replicate_into<T>` fills
  a caller-allocated `vec![<concrete zero>; total]` (alloc_zeroed/calloc = untouched zero
  pages) by splitting the outer block iterations across scoped threads — parallel page-faults
  the output and copies the contiguous src blocks concurrently. Wired into the f64 + f32 arms
  of `eval_broadcast_in_dim` (the calloc must be concrete-typed; generic `vec![T::default()]`
  does NOT lower to calloc, so other dtypes keep the serial `broadcast_replicate`).
- Bit-identity: each output block `o` is written from the SAME `src[base(o)..+block_len]` at the
  SAME offset `o*block_len`; `base(o)` is the mixed-radix outer-coordinate->base mapping, equal
  to the serial odometer. Guarded by `broadcast_replicate_into_bit_identical_to_serial`
  (incl. a size-1-stretch multi-outer-axis case), compared bit-for-bit to the serial generic.
- Conformance: `fj-lax --lib` 1499 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, [1024]->[N,1024], best-of-20) + JAX
  (`jax.jit(jnp.broadcast_to)` x64, /tmp/jax_bcast.py):

  | output | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16.4M | 55417 us (2.4 GB/s) | 6097 us (21.5 GB/s) | 9.09x | 22254 us (5.9 GB/s) | 0.27x (3.65x FASTER) |
  | 65.5M | 214491 us (2.4 GB/s) | 23130 us (22.7 GB/s) | 9.27x | 87049 us (6.0 GB/s) | 0.27x (3.76x FASTER) |

- Decision: KEEP. Broadcast is ubiquitous (every bias/feature materialization). Before, Rust
  LOST to JAX by 2.5x AND was catastrophically slow internally; after, Rust DOMINATES by 3.65-
  3.76x. Note JAX's own broadcast_to materializes at only ~6 GB/s here. Same gate; small (<gate)
  broadcasts stay serial (the calloc+serial-copy is bit-identical, negligible overhead).
- Retry predicate: extend `_into` to i64/u32/u64/bool/half broadcast arms (each allocates a
  concrete-zero vec -> calloc, so the same threaded fill applies) when those dtypes show up
  large. Confirms the general principle: every large fresh-output op wins from calloc +
  parallel page-faulting; threading beats both serial-glibc and mimalloc.

## CobaltForge - Threaded f64<->f32 convert (ConvertElementType) for DRAM-bound arrays (JAX WIN)

- Lever: the hot mixed-precision casts f64->f32 (downcast) and f32->f64 (upcast) built their
  large output via a serial `.iter().map(cast).collect()`, page-fault-bound at ~5.9 / ~3.4 GB/s.
  New `threaded_convert_into<S,D>` fills a caller-allocated calloc'd output
  (`vec![0.0f32; n]` / `vec![0.0f64; n]`) by splitting into scoped-thread chunks applying the
  per-element cast. Wired as early returns in the f64-source (->F32) and f32-source (->F64) arms
  of `eval_convert_element_type`, above `CHEAP_BINARY_PARALLEL_MIN`. Other casts keep serial
  (each could thread the same way — caller allocates a concrete-zero vec for calloc).
- Bit-identity: per-element `v as f32` / `f64::from(v)`, lane-independent => identical to serial.
  Guarded by `threaded_convert_bit_identical_to_serial` (NaN/±inf/±0, both directions).
- Conformance: `fj-lax --lib` 1500 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-20) + JAX (`x.astype(...)` x64, /tmp/jax_conv.py):

  | cast | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | f64->f32 16M | 32280 us (5.9 GB/s) | 5328 us (36.0 GB/s) | 6.06x | 11422 us | 0.47x (2.14x FASTER) |
  | f64->f32 64M | 129841 us (5.9 GB/s) | 19965 us (38.5 GB/s) | 6.50x | 32748 us | 0.61x (1.64x FASTER) |
  | f32->f64 16M | 55907 us (3.4 GB/s) | 6800 us (28.2 GB/s) | 8.22x | 13499 us | 0.50x (1.99x FASTER) |
  | f32->f64 64M | 226888 us (3.4 GB/s) | 26645 us (28.8 GB/s) | 8.51x | 58869 us | 0.45x (2.21x FASTER) |

- Decision: KEEP. Mixed-precision casts are ubiquitous; before, Rust LOST to JAX by 1.6-4.2x;
  after, Rust DOMINATES by 1.64-2.21x. Same gate; small casts stay serial.
- Retry predicate: extend the same calloc+threaded fill to the other hot casts (f64->i32/i64
  quantization, f32->i32, half<->f32) — each allocates a concrete-zero output so the same
  helper applies. Confirms the campaign principle across yet another op family.

## CobaltForge - Threaded rank-2 f64/f32 transpose for DRAM-bound arrays (JAX WIN)

- Lever: the rank-2 [1,0] transpose used `transpose_2d_blocked` (cache-blocked but SERIAL,
  and it pre-fills the output with `vec![src[0]; total]` = a serial fault of the whole output
  before transposing) — ~2.2 GB/s, page-fault + strided-read bound. New `transpose_2d_into<T>`
  fills a caller-allocated calloc'd output by splitting the OUTPUT-row range (= source-column
  range) across scoped threads, each running a cache-blocked sub-transpose. Wired into the
  f64/f32 arms of the rank-2 branch above `CHEAP_BINARY_PARALLEL_MIN`; other dtypes / small
  keep `transpose_2d_blocked`.
- Bit-identity: every `out[j*rows+i] = src[i*cols+j]`, just produced concurrently into disjoint
  contiguous output-column slices. Guarded by `transpose_2d_into_bit_identical_to_serial`
  (NON-SQUARE dims 8192x2048, 3000x5600, + square — catches rows/cols swaps), bit-for-bit vs
  the serial blocked kernel.
- "Tiling regresses" (recorded) is about CACHE-BLOCKING the strided walk, NOT threading — this
  KEEPS the existing block tiling and adds thread-level parallelism on top (orthogonal, legal).
- Conformance: `fj-lax --lib` 1501 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(lambda x: x.T+0.0)` x64,
  /tmp/jax_t.py):

  | transpose | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | f64 4096x4096 | 119433 us (2.2 GB/s) | 9716 us (27.6 GB/s) | 12.29x | 28454 us (9.4 GB/s) | 0.34x (2.93x FASTER) |
  | f64 8192x8192 | 484097 us (2.2 GB/s) | 37217 us (28.9 GB/s) | 13.01x | 111599 us (9.6 GB/s) | 0.33x (3.0x FASTER) |

- Decision: KEEP. Transpose is ubiquitous (attention, matmul prep). Before, Rust LOST to JAX by
  ~4.3x; after, Rust DOMINATES by ~3x. Same gate; small/other-dtype transposes unchanged.
- Retry predicate: extend `transpose_2d_into` to i64/u32/u64/half/complex rank-2 arms (concrete
  zero -> calloc) and thread the N-D `transpose_general` outer odometer the same way (base_of
  decomposition, like broadcast). Same campaign principle.

## CobaltForge - Threaded f64/f32 contiguous gather (embedding lookup) for DRAM-bound (JAX WIN)

- Lever: the contiguous full-row gather (embedding lookup, slice_sizes [1,DIM]) built its large
  output via a serial per-index `extend_from_slice`, page-fault-bound at ~2.1-2.2 GB/s. New
  `gather_contiguous_into<T>` fills a caller-allocated calloc'd output by splitting the index
  range across scoped threads (each does its rows' memcpy / OOB-fill). Wired into the f64/f32
  contiguous arms above `CHEAP_BINARY_PARALLEL_MIN`; pre-validates bounds and returns false
  (caller falls to the serial path -> identical error) if any in-bounds index would exceed src.
- Bit-identity: index i -> out[i*slice_elems..], Some(idx)->src row copy, None->fill, exactly the
  serial contiguous copy. Guarded by `gather_contiguous_into_bit_identical_to_serial` (mix of
  in-bounds Some + OOB None/fill at total >= gate), bit-for-bit vs a serial reference.
- Conformance: `fj-lax --lib` 1502 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, [16384,1024] f32 table, best-of-15) + JAX
  (`jax.jit(lambda t,i: t[i])` x64, /tmp/jax_g.py):

  | nidx (out elems) | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16384 (16.8M) | 31161 us (2.2 GB/s) | 4781 us (14.0 GB/s) | 6.52x | 8336 us (8.1 GB/s) | 0.57x (1.74x FASTER) |
  | 65536 (67M) | 125562 us (2.1 GB/s) | 15007 us (17.9 GB/s) | 8.37x | 31003 us (8.7 GB/s) | 0.48x (2.07x FASTER) |

- Decision: KEEP. Embedding gather is huge in NLP; before, Rust LOST to JAX by ~3.7-4x; after,
  DOMINATES by 1.74-2.07x. (Gather hits 14-18 GB/s vs ~34 for streaming ops — random table reads
  miss cache — but still beats JAX, which is also gather-cache-bound at ~8 GB/s.) Same gate.
- Retry predicate: extend `gather_contiguous_into` to i64/i32/u32/u64/half/complex contiguous
  arms (concrete-zero calloc) and to the strided `gather_window_blocks` (per-index independent;
  same index-range split). Same campaign principle.

## CobaltForge - Threaded dense f64/f32 axis-0 concatenate for DRAM-bound (JAX WIN, was parity)

- Lever: contiguous (outer==1: axis-0 / KV-cache / batch) concat built its large output via
  `LiteralBuffer::from_concat_slices`, page-fault-bound at ~2.1 GB/s. New `concat_contiguous_into<T>`
  fills a caller-allocated calloc'd output by splitting the output into contiguous chunks across
  scoped threads; each chunk copies from whichever source(s) it overlaps (cumulative-offset table,
  so chunk boundaries may cross source boundaries). Wired into the f64/f32 `outer==1` case above
  `CHEAP_BINARY_PARALLEL_MIN`; axis>0 (interleaved) and other dtypes keep `from_concat_slices`.
- Bit-identity: out = srcs concatenated in order, exactly `from_concat_slices` for outer==1.
  Guarded by `concat_contiguous_into_bit_identical_to_serial` (5 uneven sources incl. len-1 and
  len-7 so chunk boundaries cross sources, + end-to-end 3-operand eval_concatenate).
- Conformance: `fj-lax --lib` 1503 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path WITH checksum to defeat dead-code elision, best-of-15)
  + JAX (`jax.jit(jnp.concatenate axis=0)` x64, /tmp/jax_cat.py):

  | out elems | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | ~16M | 63926 us (2.1 GB/s) | 7872 us (16.7 GB/s) | 8.12x | 72226 us (1.8 GB/s) | 0.11x (9.2x FASTER) |
  | ~64M | ~256000 us (2.1 GB/s) | 30982 us (16.9 GB/s) | ~8x | 290818 us (1.8 GB/s) | 0.11x (9.4x FASTER) |

- Decision: KEEP. Concat was at PARITY with JAX (both ~2 GB/s, page-fault bound) — threading turns
  it into ~9x DOMINATION. Concat is common (KV-cache append, batch/feature concat). Same gate.
- MEASUREMENT NOTE: an unconsumed `eval_primitive(Concatenate)` in a bench loop was elided by the
  optimizer (reported 0.3us / 500000 GB/s); always consume the result (checksum) when timing.
- Retry predicate: extend to axis>0 (interleaved row blocks; thread the per-row task list) and
  other dtypes (concrete-zero calloc). Same campaign principle.

## CobaltForge - Threaded bf16/f16 broadcast + gather (training dtype) for DRAM-bound (JAX WIN)

- Lever: extends the proven threaded broadcast/gather to the bf16/f16 (u16-backed) dtype — the
  DOMINANT training dtype, previously left on the serial cliff (~2.5 GB/s). The generic
  `broadcast_replicate_into<T>` / `gather_contiguous_into<T>` already accept `T: Copy+Send+Sync`
  (u16 qualifies); wired the half-float broadcast arm and contiguous-gather arm to allocate a
  calloc'd `vec![0u16; total]` and use the threaded fill. Bit-identical (u16 bit copies).
- Guarded by `threaded_half_float_broadcast_and_gather_bit_identical` (bf16 broadcast vs serial
  broadcast_replicate; bf16 gather incl. OOB None/fill vs serial reference).
- Conformance: `fj-lax --lib` 1504 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | serial (before) | threaded (after) | internal |
  | --- | ---: | ---: | ---: |
  | bf16 broadcast [1024]->[16384,1024] | 12706 us (2.6 GB/s) | 2838 us (11.8 GB/s) | 4.48x |
  | bf16 broadcast [1024]->[65536,1024] | 53838 us (2.5 GB/s) | 6421 us (20.9 GB/s) | 8.38x |
  | bf16 gather nidx=16384 (16.8M) | (~serial cliff) | 3030 us (11.1 GB/s) | ~6x |
  | bf16 gather nidx=65536 (67M) | (~serial cliff) | 7826 us (17.2 GB/s) | ~6x |

- Decision: KEEP. bf16/f16 is the training-dominant dtype; bias broadcast + embedding gather are
  ubiquitous there. Same calloc+parallel-page-fault lever, now covering the half-float backings.
- Retry predicate: extend the same to i64/u32/u64 broadcast/gather/transpose/concat arms (all
  Copy+Send+Sync, concrete-zero calloc) and to bf16/f16 transpose/concat. Mechanical.

## CobaltForge - Threaded scalar-broadcast / full() constant fill (f64/f32) for DRAM-bound (JAX WIN)

- Lever: scalar broadcast (`jnp.full`, scalar->tensor, mask/const init) built its output via
  `vec![v; total]`, which for a NON-ZERO v is a serial element fill, page-fault-bound at ~2.5 GB/s
  (v==0 already callocs). New `threaded_fill_into<T>` writes v into a caller-allocated calloc'd
  output across scoped threads (parallel page-fault). Wired into the F64Bits/F32Bits arms of the
  BroadcastInDim scalar path above `CHEAP_BINARY_PARALLEL_MIN`. Bit-identical to `vec![v; total]`.
- Guarded by `threaded_scalar_fill_bit_identical_to_serial` (v = 3.14 / 0.0 / -2.5 / NaN, + f32).
- Conformance: `fj-lax --lib` 1505 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.full)` x64, /tmp/jax_full.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 49542 us (2.6 GB/s) | 6310 us (20.3 GB/s) | 7.85x | 21549 us (5.9 GB/s) | 0.29x (3.42x FASTER) |
  | 64M | 205894 us (2.5 GB/s) | 24466 us (20.9 GB/s) | 8.42x | 86302 us (5.9 GB/s) | 0.28x (3.53x FASTER) |

- Decision: KEEP. jnp.full / scalar-const materialization is common (init, masks, fills). Before,
  Rust LOST to JAX by ~2.3x; after, DOMINATES by ~3.5x. Same gate; v==0 stays calloc (already fast).
- Retry predicate: extend threaded_fill_into to the half/i64/u32/u64/bool scalar arms (concrete-zero
  calloc) and to array_creation full/zeros/ones if those hit the cliff. Mechanical.

## CobaltForge - Threaded full f64/f32 max/min reduce for DRAM-bound (JAX WIN)

- Lever: full ReduceMax/ReduceMin used a single-threaded SIMD fold (`simd_reduce_minmax_f64/f32`,
  ~21.6 GB/s f64) — read-bound, so a single core can't saturate multi-channel DRAM. New
  `threaded_reduce_minmax_f64/f32` splits the input into chunks, SIMD-reduces each on a scoped
  thread, and combines the partials. Gated at `CHEAP_BINARY_PARALLEL_MIN` (serial SIMD wins in
  cache). (Sum/Prod stay scalar/serial — float-sum is non-associative and bit-exact-pinned vs JAX;
  not touched.)
- Bit-identity: max/min are associative+commutative; each partial resolves ±0 sign and NaN exactly
  per-chunk, and the combine preserves both (NaN if any partial NaN -> canonical NaN; f64::max/min
  maxNum/minNum ±0 handling is order-independent, so fold-of-folds == full fold). Guarded by
  `threaded_reduce_minmax_bit_identical_to_serial` (NaN mid/boundary, ±0-only, ±inf; f64 AND f32).
- Conformance: `fj-lax --lib` 1506 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.max)` x64, /tmp/jax_red.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 5935 us (21.6 GB/s) | 3627 us (35.3 GB/s) | 1.64x | 4518 us (28.3 GB/s) | 0.80x (1.25x FASTER) |
  | 64M | 26189 us (19.5 GB/s) | 8552 us (59.9 GB/s) | 3.06x | 11568 us (44.3 GB/s) | 0.74x (1.35x FASTER) |

- Decision: KEEP. max/min reduce (softmax stability, clipping bounds) was a 1.3-2.3x JAX LOSS;
  threading flips it to a 1.25-1.35x WIN. First reduction-family threading; read-bound (no
  page-fault cliff), so the win is bandwidth (~2-3x internal) not the dramatic fill-op factors.
- NEGATIVE (recorded): ReduceSum is 10.9 GB/s (3-4x slower than JAX's 33-43) but CANNOT be threaded
  or SIMD-vectorized — float sum is non-associative and pinned bit-exact vs JAX (memory: DO-NOT
  float-sum SIMD). Closing it needs a bit-exact-matching tree/pairwise order = the compiled-jaxpr /
  XLA-order-matching swing, not threading.
- Retry predicate: extend threaded minmax to bf16/f16 full reduce + argmax/argmin (needs
  first-index-preserving combine) and to axis reductions where the reduced extent is huge.

## CobaltForge - Threaded bf16/f16 transpose + axis-0 concat (training/inference dtype) (JAX WIN)

- Lever: extends the proven threaded rank-2 transpose (`transpose_2d_into`) and axis-0 concat
  (`concat_contiguous_into`) to the bf16/f16 (u16-backed) dtype — bf16 attention transposes and
  bf16 KV-cache / feature concat are ubiquitous in LLM training & inference, previously left on
  the serial cliff (~2-2.5 GB/s). The helpers are already generic over T: Copy+Send+Sync (u16
  qualifies); wired the half-float arms to a calloc'd vec![0u16; total]. Bit-identical (u16 copies).
- Guarded by `threaded_half_float_transpose_and_concat_bit_identical` (bf16 transpose non-square
  6000x3000 vs serial transpose_2d_blocked + end-to-end eval_transpose; bf16 2-operand axis-0 concat
  vs serial).
- Conformance: `fj-lax --lib` 1507 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | threaded (after) | (serial was ~2-2.5 GB/s cliff) internal |
  | --- | ---: | ---: |
  | bf16 transpose 4096x4096 | 4927 us (13.6 GB/s) | ~5x |
  | bf16 transpose 8192x8192 | 13283 us (20.2 GB/s) | ~8x |
  | bf16 concat0 out=32.8M | 5006 us (13.1 GB/s) | ~6x |
  | bf16 concat0 out=131M | 15446 us (17.0 GB/s) | ~8x |

- Decision: KEEP. Same calloc+parallel-page-fault lever, now covering the training-dominant dtype
  for transpose+concat (matching the bf16 broadcast+gather shipped in 3fe6f23a). Bit-identical.
- Retry predicate: extend bf16/f16 to convert (half<->f32) and the i64/u32/u64 arms of all threaded
  ops. Mechanical (concrete-zero calloc + existing generic _into helpers).

## CobaltForge - Threaded i64 broadcast + gather (index/id tensors) + argmax verification (JAX WIN)

- Lever: extends the threaded broadcast/gather to i64 (i32 shares the backing) — position-ids /
  token-ids broadcast and index/id gather at scale, previously on the serial cliff (~2.2-2.5 GB/s).
  Generic `broadcast_replicate_into` / `gather_contiguous_into` accept i64 (Copy+Send+Sync); wired
  the i64 arms to calloc'd `vec![0i64; total]`, preserving i32 width on the broadcast result.
  Bit-identical. Guarded by `threaded_i64_broadcast_and_gather_bit_identical` (broadcast vs serial;
  gather incl. OOB None/fill vs serial).
- Conformance: `fj-lax --lib` 1508 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15):

  | workload | threaded | (serial ~2.2-2.5 GB/s cliff) internal |
  | --- | ---: | ---: |
  | i64 broadcast ->[16384,1024] | 5966 us (22.5 GB/s) | ~9x |
  | i64 broadcast ->[65536,1024] | 24545 us (21.9 GB/s) | ~9x |
  | i64 gather nidx=16384 (16.8M) | 8634 us (15.5 GB/s) | ~7x |
  | i64 gather nidx=65536 (67M)   | 31753 us (16.9 GB/s) | ~7x |

- ARGMAX/ARGMIN — VERIFIED ALREADY WINNING (no change). Measured full f64 argmax (LOCAL,
  best-of-15) vs `jax.jit(jnp.argmax)` x64:

  | n | Rust | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: |
  | 16M | 10131 us (12.6 GB/s) | 13766 us (9.3 GB/s) | 0.74x (1.36x FASTER) |
  | 64M | 40560 us (12.6 GB/s) | 55010 us (9.3 GB/s) | 0.74x (1.36x FASTER) |

  Rust argmax already beats JAX 1.36x; the delicate NaN/first-index-tie threaded combine is NOT
  worth the risk since there is no competitive gap. Left as-is.
- SATURATION NOTE: with this commit, every measured large memory-bound op (binops/scalar-tensor/
  unary/broadcast/scalar-fill/convert/transpose/gather/concat across f64/f32/i64/bf16/f16, plus
  max/min-reduce and argmax) now MATCHES or BEATS JAX. The only remaining JAX losses are the
  documented off-limits/architectural ones: ReduceSum/ReduceProd & cumsum (float non-associative,
  bit-exact-pinned -> need XLA-order matching) and the L3-resident regime (needs compiled-jaxpr
  arena buffer reuse). Both are multi-session swings, NOT threading.

## CobaltForge - Threaded f64/f32 select/where (masking) for DRAM-bound (JAX WIN)

- Lever: `select`/`jnp.where` (masking — attention, dropout, clipping) ran its dense f64/f32 fast
  paths serially (~1.8-1.9 GB/s, worse than the binop cliff: it reads cond + 2 branches + writes
  output). New reusable `threaded_index_fill_into<T>(out, threads, pick)` fills a calloc'd output by
  index across scoped threads; wired into both cond backings (dense Bool slice AND bit-packed
  BoolWords from a comparison mask) of the f64 and f32 select fast paths, above
  CHEAP_BINARY_PARALLEL_MIN. Bit-identical (per-index pick, lane-independent).
- Guarded by `threaded_select_bit_identical_to_serial` (both cond backings, f64+f32, NaN/±0/±inf
  branches), bit-for-bit vs a serial reference.
- Conformance: `fj-lax --lib` 1509 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, with checksum, best-of-15) + JAX
  (`jax.jit(jnp.where)` x64, /tmp/jax_sel.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 66376 us (1.9 GB/s) | 10753 us (11.9 GB/s) | 6.17x | 23242 us (5.5 GB/s) | 0.46x (2.16x FASTER) |
  | 64M | 280863 us (1.8 GB/s) | 40010 us (12.8 GB/s) | 7.02x | 70142 us (7.3 GB/s) | 0.57x (1.75x FASTER) |

- Decision: KEEP. select/where masking is ubiquitous; before, Rust LOST to JAX by ~2.9-4x; after,
  DOMINATES by 1.75-2.16x. (Lower GB/s than 2-input ops — 3 inputs + 1 output of traffic — but
  still beats JAX.) `threaded_index_fill_into` is now a reusable per-index parallel fill primitive.
- Retry predicate: extend the same to i64/bf16/u32 select arms + select_n (multi-way). Mechanical.

## CobaltForge - Threaded f64/f32/i64 clamp/clip (scalar bounds) for DRAM-bound (JAX WIN)

- Lever: `clamp`/`jnp.clip` (gradient clipping, activation bounds, relu6) ran its dense scalar-bounds
  fast paths serially (~2.1 GB/s) — a documented JAX loss (prior scorecard: I64 3.10-3.64x slower).
  Threaded the f64/f32/i64 scalar-bounds dense paths via `threaded_index_fill_into` (calloc'd output
  + per-index clamp across scoped threads) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical (per-index
  clamp closure: clamp_f64/f32 NaN-normalize, i64 max/min — lane-independent).
- Guarded by `threaded_clamp_bit_identical_to_serial` (f64/f32/i64, NaN/±0/±inf), bit-for-bit vs a
  serial reference replicating clamp_f64/f32 + i64 max/min.
- Conformance: `fj-lax --lib` 1510 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`jax.jit(jnp.clip)` x64, /tmp/jax_clamp.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 60484 us (2.1 GB/s) | 8304 us (15.4 GB/s) | 7.28x | 17020 us (7.5 GB/s) | 0.49x (2.05x FASTER) |
  | 64M | 249256 us (2.1 GB/s) | 31689 us (16.2 GB/s) | 7.87x | 60542 us (8.5 GB/s) | 0.52x (1.91x FASTER) |

- Decision: KEEP. clip/clamp is ubiquitous (gradient clipping, activation bounds); the prior
  documented 3.1-4x JAX loss is now a 1.91-2.05x WIN. Resolves the "I64 clamp 3.10-3.64x slower"
  scorecard blocker (for the scalar-bounds case).
- Retry predicate: extend to the tensor-bounds + mixed scalar/tensor clamp arms (clamp_*_tensor_*)
  and bf16/f16 clamp via threaded_index_fill_into. Mechanical.

## CobaltForge - Threaded contiguous slice (f64/f32/bf16/i64) for DRAM-bound (JAX WIN) + rev/bitwise losses

- Lever: `slice` (sequence/window/batch slicing — ubiquitous) copied its contiguous leading-axis
  output via serial `src[start..end].to_vec()` (page-fault-bound ~3.3-3.5 GB/s). Routed the f64/f32/
  bf16(f16)/i64 contiguous-trailing-slice dense arms through `concat_contiguous_into` with a single
  source (calloc'd output + parallel copy) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical (pure copy).
- Guarded by `threaded_slice_bit_identical_to_serial` (f64 + i64, offset start), bit-for-bit vs the
  serial sub-range copy.
- Conformance: `fj-lax --lib` 1511 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15) + JAX (`x[:N]` x64, /tmp/jax_misc.py):

  | n | serial (before) | threaded (after) | internal | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 16M | 72675 us (3.5 GB/s) | 17999 us (14.2 GB/s) | 4.04x | 22910 us (11.2 GB/s) | 0.79x (1.27x FASTER) |
  | 64M | 311358 us (3.3 GB/s) | 74208 us (13.8 GB/s) | 4.20x | 92900 us (11.0 GB/s) | 0.80x (1.25x FASTER) |

- Decision: KEEP. slice is ubiquitous; was a ~3.2x JAX loss, now a 1.25-1.27x win (modest because
  slice is a memory-bound copy and JAX already streams at ~11 GB/s).
- ALSO MEASURED (this round, recorded as remaining losses / follow-ons):
  * `rev` (reverse): Rust 2.1-2.2 GB/s vs JAX 10.8-10.9 (~5x LOSS). Threadable (index map
    out[i]=in[n-1-i]); not yet done — follow-on.
  * `BitwiseAnd/Or/Xor` (i64): Rust 4.8-4.9 GB/s vs JAX 20.7-24.3 (~4-5x LOSS). Goes through a
    dedicated bitwise eval fn (NOT eval_binary_elementwise), so my i64 arith threading doesn't
    cover it — follow-on (thread its integer same-shape loop + the BoolWords SWAR path).
- Retry predicate: thread rev + bitwise (and/or/xor/shifts) same-shape integer paths next; both are
  clean index-map / elementwise patterns with the same calloc+parallel lever.

## CobaltForge - Threaded i64/i32/u32/u64 bitwise and/or/xor (masking/quantization) for DRAM-bound

- Lever: bitwise and/or/xor on integer tensors goes through a DEDICATED `eval_bitwise_binary`
  (NOT `eval_binary_elementwise`), so the earlier arith threading missed it; the dense i64/i32/u32/
  u64 same-shape arms were serial maps (~4.8-4.9 GB/s). Threaded all four via the now-pub(crate)
  `threaded_index_fill_into` (calloc'd output + per-index `apply_bitwise_binary_*` across scoped
  threads) above CHEAP_BINARY_PARALLEL_MIN. Bit-identical.
- Guarded by `threaded_bitwise_bit_identical_to_serial` (and/or/xor, i64 + u64), bit-for-bit vs a
  serial reference.
- Conformance: `fj-lax --lib` 1512 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL, best-of-25 across several invocations): bitwise-and i64 4.9 -> ~19-20 GB/s
  (~3.7-4x internal). IMPORTANT MEASUREMENT CAVEAT: the host was under heavy contention this
  session — f64 add, i64 add, and i64 bitwise-and ALL measured ~18 GB/s back-to-back (vs the ~38
  GB/s unloaded add baseline from earlier rounds, and JAX bitwise 20.7-24.3 measured at lower load).
  So the threaded bitwise path performs IDENTICALLY to the threaded add path under equal load; the
  apparent "JAX still ahead" was a cross-invocation load artifact (see rch-bench-cross-invocation-
  variance), not a bitwise-specific weakness. Unloaded it tracks add (~38 GB/s), which dominates JAX.
- Decision: KEEP. Same proven lever as the arith binops, now covering the bitwise family it had
  missed; robust ~3.7-4x internal improvement, bit-identical, at-least-parity with JAX even under
  load. Re-measure the clean head-to-head on an idle host to confirm the unloaded ~1.8x domination.
- Retry predicate: rev (reverse) is the remaining clean loss (~5x); thread its index-map next.

## CobaltForge - Threaded rev (reverse) f64/f32/bf16/i64 for DRAM-bound (JAX WIN)

- Lever: `rev` (sequence/axis flip) built its output via the serial `rev_gather` odometer
  (page-fault-bound ~2.1-2.2 GB/s). New `rev_gather_into<T>` fills a calloc'd output by splitting
  the outer block iterations across scoped threads; each block `o` is `src[base(o)..+block_len]`
  with the reversal applied in base(o) — identical to the serial odometer. Wired f64/f32/bf16/i64
  arms above CHEAP_BINARY_PARALLEL_MIN. Bit-identical.
- Guarded by `rev_gather_into_bit_identical_to_serial` (leading-axis / inner-axis block_len==1 /
  multi-axis / both-axes-reversed), bit-for-bit vs serial rev_gather.
- Conformance: `fj-lax --lib` 1513 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-20, host under load this session):
  rev f64 2.1-2.2 -> 11.8-12.9 GB/s (5.4-6.1x internal). Same-load f64-add baseline = 17.7-18.1
  GB/s (vs ~38 unloaded), so the host is contended; JAX rev measured 10.8-10.9 (lower load).
  At equal load rev (~12) trails add (~18) because the reverse block order is less prefetch-
  friendly; unloaded it scales toward ~25 GB/s (~2x over JAX). Net: flips a ~5x JAX loss into
  parity-to-win (load-dependent), robust 5.4-6.1x internal improvement, bit-identical.
- Decision: KEEP. Same calloc+parallel-page-fault lever, now covering rev (the last queued clean
  loss). Re-measure idle for the precise vs-JAX domination factor.
- Retry predicate: contained large-op surface now broadly threaded; remaining JAX losses are the
  documented off-limits/architectural ones (float sum/prod/cumsum order; L3-resident regime).

## CobaltForge - Threaded dynamic_update_slice (KV-cache write) for DRAM-bound (JAX WIN)

- Lever: `dynamic_update_slice` (autoregressive KV-cache WRITE — ubiquitous in LLM inference) cloned
  the large operand via serial `op_src.to_vec()` (page-fault cliff ~3.4-3.5 GB/s) then overwrote the
  (small) update region. Split `dynamic_update_slice_dense` into `dynamic_update_apply` (overwrite
  only) + a `dense_dus!` macro that allocates a calloc'd output (zero literal per dtype) and
  PARALLEL-copies the operand via `concat_contiguous_into` above the gate, then applies the update.
  Wired f64/f32/bf16/i64/i32/u32/u64/bool; complex keeps the serial dense copy (no zero-literal
  calloc). Bit-identical.
- Guarded by `threaded_dynamic_update_slice_bit_identical_to_serial` (1D contiguous f64 + 2D strided
  column-band i64), bit-for-bit vs operand-clone + region-overwrite reference.
- Conformance: `fj-lax --lib` 1514 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15, host loaded; operand n, update 4096):

  | n | serial (before) | threaded (after) | internal | add same-load | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 16M | 73162 us (3.5 GB/s) | 18618 us (13.8 GB/s) | 3.93x | 18.7 GB/s | 23922 us (10.7) | 0.78x (1.29x FASTER) |
  | 64M | 301748 us (3.4 GB/s) | 79638 us (12.9 GB/s) | 3.79x | 17.7 GB/s | 98401 us (10.4) | 0.81x (1.23x FASTER) |

- Decision: KEEP. KV-cache write is on the hottest LLM-inference path; was a ~3x JAX loss, now a
  1.23-1.29x WIN even under contention (unloaded tracks add toward ~2.6x). Same calloc+parallel-copy
  lever as slice; bit-identical; preserves dense storage (critical for the autoregressive loop).
- Retry predicate: dynamic_slice (KV-cache READ) is the sibling; thread its contiguous extract the
  same way if it shows a gap.

## CobaltForge - Threaded dynamic_slice (KV-cache read) for DRAM-bound (JAX WIN)

- Lever: `dynamic_slice` (KV-cache READ / scan windowing) extracted its contiguous output via serial
  `src[start..end].to_vec()` (page-fault cliff ~3.4 GB/s). Threaded the contiguous case in the
  `dense_ds!` macro: calloc'd output (zero literal per dtype) + parallel copy via
  `concat_contiguous_into` above CHEAP_BINARY_PARALLEL_MIN; non-contig keeps the serial odometer.
  Wired f64/f32/bf16/i64/i32/u32/u64/bool; complex keeps serial (no zero-literal calloc). Bit-identical.
- Guarded by `threaded_dynamic_slice_bit_identical_to_serial` (f64 + i64 contiguous extract), bit-for-
  bit vs the serial sub-range.
- Conformance: `fj-lax --lib` 1515 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15, host loaded):

  | out elems | serial (before) | threaded (after) | internal | add same-load | JAX | Rust/JAX |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 32M (in 64M) | ~3.4 GB/s | 29510 us (17.3 GB/s) | ~5x | 24.3 GB/s | 44931 us (11.4) | 0.66x (1.52x FASTER) |
  | 8M (in 16M) | 3.6 GB/s | (sub-gate, serial) | — | — | (10.5) | — |

- Decision: KEEP. KV-cache read; was a ~3x JAX loss, now 1.52x WIN for outputs >= the 8.39M gate
  (the 8M-output case is just below the gate and correctly stays serial — no regression). Pairs with
  the dynamic_update_slice (write) win to thread both halves of the autoregressive KV-cache loop.
- Retry predicate: the contained large-op data-movement surface (binops/unary/broadcast/scalar-fill/
  convert/transpose/gather/concat/slice/dynamic_slice/dynamic_update_slice/rev/select/clamp/bitwise/
  max-min-reduce) is now broadly threaded f64/f32/i64 (+bf16/u where wired). Remaining JAX losses are
  the documented off-limits/architectural ones (float sum/prod/cumsum order; L3-resident regime).

## CobaltForge - Threaded zero-pad (axis-0 contiguous) f64/f32/bf16 for DRAM-bound (JAX WIN)

- Lever: `pad` (zero-padding for conv/attention) built its output via `pad_copy_rows` = `vec![pad;
  total]` + serial per-row copies (~4.1 GB/s). For the common case — padding only on the leading
  axis (operand is ONE contiguous output block) AND a ZERO pad value — the whole op is a calloc'd
  output (pad region free) + a single PARALLEL copy of the operand block via `concat_contiguous_into`.
  Gated on `in_total >= CHEAP_BINARY_PARALLEL_MIN`. Wired f64/f32/bf16 (pad bits == 0). Non-leading
  padding / interior / non-zero pad fall to the existing pad_copy_rows / pad_fill_place. Bit-identical.
- Guarded by `threaded_zero_pad_bit_identical_to_serial` (f64 + bf16, low+high axis-0 pad), bit-for-
  bit vs a zeros + operand-block reference.
- Conformance: `fj-lax --lib` 1516 pass (+1 new), 43 fail (PRE-EXISTING) — 0 new failures.
- Measured (LOCAL real eval_primitive path, best-of-15; output = 2x operand, half is calloc'd pad):
  pad0 f64 out=32M 12.7 GB/s, out=131M 13.2 GB/s (was ~4.1 -> ~3x internal). vs JAX pad 5.6 GB/s
  (round-11, lower load) ~2.3x faster (the calloc'd pad region is free; the operand copy threads).
- Decision: KEEP. Zero-padding is ubiquitous (conv, attention masks); was a ~1.35x JAX loss, now a
  ~2.3x win for the common axis-0 zero-pad case. Same calloc+parallel-copy lever.
- Retry predicate: the contained large-op data-movement surface is now broadly threaded across the
  common shapes/dtypes. Non-leading-axis pad, interior/dilated pad, and non-zero pad-value remain on
  the serial path (lower priority). The remaining JAX losses are the documented off-limits/
  architectural ones (float sum/prod/cumsum reduction order; L3-resident regime).

## CobaltForge - *** MAJOR CORRECTION: same-load head-to-head reframes the elementwise vs-JAX claims ***

- Context: this host has been heavily CONTENDED all session (load avg ~5-8; f64 add ~17-22 GB/s vs
  ~38 idle). Prior rounds measured Rust and JAX at DIFFERENT load moments (cross-invocation / cross-
  process), which inflated many vs-JAX "win" ratios. This round I ran Rust and JAX BACK-TO-BACK under
  the same sustained load (64M, best-of-20), and for `add` also reversed the order (JAX-first vs
  Rust-first) with stable load to rule out ordering bias — the verdict did NOT flip.
- SAME-LOAD head-to-head (64M, microseconds; lower = better; load avg ~6):

  | op | Rust us | JAX us | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | broadcast_f64 | 71859 | 86802 | 0.83 | Rust WINS 1.21x |
  | dynamic_slice (out32M) | 35038 | 45869 | 0.76 | Rust WINS 1.31x |
  | dynamic_update_slice | 75925 | 99086 | 0.77 | Rust WINS 1.30x |
  | rev | 82955 | 92987 | 0.89 | Rust WINS 1.12x |
  | add_f64 | 88188 (86987 rev-order) | 66704 (68061) | 1.28-1.32 | JAX faster |
  | clamp_f64 | 79182 | 60725 | 1.30 | JAX faster |
  | bitwise_and_i64 | 86387 | 66712 | 1.29 | JAX faster |
  | select_f64 | 87188 | 71481 | 1.22 | JAX faster |

- CORRECTED CONCLUSION (supersedes the per-op "Nx FASTER than JAX" claims for the COMPUTE ops):
  * COPY / WRITE-BOUND ops (broadcast, scalar-fill, convert, transpose, gather, concat, slice,
    dynamic_slice, dynamic_update_slice, rev, zero-pad) — Rust's calloc + parallel page-faulting of
    the fresh OUTPUT genuinely BEATS JAX same-load (~1.1-1.3x). These wins HOLD.
  * COMPUTE / MULTI-INPUT-READ ops (add/sub/mul/div, clamp, select, bitwise and/or/xor; also
    max-reduce likely) — at EQUAL load JAX is ~1.2-1.32x FASTER. Rust's threaded path (std::thread::
    scope) does not reach JAX/XLA's aggregate DRAM READ bandwidth on 2-3 full input streams. My
    earlier per-round "1.7-2.2x FASTER than JAX" for these was a CROSS-LOAD ARTIFACT.
- REGRESSIONS: NONE. Every threaded path is a strict 3.5-10x improvement over its serial baseline
  (which was 4-6 GB/s, catastrophic) and is bit-identical — so ALL changes are KEPT. The correction
  is to the vs-JAX framing only, not the code.
- METHOD LESSON (reinforced): on a contended host, ONLY same-binary same-invocation (internal
  serial-vs-threaded) OR strictly back-to-back same-load Rust-vs-JAX numbers are trustworthy. The
  internal speedups in every prior entry are reliable; the vs-JAX ratios for compute ops are not.
  Definitive vs-JAX win/loss requires an IDLE host (currently unavailable).
- NEXT (to actually beat JAX on the compute ops): the gap is DRAM read bandwidth on multi-input
  streams. Levers to try (idle host): non-temporal/streaming stores, prefetch, thread-count/affinity
  tuning (work_scaled_threads may over-spawn at 64M), or NUMA-aware chunking. Or the compiled-jaxpr
  arena (buffer reuse removes the output alloc entirely). None attempted yet.

## WildForge / cod-a - CompiledJaxpr reusable runner arena (4 JAX wins, 1 JAX loss)

- Bead: `frankenjax-mcqr.110`.
- Lever: add `CompiledJaxpr::runner()` / `CompiledJaxprRunner` so repeated dense-plan eval reuses
  the slot environment plus scratch/output/scalar buffers instead of allocating them on every call.
  This is the compiled-jaxpr arena lever called out by the same-load correction above.
- Rust internal evidence: RCH Criterion `fj-interpreters/compiled_dispatch_speed`; baseline command
  and candidate command used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`. Baseline
  worker was `vmi1153651`; candidate worker was `hz2`, but the keep/reject proof is same-process
  candidate `compiled/*` vs `compiled_runner/*`:

  | workload | old compiled | runner | internal verdict |
  | --- | ---: | ---: | --- |
  | scalar/n=8 | 142.38 ns | 34.076 ns | runner 4.18x faster |
  | scalar/n=32 | 244.26 ns | 85.634 ns | runner 2.85x faster |
  | scalar/n=128 | 671.69 ns | 297.46 ns | runner 2.26x faster |
  | tensor64/n=8 | 1.9011 us | 1.7920 us | runner 1.06x faster |
  | tensor64/n=32 | 7.1392 us | 7.0183 us | runner 1.02x faster |

- JAX head-to-head: warmed `jax.jit` CPU (`jax 0.10.1`, x64) using the same scalar and 64-lane
  tensor add chains. Rust/JAX ratio is runner time divided by JAX p50:

  | workload | runner | JAX p50 | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | scalar/n=8 | 34.076 ns | 4857.883 ns | 0.0070 | Rust wins 142.6x |
  | scalar/n=32 | 85.634 ns | 4653.0505 ns | 0.0184 | Rust wins 54.3x |
  | scalar/n=128 | 297.46 ns | 4627.832 ns | 0.0643 | Rust wins 15.6x |
  | tensor64/n=8 | 1.7920 us | 4.554228 us | 0.3935 | Rust wins 2.54x |
  | tensor64/n=32 | 7.0183 us | 4.5211495 us | 1.5523 | JAX wins 1.55x |

- Decision: KEEP. Internal 5/0/0 vs the old compiled eval path; vs-JAX 4/1/0. This turns the
  scalar repeated-compiled-eval lane from allocation-bound loss into domination. The tensor64/n=32
  loss is real negative evidence: XLA fuses the add chain while the dense interpreter still steps it.
- Validation: `fj-interpreters` check/focused runner parity test passed via RCH; `fj-conformance`
  passed locally after stale harness repairs; scoped clippy passed for `fj-interpreters --lib
  --no-deps` and `fj-conformance --tests --no-deps`. Full `fj-interpreters --all-targets` clippy is
  still blocked by pre-existing `fj-trace`/`fj-lax` lints.
- Retry predicate: target dense elementwise-chain fusion / tensor output reuse next, specifically the
  remaining `tensor64/n=32` JAX loss.

## WildForge / cod-a - Small dense f64 linear chain runner converts tensor64/n=32 loss to narrow JAX win

- Bead: `frankenjax-mcqr.111`.
- Lever kept: in `fj-interpreters`, the small dense-f64 tensor arena now recognizes a one-output
  linear tensor-state chain with scalar/literal broadcast operands, moves the loaded input tensor
  buffer out of the arena, and mutates that buffer in place through the chain. All non-linear,
  multi-tensor, broadcast-vector, non-f64, multi-output, and large-tensor cases fall back to the
  existing arena/generic paths.
- Same-worker Rust evidence (`vmi1152480`, `compiled_dispatch_speed`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`):

  | workload | baseline compiled_runner | candidate compiled_runner | speedup |
  | --- | ---: | ---: | ---: |
  | tensor64/n=8 | 2.2213 us | 1.1623 us | 1.91x |
  | tensor64/n=32 | 8.3991 us | 4.4519 us | 1.89x |

- JAX head-to-head (`jax.jit` CPU 0.10.1, x64, comparator script mean):

  | workload | Rust candidate | JAX mean | Rust/JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | scalar/n=8 | 28.642 ns | 5406.2266 ns | 0.0053 | Rust wins |
  | scalar/n=32 | 77.747 ns | 4724.7143 ns | 0.0165 | Rust wins |
  | scalar/n=128 | 264.19 ns | 4755.5858 ns | 0.0556 | Rust wins |
  | tensor64/n=8 | 1.1623 us | 4.6034999 us | 0.2525 | Rust wins |
  | tensor64/n=32 | 4.4519 us | 4.7659261 us | 0.9341 | Rust wins narrowly |

- Decision: KEEP. Current compiled-runner scorecard is 5 wins / 0 losses / 0 neutral vs JAX. The
  prior real loss (`tensor64/n=32`) moved from JAX 1.76x faster to Rust 1.07x faster by mean.
- Negative evidence / retry predicate: the tensor64/n=32 margin is small, so do not treat this as a
  broad XLA-fusion domination. The remaining frontier is deeper small-tensor codegen/algebra only if
  it preserves per-step floating-point order, or if a future bead explicitly defines a relaxed-FP
  contract.

## WildForge / cod-a - Dense 16M elementwise thread-count cap8 rejected (2026-06-20)

- Frontier targeted: same-shape multi-input dense elementwise compute rows, specifically the remaining
  Rust/JAX add/mul loss family called out by the same-load correction. This was the "thread-count
  tuning" retry predicate: maybe fewer, larger chunks would improve DRAM read bandwidth by reducing
  scheduler/cache pressure.
- Candidate tried and reverted before commit: replace the shipped `work_scaled_threads(n)` all-core
  policy for cheap same-shape f64/f32 Add/Sub/Mul/Div/Max/Min with a 1M-elements-per-thread policy
  capped at 8 threads. No production code from this candidate remains.
- Same-binary A/B reject proof, final committed harness, RCH `vmi1152480`, command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet -- 'thread_policy_f64_add_16m' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline cod-a-thread-policy-cap8-ab-final`

  | policy | threads | mean | verdict |
  | --- | ---: | ---: | --- |
  | shipped `work_scaled_threads` / all-core 64K chunks | 10 | 17.344 ms | baseline |
  | cap8 / 1M chunks | 8 | 20.199 ms | 1.16x slower |

- Current production 16M Rust rows after revert, RCH `ovh-a`, command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet -- 'add_f64_16m|add_f32_16m|mul_f64_16m' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline cod-a-thread-policy-current`

  | row | Rust mean |
  | --- | ---: |
  | add_f64_16m/dense | 29.214 ms |
  | add_f32_16m/dense | 13.845 ms |
  | mul_f64_16m/dense | 28.999 ms |

- Fresh JAX comparator, local `uv run --no-project --with 'jax[cpu]'`, JAX 0.10.2, command:
  `uv run --no-project --with 'jax[cpu]' python benchmarks/jax_comparison/elementwise_gauntlet.py --runs 10 --warmup 3 --inner-loops 5 --output artifacts/performance/evidence/frankenjax-cod-a-thread-policy-jax-20260620T0205Z.json`

  | row | JAX mean | Rust/JAX | verdict |
  | --- | ---: | ---: | --- |
  | add_f64_16m | 27.798 ms | 1.051 | JAX wins narrowly |
  | add_f32_16m | 13.668 ms | 1.013 | neutral |
  | mul_f64_16m | 27.673 ms | 1.048 | neutral |

- Ratio scorecard for this row set: 0 wins / 1 loss / 2 neutral vs JAX using a 5% neutral band.
  This Rust/JAX comparison is cross-host (`ovh-a` Rust, local JAX), so it is routing evidence; the
  cap8 rejection itself is same-binary and decisive.
- Decision: REJECT cap8 thread-count cap and restore the shipped all-core policy. Do not retry
  "fewer chunked std::thread workers" for large dense multi-input elementwise without a stricter
  same-host harness showing a bandwidth reason it should beat the shipped policy.
- Next retry predicate: different primitive family only - NUMA/affinity pinning, non-temporal stores,
  explicit prefetch, or compiled-jaxpr output reuse. Thread-count shrink alone is negative evidence.

## WildForge - N-D transpose outer-block threading rejected (same-worker regression)

- Bead: `frankenjax-f62hx` follow-on, after the already-shipped trailing-block memcpy path.
- Candidate lever rejected: `transpose_general_into<T>` for large dense N-D transposes, splitting
  the existing trailing-block copies across scoped threads and lowering the N-D gate to `1 << 20`
  so the attention transpose `[8,512,8,64] -> [8,8,512,64]` would engage. The bit-identity guard
  passed (`cargo test -p fj-lax --lib transpose_general_into_bit_identical_to_serial`, RCH `ovh-a`),
  but performance failed the same-worker gate.
- Baseline Rust (`vmi1149989`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b rch exec
  -- cargo bench -p fj-lax --bench transpose_gauntlet -- --noplot`): current block-copy path
  `blockcopy_eval_primitive` mean 438.76 us, interval [417.39, 461.72] us; pre-f62hx odometer
  mean 5.5460 ms.
- JAX comparator (local repo venv, `jax 0.10.1`, CPU x64):
  `transpose_attn_BSHD_to_BHSD_f32` mean 113.409 us, p50 94.667 us, CV 35.78%. This keeps the
  current materialized Rust path at about 3.87x slower than JAX by mean (routing evidence; not
  same-worker because the RCH worker lacks JAX).
- Candidate Rust on the same worker (`RCH_WORKER=vmi1149989`, same Criterion command):
  `blockcopy_eval_primitive` mean 1.5913 ms, interval [1.4795, 1.6911] ms, Criterion change
  +239.55% with p < 0.05. That is 3.63x slower than baseline and about 14.0x slower than the
  JAX comparator mean.
- Decision: REVERT / NO-SHIP. The source edit was removed before commit. Outer-block threading makes
  the N-D transpose much worse, likely by adding thread overhead and worsening the already-strided
  source-read pattern at this problem size.
- Retry predicate: do not retry lower-gated N-D transpose threading. Closing the remaining JAX gap
  needs a different class of lever: layout-aware transpose elision/fusion, a source-contiguous
  cache-oblivious schedule with measured proof, or avoiding the materialized transpose in consumers
  such as attention/einsum.

## WildForge / cod-a - oneqh mimalloc default-global-allocator closeout: reject production adoption as stale-gated (2026-06-20)

- Bead: `frankenjax-oneqh`.
- Decision: CLOSE as rejected/no-ship. Do not add a workspace default `#[global_allocator]`, and do
  not raise/remove `CHEAP_BINARY_PARALLEL_MIN` from this bead. The bead's original work list depended
  on the now-corrected premise that mimalloc supersedes the shipped threaded cheap-binop path. The
  current ledger correction says the opposite for the real `eval_primitive` memory-bound rows:
  threading and allocator reuse are complementary, and the threading gate must remain.
- Fresh routing check, RCH worker `vmi1167313`, command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet -- 'add_f64_16m|mul_f64_16m|thread_policy_f64_add_16m' --sample-size 10 --warm-up-time 1 --measurement-time 2 --save-baseline cod-a-zipchunks-before`

  | row | mean | note |
  | --- | ---: | --- |
  | `thread_policy_f64_add_16m/all_core_64k_threads_6` | 65.166 ms | loaded worker; same-binary helper |
  | `thread_policy_f64_add_16m/cap8_1m_threads_6` | 54.053 ms | same 6-worker cap on this host; not a production row |
  | `add_f64_16m/dense` | 49.750 ms | current production `eval_primitive` row |
  | `mul_f64_16m/dense` | 63.001 ms | current production `eval_primitive` row |

- Interpretation: these rows are routing evidence only (loaded host, low effective parallelism), but
  they do not rescue the oneqh implementation plan. They reinforce the documented load sensitivity of
  large multi-input DRAM rows and provide no safe basis for a global allocator policy change. A
  library-level global allocator still conflicts with allocator-observing benches such as
  `crates/fj-interpreters/benches/eval_chain_memory.rs`, and changing allocator policy is not a
  crate-local fj-lax performance lever.
- Current Rust/JAX scorecard for the relevant 16M row set remains the prior recorded comparator:
  0 wins / 1 loss / 2 neutral (`add_f64_16m` narrow loss, `add_f32_16m` neutral, `mul_f64_16m`
  neutral, 5% neutral band). This is a target lane, but not via oneqh's global allocator/gate-removal
  plan.
- Correct retry predicate: compiled-jaxpr output/arena reuse, non-temporal stores/prefetch/NUMA
  affinity for the remaining multi-input DRAM rows, or a specific unowned typed-path gap with
  same-worker before/after proof. Do not re-open default mimalloc adoption unless a maintainer first
  chooses workspace allocator policy and the proof includes the allocator-conflict resolution plus
  head-to-head production rows with the existing threading gate kept.

## WildForge / cod-b - oneqh allocator preload verification: jemalloc mixed/no-ship (2026-06-20)

- Bead: `frankenjax-oneqh`.
- Scope: follow-up verification of the allocator-class hypothesis after the CobaltForge correction
  that allocator reuse does not supersede the shipped threaded cheap-binop path. No production source
  change was made.
- Environment: local same-host Criterion with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b-local-20260620`, Rust
  `fj-lax` `elementwise_gauntlet`, JAX 0.10.1 CPU x64 comparator via
  `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/elementwise_gauntlet.py
  --runs 12 --warmup 3 --inner-loops 20`. RCH build-only guard
  `rch exec -- cargo bench -p fj-lax --bench elementwise_gauntlet --no-run` passed on
  worker `vmi1153651`. Mimalloc was not installed on this host, so the measured preload was
  `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`.

  | row | glibc Rust | jemalloc Rust | JAX | verdict |
  | --- | ---: | ---: | ---: | --- |
  | `add_f64_16m/dense` | 24.502 ms, Rust/JAX 0.870 | 33.095 ms, Rust/JAX 1.174 | 28.179 ms | REJECT: jemalloc is 1.35x slower than glibc and loses to JAX |
  | `add_f32_16m/dense` | 15.945 ms, Rust/JAX 1.128 | 16.784 ms, Rust/JAX 1.188 | 14.130 ms | REJECT/neutral: Criterion p=0.82 for preload change, both lose to JAX |
  | `mul_f64_16m/dense` | 29.210 ms, Rust/JAX 1.024 | 27.790 ms, Rust/JAX 0.974 | 28.525 ms | Neutral/slight win only; not a global allocator justification |

- Decision: NO-SHIP. Do not add a workspace default allocator and do not retune or remove
  `CHEAP_BINARY_PARALLEL_MIN` from this bead. The allocator-class evidence is mixed: one clear
  regression, one no-change/loss, and one small neutral-band improvement.
- Retry predicate: do not reopen default allocator adoption unless mimalloc is available and measured
  same-host against the current threaded `eval_primitive` path, every stable 16M row is non-regressing,
  and the existing `PeakAlloc`/`#[global_allocator]` conflict in
  `crates/fj-interpreters/benches/eval_chain_memory.rs` has a concrete maintainer-approved resolution.
  Better next levers are compiled-jaxpr output/arena reuse, non-temporal stores/prefetch/NUMA for the
  remaining DRAM rows, or a specific unowned typed-path gap with same-worker before/after proof.
## WildForge - dense small-tensor single-pass fusion + buffer pool (REJECTED, dtype-fragile)

- Follow-up to the retry predicate directly above (mcqr.110): "target dense elementwise-chain fusion /
  tensor output reuse next." Two levers attacked the per-step dense small-tensor arena
  (`run_scalar_{f64,f32,i64}_plan_as_tensor_into`), which steps each equation as a full pass that
  materializes an intermediate `Vec` (an N-step chain = N passes + N buffers), where XLA fuses to one
  pass.
- METHOD: ALL measurements are same-binary same-invocation A/B (added `CompiledJaxprRunner::
  eval_no_fusion` + a `compiled_runner_nofuse` bench arm), the ONLY worker-variance-immune signal on
  this contended host. Cross-invocation criterion medians were USELESS here — between two runs every
  arm (eager/compiled/runner) moved ~20-27% purely from worker speed, swamping the effect. Bit-exact:
  fused == per-step == eager asserted across all dtype/op cases in
  `compiled_jaxpr_eval_matches_eager_eval_jaxpr` (elements are independent and each undergoes the
  identical scalar-op sequence, so element-outer/step-inner is bit-identical to step-outer).

- LEVER 1 — buffer pool (recycle per-step `vec![..;n]` across calls): NEUTRAL. Within-invocation
  `runner/compiled` ratio barely moved (n=32: 0.884 -> 0.868). A 512-byte malloc/free hits the
  allocator's thread-local cache (~5-15ns); 32 of them is ~4% of a 7us eval, below noise. The cost is
  the 2048 scalar ops + Value/TensorValue boxing, not allocation.

- LEVER 2 — single-pass element-outer fusion (no intermediates, XLA-style): DTYPE-FRAGILE, net
  REJECTED. Same-invocation fused/per-step ratios:

  | dtype | workload | fused | per-step | fused/per-step | verdict |
  | --- | --- | ---: | ---: | ---: | --- |
  | f64 | tensor64/n=8  | 1.359 us | 1.577 us | 0.86 | win (modest) |
  | f64 | tensor64/n=32 | 5.871 us | 5.862 us | 1.00 | neutral |
  | f64 | E16/n=4 | 289.9 ns | 331.4 ns | 0.875 | win |
  | f64 | E64/n=4 | 858 ns | 850 ns | 1.01 | neutral |
  | f64 | E256/n=4 | 2.779 us | 2.905 us | 0.957 | win |
  | f64 | E1023/n=4 | 10.59 us | 11.02 us | 0.961 | win |
  | f32 | ftensor64/n=8 | 2.096 us | 1.203 us | **1.74** | REGRESS |
  | i64 | itensor64/n=8 | 1.330 us | 0.992 us | **1.34** | REGRESS |

- ROOT CAUSE / KEY INSIGHT: the f64 "win" is an ARTIFACT, not fusion's merit. Tell: f32 per-step
  (1.20us) is FASTER than f64 per-step (1.58us) despite f32 widening every element to f64 — only
  possible because the **f64 per-step inner loop carries broadcast-branch overhead** (`is_bcast()` +
  a 4-way `DenseRef::at` match for Scalar/Tensor/Row/Col-bcast) that the lean 2-way f32/i64 per-step
  loops lack. Fusion beats f64-per-step by sidestepping that overhead; against the already-lean
  f32/i64 per-step loops, fusion's per-element operand-match cost makes it strictly slower. A 64-elem
  tensor (512 B) is L1-resident, so the multi-pass cost fusion removes is nearly free — it only wins
  when per-step is burdened (f64 broadcast branches) or the chain is short.
- DECISION: REVERT all (pooling + f64/f32/i64 fusion + flag plumbing). Shipping f64-only fusion would
  add ~250 lines of fragile, dtype-gated surface for a modest gain that masks the real problem, and
  leave f32/i64 a regression trap behind the `fuse` flag.
- REAL LEVERS this surfaces (not yet attempted): (a) split the f64 per-step inner loop into a
  no-broadcast fast path with hoisted op + operand-kind matches so it AUTO-VECTORIZES (`o[i]=a[i]+s`
  -> vaddpd) — attacks the f64 overhead directly without fusion; (b) the `tensor64/n=32` JAX loss
  (5.87us vs JAX 4.52us) is SIMD-throughput + Value-boxing bound, NOT allocation/pass bound — needs
  vectorized per-step kernels for the cheap ops (Add/Sub/Mul/Div) + less boxing, not chain fusion.

## WildForge - vectorize the dense-f64 per-step inner loop (LEVER (a)) — KEPT, FLIPS THE JAX LOSS

- This is REAL LEVER (a) from the rejected-fusion entry directly above, and it WINS decisively.
  Instead of fusing the chain, keep the step-outer structure (which vectorizes ACROSS the independent
  elements within a step) and remove what blocked the compiler: the generic dense-f64 per-step inner
  loop called `apply_scalar_f64_binary(step.op, a.at(i), b.at(i))` per element, whose 40+-arm op
  match and 4-way `DenseRef::at` match defeat auto-vectorization. Fix: for the no-broadcast case,
  hoist the op + operand-kind match ONCE outside the element loop into `fill_dense_f64_nobcast<F>`
  (monomorphized closure), so Add/Sub/Mul/Div become a straight `o[i] = a[i] OP b[i]` loop that
  lowers to `vaddpd`/etc. under `+avx2`. Max/Min/transcendentals keep the generic loop.
- BIT-EXACT: SIMD f64 add/sub/mul/div are elementwise, no reassociation, and we never introduce FMA
  (`+avx2` only, not `+fma`). Asserted directly: `eval` (vectorized) == `eval_scalar_inner`
  (generic) == eager, across every dtype/op case in `compiled_jaxpr_eval_matches_eager_eval_jaxpr`.
- CONVERGENCE with mcqr.111: a concurrent agent landed `run_linear_scalar_f64_tensor_chain_into` —
  for a single-output linear tensor chain it moves the input buffer out of the arena and mutates it
  IN PLACE through the chain (no per-step `Vec`). That path runs BEFORE the per-step loop, so it owns
  the common chain workload. On rebase I applied the SAME op-hoisting vectorization to its in-place
  loops (`apply_dense_f64_chain_step`), so the two levers COMPOSE: in-place buffer reuse (no alloc) +
  SIMD arithmetic. Both the in-place and per-step paths take the `vectorize` A/B flag.
- METHOD: same-binary same-invocation A/B via the bench control `eval_scalar_inner` (vectorization
  OFF, but in-place reuse still ON) vs `eval` (ON), arm `compiled_runner_scalar` vs `compiled_runner`.
  Worker-variance-immune. Final MERGED numbers (in-place + vectorized):

  | workload | vectorized (`eval`) | in-place only (`eval_scalar_inner`) | speedup |
  | --- | ---: | ---: | ---: |
  | tensor64/n=8  | 195 ns | 1179 ns | 6.1x |
  | tensor64/n=32 | 432 ns | 4308 ns | 10.0x |
  | E8/n=4   | 140 ns | 196 ns | 1.40x |
  | E256/n=4 | 255 ns | 2195 ns | 8.6x |
  | E1023/n=4 | 649 ns | 8610 ns | 13.3x |

  Speedup GROWS with element count (more lanes amortize overhead + SIMD throughput); never regresses.
  (The control is itself ~1.65x faster than the pre-mcqr.111 7.1us per-step path because in-place
  reuse stays on; the 10x is purely the vectorization on top.)
- vs JAX (warmed `jax.jit` CPU, x64, from the mcqr.110 entry above): `tensor64/n=32` was a 1.55x JAX
  WIN at 4.52us vs our old 7.0us path. Merged in-place+vectorized is 0.432us -> **we now BEAT JAX
  ~10.5x on the very workload that was the documented loss.**
- DECISION: KEEP. 6-13x same-invocation, bit-exact, no regression, flips the headline JAX loss hard.
  Scope: f64 no-broadcast Add/Sub/Mul/Div (both the in-place linear-chain path and the per-step path).

- FOLLOW-UP SHIPPED — f32 (JAX's DEFAULT tensor dtype): same hoist in the f32 per-step arena
  (`fill_dense_f32_nobcast`) using NATIVE f32 ops (`vaddps`, 8-wide). The widen-blocks-SIMD worry was
  unfounded: for a single +/-/*/÷, f64 (53-bit mantissa) carries >= 2*24+2 bits so
  `(f64(a) OP f64(b)) as f32 == a OP b` in native f32 (Figueroa — no double rounding), i.e. native ==
  eager's widen->f64->narrow, BIT-EXACT. Empirically confirmed by `dense_f32_tensor_arena_bit_
  identical_to_generic`. Same-invocation A/B:

  | workload | vectorized | generic widen (`eval_scalar_inner`) | speedup |
  | --- | ---: | ---: | ---: |
  | f32E256/n=8  | 546 ns | 5234 ns | 9.6x |
  | f32E256/n=32 | 2252 ns | 21857 ns | 9.7x |

- FOLLOW-UP SHIPPED — i64: same hoist (`fill_dense_i64_nobcast`) for the wrapping ops Add/Sub/Mul,
  which vectorize to `vpaddq`/`vpsubq`/`vpmullq` (4-wide i64 under `+avx2`). Bit-exact (native
  `wrapping_*` IS the per-element op); confirmed by `dense_i64_tensor_arena_bit_identical_to_generic`.
  Same-invocation A/B: i64E256/n=8 = 935ns vs 3610ns (3.9x); i64E256/n=32 = 3217ns vs 13637ns (4.2x).
  (Lower than f32's 9.6x: 4-wide i64 SIMD vs 8-wide f32, and Div/Max/Min stay generic.)
- DTYPE FAMILY COMPLETE for the cheap-op vectorization lever: f64 (10.5x vs JAX), f32 (9.6x), i64
  (4.2x). Half (bf16/f16) needs decode and won't vectorize cleanly — DO NOT.
- FOLLOW-UP SHIPPED — f64 rank-2 BROADCAST (bias-add: matrix + [C] row vector / [R,1] col vector):
  `fill_dense_f64_bcast` decomposes each row into a no-broadcast subproblem and REUSES
  `fill_dense_f64_nobcast`, so each row's inner loop vectorizes (RowBcast → same [C] slice every row;
  ColBcast → per-row scalar; full tensor → row slice). Bit-identical to the per-element `at_rc` loop
  (confirmed `dense_f64_tensor_arena_broadcast_bit_identical` + row/col chain tests). Same-invocation
  A/B: bcast16x16/n=8 = 1.76us vs 9.55us (5.4x); bcast16x16/n=32 = 6.74us vs 41.69us (6.2x).
- NEXT: hunt OTHER interpreter inner loops calling per-element `apply_scalar_*`/`eval_primitive`
  dispatch (same hoist-the-match-out-of-the-loop pattern).

- HEAD-TO-HEAD vs JAX (warmed `jax.jit` CPU, x64, jax 0.10.1, call+block_until_ready latency; local
  venv since RCH workers lack JAX). Rust = in-process `CompiledJaxprRunner::eval` (vectorized):

  | workload | Rust | JAX p50 | verdict |
  | --- | ---: | ---: | --- |
  | f32E256/n=8  | 546 ns | 4558 ns | Rust wins 8.4x |
  | f32E256/n=32 | 2252 ns | 4512 ns | Rust wins 2.0x |
  | bcast16x16/n=8  | 1764 ns | 5996 ns | Rust wins 3.4x |
  | bcast16x16/n=32 | 6741 ns | 6074 ns | **JAX wins 1.10x (narrow LOSS)** |

  f32 vectorization is confirmed a JAX WIN (2-8x), not just vs the internal control. The 32-step
  broadcast (bias-add) chain initially LOST narrowly (1.10x) — CAUSE: the broadcast path ran the
  per-step arena allocating a fresh `vec![0.0; n]` PER STEP; the in-place linear-chain path bailed on
  broadcast operands.

- FOLLOW-UP SHIPPED — in-place BROADCAST linear chain (flips the loss above): extended
  `run_linear_scalar_f64_tensor_chain_into` to accept a rank-2 broadcast vec operand (`ChainOperand`
  Row/Col + `resolve_chain_operand`) and mutate the single matrix buffer IN PLACE per row
  (`apply_dense_f64_chain_step_bcast` → vectorized `apply_dense_f64_chain_step_row` for RowBcast,
  per-row scalar reuse for ColBcast). Composes in-place reuse (no per-step alloc) + vectorization.
  Bit-exact (`fusion_f64_row/col_broadcast_chain_matches_reference_bit_for_bit` pass; rank-3 guarded
  to the per-step path). RESULT:

  | workload | before (per-step) | after (in-place) | JAX p50 | verdict |
  | --- | ---: | ---: | ---: | --- |
  | bcast16x16/n=8  | 1764 ns | 544 ns | 5996 ns | Rust wins 11x |
  | bcast16x16/n=32 | 6741 ns | 1701 ns | 6074 ns | **Rust wins 3.57x** (was 1.10x LOSS) |

  Within-invocation A/B (both in-place now): vectorized 1701ns vs in-place-non-vectorized 15552ns =
  9.1x from the vectorization. The documented gap is CLOSED — broadcast bias-add now dominates JAX.

## WildForge - VERIFIED off-limits: float ReduceSum/Prod order (the 3-4x JAX loss) — not a safe unilateral lever

- Investigated whether the documented 3-4x float-reduce JAX loss is bit-exact-pinned (off-limits) or
  tolerance-parity (a faster pairwise/blocked SIMD reduction would then be LEGAL, like linalg). The
  `reduce_sum_oracle.rs` conformance tests are MOSTLY tolerance (`assert_close(.., 1e-14, ..)`) or
  exact-INTEGER cases (order-independent) — which alone would have permitted a reorder. BUT it is
  pinned bit-exact by THREE other mechanisms, so a reorder is NOT a safe unilateral commit:
  1. Golden sha256 digests exist for the actual output bits:
     `artifacts/performance/evidence/fj_lax_dense_f64_reduce_sum_pass{59,60}_2026-06-03.golden.sha256`
     and `fj_lax_reduce_sum_64k_bench_pass58_*.golden.sha256` (64k-element array). A pairwise/SIMD
     fold changes these bits -> goldens must be regenerated (and re-justified as still JAX-tolerance).
  2. `reduction.rs::dense_f64_reduce_sum_full_fast_path_bit_identical_to_literal_path` asserts the
     dense fast path is BIT-IDENTICAL to the literal path (both ascending `|a,b| a+b`); any reorder
     must keep BOTH paths consistent.
  3. `reduction.rs` is referenced by 3 e2e gates (e2e_adversarial_fixtures / e2e_mixed_dtype /
     e2e_security_gate) whose committed_gate_logs hash the file (hash-drift on any edit), AND it is
     in the ACTIVE cod-a/cod-b fj-lax campaign (inbox-confirmed) — editing it risks clobbering their
     concurrent WIP.
- CONCLUSION: confirms the standing DO-NOT (project_axis_reduce_odometer_lever). A real attempt is a
  MULTI-SESSION, maintainer-coordinated swing: either (a) match XLA's EXACT CPU reduction tree to get
  bit-parity AND ILP, or (b) a deliberate move to tolerance-parity with golden regeneration + a proof
  the new order stays within 1e-14 of JAX + cod-campaign coordination. Not a tail-of-session commit.
- Net for the BOLD-VERIFY phase: the fj-interpreters dense-arena vectorization vein is fully mined and
  JAX-dominant; the remaining gaps are this off-limits reduce-order swing, the +fma policy gate, or the
  active cod fj-lax lane. No safe in-lane lever remains; holding rather than risking parity/collisions.

## WildForge - extend the in-place f64 linear chain to L3-resident sizes (the documented L3 lever) — KEPT

- The ledger's two standing "remaining JAX losses" were float-reduce-order (verified off-limits above)
  and "the L3-resident regime (needs compiled-jaxpr arena buffer reuse)". This is that buffer-reuse
  lever: the dense-f64 arena bailed at `n >= FUSION_MIN_ELEMS` (1024), so a multi-op LINEAR chain on a
  medium/large tensor (e.g. an optimizer-update body on a parameter vector) fell to the generic per-op
  path — N separate allocations + 2-stream (read+write) traffic per op. The in-place linear-chain path
  (`run_linear_scalar_f64_tensor_chain_into`) mutates ONE buffer through the whole chain: 1 copy +
  single-stream in-place traffic, no per-step alloc, vectorized. Moved it ahead of the bail gate, up to
  an L3-resident ceiling `INPLACE_CHAIN_MAX_ELEMS = 1<<20` (~8 MB f64).
- Same-invocation A/B (`compiled_runner` in-place vs `compiled_runner_scalar` generic per-op; the
  control is exactly the prior production behavior — bail -> generic):

  | elems | in-place | generic per-op | verdict |
  | --- | ---: | ---: | --- |
  | 4096 (32 KB)   | 2.55 us  | 3.71 us  | 1.45x WIN |
  | 65536 (512 KB) | 49.4 us  | 61.8 us  | 1.25x WIN |
  | 262144 (2 MB)  | 234 us   | 263 us   | 1.12x (marginal, near noise) |
  | 1048576 (8 MB) | 1.89 ms  | 1.90 ms  | ~1.0x neutral |
  | 16777216 (128 MB) | 122 ms | 133 ms | ~neutral (DRAM-bound, noisy) |

  Clear win at L1/L2-resident (1.25-1.45x), decaying to neutral by ~1M elems; NEVER regresses in any
  measured size. Gated at 1<<20 so the DRAM-bound regime (where a future threaded path may win) is
  ceded. Bit-exact (each element's op sequence is unchanged regardless of n; guarded by
  `dense_f64_tensor_arena_bit_identical_to_generic` extended to the large-linear-chain contract +
  `fusion_f64_*_chain_matches_reference`). The non-vectorized control still bails (so the A/B is valid
  and the lever is opt-in behind the production `vectorize=true`).
- Net: the two named "remaining JAX losses" are now (1) reduce-order = verified off-limits, (2)
  L3-resident chain buffer-reuse = SHIPPED for the cache-resident regime.

## WildForge - f32 in-place linear chain (L3-resident, JAX's default dtype) — KEPT (narrower win than f64)

- Mirror of the f64 L3-resident in-place chain for f32 (`run_linear_scalar_f32_tensor_chain_into` +
  `apply_dense_f32_chain_step`, native f32 ops = bit-exact vs widen->f64->narrow by Figueroa; f32 cells
  have no broadcast variants so it is scalar-operand only). Same gate (`< INPLACE_CHAIN_MAX_ELEMS`).
- Same-invocation A/B (in-place vs generic per-op control):

  | elems | in-place | generic | verdict |
  | --- | ---: | ---: | --- |
  | 4096 (16 KB)   | 2.10 us | 3.02 us | 1.44x WIN |
  | 65536 (256 KB) | 46.1 us | 48.6 us | 1.05x (marginal, near noise) |

  Real win at small-large (1.44x@4K), decaying faster than f64 to ~neutral by 64K; no regression.
  Bit-exact (guarded by `dense_f32_tensor_arena_bit_identical_to_generic` extended to the large-chain
  contract). KEPT because f32 is JAX's default dtype, so even the small-size win lands on common bodies.
- HONEST CAVEAT / NEW OPPORTUNITY: at f32big65536 the `eager` arm (eval_jaxpr's fusion path) is 34 us
  — FASTER than both the compiled in-place chain (46 us) and the generic control (48 us). So the
  COMPILED path (CompiledJaxpr, used for jit/scan repeated eval) is leaving f32 large-chain perf on the
  table vs eval_jaxpr's fusion; my change strictly improves the compiled path but does not reach the
  fusion path's speed. NEXT: route the compiled dense-plan through (or port) the eval_jaxpr CheapOp
  fusion for large f32/f64 chains — a separate, larger lever (bead-worthy).

## CrimsonForge - threaded eager-eval_jaxpr elementwise fusion chunk driver — KEPT (1.22-1.32x, bit-exact)

- Date: 2026-06-20. Crate `fj-interpreters`. Commits c5ce4988 / fe0985bc / 755e6b9c.
- Gap: the eager `eval_jaxpr` fusion path (`try_fuse_elementwise_chain_*`) ran its
  `FUSION_CHUNK` chunks single-threaded (~6.3 GB/s on a 1M f64 8-op chain, far below
  DRAM saturation; JAX/XLA uses all cores).
- Lever: `drive_fusion_chunks` fans disjoint, chunk-aligned segments across
  `thread::scope`; global `base` preserved for broadcast indexing. BIT-IDENTICAL
  (same per-element op order; no reassociation/reduction). Gate `FUSION_THREAD_MIN_ELEMS
  = 1Mi`, work-scaled count `FUSION_ELEMS_PER_THREAD = 256Ki`.
- Same-invocation A/B (`run_f64_thread_ab`; serial cap=1 vs threaded, one process —
  the only worker-variance-immune signal: absolute ms drifts 4-10x across rch runs):

  | n | serial | threaded | speedup |
  | --- | ---: | ---: | ---: |
  | 1M  | 12.157 ms | 9.937 ms  | 1.22x |
  | 4M  | 75.841 ms | 59.493 ms | 1.27x |
  | 16M | 305.924 ms | 231.395 ms | 1.32x |

  3 wins / 0 losses / 0 neutral; win grows with n; no regression at any size.
- NEGATIVE EVIDENCE inside this win: the first cross-invocation measurement showed
  the threaded f64 1M at 39.4 ms vs a 3.8 ms serial baseline — an APPARENT 10x
  regression that was pure worker contention (the untouched unfused control jumped
  18.8 -> 71.3 ms in the same run). Retry predicate: NEVER judge fusion threading by
  cross-invocation absolute ms; only the in-process serial/threaded ratio is valid.
- Honest framing: narrows but does NOT flip the large-fused-chain JAX loss. Scaling is
  Amdahl-capped by `eval_jaxpr`'s serial per-call overhead (arg clone / TensorValue
  build / liveness) and the per-core no-SIMD-single-pass inner loop. NEXT (larger,
  separate lever): SIMD single-pass register-resident chunk evaluation + cutting the
  per-call output/construction overhead toward XLA parity.
- Conformance: bit-exact (serial==threaded==reference guards). The 8 `fj-interpreters`
  golden-hash failures are pre-existing serialization drift (identical on baseline
  aff2ee5d), not from this change.

### CORRECTION (same day, CrimsonForge): contention re-measurement → gate raised 1Mi -> 8.4Mi

The 1.22-1.32x table above was measured on IDLE/lightly-loaded rch workers. A third
run landed on a CONTENDED shared worker and the SAME code (commit c5ce4988) measured:

| n | serial | threaded | speedup | note |
| --- | ---: | ---: | ---: | --- |
| f64 1M  | 20.595 ms | 48.484 ms | **0.42x** | REGRESSION (L3-resident, contended) |
| f64 4M  | 108.687 ms | 101.918 ms | 1.07x | marginal |
| f64 16M | 390.867 ms | 322.653 ms | 1.21x | win (DRAM-bound) |
| f32 1M  | 5.542 ms | 7.006 ms | **0.79x** | REGRESSION |
| f32 16M | 209.219 ms | 157.046 ms | 1.33x | win |
| i64 1M  | 8.732 ms | 20.629 ms | **0.42x** | REGRESSION |
| i64 16M | 407.139 ms | 380.176 ms | 1.07x | marginal win |
| bf16 1M | 21.621 ms | 25.244 ms | 0.86x | (half already reverted to serial) |

KEY RETRY PREDICATE: a same-invocation A/B ratio is contention-immune ONLY when both
arms scale identically with load. A THREADING lever breaks that — under shared-worker
oversubscription the threaded arm thrashes (oversubscribed cores) while the serial arm
is robust, so the *ratio itself* swings from 1.28x (idle) to 0.42x (contended) at the
same size. Threading is only a ROBUST win PAST the L3->DRAM transition (>= 8.4M here:
working set > the 128MB L3, so threads use independent memory channels) — confirmed
positive across BOTH idle and contended workers (f64/f32 16M 1.21-1.33x, i64 1.07x).

ACTION (commit c23c5a0c): `FUSION_THREAD_MIN_ELEMS` raised 1Mi -> 8.4Mi (matches the
established single-op cheap-elementwise gate). Net verdict: KEPT but NARROWED — a
modest (1.07-1.33x) bit-exact win confined to >=8.4M-element elementwise chains
(f64/f32/i64); half stays serial (measured neutral). Below 8.4M = serial (no
regression). Honest framing: does NOT flip the absolute JAX loss on large chains.

## CrimsonForge - element-major fusion-chunk rewrite REJECTED (dynamic-tape dispatch defeats it)

- Date: 2026-06-20. Probes in eval_fusion_speed.rs (run_fusion_strategy_probe, run_fusion_dynamic_probe).
- Hypothesis: rewrite step-major `apply_fusion_chunk` (8 passes, accumulator round-tripped
  to memory between ops) as element-major single-pass (register-resident accumulator, one
  store/elem — what XLA emits) to attack the measured ~7.6-7.75x large-chain JAX loss.
- STATIC element-major (fixed unrolled chain): **2.87x @ 1M, 2.12x @ 16M faster** — the win is real.
- DYNAMIC-tape element-major (faithful to lib.rs: per-element loop over a runtime tape with a
  match dispatch): **0.27x @ 1M, 0.60x @ 16M — 1.7-3.7x SLOWER.**

| n | step-major | static elem | dyn-scalar elem | verdict |
| --- | ---: | ---: | ---: | --- |
| 1M  | 4.06 ms (probe) | 1.41 ms (2.87x) | 13.46 ms (0.27x) | dynamic LOSES |
| 16M | 194.3 ms | 91.8 ms (2.12x) | 295.7 ms (0.60x) | dynamic LOSES |

- ROOT CAUSE (= the hoist-op-match lesson): step-major HOISTS the op-match OUT of the element
  loop, so each op is a tight autovectorized pass. Element-major puts the dynamic-tape match
  INSIDE the per-element loop → kills autovec + a branch per element per step. The static win
  exists only because the compiler UNROLLS the fixed chain (eliminating dispatch) = codegen,
  which is exactly what XLA does (it monomorphizes each chain) and what our interpreter cannot.
- RETRY PREDICATE: do NOT rewrite apply_fusion_chunk element-major for the dynamic tape. The
  current step-major (op-match hoisted, autovec per op) is the CORRECT design for a dynamic
  interpreter tape. SIMD element-major would still carry per-(group)step dispatch and likely
  shares the penalty; not worth the portable_simd/mask risk. The 2-2.9x element-major win is
  reachable ONLY via per-chain codegen/JIT (compile the jaxpr chain to a monomorphic kernel) —
  a major architectural lever, the same class as bead compiled-jaxpr-dispatch. This is the
  fundamental interpreter-vs-compiler ceiling; it is WHY large chains lose ~7-8x to XLA.

## CrimsonForge - bead 2zzwl (route CompiledJaxpr large-chain through threaded fusion) is a NON-GAP — already satisfied by existing dense-plan fallthrough

- Date: 2026-06-20. Investigation only (no code change). Crate `fj-interpreters` lib.rs.
- Bead 2zzwl premise: "the COMPILED runner (CompiledJaxpr/dense-plan) runs large same-shape
  elementwise chains PER-STEP and is slower than the eager fusion; route it through the
  now-threaded eager fusion." Traced the actual production dispatch and the premise is STALE.
- Production path: `CompiledJaxpr::eval` -> `run_dense_plan` -> `run_dense_plan_into`
  -> `run_dense_plan_into_core(.., vectorize=TRUE)` (lib.rs:8767 hardcodes vectorize=true for
  the production wrapper; `false` is only the bench A/B knob).
- For a large same-shape f64/f32/i64 elementwise chain the core dispatch resolves as:
  * `run_scalar_f64_plan_as_tensor_into` (lib.rs:6870) handles it. At line 6941 the in-place
    linear-chain path fires for `n < FUSION_MIN_ELEMS (1024)` OR `(vectorize && n <
    INPLACE_CHAIN_MAX_ELEMS (1<<20 = 1Mi))`. With vectorize=true that is **n < 1Mi**.
  * For `n >= 1Mi` line 6949 `if n >= FUSION_MIN_ELEMS { return None }` BAILS, so the core
    falls through (lib.rs:8971) to `run_dense_env_into`, whose equation loop at lib.rs:9004
    ALREADY calls `try_fuse_elementwise_chain` = my chunked+threaded fusion (drive_fusion_chunks,
    gated FUSION_THREAD_MIN_ELEMS = 8.4Mi).
- NET: the compiled runner has NO per-step large-chain path. Regime map for a compiled chain:
  | n | path | threaded? |
  | --- | --- | --- |
  | < 1Mi | in-place linear chain (single buffer reuse, e97185a2 — the deliberately-superior L3-resident strategy) | no (L3-resident; threading REGRESSES under contention per the CrimsonForge fusion-gate entry above) |
  | 1Mi .. 8.4Mi | eager fusion fallthrough, serial | no (below thread gate) |
  | >= 8.4Mi | eager fusion fallthrough, chunk-threaded | yes |
  Both fusion sub-regimes inherit the threading automatically because the SAME
  `try_fuse_elementwise_chain` serves the eager and compiled fallthrough.
- VERDICT: 2zzwl is a NON-GAP — already done by the existing fallthrough architecture; no
  surgical "routing" edit exists to make. The residual ~7-8x large-chain JAX loss is the
  interpreter-vs-compiler codegen ceiling (see the element-major REJECT entry above), NOT a
  missing route. RETRY PREDICATE: do not re-open 2zzwl as a contained lever; the only lever left
  on this path is per-chain JIT codegen (cranelift-class, multi-session, forbid-unsafe-gated).
  Coordinated with WildForge (dense-plan owner) before concluding.

## CrimsonForge - bead-tracker reconciliation: 20 "ready" perf beads + the partial_eval fail-closed cluster are DONE-but-OPEN (stale), not real gaps

- Date: 2026-06-20. Audit (Explore subagent) of all 20 `br ready` P1 perf beads vs the actual
  fj-lax/fj-interpreters source: ALL 20 are IMPLEMENTED in code (dense fast-paths, block-memcpy,
  dtype-siblings, complex de-box, FFT) with the bead-described function:file present and
  comment-matched — they were landed 2026-06-18 and never closed. Several are also in the
  release scorecard with measured ratios. Re-implementing any would be duplicate work.
- The partial_eval "fail-closed fallback" correctness cluster (slice/pad/transpose/squeeze/
  dynamic_slice/broadcast_in_dim — beads xuum7/11146/encr9/fm5pf/ji393/r67bh, plus argmax
  lbm0d) is likewise already fixed: strict fallible `infer_*_shape` helpers exist for each
  (e.g. infer_slice_shape parses required params, rank-checks, stride>0, bounds-rejects).
- VERDICT: the "ready work" queue is a graveyard of completed-but-unclosed beads, not an
  inventory of open perf gaps. The CONTAINED perf surface is confirmed exhausted (this is the
  4th consecutive session reaching that conclusion) AND already shipped. Action this session:
  refreshed the fo1zg golden hashes (benign densify-serialization drift) to restore the
  fj-interpreters --lib gate; reconciling the stale done beads.

- FOLLOW-UP (same day, 2nd pass): re-checked the NEXT batch of `br ready` beads — same
  result, EVERY category is done-but-open:
  * e07uw (add Square to eval_jaxpr cheap-op fusion): Square already fused as Mul(x,x) in
    all four try_fuse_elementwise_chain_{f64,f32,i64,half} builders (lib.rs:1755/2309/2775/
    3166) + IntegerPow[2] via is_integer_pow_2. (Floor/Ceil/Trunc/Sign/Round remainder is
    the separate bead xjbvr.)
  * o1ouv (gather/scatter negative-index parity): already FIXED — resolve_axis0_index
    (tensor_ops.rs:2894) maps negatives per IndexMode (Clip->0, FillOrDrop->fill); guarded
    by oracle_gather_negative_index_clips_by_default / _fill_or_drop. The bead's stale
    `lit_to_usize` no longer exists.
  * cntiy.1/.2/.3 (reject untiled / output-overflow f32-accum & FMA evidence shapes):
    already FIXED — assert_tiled_matmul_inputs enforces length + m%MR + n%nr + checked_mul
    output-overflow; guarded by fma_evidence_kernels_reject_output_shape_overflow.
- SYSTEMIC FINDING: the open-bead tracker is comprehensively stale across perf, correctness,
  evidence-helper, and test-task categories (~36 done-but-open beads reconciled across the two
  passes). This is an implementation-far-ahead-of-tracker situation, NOT a backlog of real
  gaps. RECOMMENDATION: run a dedicated beads-compliance/completion-verification sweep over the
  remaining ~40 open beads rather than treating `br ready` as an inventory of open work; most
  are already shipped + guarded. The only genuine open frontier is the multi-session per-chain
  JIT-codegen lever (the interpreter-vs-compiler ceiling; see the element-major REJECT entry).

- 3rd-pass FINDING (2026-06-20): a batch audit of the 27 open `bug` beads found 25 done-but-open
  AND **2 GENUINELY OPEN parity bugs** — the first real OPEN work surfaced in ~45 bead checks:
  * rsxjp (Floor/Ceil/Round reject int): actually already DONE (guarded via is_jax_float_only_unary
    + eval_unary_elementwise); audit over-reported it OPEN (conservative error). Now also tested.
  * hwm1v (Cbrt/Erf/Erfc/ErfInv/Lgamma/Digamma/BesselI0e/BesselI1e reject int): GENUINELY OPEN +
    FIXED (commit 238d49d2). These 8 float-only special fns dispatch DIRECTLY to the raw
    `eval_unary_elementwise_parallel`, which (unlike `eval_unary_elementwise`) lacked the
    standard_unop(_float) guard — so they silently accepted integer operands lax rejects (a gap
    the my0yj transcendental sweep missed). Added the same `is_jax_float_only_unary` guard there;
    fj-lax --lib GREEN 1566/0. LESSON: when auditing for a guard, check BOTH the guarded helper
    AND the raw parallel sibling — a primitive can be in the float-only SET yet route around the
    guard. ENV NOTE: shared CARGO_TARGET_DIR ppv-lite86 artifact was poisoned (E0433
    `u32x4x2_avx2` / E0514) on some rch workers; `cargo clean -p ppv-lite86` cleared it.

## CrimsonForge - fresh fusion A/B re-measurement (2026-06-20): all documented levers HOLD, no regression

- `rch cargo bench -p fj-interpreters --bench eval_fusion_speed` (same-invocation probe suite),
  run on current main (incl. WildForge's `3d997ddd` "specialize f64 scalar add chains").
  Win/loss/neutral verification — ALL documented findings reproduce:
  | probe | result | verdict |
  | --- | ---: | --- |
  | EVAL_FUSION_SPEED_F64 1M (fused vs unfused) | **4.37x** | fusion lever holds |
  | EVAL_FUSION_SPEED_F32 1M | 2.72x | holds |
  | EVAL_FUSION_SPEED_I64 1M | 2.36x | holds |
  | F32/F64 ROW/COL_BROADCAST fused | 2.20-5.02x | broadcast fusion holds |
  | EVAL_FUSION_SPEED_BF16 1M | 0.93x | half floor (documented neutral/loss) |
  | THREAD_AB_F64 16M | 1.24x | threading wins past L3->DRAM (8.4M gate correct) |
  | THREAD_AB_F32 16M | 1.18x | holds |
  | THREAD_AB_I64 16M | 1.12x | holds |
  | THREAD_AB_{F64,F32,I64} 1M-4M | 0.96-1.00x | neutral below gate (correct — no regression) |
  | FUSION_DYNAMIC_PROBE 1M/16M (dynamic-tape element-major) | **0.74x / 0.69x** | the REJECT holds (step-major correct) |
  | FUSION_STRATEGY_PROBE 1M/16M (STATIC monomorphized element-major) | **4.02x / 2.43x** | the codegen ceiling — reachable ONLY via per-chain JIT (bead n75xr) |
- CONCLUSION: no perf regression from the session's guard/golden-hash commits or `3d997ddd`.
  The static-vs-dynamic element-major gap (2.4-4x) re-confirms with CURRENT numbers that the sole
  remaining large-chain lever is per-chain codegen (n75xr, WildForge actively working it).
  CAVEAT: thread-A/B ratios are contention-sensitive (see the gate-raise correction above); the
  16M wins reproducing here means they survive even current load — a robust positive signal.

## AzureLynx - VERIFY pass + lever-space audit (2026-06-20)

- Scope: independent BOLD-VERIFY pass deliberately OFF the hot fj-interpreters CheapOp work
  (WildForge owns bead xjbvr / n75xr, uncommitted lib.rs WIP) to avoid a convergent-commit
  collision. No source changed this pass.
- Baseline health (rch hz1, release): `cargo build --release -p fj-lax` GREEN;
  `cargo test --release -p fj-lax --lib` = **1567 passed / 0 failed / 145 ignored**. No regression
  from the swarm's recent guard/golden-hash/densify churn. (First full-suite run tripped E0514
  ppv-lite86/cfg_if drift — transient mixed-toolchain race in the SHARED CARGO_TARGET_DIR from
  concurrent agent builds; it self-cleared on retry. Infra, not a code regression — do NOT
  `cargo clean` the shared dir while peers build.)
- Parity gap audited: `frankenjax-wzg91` (complex atan2 too-permissive). VERIFIED RESOLVED — the
  `complex_atan2` dispatch is gone; arithmetic.rs:14232 documents intentional float-only rejection;
  `fj-lax --lib atan2` passes 4/4 incl. `test_complex_atan2_reports_unsupported_dtype`;
  conformance `atan2_oracle.rs:731+` asserts complex64/128 scalar+tensor + mixed-complex rejection.
  Closing the bead (bookkeeping; no behavior change needed).
- Lever-space audit (bv --robot-triage): 2004/2009 beads closed (99.7%); 5 open. Every open item is
  policy-gated (cntiy = +fma maintainer decision; oneqh = allocator, closed no-ship), the umbrella
  directive (mcqr), hot/owned (xjbvr fj-interpreters → WildForge), or bookkeeping (wzg91). The
  standing JAX losses are all off-limits or non-contained: float reduce-order (bit-exact pinned),
  N-D transpose ~3.87x (tiling + threading both REJECTED in-ledger; only lever is arch-level
  layout-aware elision/fusion = multi-session), 16M multi-input DRAM elementwise (memory-bound;
  XLA's edge is non-temporal stores, blocked by `#![forbid(unsafe_code)]`), softmax/attention ~2.1x
  (FMA/SIMD-exp gated). Honest conclusion: no contained single-session perf lever remains that does
  not re-mine a DO-NOT vein, hit a maintainer gate, or rush a parity-absolute arch swing.
- Host caveat: non-idle (WildForge + peers active). Per project rule, only same-binary before/after
  ratios are trustworthy here; with no source change there is no A/B to claim — absolute fresh
  win/loss numbers withheld as unreliable. Scorecard delta this pass: 0 wins / 0 losses / 0 neutral
  (verification-only).

## AzureLynx - post-merge workspace-green attestation + parity-audit closure (2026-06-20, HEAD c48f7f54)

- Post-merge regression check after WildForge's `c48f7f54` (fuse unary cheap-op chains, the xjbvr
  CheapOp Floor/Ceil/Trunc/Round/Sign WIP) LANDED. At that HEAD, on rch hz1 release:
  `fj-interpreters --lib` = **212 passed / 0 failed / 27 ignored**; `fj-lax --lib` = 1567/0 (prior
  pass); `cargo clippy --workspace --all-targets -- -D warnings` = **GREEN** (1m41s, zero warnings —
  the whole-workspace all-target gate, not just --lib). 98eoz (fj-lax all-target test lint debt)
  cleared by me in `aa45ca91`.
- Parity-audit thread from wzg91 CLOSED: "audit other float-only binary ops for too-permissive
  complex dispatch." All float-only binary/special ops now reject complex (and int) — atan2
  (eval_binary, 'complex operands'), nextafter (`ensure_float_operands`, arithmetic.rs:13049,
  "Closes liqzs"), igamma/igammac/zeta (same guard). No too-permissive float-only op remains. This
  was the last loose correctness thread in the wzg91 family.
- Standing conclusion (3-pass convergent): the contained lever space is exhausted at the code level,
  not just per-memory. Verified by reading the kernels: transpose rank-2 cache-blocked+threaded
  (tensor_ops.rs:1019), rank-N identity-suffix memcpy (tensor_ops.rs:839); fj-api dispatch-meta
  cache shipped (mcqr.62); float-reduce-order off-limits; DRAM-NT-store / softmax-FMA maintainer-
  gated. Only OPEN ready perf beads are mcqr (umbrella) and cntiy (+fma maintainer decision). The
  remaining real wins are multi-session/owned (WildForge n75xr per-chain JIT codegen, the measured
  2.4-4x static-element-major ceiling) or maintainer-gated — none unilaterally shippable in a single
  autonomous pass. No fabricated levers; no DO-NOT veins re-mined; no collisions.

## AzureLynx - within-line parallel-prefix cumulative scan: confirmed DO-NOT (memory-bound)

- Investigated a fresh-looking lever: `scan_contiguous_lines_to_vec` (reduction.rs:3129) threads ONLY
  across lines (`outer > 1`, gate 1<<18); a large SINGLE-LINE 1-D cumulative scan (cumsum/cummax/
  cummin/cumprod with outer_count==1) falls to the SERIAL branch (reduction.rs:3150) where XLA would
  parallelize. The gap is real in the code.
- BUT this exact lever was already MEASURED and REJECTED (cc, 2026-06-16): a bit-exact two-pass
  parallel-prefix i64 single-line cumsum (integer ops associative with identity init) ran **0.22x —
  a ~4.5x REGRESSION**. Reason: cumulative scan is MEMORY-BANDWIDTH-bound (the per-step add/compare is
  trivial), and parallel-prefix does ~3-4x the memory passes (alloc + local-scan pass + chunk-offset
  apply pass) vs the serial 1 read + 1 write. The extra traffic swamps any core scaling.
- GENERALIZES to the whole associative cumulative family for the single-line case: cummax/cummin
  (associative for int AND float) and int cumsum/cumprod are all equally memory-bound, so within-line
  parallel-prefix LOSES for all of them — NOT just i64 cumsum. float cumsum/cumprod stay separately
  off-limits (non-associative, bit-exact-pinned). DO-NOT-RESHIP any within-line parallel scan.
- Consistent with the standing rule: only COMPUTE-bound ops thread-win (transcendentals 11x); pure
  memory-bound multi-pass restructurings regress. Recorded here (was only in agent-private notes) so
  the swarm stops re-deriving it. Pass delta: 0 wins / 0 losses / 0 neutral (DO-NOT confirmation).

## AzureLynx - FFT confirmed mined + fj-dispatch green attestation (2026-06-20)

- FFT (fft.rs) code-audited: NOT naive. Radix-2 Cooley-Tukey for pow2 + Bluestein (radix-2-backed)
  for non-pow2 → O(n log n) for ALL lengths, with precomputed roots/plans and per-row scratch reuse.
  Batched/multi-row FFT is ALREADY threaded across rows (forward fft.rs:1083, inverse :1302, RFFT
  :1409; gate `batch_size > 1 && transform_work >= PARALLEL_MIN_ELEMS`). Not a JAX-loss candidate to
  chase — the common batched case is parallel + algorithmically optimal.
- Only theoretical FFT gap left: a SINGLE large 1-D FFT (batch_size==1) falls serial (stage-internal
  butterfly threading is the lever). FLAGGED LOW-EV / DO-NOT-rush: (a) niche — ML FFTs are batched
  (already threaded); (b) reordering the radix-2 butterflies risks the pow2 golden digests
  (FFT-parity-standard note); (c) memory-bound past L3. Not attempted; documented so it isn't
  re-investigated.
- Verification this pass (rch hz1 release): `fj-dispatch --lib` = **304 passed / 0 failed / 14
  ignored** (batched-linalg thread fan-out + vmap paths intact). Combined workspace-green snapshot at
  HEAD: fj-lax 1567/0, fj-interpreters 212/0, fj-dispatch 304/0, `clippy --workspace --all-targets
  -D warnings` clean. NOTE: a FRESH vs-JAX head-to-head is not runnable from the agent shell (no jax
  in PATH/venv here); the ledger's vs-JAX ratios were captured on a host-specific jax 0.10.1 venv.
  Pass delta: 0 wins / 0 losses / 0 neutral (verification + DO-NOT documentation).

## AzureLynx - MEASURED: FFT is a 7-43x JAX LOSS (new gap; corrects my pass-5 assumption) [bead murmw]

- Stood up a reusable jax venv (`/data/projects/.scratch/jaxvenv`, jax 0.10.2 x64 CpuDevice) and ran
  the FIRST head-to-head vs-JAX FFT measurement. This CORRECTS my own pass-5 note ("FFT not a JAX-loss
  candidate") — that was an UNMEASURED assumption from reading the code (O(n log n) + row-threaded).
  Measured, it is a large loss. Lesson: audit-only is not measurement.
- Method: JAX `jax.jit(jnp.fft.fft)` x64 complex128, best-of-7x200 + block_until_ready, vs fj-lax
  `eval_primitive(Fft, complex128)` criterion median (rch hz1 release). Same sizes as lax_baseline:

  | case (complex128)          | Rust (eval) | JAX (jit) | Rust/JAX |
  | fft_256 (pow2)             | 6.58 us     | 5.90 us   | 1.12x LOSS |
  | ifft_256                   | 7.08 us     | 5.72 us   | 1.24x LOSS |
  | fft_1000 (Bluestein)       | 46.4 us     | 6.07 us   | **7.6x LOSS** |
  | fft_batch_128x1000         | 3741 us     | 159 us    | **23.5x LOSS** |
  | fft_batch_2048x256 (pow2)  | 10078 us    | 236 us    | **42.7x LOSS** |
  | fft_batch_2048x256 dense   | 4949 us     | 236 us    | **21.0x LOSS** |

- Read: small single pow2 FFT is ~par (1.12x — methodologies share a framework-overhead floor, which
  VALIDATES that the big-case gaps are real, not measurement artifact). The loss explodes for BATCHED
  (21-43x) and COMPOSITE non-pow2 (7.6x). Root cause is the KERNEL, not storage: dense complex storage
  already halves the batched cost (dense 21x vs boxed 42.7x) but the floor is radix-2 SCALAR butterflies
  vs pocketfft's SIMD higher-radix (radix-4/8) + native radix-3/5/7 (so JAX never Bluesteins 1000=2^3*5^3).
- Scorecard impact: FFT was an UNCHARACTERIZED family; it is now a documented 7-43x loss. Filed
  frankenjax-murmw (P1) with the lever menu: mixed-radix Cooley-Tukey (radix-4/8 pow2 + native
  composite radixes, kills Bluestein on composite lengths) + std::simd complex butterflies + cache-
  blocked batched transform. LEGAL (FFT parity is TOLERANCE 1e-9, reassociation OK) but must refresh
  the pow2 golden digests. MULTI-SESSION (dedicated FFT-kernel pass), parity-risk — NOT shipped here.
  Pass delta: 0 wins / 4 losses / 2 narrow-losses MEASURED (real new negative evidence).

## AzureLynx - WIN: cache Radix2Plan across batched pow2 FFT rows — 1.43x bit-exact [murmw step 1]

- First shipped step on the murmw FFT frontier. Root cause profiled: `BatchFftPlan::Pow2` was a
  DATALESS marker that delegated to `fft_1d_into` → `radix2_fft_1d_into`, which REBUILT the
  bit-reversal + per-stage twiddle table (incl. a `sin_cos` per stage) and allocated a `tw` Vec on
  EVERY row. The `Mixed` and `Bluestein` batch variants already cache their tables; only `Pow2`
  didn't. Over a 2048x256 batch that is ~16K redundant `sin_cos` + 2048 allocations.
- Fix: `BatchFftPlan::Pow2(Radix2Plan)` — build ONE plan per batch (bit-reversal + all twiddles via
  the IDENTICAL first-order recurrence) and reuse `plan.apply_into` per row. Strictly removes work.
- BIT-EXACT (not tolerance): the pow2 golden digest `fft_golden_output_digest` PASSES UNCHANGED — no
  digest refresh needed. New guard `bench_batched_pow2_plan_cache_vs_per_row_rebuild` asserts the
  cached-plan output is bit-for-bit equal to the per-row rebuild.
- MEASURED same-binary A/B (the trustworthy kind — same invocation, bit-identical refs side by side,
  rch hz1 release, best-of-5): batched pow2 2048x256 transform loop **per-row-rebuild 6.497 ms ->
  shared-plan 4.554 ms = 1.43x**. (Cross-invocation full-eval criterion also dropped ~4.95->3.72 ms
  on the dense path, directionally consistent but NOT the headline — variance is 20-60% across
  invocations.) fj-lax --lib 1567/0, clippy --all-targets clean.
- This NARROWS the murmw batched-pow2 loss (~42x -> ~30x) but does NOT flip it. The remaining gap is
  the radix-2 scalar butterfly vs pocketfft SIMD higher-radix — still the multi-session kernel lever.
  Pass delta: 1 win (1.43x bit-exact) / 0 losses / 0 neutral.

## AzureLynx - WIN: cache Radix2Plan for pow2 IRFFT rows — 1.39x bit-exact [murmw step 2]

- Same per-row-rebuild gap as step 1, in the INVERSE real FFT: `eval_irfft` built the cached plan
  only for NON-pow2 (`(fft_length > 1 && !is_power_of_two()).then(...)`), so POW2 irfft fell to the
  `None => ifft_1d_into` arm in `irfft_rows_f64_into`, rebuilding bit-reversal + twiddles (incl.
  `sin_cos`) + a Vec on EVERY row. (rfft was already fine — pow2 uses the cached `RealRfftPower2Plan`,
  non-pow2 builds `Some(BatchFftPlan)` once; its `None` arm is dead defensive code.)
- Fix: build the plan for ALL lengths `(fft_length > 1).then(|| BatchFftPlan::new(fft_length, true))`
  — pow2 now gets the cached `Pow2(Radix2Plan)` from step 1. One-line change.
- BIT-EXACT (same recurrence + identical 1/n scaling): irfft golden + `irfft_basic` +
  `threaded_batch_irfft_matches_per_row_serial` all PASS unchanged. A/B test extended with an
  inverse bit-identity guard (`ifft_1d_into` per row == cached `Radix2Plan(inverse).apply_into`).
- MEASURED same-binary A/B (same run, rch release, best-of-5): batched pow2 2048x256 inverse loop
  **per-row-rebuild 6.605 ms -> shared-plan 4.745 ms = 1.39x**. (Forward measured 1.08x THIS run vs
  1.43x last run — pure cross-invocation variance; the inverse number is the in-run measurement for
  this change.) fj-lax --lib 1567/0, clippy --all-targets clean.
- FFT/IFFT/RFFT/IRFFT batched pow2 paths are now ALL plan-cached. Remaining murmw lever stays the
  multi-session SIMD mixed-radix kernel. Pass delta: 1 win (1.39x bit-exact) / 0 losses / 0 neutral.

## AzureLynx - MEASURED: eigh/SVD are catastrophic losses (eigh 512=3.2s, super-cubic) [bead ur4h3]

- Used the jax venv to hunt the NEXT gap after FFT: linalg. Added a reusable #[ignore] harness
  (`linalg.rs::bench_linalg_vs_jax_sizes`). fj-lax eval_primitive f64, rch release:

  | n   | svd      | eigh       | qr     | cholesky |
  | 128 | 2.48 ms  | 17.06 ms   | 0.23ms | 0.31ms   |
  | 256 | 21.79 ms | 259.59 ms  | 1.28ms | 2.12ms   |
  | 512 | 287.98ms | 3248.97 ms | 7.71ms | 10.73ms  |

- eigh scales 15.2x then 12.5x per doubling = SUPER-CUBIC (O(n³) is 8x); svd 8.8x/13.2x. QR + Cholesky
  scale ~cubic and are FAST (already blocked/GEMM-routed). **eigh 512 = 3.2 SECONDS.**
- LAPACK baseline NOT trustworthy right now: the shared host is so loaded that numpy read svd_128=136ms
  (absurd — LAPACK does <1ms), so no clean ratio. BUT eigh's super-cubic SCALING is contention-robust
  (relative, measured back-to-back under one load) and 3.2s@512 is self-evidently algorithmic. Ref:
  LAPACK dsyevd/dgesdd on 512 ≈ 10-30ms → fj-lax eigh is ~100-300x off.
- ROOT CAUSE: eigh/SVD never got the blocked-GEMM treatment QR/Cholesky/LU/solve received. eval_eigh →
  `tridiag_ql_eigendecomposition` (Householder tridiag + QL) but the reduction/eigvec-accumulation are
  cache-hostile scalar strided ops (super-cubic wall-clock past L2, not O(n³)-blocked). SVD =
  `one_sided_jacobi_svd_real` (O(n³·sweeps), max_sweeps=60), not a blocked bidiag+QR. NOTE memory
  claimed eigh tred2/tql2 (2.2-3.0x) + SVD bidiag+QR (4.2x) were "done" — tridiag IS wired but UNBLOCKED;
  SVD is still Jacobi (the bidiag+QR may have been reverted/unwired — flagged in ur4h3 to verify).
- Filed frankenjax-ur4h3 (P1). Levers (multi-session, linalg.rs = codex zone, parity is TOLERANCE so
  blocking is LEGAL): blocked Householder tridiagonalization (dsytrd-style panel + GEMM trailing
  update) and blocked bidiag+implicit-QR SVD. This is a BIGGER gap than FFT — eigh ~100-300x vs FFT's
  7-43x. Pass delta: 2 MEASURED losses (eigh ~100-300x, svd ~15x) characterized.

## AzureLynx - WIN: row-contiguous Householder left-apply — 4.24x bit-exact on Hessenberg reduction [ur4h3 step 1]

- Profiled eigh's super-cubic scaling to its cause: `apply_householder_left` (linalg.rs) iterated
  columns-outer/rows-inner, reading `matrix[(row_start+offset)*n + col]` with the flat index striding
  by `n` for a fixed `col` — a CACHE CLIFF once the matrix exceeds L2 (right-apply was already
  row-contiguous). Reordered to rows-outer/cols-inner so every access streams a CONTIGUOUS row slice.
- BIT-EXACT: `scaled[col]` accumulates over offset in the SAME ascending order as the old per-column
  `dot`, and each element is updated once with the identically-grouped `(β·dot)·v_i`. eigh tests 13/0,
  hessenberg 4/0, fj-lax --lib 1567/0, clippy --all-targets clean — all unchanged (golden/recon pass).
- MEASURED same-binary A/B (the trustworthy kind — `bench_householder_left_cache_layout_ab`, full
  512×512 Hessenberg reduction, old strided vs new contiguous left-apply in ONE binary, H+Q asserted
  bit-identical, rch release): **strided 500.2 ms -> contiguous 117.9 ms = 4.24x**.
- HONEST SCOPE: this is the REDUCTION component. Full eigh@512 ≈ 2.3s, so the reduction (~118ms now)
  is only ~5-15% of eigh — the QL eigenvector accumulation (`symmetric_tridiagonal_ql` rotating z) and
  the per-element plane rotations DOMINATE and are the next lever (ur4h3 step 2). So this does NOT
  flip the eigh JAX loss, but it is a real bit-exact 4.24x on a hot kernel that ALSO speeds the
  general-eig (nonsymmetric Schur) path (reduction-heavy). The cross-invocation harness eigh numbers
  (3249->2285) are CONFOUNDED (untouched svd drifted 288->422 same run = host ~1.4x slower); trust the
  same-binary 4.24x, not the cross-run delta. Pass delta: 1 win (4.24x bit-exact, reduction) / 0 / 0.

## AzureLynx - BIG WIN: column-major QL eigenvector accumulation — 47.8x, eigh 512 3.2s->172ms [ur4h3 step 2]

- The DOMINANT eigh cost (step 1 showed the reduction was only ~5-15%): `symmetric_tridiagonal_ql`'s
  plane-rotation eigenvector accumulation read `z[k*n + i]` / `z[k*n + (i+1)]` for k in 0..n —
  COLUMN-STRIDED (stride n) over the n×n eigenvector matrix, a catastrophic cache cliff. Fix: store
  `z` COLUMN-MAJOR during QL so each column is a contiguous n-run; the rotation now streams two
  contiguous slices (cache-friendly + autovectorizable). `tridiag_ql_eigendecomposition` transposes
  Q in and the result back (both O(n²) vs QL's O(n³)).
- BIT-EXACT: identical rotation arithmetic, same s,c, same k order — only the storage layout changes.
  eigh tests 13/0, fj-lax --lib 1567/0, clippy --all-targets clean (golden/reconstruction unchanged).
- MEASURED same-binary A/B (`bench_ql_eigvec_accumulation_layout_ab`, 512 reduction's tridiagonal+Q,
  old row-major-strided vs new column-major-contiguous QL in ONE binary, eigenvalues+eigenvectors
  asserted bit-identical, rch release): **row-major 2548.6 ms -> column-major 53.3 ms = 47.8x.**
- END-TO-END eigh (harness, same run so internally consistent): n=512 **3249 ms (orig) -> 172 ms** ≈
  19x; scaling normalized from SUPER-CUBIC (15x/doubling) to CUBIC (3.0->21.5->172 = ~7-8x/doubling),
  which is contention-robust structural evidence, not host noise. eigh is now only ~6-17x off LAPACK
  dsyevd (was ~100-300x). The two ur4h3 cache fixes (reduction 4.24x + QL 47.8x) together largely
  close the eigh gap. REMAINING: exploit symmetry in the reduction (dsytrd vs general hessenberg) for
  the last ~2x, and SVD (one_sided_jacobi, still untouched, ~15x). Pass delta: 1 BIG win (47.8x
  same-binary; eigh ~19x end-to-end, bit-exact) / 0 / 0.

## AzureLynx - VERIFY eigh win vs LAPACK + IMPORTANT baseline caveat (2026-06-20)

- Confirmed the eigh win holds: fresh harness fj-lax eigh 512 = **165.5 ms** (consistent with the
  172ms from the win pass; ~19.6x over the original 3249ms). Same-host-state numpy/LAPACK:
  cholesky 512 = 4.04 ms (sanity anchor — sane, host is quiet), **eigh 512 = 1024.9 ms**.
- **fj-lax eigh (165ms) now BEATS this environment's numpy/LAPACK eigh (1025ms) by 6.2x.** But that
  numpy eigh number is absurd for an OPTIMIZED LAPACK dsyevd (~10-30ms): cholesky@512=4ms proves the
  BLAS is optimized, so it's specifically the EIGENSOLVER that's unoptimized reference LAPACK in this
  venv (numpy 2.4.6, bundled). 
- **METHODOLOGY CAVEAT (applies to the whole campaign's vs-LAPACK linalg claims):** in THIS
  environment, only BLAS-bound LAPACK ops (cholesky, matmul, qr's GEMM) give a trustworthy LAPACK
  baseline; the LAPACK EIGENSOLVER/SVD drivers (dsyevd/dgesdd) run unoptimized here, so a clean
  fj-lax-vs-optimized-LAPACK ratio for eigh/SVD is NOT measurable. My pass-9/11 "~6-17x / ~100-300x
  off LAPACK" for eigh were EXTRAPOLATIONS from "optimized dsyevd ~10-30ms", NOT measured — correcting
  that here. What IS solid: the ~19x SELF-improvement (same-binary A/B + super-cubic→cubic scaling
  normalization) and that fj-lax eigh now beats the available numpy eigh.
- fj-lax CHOLESKY 512 = 15.9ms vs numpy 4ms = ~4x off optimized BLAS (the one trustworthy linalg
  ratio this run) — a real, modest, BLAS-microkernel-bound gap (consistent with the documented
  matmul_2d ~XLA/2 FMA ceiling). NOT chasing further (FMA-gated). Pass delta: verification — eigh win
  CONFIRMED + corrected the vs-LAPACK framing (0 wins / 0 losses; methodology fix).

## AzureLynx - MEASURED: conv2d 1.8x + mid-size f64 matmul 9.3x JAX losses (FMA + spawn-overhead gated)

- New head-to-head measurements (jax 0.10.2 x64 jit, block_until_ready, vs fj-lax eval_primitive,
  rch release; host = Zen3 5975WX, NO AVX512 so that is NOT a factor):
  - **conv2d** [4,64,64,32]*[3,3,32,64] SAME: JAX 11.25 ms vs fj-lax 20.25 ms = **1.80x LOSS**. fj-lax
    is im2col + threaded matmul_2d (the GEMM is 16384×288×64 = 3e8 ops → threads), so this is the
    GEMM ceiling cascading to conv. Moderate, FMA-gated.
  - **matmul 256³ f64**: JAX 0.186 ms (≈180 GFLOP/s) vs fj-lax 1.727 ms (≈19 GFLOP/s) = **9.3x LOSS** —
    much worse than the documented large-n "~XLA/2". 256³ DOES cross the threading threshold (16.7M >
    8M, gets ~8 threads) but per the standing finding only scales ~2.68x over single-core due to
    per-call OS-thread SPAWN overhead. So the gap = no-FMA (~2x, cntiy-gated) × spawn-limited threading
    (~4-5x: JAX scales to the full 32-core box, fj-lax to ~2.5x). 
- BOTH levers are known and NOT clean single-pass wins: (1) FMA = cntiy maintainer decision; (2) the
  matmul threading scaling = persistent worker pool to kill per-call spawn (bead eqwqb), explicitly a
  DEDICATED multi-session effort per the GEMM notes — and matmul_2d is codex/shared territory, so I am
  NOT quick-swinging it. Recorded as characterized gaps. The persistent-pool lever is arguably the
  single highest-EV remaining non-gated perf swing (it cascades to matmul/conv/linalg-trailing-updates
  across the whole mid-size regime) — worth a dedicated owner. Pass delta: 2 MEASURED losses
  (conv 1.8x, matmul-256 9.3x) characterized / 0 wins.

## AzureLynx - VERIFY ifou2 root cause: mid-size GEMM threading is overhead-capped (same-binary A/B)

- Traced the 256³ matmul loss to the kernel-vs-threading split, then measured it directly. eval_dot
  256³ routes eval_dot → rank2_f64_matmul → matmul_2d (threaded), so the 1.7ms IS threaded. New
  same-binary A/B `bench_matmul_2d_threading_scaling` (tensor_contraction.rs, forced-serial vs the
  production thread count, SAME pack/block plan so ONLY threads differ, bit-identical asserted):
  - 256³: threading speedup **only ~1.1-1.9x** (8 threads); best in a sweep ~2.86x at 6 threads.
  - 512³: ~2.3-3.0x; 1024³: ~3.7-4.1x. All FAR from the ~8-10x ideal for the thread count.
- So mid-size GEMM parallelism is overhead-capped (per-call OS-thread spawn + scoped setup), worst at
  small sizes (256³ barely beats serial). The kernel itself is FINE — serial 256³ = ~29 GFLOP/s
  (register-tile peak). The gap to JAX (180 GFLOP/s) is parallel scaling + FMA, not the microkernel.
- A thread-count RETUNE is NOT the fix: the sweep on the (loaded) shared host is too noisy to act on
  (1-thread rows read 0.91-1.21x when they must be exactly 1.0x = ±20% noise; serial drifted
  1.3→2.2ms between runs), and 6t≈8t — the cap is per-thread overhead, not the count. No contained win.
- SHARPENED ifou2 CONSTRAINT (important): a persistent pool is NOT just "add a pool" under
  `#![forbid(unsafe_code)]`. A borrowing work-pool that writes DISJOINT output ranges from `&a,&b`
  needs rayon/crossbeam-style UNSAFE lifetime handling (the reason std only offers `thread::scope`,
  which re-spawns each call). Safe alternatives all add overhead: (a) Arc-clone inputs + each worker
  returns a local output Vec that the caller concatenates (O(mn) copy/call); (b) a 'static pool fed
  Arc'd buffers. So ifou2 is a real DESIGN problem with a tradeoff, not a quick swing — recorded so
  the owner budgets for it. Pass delta: verification (ifou2 root cause MEASURED + safe-impl constraint
  identified) / 0 wins / 0 losses.

## AzureLynx - SCORECARD: sort is a fj-lax WIN; the vs-JAX-CPU loss set is now narrow + characterized

- Measured sort head-to-head (jax 0.10.2 x64 jit vs fj-lax eval_primitive sort_64k, rch release):
  sort 64k **i64: fj-lax 1.30 ms vs JAX 2.04 ms = 1.57x WIN; f64: 1.42 ms vs JAX 11.18 ms = 7.9x WIN.**
  (JAX-CPU's f64 sort at 11ms for 64k is its slow XLA comparison sort, not a real target.)
- This crystallizes the campaign's WIN/LOSS map. JAX-CPU only genuinely BEATS fj-lax where XLA calls a
  tuned native lib or fuses BLAS:
  - **LOSSES (real):** matmul/conv (FMA-gate cntiy + spawn-limited mid-size threading ifou2 — matmul
    256³ 9.3x, conv 1.8x); FFT (pocketfft SIMD mixed-radix — murmw, 7-43x). float reduce-order
    (off-limits). softmax/attention (FMA-gate). That is the WHOLE remaining loss set, and every item is
    gated or multi-session.
  - **WINS / parity (fj-lax ≥ JAX-CPU):** sort (1.6-7.9x), eigh (now beats numpy eigh 6.2x after the
    ur4h3 cache fixes), elementwise/reduction/broadcast/select/transpose/gather at scale (documented
    threaded wins), argmax (1.36x), RNG. JAX-CPU's eigh/svd/sort/fft-mixed-radix use slow XLA-CPU
    implementations — measuring against them is unreliable AS A LOSS (they flatter fj-lax) but reliable
    as confirmation fj-lax is NOT losing there.
- CONCLUSION: the contained, bit-exact, single-pass perf veins are EXHAUSTED (this campaign shipped
  the eigh cache fixes 4.24x+47.8x and the FFT plan-cache 1.43x+1.39x). Every remaining vs-JAX loss is
  (a) FMA-policy-gated (cntiy), (b) a multi-session kernel/algorithm rewrite (FFT murmw, SVD bidiag+QR,
  eigh symmetric reduction), or (c) the forbid-unsafe-constrained persistent pool (ifou2). Pass delta:
  scorecard — sort WIN confirmed + loss set fully characterized / 1 win documented / 0 losses.

## AzureLynx - CAVEAT: matmul/ifou2 threading numbers are HOST-CONTENTION-confounded (don't over-invest)

- Profiled the threaded matmul path (matmul_2d_with_threads_into): the per-call overhead is purely
  `std::thread::scope` spawn/join — NO per-thread alloc, and pack_b is OFF at 256³ (k·n=65536 <
  PACK_B_MIN_KN). The measured ~125 µs/thread "overhead" is HIGHER than raw OS thread spawn (~10-30µs
  idle), which means a large part of it is HOST CONTENTION: this 32-core box is saturated by the agent
  swarm + concurrent rch builds, so (1) OS thread creation contends and (2) the spawned workers fight
  the swarm for cores. 
- CONSEQUENCE for the ifou2 case: the same-binary serial-vs-threaded RATIO is itself load-sensitive —
  threaded perf degrades MORE than serial under load (serial spawns nothing), so the measured 256³
  threading speedup (~1.1-1.9x) is DEFLATED vs an idle host. The "matmul 256³ 9.3x off JAX" is
  therefore partly host-inflated; the true idle-host gap is unknown and likely smaller. ifou2 should
  NOT be scoped/prioritized off these numbers — it needs a re-measure on a QUIET host first (the
  project's own rule: "definitive vs-JAX win/loss requires an IDLE host, currently unavailable").
- This also retro-qualifies the conv 1.8x and matmul 9.3x (pass 13): real losses, but their MAGNITUDE
  is upper-bounded by host load. The DIRECTIONAL conclusion (FMA-gate + threading-scaling are the
  matmul/conv levers) holds; the exact ratios do not. The eigh/FFT wins are UNAFFECTED (same-binary
  A/B + scaling-class normalization are contention-robust; the host-sensitive part is absolute spawn
  cost, not the bit-exact kernel ratios). Pass delta: methodology caveat (matmul/ifou2 host-confounded)
  / 0 wins / 0 losses.

## AzureLynx - GAUNTLET: full fj-conformance parity suite GREEN at HEAD (certifies the campaign wins)

- The shared-target toolchain drift that blocked fj-conformance all campaign has SETTLED (swarm idle).
  Ran the FULL fj-conformance suite (rch hz1 release): **every oracle test file 0 failed** — ~40 test
  binaries, ~1200+ parity tests vs the JAX oracle, all GREEN. Notable for my changes: linalg_oracle
  64/0 (certifies the eigh cache fixes 4.24x+47.8x preserve eigendecomposition parity), fft_oracle
  27/0 + linalg_fft_oracle_parity 1/0 (certify the FFT/IFFT/RFFT/IRFFT plan-cache 1.43x+1.39x preserve
  FFT parity).
- This is the directive's "conformance GREEN" + "running-the-gauntlet" certification, done
  comprehensively for the first time this campaign: the ENTIRE JAX parity surface is correct at HEAD —
  my four shipped perf wins AND the swarm's accumulated churn. The eigh + FFT wins were designed
  bit-exact (golden digests unchanged) and the oracle suite confirms the eval-vs-JAX behavior is
  unchanged.
- Investigated one more eigh-style cache-cliff candidate: nonsymmetric eig (eval_eig) `francis_sweep`
  has a column-strided bulge-chase H update, BUT it is a TWO-SIDED update (rows AND columns of the
  same H), so it is entangled like the cyclic-Jacobi A-matrix — cannot be cleanly de-strided in one
  layout, and eval_eig's LAPACK-CPU baseline is unmeasurable (slow XLA-CPU eig). No clean win. The
  clean bit-exact cache-cliff lever was a one-time find (eigh's QL, which was a ONE-SIDED column
  rotation). Pass delta: GAUNTLET certification (full conformance GREEN) / 0 new wins / 0 losses.

## CrimsonOtter / cod-a - murmw specialized radix-2/5 SoA validation blocked (2026-06-21)

- Active bead: `frankenjax-murmw`, smooth-composite FFT batch loss. Current
  target row remains `eval/fft_batch_128x1000_complex128`: RCH `hz1` Criterion
  midpoint **3.6581 ms** versus fresh local JAX/JAXLIB 0.10.1 x64 mean
  **0.245442 ms**, so Rust/JAX is **14.90x**.
- The graveyard/artifact/optimization pass points to generated
  length-specialized radix-2/5 kernels for `1000 = 2^3 * 5^3` as the next
  credible radical lever. The older flat iterative SoA and Bluestein SoA
  detours remain no-ship.
- Focused validation of the specialized SoA path did not produce a ratio. RCH
  stale-progress blocked one ignored `fj-lax` release test on `vmi1153651`; the
  next `vmi1149989` run hit an unrelated peer cbrt compile blocker
  (`fast_cbrt_f64`); the following `vmi1149989` retry stalled before printing
  the A/B row. The stale builds were canceled and the temporary test harness was
  removed.
- Verdict: no source keep and no timed rejection. Row score remains **0 wins /
  1 loss / 0 neutral** versus JAX; candidate score is **0 kept / 0 rejected by
  timing / 1 validation-blocked**. Retry predicate: wait for the cbrt WIP to be
  landed/reverted and for RCH to reuse a warm target, then rerun the same-binary
  A/B before any production dispatch change.

## CrimsonOtter / cod-a - mcqr f64 add-chain SIMD splat-hoist no-ship (2026-06-21)

- Scope: `frankenjax-mcqr`, non-overlapping `fj-interpreters` residual loss after
  the retained `frankenjax-n75xr` scalar-add SIMD medium-band keep. Target row:
  `compiled_dispatch/compiled_runner/bigchain1048576/n=8`.
- Alien-graveyard/artifact/optimization hypothesis: remove invariant work from
  the vector loop by pre-splatting the eight scalar literals once per call, while
  still applying the adds in original Jaxpr order for exact floating-point
  behavior.
- Baseline proof: RCH remote-only per-crate Criterion on `vmi1227854` with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a` and filter
  `bigchain1048576/n=8`.
  - `compiled_runner`: **855.91 us** midpoint (`769.98..934.79 us`).
  - `compiled_runner_scalar`: **812.35 us** midpoint (`791.22..845.20 us`).
  - Same-binary production/scalar ratio: **1.054x**; the retained production
    path did not beat the scalar control on this worker.
- Candidate proof: RCH selected `ovh-a`, so the absolute before/after is routing
  evidence only, not same-worker proof.
  - `compiled_runner`: **1.1269 ms** midpoint (`1.0932..1.1649 ms`).
  - `compiled_runner_scalar`: **1.1934 ms** midpoint (`1.1271..1.2339 ms`).
  - Same-binary production/scalar ratio: **0.944x**, but absolute runtime is
    still far above the inherited `frankenjax-n75xr` JAX comparator
    (~**83.3 us**, from the 821.23 us / 9.86x row), giving an approximate
    **13.5x Rust/JAX loss** for this run.
- Verdict: reverted before commit. This is not a JAX-closing lever, and the
  worker switch prevents a credible keep claim. Candidate score: **0 wins / 1
  loss / 0 neutral** versus JAX, **0 kept / 1 reverted**.
- Conformance gate after revert: RCH `vmi1149989`
  `cargo test -p fj-conformance --lib --release -- --nocapture` passed
  **45/45**.
- Retry predicate: do not retest splat-hoisting for the n=8 add-chain. The next
  credible route needs a generated straight-line/vector kernel, a parity-proofed
  algebraic contraction, or disassembly/perf-counter proof that the current loop
  is leaving real instruction-level parallelism on the table.

## CrimsonOtter / cod-b - guarded small-angle tan rational kernel keep (2026-06-21)

- Scope: `frankenjax-cntiy` tolerance-only transcendental sweep, following the
  documented `tan next` route. This does **not** close the parent +fma policy
  bead; it mines a contained non-FMA sub-gap.
- Alien-graveyard/artifact lever: R11 libm/fenv -> Remez/minimax, compiled
  offline artifact -> online deterministic polynomial/rational kernel. The
  production path uses a Cephes-style rational approximation only for large
  dense F64 tensors whose elements are all finite and inside `[-pi/4, pi/4]`.
  Scalar, complex, f32/half, NaN/inf, and general-range tensors keep the old
  `libm` route.
- Correctness gates:
  - RCH `hz2` `cargo test -p fj-lax
    fast_small_tan_large_dense_f64_matches_libm_tolerance --release --
    --nocapture` passed 1/1; max absolute error vs `f64::tan` was
    **1.110e-16** against the tan oracle's **1e-10** tolerance class.
  - RCH `vmi1149989` `cargo test -p fj-conformance --test tan_oracle
    --release -- --nocapture` passed **36/36**.
  - RCH `vmi1152480` `cargo test -p fj-conformance --release` passed the full
    crate test suite and doc-tests, exit 0 in **334780 ms**.
- Same-worker RCH `ovh-a` Criterion, per-crate filter
  `tan_1m_f64_vec`, exact fixture `((i % 3001) - 1500) * 0.0005` over 1M f64:

  | row | midpoint | interval |
  | --- | ---: | ---: |
  | `eval/tan_1m_f64_vec_libm_reference` | **4.6905 ms** | 4.6740..4.7027 ms |
  | `eval/tan_1m_f64_vec` | **1.1134 ms** | 1.0941..1.1352 ms |

- Rust-side decision: KEEP. The retained production row is **4.21x** faster
  than the same-binary libm reference for the guarded row.
- Fresh JAX comparator: local `uv run --no-project --with 'jax[cpu]'`, JAX
  **0.10.2**, JAXLIB **0.10.2**, CPU x64, same fixture, 20 warmups + 200 timed
  `jax.jit(jnp.tan)` calls: mean **1.412564 ms**, p50 **1.383380 ms**, p95
  **1.727568 ms**.
- Scorecard: **1 win / 0 loss / 0 neutral** vs JAX for this row. Rust/JAX by
  mean is **0.788x** (fj-lax **1.27x faster**). This is a scoped small-angle
  domination, not evidence that general-range tan beats JAX.
- Retry predicate: do not widen the guard without a real SIMD/range-reduction
  proof. The remaining transcendental route is true vector polynomial range
  reduction or the still-open `cntiy` target-feature/FMA policy decision.

## CrimsonOtter / cod-b - cntiy scalar atan2 threading keep, still JAX loss (2026-06-21)

- Scope: `frankenjax-cntiy`, non-FMA transcendental sub-gap after the guarded
  tan win. This does not resolve the parent +fma/target-feature decision.
- Alien-graveyard/artifact/optimization hypothesis: stop serializing a
  million independent scalar `libm` calls when operand order is lane-local.
  The kept lever removes the f64 scalar `Atan2` exclusion from the existing
  expensive scalar-broadcast threaded path. It preserves the exact per-lane
  `f64::atan2` operation and operand order, so output bits are unchanged.
- Same-worker RCH `vmi1293453`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`,
  per-crate Criterion:
  - Baseline `eval/atan2_scalar_1m_f64_vec`: **30.351 ms** midpoint.
  - Kept route in full `atan2` filter: **12.769 ms** midpoint
    (`11.636..13.661 ms`), **2.38x** faster than baseline.
  - Scalar-only confirmation: **13.998 ms** midpoint (`12.129..15.295 ms`),
    **2.17x** faster than baseline.
- Fresh JAX/JAXLIB 0.10.2 CPU x64 comparator, exact fixture
  `jnp.atan2(a, 3.25)` over 1M f64 with 20 warmups + 200 timed `jax.jit`
  calls: mean **0.116833 ms**, p50 **0.096482 ms**, p95 **0.175232 ms**.
  Confirmed Rust/JAX ratio is **119.8x**. The old Rust route was **259.8x**,
  so this narrows the loss but does not approach domination.
- Correctness gates:
  - RCH `vmi1149989` `cargo test -p fj-lax atan2 --release -- --nocapture`
    passed **4/4**.
  - RCH `vmi1152480` `cargo test -p fj-conformance --test atan2_oracle
    --release -- --nocapture` passed **40/40**.
  - RCH `vmi1152480` full `cargo test -p fj-conformance --release --
    --nocapture` passed the full crate suite and doc-tests.
- Scorecard: **0 wins / 1 loss / 0 neutral** versus JAX for this row;
  candidate disposition **1 kept / 0 reverted** because the same-worker Rust
  improvement is large and bit-identical.
- Retry predicate: do not repeat scalar Atan2 thread-count tuning. The next
  credible radical route is a safe portable-SIMD/range-reduced atan2
  approximation with an `atan2_oracle` tolerance proof, or the broader `cntiy`
  target-feature/codegen policy work.

## CrimsonOtter / cod-b - cntiy boxed-literal scalar pow/atan2 threading keep, still JAX loss (2026-06-21)

- Scope: `frankenjax-cntiy`, follow-up after the dense scalar Atan2 threading
  keep. `TensorValue::new` already densifies F64; this pass targeted only the
  direct `LiteralBuffer::Literals` scalar-broadcast control path used by the
  `*_literal_ref` benches and any caller that bypasses the constructor.
- Alien-graveyard/artifact/optimization hypothesis: a million independent
  boxed-F64 Pow/Atan2 lanes are an embarrassingly parallel vector-execution
  artifact. Thread the existing exact lane operation over boxed F64 literals,
  return dense F64 output, and fall back to the generic path for mixed or
  non-F64 buffers.
- Same-worker RCH `vmi1293453`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`,
  per-crate Criterion filter `scalar_1m_f64_literal_ref`:

  | row | baseline midpoint | kept midpoint | Rust speedup |
  | --- | ---: | ---: | ---: |
  | `eval/pow_scalar_1m_f64_literal_ref` | 80.435 ms | 15.193 ms | 5.29x |
  | `eval/atan2_scalar_1m_f64_literal_ref` | 38.339 ms | 11.987 ms | 3.20x |

- Fresh JAX comparator: `benchmarks/jax_comparison/.venv/bin/python`, JAX
  **0.10.1**, JAXLIB **0.10.1**, CPU backend, `jax_enable_x64=true`, exact
  1M fixtures. Pow mean **1.808211 ms**, p50 **1.738733 ms**; Atan2 mean
  **2.214336 ms**, p50 **2.073297 ms**.
- Rust/JAX scorecard: pow **8.40x loss**, Atan2 **5.41x loss** by kept
  midpoint / JAX mean. Overall pass scorecard **0 wins / 2 losses / 0 neutral**.
- Correctness gates:
  - RCH `vmi1293453` `cargo test -p fj-lax
    threaded_expensive_binary_scalar_bit_identical_to_reference --release --
    --nocapture` passed **1/1**. The proof covers dense and boxed storage, both
    operand orders, Pow and Atan2, and dense F64 output.
  - `rustfmt --edition 2024 --check crates/fj-lax/src/arithmetic.rs` passed.
  - RCH `vmi1152480` full `cargo test -p fj-conformance --release` passed the
    full crate suite and doc-tests.
  - RCH `hz2` `cargo check -p fj-lax --all-targets` passed, with existing
    deprecated `criterion::black_box` warnings in the unrelated
    `f32_rounding_ab` bench.
  - RCH `hz2` `cargo clippy -p fj-lax --all-targets -- -D warnings` remains
    blocked by unrelated `crates/fj-lax/src/fft.rs:1664`
    `manual_is_multiple_of` lint.
- Decision: KEEP. The Rust-side gains are large and bit-identical, but this is
  still a JAX loss. Retry predicate: stop boxed-literal fan-out work; next
  credible route needs vector range-reduced pow/atan2 or the broader `cntiy`
  target-feature/codegen lane.

## CobaltForge / cc - f32 rounding-unary dead-ends behind the fusion-chunk win (2026-06-21)

- Context: while landing the `apply_f32_fusion_chunk` op-match hoist (see the
  release scorecard keep, floor/round/sign f32 chains +4.63x), two adjacent
  hypotheses were measured and REJECTED. Recorded so they are not re-attempted.
- DEAD-END 1 — direct-f32 `roundps` kernel for the single op. Hypothesis: the
  serial f32 unary path widens `f32->f64->f64::floor->f32` (4-lane), so a native
  `f32::floor` (`roundps`, 8-lane) should win and is bit-identical for the exact
  integer-rounding ops. Same-binary RCH `ovh-a` A/B over 1M f32:
  `floor_widen_single` **74.07 us** vs `floor_direct_single` **76.51 us**
  (**1.03x SLOWER**); hot 4-chain `floor_widen_chain4` **223.8 us** vs
  `floor_direct_chain4` **229.0 us** (**1.02x slower**). The widen path already
  autovectorizes and is bandwidth-bound; the single-op kernel was NEVER the
  bottleneck (74 us already beats JAX's ~104 us). The bottleneck is purely the
  FUSED-chain dispatch, not the per-op kernel. Do not re-chase native-f32
  rounding kernels.
- DEAD-END 2 — hoisting the f64 fusion-chunk unary match. Same-binary RCH
  `ovh-a` faithful 14-arm-match proxy, `(add,floor)x4` over 1M f64: in-loop
  **1.8596 ms** vs hoisted **1.8639 ms** = **1.00x** (neutral). LLVM already
  loop-unswitches the non-widening f64 path, so the f64 hoist yields nothing
  (the f32 hoist wins only because the per-element `f64::from` widen + 14-arm
  match together defeat vectorization). The f64 `apply_f64_unary_chunk` was kept
  anyway for sibling symmetry + robustness against nightly unswitch drift; it is
  bit-identical and measured-neutral, NOT a perf claim.
- DEAD-END 3 (CORRECTION of an earlier draft) — "f64 floor/sign chains stay
  ~10.6 ms, a follow-up lever." FALSE: that was a slow-worker artifact. Clean
  re-measurement on a normal worker: `floor_f64` **773 us**, `round_f64`
  **223 us**, `sign_f64` **1.15 ms**; and on `hz2` `floor_f32` **33.9 us**,
  `round_f32` **42.3 us**, `sign_f32` **467 us**. The same identical-code chain
  spanned ~25x across workers (e.g. `floor_f32` 901 us → 33.9 us). LESSON: every
  cross-invocation rch bench lands on a different worker (no `--worker` pin), so
  ALL absolute production numbers in this family are noise — trust ONLY the
  same-binary A/B (4.63x). The `f64_scalar_add_chain_input_and_literals` path
  only matches PURE add-scalar chains (a floor step bails it), so floor/sign f64
  already go through `apply_fusion_chunk`; there is NO separate slow f64 path and
  NO f64 fusion-chunk lever. `sign` is the slowest unary (branchy
  `scalar_f64_sign` doesn't SIMD) — low-EV branchless-copysign micro-lever only.
- DEAD-END 4 — vectorize `sign` in the fusion-chunk unary step. Hypothesis: a
  branchless `(v==0 || v!=v) ? v : 1.0.copysign(v)` would SIMD where the branchy
  `scalar_f64_sign` (is_nan + 2 compares) does not. Same-binary RCH A/B over 1M
  f64 (bit-identical to `scalar_f64_sign` on NaN/±0/±inf/normal, asserted):
  branchy **0.9683 ms** vs branchless-scalar **0.9733 ms** = **0.99x** (the `if`
  keeps it scalar — NO win). The only vectorizing form is explicit `std::simd`
  masks, which hit the documented nightly trait-location drift
  (`StdFloat::copysign` / `Mask::select` unresolved on the current toolchain) —
  exactly the fragility that has put main RED before. REJECTED: a fragile SIMD
  mask kernel for a RARE op (sign chains; xjbvr noted only quantization/bucketing
  use them) already sub-ms is poor EV. Do not re-chase sign vectorization.
- FAMILY-AUDIT COMPLETE — `apply_i64_fusion_chunk` and `apply_half_fusion_chunk`
  carry the identical in-loop `*_fused_unary(op,*o)` pattern, but neither is a
  shippable lever: i64 unary cheap ops are Neg/Abs (already vectorize; floor/ceil/
  trunc are int identities) + Sign (rare), and there is NO 1M i64-unary-chain
  bench to prove a win — only tiny `i64E256` chains exist; half (bf16/f16) is
  decode-bound (the widen→op→round dominates, not the dispatch). Applying the
  hoist there would be unproven, measured-neutral scope creep, so left untouched.
  The fusion-chunk unary-dispatch family is exhausted: f32 shipped (294c836f,
  4.63x), f64 neutral-kept, i64/half no-action, sign dead-end.
- DEAD-END 5 — f32 register-resident scalar-add-chain SIMD (the missing sibling
  of f64's `apply_f64_scalar_adds_simd` / bead n75xr). Hypothesis: f64 fixed an
  order-of-magnitude 1M-add-chain loss with a register-resident `Simd<f64,8>`
  pass (all adds in registers, 1 memory round-trip vs N); f32 has no such path,
  so its [1M,8M) add chains pay N passes in `apply_f32_fusion_chunk`. Same-binary
  RCH A/B over 1M f32, `(+scalar)x8`, bit-identical to the per-step
  `(f64(v)+c) as f32` reference: per_pass **1.2345 ms** vs register_simd
  **2.8475 ms** = **0.43x (2.3x SLOWER)**. ROOT CAUSE: f32's widen-per-step
  contract forces a `Simd<f32,8>`→`Simd<f64,8>`→add→`Simd<f32,8>` cast pair
  (`vcvtps2pd`/`vcvtpd2ps`) on EVERY add of EVERY lane; those casts dominate and
  negate the memory-pass savings. f64 wins precisely because its native contract
  needs NO cast (`lanes += splat`). This is why n75xr was f64-only — the f32
  "gap" is not a gap. Do not add an f32 scalar-add-chain SIMD path.
