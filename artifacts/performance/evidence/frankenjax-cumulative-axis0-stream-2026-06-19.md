# Streaming leading-axis (axis 0) cumulative — closes a 3.4x JAX LOSS to parity

Agent: `WildForge`
Date: 2026-06-19
Files: `crates/fj-lax/src/reduction.rs`, `crates/fj-lax/benches/lax_baseline.rs`

## Lever

`eval_cumulative_dense` threads the contiguous (trailing-axis) scan and now also
streams the F32 trailing scan, but **leading-axis cumulatives (`axis == 0`, scan DOWN
the columns) fell to the generic strided serial loop**:

```rust
for outer in 0..outer_count {            // one column at a time
    for i in 0..axis_dim { acc = op(acc, out[base + i*axis_stride]); out[..] = acc; }
}
```

For `[16384, 1024]` that re-reads each output column at **stride 1024** — cache-hostile.

New `scan_leading_axis_to_vec` restructures to **k-outer / column-inner**: it keeps a
`cols`-wide f64 accumulator (L1-resident) and at each row reads/writes the CONTIGUOUS
slice `src[k*cols ..][..cols]`. Each column folds `k` in the same ascending (or
reverse) order as the serial scan, so it is BIT-IDENTICAL (the f64 accumulator + per
step `narrow` matches both the F64 and F32 serial contracts; NaN sign of inf−inf
cancellations is non-contractual, as for any reduction). Wired for dense F64 and F32
(`axis == 0`); other strided axes keep the serial gather.

## Head-to-head vs JAX (worker hz2, EPYC-Genoa 16 vCPU)

Workload: `cumsum` over axis 0 of a `[16384, 1024]` f64 tensor → `[16384, 1024]`
(16.7M elements; 1024 independent 16384-deep columns).

| Variant | Time (median) | vs JAX |
| --- | ---: | ---: |
| JAX `jit(cumsum, axis=0)` x64 (p50) | 81.98 ms | 1.00x (baseline) |
| fj-lax serial strided (pre-lever) | 275.99 ms | **0.30x (3.37x LOSS)** |
| fj-lax streaming-serial (this lever) | 79.45 ms | **1.03x (parity, loss removed)** |

- Speedup over the old strided-serial path: `275.99 / 79.45 = 3.47x`, bit-exact
  (finite/±inf; NaN canonicalized).
- Turns a 3.37x JAX **loss** into rough **parity** (1.03x) — the cache-access fix
  alone, no threading.
- F32 leading-axis rides the same path (JAX default dtype).

### Note / follow-up

79 ms for ~268 MB of traffic is ~3.4 GB/s — well below DRAM bandwidth — so the
streaming path is **closure/compute-bound** (the `op: impl Fn` per element), not yet
memory-bound. A threaded version could win outright, but the cumsum output is
column-strided (full-size), so safe disjoint threading needs a column-major scratch +
strided scatter-back; that scatter risks eating the gain. Left as a follow-up; the
safe streaming-serial already removes the loss.

JAX baseline command:

```bash
benchmarks/jax_comparison/.venv/bin/python \
  benchmarks/jax_comparison/cumsum_axis0_gauntlet.py
# cumsum_axis0_f64 {"p50_ms": 81.98, ...}
```

fj-lax bench command:

```bash
RCH_WORKER=hz2 RCH_WORKERS=hz2 RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- cumsum_16kx1k_f64_axis0
# streaming-serial: time: [77.684 ms 79.447 ms 81.500 ms]
# serial strided (pre-edit): time: [273.59 ms 275.99 ms 278.80 ms]
```

## Behavior proof (bit-exact)

`cargo test --release -p fj-lax --lib cumulative` → `8 passed; 0 failed`, including the
new `leading_axis_cumulative_matches_serial_reference` (512×384 F64 + F32, all four
ops × forward/reverse, special values −0.0/NaN/±inf, vs an independent per-column
serial reference with NaN canonicalized).

## Negative evidence — attempts to push parity → a clear win (all REJECTED)

After landing the streaming-serial path at ~79 ms (parity), four follow-ups were
measured on the same bench to try to dominate JAX. All failed — cumsum-axis0 is
**parity-bound**:

| Attempt | Result | Verdict |
| --- | ---: | --- |
| Iterator-zip inner loop (elide bounds checks) | 74.69 ms | kept (6% + cleaner; only real gain) |
| Direct `+=` specialization for Cumsum (no `op` closure) | 80.59 ms | no vectorization win — REVERTED |
| `Vec::push` forward (skip the `vec![fill;total]` memset) | 87.35 ms | push capacity-checks cost MORE than the memset — REVERTED |
| Threaded over columns | not attempted to ship | output is column-strided/full-size → needs a column-major scratch + strided scatter-back; the scatter risks eating the gain |

Conclusion: the bottleneck is neither closure indirection (direct `+` didn't help) nor
the output zero-fill (push was slower); at ~3.6 GB/s the op is bound by the
strided/dependent write+accumulate pattern that does not vectorize. The iterator-zip
form is the safe optimum; do NOT re-attempt direct-op specialization or push-based
output for leading-axis cumulative.

## Ledger

NEUTRAL/gap-closer vs JAX (1.07x at the kept 74.7 ms) but a clear **3.7x internal**
win that removes a 3.37x JAX loss; Confidence 0.97 (bit-exact finite/inf, gated
`axis==0`); Effort low. Companion to the same-day argmax/cumulative line-threading
wins.
