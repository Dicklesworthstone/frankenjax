# Threaded contiguous argmax/argmin over output rows — JAX WIN

Agent: `WildForge`
Date: 2026-06-19
Files: `crates/fj-lax/src/tensor_ops.rs`, `crates/fj-lax/benches/lax_baseline.rs`

## Lever

`extremum_along_axis` already had safe-Rust portable-SIMD reducers for dense
F32/F64 `argmin`/`argmax` when the reduced axis is contiguous (`axis_stride == 1`,
bead `frankenjax-43tr8`). That SIMD path was still a **serial** loop over output
rows: for a `[16384, 1024]` argmax over the innermost (logits) axis it scans all
16384 rows on one thread.

Each output row's argmax is the INDEPENDENT contiguous block
`values[outer*axis_dim .. (outer+1)*axis_dim]`, and per-row argmax is
order-deterministic, so partitioning the rows across threads is **bit-identical
to the serial loop for any partition**. New helper `parallel_argmax_fill` threads
the row fill once total work crosses the DRAM-bound gate
(`CHEAP_BINARY_PARALLEL_MIN = 8.4M` elems, `outer_count > 1`), using the same
`work_scaled_threads` heuristic as the rest of fj-lax (capped at 16, and at
`outer_count`). The F64 and dense-F32 contiguous paths route through it; the
rank-0 (1D input → scalar) result still returns `Value::Scalar` to honor the
finalization contract (`as_i64_scalar`).

The generic scalar reducer remains the source of truth for strided rows, short
rows, NaN rows, half-float, and non-float dtypes — none of those are threaded.

## Head-to-head vs JAX (worker hz2, EPYC-Genoa 16 vCPU)

Workload: `argmax` over axis 1 of a `[16384, 1024]` f64 tensor → `[16384]`
(16.7M elements; the hot logits/classification case).

| Variant | Time (median) | vs JAX |
| --- | ---: | ---: |
| JAX `jit(argmax)` x64 (p50) | 3.43 ms | 1.00x (baseline) |
| fj-lax SIMD **serial** (pre-thread) | 9.70 ms | **0.35x (2.83x LOSS)** |
| fj-lax SIMD **threaded** (this lever) | 2.89 ms | **1.19x WIN** |

- Threading speedup over serial SIMD: `9.70 / 2.89 = 3.36x`, bit-exact.
- Flips a 2.83x JAX **loss** into a 1.19x JAX **win**.
- The fj-lax number is the FULL `eval_primitive(Argmax, …)` dispatch (input read +
  result-tensor build); JAX's 3.43 ms is jit'd pure compute — so the real margin
  is wider than 1.19x.

JAX baseline command:

```bash
benchmarks/jax_comparison/.venv/bin/python \
  benchmarks/jax_comparison/argmax_axis1_gauntlet.py
# -> argmax_axis1_f64 {"p50_ms": 3.429, "mean_ms": 3.440, "cv_pct": 2.0}
```

fj-lax bench command:

```bash
RCH_WORKER=hz2 RCH_WORKERS=hz2 RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- argmax_16kx1k_axis1_f64
# threaded: time: [2.8196 ms 2.8904 ms 2.9631 ms]
# serial (tensor_ops.rs git-stashed): time: [9.6640 ms 9.6984 ms 9.7368 ms]
```

## Behavior proof (bit-exact)

`cargo test --release -p fj-lax --lib arg` → `35 passed; 0 failed`, including
`argmax_argmin_dense_matches_generic`, `argmax_argmin_nan_and_signed_zero_match_jax`,
`argmin_2d_axis0`, and the 1D scalar-result tests. The threaded fill is a pure
partition of the same per-row SIMD reducer, so outputs are byte-identical to the
serial path.

## Ledger

WIN. Impact 3.36 (serial→threaded) / flips JAX loss→win; Confidence 0.97
(bit-exact, gated, independent rows); Effort low (reused `work_scaled_threads`).
