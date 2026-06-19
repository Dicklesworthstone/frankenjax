#!/usr/bin/env python3
"""JAX CPU head-to-head for the FrankenJAX convert f64->bf16 downcast lever
(CobaltForge, commit 9bebc33c). Mirrors eval/convert_16m_f64_to_bf16: a dense
16,777,216-element f64 -> bf16 ConvertElementType (materialized)."""
import json, statistics, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

N = 16_777_216  # 1<<24, matches the Rust bench

def percentile(sv, pct):
    k = (len(sv) - 1) * pct / 100.0
    f = int(k); c = min(f + 1, len(sv) - 1)
    return sv[f] + (k - f) * (sv[c] - sv[f])

def bench(name, fn, args, runs=20, warmup=5, inner=20):
    compiled = jax.jit(fn)
    compiled(*args).block_until_ready()
    for _ in range(warmup):
        for _ in range(inner):
            compiled(*args).block_until_ready()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            compiled(*args).block_until_ready()
        times.append((time.perf_counter_ns() - t0) / inner)
    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {"name": name, "p50_ms": percentile(times, 50)/1e6, "mean_ms": mean/1e6,
            "cv_pct": (std/mean*100.0) if mean else 0.0}

x = jnp.asarray(np.linspace(-5.0, 5.0, N), dtype=jnp.float64)
# astype to bf16 forces the downcast; block_until_ready materializes it.
r = bench("convert_16m_f64_to_bf16", lambda a: a.astype(jnp.bfloat16), (x,))
print(json.dumps(r, indent=2))
