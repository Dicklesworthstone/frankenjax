#!/usr/bin/env python3
"""JAX CPU head-to-head for FrankenJAX fused 2D softmax (nn/softmax_2d_65536x16_fused).
Shape [65536, 16] f64, softmax along axis=-1 (the row), matching the Rust bench input."""
import json, statistics, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROWS, COLS = 65_536, 16

def percentile(sv, pct):
    k = (len(sv) - 1) * pct / 100.0
    f = int(k); c = min(f + 1, len(sv) - 1)
    return sv[f] + (k - f) * (sv[c] - sv[f])

def bench(name, fn, args, runs=30, warmup=8, inner=50):
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
    return {"name": name, "p50_us": percentile(times, 50)/1e3, "mean_us": mean/1e3,
            "cv_pct": (std/mean*100.0) if mean else 0.0}

k = np.arange(ROWS*COLS, dtype=np.float64)
x = jnp.asarray((np.sin(k*0.013)*4.0).reshape(ROWS, COLS))
r = bench("softmax_2d_65536x16", lambda a: jax.nn.softmax(a, axis=-1), (x,))
print(json.dumps(r, indent=2))
