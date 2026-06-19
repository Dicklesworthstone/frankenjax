#!/usr/bin/env python3
"""JAX CPU baseline for the FrankenJAX clamp gauntlet workloads."""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

N = 1_048_576


def percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def bench(name, fn, args, runs, warmup, inner_loops):
    compiled = jax.jit(fn)
    compiled(*args).block_until_ready()
    for _ in range(warmup):
        for _ in range(inner_loops):
            compiled(*args).block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(inner_loops):
            compiled(*args).block_until_ready()
        times.append((time.perf_counter_ns() - start) / inner_loops)
    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    cv_pct = std / mean * 100.0 if mean else 0.0
    return {
        "name": name,
        "engine": "jax_jit_cpu",
        "runs": runs,
        "inner_loops": inner_loops,
        "p50_ns": percentile(times, 50),
        "p95_ns": percentile(times, 95),
        "p99_ns": percentile(times, 99),
        "mean_ns": mean,
        "std_ns": std,
        "cv_pct": cv_pct,
        "throughput_elements_per_sec": N / (mean / 1e9) if mean else 0.0,
    }


def f32_mixed():
    x = jnp.arange(N, dtype=jnp.float32) * jnp.float32(0.001) - jnp.float32(500.0)
    hi = jnp.float32(2.0) + (jnp.arange(N, dtype=jnp.float32) % jnp.float32(8192.0)) * jnp.float32(0.0005)
    lo = jnp.array(0.0, dtype=jnp.float32)
    return "f32_mixed_scalar_tensor_1m", lambda lo, x, hi: jnp.clip(x, lo, hi), (lo, x, hi)


def f64_mixed():
    x = jnp.arange(N, dtype=jnp.float64) * jnp.float64(0.001) - jnp.float64(500.0)
    hi = jnp.float64(2.0) + (jnp.arange(N, dtype=jnp.float64) % jnp.float64(8192.0)) * jnp.float64(0.0005)
    lo = jnp.array(0.0, dtype=jnp.float64)
    return "f64_mixed_scalar_tensor_1m", lambda lo, x, hi: jnp.clip(x, lo, hi), (lo, x, hi)


def half_mixed(dtype, name):
    base = jnp.arange(N, dtype=jnp.float32) % jnp.float32(64.0)
    x = (jnp.float32(1.0) + base * jnp.float32(0.015625)).astype(dtype)
    hi = (jnp.float32(2.0) + base * jnp.float32(0.015625)).astype(dtype)
    lo = jnp.array(0.0, dtype=dtype)
    return name, lambda lo, x, hi: jnp.clip(x, lo, hi), (lo, x, hi)


def half_tensor(dtype, name):
    base = jnp.arange(N, dtype=jnp.float32) % jnp.float32(64.0)
    lo = (jnp.float32(0.5) + base * jnp.float32(0.0078125)).astype(dtype)
    x = (jnp.float32(1.0) + base * jnp.float32(0.015625)).astype(dtype)
    hi = (jnp.float32(2.0) + base * jnp.float32(0.015625)).astype(dtype)
    return name, lambda lo, x, hi: jnp.clip(x, lo, hi), (lo, x, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--inner-loops", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workloads = [
        f32_mixed(),
        f64_mixed(),
        half_mixed(jnp.bfloat16, "bf16_mixed_scalar_tensor_1m"),
        half_mixed(jnp.float16, "f16_mixed_scalar_tensor_1m"),
        half_tensor(jnp.bfloat16, "bf16_tensor_tensor_tensor_1m"),
        half_tensor(jnp.float16, "f16_tensor_tensor_tensor_1m"),
    ]
    results = [
        bench(name, fn, fn_args, args.runs, args.warmup, args.inner_loops)
        for name, fn, fn_args in workloads
    ]
    payload = {
        "schema_version": "frankenjax.clamp-gauntlet.jax.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs,
        "warmup": args.warmup,
        "inner_loops": args.inner_loops,
        "element_count": N,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
