#!/usr/bin/env python3
"""JAX CPU baseline for FrankenJAX BitcastConvertType gauntlet workloads."""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

N = 1_048_576


def percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def bench(name, fn, arg, runs, warmup, inner_loops):
    compiled = jax.jit(fn)
    compiled(arg).block_until_ready()
    for _ in range(warmup):
        for _ in range(inner_loops):
            compiled(arg).block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(inner_loops):
            compiled(arg).block_until_ready()
        times.append((time.perf_counter_ns() - start) / inner_loops)
    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
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
        "cv_pct": std / mean * 100.0 if mean else 0.0,
        "throughput_elements_per_sec": N / (mean / 1e9) if mean else 0.0,
    }


def f32_payload():
    i = jnp.arange(N, dtype=jnp.float32)
    return jnp.sin(i * jnp.float32(0.000_017_3)) + (
        (i % jnp.float32(97.0)) * jnp.float32(0.125)
    )


def f64_payload():
    i = jnp.arange(N, dtype=jnp.float64)
    return jnp.sin(i * jnp.float64(0.000_017_3)) + (
        (i % jnp.float64(97.0)) * jnp.float64(0.125)
    )


def workloads():
    f32 = f32_payload()
    f64 = f64_payload()
    i32 = jax.lax.bitcast_convert_type(f32, jnp.int32)
    u64 = jax.lax.bitcast_convert_type(f64, jnp.uint64)
    u32_chunks = jax.lax.bitcast_convert_type(f64, jnp.uint32)
    bf16_chunks = jax.lax.bitcast_convert_type(f32, jnp.bfloat16)
    return [
        (
            "bitcast_f32_u32_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32),
            f32,
        ),
        (
            "bitcast_f64_i64_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.int64),
            f64,
        ),
        (
            "bitcast_f32_i32_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.int32),
            f32,
        ),
        (
            "bitcast_i32_f32_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.float32),
            i32,
        ),
        (
            "bitcast_f64_u64_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.uint64),
            f64,
        ),
        (
            "bitcast_u64_f64_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.float64),
            u64,
        ),
        (
            "bitcast_f64_u32_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32),
            f64,
        ),
        (
            "bitcast_u32_f64_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.float64),
            u32_chunks,
        ),
        (
            "bitcast_f32_bf16_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.bfloat16),
            f32,
        ),
        (
            "bitcast_bf16_f32_1m",
            lambda x: jax.lax.bitcast_convert_type(x, jnp.float32),
            bf16_chunks,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--inner-loops", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = [
        bench(name, fn, arg, args.runs, args.warmup, args.inner_loops)
        for name, fn, arg in workloads()
    ]
    payload = {
        "schema_version": "frankenjax.bitcast-gauntlet.jax.v1",
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
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
