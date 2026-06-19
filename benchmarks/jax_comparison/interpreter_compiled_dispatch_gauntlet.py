#!/usr/bin/env python3
"""JAX CPU comparator for fj-interpreters compiled_dispatch_speed.

Mirrors crates/fj-interpreters/benches/compiled_dispatch_speed.rs:
- scalar_i64_chain_n{8,32,128}: x -> x + 1 repeated n times.
- tensor64_chain_n{8,32}: x[64] -> x + 1.0 repeated n times.

The Rust Criterion rows compare eager eval_jaxpr vs CompiledJaxpr in-process.
The JAX rows measure jax.jit CPU call+ready latency for the same logical chain.
"""

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


def percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def bench(name, fn, runs, warmup, inner_loops):
    fn()
    for _ in range(warmup):
        for _ in range(inner_loops):
            fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(inner_loops):
            fn()
        times.append((time.perf_counter_ns() - start) / inner_loops)

    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "name": name,
        "engine": "jax_jit_cpu",
        "p50_ns": percentile(times, 50),
        "p95_ns": percentile(times, 95),
        "p99_ns": percentile(times, 99),
        "mean_ns": mean,
        "cv_pct": (std / mean * 100.0) if mean else 0.0,
    }


def scalar_chain_fn(n):
    def inner(x):
        y = x
        for _ in range(n):
            y = y + jnp.int64(1)
        return y

    return jax.jit(inner)


def tensor64_chain_fn(n):
    def inner(x):
        y = x
        for _ in range(n):
            y = y + 1.0
        return y

    return jax.jit(inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=200)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    scalar_arg = jnp.asarray(0, dtype=jnp.int64)
    tensor_arg = jnp.ones((64,), dtype=jnp.float64)

    results = []
    for n in (8, 32, 128):
        compiled = scalar_chain_fn(n)
        compiled(scalar_arg).block_until_ready()
        results.append(
            bench(
                f"compiled_dispatch_jax_scalar_n={n}",
                lambda compiled=compiled: compiled(scalar_arg).block_until_ready(),
                args.runs,
                args.warmup,
                args.inner_loops,
            )
        )

    for n in (8, 32):
        compiled = tensor64_chain_fn(n)
        compiled(tensor_arg).block_until_ready()
        results.append(
            bench(
                f"compiled_dispatch_jax_tensor64_n={n}",
                lambda compiled=compiled: compiled(tensor_arg).block_until_ready(),
                args.runs,
                args.warmup,
                args.inner_loops,
            )
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "workloads": {
            "scalar": "x -> x + 1 repeated n times, dtype=int64",
            "tensor64": "x[64] -> x + 1.0 repeated n times, dtype=float64",
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    for result in results:
        print(
            f"{result['name']}: mean={result['mean_ns']:.1f}ns "
            f"p50={result['p50_ns']:.1f}ns p95={result['p95_ns']:.1f}ns "
            f"p99={result['p99_ns']:.1f}ns cv={result['cv_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
