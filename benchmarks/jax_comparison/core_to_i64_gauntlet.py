#!/usr/bin/env python3
"""JAX CPU head-to-head for fj-core TensorValue::to_i64_vec.

Mirrors crates/fj-core/benches/core_baseline.rs:
core/tensor_to_i64_vec_dense_4k extracts one 4096-element I64 tensor into an
owned host vector.

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/core_to_i64_gauntlet.py \
      --runs 100 --warmup 10 --inner-loops 1000 --output /tmp/frankenjax_to_i64_jax.json
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
import numpy as np  # noqa: E402

WIDTH = 4096


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=500)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    x = jnp.arange(WIDTH, dtype=jnp.int64) - jnp.int64(WIDTH // 2)
    identity = jax.jit(lambda v: v)
    identity(x).block_until_ready()

    def identity_ready():
        identity(x).block_until_ready()

    def identity_numpy_copy():
        np.asarray(identity(x).block_until_ready()).copy()

    results = [
        bench(
            "tensor_to_i64_vec_i64_4k_identity_ready",
            identity_ready,
            args.runs,
            args.warmup,
            args.inner_loops,
        ),
        bench(
            "tensor_to_i64_vec_i64_4k_numpy_copy",
            identity_numpy_copy,
            args.runs,
            args.warmup,
            args.inner_loops,
        ),
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "workloads": {
            "tensor_to_i64_vec_i64_4k_identity_ready": {
                "shape": [WIDTH],
                "elements": WIDTH,
                "expr": "jax.jit(lambda x: x)(x).block_until_ready()",
            },
            "tensor_to_i64_vec_i64_4k_numpy_copy": {
                "shape": [WIDTH],
                "elements": WIDTH,
                "expr": "np.asarray(jax.jit(lambda x: x)(x).block_until_ready()).copy()",
            },
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
