#!/usr/bin/env python3
"""JAX head-to-head for dense elementwise binary (add/mul) 1M f64/f32. Mirrors
crates/fj-lax/benches/elementwise_gauntlet.rs."""
import argparse, json, platform, statistics, time
from datetime import datetime, timezone
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
N = 1_048_576
DRAM_N = 16_777_216

def percentile(s,p):
    k=(len(s)-1)*p/100.0; f=int(k); c=min(f+1,len(s)-1)
    return s[f]+(k-f)*(s[c]-s[f])

def bench(name, fn, args, runs, warmup, inner):
    cf=jax.jit(fn); cf(*args).block_until_ready()
    for _ in range(warmup):
        for _ in range(inner): cf(*args).block_until_ready()
    times=[]
    for _ in range(runs):
        st=time.perf_counter_ns()
        for _ in range(inner): cf(*args).block_until_ready()
        times.append((time.perf_counter_ns()-st)/inner)
    times.sort(); mean=statistics.fmean(times); std=statistics.pstdev(times) if len(times)>1 else 0.0
    return {"name":name,"engine":"jax_jit_cpu","p50_ns":percentile(times,50),"mean_ns":mean,"cv_pct":(std/mean*100.0) if mean else 0.0}

def add(a,b): return a+b
def mul(a,b): return a*b
def fma(a,b,c): return a*b+c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",type=int,default=30); ap.add_argument("--warmup",type=int,default=5)
    ap.add_argument("--inner-loops",type=int,default=50); ap.add_argument("--output",type=str,default="")
    a=ap.parse_args()
    a64=jnp.arange(N,dtype=jnp.float64)*1e-6-0.5; b64=jnp.arange(N,dtype=jnp.float64)*2e-6+0.25
    c64=jnp.arange(N,dtype=jnp.float64)*3e-6-0.125
    a32=(jnp.arange(N,dtype=jnp.float32)*1e-6-0.5); b32=(jnp.arange(N,dtype=jnp.float32)*2e-6+0.25)
    c32=(jnp.arange(N,dtype=jnp.float32)*3e-6-0.125)
    da64=jnp.arange(DRAM_N,dtype=jnp.float64)*1e-9-0.5; db64=jnp.arange(DRAM_N,dtype=jnp.float64)*2e-9+0.25
    da32=(jnp.arange(DRAM_N,dtype=jnp.float32)*1e-9-0.5); db32=(jnp.arange(DRAM_N,dtype=jnp.float32)*2e-9+0.25)
    res=[bench("add_f64_1m",add,(a64,b64),a.runs,a.warmup,a.inner_loops),
         bench("add_f32_1m",add,(a32,b32),a.runs,a.warmup,a.inner_loops),
         bench("mul_f64_1m",mul,(a64,b64),a.runs,a.warmup,a.inner_loops),
         bench("fma_f64_1m",fma,(a64,b64,c64),a.runs,a.warmup,a.inner_loops),
         bench("fma_f32_1m",fma,(a32,b32,c32),a.runs,a.warmup,a.inner_loops),
         bench("add_f64_16m",add,(da64,db64),a.runs,a.warmup,a.inner_loops),
         bench("add_f32_16m",add,(da32,db32),a.runs,a.warmup,a.inner_loops),
         bench("mul_f64_16m",mul,(da64,db64),a.runs,a.warmup,a.inner_loops)]
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"engine":"jax_jit_cpu","jax_version":jax.__version__,"platform":platform.platform(),"results":res}
    t=json.dumps(payload,indent=2)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(t)
    print(t)
    for r in res: print(f"{r['name']}: mean={r['mean_ns']:.1f}ns p50={r['p50_ns']:.1f}ns cv={r['cv_pct']:.2f}%")

if __name__=="__main__": main()
