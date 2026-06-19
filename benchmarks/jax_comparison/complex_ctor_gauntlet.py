#!/usr/bin/env python3
"""JAX head-to-head for the complex constructor (frankenjax-7eqrs): lax.complex(re,im)
-> complex128, 1M. Mirrors crates/fj-lax/benches/complex_ctor_gauntlet.rs."""
import argparse, json, platform, statistics, time
from datetime import datetime, timezone
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
N = 1_048_576

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

def complex_ctor(re, im):
    return jax.lax.complex(re, im)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",type=int,default=30); ap.add_argument("--warmup",type=int,default=5)
    ap.add_argument("--inner-loops",type=int,default=50); ap.add_argument("--output",type=str,default="")
    a=ap.parse_args()
    re=jnp.arange(N,dtype=jnp.float64)*1e-6-0.5
    im=jnp.arange(N,dtype=jnp.float64)*2e-6+0.25
    res=[bench("complex_ctor_re_im_to_c128_1m",complex_ctor,(re,im),a.runs,a.warmup,a.inner_loops)]
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"engine":"jax_jit_cpu","jax_version":jax.__version__,"platform":platform.platform(),"results":res}
    t=json.dumps(payload,indent=2)
    if a.output: open(a.output,"w").write(t)
    print(t)
    for r in res: print(f"{r['name']}: mean={r['mean_ns']:.1f}ns p50={r['p50_ns']:.1f}ns cv={r['cv_pct']:.2f}%")

if __name__=="__main__": main()
