import json, statistics, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
N=16_777_216
def pct(sv,p):
    k=(len(sv)-1)*p/100.0;f=int(k);c=min(f+1,len(sv)-1);return sv[f]+(k-f)*(sv[c]-sv[f])
def bench(fn,args,runs=20,warm=5,inner=20):
    cj=jax.jit(fn);cj(*args).block_until_ready()
    for _ in range(warm):
        for _ in range(inner): cj(*args).block_until_ready()
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter_ns()
        for _ in range(inner): cj(*args).block_until_ready()
        ts.append((time.perf_counter_ns()-t0)/inner)
    ts.sort();m=statistics.fmean(ts)
    return {"p50_ms":pct(ts,50)/1e6,"mean_ms":m/1e6,"cv_pct":statistics.pstdev(ts)/m*100}
x=jnp.asarray((np.arange(N)%1000-500).astype(np.int64))
print(json.dumps(bench(lambda a: jnp.sum(a),(x,)),indent=2))
