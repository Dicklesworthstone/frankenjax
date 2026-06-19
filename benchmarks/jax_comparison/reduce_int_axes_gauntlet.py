import json, statistics, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
R,C=16384,1024
def pct(sv,p):
    k=(len(sv)-1)*p/100.0;f=int(k);c=min(f+1,len(sv)-1);return sv[f]+(k-f)*(sv[c]-sv[f])
def bench(fn,args,runs=25,warm=6,inner=30):
    cj=jax.jit(fn);cj(*args).block_until_ready()
    for _ in range(warm):
        for _ in range(inner): cj(*args).block_until_ready()
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter_ns()
        for _ in range(inner): cj(*args).block_until_ready()
        ts.append((time.perf_counter_ns()-t0)/inner)
    ts.sort();m=statistics.fmean(ts);return {"p50_ms":pct(ts,50)/1e6,"mean_ms":m/1e6}
x=jnp.asarray((np.arange(R*C)%977-488).astype(np.int64).reshape(R,C))
print("axis0", json.dumps(bench(lambda a: jnp.sum(a,axis=0),(x,))))
print("axis1", json.dumps(bench(lambda a: jnp.sum(a,axis=1),(x,))))
