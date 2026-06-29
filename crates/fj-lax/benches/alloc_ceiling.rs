// Probes the so4wo "eval-model" gate: BW-bound fj-lax ops allocate a fresh output `Vec`
// per call and pay first-touch page-faults that JAX amortizes by reusing buffers. The
// TRUSTWORTHY signal here is the SAME-BINARY `fresh-alloc` vs `reused-buffer` delta
// printed per row — that isolates the alloc/fault overhead within ONE binary and shows
// the so4wo ceiling is real (fresh-alloc costs materially more than the compute floor).
//
// ⚠️ METHODOLOGY WARNING (learned the hard way, 2026-06-27): the cross-allocator
// comparison (`--features mimalloc-alloc` build vs default build) is NOT trustworthy.
// A global allocator is compile-time, so comparing it needs TWO separate binaries — and
// under variable shared-RCH-worker contention the load differential between the two
// builds dwarfs any allocator effect. An early cross-build run here showed a fake ~2-3x
// "mimalloc win"; ProudSalmon's same-load back-to-back A/B then measured mimalloc as
// NEUTRAL-to-2x-WORSE (see docs/NEGATIVE_EVIDENCE.md retraction). The caching-allocator
// lever is DEAD; do NOT draw allocator conclusions from cross-build numbers here. The
// real so4wo fix is eval-model buffer reuse (architectural), not a global allocator.
//
//   rch exec -- cargo bench -p fj-lax --bench alloc_ceiling   # read the SAME-BINARY ratio only

#[cfg(feature = "mimalloc-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "mimalloc-alloc")]
const ALLOC: &str = "mimalloc";
#[cfg(not(feature = "mimalloc-alloc"))]
const ALLOC: &str = "system";

use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

fn best_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::MAX;
    for _ in 0..iters {
        let s = Instant::now();
        f();
        b = b.min(s.elapsed().as_secs_f64());
    }
    b * 1e3
}

fn main() {
    let n = 16_000_000usize;
    let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 9973) as f64 * 0.001).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let p = BTreeMap::new();

    // Production path: each call allocates a fresh 128MB output (dropped -> freed each
    // iter), so the system allocator re-faults every call.
    let fresh = best_ms(20, || {
        let out = eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &p).unwrap();
        black_box(&out);
    });

    // Reused-buffer reference: the same negate written into ONE pre-faulted buffer (no
    // per-call alloc/fault) — the floor the eval-model could reach.
    let mut reuse = vec![0.0f64; n];
    let reused = best_ms(20, || {
        for (o, &x) in reuse.iter_mut().zip(data.iter()) {
            *o = -x;
        }
        black_box(&reuse);
    });

    println!(
        "alloc-ceiling neg 16M f64 [{ALLOC}]: fresh-alloc={fresh:.3}ms reused-buffer={reused:.3}ms ratio={:.2}x",
        fresh / reused
    );

    // Reciprocal: a real JAX-comparable BW-bound op (SlateHarrier 2026-06-25 measured
    // fj-lax ~20-25ms vs JAX 14ms = ~1.5x loss, attributed to per-call alloc/faults).
    // Under a caching allocator the fresh-alloc cost should fall enough to flip that.
    let recip = best_ms(20, || {
        let out = eval_primitive(Primitive::Reciprocal, std::slice::from_ref(&input), &p).unwrap();
        black_box(&out);
    });
    println!("alloc-ceiling reciprocal 16M f64 [{ALLOC}]: fresh-alloc={recip:.3}ms");
}
