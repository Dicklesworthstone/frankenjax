//! Same-invocation A/B for threaded distribution transcendental transforms.
//!
//! The inverse-transform distributions draw uniforms (already threaded) then apply
//! a per-element `ln`/`tan` map. Arm B (reference) is that map run SERIALLY over
//! freshly-drawn uniforms — exactly the pre-change distribution body. Arm A is the
//! shipped public distribution, whose map now fans out across threads via
//! `map_uniforms_parallel`. Both arms draw the SAME (threaded) uniforms, so the
//! ratio isolates the transform-threading lever. Each case asserts arm A == arm B
//! bit-for-bit before timing — the win must be free.
//!
//! Run: `cargo bench -p fj-lax --bench rng_dist_transform_threading`.

use std::hint::black_box;
use std::time::Instant;

use fj_lax::threefry::{random_cauchy, random_gumbel, random_key, random_uniform};

fn time<F: FnMut() -> Vec<f64>>(iters: usize, mut f: F) -> f64 {
    for _ in 0..2 {
        black_box(f());
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

fn bench_case(
    name: &str,
    count: usize,
    mut serial: impl FnMut() -> Vec<f64>,
    mut threaded: impl FnMut() -> Vec<f64>,
) {
    let s_once = serial();
    let t_once = threaded();
    assert_eq!(s_once.len(), t_once.len(), "{name}: len mismatch");
    for (idx, (s, t)) in s_once.iter().zip(t_once.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            t.to_bits(),
            "{name}: threaded != serial at idx={idx} (count={count})"
        );
    }
    let iters = (30_000_000 / count).max(6);
    let serial_us = time(iters, &mut serial) * 1e6;
    let threaded_us = time(iters, &mut threaded) * 1e6;
    println!(
        "{name:<14} count={count:>9}  serial={serial_us:>9.3}us  threaded={threaded_us:>9.3}us  speedup={:.2}x",
        serial_us / threaded_us
    );
}

fn main() {
    println!(
        "parallelism = {}",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    );
    let key = random_key(0x1234_5678_9ABC_DEF0);
    let eps = f64::from(f32::EPSILON);
    let tiny = f64::from(f32::MIN_POSITIVE);
    for &count in &[131_072_usize, 262_144, 524_288, 1_048_576, 4_194_304, 16_777_216] {
        // Cauchy: tan transform.
        bench_case(
            "cauchy(tan)",
            count,
            || {
                random_uniform(key, count, eps, 1.0)
                    .into_iter()
                    .map(|u| (std::f64::consts::PI * (u - 0.5)).tan())
                    .collect()
            },
            || random_cauchy(key, count),
        );
        // Gumbel: double-ln transform (loc=0, scale=1).
        bench_case(
            "gumbel(2*ln)",
            count,
            || {
                random_uniform(key, count, tiny, 1.0)
                    .into_iter()
                    .map(|u| 0.0 - 1.0 * (-u.ln()).ln())
                    .collect()
            },
            || random_gumbel(key, count, 0.0, 1.0),
        );
    }
}
