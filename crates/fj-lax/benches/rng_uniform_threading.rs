//! Same-invocation A/B for threaded `random_uniform`.
//!
//! Arm B (reference) is the single-threaded SIMD generator
//! (`random_uniform_serial_simd`) — exactly the pre-change `random_uniform` body.
//! Arm A is the new `random_uniform`, which fans the counter-based ThreeFry
//! generation out across cores. Both run in ONE process so the ratio is
//! trustworthy (no cross-invocation worker drift). The bench asserts the two
//! produce BIT-IDENTICAL output before timing — the win must be free, since
//! ThreeFry is counter-based (element i always uses absolute counter i).
//!
//! Run: `cargo bench -p fj-lax --bench rng_uniform_threading`.

use std::hint::black_box;
use std::time::Instant;

use fj_lax::threefry::{random_key, random_uniform, random_uniform_serial_simd};

fn bench_one(count: usize) {
    let key = random_key(0x1234_5678_9ABC_DEF0);
    let (lo, hi) = (-1.0_f64, 1.0_f64);

    // Bit-identity gate: the threaded output must equal the serial baseline.
    let serial_once = random_uniform_serial_simd(key, count, lo, hi);
    let threaded_once = random_uniform(key, count, lo, hi);
    assert_eq!(serial_once.len(), threaded_once.len());
    for (idx, (s, t)) in serial_once.iter().zip(threaded_once.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            t.to_bits(),
            "threaded != serial at idx={idx} (count={count})"
        );
    }

    let iters = (50_000_000 / count).max(8);

    // Warm up both arms.
    for _ in 0..2 {
        black_box(random_uniform_serial_simd(key, count, lo, hi));
        black_box(random_uniform(key, count, lo, hi));
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(random_uniform_serial_simd(
            black_box(key),
            black_box(count),
            lo,
            hi,
        ));
    }
    let serial = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        black_box(random_uniform(black_box(key), black_box(count), lo, hi));
    }
    let threaded = t1.elapsed().as_secs_f64() / iters as f64;

    println!(
        "count={count:>9}  serial={:>9.3}us  threaded={:>9.3}us  speedup={:.2}x",
        serial * 1e6,
        threaded * 1e6,
        serial / threaded
    );
}

fn main() {
    println!(
        "parallelism = {}",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    );
    for &count in &[
        65_536_usize,
        262_144,
        524_288,
        1_048_576,
        4_194_304,
        16_777_216,
    ] {
        bench_one(count);
    }
}
