//! Same-invocation A/B for threaded `generate_u32_bits`.
//!
//! Arm B (reference) is the single-threaded SIMD generator
//! (`generate_u32_bits_serial`) — exactly the pre-change `generate_u32_bits` body.
//! Arm A is the new `generate_u32_bits`, which fans the counter-based ThreeFry bit
//! generation out across cores. Both run in ONE process so the ratio is
//! trustworthy. The bench asserts word-for-word identity before timing — the win
//! must be free, since ThreeFry is counter-based (word i always uses absolute
//! counter i). This primitive underpins randint (two draws), random_bits, and the
//! sort/argsort key path.
//!
//! Run: `cargo bench -p fj-lax --bench rng_bits_threading`.

use std::hint::black_box;
use std::time::Instant;

use fj_lax::threefry::{generate_u32_bits, generate_u32_bits_serial, random_key};

fn bench_one(count: usize) {
    let key = random_key(0x1234_5678_9ABC_DEF0);

    // Word-identity gate: the threaded output must equal the serial baseline.
    let serial_once = generate_u32_bits_serial(key, count);
    let threaded_once = generate_u32_bits(key, count);
    assert_eq!(serial_once.len(), threaded_once.len());
    assert!(
        serial_once == threaded_once,
        "threaded != serial at count={count}"
    );

    let iters = (200_000_000 / count).max(8);

    for _ in 0..2 {
        black_box(generate_u32_bits_serial(key, count));
        black_box(generate_u32_bits(key, count));
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(generate_u32_bits_serial(black_box(key), black_box(count)));
    }
    let serial = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        black_box(generate_u32_bits(black_box(key), black_box(count)));
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
        131_072_usize,
        262_144,
        524_288,
        1_048_576,
        4_194_304,
        16_777_216,
    ] {
        bench_one(count);
    }
}
