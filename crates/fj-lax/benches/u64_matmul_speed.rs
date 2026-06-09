//! Same-invocation A/B for the u64/u32 matmul kernel (`rank2_u64_matmul`).
//!
//! Arm B (reference) is the ORIGINAL single-row i-k-j wrapping loop; arm A is the
//! NEW 4-row register-blocked kernel sharing one `brow` load across four output
//! rows. Both run single-threaded over the full matrix in ONE process and the
//! bench asserts bit-identical output before timing. u64 wrapping arithmetic is a
//! commutative ring (`Z/2^64`), so the interleaving is exact. This kernel backs
//! both u64 and u32 (narrowed) canonical matmul.
//!
//! Run: `cargo bench -p fj-lax --bench u64_matmul_speed`.

use std::hint::black_box;
use std::time::Instant;

/// Original single-row i-k-j wrapping kernel (pre-optimization reference).
fn matmul_u64_single_row(a: &[u64], b: &[u64], k: usize, n: usize, c: &mut [u64]) {
    for (ri, crow) in c.chunks_mut(n).enumerate() {
        let a_off = ri * k;
        for l in 0..k {
            let av = a[a_off + l];
            let brow = &b[l * n..l * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv = cv.wrapping_add(av.wrapping_mul(bv));
            }
        }
    }
}

/// New 4-row register-blocked kernel (mirrors the shipped `rank2_u64_matmul`).
fn matmul_u64_row_block4(a: &[u64], b: &[u64], k: usize, n: usize, c: &mut [u64]) {
    let rows = c.len() / n;
    let full = rows - rows % 4;
    let (blocked, tail) = c.split_at_mut(full * n);
    for (g, four) in blocked.chunks_mut(4 * n).enumerate() {
        let (c0, rest) = four.split_at_mut(n);
        let (c1, rest) = rest.split_at_mut(n);
        let (c2, c3) = rest.split_at_mut(n);
        let base = (g * 4) * k;
        let (a0o, a1o, a2o, a3o) = (base, base + k, base + 2 * k, base + 3 * k);
        for l in 0..k {
            let a0 = a[a0o + l];
            let a1 = a[a1o + l];
            let a2 = a[a2o + l];
            let a3 = a[a3o + l];
            let brow = &b[l * n..l * n + n];
            for ((((e0, e1), e2), e3), &bv) in c0
                .iter_mut()
                .zip(c1.iter_mut())
                .zip(c2.iter_mut())
                .zip(c3.iter_mut())
                .zip(brow)
            {
                *e0 = e0.wrapping_add(a0.wrapping_mul(bv));
                *e1 = e1.wrapping_add(a1.wrapping_mul(bv));
                *e2 = e2.wrapping_add(a2.wrapping_mul(bv));
                *e3 = e3.wrapping_add(a3.wrapping_mul(bv));
            }
        }
    }
    for (ri_rem, crow) in tail.chunks_mut(n).enumerate() {
        let a_off = (full + ri_rem) * k;
        for l in 0..k {
            let av = a[a_off + l];
            let brow = &b[l * n..l * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv = cv.wrapping_add(av.wrapping_mul(bv));
            }
        }
    }
}

fn bench_size(m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<u64> = (0..(m * k) as u64)
        .map(|i| (1u64 << 40).wrapping_add(i.wrapping_mul(2_654_435_761)))
        .collect();
    let b: Vec<u64> = (0..(k * n) as u64)
        .map(|i| (1u64 << 40).wrapping_add(i.wrapping_mul(40_503).wrapping_add(3)))
        .collect();

    let mut c_ref = vec![0u64; m * n];
    let mut c_new = vec![0u64; m * n];
    matmul_u64_single_row(&a, &b, k, n, &mut c_ref);
    matmul_u64_row_block4(&a, &b, k, n, &mut c_new);
    assert_eq!(c_ref, c_new, "[{m},{k}]@[{k},{n}] row-block4 != single-row");

    let mut c = vec![0u64; m * n];
    matmul_u64_single_row(&a, &b, k, n, &mut c);
    let t0 = Instant::now();
    for _ in 0..iters {
        c.iter_mut().for_each(|x| *x = 0);
        matmul_u64_single_row(black_box(&a), black_box(&b), k, n, &mut c);
        black_box(&c);
    }
    let single = t0.elapsed().as_nanos() as f64 / iters as f64;

    matmul_u64_row_block4(&a, &b, k, n, &mut c);
    let t1 = Instant::now();
    for _ in 0..iters {
        c.iter_mut().for_each(|x| *x = 0);
        matmul_u64_row_block4(black_box(&a), black_box(&b), k, n, &mut c);
        black_box(&c);
    }
    let blocked = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "U64_MATMUL m={m} k={k} n={n} single_row={:.3}ms row_block4={:.3}ms speedup={:.2}x",
        single / 1e6,
        blocked / 1e6,
        single / blocked,
    );
}

fn main() {
    bench_size(512, 512, 512, 12); // L2/L3-resident
    bench_size(1536, 1536, 1536, 3); // B = 18 MB, >L3
    bench_size(2048, 2048, 2048, 2); // B = 32 MB, RAM-bound
}
