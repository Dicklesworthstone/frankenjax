//! Same-invocation A/B for the i64 matmul row-block kernel.
//!
//! Arm B (reference) is the ORIGINAL single-row i-k-j wrapping loop — exactly the
//! pre-change `rank2_i64_row_block` body. Arm A is the NEW 4-row register-blocked
//! kernel that shares one `b_row` load across four output rows. Both run
//! single-threaded over the full matrix in ONE process, so the ratio is
//! trustworthy (no cross-invocation worker drift). The bench also asserts the two
//! kernels produce BIT-IDENTICAL output before timing — the win must be free.
//!
//! Run: `cargo bench -p fj-lax --bench i64_matmul_speed`.

use std::hint::black_box;
use std::time::Instant;

/// Original single-row i-k-j wrapping kernel (the pre-optimization reference).
fn matmul_i64_single_row(a: &[i64], b: &[i64], k: usize, n: usize, c: &mut [i64]) {
    for (ri, c_row) in c.chunks_mut(n).enumerate() {
        let a_off = ri * k;
        for l in 0..k {
            let a_il = a[a_off + l];
            let b_row = &b[l * n..l * n + n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row) {
                *cv = cv.wrapping_add(a_il.wrapping_mul(bv));
            }
        }
    }
}

/// New 4-row register-blocked kernel (mirrors the shipped `rank2_i64_row_block`).
fn matmul_i64_row_block4(a: &[i64], b: &[i64], k: usize, n: usize, c: &mut [i64]) {
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
            let b_row = &b[l * n..l * n + n];
            for ((((e0, e1), e2), e3), &bv) in c0
                .iter_mut()
                .zip(c1.iter_mut())
                .zip(c2.iter_mut())
                .zip(c3.iter_mut())
                .zip(b_row)
            {
                *e0 = e0.wrapping_add(a0.wrapping_mul(bv));
                *e1 = e1.wrapping_add(a1.wrapping_mul(bv));
                *e2 = e2.wrapping_add(a2.wrapping_mul(bv));
                *e3 = e3.wrapping_add(a3.wrapping_mul(bv));
            }
        }
    }
    for (ri_rem, c_row) in tail.chunks_mut(n).enumerate() {
        let a_off = (full + ri_rem) * k;
        for l in 0..k {
            let a_il = a[a_off + l];
            let b_row = &b[l * n..l * n + n];
            for (cv, &bv) in c_row.iter_mut().zip(b_row) {
                *cv = cv.wrapping_add(a_il.wrapping_mul(bv));
            }
        }
    }
}

fn bench_size(m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<i64> = (0..m * k)
        .map(|i| (i as i64).wrapping_mul(2_654_435_761).wrapping_sub(7))
        .collect();
    let b: Vec<i64> = (0..k * n)
        .map(|i| (i as i64).wrapping_mul(40_503).wrapping_add(3))
        .collect();

    // Bit-identity gate: the optimization must change nothing observable.
    let mut c_ref = vec![0i64; m * n];
    let mut c_new = vec![0i64; m * n];
    matmul_i64_single_row(&a, &b, k, n, &mut c_ref);
    matmul_i64_row_block4(&a, &b, k, n, &mut c_new);
    assert_eq!(c_ref, c_new, "[{m},{k}]@[{k},{n}] row-block4 != single-row");

    // Warm + time reference.
    let mut c = vec![0i64; m * n];
    matmul_i64_single_row(&a, &b, k, n, &mut c);
    let t0 = Instant::now();
    for _ in 0..iters {
        c.iter_mut().for_each(|x| *x = 0);
        matmul_i64_single_row(black_box(&a), black_box(&b), k, n, &mut c);
        black_box(&c);
    }
    let single = t0.elapsed().as_nanos() as f64 / iters as f64;

    // Warm + time new.
    matmul_i64_row_block4(&a, &b, k, n, &mut c);
    let t1 = Instant::now();
    for _ in 0..iters {
        c.iter_mut().for_each(|x| *x = 0);
        matmul_i64_row_block4(black_box(&a), black_box(&b), k, n, &mut c);
        black_box(&c);
    }
    let blocked = t1.elapsed().as_nanos() as f64 / iters as f64;

    // Production THREADED path (what eval_primitive(Dot) on i64 tensors uses) — the real
    // head-to-head vs JAX. XLA has no integer BLAS, so JAX falls back to a scalar int matmul.
    let _ = fj_lax::tensor_contraction::rank2_i64_matmul(&a, m, k, &b, n);
    let t2 = Instant::now();
    for _ in 0..iters {
        let p = fj_lax::tensor_contraction::rank2_i64_matmul(black_box(&a), m, k, black_box(&b), n);
        black_box(&p);
    }
    let prod = t2.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "I64_MATMUL m={m} k={k} n={n} single_row={:.3}ms row_block4={:.3}ms speedup={:.2}x prod_threaded={:.3}ms",
        single / 1e6,
        blocked / 1e6,
        single / blocked,
        prod / 1e6,
    );
}

fn main() {
    // L2/L3-resident (B small): blocking's B-reuse barely matters here.
    bench_size(512, 512, 512, 12);
    bench_size(1024, 1024, 1024, 6); // direct head-to-head vs JAX i64 1024^3
    // RAM-bound (B > L3): re-streaming B from RAM `rows` times dominates, so the
    // 4x fewer B passes from 4-row blocking is where the structural win appears.
    bench_size(1536, 1536, 1536, 3); // B = 18 MB
    bench_size(2048, 2048, 2048, 2); // B = 32 MB
}
