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

    // Complex128 matmul head-to-head context (vs JAX zgemm, which IS fast unlike int).
    for (nsz, iters) in [(512usize, 8usize), (1024, 4)] {
        let a: Vec<(f64, f64)> = (0..nsz * nsz)
            .map(|i| ((i % 13) as f64 - 6.0, (i % 7) as f64 - 3.0))
            .collect();
        let b: Vec<(f64, f64)> = (0..nsz * nsz)
            .map(|i| ((i % 11) as f64 - 5.0, (i % 5) as f64 - 2.0))
            .collect();
        // Single-row reference (the PRE-blocking kernel), single-threaded over the whole
        // matrix — to A/B the 4-row register blocking same-binary and prove bit-identity.
        let single_row_ref = |a: &[(f64, f64)], b: &[(f64, f64)]| -> Vec<(f64, f64)> {
            let mut c = vec![(0.0f64, 0.0f64); nsz * nsz];
            for (ri, c_row) in c.chunks_mut(nsz).enumerate() {
                let a_off = ri * nsz;
                for l in 0..nsz {
                    let (ar, ai) = a[a_off + l];
                    let b_row = &b[l * nsz..l * nsz + nsz];
                    for (cc, &(br, bi)) in c_row.iter_mut().zip(b_row) {
                        cc.0 += ar * br - ai * bi;
                        cc.1 += ar * bi + ai * br;
                    }
                }
            }
            c
        };
        // 1-thread 4-row-BLOCKED reference (the production inner kernel's shape) — isolates
        // the register-blocking win vs single_row_ref, same-binary (both 1-thread).
        let blocked_ref = |a: &[(f64, f64)], b: &[(f64, f64)]| -> Vec<(f64, f64)> {
            let mut c = vec![(0.0f64, 0.0f64); nsz * nsz];
            let full = nsz - nsz % 4;
            let (blocked, tail) = c.split_at_mut(full * nsz);
            for (g, four) in blocked.chunks_mut(4 * nsz).enumerate() {
                let (c0, rest) = four.split_at_mut(nsz);
                let (c1, rest) = rest.split_at_mut(nsz);
                let (c2, c3) = rest.split_at_mut(nsz);
                let base = g * 4 * nsz;
                for l in 0..nsz {
                    let (a0r, a0i) = a[base + l];
                    let (a1r, a1i) = a[base + nsz + l];
                    let (a2r, a2i) = a[base + 2 * nsz + l];
                    let (a3r, a3i) = a[base + 3 * nsz + l];
                    let b_row = &b[l * nsz..l * nsz + nsz];
                    for ((((e0, e1), e2), e3), &(br, bi)) in c0
                        .iter_mut()
                        .zip(c1.iter_mut())
                        .zip(c2.iter_mut())
                        .zip(c3.iter_mut())
                        .zip(b_row)
                    {
                        e0.0 += a0r * br - a0i * bi;
                        e0.1 += a0r * bi + a0i * br;
                        e1.0 += a1r * br - a1i * bi;
                        e1.1 += a1r * bi + a1i * br;
                        e2.0 += a2r * br - a2i * bi;
                        e2.1 += a2r * bi + a2i * br;
                        e3.0 += a3r * br - a3i * bi;
                        e3.1 += a3r * bi + a3i * br;
                    }
                }
            }
            for (ri, c_row) in tail.chunks_mut(nsz).enumerate() {
                let a_off = (full + ri) * nsz;
                for l in 0..nsz {
                    let (ar, ai) = a[a_off + l];
                    let b_row = &b[l * nsz..l * nsz + nsz];
                    for (cc, &(br, bi)) in c_row.iter_mut().zip(b_row) {
                        cc.0 += ar * br - ai * bi;
                        cc.1 += ar * bi + ai * br;
                    }
                }
            }
            c
        };

        let prod = fj_lax::tensor_contraction::rank2_complex_matmul(&a, nsz, nsz, &b, nsz);
        let sref = single_row_ref(&a, &b);
        assert_eq!(
            prod, sref,
            "threaded blocked complex matmul != single-row ref"
        );
        assert_eq!(blocked_ref(&a, &b), sref, "1thr blocked != single-row ref");
        let t = Instant::now();
        for _ in 0..iters {
            let r = fj_lax::tensor_contraction::rank2_complex_matmul(
                black_box(&a),
                nsz,
                nsz,
                black_box(&b),
                nsz,
            );
            black_box(&r);
        }
        let prod_ms = t.elapsed().as_nanos() as f64 / iters as f64 / 1e6;
        let _ = single_row_ref(&a, &b);
        let t2 = Instant::now();
        black_box(single_row_ref(black_box(&a), black_box(&b)));
        let single_ms = t2.elapsed().as_nanos() as f64 / 1e6;
        let _ = blocked_ref(&a, &b);
        let t3 = Instant::now();
        black_box(blocked_ref(black_box(&a), black_box(&b)));
        let blocked_ms = t3.elapsed().as_nanos() as f64 / 1e6;
        println!(
            "C128_MATMUL {nsz}^3 prod_threaded={prod_ms:.3}ms | 1thr single_row={single_ms:.3}ms blocked={blocked_ms:.3}ms block_speedup={:.2}x",
            single_ms / blocked_ms,
        );
    }

    // BATCHED complex128 matmul: 1-thread naive-single-row vs 1-thread batch-aware-4row-blocked
    // (same-binary, isolates the per-batch register blocking now shipped in batched_complex_row_block).
    for (batch, msz) in [(32usize, 128usize)] {
        let (k, n) = (msz, msz);
        let a: Vec<(f64, f64)> = (0..batch * msz * k)
            .map(|i| ((i % 13) as f64 - 6.0, (i % 7) as f64 - 3.0))
            .collect();
        let b: Vec<(f64, f64)> = (0..batch * k * n)
            .map(|i| ((i % 11) as f64 - 5.0, (i % 5) as f64 - 2.0))
            .collect();
        let naive = || -> Vec<(f64, f64)> {
            let mut c = vec![(0.0f64, 0.0f64); batch * msz * n];
            for bt in 0..batch {
                let (ao, bo, co) = (bt * msz * k, bt * k * n, bt * msz * n);
                for row in 0..msz {
                    for l in 0..k {
                        let (ar, ai) = a[ao + row * k + l];
                        for j in 0..n {
                            let (br, bi) = b[bo + l * n + j];
                            let cc = &mut c[co + row * n + j];
                            cc.0 += ar * br - ai * bi;
                            cc.1 += ar * bi + ai * br;
                        }
                    }
                }
            }
            c
        };
        let prod =
            fj_lax::tensor_contraction::batched_rank2_complex_matmul(&a, batch, msz, k, &b, n);
        assert_eq!(prod, naive(), "batched blocked complex != naive");
        let _ = naive();
        let t = Instant::now();
        black_box(naive());
        let naive_ms = t.elapsed().as_nanos() as f64 / 1e6;
        let _ = fj_lax::tensor_contraction::batched_rank2_complex_matmul(&a, batch, msz, k, &b, n);
        let t2 = Instant::now();
        black_box(fj_lax::tensor_contraction::batched_rank2_complex_matmul(
            black_box(&a),
            batch,
            msz,
            k,
            black_box(&b),
            n,
        ));
        let prod_ms = t2.elapsed().as_nanos() as f64 / 1e6;
        println!(
            "C128_BATCHED_MATMUL b={batch} {msz}^3 naive_1thr={naive_ms:.3}ms prod_threaded_blocked={prod_ms:.3}ms speedup={:.2}x",
            naive_ms / prod_ms,
        );
    }
}
