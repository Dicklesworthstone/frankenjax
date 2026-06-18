//! FMA-vs-non-FMA matmul microkernel — EVIDENCE for `frankenjax-cz0g0`, complementing
//! the SIMD-poly exp kernel in [`crate::simd_exp`].
//!
//! The production GEMM (`tensor_contraction::matmul_2d_row_block`) deliberately accumulates
//! with SEPARATE `*` and `+` (`c += a*b`), NOT `mul_add` — because `fma(a,b,c)` (one
//! rounding) ≠ `a*b + c` (two roundings), so an FMA kernel produces DIFFERENT bits and
//! breaks the bit-exact-to-ijk self-goldens. That is the SAME bit-exact-to-self relaxation
//! cz0g0 is blocked on. This module measures what that bit-exactness COSTS: two otherwise
//! identical register-tiled (MR×NR, `Simd<f64,8>`) kernels, one using `mul_add` (fuses to
//! a single FMA instruction when `+fma`/`target-cpu=native` is enabled) and one using
//! `*`+`+`, benchmarked head-to-head. The FMA result stays within ~1e-12 relative of the
//! non-FMA result (`fma_kernel_within_tolerance`) — i.e. WELL within JAX/XLA tolerance
//! (XLA itself emits FMA), so enabling FMA only breaks the SELF-golden, not JAX parity.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native"` to see the FMA win; the default portable
//! build has no `+fma` so `mul_add` either libcalls `fma()` or de-fuses (see
//! [`crate::simd_exp`] for the same effect on exp).

use std::simd::{Simd, StdFloat};

const NR: usize = 8;
const MR: usize = 4;
type F64s = Simd<f64, NR>;

#[track_caller]
fn assert_tiled_matmul_inputs(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> usize {
    assert_eq!(
        a.len(),
        m.checked_mul(k).expect("FMA evidence lhs shape overflow"),
        "FMA evidence lhs length must equal m*k"
    );
    assert_eq!(
        b.len(),
        k.checked_mul(n).expect("FMA evidence rhs shape overflow"),
        "FMA evidence rhs length must equal k*n"
    );
    assert_eq!(
        m % MR,
        0,
        "FMA evidence kernels require m to be a multiple of MR"
    );
    assert_eq!(
        n % NR,
        0,
        "FMA evidence kernels require n to be a multiple of NR"
    );
    m.checked_mul(n)
        .expect("FMA evidence output shape overflow")
}

/// Register-tiled `[m,k]@[k,n]` f64 GEMM, NON-FMA accumulation (`c += a*b`) — mirrors the
/// production kernel's arithmetic exactly (bit-identical to `matmul_2d_row_block`).
pub fn matmul_muladd_free(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let output_len = assert_tiled_matmul_inputs(a, m, k, b, n);
    let mut c = vec![0.0_f64; output_len];
    let full_rows = m / MR * MR;
    let full_cols = n / NR * NR;
    let mut j = 0;
    while j < full_cols {
        let mut i = 0;
        while i < full_rows {
            let mut acc = [F64s::splat(0.0); MR];
            for l in 0..k {
                let bv = F64s::from_slice(&b[l * n + j..l * n + j + NR]);
                for (t, accum) in acc.iter_mut().enumerate() {
                    *accum += F64s::splat(a[(i + t) * k + l]) * bv; // separate mul + add
                }
            }
            for (t, accum) in acc.iter().enumerate() {
                c[(i + t) * n + j..(i + t) * n + j + NR].copy_from_slice(accum.as_array());
            }
            i += MR;
        }
        j += NR;
    }
    c
}

/// Identical tiling, FMA accumulation (`c = a.mul_add(b, c)`) — fuses to a single FMA
/// instruction under `+fma`. Different rounding ⇒ different bits ⇒ would break the
/// bit-exact-to-ijk self-golden (but stays within JAX tolerance).
pub fn matmul_fma(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let output_len = assert_tiled_matmul_inputs(a, m, k, b, n);
    let mut c = vec![0.0_f64; output_len];
    let full_rows = m / MR * MR;
    let full_cols = n / NR * NR;
    let mut j = 0;
    while j < full_cols {
        let mut i = 0;
        while i < full_rows {
            let mut acc = [F64s::splat(0.0); MR];
            for l in 0..k {
                let bv = F64s::from_slice(&b[l * n + j..l * n + j + NR]);
                for (t, accum) in acc.iter_mut().enumerate() {
                    *accum = F64s::splat(a[(i + t) * k + l]).mul_add(bv, *accum); // fused
                }
            }
            for (t, accum) in acc.iter().enumerate() {
                c[(i + t) * n + j..(i + t) * n + j + NR].copy_from_slice(accum.as_array());
            }
            i += MR;
        }
        j += NR;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(len: usize, salt: f64) -> Vec<f64> {
        (0..len)
            .map(|i| (i as f64 * salt).sin() * 1.3 - 0.2)
            .collect()
    }

    #[test]
    fn fma_kernel_within_tolerance() {
        // The FMA and non-FMA kernels must agree to JAX/XLA tolerance — they are NOT
        // bit-identical (that is the whole point: FMA changes the rounding and breaks the
        // self-golden) but stay well within ~1e-12 relative, i.e. within XLA tolerance.
        let (m, k, n) = (64usize, 96usize, 64usize);
        let a = mk(m * k, 0.013);
        let b = mk(k * n, 0.019);
        let cf = matmul_fma(&a, m, k, &b, n);
        let cm = matmul_muladd_free(&a, m, k, &b, n);
        let mut max_rel = 0.0_f64;
        for (&f, &mm) in cf.iter().zip(&cm) {
            if mm.abs() > 1e-9 {
                max_rel = max_rel.max(((f - mm) / mm).abs());
            }
        }
        assert!(
            max_rel < 1e-12,
            "FMA vs non-FMA relative delta {max_rel:e} exceeds 1e-12 (should be within XLA tol)"
        );
    }

    #[test]
    fn fma_evidence_kernels_reject_non_tiled_shapes() {
        let (m, k, n) = (5usize, 3usize, 9usize);
        let a = mk(m * k, 0.013);
        let b = mk(k * n, 0.019);
        assert!(
            std::panic::catch_unwind(|| matmul_muladd_free(&a, m, k, &b, n)).is_err(),
            "non-FMA evidence kernel must reject shapes that would leave zeroed tail cells"
        );
        assert!(
            std::panic::catch_unwind(|| matmul_fma(&a, m, k, &b, n)).is_err(),
            "FMA evidence kernel must reject shapes that would leave zeroed tail cells"
        );
    }

    #[test]
    fn fma_evidence_kernels_reject_output_shape_overflow() {
        let (m, k, n) = (usize::MAX - 3, 0usize, NR);
        let a = Vec::new();
        let b = Vec::new();
        assert!(
            std::panic::catch_unwind(|| matmul_muladd_free(&a, m, k, &b, n)).is_err(),
            "non-FMA evidence kernel must reject m*n overflow before allocation"
        );
        assert!(
            std::panic::catch_unwind(|| matmul_fma(&a, m, k, &b, n)).is_err(),
            "FMA evidence kernel must reject m*n overflow before allocation"
        );
    }

    #[test]
    #[ignore = "benchmark: run with RUSTFLAGS=\"-C target-cpu=native\" --ignored --nocapture"]
    fn bench_fma_vs_nonfma_matmul() {
        for &n in &[256usize, 512usize] {
            let a = mk(n * n, 0.013);
            let b = mk(n * n, 0.019);
            let iters = if n <= 256 { 40 } else { 12 };

            let _ = matmul_muladd_free(&a, n, n, &b, n);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(matmul_muladd_free(&a, n, n, &b, n));
            }
            let nofma = t0.elapsed().as_nanos() as f64 / iters as f64;

            let _ = matmul_fma(&a, n, n, &b, n);
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(matmul_fma(&a, n, n, &b, n));
            }
            let fma = t1.elapsed().as_nanos() as f64 / iters as f64;

            let gflop = 2.0 * (n * n * n) as f64;
            println!(
                "BENCH matmul {n}x{n} f64: non_fma={:.3}ms ({:.1} GFLOP/s) fma={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x",
                nofma / 1e6,
                gflop / nofma,
                fma / 1e6,
                gflop / fma,
                nofma / fma
            );
        }
    }
}
