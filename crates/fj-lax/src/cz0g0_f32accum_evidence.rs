//! f32-accumulate vs f64-accumulate GEMM — EVIDENCE for `frankenjax-cz0g0`'s
//! "GEMM f32-accum" sub-lever, complementing the FMA evidence in
//! [`crate::cz0g0_fma_evidence`] and the SIMD-poly exp evidence in
//! [`crate::simd_exp`].
//!
//! ## The decision this quantifies
//!
//! fj's original production f32 matmul (`tensor_contraction::batched_matmul_2d_f32_in`,
//! used by `general_real_tensordot`) loaded f32, **accumulated in f64**, and rounded to
//! f32 at the end — bit-identical to promoting f32→f64 and running the f64 GEMM, which
//! was fj's self-golden. JAX/XLA's f32 matmul instead **accumulates in f32** (the element
//! type). So the old fj path was MORE accurate than JAX here, at a cost:
//!
//! 1. **Parity:** fj's f32 matmul does NOT bit-match (nor tolerance-trivially match) XLA's
//!    f32 matmul — XLA's f32 accumulation has ~√K·ε error that fj's f64 accumulation does
//!    not. Matching XLA means accepting that lower precision (the cz0g0 policy question).
//! 2. **Speed:** f64 accumulation uses `Simd<f64,8>` (8 lanes / 512-bit); f32 accumulation
//!    uses `Simd<f32,16>` (16 lanes) → ~2× the MAC throughput, and the inputs already live
//!    as f32 so no widening is needed in the inner loop.
//!
//! This module provides two otherwise-identical register-tiled (MR×NR) kernels — one
//! f64-accumulate (mirrors production arithmetic), one f32-accumulate (mirrors XLA) — so
//! the speed win and the accuracy/parity delta can be measured head-to-head. The production
//! path is now the f32-accumulate SIMD kernel; this module keeps the old-vs-new comparison
//! and precision-delta proof local to `frankenjax-cz0g0`.

use std::simd::Simd;

const MR: usize = 4;
const NR64: usize = 8; // 512-bit / f64
const NR32: usize = 16; // 512-bit / f32
type F64s = Simd<f64, NR64>;
type F32s = Simd<f32, NR32>;

/// Register-tiled `[m,k]@[k,n]` GEMM: f32 in, **f64 accumulate**, f32 out — mirrors fj's
/// previous `batched_matmul_2d_f32_in` arithmetic (load f32 → widen f64 → `c += a*b` in
/// f64 → round to f32). `n` and `m` are assumed multiples of `NR64`/`MR` (the evidence
/// benches use such shapes); a scalar tail is omitted for clarity.
pub fn matmul_f32_f64_accumulate(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    let full_rows = m / MR * MR;
    let full_cols = n / NR64 * NR64;
    let mut j = 0;
    while j < full_cols {
        let mut i = 0;
        while i < full_rows {
            let mut acc = [F64s::splat(0.0); MR];
            for l in 0..k {
                // Widen the f32 B row to f64 (lossless), as the production f64-accum path does.
                let mut bw = [0.0_f64; NR64];
                for (d, slot) in bw.iter_mut().enumerate() {
                    *slot = f64::from(b[l * n + j + d]);
                }
                let bv = F64s::from_array(bw);
                for (t, accum) in acc.iter_mut().enumerate() {
                    *accum += F64s::splat(f64::from(a[(i + t) * k + l])) * bv;
                }
            }
            for (t, accum) in acc.iter().enumerate() {
                let arr = accum.as_array();
                for d in 0..NR64 {
                    c[(i + t) * n + j + d] = arr[d] as f32; // round f64 -> f32 at output
                }
            }
            i += MR;
        }
        j += NR64;
    }
    c
}

/// Register-tiled `[m,k]@[k,n]` GEMM: f32 in, **f32 accumulate**, f32 out — mirrors XLA's
/// f32 matmul (everything stays f32). Uses `Simd<f32,16>` (2× the lanes of the f64-accum
/// kernel). Same tiling/order as [`matmul_f32_f64_accumulate`]; only the accumulator
/// precision differs. `n`/`m` assumed multiples of `NR32`/`MR`.
pub fn matmul_f32_f32_accumulate(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    let full_rows = m / MR * MR;
    let full_cols = n / NR32 * NR32;
    let mut j = 0;
    while j < full_cols {
        let mut i = 0;
        while i < full_rows {
            let mut acc = [F32s::splat(0.0); MR];
            for l in 0..k {
                let bv = F32s::from_slice(&b[l * n + j..l * n + j + NR32]);
                for (t, accum) in acc.iter_mut().enumerate() {
                    *accum += F32s::splat(a[(i + t) * k + l]) * bv;
                }
            }
            for (t, accum) in acc.iter().enumerate() {
                c[(i + t) * n + j..(i + t) * n + j + NR32].copy_from_slice(accum.as_array());
            }
            i += MR;
        }
        j += NR32;
    }
    c
}

/// Scalar f64-accumulate reference (the "true" value at f64 precision) for a single output
/// cell — used to measure how far each kernel's f32 output sits from the high-precision
/// result. Ascending-`l` fold, matching the production accumulation order.
#[cfg(test)]
fn cell_f64_reference(a: &[f32], k: usize, b: &[f32], n: usize, i: usize, jcol: usize) -> f64 {
    let mut s = 0.0_f64;
    for l in 0..k {
        s += f64::from(a[i * k + l]) * f64::from(b[l * n + jcol]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(len: usize, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * salt).sin() * 1.3 - 0.2)
            .collect()
    }

    #[test]
    fn f64accum_kernel_matches_old_production_semantics() {
        // The f64-accumulate evidence kernel must reproduce fj's old production f32 matmul
        // (f32 in, f64 accumulate, round to f32) bit-for-bit: same ascending-`l` fold,
        // f64 accumulator, single round at output. Verify against a direct scalar
        // f64-accumulate-then-round-to-f32 reference (the former production semantics).
        let (m, k, n) = (8usize, 24usize, 16usize);
        let a = mk(m * k, 0.011);
        let b = mk(k * n, 0.017);
        let got = matmul_f32_f64_accumulate(&a, m, k, &b, n);
        for i in 0..m {
            for jcol in 0..n {
                let want = cell_f64_reference(&a, k, &b, n, i, jcol) as f32;
                assert_eq!(
                    got[i * n + jcol].to_bits(),
                    want.to_bits(),
                    "f64-accum kernel must match old f64-accumulate-then-round at ({i},{jcol})"
                );
            }
        }
    }

    #[test]
    fn f32accum_diverges_from_f64accum_but_both_bounded() {
        // Quantify the cz0g0 precision delta: f32-accumulate (XLA semantics) vs
        // f64-accumulate (fj production). They are NOT equal (that is the whole point —
        // switching to f32-accum changes fj's output bits and breaks the self-golden),
        // but both stay within f32-matmul tolerance of the true f64 value. The f64-accum
        // kernel is at least as close to truth as the f32-accum kernel for every cell.
        let (m, k, n) = (16usize, 256usize, 16usize); // K=256 so f32-accum error accrues
        let a = mk(m * k, 0.0007);
        let b = mk(k * n, 0.0011);
        let c32 = matmul_f32_f32_accumulate(&a, m, k, &b, n);
        let c64 = matmul_f32_f64_accumulate(&a, m, k, &b, n);

        let mut any_differ = false;
        let mut max_rel_32_vs_64 = 0.0_f64;
        for i in 0..m {
            for jcol in 0..n {
                let idx = i * n + jcol;
                let truth = cell_f64_reference(&a, k, &b, n, i, jcol);
                let e32 = (f64::from(c32[idx]) - truth).abs();
                let e64 = (f64::from(c64[idx]) - truth).abs();
                if c32[idx].to_bits() != c64[idx].to_bits() {
                    any_differ = true;
                }
                if truth.abs() > 1e-6 {
                    max_rel_32_vs_64 = max_rel_32_vs_64
                        .max((f64::from(c32[idx]) - f64::from(c64[idx])).abs() / truth.abs());
                }
                // f64-accumulate is never farther from truth than f32-accumulate (modulo
                // the final identical round-to-f32) — that is the accuracy fj would give up.
                assert!(
                    e64 <= e32 + 8.0 * f64::from(f32::EPSILON) * truth.abs().max(1.0),
                    "f64-accum should be at least as accurate as f32-accum at ({i},{jcol}): e64={e64:e} e32={e32:e}"
                );
            }
        }
        assert!(
            any_differ,
            "K=256 f32-accum vs f64-accum should differ in at least one cell (else no decision to make)"
        );
        // Within f32 matmul tolerance (~K·ε): a loose but real bound documenting the cost.
        assert!(
            max_rel_32_vs_64 < 1e-3,
            "f32-accum vs f64-accum relative delta {max_rel_32_vs_64:e} unexpectedly large"
        );
    }

    #[test]
    #[ignore = "benchmark: run with RUSTFLAGS=\"-C target-cpu=native\" --ignored --nocapture"]
    fn bench_f32accum_vs_f64accum_matmul() {
        for &n in &[256usize, 512usize] {
            let a = mk(n * n, 0.013);
            let b = mk(n * n, 0.019);
            let iters = if n <= 256 { 40 } else { 12 };

            let _ = matmul_f32_f64_accumulate(&a, n, n, &b, n);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(matmul_f32_f64_accumulate(&a, n, n, &b, n));
            }
            let f64acc = t0.elapsed().as_nanos() as f64 / iters as f64;

            let _ = matmul_f32_f32_accumulate(&a, n, n, &b, n);
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(matmul_f32_f32_accumulate(&a, n, n, &b, n));
            }
            let f32acc = t1.elapsed().as_nanos() as f64 / iters as f64;

            let gflop = 2.0 * (n * n * n) as f64;
            println!(
                "BENCH matmul {n}x{n} f32-in: f64_accum={:.3}ms ({:.1} GFLOP/s) f32_accum={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x",
                f64acc / 1e6,
                gflop / f64acc,
                f32acc / 1e6,
                gflop / f32acc,
                f64acc / f32acc
            );
        }
    }
}
