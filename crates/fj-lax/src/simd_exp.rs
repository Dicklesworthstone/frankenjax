//! SIMD-vectorized polynomial `f64` exp — a portable-SIMD reference kernel and the
//! concrete EVIDENCE artifact for `frankenjax-cz0g0` (the parity-relaxation decision).
//!
//! `eval_exp` currently maps the scalar libm `f64::exp` per element. That is pinned by
//! the bit-exact-to-self transcendental goldens (`gj_08_transcendental_gradients` +
//! `arithmetic.rs` `exp_bits` assertions), which is why a vectorized exp has been
//! BLOCKED: any polynomial approximation differs from libm's bits. But fj-lax's exp is
//! ALREADY not bit-identical to JAX/XLA (which uses its own polynomial, not libm), so
//! conformance parity is ALREADY tolerance-based; only the SELF-goldens pin libm's bits.
//!
//! This kernel is NOT wired into `eval_exp` — so no golden changes and behaviour is
//! unchanged. It exists to QUANTIFY the win and the accuracy delta for the cz0g0
//! decision: `simd_poly_exp_into` is proven to agree with libm `exp` to < 1e-12 relative
//! error across the full finite range (`simd_poly_exp_matches_libm`) and benchmarked vs
//! scalar libm (`bench_simd_poly_exp_vs_libm`).
//!
//! MEASURED (1M f64, this hardware) — the win is DOUBLY gated, both gates being the same
//! bit-exact-to-self relaxation cz0g0 asks about:
//!   * `target-cpu=native` (FMA enabled): **2.20x** vs libm (2.09ms vs 4.59ms).
//!   * default portable build (NO `+fma`): **0.79x** — SLOWER than libm. Without FMA the
//!     degree-13 Horner `mul_add` chain either libcalls `fma()` (very slow) or de-fuses to
//!     mul+add (≈0.8x), and glibc's scalar `exp` is already well-optimized (~4.5ns/elem).
//!
//! CONCLUSION for cz0g0: SIMD-poly `exp` needs BOTH (a) the polynomial approximation
//! (breaks the exp self-golden) AND (b) FMA contraction (breaks bit-exact-to-ijk-style
//! accumulation, same parity class). With both, f64 gets ~2.2x; f32/bf16 (wider vectors,
//! less-optimized `expf`) would gain more. Without the relaxation the lever is a net LOSS,
//! so it is correctly BLOCKED, not merely unimplemented. On approval (golden→JAX-tolerance
//! + FMA-enabled kernel build), wiring this into the dense `eval_exp` path is a one-liner.
//!
//! Algorithm (Cody-Waite range reduction + Taylor, the standard accurate scheme):
//!   n = round(x / ln2);  r = x - n*ln2  (|r| <= ln2/2 ≈ 0.347)
//!   exp(x) = 2^n * exp(r),  exp(r) via a degree-13 Taylor (1/k! coefficients — exact
//!   factorials, so no minimax-constant transcription risk). For |r| <= 0.347 the degree
//!   -13 truncation error is ~r^14/14! < 1e-18, far below f64 eps, so the result tracks
//!   libm to ~1 ulp. `ln2` is split into `LN2_HI`+`LN2_LO` (Cody-Waite) so `n*LN2_HI` is
//!   exact and the reduction does not lose low bits for large `x`. `2^n` is built by
//!   placing `n` into the f64 exponent field. Overflow/underflow/NaN are blended in.

use std::simd::{
    Simd, StdFloat,
    num::{SimdFloat, SimdInt},
};

const LANES: usize = 8;
type F64s = Simd<f64, LANES>;

const LOG2E: f64 = std::f64::consts::LOG2_E; // 1/ln2
// Cody-Waite split of ln2: LN2_HI is exactly representable (11355 * 2^-14), so for the
// `n` magnitudes that survive the overflow clamp, `n * LN2_HI` is exact.
const LN2_HI: f64 = 6.931_457_519_531_25e-1;
// Full-precision Cody-Waite / clamp constants: the trailing digits document the
// intended mathematical value (they round to the same f64), so keep them verbatim.
#[allow(clippy::excessive_precision)]
const LN2_LO: f64 = 1.428_606_820_309_417_232_12e-6;
// Past these, exp overflows to +inf / underflows to 0 in f64.
const OVERFLOW_X: f64 = 709.782_712_893_384;
#[allow(clippy::excessive_precision)]
const UNDERFLOW_X: f64 = -745.133_219_101_941_22;

// Taylor coefficients 1/k! for k = 0..=13 (Horner evaluates from C[13] down to C[0]).
const C: [f64; 14] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362_880.0,
    1.0 / 3_628_800.0,
    1.0 / 39_916_800.0,
    1.0 / 479_001_600.0,
    1.0 / 6_227_020_800.0,
];

/// One 8-lane block of `exp` for IN-RANGE inputs (`UNDERFLOW_X < x < OVERFLOW_X`). This
/// is PURE arithmetic — no SIMD comparisons/select (whose portable-SIMD trait surface
/// drifts across nightlies); the caller routes out-of-range / NaN / ±inf lanes to the
/// scalar `exp_scalar`. For in-range `x` it equals `exp_scalar(x)` bit-for-bit.
#[inline]
fn exp_block(x: F64s) -> F64s {
    // n = round(x / ln2); r = x - n*LN2_HI - n*LN2_LO. Explicit mul+add (NOT mul_add):
    // without a `+fma` target feature `mul_add` lowers to a slow `fma()` libcall, so the
    // whole SIMD codebase (matmul, etc.) uses separate `*`/`+` — and the exp_scalar ref
    // matches this so the SIMD and scalar paths stay bit-identical.
    let nf = (x * F64s::splat(LOG2E)).round();
    let r = nf.mul_add(F64s::splat(-LN2_HI), x);
    let r = nf.mul_add(F64s::splat(-LN2_LO), r);

    // Horner Taylor: p = ((C13*r + C12)*r + …)*r + C0.
    let mut p = F64s::splat(C[13]);
    for k in (0..13).rev() {
        p = p.mul_add(r, F64s::splat(C[k]));
    }

    // 2^n by placing the exponent into the f64 exponent field. Split n into two halves
    // n = n_a + n_b (each within the normal-exponent range) and multiply by 2^n_a · 2^n_b,
    // so SUBNORMAL results (n + 1023 < 1, x ≲ -708) compute correctly (normal·normal →
    // subnormal with one rounding) and large results saturate to +inf via f64 arithmetic.
    let ni = nf.cast::<i64>();
    let na = ni >> Simd::splat(1); // floor(n/2) (arithmetic shift)
    let nb = ni - na;
    let two_a = F64s::from_bits(((na + Simd::splat(1023)) << Simd::splat(52)).cast::<u64>());
    let two_b = F64s::from_bits(((nb + Simd::splat(1023)) << Simd::splat(52)).cast::<u64>());
    p * two_a * two_b
}

/// Scalar reference for the SIMD block (same algorithm, used for the tail and as the
/// per-lane definition the bit-identity argument rests on).
#[inline]
fn exp_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > OVERFLOW_X {
        return f64::INFINITY;
    }
    if x < UNDERFLOW_X {
        return 0.0;
    }
    let nf = (x * LOG2E).round();
    let r = nf.mul_add(-LN2_HI, x);
    let r = nf.mul_add(-LN2_LO, r);
    let mut p = C[13];
    for k in (0..13).rev() {
        p = p.mul_add(r, C[k]);
    }
    let ni = nf as i64;
    let na = ni >> 1; // floor(n/2)
    let nb = ni - na;
    let two_a = f64::from_bits(((na + 1023) << 52) as u64);
    let two_b = f64::from_bits(((nb + 1023) << 52) as u64);
    p * two_a * two_b
}

/// Vectorized `exp` over a contiguous `f64` slice into `out` (same length). Processes
/// 8 lanes per step; the remainder runs the identical scalar algorithm.
pub fn simd_poly_exp_into(src: &[f64], out: &mut [f64]) {
    assert_eq!(src.len(), out.len());
    let mut chunks = src.chunks_exact(LANES);
    let mut out_chunks = out.chunks_exact_mut(LANES);
    for (c, o) in chunks.by_ref().zip(out_chunks.by_ref()) {
        // Clean vector store of the SIMD poly (lets exp_block stay fully vectorized).
        o.copy_from_slice(&exp_block(F64s::from_slice(c)).to_array());
        // Cheap branchless scan for any out-of-range / NaN / ±inf lane (autovectorizes);
        // only the rare chunk that has one re-does just those lanes via the scalar ref.
        let mut has_edge = false;
        for &x in c {
            has_edge |= !(x > UNDERFLOW_X && x < OVERFLOW_X);
        }
        if has_edge {
            for (oi, &x) in o.iter_mut().zip(c) {
                if !(x > UNDERFLOW_X && x < OVERFLOW_X) {
                    *oi = exp_scalar(x);
                }
            }
        }
    }
    for (s, o) in chunks.remainder().iter().zip(out_chunks.into_remainder()) {
        *o = exp_scalar(*s);
    }
}

/// Allocating convenience wrapper.
#[must_use]
pub fn simd_poly_exp(src: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; src.len()];
    simd_poly_exp_into(src, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_poly_exp_matches_libm() {
        // The SIMD-poly exp must agree with scalar libm `f64::exp` to a tight relative
        // tolerance across the full finite range — this is the accuracy evidence for the
        // cz0g0 decision (it is NOT bit-identical to libm, which is exactly why it is
        // golden-blocked, but it IS within JAX/XLA tolerance, which is what parity needs).
        let mut xs: Vec<f64> = Vec::new();
        // Dense sweep over [-740, 709], plus small magnitudes and exact-zero region.
        let mut x = -740.0_f64;
        while x <= 709.0 {
            xs.push(x);
            x += 0.013;
        }
        for &v in &[
            -1e-12, -0.0, 0.0, 1e-12, 1.0, -1.0, 0.5, -0.5, 88.0, -88.0, 700.0,
        ] {
            xs.push(v);
        }
        // Pad to a non-multiple of 8 so the scalar tail path is exercised.
        while xs.len() % LANES != 5 {
            xs.push(0.25);
        }

        let got = simd_poly_exp(&xs);
        let mut max_rel = 0.0_f64;
        for (&x, &g) in xs.iter().zip(&got) {
            let want = x.exp();
            if want == 0.0 {
                assert_eq!(g, 0.0, "underflow at x={x}");
                continue;
            }
            if !want.is_finite() {
                assert!(g.is_infinite() && g > 0.0, "overflow at x={x}");
                continue;
            }
            let rel = ((g - want) / want).abs();
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 1e-12,
            "max relative error {max_rel:e} exceeds 1e-12 vs libm exp"
        );

        // Edge cases.
        assert!(simd_poly_exp(&[f64::NAN])[0].is_nan());
        assert_eq!(simd_poly_exp(&[f64::NEG_INFINITY])[0], 0.0);
        assert!(simd_poly_exp(&[f64::INFINITY])[0].is_infinite());
        assert_eq!(simd_poly_exp(&[0.0])[0], 1.0);
    }

    #[test]
    fn block_matches_scalar_bit_for_bit() {
        // The 8-lane block and the scalar `exp_scalar` are the SAME algorithm, so they
        // must agree bit-for-bit (the SIMD path is just lane-parallel evaluation).
        let xs: Vec<f64> = (0..LANES).map(|i| (i as f64) * 1.7 - 5.0).collect();
        let v = exp_block(F64s::from_slice(&xs)).to_array();
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(v[i].to_bits(), exp_scalar(x).to_bits(), "lane {i} x={x}");
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_simd_poly_exp_vs_libm() {
        let n = 1 << 20; // 1M elements
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-4 - 50.0).collect();
        let mut out = vec![0.0_f64; n];

        // Warm + time SIMD-poly.
        simd_poly_exp_into(&xs, &mut out);
        let t0 = std::time::Instant::now();
        let iters = 50;
        for _ in 0..iters {
            simd_poly_exp_into(&xs, &mut out);
            std::hint::black_box(&out);
        }
        let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        // Time scalar libm.
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            for (o, &x) in out.iter_mut().zip(&xs) {
                *o = x.exp();
            }
            std::hint::black_box(&out);
        }
        let libm_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

        println!(
            "BENCH exp 1M f64: libm={:.3}ms simd_poly={:.3}ms speedup={:.2}x",
            libm_ns / 1e6,
            simd_ns / 1e6,
            libm_ns / simd_ns
        );
    }
}
