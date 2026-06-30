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
    let r = x - nf * F64s::splat(LN2_HI);
    let r = r - nf * F64s::splat(LN2_LO);

    // Horner Taylor: p = ((C13*r + C12)*r + …)*r + C0.
    let mut p = F64s::splat(C[13]);
    for k in (0..13).rev() {
        p = p * r + F64s::splat(C[k]);
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
    let r = x - nf * LN2_HI;
    let r = r - nf * LN2_LO;
    let mut p = C[13];
    for k in (0..13).rev() {
        p = p * r + C[k];
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

/// 8-wide f32 `exp` (Cephes `expf`: Cody-Waite ln2 range reduction + degree-6 minimax poly,
/// 2^n by exponent-field injection). Explicit `*`/`+` (NO `mul_add`) so it never libcalls `fma`.
///
/// KEY RESULT: unlike the f64 [`simd_poly_exp`] (degree-13, 4-wide → 0.79x without `+fma`), this
/// f32 8-wide kernel is **~2.27x FASTER than scalar libm `expf` even WITHOUT `+fma`** (measured
/// single-thread 4M, `bench_exp_block_f32_vs_libm`) — the lane count (8 vs 4 on AVX2) and the far
/// shorter poly (degree 6 vs 13) flip the f64 finding. Accuracy ~1 ulp (≤1.2e-7 rel over [-30,30],
/// `exp_block_f32_accuracy`), i.e. within tolerance parity but NOT bit-identical to libm. f32 is
/// JAX's default float, so this UNBLOCKS SIMD f32 transcendentals (exp/tanh/sigmoid/softplus and,
/// with a companion SIMD log, pow/zeta) for any TOLERANCE-parity consumer — the residual gate for
/// the `Exp` PRIMITIVE itself is its frozen golden (RNG reproducibility), a separate decision from
/// `+fma`. Not yet wired into an eval path.
#[inline]
pub fn exp_block_f32(x: std::simd::Simd<f32, 8>) -> std::simd::Simd<f32, 8> {
    use std::simd::Simd;
    use std::simd::StdFloat;
    use std::simd::num::SimdFloat;
    use std::simd::num::SimdInt;
    type F = Simd<f32, 8>;
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_375;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let nf = (x * F::splat(LOG2E)).round();
    let r = x - nf * F::splat(LN2_HI);
    let r = r - nf * F::splat(LN2_LO);
    const C: [f32; 6] = [
        1.987_569_15e-4,
        1.398_199_95e-3,
        8.333_451_9e-3,
        4.166_579_6e-2,
        1.666_666_55e-1,
        5.000_000_1e-1,
    ];
    let mut p = F::splat(C[0]);
    for &c in &C[1..] {
        p = p * r + F::splat(c);
    }
    let z = r * r;
    p = p * z + r + F::splat(1.0);
    let ni = nf.cast::<i32>();
    let two_n = F::from_bits(((ni + Simd::splat(127)) << Simd::splat(23)).cast::<u32>());
    p * two_n
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
    fn simd_poly_exp_matches_scalar_algorithm_across_chunks_and_tail() {
        let xs = [
            -740.0,
            -120.25,
            -12.5,
            -1.0,
            -0.0,
            0.0,
            0.125,
            1.0,
            12.5,
            120.25,
            700.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            0.5,
            -0.5,
            709.0,
        ];
        let got = simd_poly_exp(&xs);
        for (&x, &g) in xs.iter().zip(&got) {
            let want = exp_scalar(x);
            if want.is_nan() {
                assert!(g.is_nan(), "x={x}");
            } else {
                assert_eq!(g.to_bits(), want.to_bits(), "x={x}");
            }
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

    #[test]
    fn exp_block_f32_accuracy() {
        use std::simd::Simd;
        // Moderate range (the activation-relevant regime); check max relative error vs libm.
        let mut maxrel = 0.0f32;
        let mut x = -30.0f32;
        while x <= 30.0 {
            let mut arr = [0.0f32; 8];
            for (j, a) in arr.iter_mut().enumerate() {
                *a = x + j as f32 * 0.001;
            }
            let got = exp_block_f32(Simd::from_array(arr)).to_array();
            for (j, &g) in got.iter().enumerate() {
                let e = (arr[j]).exp();
                if e.is_finite() && e > 0.0 {
                    maxrel = maxrel.max(((g - e) / e).abs());
                }
            }
            x += 0.017;
        }
        // f32 has ~6-7 significant digits; tolerance parity needs ~1e-6 relative.
        eprintln!("[exp_block_f32] max relative error vs libm over [-30,30] = {maxrel:e}");
        assert!(maxrel < 5e-6, "f32 SIMD exp rel err {maxrel:e} too large");
    }

    #[test]
    #[ignore = "perf; run explicitly"]
    fn bench_exp_block_f32_vs_libm() {
        use std::simd::Simd;
        let n = 1 << 22; // 4M f32
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-5 - 20.0).collect();
        let mut out = vec![0.0f32; n];
        let best = |f: &mut dyn FnMut()| -> f64 {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let simd = best(&mut || {
            let mut ch = xs.chunks_exact(8);
            let mut oc = out.chunks_exact_mut(8);
            for (c, o) in ch.by_ref().zip(oc.by_ref()) {
                exp_block_f32(Simd::from_slice(c)).copy_to_slice(o);
            }
            std::hint::black_box(&out);
        });
        let libm = best(&mut || {
            for (o, &x) in out.iter_mut().zip(&xs) {
                *o = x.exp();
            }
            std::hint::black_box(&out);
        });
        eprintln!(
            "[exp f32 4M, single-thread, NO-FMA] libm={:.3}ms simd8={:.3}ms speedup={:.2}x",
            libm * 1e3,
            simd * 1e3,
            libm / simd
        );
    }
}
