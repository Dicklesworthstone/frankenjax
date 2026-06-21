#![forbid(unsafe_code)]

//! FFT primitives: Fft, Ifft, Rfft, Irfft.
//!
//! The 1D transform runs along the last axis for each batch row (all leading
//! dimensions). Per-row engine by length: radix-2 Cooley-Tukey for powers of two,
//! native mixed-radix for smooth composites (every prime factor <= 13), and
//! Bluestein (chirp-z) for prime / rough lengths. (The O(n²) `dft_1d`/`idft_1d` are
//! TEST ORACLES, not a production fallback.)
//!
//! ## Batch SoA vectorization (frankenjax-murmw)
//!
//! For batches above a floor, the per-row kernels are run *vertically* over the
//! rows: the batch is transposed to a structure-of-arrays `[index][row]` layout so
//! each butterfly applies one shared scalar twiddle across a contiguous run of rows,
//! which the compiler autovectorizes. This is a MEASURED WIN only for FLAT/iterative
//! kernels, which keep one cache-resident buffer pair:
//!   - radix-2 full-complex fft/ifft + real-FFT rfft/irfft (~1.6-1.8x),
//!   - Bluestein via its flat radix-2 convolution FFTs (~3x, m up to 16384),
//!   - the iterative flat-stage mixed-radix (smooth composites).
//!
//! It is a LOSS for the RECURSIVE mixed-radix (no-ship 0.5-0.81x): its three
//! ping-ponging buffers spill L1 and its strided sub-DFT access does not vectorize —
//! at any tile size. RULE: SoA-vectorize FLAT kernels, never RECURSIVE ones.
//!
//! `transform_batches_dense` dispatch order — pow2 -> Bluestein (non-smooth) ->
//! iterative mixed-radix (smooth, n <= 1024) -> per-row fallback — partitions every
//! length exactly once. NB: only same-invocation INTERLEAVED A/B ratios are
//! trustworthy here; threaded and sequential-old-then-new timings drift badly under
//! swarm contention.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::f64::consts::PI;

use crate::EvalError;

// ── Helpers ──────────────────────────────────────────────────────────

/// Extract (re, im) from a Literal, treating real types as having zero imaginary part.
fn literal_to_complex(lit: Literal) -> Result<(f64, f64), &'static str> {
    match lit {
        Literal::Complex64Bits(re, im) => {
            Ok((f32::from_bits(re) as f64, f32::from_bits(im) as f64))
        }
        Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
        other => match other.as_f64() {
            Some(v) => Ok((v, 0.0)),
            None => Err("unsupported literal type for FFT"),
        },
    }
}

/// Determine the complex output dtype for FFT based on input dtype.
fn complex_dtype_for(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 | DType::Complex128 => dtype,
        DType::BF16 | DType::F16 | DType::F32 => DType::Complex64,
        _ => DType::Complex128,
    }
}

/// Determine the real output dtype from a complex dtype (for IRFFT).
fn real_dtype_for(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => DType::F64,
    }
}

/// Build a real Literal from an f64 value, preserving the tensor's logical dtype.
fn make_real_literal(val: f64, dtype: DType) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f64(val),
        DType::F16 => Literal::from_f16_f64(val),
        DType::F32 => Literal::from_f32(val as f32),
        _ => Literal::from_f64(val),
    }
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Complex64 | DType::Complex128)
}

fn last_axis_len(primitive: Primitive, shape: &Shape) -> Result<usize, EvalError> {
    shape
        .dims
        .last()
        .copied()
        .map(|dim| dim as usize)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "FFT expects rank >= 1 tensor, got empty shape".to_owned(),
        })
}

fn replace_last_axis_len(
    primitive: Primitive,
    dims: &mut [u32],
    len: usize,
) -> Result<(), EvalError> {
    let len = u32::try_from(len).map_err(|_| EvalError::Unsupported {
        primitive,
        detail: format!("FFT output last dimension exceeds u32: {len}"),
    })?;
    let last = dims.last_mut().ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: "FFT expects rank >= 1 tensor, got empty shape".to_owned(),
    })?;
    *last = len;
    Ok(())
}

fn checked_fft_len(
    primitive: Primitive,
    len: usize,
    description: &'static str,
) -> Result<(), EvalError> {
    u32::try_from(len)
        .map(|_| ())
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("{description} exceeds u32: {len}"),
        })
}

fn checked_output_element_count(
    primitive: Primitive,
    batch_size: usize,
    output_last: usize,
) -> Result<usize, EvalError> {
    batch_size
        .checked_mul(output_last)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: format!(
                "FFT output element count overflows usize: batch_size={batch_size}, last_dim={output_last}"
            ),
        })
}

fn parse_optional_fft_length(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
    default: usize,
) -> Result<usize, EvalError> {
    if let Some(raw) = params.get("fft_length") {
        return raw
            .trim()
            .parse::<usize>()
            .map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid fft_length: {raw}"),
            });
    }

    if let Some(raw) = params.get("fft_lengths") {
        let len = raw
            .split(',')
            .map(str::trim)
            .rfind(|part| !part.is_empty())
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("invalid fft_lengths: {raw}"),
            })?
            .parse::<usize>()
            .map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid fft_lengths: {raw}"),
            })?;
        return Ok(len);
    }

    Ok(default)
}

/// Extract a rank-1+ tensor from a Value, returning shape and elements as (re, im) pairs.
#[allow(clippy::type_complexity)]
fn extract_tensor_complex(
    primitive: Primitive,
    value: &Value,
) -> Result<(Shape, DType, Vec<(f64, f64)>), EvalError> {
    let tensor = match value {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "FFT expects a tensor (rank >= 1), got scalar".to_owned(),
            });
        }
    };

    if tensor.shape.rank() == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "FFT expects rank >= 1 tensor, got rank 0".to_owned(),
        });
    }

    // Fast path: dense complex-backed buffers expose packed `(re, im)` f64 pairs,
    // so we bulk-copy them instead of matching+converting each `Literal`. The dense
    // representation is bit-identical to what `literal_to_complex` would produce
    // (Complex128: exact f64; Complex64: the same f32→f64 widening).
    if let Some(dense) = tensor.elements.as_complex_slice() {
        return Ok((tensor.shape.clone(), tensor.dtype, dense.to_vec()));
    }

    // Dense real-F64 input (e.g. RFFT operand): read the packed f64 slice and lift
    // each value to `(v, 0.0)` — bit-identical to `literal_to_complex` on an
    // `F64Bits` literal, without the per-element enum match.
    if tensor.dtype == DType::F64
        && let Some(reals) = tensor.elements.as_f64_slice()
    {
        let elements: Vec<(f64, f64)> = reals.iter().map(|&v| (v, 0.0)).collect();
        return Ok((tensor.shape.clone(), tensor.dtype, elements));
    }

    // Dense real-F32 input (f32 is JAX's default float, so f32-real RFFT/FFT is the
    // common signal case). Lift each f32 to `(f64::from(v), 0.0)` — bit-identical to
    // `literal_to_complex` on an `F32Bits` literal (real part = f64::from(f32),
    // imag 0), without the per-element enum match + Literal dispatch.
    if tensor.dtype == DType::F32
        && let Some(reals) = tensor.elements.as_f32_slice()
    {
        let elements: Vec<(f64, f64)> = reals.iter().map(|&v| (f64::from(v), 0.0)).collect();
        return Ok((tensor.shape.clone(), tensor.dtype, elements));
    }

    let elements: Vec<(f64, f64)> = tensor
        .elements
        .iter()
        .map(|lit| {
            literal_to_complex(*lit).map_err(|_| EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric elements",
            })
        })
        .collect::<Result<_, _>>()?;

    Ok((tensor.shape.clone(), tensor.dtype, elements))
}

// ── DFT core ─────────────────────────────────────────────────────────

/// Precompute n-th roots of unity: e^{-2πi·k/n} for k = 0..n (forward DFT).
/// Returns (cos, sin) pairs for lookup by index.
#[inline]
fn precompute_twiddles(n: usize, inverse: bool) -> Vec<(f64, f64)> {
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            let angle = sign * 2.0 * PI * (k as f64) / (n as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            (cos_a, sin_a)
        })
        .collect()
}

/// Compute 1D DFT of a complex signal of length n.
/// X[k] = Σ_{j=0}^{n-1} x[j] * e^{-2πi·j·k/n}
///
/// Uses precomputed twiddle factors to reduce trig calls from O(n²) to O(n).
///
/// Retained as the O(n²) reference for validating the [`bluestein_dft_into`]
/// fast path in tests; production non-power-of-two transforms use Bluestein.
#[cfg(test)]
fn dft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let twiddles = precompute_twiddles(n, false);
    let mut output = vec![(0.0, 0.0); n];

    for (k, out) in output.iter_mut().enumerate() {
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;
        for (j, &(xr, xi)) in input.iter().enumerate() {
            let idx = (j * k) % n;
            let (cos_a, sin_a) = twiddles[idx];
            re_sum += xr * cos_a - xi * sin_a;
            im_sum += xr * sin_a + xi * cos_a;
        }
        *out = (re_sum, im_sum);
    }
    output
}

/// Compute 1D inverse DFT of a complex signal of length n.
/// x[j] = (1/n) Σ_{k=0}^{n-1} X[k] * e^{2πi·j·k/n}
///
/// Uses precomputed twiddle factors to reduce trig calls from O(n²) to O(n).
///
/// Retained as the O(n²) reference for validating the [`bluestein_dft_into`]
/// fast path in tests; production non-power-of-two transforms use Bluestein.
#[cfg(test)]
fn idft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let twiddles = precompute_twiddles(n, true);
    let inv_n = 1.0 / (n as f64);
    let mut output = vec![(0.0, 0.0); n];

    for (j, out) in output.iter_mut().enumerate() {
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;
        for (k, &(xr, xi)) in input.iter().enumerate() {
            let idx = (j * k) % n;
            let (cos_a, sin_a) = twiddles[idx];
            re_sum += xr * cos_a - xi * sin_a;
            im_sum += xr * sin_a + xi * cos_a;
        }
        *out = (re_sum * inv_n, im_sum * inv_n);
    }
    output
}

/// Bluestein's algorithm (chirp-z transform): the length-`n` DFT for ARBITRARY
/// `n` in O(n log n), by re-expressing the transform as a convolution evaluated
/// with power-of-two radix-2 FFTs — replacing the O(n²) direct [`dft_1d`] for
/// non-power-of-two lengths.
///
/// `inverse == false` matches [`dft_1d`] (X[k] = Σ x[j]·e^{-2πi·jk/n}); `inverse
/// == true` matches [`idft_1d`] (the +sign transform scaled by 1/n). The result
/// is bit-different from the direct DFT but agrees to floating tolerance (both
/// compute the same mathematical transform).
///
/// Derivation: with W = e^{sign·2πi/n}, jk = (j²+k²−(k−j)²)/2 gives
/// W^{jk} = chirp[j]·chirp[k]·conj(chirp[k−j]) where chirp[t] = e^{sign·πi·t²/n},
/// so X[k] = chirp[k]·Σ_j (x[j]·chirp[j])·conj(chirp[k−j]) — a convolution of
/// a[j] = x[j]·chirp[j] with the kernel conj(chirp). The chirp exponent is
/// 2n-periodic in t², so t²·mod·2n keeps the angle small and accurate.
/// A Bluestein "plan" for a fixed transform length `n` and direction: the chirp
/// table and the FFT of the convolution kernel. For a batched transform (an FFT
/// along the last axis of `[B, n]`) these are identical for every one of the `B`
/// rows, so building the plan once and reusing it across rows removes the
/// per-row chirp `sin_cos` table builds, the per-row kernel FFT, and the per-row
/// allocations — turning ~3 FFTs + ~3n trig calls per row into ~2 FFTs and zero
/// trig per row.
struct BluesteinPlan {
    n: usize,
    m: usize,
    inverse: bool,
    /// chirp[t] = e^{sign·πi·t²/n} for t in 0..n (sign = -1 forward, +1 inverse).
    chirp: Vec<(f64, f64)>,
    /// FFT of the symmetric conj(chirp) kernel, length m (invariant across rows).
    fb: Vec<(f64, f64)>,
    radix2_forward: Radix2Plan,
    radix2_inverse: Radix2Plan,
}

/// Reusable per-row work buffers for [`BluesteinPlan::apply_into`].
#[derive(Default)]
struct BluesteinScratch {
    a: Vec<(f64, f64)>,
    fa: Vec<(f64, f64)>,
    prod: Vec<(f64, f64)>,
    conv: Vec<(f64, f64)>,
}

/// A fixed-size radix-2 FFT execution plan.
///
/// The twiddle table is generated by the exact same per-stage recurrence as
/// [`radix2_fft_1d_into`]. Reusing those values across Bluestein rows removes
/// the per-row recurrence work without changing any butterfly arithmetic.
struct Radix2Plan {
    n: usize,
    inverse: bool,
    bit_reversed: Vec<usize>,
    twiddles: Vec<(f64, f64)>,
}

impl Radix2Plan {
    fn new(n: usize, inverse: bool) -> Self {
        debug_assert!(n.is_power_of_two());
        let bit_reversed = bit_reverse_permutation(n);
        let mut twiddles = Vec::with_capacity(n.saturating_sub(1));

        let mut len = 2_usize;
        while len <= n {
            let half = len / 2;
            let angle = if inverse {
                2.0 * PI / (len as f64)
            } else {
                -2.0 * PI / (len as f64)
            };
            let (sin_step, cos_step) = angle.sin_cos();
            let mut twiddle_re = 1.0;
            let mut twiddle_im = 0.0;

            for _ in 0..half {
                twiddles.push((twiddle_re, twiddle_im));
                let next_re = twiddle_re * cos_step - twiddle_im * sin_step;
                let next_im = twiddle_re * sin_step + twiddle_im * cos_step;
                twiddle_re = next_re;
                twiddle_im = next_im;
            }

            len *= 2;
        }

        Self {
            n,
            inverse,
            bit_reversed,
            twiddles,
        }
    }

    fn apply_into(&self, input: &[(f64, f64)], output: &mut Vec<(f64, f64)>) {
        debug_assert_eq!(input.len(), self.n);

        output.clear();
        output.reserve(self.n);
        output.extend(self.bit_reversed.iter().map(|&idx| input[idx]));

        let mut twiddle_base = 0_usize;
        let mut len = 2_usize;
        while len <= self.n {
            let half = len / 2;
            let stage_twiddles = &self.twiddles[twiddle_base..twiddle_base + half];

            for start in (0..self.n).step_by(len) {
                for (offset, &(twiddle_re, twiddle_im)) in stage_twiddles.iter().enumerate() {
                    let even = output[start + offset];
                    let odd = output[start + offset + half];
                    let rotated_re = odd.0 * twiddle_re - odd.1 * twiddle_im;
                    let rotated_im = odd.0 * twiddle_im + odd.1 * twiddle_re;

                    output[start + offset] = (even.0 + rotated_re, even.1 + rotated_im);
                    output[start + offset + half] = (even.0 - rotated_re, even.1 - rotated_im);
                }
            }

            twiddle_base += half;
            len *= 2;
        }

        if self.inverse {
            let inv_n = 1.0 / (self.n as f64);
            for value in output.iter_mut() {
                value.0 *= inv_n;
                value.1 *= inv_n;
            }
        }
    }
}

/// Reusable plan for an even power-of-two real FFT.
///
/// The N-point real signal is packed into an N/2-point complex signal and
/// transformed with a planned radix-2 FFT, then recombined with precomputed
/// twiddles. This changes floating-point operation order versus the full complex
/// FFT path, so tests validate mathematical equivalence by tolerance.
struct RealRfftPower2Plan {
    fft_length: usize,
    half_len: usize,
    half_fft: Radix2Plan,
    twiddles: Vec<(f64, f64)>,
}

impl RealRfftPower2Plan {
    fn new(fft_length: usize) -> Self {
        debug_assert!(fft_length >= 2);
        debug_assert!(fft_length.is_power_of_two());
        let half_len = fft_length / 2;
        let half_fft = Radix2Plan::new(half_len, false);
        let twiddles = (1..half_len)
            .map(|k| {
                let angle = -2.0 * PI * (k as f64) / (fft_length as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                (cos_a, sin_a)
            })
            .collect();
        Self {
            fft_length,
            half_len,
            half_fft,
            twiddles,
        }
    }

    fn apply_into(
        &self,
        input: &[(f64, f64)],
        copy_len: usize,
        packed: &mut Vec<(f64, f64)>,
        transformed: &mut Vec<(f64, f64)>,
        out: &mut [(f64, f64)],
    ) {
        debug_assert_eq!(out.len(), self.half_len + 1);
        packed.clear();
        packed.resize(self.half_len, (0.0, 0.0));

        for (idx, slot) in packed.iter_mut().enumerate() {
            let even = 2 * idx;
            let odd = even + 1;
            let even_re = if even < copy_len { input[even].0 } else { 0.0 };
            let odd_re = if odd < copy_len { input[odd].0 } else { 0.0 };
            *slot = (even_re, odd_re);
        }

        self.half_fft.apply_into(packed, transformed);

        let (z0_re, z0_im) = transformed[0];
        out[0] = (z0_re + z0_im, 0.0);
        out[self.half_len] = (z0_re - z0_im, 0.0);

        for k in 1..self.half_len {
            let (a_re, a_im) = transformed[k];
            let (b_re, b_im) = transformed[self.half_len - k];
            let avg_re = 0.5 * (a_re + b_re);
            let avg_im = 0.5 * (a_im - b_im);
            let diff_re = 0.5 * (a_re - b_re);
            let diff_im = 0.5 * (a_im + b_im);

            let (cos_a, sin_a) = self.twiddles[k - 1];
            let rotated_re = diff_re * cos_a - diff_im * sin_a;
            let rotated_im = diff_re * sin_a + diff_im * cos_a;
            out[k] = (avg_re + rotated_im, avg_im - rotated_re);
        }
    }
}

impl BluesteinPlan {
    /// Precompute the chirp table and kernel FFT for length `n` (`n >= 2`).
    fn new(n: usize, inverse: bool) -> Self {
        let sign = if inverse { 1.0 } else { -1.0 };
        let two_n = 2_u128 * n as u128;
        // chirp exponent is 2n-periodic in t², so t²·mod·2n keeps the angle
        // small and accurate.
        let chirp: Vec<(f64, f64)> = (0..n)
            .map(|t| {
                let m2 = ((t as u128 * t as u128) % two_n) as f64;
                let angle = sign * PI * m2 / (n as f64);
                let (s, c) = angle.sin_cos();
                (c, s)
            })
            .collect();

        // Convolution length: a power of two >= 2n-1, so the size-m circular
        // convolution reproduces the linear convolution Bluestein requires.
        let m = (2 * n - 1).next_power_of_two();
        let radix2_forward = Radix2Plan::new(m, false);
        let radix2_inverse = Radix2Plan::new(m, true);

        let mut b = vec![(0.0_f64, 0.0_f64); m];
        // Kernel g[d] = conj(chirp[d]); symmetric-extended.
        b[0] = (1.0, 0.0);
        for j in 1..n {
            let (cr, ci) = chirp[j];
            let g = (cr, -ci);
            b[j] = g;
            b[m - j] = g;
        }
        let mut fb = Vec::new();
        radix2_forward.apply_into(&b, &mut fb);

        Self {
            n,
            m,
            inverse,
            chirp,
            fb,
            radix2_forward,
            radix2_inverse,
        }
    }

    /// Transform one length-`n` row into `output`, reusing `scratch`.
    fn apply_into(
        &self,
        input: &[(f64, f64)],
        scratch: &mut BluesteinScratch,
        output: &mut Vec<(f64, f64)>,
    ) {
        let BluesteinScratch { a, fa, prod, conv } = scratch;
        let n = self.n;
        let m = self.m;

        a.clear();
        a.resize(m, (0.0, 0.0));
        for (j, slot) in a.iter_mut().take(n).enumerate() {
            let (cr, ci) = self.chirp[j];
            let (xr, xi) = input[j];
            *slot = (xr * cr - xi * ci, xr * ci + xi * cr);
        }

        self.radix2_forward.apply_into(a, fa);

        prod.clear();
        prod.resize(m, (0.0, 0.0));
        for (p, (&(ar, ai), &(br, bi))) in prod.iter_mut().zip(fa.iter().zip(self.fb.iter())) {
            *p = (ar * br - ai * bi, ar * bi + ai * br);
        }

        self.radix2_inverse.apply_into(prod, conv);

        let inv_n = if self.inverse { 1.0 / (n as f64) } else { 1.0 };
        output.clear();
        output.reserve(n);
        for (k, &(vr, vi)) in conv.iter().take(n).enumerate() {
            let (cr, ci) = self.chirp[k];
            output.push(((cr * vr - ci * vi) * inv_n, (cr * vi + ci * vr) * inv_n));
        }
    }
}

/// Bluestein's algorithm (chirp-z transform): the length-`n` DFT for ARBITRARY
/// `n` in O(n log n), by re-expressing the transform as a convolution evaluated
/// with power-of-two radix-2 FFTs — replacing the O(n²) direct [`dft_1d`] for
/// non-power-of-two lengths.
///
/// `inverse == false` matches [`dft_1d`] (X[k] = Σ x[j]·e^{-2πi·jk/n}); `inverse
/// == true` matches [`idft_1d`] (the +sign transform scaled by 1/n). The result
/// is bit-different from the direct DFT but agrees to floating tolerance (both
/// compute the same mathematical transform).
///
/// Single-shot entry point (builds a one-row plan). Batched callers should build
/// a [`BluesteinPlan`] once and reuse it across rows via [`BluesteinPlan::apply_into`].
///
/// Derivation: with W = e^{sign·2πi/n}, jk = (j²+k²−(k−j)²)/2 gives
/// W^{jk} = chirp[j]·chirp[k]·conj(chirp[k−j]) where chirp[t] = e^{sign·πi·t²/n},
/// so X[k] = chirp[k]·Σ_j (x[j]·chirp[j])·conj(chirp[k−j]) — a convolution of
/// a[j] = x[j]·chirp[j] with the kernel conj(chirp).
fn bluestein_dft_into(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>, inverse: bool) {
    let n = input.len();
    output.clear();
    if n == 0 {
        return;
    }
    if n == 1 {
        // Length-1 DFT and its (1/1-scaled) inverse are both the identity.
        output.push(input[0]);
        return;
    }
    let plan = BluesteinPlan::new(n, inverse);
    let mut scratch = BluesteinScratch::default();
    plan.apply_into(input, &mut scratch, output);
}

fn bit_reverse_permute(values: &mut [(f64, f64)]) {
    let n = values.len();
    let mut j = 0_usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while bit != 0 && (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            values.swap(i, j);
        }
    }
}

fn bit_reverse_permutation(n: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut j = 0_usize;

    for i in 1..n {
        let mut bit = n >> 1;
        while bit != 0 && (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            indices.swap(i, j);
        }
    }

    indices
}

fn radix2_fft_1d_into(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>, inverse: bool) {
    let n = input.len();
    output.clear();
    output.extend_from_slice(input);
    if n <= 1 {
        return;
    }
    bit_reverse_permute(output);

    // Twiddle scratch reused across stages (max length n/2). Precomputing each stage's
    // twiddles via the SAME first-order recurrence the inline loop used produces
    // BIT-IDENTICAL values (deterministic from (1,0)), but lifts the loop-carried
    // recurrence dependency out of the butterfly loop so consecutive butterflies are
    // independent and the compiler can pipeline/vectorize the complex multiply.
    let mut tw: Vec<(f64, f64)> = Vec::with_capacity(n / 2);
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / (len as f64)
        } else {
            -2.0 * PI / (len as f64)
        };
        let (sin_step, cos_step) = angle.sin_cos();

        // Build the stage's twiddle table via the identical recurrence (tw[k] equals the
        // inline `twiddle` at offset k, so the butterfly results are bit-for-bit unchanged).
        tw.clear();
        let mut twiddle_re = 1.0;
        let mut twiddle_im = 0.0;
        for _ in 0..half {
            tw.push((twiddle_re, twiddle_im));
            let next_re = twiddle_re * cos_step - twiddle_im * sin_step;
            let next_im = twiddle_re * sin_step + twiddle_im * cos_step;
            twiddle_re = next_re;
            twiddle_im = next_im;
        }

        for start in (0..n).step_by(len) {
            let (lo, hi) = output[start..start + len].split_at_mut(half);
            for offset in 0..half {
                let (twiddle_re, twiddle_im) = tw[offset];
                let even = lo[offset];
                let odd = hi[offset];
                let rotated_re = odd.0 * twiddle_re - odd.1 * twiddle_im;
                let rotated_im = odd.0 * twiddle_im + odd.1 * twiddle_re;

                lo[offset] = (even.0 + rotated_re, even.1 + rotated_im);
                hi[offset] = (even.0 - rotated_re, even.1 - rotated_im);
            }
        }

        len *= 2;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for value in output.iter_mut() {
            value.0 *= inv_n;
            value.1 *= inv_n;
        }
    }
}

#[cfg(test)]
fn fft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut output = Vec::new();
    fft_1d_into(input, &mut output);
    output
}

fn fft_1d_into(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>) {
    let n = input.len();
    if n.is_power_of_two() {
        radix2_fft_1d_into(input, output, false);
    } else if is_mixed_radix_smooth(n) {
        // Smooth composite (every prime factor ≤ 13): mixed-radix Cooley-Tukey runs
        // on a length-`n` buffer, while Bluestein zero-pads to the next pow2 ≥ 2n-1
        // and runs FFTs of THAT length — ~2-4x more memory for the same result. The
        // BATCHED path already chooses mixed-radix here; this makes the single-row
        // path consistent. Tolerance parity (mixed_radix == DFT, like Bluestein).
        let roots = precompute_twiddles(n, false);
        let mut scratch = Vec::new();
        mixed_radix_into(input, &roots, false, output, &mut scratch);
    } else {
        // Large prime factor / prime length: Bluestein O(n log n) vs the O(n²) DFT.
        bluestein_dft_into(input, output, false);
    }
}

#[cfg(test)]
fn ifft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut output = Vec::new();
    ifft_1d_into(input, &mut output);
    output
}

fn ifft_1d_into(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>) {
    let n = input.len();
    if n.is_power_of_two() {
        radix2_fft_1d_into(input, output, true);
    } else if is_mixed_radix_smooth(n) {
        // Smooth composite: mixed-radix (the inverse `1/n` normalization is applied
        // inside mixed_radix_into, matching radix-2 / Bluestein). See fft_1d_into.
        let roots = precompute_twiddles(n, true);
        let mut scratch = Vec::new();
        mixed_radix_into(input, &roots, true, output, &mut scratch);
    } else {
        // Large prime factor / prime length: Bluestein O(n log n) vs the O(n²) IDFT.
        bluestein_dft_into(input, output, true);
    }
}

/// Largest prime factor a length may have to take the mixed-radix path instead
/// of Bluestein. The radix base-case is a direct O(p²) DFT, so this is kept small
/// — any larger prime factor (e.g. a prime length) falls back to Bluestein.
const MIXED_RADIX_MAX_PRIME: usize = 13;

/// Smallest prime factor of `n` (n >= 2).
fn smallest_prime_factor(n: usize) -> usize {
    if n.is_multiple_of(2) {
        return 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return d;
        }
        d += 2;
    }
    n
}

/// Whether `n` should use the mixed-radix FFT: composite, not a power of two, and
/// with every prime factor ≤ `MIXED_RADIX_MAX_PRIME`.
fn is_mixed_radix_smooth(n: usize) -> bool {
    if n < 2 || n.is_power_of_two() {
        return false;
    }
    let mut m = n;
    while m > 1 {
        let p = smallest_prime_factor(m);
        if p > MIXED_RADIX_MAX_PRIME {
            return false;
        }
        m /= p;
    }
    true
}

/// Allocation-free recursive mixed-radix Cooley-Tukey DFT (decimation-in-time).
///
/// Computes the length-`nn` DFT of the strided input `x[offset], x[offset+stride],
/// …` into `out[0..nn]`, using `scratch[0..nn]` (a same-size sibling buffer) as
/// workspace. At each level the length is split by its smallest prime factor `p`
/// into `p` subsequences of length `m = nn/p`; the `p` sub-DFTs are written into
/// disjoint blocks of `scratch` (each using the matching block of `out` as ITS
/// scratch — the two buffers ping-pong roles down the recursion), then combined
/// via `X[q] = Σ_r W_nn^{r·q} · F_r[q mod m]`. The global `roots` table of size
/// `big_n` (`roots[k] = e^{sign·2πi·k/big_n}`) supplies every twiddle by stride
/// `big_n/nn`, so no trig and **no per-node allocation** happens in the recursion.
/// For a smooth `nn` this is O(nn · Σ prime factors) flops with two buffers total.
#[allow(clippy::too_many_arguments)]
fn mixed_radix_ping(
    x: &[(f64, f64)],
    offset: usize,
    stride: usize,
    out: &mut [(f64, f64)],
    scratch: &mut [(f64, f64)],
    nn: usize,
    roots: &[(f64, f64)],
    big_n: usize,
) {
    if nn == 1 {
        out[0] = x[offset];
        return;
    }
    let p = smallest_prime_factor(nn);
    let m = nn / p;

    // Recurse: sub-DFT r -> scratch[r*m..], using out[r*m..] as its workspace.
    for r in 0..p {
        let base = r * m;
        mixed_radix_ping(
            x,
            offset + r * stride,
            stride * p,
            &mut scratch[base..base + m],
            &mut out[base..base + m],
            m,
            roots,
            big_n,
        );
    }

    // Combine the p sub-DFTs (now in `scratch`) into `out`:
    // out[q] = Σ_r W_nn^{r·q} · scratch[r·m + (q mod m)].
    // The twiddle index `(r·q) mod nn` and the source index `q mod m` are advanced
    // by increment-and-wrap counters rather than a per-element integer modulo
    // (division), which dominated at these lengths. Each `out[q]` accumulates the
    // r-terms in ascending r, so the result is identical to the direct form.
    let s_tw = big_n / nn;

    // Specialized radix-2 butterfly: out[s] = E + W·O, out[s+m] = E − W·O, one
    // pass over the m groups (the ± symmetry halves the work vs the general
    // accumulate, and radix-2 is the most common split for power-of-two-rich n).
    if p == 2 {
        let (lo, hi) = out.split_at_mut(m);
        for s in 0..m {
            let (e_re, e_im) = scratch[s];
            let (o_re, o_im) = scratch[m + s];
            let (tr, ti) = roots[s * s_tw]; // W_nn^s
            let wr = tr * o_re - ti * o_im;
            let wi = tr * o_im + ti * o_re;
            lo[s] = (e_re + wr, e_im + wi);
            hi[s] = (e_re - wr, e_im - wi);
        }
        return;
    }

    // Specialized radix-3 butterfly. After twiddling the two non-zero sub-DFTs
    // (u1 = W_nn^s·F1, u2 = W_nn^{2s}·F2), the length-3 DFT of (u0,u1,u2) factors
    // into one cos/sin pair because W_3^2 = conj(W_3): writing w3 = (c,d) =
    // roots[big_n/3] (so c = cos(2π/3), d = ±sin(2π/3) carrying the transform's
    // sign), the three outputs are u0+u1+u2 and (u0 + c·(u1+u2)) ∓ d·i·(u1−u2).
    // Pulling c,d from the same roots table keeps the inverse sign exact. This
    // replaces 9 complex mults (3-pass accumulate) with 2 twiddle mults + a real
    // butterfly per group, and agrees with the direct DFT to floating tolerance.
    if p == 3 {
        let (c, d) = roots[big_n / 3]; // W_3 = (cos 2π/3, ±sin 2π/3)
        let m2 = 2 * m;
        for s in 0..m {
            let u0 = scratch[s];
            let (a1r, a1i) = scratch[m + s];
            let (a2r, a2i) = scratch[m2 + s];
            let (t1r, t1i) = roots[s * s_tw]; // W_nn^s
            let (t2r, t2i) = roots[2 * s * s_tw]; // W_nn^{2s}
            let u1 = (t1r * a1r - t1i * a1i, t1r * a1i + t1i * a1r);
            let u2 = (t2r * a2r - t2i * a2i, t2r * a2i + t2i * a2r);
            let sr = u1.0 + u2.0; // (u1+u2).re
            let si = u1.1 + u2.1; // (u1+u2).im
            let dr = u1.0 - u2.0; // (u1-u2).re
            let di = u1.1 - u2.1; // (u1-u2).im
            out[s] = (u0.0 + sr, u0.1 + si);
            let cr = u0.0 + c * sr;
            let ci = u0.1 + c * si;
            let xr = d * di; // d·(u1-u2).im
            let xi = d * dr; // d·(u1-u2).re
            out[m + s] = (cr - xr, ci + xi);
            out[m2 + s] = (cr + xr, ci - xi);
        }
        return;
    }

    // Specialized radix-5 butterfly. Twiddle u1..u4, then exploit W_5^4=conj(W_5),
    // W_5^3=conj(W_5^2): with (c1,d1)=roots[big_n/5] and (c2,d2)=roots[2·big_n/5],
    // the five outputs collapse into two cos/sin pairs over the conjugate sums
    // t1=u1+u4, t2=u2+u3 and the conjugate diffs t1d=u1−u4, t2d=u2−u3. Constants
    // come from the roots table so the inverse sign is exact. Replaces 25 complex
    // mults with 4 twiddle mults + a real butterfly; tolerance-equal to the DFT.
    if p == 5 {
        let (c1, d1) = roots[big_n / 5];
        let (c2, d2) = roots[2 * (big_n / 5)];
        let (m2, m3, m4) = (2 * m, 3 * m, 4 * m);
        for s in 0..m {
            let u0 = scratch[s];
            let tw = |r: usize, a: (f64, f64)| {
                let (tr, ti) = roots[r * s * s_tw];
                (tr * a.0 - ti * a.1, tr * a.1 + ti * a.0)
            };
            let u1 = tw(1, scratch[m + s]);
            let u2 = tw(2, scratch[m2 + s]);
            let u3 = tw(3, scratch[m3 + s]);
            let u4 = tw(4, scratch[m4 + s]);
            let t1 = (u1.0 + u4.0, u1.1 + u4.1);
            let t1d = (u1.0 - u4.0, u1.1 - u4.1);
            let t2 = (u2.0 + u3.0, u2.1 + u3.1);
            let t2d = (u2.0 - u3.0, u2.1 - u3.1);
            out[s] = (u0.0 + t1.0 + t2.0, u0.1 + t1.1 + t2.1);
            // X1/X4 share cos-combo (c1·t1 + c2·t2), differ by cross terms.
            let ar = u0.0 + c1 * t1.0 + c2 * t2.0;
            let ai = u0.1 + c1 * t1.1 + c2 * t2.1;
            let xr = d1 * t1d.1 + d2 * t2d.1;
            let xi = d1 * t1d.0 + d2 * t2d.0;
            out[m + s] = (ar - xr, ai + xi);
            out[m4 + s] = (ar + xr, ai - xi);
            // X2/X3 share cos-combo (c2·t1 + c1·t2).
            let br = u0.0 + c2 * t1.0 + c1 * t2.0;
            let bi = u0.1 + c2 * t1.1 + c1 * t2.1;
            let yr = d2 * t1d.1 - d1 * t2d.1;
            let yi = d2 * t1d.0 - d1 * t2d.0;
            out[m2 + s] = (br - yr, bi + yi);
            out[m3 + s] = (br + yr, bi - yi);
        }
        return;
    }

    for slot in out.iter_mut() {
        *slot = (0.0, 0.0);
    }
    for r in 0..p {
        let row = r * m;
        let mut idx = 0usize; // (r·q) mod nn
        let mut qm = 0usize; // q mod m
        for slot in out.iter_mut() {
            let (tr, ti) = roots[idx * s_tw];
            let (sr, si) = scratch[row + qm];
            slot.0 += tr * sr - ti * si;
            slot.1 += tr * si + ti * sr;
            idx += r;
            if idx >= nn {
                idx -= nn;
            }
            qm += 1;
            if qm == m {
                qm = 0;
            }
        }
    }
}

/// Apply the mixed-radix FFT for one length-`n` row using a precomputed `roots`
/// table (`precompute_twiddles(n, inverse)`), writing into `output` and reusing
/// the caller-owned `scratch` buffer (both resized to `n`). For the inverse
/// transform the `1/n` normalization is applied here, matching `radix2_fft_1d_into`
/// / Bluestein so callers need no extra scaling.
fn mixed_radix_into(
    input: &[(f64, f64)],
    roots: &[(f64, f64)],
    inverse: bool,
    output: &mut Vec<(f64, f64)>,
    scratch: &mut Vec<(f64, f64)>,
) {
    let n = input.len();
    output.clear();
    output.resize(n, (0.0, 0.0));
    scratch.clear();
    scratch.resize(n, (0.0, 0.0));
    mixed_radix_ping(input, 0, 1, output, scratch, n, roots, n);
    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in output.iter_mut() {
            v.0 *= inv_n;
            v.1 *= inv_n;
        }
    }
}

fn mixed_radix_factors(n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut nn = n;
    while nn > 1 {
        let p = smallest_prime_factor(nn);
        factors.push(p);
        nn /= p;
    }
    factors
}

/// Largest smooth-composite length routed through the flat iterative SoA mixed-radix
/// candidate. DISABLED (= 0): the RCH A/B (`bench_mixed_radix_iterative_soa_vs_per_row`,
/// interleaved min-of-9, 2026-06-21) measured `128x1000` at per-row=2.947ms vs
/// iter=19.096ms = **0.15x (6.5x SLOWER)** — a no-ship like the recursive-SoA mixed-radix
/// before it (the SoA transpose + strided per-lane access is memory-bound at n=1000; the
/// inner butterflies don't autovectorize the way flat radix-2 / Bluestein do). Smooth
/// composites stay on the proven recursive per-row path. The kernel + its tests are kept
/// (correctness-validated) as a documented no-ship; flip to a positive cap only if a future
/// rewrite beats per-row in this same A/B. See the murmw FFT ledger entry.
const MIXED_RADIX_ITERATIVE_SOA_MAX_N: usize = 0;
const MIXED_RADIX_ITERATIVE_SOA_MAX_ELEMS: usize = 1 << 18;

fn mixed_radix_iterative_soa_gate_allows_n(n: usize) -> bool {
    match MIXED_RADIX_ITERATIVE_SOA_MAX_N {
        0 => false,
        max_n => n <= max_n,
    }
}
const MIXED_RADIX_ITERATIVE_SOA_TILE_ROWS: usize = 4;

/// Flat-stage mixed-radix SoA candidate for smooth-composite batched FFT/IFFT.
///
/// This is the production lift of the test-only disk-low prototype: digit-reverse
/// each row into `index x row` SoA, then run iterative mixed-radix stages over
/// contiguous row lanes. It intentionally changes floating-point op order versus
/// the recursive `mixed_radix_into` path, so it is tolerance-equivalent rather
/// than bit-identical; FFT oracle parity is tolerance-based for these composite
/// lengths. Bench/proof remains pending because this was committed under a
/// disk-low no-build/no-bench instruction.
fn transform_batches_mixed_radix_iterative_soa(
    elements: &[(f64, f64)],
    n: usize,
    batch_size: usize,
    inverse: bool,
    roots: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let total = batch_size * n;
    let mut out = vec![(0.0, 0.0); total];
    if n <= 1 {
        out.copy_from_slice(&elements[..total]);
        return out;
    }

    let factors = mixed_radix_factors(n);
    let max_r = factors.iter().copied().max().unwrap_or(1);
    let rows_cap = MIXED_RADIX_ITERATIVE_SOA_TILE_ROWS;
    let mut re = vec![0.0f64; rows_cap * n];
    let mut im = vec![0.0f64; rows_cap * n];
    let mut y_re = vec![0.0f64; rows_cap * max_r];
    let mut y_im = vec![0.0f64; rows_cap * max_r];

    let mut row0 = 0usize;
    while row0 < batch_size {
        let rows = rows_cap.min(batch_size - row0);
        let s = row0 * n;
        let e = s + rows * n;
        mixed_radix_iterative_soa_block(
            &elements[s..e],
            n,
            rows,
            inverse,
            roots,
            &factors,
            &mut re[..rows * n],
            &mut im[..rows * n],
            &mut y_re[..rows * max_r],
            &mut y_im[..rows * max_r],
            &mut out[s..e],
        );
        row0 += rows;
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn mixed_radix_iterative_soa_block(
    elements: &[(f64, f64)],
    n: usize,
    rows: usize,
    inverse: bool,
    roots: &[(f64, f64)],
    factors: &[usize],
    re: &mut [f64],
    im: &mut [f64],
    y_re: &mut [f64],
    y_im: &mut [f64],
    out: &mut [(f64, f64)],
) {
    debug_assert_eq!(elements.len(), rows * n);
    debug_assert_eq!(out.len(), rows * n);
    debug_assert_eq!(re.len(), rows * n);
    debug_assert_eq!(im.len(), rows * n);
    let w = rows;

    for i in 0..n {
        let mut x = i;
        let mut rev = 0usize;
        for &f in factors {
            rev = rev * f + (x % f);
            x /= f;
        }
        for row in 0..w {
            let (vr, vi) = elements[row * n + rev];
            re[i * w + row] = vr;
            im[i * w + row] = vi;
        }
    }

    let mut len = 1usize;
    for &r in factors {
        let next = len * r;
        let input_twiddle_stride = n / next;
        let dft_twiddle_stride = n / r;
        let mut start = 0usize;
        while start < n {
            for j in 0..len {
                for t in 0..r {
                    let (twr, twi) = roots[(input_twiddle_stride * j * t) % n];
                    let base = (start + j + t * len) * w;
                    let ybase = t * w;
                    for row in 0..w {
                        let vr = re[base + row];
                        let vi = im[base + row];
                        y_re[ybase + row] = vr * twr - vi * twi;
                        y_im[ybase + row] = vr * twi + vi * twr;
                    }
                }

                for s in 0..r {
                    let out_base = (start + j + s * len) * w;
                    for row in 0..w {
                        let mut acc_re = 0.0f64;
                        let mut acc_im = 0.0f64;
                        for t in 0..r {
                            let (wr, wi) = roots[(dft_twiddle_stride * s * t) % n];
                            let ybase = t * w + row;
                            let yr = y_re[ybase];
                            let yi = y_im[ybase];
                            acc_re += yr * wr - yi * wi;
                            acc_im += yr * wi + yi * wr;
                        }
                        re[out_base + row] = acc_re;
                        im[out_base + row] = acc_im;
                    }
                }
            }
            start += next;
        }
        len = next;
    }

    let inv_n = if inverse { 1.0 / (n as f64) } else { 1.0 };
    for row in 0..w {
        let dst = &mut out[row * n..row * n + n];
        for (k, slot) in dst.iter_mut().enumerate() {
            *slot = (re[k * w + row] * inv_n, im[k * w + row] * inv_n);
        }
    }
}

/// Convolution-length ceiling for the SoA Bluestein path. The two internal pow2
/// FFTs are the proven flat radix-2 kernel (`vectorized_pow2_block`); the win is
/// robust ~3x across the whole measured range (m=256..16384, see
/// `bench_vectorized_bluestein_vs_per_row`) because flat radix-2 vectorizes
/// regardless of size. Capped at the largest measured-winning `m`; raising further
/// is plausible but unverified.
const BLUESTEIN_SOA_MAX_M: usize = 16384;
const BLUESTEIN_TILE_ROWS: usize = 4;

/// Cache-blocked batched Bluestein FFT/IFFT via the SoA kernel: the chirp pre/post
/// multiplies and the kernel pointwise multiply are vectorized vertically over a
/// row tile, and the two internal pow2 convolution FFTs run through the proven
/// `vectorized_pow2_block`. Bit-identical to per-row `BluesteinPlan::apply_into`
/// (same chirp arithmetic, same radix-2 plans, same kernel `fb`, same `1/n` scale).
fn transform_batches_bluestein_vectorized(
    plan: &BluesteinPlan,
    elements: &[(f64, f64)],
    n: usize,
    batch_size: usize,
) -> Vec<(f64, f64)> {
    let m = plan.m;
    let inv_n = if plan.inverse { 1.0 / (n as f64) } else { 1.0 };
    let total = batch_size * n;
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

    const PARALLEL_MIN_ELEMS: usize = 1 << 18;
    let threads = if batch_size.saturating_mul(m) >= PARALLEL_MIN_ELEMS && batch_size > 1 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(batch_size)
    } else {
        1
    };

    let run_block = |elem: &[(f64, f64)], rows: usize, out_blk: &mut [(f64, f64)]| {
        let cap = BLUESTEIN_TILE_ROWS * m;
        let mut a = vec![(0.0, 0.0); cap]; // a -> prod (reused)
        let mut fa = vec![(0.0, 0.0); cap]; // fa -> conv (reused)
        let mut re = vec![0.0f64; cap];
        let mut im = vec![0.0f64; cap];
        let mut row0 = 0usize;
        while row0 < rows {
            let tile = BLUESTEIN_TILE_ROWS.min(rows - row0);
            let w = tile;
            let span = w * m;
            // a[b*m + j] = chirp[j] * input_row_b[j] for j<n, else 0.
            for slot in a[..span].iter_mut() {
                *slot = (0.0, 0.0);
            }
            for b in 0..tile {
                let src = &elem[(row0 + b) * n..(row0 + b) * n + n];
                for j in 0..n {
                    let (cr, ci) = plan.chirp[j];
                    let (xr, xi) = src[j];
                    a[b * m + j] = (xr * cr - xi * ci, xr * ci + xi * cr);
                }
            }
            // Forward convolution FFT (SoA): fa <- FFT(a).
            vectorized_pow2_block(
                &plan.radix2_forward,
                &a[..span],
                w,
                m,
                false,
                &mut re[..span],
                &mut im[..span],
                &mut fa[..span],
            );
            // Pointwise kernel multiply: a (reused as prod) = fa * fb.
            for b in 0..tile {
                for p in 0..m {
                    let (ar, ai) = fa[b * m + p];
                    let (br, bi) = plan.fb[p];
                    a[b * m + p] = (ar * br - ai * bi, ar * bi + ai * br);
                }
            }
            // Inverse convolution FFT (SoA, scales by 1/m): fa (reused as conv) <- IFFT(prod).
            vectorized_pow2_block(
                &plan.radix2_inverse,
                &a[..span],
                w,
                m,
                true,
                &mut re[..span],
                &mut im[..span],
                &mut fa[..span],
            );
            // Post-chirp + 1/n inverse scale: out[k] = chirp[k]*conv[k]*inv_n, k<n.
            for b in 0..tile {
                let dst = &mut out_blk[(row0 + b) * n..(row0 + b) * n + n];
                for (k, slot) in dst.iter_mut().enumerate() {
                    let (cr, ci) = plan.chirp[k];
                    let (vr, vi) = fa[b * m + k];
                    *slot = ((cr * vr - ci * vi) * inv_n, (cr * vi + ci * vr) * inv_n);
                }
            }
            row0 += tile;
        }
    };

    if threads <= 1 {
        run_block(elements, batch_size, &mut out);
        return out;
    }

    let rows_per = batch_size.div_ceil(threads);
    let run_block_ref = &run_block;
    std::thread::scope(|scope| {
        let mut rest: &mut [(f64, f64)] = out.as_mut_slice();
        let mut row0 = 0usize;
        while row0 < batch_size {
            let rows = rows_per.min(batch_size - row0);
            let (blk, tail) = rest.split_at_mut(rows * n);
            rest = tail;
            let base = row0 * n;
            let src = &elements[base..base + rows * n];
            scope.spawn(move || {
                run_block_ref(src, rows, blk);
            });
            row0 += rows;
        }
    });
    out
}

/// Per-row transform engine for a batch of equal-length rows. Built once and
/// shared across rows: power-of-two lengths use radix-2, smooth composite lengths
/// use mixed-radix (caching the roots table), and everything else uses a cached
/// Bluestein plan.
enum BatchFftPlan {
    Pow2(Radix2Plan),
    Mixed(Vec<(f64, f64)>),
    Bluestein(BluesteinPlan),
}

impl BatchFftPlan {
    fn new(n: usize, inverse: bool) -> Self {
        if n <= 1 || n.is_power_of_two() {
            // Cache the radix-2 plan (bit-reversal + per-stage twiddles) ONCE for the
            // whole batch. The dataless marker previously delegated to `fft_1d_into`,
            // which rebuilt the twiddle table — including per-stage `sin_cos` — and the
            // bit-reversal on EVERY row (e.g. ~16K redundant `sin_cos` over a 2048x256
            // batch). `Radix2Plan` derives twiddles from the IDENTICAL first-order
            // recurrence, so `apply_into` is bit-for-bit equal to the per-row rebuild.
            BatchFftPlan::Pow2(Radix2Plan::new(n.max(1), inverse))
        } else if is_mixed_radix_smooth(n) {
            BatchFftPlan::Mixed(precompute_twiddles(n, inverse))
        } else {
            BatchFftPlan::Bluestein(BluesteinPlan::new(n, inverse))
        }
    }

    fn work_len(&self, n: usize) -> usize {
        match self {
            BatchFftPlan::Pow2(_) | BatchFftPlan::Mixed(_) => n,
            BatchFftPlan::Bluestein(plan) => plan.m,
        }
    }

    /// The cached radix-2 plan for power-of-two lengths (else `None`). Lets the
    /// real-inverse-FFT path reuse the already-built bit-reversal + twiddles for the
    /// SoA kernel instead of rebuilding them.
    fn as_pow2(&self) -> Option<&Radix2Plan> {
        match self {
            BatchFftPlan::Pow2(plan) => Some(plan),
            _ => None,
        }
    }

    fn apply_into(
        &self,
        input: &[(f64, f64)],
        scratch: &mut BluesteinScratch,
        mixed_scratch: &mut Vec<(f64, f64)>,
        inverse: bool,
        output: &mut Vec<(f64, f64)>,
    ) {
        match self {
            BatchFftPlan::Pow2(plan) => {
                // The radix-2 direction is baked into the cached plan at build time;
                // the batch engine always builds and applies with the same `inverse`.
                debug_assert_eq!(plan.inverse, inverse);
                plan.apply_into(input, output);
            }
            BatchFftPlan::Mixed(roots) => {
                mixed_radix_into(input, roots, inverse, output, mixed_scratch)
            }
            BatchFftPlan::Bluestein(plan) => plan.apply_into(input, scratch, output),
        }
    }
}

/// Transform every length-`n` row of `elements` along the last axis into a dense
/// `(re, im)` output buffer. Rows are independent, so for large batches the work
/// is split across threads — each thread owns its own per-row Bluestein scratch
/// and buffer, and writes a disjoint slice of the output. The Bluestein plan is
/// built once and shared immutably. The result is bit-identical to the serial
/// path (no cross-row interaction, no reordering of any butterfly). `inverse`
/// selects FFT vs IFFT.
fn transform_batches_dense(
    elements: &[(f64, f64)],
    n: usize,
    batch_size: usize,
    inverse: bool,
) -> Vec<(f64, f64)> {
    // Power-of-two batches go through the vectorized structure-of-arrays kernel:
    // transpose a row-block to `[index][row]` SoA, run the SAME radix-2 butterfly
    // schedule vertically over the batch (one scalar twiddle broadcast across all
    // rows -> the inner loop autovectorizes over contiguous rows), then transpose
    // back. This is bit-identical to the per-row `Radix2Plan` path (identical
    // twiddle recurrence + butterfly op order) but turns the log2(n) butterfly
    // passes into wide contiguous SIMD over the batch dimension. Gated on a
    // minimum batch so small/single FFTs (already near JAX parity) keep the cheap
    // per-row path and avoid the transpose round-trip.
    const POW2_VECTORIZED_MIN_BATCH: usize = 8; // A/B TOGGLE: MAX=HEAD baseline, 8=vectorized
    if n > 1 && n.is_power_of_two() && batch_size >= POW2_VECTORIZED_MIN_BATCH {
        return transform_batches_pow2_vectorized(elements, n, batch_size, inverse);
    }

    // Bluestein-eligible lengths (non-pow2, non-smooth-composite, i.e. large-prime
    // or rough) run the SoA Bluestein path: the chirp/kernel work and the two
    // internal pow2 convolution FFTs vectorize across rows via the proven flat
    // radix-2 kernel. Capped at `BLUESTEIN_SOA_MAX_M` so the tile buffers stay
    // cache-warm. Bit-identical to per-row `BluesteinPlan::apply_into`.
    const BLUESTEIN_VECTORIZED_MIN_BATCH: usize = 8;
    if n > 1
        && batch_size >= BLUESTEIN_VECTORIZED_MIN_BATCH
        && !n.is_power_of_two()
        && !is_mixed_radix_smooth(n)
        && (2 * n - 1).next_power_of_two() <= BLUESTEIN_SOA_MAX_M
    {
        let bplan = BluesteinPlan::new(n, inverse);
        return transform_batches_bluestein_vectorized(&bplan, elements, n, batch_size);
    }

    // Smooth-composite batches (e.g. 1000 = 2^3*5^3) are the remaining FFT gap:
    // Bluestein is avoided, but the current recursive mixed-radix path is row-wise
    // and hard to vectorize. The flat iterative SoA candidate below was MEASURED (RCH
    // A/B 2026-06-21) at 0.15x (6.5x SLOWER) and is DISABLED via MAX_N=0 — this gate
    // is now inert; smooth composites keep the proven recursive per-row path. The block
    // is retained (with its kernel + correctness tests) as a documented no-ship.
    const MIXED_RADIX_ITERATIVE_SOA_MIN_BATCH: usize = 8;
    if n > 1
        && batch_size >= MIXED_RADIX_ITERATIVE_SOA_MIN_BATCH
        && is_mixed_radix_smooth(n)
        && mixed_radix_iterative_soa_gate_allows_n(n)
        && batch_size.saturating_mul(n) <= MIXED_RADIX_ITERATIVE_SOA_MAX_ELEMS
    {
        let roots = precompute_twiddles(n, inverse);
        return transform_batches_mixed_radix_iterative_soa(
            elements, n, batch_size, inverse, &roots,
        );
    }

    // Shared, immutable per-row plan (built once): radix-2 / mixed-radix / Bluestein.
    let plan = BatchFftPlan::new(n, inverse);
    let total = batch_size * n;
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

    // Only fan out when there is enough work to amortize thread spawn; dense
    // storage makes the extract/output trivial, so the per-row transform is the
    // dominant cost and parallelizes cleanly across rows.
    // NO-SHIP (2026-06-21, CrimsonOtter): lowering this to 1<<16 so fft_batch_128x1000
    // (128_000 elems) threads measured 5.82ms (32 threads) / 7.40ms (work-capped ~7
    // threads) vs 2.46ms single-thread — a 2.4-3x REGRESSION under swarm contention. The
    // ~40us per-row 1000-pt FFT is too short to amortize thread overhead when cores are
    // contended. JAX threads this on idle cores; that win is unmeasurable on this shared
    // host. Retry only on a quiesced host. See the murmw ledger entry.
    const PARALLEL_MIN_ELEMS: usize = 1 << 18; // 262_144
    let threads = if total >= PARALLEL_MIN_ELEMS && batch_size > 1 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(batch_size)
    } else {
        1
    };

    if threads <= 1 {
        let mut scratch = BluesteinScratch::default();
        let mut mixed_scratch = Vec::new();
        let mut buf = Vec::with_capacity(n);
        for batch in 0..batch_size {
            let s = batch * n;
            plan.apply_into(
                &elements[s..s + n],
                &mut scratch,
                &mut mixed_scratch,
                inverse,
                &mut buf,
            );
            out[s..s + n].copy_from_slice(&buf);
        }
        return out;
    }

    let rows_per = batch_size.div_ceil(threads);
    let plan_ref = &plan;
    std::thread::scope(|scope| {
        let mut rest: &mut [(f64, f64)] = out.as_mut_slice();
        let mut row0 = 0usize;
        while row0 < batch_size {
            let rows = rows_per.min(batch_size - row0);
            let (blk, tail) = rest.split_at_mut(rows * n);
            rest = tail;
            let base = row0 * n;
            scope.spawn(move || {
                let mut scratch = BluesteinScratch::default();
                let mut mixed_scratch = Vec::new();
                let mut buf = Vec::with_capacity(n);
                for r in 0..rows {
                    let gs = base + r * n;
                    let src = &elements[gs..gs + n];
                    plan_ref.apply_into(src, &mut scratch, &mut mixed_scratch, inverse, &mut buf);
                    blk[r * n..r * n + n].copy_from_slice(&buf);
                }
            });
            row0 += rows;
        }
    });
    out
}

/// One radix-2 butterfly stage applied across a contiguous block of `batch`
/// rows that share the same twiddle factor. `even_*`/`odd_*` are the real and
/// imaginary lanes of the two butterfly halves (each `batch` elements long, all
/// four disjoint and contiguous). The arithmetic is identical, op-for-op, to the
/// scalar `Radix2Plan::apply_into` butterfly, so the SoA result is bit-for-bit
/// equal to the per-row path. The four-way zip over equal-length slices with the
/// scalar twiddle broadcast is what the autovectorizer turns into wide FMA-free
/// SIMD over the batch dimension.
#[inline]
fn batch_butterfly_block(
    even_re: &mut [f64],
    even_im: &mut [f64],
    odd_re: &mut [f64],
    odd_im: &mut [f64],
    tw_re: f64,
    tw_im: f64,
) {
    for (((er, ei), or_), oi) in even_re
        .iter_mut()
        .zip(even_im.iter_mut())
        .zip(odd_re.iter_mut())
        .zip(odd_im.iter_mut())
    {
        let o_re = *or_;
        let o_im = *oi;
        let rot_re = o_re * tw_re - o_im * tw_im;
        let rot_im = o_re * tw_im + o_im * tw_re;
        let e_re = *er;
        let e_im = *ei;
        *er = e_re + rot_re;
        *ei = e_im + rot_im;
        *or_ = e_re - rot_re;
        *oi = e_im - rot_im;
    }
}

/// Run the radix-2 butterfly stages of a length-`n` FFT vertically over `w` rows
/// laid out as SoA `re[index*w + row]` / `im[...]` (already bit-reversed). For each
/// `(start, offset)` the two butterfly halves are the disjoint contiguous row-blocks
/// `[i_even, i_even+w)` and `[i_odd, i_odd+w)` (`i_odd > i_even` always, so
/// `split_at_mut` yields both). Op-for-op identical to `Radix2Plan::apply_into`'s
/// butterfly loop, so the result is bit-identical to the per-row path. Shared by the
/// full-complex (`vectorized_pow2_block`) and real (`vectorized_rfft_pow2_block`) SoA
/// kernels.
fn soa_radix2_butterfly_stages(plan: &Radix2Plan, w: usize, n: usize, re: &mut [f64], im: &mut [f64]) {
    let mut twiddle_base = 0usize;
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let stage_tw = &plan.twiddles[twiddle_base..twiddle_base + half];
        let mut start = 0usize;
        while start < n {
            for (offset, &(tw_re, tw_im)) in stage_tw.iter().enumerate() {
                let i_even = (start + offset) * w;
                let i_odd = (start + offset + half) * w;
                let (re_lo, re_hi) = re.split_at_mut(i_odd);
                let (im_lo, im_hi) = im.split_at_mut(i_odd);
                batch_butterfly_block(
                    &mut re_lo[i_even..i_even + w],
                    &mut im_lo[i_even..i_even + w],
                    &mut re_hi[..w],
                    &mut im_hi[..w],
                    tw_re,
                    tw_im,
                );
            }
            start += len;
        }
        twiddle_base += half;
        len *= 2;
    }
}

/// Vectorized radix-2 FFT over a block of `batch` length-`n` rows using a
/// structure-of-arrays layout. `elements`/`out` are the AoS `(re, im)` input and
/// output for this block (`batch * n` each, row-major). Scratch `re`/`im` hold the
/// transposed SoA data (`re[index * batch + row]`).
///
/// Bit-identical to running `plan.apply_into` on each row independently: same
/// bit-reversal, same precomputed twiddle table, same per-stage `start`/`offset`
/// traversal, same butterfly arithmetic, same `1/n` inverse scale.
#[allow(clippy::too_many_arguments)]
fn vectorized_pow2_block(
    plan: &Radix2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    n: usize,
    inverse: bool,
    re: &mut [f64],
    im: &mut [f64],
    out: &mut [(f64, f64)],
) {
    debug_assert_eq!(elements.len(), batch * n);
    debug_assert_eq!(out.len(), batch * n);
    debug_assert_eq!(re.len(), batch * n);
    debug_assert_eq!(im.len(), batch * n);
    let w = batch;

    // Transpose into SoA while applying the radix-2 bit-reversal to the index
    // axis: re[i*w + b] = elements[b*n + bit_reversed[i]].re (likewise im).
    let bitrev = &plan.bit_reversed;
    for b in 0..batch {
        let row = &elements[b * n..b * n + n];
        for (i, &src_idx) in bitrev.iter().enumerate() {
            let (sr, si) = row[src_idx];
            re[i * w + b] = sr;
            im[i * w + b] = si;
        }
    }

    // Butterfly stages, vertical over the batch.
    soa_radix2_butterfly_stages(plan, w, n, re, im);

    // Transpose back to AoS row-major, folding in the 1/n inverse scale (matches
    // `Radix2Plan::apply_into`'s post-loop `value.* *= inv_n`).
    if inverse {
        let inv_n = 1.0 / (n as f64);
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b] * inv_n, im[k * w + b] * inv_n);
            }
        }
    } else {
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b], im[k * w + b]);
            }
        }
    }
}

/// True for n = 4^k (k >= 1): a pure power of four, eligible for the radix-4 SoA kernel.
#[cfg(test)]
fn is_power_of_four(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros() % 2 == 0
}

/// Base-4 digit-reversal permutation for n = 4^k (the radix-4 analogue of bit-reversal).
#[cfg(test)]
fn digit_reverse_base4(n: usize) -> Vec<usize> {
    let digits = (n.trailing_zeros() / 2) as usize;
    (0..n)
        .map(|i| {
            let mut x = i;
            let mut rev = 0usize;
            for _ in 0..digits {
                rev = (rev << 2) | (x & 3);
                x >>= 2;
            }
            rev
        })
        .collect()
}

/// Radix-4 plan for pow-of-four lengths: base-4 digit reversal + per-stage `W_len^j`
/// twiddles (stage `len` = 4,16,...,n; `quarter` = len/4 twiddles each). `W_len^2j` and
/// `W_len^3j` are derived inline in the butterfly (one square + one mul) to keep the table
/// the same size as radix-2's.
#[cfg(test)]
struct Radix4Plan {
    digit_reversed: Vec<usize>,
    twiddles: Vec<(f64, f64)>,
    inverse: bool,
}

#[cfg(test)]
impl Radix4Plan {
    fn new(n: usize, inverse: bool) -> Self {
        debug_assert!(is_power_of_four(n));
        let digit_reversed = digit_reverse_base4(n);
        let mut twiddles = Vec::new();
        let mut len = 4usize;
        while len <= n {
            let quarter = len / 4;
            let angle = if inverse {
                2.0 * PI / (len as f64)
            } else {
                -2.0 * PI / (len as f64)
            };
            let (sin_step, cos_step) = angle.sin_cos();
            let (mut tr, mut ti) = (1.0, 0.0);
            for _ in 0..quarter {
                twiddles.push((tr, ti));
                let nr = tr * cos_step - ti * sin_step;
                let ni = tr * sin_step + ti * cos_step;
                tr = nr;
                ti = ni;
            }
            len *= 4;
        }
        Self {
            digit_reversed,
            twiddles,
            inverse,
        }
    }
}

/// Radix-4 DIT butterfly stages applied vertically across `w` SoA lanes. Halves the stage
/// count vs radix-2 (log4 vs log2), so it sweeps the SoA buffer fewer times — the dominant
/// cost per the `profile_soa_pow2_phases` measurement (butterflies were 57% of the kernel).
/// Tolerance-equal (not bit-identical) to the radix-2 path: a different, fewer-op schedule.
#[cfg(test)]
fn soa_radix4_butterfly_stages(plan: &Radix4Plan, w: usize, n: usize, re: &mut [f64], im: &mut [f64]) {
    let inv = plan.inverse;
    let mut tw_base = 0usize;
    let mut len = 4usize;
    while len <= n {
        let quarter = len / 4;
        let stage_tw = &plan.twiddles[tw_base..tw_base + quarter];
        let mut start = 0usize;
        while start < n {
            for (j, &(w1r, w1i)) in stage_tw.iter().enumerate() {
                // W_len^2j and W_len^3j from W_len^j.
                let w2r = w1r * w1r - w1i * w1i;
                let w2i = 2.0 * w1r * w1i;
                let w3r = w2r * w1r - w2i * w1i;
                let w3i = w2r * w1i + w2i * w1r;
                let i0 = (start + j) * w;
                let i1 = (start + j + quarter) * w;
                let i2 = (start + j + 2 * quarter) * w;
                let i3 = (start + j + 3 * quarter) * w;
                for lane in 0..w {
                    let ar = re[i0 + lane];
                    let ai = im[i0 + lane];
                    let (b0r, b0i) = (re[i1 + lane], im[i1 + lane]);
                    let (c0r, c0i) = (re[i2 + lane], im[i2 + lane]);
                    let (d0r, d0i) = (re[i3 + lane], im[i3 + lane]);
                    // Twiddle B,C,D by W^j, W^2j, W^3j.
                    let br = b0r * w1r - b0i * w1i;
                    let bi = b0r * w1i + b0i * w1r;
                    let cr = c0r * w2r - c0i * w2i;
                    let ci = c0r * w2i + c0i * w2r;
                    let dr = d0r * w3r - d0i * w3i;
                    let di = d0r * w3i + d0i * w3r;
                    // t0=A+C, u1=A-C, s2=B+D, s3=B-D.
                    let t0r = ar + cr;
                    let t0i = ai + ci;
                    let u1r = ar - cr;
                    let u1i = ai - ci;
                    let s2r = br + dr;
                    let s2i = bi + di;
                    let s3r = br - dr;
                    let s3i = bi - di;
                    re[i0 + lane] = t0r + s2r;
                    im[i0 + lane] = t0i + s2i;
                    re[i2 + lane] = t0r - s2r;
                    im[i2 + lane] = t0i - s2i;
                    // forward: X1 = u1 - i*s3, X3 = u1 + i*s3 (i*s3 = (-s3i, s3r));
                    // inverse swaps the two.
                    if inv {
                        re[i1 + lane] = u1r - s3i;
                        im[i1 + lane] = u1i + s3r;
                        re[i3 + lane] = u1r + s3i;
                        im[i3 + lane] = u1i - s3r;
                    } else {
                        re[i1 + lane] = u1r + s3i;
                        im[i1 + lane] = u1i - s3r;
                        re[i3 + lane] = u1r - s3i;
                        im[i3 + lane] = u1i + s3r;
                    }
                }
            }
            start += len;
        }
        tw_base += quarter;
        len *= 4;
    }
}

/// Radix-4 SoA FFT over a block of `batch` length-`n` (n = 4^k) rows. Same SoA framing as
/// `vectorized_pow2_block` (one transpose in, butterflies vertical over the batch, one
/// transpose out) but with the radix-4 stage schedule. Tolerance-equal to the radix-2 path.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn vectorized_pow4_block(
    plan: &Radix4Plan,
    elements: &[(f64, f64)],
    batch: usize,
    n: usize,
    inverse: bool,
    re: &mut [f64],
    im: &mut [f64],
    out: &mut [(f64, f64)],
) {
    let w = batch;
    let digitrev = &plan.digit_reversed;
    for b in 0..batch {
        let row = &elements[b * n..b * n + n];
        for (i, &src_idx) in digitrev.iter().enumerate() {
            let (sr, si) = row[src_idx];
            re[i * w + b] = sr;
            im[i * w + b] = si;
        }
    }
    soa_radix4_butterfly_stages(plan, w, n, re, im);
    if inverse {
        let inv_n = 1.0 / (n as f64);
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b] * inv_n, im[k * w + b] * inv_n);
            }
        }
    } else {
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b], im[k * w + b]);
            }
        }
    }
}

/// Cache-blocked tile width (rows processed together in one SoA pass). Kept small
/// so the SoA scratch (`TILE * n` complex split into re/im) stays L1-resident:
/// a full-batch transpose would make every butterfly stage stream the entire
/// batch from RAM (memory-bound, ~2x slower than the L1-resident per-row path),
/// whereas an 8-row tile of length 256 is 8*256*8*2 = 32 KiB — the butterflies
/// run vertically over the tile (SIMD across rows) while staying in cache.
const POW2_TILE_ROWS: usize = 8;

/// Run the vectorized SoA kernel over `batch` rows in L1-resident tiles of
/// `POW2_TILE_ROWS` rows, reusing one scratch pair. Bit-identical to per-row
/// `Radix2Plan` (each tile is independent and each row's op order is unchanged).
fn vectorized_pow2_tiled(
    plan: &Radix2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    n: usize,
    inverse: bool,
    out: &mut [(f64, f64)],
) {
    let cap = POW2_TILE_ROWS * n;
    let mut re = vec![0.0f64; cap];
    let mut im = vec![0.0f64; cap];
    let mut row0 = 0usize;
    while row0 < batch {
        let rows = POW2_TILE_ROWS.min(batch - row0);
        let s = row0 * n;
        let e = s + rows * n;
        vectorized_pow2_block(
            plan,
            &elements[s..e],
            rows,
            n,
            inverse,
            &mut re[..rows * n],
            &mut im[..rows * n],
            &mut out[s..e],
        );
        row0 += rows;
    }
}

/// Batched power-of-two FFT/IFFT via the cache-blocked vectorized SoA kernel.
/// Builds one shared `Radix2Plan`, then fans row-chunks across threads for large
/// batches; each thread tiles its chunk. Bit-identical to `transform_batches_dense`'s
/// per-row `Radix2Plan` path.
fn transform_batches_pow2_vectorized(
    elements: &[(f64, f64)],
    n: usize,
    batch_size: usize,
    inverse: bool,
) -> Vec<(f64, f64)> {
    let plan = Radix2Plan::new(n, inverse);
    let total = batch_size * n;
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

    const PARALLEL_MIN_ELEMS: usize = 1 << 18; // 262_144
    let threads = if total >= PARALLEL_MIN_ELEMS && batch_size > 1 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(batch_size)
    } else {
        1
    };

    if threads <= 1 {
        vectorized_pow2_tiled(&plan, elements, batch_size, n, inverse, &mut out);
        return out;
    }

    let rows_per = batch_size.div_ceil(threads);
    let plan_ref = &plan;
    std::thread::scope(|scope| {
        let mut rest: &mut [(f64, f64)] = out.as_mut_slice();
        let mut row0 = 0usize;
        while row0 < batch_size {
            let rows = rows_per.min(batch_size - row0);
            let (blk, tail) = rest.split_at_mut(rows * n);
            rest = tail;
            let base = row0 * n;
            let src = &elements[base..base + rows * n];
            scope.spawn(move || {
                vectorized_pow2_tiled(plan_ref, src, rows, n, inverse, blk);
            });
            row0 += rows;
        }
    });
    out
}

// ── FFT ──────────────────────────────────────────────────────────────

/// Compute the 1D FFT along the last axis.
///
/// Input: rank-1+ tensor of real or complex values.
/// Output: complex tensor with the same shape.
pub(crate) fn eval_fft(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Fft;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (shape, in_dtype, elements) = extract_tensor_complex(primitive, &inputs[0])?;
    let out_dtype = complex_dtype_for(in_dtype);

    let n = last_axis_len(primitive, &shape)?;
    if n == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "FFT requires last dimension > 0".to_owned(),
        });
    }
    let batch_size = elements.len() / n;

    // Dense complex output via the threaded per-row transform: extract already
    // borrowed a packed slice (no per-`Literal` conversion), and the output is
    // built straight into dense `(re, im)` storage. Rows are independent so large
    // batches fan out across threads — bit-identical to the serial path.
    let out_values = transform_batches_dense(&elements, n, batch_size, false);
    let tensor = TensorValue::new_complex_values(out_dtype, shape, out_values)
        .map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

// ── IFFT ─────────────────────────────────────────────────────────────

/// Compute the 1D inverse FFT along the last axis.
///
/// Input: rank-1+ tensor of complex values.
/// Output: complex tensor with the same shape.
pub(crate) fn eval_ifft(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Ifft;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (shape, in_dtype, elements) = extract_tensor_complex(primitive, &inputs[0])?;
    if !is_complex_dtype(in_dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "IFFT expects complex-valued input, got real tensor".to_owned(),
        });
    }
    let out_dtype = complex_dtype_for(in_dtype);

    let n = last_axis_len(primitive, &shape)?;
    if n == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "IFFT requires last dimension > 0".to_owned(),
        });
    }
    let batch_size = elements.len() / n;

    // Dense complex output via the threaded per-row inverse transform (see eval_fft).
    let out_values = transform_batches_dense(&elements, n, batch_size, true);
    let tensor = TensorValue::new_complex_values(out_dtype, shape, out_values)
        .map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

// ── RFFT ─────────────────────────────────────────────────────────────

/// Compute the 1D real-to-complex FFT along the last axis.
///
/// Input: rank-1+ tensor of real values.
/// Output: complex tensor where the last dimension is `fft_length / 2 + 1`
///         (exploiting Hermitian symmetry of the DFT of a real signal).
///
/// The input is zero-padded or truncated to `fft_length` before the transform.
pub(crate) fn eval_rfft(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Rfft;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (shape, in_dtype, elements) = extract_tensor_complex(primitive, &inputs[0])?;
    if is_complex_dtype(in_dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "RFFT expects real-valued input, got complex tensor".to_owned(),
        });
    }
    let out_dtype = complex_dtype_for(in_dtype);

    let input_last = last_axis_len(primitive, &shape)?;
    if input_last == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "RFFT requires last dimension > 0".to_owned(),
        });
    }

    // Parse fft_length from params (defaults to input last dim)
    let fft_length = parse_optional_fft_length(primitive, params, input_last)?;

    if fft_length == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "fft_length must be > 0".to_owned(),
        });
    }

    let out_last = fft_length / 2 + 1;
    let batch_size = elements.len() / input_last;
    checked_fft_len(primitive, fft_length, "RFFT fft_length")?;
    let output_count = checked_output_element_count(primitive, batch_size, out_last)?;

    // Build output shape before allocating transform buffers so hostile lengths fail closed.
    let mut out_dims = shape.dims;
    replace_last_axis_len(primitive, &mut out_dims, out_last)?;
    let out_shape = Shape { dims: out_dims };

    let copy_len = fft_length.min(input_last);
    // Non-power-of-two transform length: build the per-row plan once and reuse it
    // across every row. BatchFftPlan picks mixed-radix for smooth lengths (e.g.
    // 1000 = 2³·5³) and Bluestein for large-prime lengths.
    let plan = (fft_length > 1 && !fft_length.is_power_of_two())
        .then(|| BatchFftPlan::new(fft_length, false));
    let real_plan = (fft_length >= 2 && fft_length.is_power_of_two())
        .then(|| RealRfftPower2Plan::new(fft_length));

    // Dense complex output, one row-block per thread for large batches. Each row
    // is independent (pad -> transform -> keep Hermitian half), so this is
    // bit-identical to the serial path.
    let mut out_values: Vec<(f64, f64)> = vec![(0.0, 0.0); output_count];
    const PARALLEL_MIN_ELEMS: usize = 1 << 18; // 262_144
    let transform_work = batch_size.saturating_mul(
        plan.as_ref()
            .map_or(fft_length, |plan| plan.work_len(fft_length)),
    );
    let threads = if transform_work >= PARALLEL_MIN_ELEMS && batch_size > 1 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(batch_size)
    } else {
        1
    };

    if threads <= 1 {
        rfft_rows_into(
            &elements,
            plan.as_ref(),
            real_plan.as_ref(),
            fft_length,
            input_last,
            copy_len,
            out_last,
            0,
            batch_size,
            &mut out_values,
        );
    } else {
        let mut rows_per = batch_size.div_ceil(threads);
        if real_plan.is_none() && rows_per % 2 != 0 {
            rows_per += 1;
        }
        let plan_ref = plan.as_ref();
        let real_plan_ref = real_plan.as_ref();
        let elements_ref = &elements;
        std::thread::scope(|scope| {
            let mut rest: &mut [(f64, f64)] = out_values.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < batch_size {
                let rows = rows_per.min(batch_size - row0);
                let (blk, tail) = rest.split_at_mut(rows * out_last);
                rest = tail;
                scope.spawn(move || {
                    rfft_rows_into(
                        elements_ref,
                        plan_ref,
                        real_plan_ref,
                        fft_length,
                        input_last,
                        copy_len,
                        out_last,
                        row0,
                        rows,
                        blk,
                    );
                });
                row0 += rows;
            }
        });
    }

    let tensor = TensorValue::new_complex_values(out_dtype, out_shape, out_values)
        .map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

/// Vectorized SoA real FFT for a block of `batch` rows (pow2 `fft_length`). Mirrors
/// `RealRfftPower2Plan::apply_into` op-for-op but runs all three stages — pack,
/// half-length complex FFT, Hermitian recombination — vertically over the batch, so
/// it is bit-identical to the per-row path. `re`/`im` are SoA scratch of length
/// `batch * half_len`; `out_blk` holds `batch * out_last` complex outputs row-major.
#[allow(clippy::too_many_arguments)]
fn vectorized_rfft_pow2_block(
    real_plan: &RealRfftPower2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    input_last: usize,
    copy_len: usize,
    out_last: usize,
    re: &mut [f64],
    im: &mut [f64],
    out_blk: &mut [(f64, f64)],
) {
    let half_len = real_plan.half_len;
    let w = batch;
    debug_assert_eq!(out_last, half_len + 1);
    debug_assert_eq!(re.len(), batch * half_len);
    debug_assert_eq!(im.len(), batch * half_len);
    debug_assert_eq!(out_blk.len(), batch * out_last);

    // Pack each row's real samples (even -> re, odd -> im of the half-length complex
    // signal) into SoA while applying the half FFT's bit-reversal to the index axis.
    let bitrev = &real_plan.half_fft.bit_reversed;
    for b in 0..batch {
        let row = &elements[b * input_last..b * input_last + input_last];
        for (i, &idx) in bitrev.iter().enumerate() {
            let even = 2 * idx;
            let odd = even + 1;
            let even_re = if even < copy_len { row[even].0 } else { 0.0 };
            let odd_re = if odd < copy_len { row[odd].0 } else { 0.0 };
            re[i * w + b] = even_re;
            im[i * w + b] = odd_re;
        }
    }

    // Half-length complex FFT, vertical over the batch. Now re/im hold `transformed`
    // in SoA [k][row] layout (k in 0..half_len).
    soa_radix2_butterfly_stages(&real_plan.half_fft, w, half_len, re, im);

    // Hermitian recombination — identical arithmetic to the per-row path. DC/Nyquist
    // come from transformed[0]; the rest from the conjugate-symmetric pair (k, N/2-k).
    for b in 0..batch {
        let z0_re = re[b];
        let z0_im = im[b];
        out_blk[b * out_last] = (z0_re + z0_im, 0.0);
        out_blk[b * out_last + half_len] = (z0_re - z0_im, 0.0);
    }
    for k in 1..half_len {
        let (cos_a, sin_a) = real_plan.twiddles[k - 1];
        let kb = k * w;
        let mkb = (half_len - k) * w;
        for b in 0..batch {
            let a_re = re[kb + b];
            let a_im = im[kb + b];
            let b_re = re[mkb + b];
            let b_im = im[mkb + b];
            let avg_re = 0.5 * (a_re + b_re);
            let avg_im = 0.5 * (a_im - b_im);
            let diff_re = 0.5 * (a_re - b_re);
            let diff_im = 0.5 * (a_im + b_im);
            let rotated_re = diff_re * cos_a - diff_im * sin_a;
            let rotated_im = diff_re * sin_a + diff_im * cos_a;
            out_blk[b * out_last + k] = (avg_re + rotated_im, avg_im - rotated_re);
        }
    }
}

/// Run the SoA real-FFT kernel over `batch` rows in L1-resident tiles of
/// `POW2_TILE_ROWS` rows, reusing one scratch pair. `elements` is the real-input
/// slice for exactly these `batch` rows; `out` holds `batch * out_last` outputs.
/// Bit-identical to per-row `RealRfftPower2Plan` (each tile/row is independent).
#[allow(clippy::too_many_arguments)]
fn vectorized_rfft_pow2_tiled(
    real_plan: &RealRfftPower2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    input_last: usize,
    copy_len: usize,
    out_last: usize,
    out: &mut [(f64, f64)],
) {
    let half_len = real_plan.half_len;
    let cap = POW2_TILE_ROWS * half_len;
    let mut re = vec![0.0f64; cap];
    let mut im = vec![0.0f64; cap];
    let mut row0 = 0usize;
    while row0 < batch {
        let rows = POW2_TILE_ROWS.min(batch - row0);
        let in_s = row0 * input_last;
        let out_s = row0 * out_last;
        vectorized_rfft_pow2_block(
            real_plan,
            &elements[in_s..in_s + rows * input_last],
            rows,
            input_last,
            copy_len,
            out_last,
            &mut re[..rows * half_len],
            &mut im[..rows * half_len],
            &mut out[out_s..out_s + rows * out_last],
        );
        row0 += rows;
    }
}

/// Transform `rows` real-input rows (starting at `row_start`) into the dense
/// complex `out_blk` for RFFT: zero-pad/truncate each length-`input_last` row to
/// `fft_length`, run the forward transform, and keep the first `out_last`
/// Hermitian-half outputs. Owns its per-row scratch so it is callable serially or
/// from one thread per row-block.
#[allow(clippy::too_many_arguments)]
fn rfft_rows_into(
    elements: &[(f64, f64)],
    plan: Option<&BatchFftPlan>,
    real_plan: Option<&RealRfftPower2Plan>,
    fft_length: usize,
    input_last: usize,
    copy_len: usize,
    out_last: usize,
    row_start: usize,
    rows: usize,
    out_blk: &mut [(f64, f64)],
) {
    if let Some(real_plan) = real_plan {
        debug_assert_eq!(real_plan.fft_length, fft_length);
        // Above a minimum row count, run the cache-blocked SoA real-FFT kernel
        // (pack + half-length FFT + recombine vectorized vertically over rows) —
        // bit-identical to the per-row path, same stable single-thread win as the
        // full-complex SoA path. Tiny blocks keep the cheap per-row scratch loop.
        const RFFT_VECTORIZED_MIN_ROWS: usize = 8;
        if rows >= RFFT_VECTORIZED_MIN_ROWS {
            let base = row_start * input_last;
            vectorized_rfft_pow2_tiled(
                real_plan,
                &elements[base..base + rows * input_last],
                rows,
                input_last,
                copy_len,
                out_last,
                out_blk,
            );
            return;
        }
        let mut packed = Vec::with_capacity(real_plan.half_len);
        let mut transformed = Vec::with_capacity(real_plan.half_len);
        for r in 0..rows {
            let start = (row_start + r) * input_last;
            let batch_slice = &elements[start..start + input_last];
            real_plan.apply_into(
                batch_slice,
                copy_len,
                &mut packed,
                &mut transformed,
                &mut out_blk[r * out_last..r * out_last + out_last],
            );
        }
        return;
    }

    // Non-power-of-two (Bluestein / mixed-radix) path. The transform is full-complex
    // regardless of the real input, so we PACK ROWS IN PAIRS into one complex signal
    // z = x_a + i·x_b and run a single length-`fft_length` transform per pair, halving
    // the number of (dominant-cost) transforms. Each real spectrum is recovered from
    // the conjugate symmetry of the packed transform Z (N = fft_length):
    //   X_a[k] = (Z[k] + conj(Z[(N−k) mod N])) / 2,
    //   X_b[k] = (Z[k] − conj(Z[(N−k) mod N])) / (2i).
    // A leftover odd row is transformed directly. This is tolerance-equal to the
    // per-row path (only extra pack/unpack rounding); non-pow2 transforms are not
    // bit-frozen (golden FFT digests are pow2-only). Pairing stays within this
    // row-block, so threading partition / row order / determinism are unchanged.
    let n = fft_length;
    let mut padded = vec![(0.0, 0.0); fft_length];
    let mut transformed = Vec::with_capacity(fft_length);
    let mut scratch = BluesteinScratch::default();
    let mut mixed_scratch = Vec::new();

    let mut r = 0;
    while r + 1 < rows {
        let start_a = (row_start + r) * input_last;
        let start_b = (row_start + r + 1) * input_last;
        for j in 0..copy_len {
            // Real input ⇒ imaginary parts are zero; pack row a → real, row b → imag.
            padded[j] = (elements[start_a + j].0, elements[start_b + j].0);
        }
        padded[copy_len..].fill((0.0, 0.0));
        match plan {
            Some(plan) => plan.apply_into(
                &padded,
                &mut scratch,
                &mut mixed_scratch,
                false,
                &mut transformed,
            ),
            None => fft_1d_into(&padded, &mut transformed),
        }

        let (blk_a, blk_b) = out_blk[r * out_last..(r + 2) * out_last].split_at_mut(out_last);
        for k in 0..out_last {
            let nk = if k == 0 { 0 } else { n - k };
            let (zr, zi) = transformed[k];
            let (zr_nk, zi_nk) = transformed[nk];
            // conj(Z[nk]) = (zr_nk, −zi_nk).
            blk_a[k] = (0.5 * (zr + zr_nk), 0.5 * (zi - zi_nk));
            // (Z[k] − conj(Z[nk]))/(2i): d = (dr, di) ⇒ d/(2i) = (di/2, −dr/2).
            let dr = zr - zr_nk;
            let di = zi + zi_nk;
            blk_b[k] = (0.5 * di, -0.5 * dr);
        }
        r += 2;
    }
    if r < rows {
        let start = (row_start + r) * input_last;
        padded[..copy_len].copy_from_slice(&elements[start..start + copy_len]);
        padded[copy_len..].fill((0.0, 0.0));
        match plan {
            Some(plan) => plan.apply_into(
                &padded,
                &mut scratch,
                &mut mixed_scratch,
                false,
                &mut transformed,
            ),
            None => fft_1d_into(&padded, &mut transformed),
        }
        out_blk[r * out_last..r * out_last + out_last].copy_from_slice(&transformed[..out_last]);
    }
}

// ── IRFFT ────────────────────────────────────────────────────────────

/// Compute the 1D complex-to-real inverse FFT along the last axis.
///
/// Input: rank-1+ complex tensor with last dim = n/2 + 1 (half-spectrum).
/// Output: real tensor with last dim = fft_length.
///
/// Reconstructs the full spectrum from Hermitian symmetry, computes IDFT,
/// and takes the real part.
/// Vectorized SoA inverse real FFT for a block of `batch` rows (pow2 `fft_length`).
/// Fuses the Hermitian-spectrum reconstruction into the SoA transpose-in (so no
/// extra full-spectrum buffer is needed — only the `re`/`im` scratch, kept
/// L1-resident), runs the inverse radix-2 butterflies vertically over the batch,
/// then extracts the real part with the `1/n` scale. Bit-identical to the per-row
/// `irfft_rows_f64_into` path: same reconstruction, same `Radix2Plan(n, true)`
/// butterfly order, same `1/n` scaling, same real extraction.
#[allow(clippy::too_many_arguments)]
fn vectorized_irfft_pow2_block(
    plan: &Radix2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    fft_length: usize,
    input_last: usize,
    copy_len: usize,
    re: &mut [f64],
    im: &mut [f64],
    out_blk: &mut [f64],
) {
    let w = batch;
    let n = fft_length;
    debug_assert_eq!(re.len(), batch * n);
    debug_assert_eq!(im.len(), batch * n);
    debug_assert_eq!(out_blk.len(), batch * n);

    // Transpose into SoA with the radix-2 bit-reversal, reconstructing the full
    // Hermitian spectrum value at source index `j` on the fly:
    //   j < copy_len                                  -> half_spectrum[j]
    //   j >= input_last and (n-j) < input_last        -> conj(half_spectrum[n-j])
    //   otherwise                                     -> 0
    // (identical to `irfft_rows_f64_into`'s full-spectrum fill, where
    //  full[..copy_len] = hs, the tail is zero-filled, and the upper half mirrors
    //  the conjugate.)
    let bitrev = &plan.bit_reversed;
    for b in 0..batch {
        let hs = &elements[b * input_last..b * input_last + input_last];
        for (i, &j) in bitrev.iter().enumerate() {
            let (vr, vi) = if j < copy_len {
                hs[j]
            } else if j >= input_last {
                let m = n - j;
                if m < input_last {
                    let (r, im2) = hs[m];
                    (r, -im2)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
            re[i * w + b] = vr;
            im[i * w + b] = vi;
        }
    }

    soa_radix2_butterfly_stages(plan, w, n, re, im);

    // Real part with the inverse 1/n scale (matches `Radix2Plan::apply_into`'s
    // post-loop `value.* *= inv_n`, then irfft taking `.re`).
    let inv_n = 1.0 / (n as f64);
    for b in 0..batch {
        let dst = &mut out_blk[b * n..b * n + n];
        for (k, slot) in dst.iter_mut().enumerate() {
            *slot = re[k * w + b] * inv_n;
        }
    }
}

/// Run the SoA inverse real-FFT kernel over `batch` rows in L1-resident tiles of
/// `POW2_TILE_ROWS` rows, reusing one scratch pair. `elements` is the half-spectrum
/// slice for exactly these `batch` rows; `out` holds `batch * fft_length` reals.
#[allow(clippy::too_many_arguments)]
fn vectorized_irfft_pow2_tiled(
    plan: &Radix2Plan,
    elements: &[(f64, f64)],
    batch: usize,
    fft_length: usize,
    input_last: usize,
    copy_len: usize,
    out: &mut [f64],
) {
    let n = fft_length;
    let cap = POW2_TILE_ROWS * n;
    let mut re = vec![0.0f64; cap];
    let mut im = vec![0.0f64; cap];
    let mut row0 = 0usize;
    while row0 < batch {
        let rows = POW2_TILE_ROWS.min(batch - row0);
        let in_s = row0 * input_last;
        let out_s = row0 * n;
        vectorized_irfft_pow2_block(
            plan,
            &elements[in_s..in_s + rows * input_last],
            rows,
            fft_length,
            input_last,
            copy_len,
            &mut re[..rows * n],
            &mut im[..rows * n],
            &mut out[out_s..out_s + rows * n],
        );
        row0 += rows;
    }
}

/// Reconstruct the Hermitian-symmetric full spectrum and inverse-transform each
/// of `rows` half-spectra (starting at `row_start`) into the dense real (`f64`)
/// output block. Each row is independent and computed in the exact same order as
/// the serial path, so this is bit-identical (and safe to run one thread per
/// row-block). Owns its per-row scratch.
#[allow(clippy::too_many_arguments)]
fn irfft_rows_f64_into(
    elements: &[(f64, f64)],
    plan: Option<&BatchFftPlan>,
    fft_length: usize,
    input_last: usize,
    copy_len: usize,
    row_start: usize,
    rows: usize,
    out_blk: &mut [f64],
) {
    // Above a minimum row count, pow2 inverse real FFT runs the cache-blocked SoA
    // kernel (Hermitian reconstruct + inverse FFT + real extract, vectorized
    // vertically over rows) — bit-identical to the per-row path, same stable
    // single-thread win as the forward SoA paths. Reuses the cached pow2 plan.
    const IRFFT_VECTORIZED_MIN_ROWS: usize = 8;
    if rows >= IRFFT_VECTORIZED_MIN_ROWS
        && let Some(r2) = plan.and_then(|p| p.as_pow2())
    {
        let base = row_start * input_last;
        vectorized_irfft_pow2_tiled(
            r2,
            &elements[base..base + rows * input_last],
            rows,
            fft_length,
            input_last,
            copy_len,
            out_blk,
        );
        return;
    }

    let mut full = vec![(0.0, 0.0); fft_length];
    let mut transformed = Vec::with_capacity(fft_length);
    let mut scratch = BluesteinScratch::default();
    let mut mixed_scratch = Vec::new();
    for r in 0..rows {
        let start = (row_start + r) * input_last;
        let half_spectrum = &elements[start..start + input_last];

        full[..copy_len].copy_from_slice(&half_spectrum[..copy_len]);
        full[copy_len..].fill((0.0, 0.0));
        for (k, slot) in full
            .iter_mut()
            .enumerate()
            .take(fft_length)
            .skip(input_last)
        {
            let mirror = fft_length - k;
            if mirror < input_last {
                let (re, im) = half_spectrum[mirror];
                *slot = (re, -im);
            }
        }

        match plan {
            Some(plan) => plan.apply_into(
                &full,
                &mut scratch,
                &mut mixed_scratch,
                true,
                &mut transformed,
            ),
            None => ifft_1d_into(&full, &mut transformed),
        }

        let dst = &mut out_blk[r * fft_length..r * fft_length + fft_length];
        for (o, &(re, _)) in dst.iter_mut().zip(transformed.iter()) {
            *o = re;
        }
    }
}

pub(crate) fn eval_irfft(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Irfft;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (shape, in_dtype, elements) = extract_tensor_complex(primitive, &inputs[0])?;
    if !is_complex_dtype(in_dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "IRFFT expects complex-valued input, got real tensor".to_owned(),
        });
    }
    let out_dtype = real_dtype_for(in_dtype);

    let input_last = last_axis_len(primitive, &shape)?;
    if input_last == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "IRFFT requires last dimension > 0".to_owned(),
        });
    }

    // Parse fft_length from params. Default: (input_last - 1) * 2
    let default_fft_length = input_last.saturating_sub(1).saturating_mul(2);
    let fft_length = parse_optional_fft_length(primitive, params, default_fft_length)?;

    if fft_length == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "fft_length must be > 0".to_owned(),
        });
    }

    let batch_size = elements.len() / input_last;
    checked_fft_len(primitive, fft_length, "IRFFT fft_length")?;

    // Build output shape before allocating transform buffers so hostile lengths fail closed.
    let mut out_dims = shape.dims;
    replace_last_axis_len(primitive, &mut out_dims, fft_length)?;
    let out_shape = Shape { dims: out_dims };
    let output_count = checked_output_element_count(primitive, batch_size, fft_length)?;

    let copy_len = fft_length.min(input_last);
    // Build the inverse per-row plan ONCE for the whole batch and reuse it across
    // rows. Pow2 lengths get the cached `Radix2Plan` (bit-reversal + twiddles) instead
    // of `ifft_1d_into` rebuilding them every row (the same per-row-rebuild fix already
    // applied to forward FFT); mixed-radix for smooth lengths and Bluestein for
    // large-prime lengths. Bit-identical: the cached plan uses the identical recurrence
    // and 1/n scaling as the per-row `ifft_1d_into`.
    let plan = (fft_length > 1).then(|| BatchFftPlan::new(fft_length, true));

    // Dense + threaded fast path for F64 (Complex128 input) AND F32 (Complex64
    // input — JAX's default-precision inverse real FFT, previously stuck on the
    // serial boxed path). Rows are independent (Hermitian reconstruct + inverse
    // transform + real extraction), so large batches fan out across threads into a
    // dense f64 buffer; F32 then casts each value down (`v as f32` ==
    // make_real_literal(re, F32) = from_f32(re as f32)). Bit-identical to the serial
    // path: irfft_rows_f64_into runs the identical per-row algorithm in the same order.
    if matches!(out_dtype, DType::F64 | DType::F32) {
        let mut out = vec![0.0f64; output_count];
        const IRFFT_PARALLEL_MIN_ELEMS: usize = 1 << 17; // 131_072
        let threads = if output_count >= IRFFT_PARALLEL_MIN_ELEMS && batch_size > 1 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .min(batch_size)
        } else {
            1
        };
        if threads <= 1 {
            irfft_rows_f64_into(
                &elements,
                plan.as_ref(),
                fft_length,
                input_last,
                copy_len,
                0,
                batch_size,
                &mut out,
            );
        } else {
            let rows_per = batch_size.div_ceil(threads);
            let plan_ref = plan.as_ref();
            let elements_ref = &elements;
            std::thread::scope(|scope| {
                let mut rest: &mut [f64] = out.as_mut_slice();
                let mut row0 = 0usize;
                while row0 < batch_size {
                    let rows = rows_per.min(batch_size - row0);
                    let (blk, tail) = rest.split_at_mut(rows * fft_length);
                    rest = tail;
                    scope.spawn(move || {
                        irfft_rows_f64_into(
                            elements_ref,
                            plan_ref,
                            fft_length,
                            input_last,
                            copy_len,
                            row0,
                            rows,
                            blk,
                        );
                    });
                    row0 += rows;
                }
            });
        }
        let tensor = if out_dtype == DType::F64 {
            TensorValue::new_f64_values(out_shape, out)
        } else {
            TensorValue::new_f32_values(out_shape, out.iter().map(|&v| v as f32).collect())
        }
        .map_err(EvalError::InvalidTensor)?;
        return Ok(Value::Tensor(tensor));
    }

    // Remaining output dtypes (F16/BF16 from Complex64 etc.) keep the serial
    // Literal path.
    let mut out_elements = Vec::with_capacity(output_count);
    let mut full = vec![(0.0, 0.0); fft_length];
    let mut transformed = Vec::with_capacity(fft_length);
    let mut scratch = BluesteinScratch::default();
    let mut mixed_scratch = Vec::new();

    for batch in 0..batch_size {
        let start = batch * input_last;
        let half_spectrum = &elements[start..start + input_last];

        full[..copy_len].copy_from_slice(&half_spectrum[..copy_len]);
        full[copy_len..].fill((0.0, 0.0));

        for (k, slot) in full
            .iter_mut()
            .enumerate()
            .take(fft_length)
            .skip(input_last)
        {
            let mirror = fft_length - k;
            if mirror < input_last {
                let (re, im) = half_spectrum[mirror];
                *slot = (re, -im); // conjugate
            }
        }

        match &plan {
            Some(plan) => plan.apply_into(
                &full,
                &mut scratch,
                &mut mixed_scratch,
                true,
                &mut transformed,
            ),
            None => ifft_1d_into(&full, &mut transformed),
        }

        for &(re, _) in &transformed {
            out_elements.push(make_real_literal(re, out_dtype));
        }
    }

    let tensor =
        TensorValue::new(out_dtype, out_shape, out_elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// PROTOTYPE (test-only, pending build/bench validation — committed under a disk-low
    /// build pause): an ITERATIVE flat-stage mixed-radix DIT FFT. The recursive SoA
    /// mixed-radix was a no-ship (memory-bound; the recursion's strided per-lane access
    /// won't vectorize). This flat iterative form keeps the data in ONE buffer with
    /// contiguous per-stage butterflies — the structure that DID vectorize for radix-2
    /// and Bluestein — so it is the credible path to a smooth-composite SoA win.
    ///
    /// Tolerance-equal (NOT bit-identical) to `mixed_radix_into`: different op order, same
    /// `roots`/sign convention and `1/n` inverse scale. Validated by
    /// `iterative_mixed_radix_matches_recursive_to_tolerance`. NEXT (disk recovered):
    /// confirm this test passes, then port to a SoA (`w`-lane) production kernel and A/B it.
    ///
    /// Algorithm: factor `n` smallest-prime-first into `factors`; permute the input by the
    /// generalized digit-reversal (`r = r*f + (i%f)` over factors in order); then one stage
    /// per factor `r` with butterfly span `next = len*r`. Within a group, each of the `r`
    /// inputs at `start+j+t*len` is twiddled by `W_next^{j t} = roots[j*t*(n/next)]`, then an
    /// `r`-point DFT (`W_r^{s t} = roots[s*t*(n/r)]`) scatters the outputs.
    #[cfg(test)]
    fn mixed_radix_iterative_scalar(
        input: &[(f64, f64)],
        roots: &[(f64, f64)],
        inverse: bool,
        output: &mut Vec<(f64, f64)>,
    ) {
        let n = input.len();
        output.clear();
        output.resize(n, (0.0, 0.0));
        if n <= 1 {
            if n == 1 {
                output[0] = input[0];
            }
            return;
        }

        let mut factors: Vec<usize> = Vec::new();
        let mut nn = n;
        while nn > 1 {
            let p = smallest_prime_factor(nn);
            factors.push(p);
            nn /= p;
        }

        // Generalized digit-reversal permutation of the input.
        for (i, slot) in output.iter_mut().enumerate() {
            let mut x = i;
            let mut r = 0usize;
            for &f in &factors {
                r = r * f + (x % f);
                x /= f;
            }
            *slot = input[r];
        }

        let max_r = *factors.iter().max().unwrap();
        let mut y: Vec<(f64, f64)> = vec![(0.0, 0.0); max_r];

        let mut len = 1usize;
        for &r in &factors {
            let next = len * r;
            let ts = n / next; // input twiddle stride: W_next^{j t} = roots[j*t*ts]
            let ds = n / r; //    DFT twiddle stride:   W_r^{s t}   = roots[s*t*ds]
            let mut start = 0usize;
            while start < n {
                for j in 0..len {
                    // Gather + input-twiddle the r elements.
                    for t in 0..r {
                        let (vr, vi) = output[start + j + t * len];
                        let (twr, twi) = roots[(ts * j * t) % n];
                        y[t] = (vr * twr - vi * twi, vr * twi + vi * twr);
                    }
                    // r-point DFT scatter.
                    for s in 0..r {
                        let mut acc_r = 0.0f64;
                        let mut acc_i = 0.0f64;
                        for t in 0..r {
                            let (yr, yi) = y[t];
                            let (wr, wi) = roots[(ds * s * t) % n];
                            acc_r += yr * wr - yi * wi;
                            acc_i += yr * wi + yi * wr;
                        }
                        output[start + j + s * len] = (acc_r, acc_i);
                    }
                }
                start += next;
            }
            len = next;
        }

        if inverse {
            let inv_n = 1.0 / (n as f64);
            for v in output.iter_mut() {
                v.0 *= inv_n;
                v.1 *= inv_n;
            }
        }
    }

    /// Validates the iterative-DIT prototype against the production recursive
    /// `mixed_radix_into` to floating tolerance (both compute the same DFT). Covers
    /// radix-2/3/5 and general-factor (7/11/13) smooth composites, both directions.
    #[test]
    fn iterative_mixed_radix_matches_recursive_to_tolerance() {
        for &n in &[6usize, 10, 12, 14, 15, 21, 30, 35, 77, 143, 700, 1000] {
            assert!(is_mixed_radix_smooth(n));
            for inverse in [false, true] {
                let roots = precompute_twiddles(n, inverse);
                let input: Vec<(f64, f64)> = (0..n)
                    .map(|i| {
                        let f = i as f64;
                        ((f * 0.011).sin() - 0.4, (f * 0.019).cos() * 0.6)
                    })
                    .collect();
                let mut rec = Vec::new();
                let mut scr = Vec::new();
                mixed_radix_into(&input, &roots, inverse, &mut rec, &mut scr);
                let mut iter = Vec::new();
                mixed_radix_iterative_scalar(&input, &roots, inverse, &mut iter);
                for (k, (a, b)) in rec.iter().zip(iter.iter()).enumerate() {
                    assert!(
                        (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                        "iterative != recursive at n={n} inverse={inverse} k={k}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    /// PROTOTYPE (test-only, pending build/bench validation — disk-low pause): the SoA
    /// (`w`-lane) sibling of `mixed_radix_iterative_scalar`. Computes `batch` independent
    /// length-`n` mixed-radix DFTs at once with each element occupying `w=batch`
    /// contiguous lanes; the digit-reversal, stage structure, twiddle indices, and
    /// per-butterfly arithmetic are IDENTICAL to the scalar version (every scalar op
    /// replicated across `w` lanes with a shared scalar twiddle), so it is bit-for-bit
    /// equal to running the scalar iterative per row. This is the FLAT, single-buffer,
    /// contiguous-stage shape that vectorized for radix-2/Bluestein — the candidate
    /// production kernel for the smooth-composite SoA win.
    ///
    /// NEXT (disk recovered): confirm `iterative_soa_bit_identical_to_scalar`, then lift to
    /// module scope, tile for L1, and same-binary A/B vs per-row `mixed_radix_into`.
    #[cfg(test)]
    fn mixed_radix_iterative_soa_batch(
        elements: &[(f64, f64)],
        n: usize,
        batch: usize,
        inverse: bool,
        roots: &[(f64, f64)],
    ) -> Vec<(f64, f64)> {
        let w = batch;
        let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); batch * n];
        if n <= 1 {
            if n == 1 {
                out.copy_from_slice(&elements[..batch]);
            }
            return out;
        }

        let mut factors: Vec<usize> = Vec::new();
        let mut nn = n;
        while nn > 1 {
            let p = smallest_prime_factor(nn);
            factors.push(p);
            nn /= p;
        }

        // SoA work buffers (re[idx*w + lane]); digit-reversal happens during transpose-in.
        let mut re = vec![0.0f64; batch * n];
        let mut im = vec![0.0f64; batch * n];
        for i in 0..n {
            let mut x = i;
            let mut rev = 0usize;
            for &f in &factors {
                rev = rev * f + (x % f);
                x /= f;
            }
            for b in 0..batch {
                let (vr, vi) = elements[b * n + rev];
                re[i * w + b] = vr;
                im[i * w + b] = vi;
            }
        }

        let max_r = *factors.iter().max().unwrap();
        let mut y_re = vec![0.0f64; max_r * w];
        let mut y_im = vec![0.0f64; max_r * w];

        let mut len = 1usize;
        for &r in &factors {
            let next = len * r;
            let ts = n / next;
            let ds = n / r;
            let mut start = 0usize;
            while start < n {
                for j in 0..len {
                    for t in 0..r {
                        let (twr, twi) = roots[(ts * j * t) % n];
                        let base = (start + j + t * len) * w;
                        for b in 0..w {
                            let vr = re[base + b];
                            let vi = im[base + b];
                            y_re[t * w + b] = vr * twr - vi * twi;
                            y_im[t * w + b] = vr * twi + vi * twr;
                        }
                    }
                    for s in 0..r {
                        let obase = (start + j + s * len) * w;
                        for b in 0..w {
                            let mut ar = 0.0f64;
                            let mut ai = 0.0f64;
                            for t in 0..r {
                                let (wr, wi) = roots[(ds * s * t) % n];
                                let yr = y_re[t * w + b];
                                let yi = y_im[t * w + b];
                                ar += yr * wr - yi * wi;
                                ai += yr * wi + yi * wr;
                            }
                            re[obase + b] = ar;
                            im[obase + b] = ai;
                        }
                    }
                }
                start += next;
            }
            len = next;
        }

        let inv_n = if inverse { 1.0 / (n as f64) } else { 1.0 };
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b] * inv_n, im[k * w + b] * inv_n);
            }
        }
        out
    }

    /// The SoA iterative prototype must be bit-for-bit identical to running the scalar
    /// iterative per row (same op order; lanes don't change FP). If both match the
    /// recursive reference to tolerance (see the other test), the SoA kernel is correct.
    #[test]
    fn iterative_soa_bit_identical_to_scalar() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        for &n in &[6usize, 10, 12, 15, 30, 35, 77, 1000] {
            for inverse in [false, true] {
                let roots = precompute_twiddles(n, inverse);
                for &batch in &[1usize, 3, 4, 8] {
                    let elements: Vec<(f64, f64)> = (0..batch * n)
                        .map(|i| {
                            let f = i as f64;
                            ((f * 0.011).sin() - 0.4, (f * 0.019).cos() * 0.6)
                        })
                        .collect();
                    let mut reference = vec![(0.0, 0.0); batch * n];
                    for b in 0..batch {
                        let mut row = Vec::new();
                        mixed_radix_iterative_scalar(
                            &elements[b * n..b * n + n],
                            &roots,
                            inverse,
                            &mut row,
                        );
                        reference[b * n..b * n + n].copy_from_slice(&row);
                    }
                    let got = mixed_radix_iterative_soa_batch(&elements, n, batch, inverse, &roots);
                    assert_eq!(
                        bits(&reference),
                        bits(&got),
                        "SoA iterative != scalar iterative (n={n} batch={batch} inverse={inverse})"
                    );
                }
            }
        }
    }

    /// Production smooth-composite SoA routing should preserve the scalar iterative
    /// op order within each row. This is the focused validation gate for the
    /// disk-low code-only lift into `transform_batches_dense`.
    #[test]
    fn production_mixed_radix_soa_matches_scalar_iterative_by_bits() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        for &n in &[6usize, 10, 12, 15, 30, 35, 77, 1000] {
            for inverse in [false, true] {
                let roots = precompute_twiddles(n, inverse);
                for &batch in &[1usize, 3, 4, 8] {
                    let elements: Vec<(f64, f64)> = (0..batch * n)
                        .map(|i| {
                            let f = i as f64;
                            ((f * 0.017).cos() * 0.7, (f * 0.023).sin() - 0.1)
                        })
                        .collect();
                    let mut reference = vec![(0.0, 0.0); batch * n];
                    for b in 0..batch {
                        let mut row = Vec::new();
                        mixed_radix_iterative_scalar(
                            &elements[b * n..b * n + n],
                            &roots,
                            inverse,
                            &mut row,
                        );
                        reference[b * n..b * n + n].copy_from_slice(&row);
                    }
                    let got = transform_batches_mixed_radix_iterative_soa(
                        &elements, n, batch, inverse, &roots,
                    );
                    assert_eq!(
                        bits(&reference),
                        bits(&got),
                        "production SoA route != scalar iterative (n={n} batch={batch} inverse={inverse})"
                    );
                }
            }
        }
    }

    /// INDEPENDENT oracle check for the iterative mixed-radix prototypes: validate the
    /// SoA kernel directly against the O(n^2) `dft_1d`/`idft_1d` reference (which shares
    /// no code with the iterative path), so a bug common to BOTH the scalar and SoA
    /// iterative impls — which the cross-checks against each other / the recursive form
    /// would miss — is still caught. Small `n` only (O(n^2) reference).
    #[test]
    fn iterative_soa_matches_dft_oracle() {
        let close = |a: &[(f64, f64)], b: &[(f64, f64)], tag: &str| {
            assert_eq!(a.len(), b.len(), "len mismatch {tag}");
            for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x.0 - y.0).abs() < 1e-9 && (x.1 - y.1).abs() < 1e-9,
                    "{tag} k={k}: {x:?} vs {y:?}"
                );
            }
        };
        for &n in &[6usize, 10, 12, 14, 15, 21, 30, 35, 77, 143] {
            assert!(is_mixed_radix_smooth(n));
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let f = i as f64;
                    ((f * 0.031).cos() - 0.2, (f * 0.013).sin() * 0.8)
                })
                .collect();
            // Forward vs dft_1d.
            let fwd_roots = precompute_twiddles(n, false);
            let got_fwd = mixed_radix_iterative_soa_batch(&input, n, 1, false, &fwd_roots);
            close(&dft_1d(&input), &got_fwd, &format!("fwd n={n}"));
            // Inverse vs idft_1d (both 1/n-scaled).
            let inv_roots = precompute_twiddles(n, true);
            let got_inv = mixed_radix_iterative_soa_batch(&input, n, 1, true, &inv_roots);
            close(&idft_1d(&input), &got_inv, &format!("inv n={n}"));
        }
    }

    /// VALIDATES THE WIRED PRODUCTION ROUTE `transform_batches_mixed_radix_iterative_soa`
    /// (edd01b52, "pending-bench" — committed enabled but without a correctness test).
    /// This is the route `transform_batches_dense` actually takes for smooth composites
    /// `n <= MIXED_RADIX_ITERATIVE_SOA_MAX_N`, so it MUST agree with the established
    /// per-row recursive `mixed_radix_into` (to floating tolerance — iterative is a
    /// different op order) AND the independent `dft_1d`/`idft_1d` oracle. Mirrors the
    /// dispatch's exact `roots = precompute_twiddles(n, inverse)` call. Covers
    /// radix-2/3/5 + general (7/11/13) factors, both directions, single- and multi-row
    /// tiles. (Authored under a disk-critical no-cargo pause: harden-by-inspection now;
    /// this test executes when the build pause lifts.)
    #[test]
    fn production_mixed_radix_iterative_soa_matches_reference() {
        // Includes pure odd-prime-POWER lengths (27=3^3, 49=7^2, 81=3^4, 121=11^2,
        // 125=5^3, 169=13^2, 343=7^3) — repeated-same-odd-factor digit-reversal/stages
        // are structurally distinct from mixed-factor composites and could hide a bug.
        for &n in &[
            6usize, 10, 12, 14, 15, 21, 27, 30, 35, 49, 77, 81, 121, 125, 143, 169, 343, 360, 700,
            1000,
        ] {
            // Route is DISABLED in production (MAX_N=0, measured 0.15x no-ship), but the
            // kernel stays correctness-validated here for these smooth lengths.
            assert!(is_mixed_radix_smooth(n));
            for inverse in [false, true] {
                let roots = precompute_twiddles(n, inverse);
                for &batch in &[1usize, 3, 4, 8, 17] {
                    let elements: Vec<(f64, f64)> = (0..batch * n)
                        .map(|i| {
                            let f = (i % (5 * n)) as f64;
                            ((f * 0.011).sin() - 0.4, (f * 0.019).cos() * 0.6)
                        })
                        .collect();
                    // Reference: established per-row recursive path.
                    let mut reference = vec![(0.0, 0.0); batch * n];
                    let mut ob = Vec::new();
                    let mut scr = Vec::new();
                    for b in 0..batch {
                        mixed_radix_into(
                            &elements[b * n..b * n + n],
                            &roots,
                            inverse,
                            &mut ob,
                            &mut scr,
                        );
                        reference[b * n..b * n + n].copy_from_slice(&ob);
                    }
                    // Production route, called exactly as the dispatch does.
                    let got = transform_batches_mixed_radix_iterative_soa(
                        &elements, n, batch, inverse, &roots,
                    );
                    for (k, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                        assert!(
                            (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                            "production iterative SoA != recursive (n={n} batch={batch} inverse={inverse} k={k}): {a:?} vs {b:?}"
                        );
                    }
                }
            }
        }
        // Independent O(n^2) DFT oracle on a single row (small n).
        for &n in &[6usize, 12, 15, 27, 30, 35, 49, 77, 121, 125, 143] {
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let f = i as f64;
                    ((f * 0.031).cos() - 0.2, (f * 0.013).sin() * 0.8)
                })
                .collect();
            let fwd_roots = precompute_twiddles(n, false);
            let got_f =
                transform_batches_mixed_radix_iterative_soa(&input, n, 1, false, &fwd_roots);
            for (k, (a, b)) in dft_1d(&input).iter().zip(got_f.iter()).enumerate() {
                assert!(
                    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                    "production iterative SoA != dft_1d oracle (n={n} k={k})"
                );
            }
            let inv_roots = precompute_twiddles(n, true);
            let got_i = transform_batches_mixed_radix_iterative_soa(&input, n, 1, true, &inv_roots);
            for (k, (a, b)) in idft_1d(&input).iter().zip(got_i.iter()).enumerate() {
                assert!(
                    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                    "production iterative SoA != idft_1d oracle (n={n} k={k})"
                );
            }
        }
    }

    /// A/B HARNESS for the keep-or-disable decision on the enabled iterative SoA route
    /// (the bench the convergent `edd01b52` shipped WITHOUT — see the murmw pending-bench
    /// ledger entry). OLD = per-row recursive `mixed_radix_into`; NEW = the wired
    /// production route. Interleaved min-of-9 single-thread ratio (the only trustworthy
    /// FFT A/B — threaded/sequential drift, see the threaded-FFT contention caveat).
    /// Targets `fft_batch_128x1000` (n=1000, batch=128; batch*n=128000 < 1<<18, in the
    /// gate) — last measured 12.55x off JAX. Run with `--ignored --nocapture` once the
    /// disk pressure clears. DECISION: keep the gate only if NEW < OLD (a real win); if
    /// it regresses like the recursive SoA no-ship did, set `MIXED_RADIX_ITERATIVE_SOA_MAX_N
    /// = 0`. (Authored under the disk-critical no-cargo pause; executes when cargo returns.)
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_mixed_radix_iterative_soa_vs_per_row() {
        let n = 1000usize;
        let rows = 128usize;
        let roots = precompute_twiddles(n, false);
        let elements: Vec<(f64, f64)> = (0..rows * n)
            .map(|i| {
                let f = i as f64;
                ((f * 0.011).sin(), (f * 0.019).cos())
            })
            .collect();

        let run_old = || -> (u64, u64) {
            let mut out = vec![(0.0, 0.0); rows * n];
            let mut ob = Vec::new();
            let mut scr = Vec::new();
            let t0 = std::time::Instant::now();
            for b in 0..rows {
                mixed_radix_into(&elements[b * n..b * n + n], &roots, false, &mut ob, &mut scr);
                out[b * n..b * n + n].copy_from_slice(&ob);
            }
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
        };
        let run_new = || -> (u64, u64) {
            let t0 = std::time::Instant::now();
            let out = transform_batches_mixed_radix_iterative_soa(&elements, n, rows, false, &roots);
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
        };
        let (mut old_min, mut new_min) = (u64::MAX, u64::MAX);
        let mut chk = 0u64;
        for _ in 0..9 {
            let (o, co) = run_old();
            let (nn, cn) = run_new();
            chk ^= co ^ cn;
            old_min = old_min.min(o);
            new_min = new_min.min(nn);
        }
        std::hint::black_box(chk);
        eprintln!(
            "[mixed-radix iterative SoA {rows}x{n}] 1T per-row={:.3}ms iter={:.3}ms ratio={:.2}x (min of 9 interleaved)",
            old_min as f64 / 1e6,
            new_min as f64 / 1e6,
            old_min as f64 / new_min as f64,
        );
    }

    /// A/B HARNESS for the smooth-composite lever: route batched smooth composites
    /// (e.g. 128x1000) through the PROVEN Bluestein SoA instead of per-row recursive
    /// `mixed_radix_into`. The iterative mixed-radix SoA no-shipped (0.15x, memory-bound),
    /// but Bluestein's two internal convolution FFTs are the FLAT radix-2 kernel that DOES
    /// vectorize across rows (it wins 3x on prime lengths). Result: even with Bluestein's
    /// ~4x flop overhead (n=1000 -> m=2048 conv), vectorizing across 128 rows beats the
    /// scalar per-row recursive path. OLD = per-row recursive; NEW =
    /// `transform_batches_bluestein_vectorized`. Interleaved min-of-9 (the only trustworthy
    /// single-thread FFT A/B). Drove the `BLUESTEIN_SMOOTH_MIN_N` gate in
    /// `transform_batches_dense`.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_bluestein_soa_vs_per_row_smooth() {
        let n = 1000usize;
        let rows = 128usize;
        let roots = precompute_twiddles(n, false);
        let bplan = BluesteinPlan::new(n, false);
        let elements: Vec<(f64, f64)> = (0..rows * n)
            .map(|i| {
                let f = i as f64;
                ((f * 0.011).sin(), (f * 0.019).cos())
            })
            .collect();

        // Both closures build their plan INSIDE the timed region — production
        // `transform_batches_dense` builds the per-row mixed-radix roots / the `BluesteinPlan`
        // ONCE per batch call, and the Bluestein plan (chirp + an m=2048 kernel FFT for n=1000)
        // is far more expensive to build than the recursive roots. Excluding it (as an earlier
        // pass did) wrongly favored Bluestein; including it is the production-realistic A/B.
        let run_old = || -> (u64, u64) {
            let mut out = vec![(0.0, 0.0); rows * n];
            let mut ob = Vec::new();
            let mut scr = Vec::new();
            let t0 = std::time::Instant::now();
            let roots = precompute_twiddles(n, false);
            for b in 0..rows {
                mixed_radix_into(&elements[b * n..b * n + n], &roots, false, &mut ob, &mut scr);
                out[b * n..b * n + n].copy_from_slice(&ob);
            }
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
        };
        let run_new = || -> (u64, u64) {
            let t0 = std::time::Instant::now();
            let bplan = BluesteinPlan::new(n, false);
            let out = transform_batches_bluestein_vectorized(&bplan, &elements, n, rows);
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
        };
        // Correctness: Bluestein must match the recursive reference to FFT tolerance.
        {
            let mut ob = Vec::new();
            let mut scr = Vec::new();
            let bl = transform_batches_bluestein_vectorized(&bplan, &elements, n, rows);
            for b in [0usize, 1, rows / 2, rows - 1] {
                mixed_radix_into(&elements[b * n..b * n + n], &roots, false, &mut ob, &mut scr);
                for k in 0..n {
                    let (a, c) = (ob[k], bl[b * n + k]);
                    assert!(
                        (a.0 - c.0).abs() < 1e-6 && (a.1 - c.1).abs() < 1e-6,
                        "Bluestein SoA != recursive (b={b} k={k}): {a:?} vs {c:?}"
                    );
                }
            }
        }
        let (mut old_min, mut new_min) = (u64::MAX, u64::MAX);
        let mut chk = 0u64;
        for _ in 0..9 {
            let (o, co) = run_old();
            let (nn, cn) = run_new();
            chk ^= co ^ cn;
            old_min = old_min.min(o);
            new_min = new_min.min(nn);
        }
        std::hint::black_box(chk);
        eprintln!(
            "[bluestein SoA vs per-row {rows}x{n}] 1T per-row={:.3}ms bluestein={:.3}ms ratio={:.2}x (min of 9 interleaved)",
            old_min as f64 / 1e6,
            new_min as f64 / 1e6,
            old_min as f64 / new_min as f64,
        );
    }

    /// PROFILE the three phases of the SoA pow2 kernel (transpose-in / butterfly stages /
    /// transpose-out) for one L1-resident tile. DECIDES whether a radix-4 SoA is worth
    /// building: radix-4 only HALVES the butterfly passes (log2->log4 stages), so it pays
    /// off only if the butterfly stages dominate the per-tile cost. If the transposes
    /// dominate, radix-4 cannot help (same transpose either way) and is a no-go.
    #[test]
    #[ignore = "informational profile; run with --ignored --nocapture"]
    fn profile_soa_pow2_phases() {
        let n = 256usize;
        let w = POW2_TILE_ROWS; // one tile
        let plan = Radix2Plan::new(n, false);
        let elements: Vec<(f64, f64)> = (0..w * n)
            .map(|i| ((i as f64 * 0.01).sin(), (i as f64 * 0.02).cos()))
            .collect();
        let mut re = vec![0.0f64; w * n];
        let mut im = vec![0.0f64; w * n];
        let mut out = vec![(0.0f64, 0.0f64); w * n];
        let iters = 50_000usize;

        let mut t_in = u64::MAX;
        let mut t_bf = u64::MAX;
        let mut t_out = u64::MAX;
        for _ in 0..9 {
            // transpose-in (AoS -> SoA with bit-reversal)
            let s0 = std::time::Instant::now();
            for _ in 0..iters {
                for b in 0..w {
                    let row = &elements[b * n..b * n + n];
                    for (i, &src) in plan.bit_reversed.iter().enumerate() {
                        let (sr, si) = row[src];
                        re[i * w + b] = sr;
                        im[i * w + b] = si;
                    }
                }
                std::hint::black_box(&re);
            }
            t_in = t_in.min(s0.elapsed().as_nanos() as u64);
            // butterfly stages (the only phase radix-4 would change)
            let s1 = std::time::Instant::now();
            for _ in 0..iters {
                soa_radix2_butterfly_stages(&plan, w, n, &mut re, &mut im);
                std::hint::black_box(&re);
            }
            t_bf = t_bf.min(s1.elapsed().as_nanos() as u64);
            // transpose-out (SoA -> AoS)
            let s2 = std::time::Instant::now();
            for _ in 0..iters {
                for b in 0..w {
                    let dst = &mut out[b * n..b * n + n];
                    for (k, slot) in dst.iter_mut().enumerate() {
                        *slot = (re[k * w + b], im[k * w + b]);
                    }
                }
                std::hint::black_box(&out);
            }
            t_out = t_out.min(s2.elapsed().as_nanos() as u64);
        }
        let total = t_in + t_bf + t_out;
        eprintln!(
            "[SoA pow2 phases {w}x{n}, min-of-9 x{iters}] transpose-in={:.2}ms ({:.0}%) butterflies={:.2}ms ({:.0}%) transpose-out={:.2}ms ({:.0}%)",
            t_in as f64 / 1e6,
            100.0 * t_in as f64 / total as f64,
            t_bf as f64 / 1e6,
            100.0 * t_bf as f64 / total as f64,
            t_out as f64 / 1e6,
            100.0 * t_out as f64 / total as f64,
        );
    }

    /// CORRECTNESS: the radix-4 SoA kernel must match the independent O(n^2) DFT oracle
    /// (to tolerance — radix-4 is a different, fewer-op schedule than radix-2) for every
    /// pow-of-four length, both directions, across a multi-row SoA tile.
    #[test]
    fn radix4_soa_matches_dft_oracle() {
        for &n in &[4usize, 16, 64, 256] {
            for inverse in [false, true] {
                let plan = Radix4Plan::new(n, inverse);
                let w = 8usize;
                let elements: Vec<(f64, f64)> = (0..w * n)
                    .map(|i| ((i as f64 * 0.017).sin() - 0.2, (i as f64 * 0.029).cos() * 0.6))
                    .collect();
                let mut re = vec![0.0f64; w * n];
                let mut im = vec![0.0f64; w * n];
                let mut out = vec![(0.0f64, 0.0f64); w * n];
                vectorized_pow4_block(&plan, &elements, w, n, inverse, &mut re, &mut im, &mut out);
                for b in 0..w {
                    let row = &elements[b * n..b * n + n];
                    let want = if inverse { idft_1d(row) } else { dft_1d(row) };
                    for k in 0..n {
                        let (a, c) = (want[k], out[b * n + k]);
                        assert!(
                            (a.0 - c.0).abs() < 1e-9 && (a.1 - c.1).abs() < 1e-9,
                            "radix4 SoA != DFT oracle (n={n} inv={inverse} b={b} k={k}): {a:?} vs {c:?}"
                        );
                    }
                }
            }
        }
    }

    /// A/B HARNESS for the radix-4 SoA lever (profiling green-lit it: butterflies were 57%
    /// of the pow2 SoA kernel, and radix-4 halves the stage count). OLD = radix-2 SoA
    /// butterfly stages; NEW = radix-4 SoA butterfly stages — same transpose framing, so the
    /// stage cost is what differs. Interleaved min-of-9 (the only trustworthy FFT A/B). KEEP
    /// + wire into the pow2 dispatch for pow4 lengths only if radix-4 < radix-2.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_radix4_vs_radix2_soa() {
        let n = 256usize;
        let w = POW2_TILE_ROWS;
        let p2 = Radix2Plan::new(n, false);
        let p4 = Radix4Plan::new(n, false);
        let base: Vec<f64> = (0..w * n).map(|i| (i as f64 * 0.013).sin()).collect();

        let run_old = || -> (u64, u64) {
            let mut re: Vec<f64> = base.clone();
            let mut im = vec![0.0f64; w * n];
            let t0 = std::time::Instant::now();
            soa_radix2_butterfly_stages(&p2, w, n, &mut re, &mut im);
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, re[0].to_bits() ^ re[w * n / 2].to_bits())
        };
        let run_new = || -> (u64, u64) {
            let mut re: Vec<f64> = base.clone();
            let mut im = vec![0.0f64; w * n];
            let t0 = std::time::Instant::now();
            soa_radix4_butterfly_stages(&p4, w, n, &mut re, &mut im);
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, re[0].to_bits() ^ re[w * n / 2].to_bits())
        };
        let (mut o, mut nn) = (u64::MAX, u64::MAX);
        let mut chk = 0u64;
        for _ in 0..9 {
            let (a, ca) = run_old();
            let (b, cb) = run_new();
            chk ^= ca ^ cb;
            o = o.min(a);
            nn = nn.min(b);
        }
        std::hint::black_box(chk);
        eprintln!(
            "[radix4 vs radix2 SoA butterflies {w}x{n}] radix2={:.4}ms radix4={:.4}ms ratio={:.2}x (min of 9 interleaved)",
            o as f64 / 1e6,
            nn as f64 / 1e6,
            o as f64 / nn as f64,
        );
    }

    /// PROTOTYPE (test-only, pending build/bench validation — disk-low pause): the
    /// SPECIALIZED-BUTTERFLY iterative SoA mixed-radix kernel. Identical to
    /// `mixed_radix_iterative_soa_batch` except the per-stage r-point DFT uses the
    /// proven low-mult radix-2/3/5 butterflies (adapted op-for-op from the recursive
    /// `mixed_radix_ping`: radix-3 trades 9 mults for 2; radix-5 trades 25 for ~4)
    /// instead of the general O(r^2) accumulate, falling back to O(r^2) for 7/11/13.
    /// The gathered `y[t]` are already input-twiddled (= the recursive's `u_t`), so the
    /// butterflies apply the same length-r DFT with `W_r = roots[n/r]`/`roots[2n/r]`.
    ///
    /// WHY: the general O(r^2) DFT is the likely reason the enabled iterative route may
    /// not beat per-row on radix-3/5-heavy composites (e.g. n=1000=2^3*5^3 — its three
    /// radix-5 stages cost 25 mults/output each). This is the optimization to deploy IF
    /// `bench_mixed_radix_iterative_soa_vs_per_row` shows the general route losing.
    /// Tolerance-equal (NOT bit-identical) to the general DFT; validated by
    /// `iterative_soa_specialized_matches_dft_oracle`.
    #[cfg(test)]
    fn mixed_radix_iterative_soa_specialized(
        elements: &[(f64, f64)],
        n: usize,
        batch: usize,
        inverse: bool,
        roots: &[(f64, f64)],
    ) -> Vec<(f64, f64)> {
        let w = batch;
        let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); batch * n];
        if n <= 1 {
            if n == 1 {
                out.copy_from_slice(&elements[..batch]);
            }
            return out;
        }
        let mut factors: Vec<usize> = Vec::new();
        let mut nn = n;
        while nn > 1 {
            let p = smallest_prime_factor(nn);
            factors.push(p);
            nn /= p;
        }
        let mut re = vec![0.0f64; batch * n];
        let mut im = vec![0.0f64; batch * n];
        for i in 0..n {
            let mut x = i;
            let mut rev = 0usize;
            for &f in &factors {
                rev = rev * f + (x % f);
                x /= f;
            }
            for b in 0..batch {
                let (vr, vi) = elements[b * n + rev];
                re[i * w + b] = vr;
                im[i * w + b] = vi;
            }
        }
        let max_r = *factors.iter().max().unwrap();
        let mut y_re = vec![0.0f64; max_r * w];
        let mut y_im = vec![0.0f64; max_r * w];

        let mut len = 1usize;
        for &r in &factors {
            let next = len * r;
            let ts = n / next;
            let ds = n / r;
            let (c3, d3) = if r == 3 { roots[ds] } else { (0.0, 0.0) };
            let (c5a, d5a) = if r == 5 { roots[ds] } else { (0.0, 0.0) };
            let (c5b, d5b) = if r == 5 { roots[2 * ds] } else { (0.0, 0.0) };
            let mut start = 0usize;
            while start < n {
                for j in 0..len {
                    // Gather + input-twiddle (identical to the general kernel).
                    for t in 0..r {
                        let (twr, twi) = roots[(ts * j * t) % n];
                        let base = (start + j + t * len) * w;
                        let yb = t * w;
                        for b in 0..w {
                            let vr = re[base + b];
                            let vi = im[base + b];
                            y_re[yb + b] = vr * twr - vi * twi;
                            y_im[yb + b] = vr * twi + vi * twr;
                        }
                    }
                    // Specialized r-point DFT (y already twiddled).
                    let o = |s: usize| (start + j + s * len) * w;
                    if r == 2 {
                        let (o0, o1) = (o(0), o(1));
                        for b in 0..w {
                            let (y0r, y0i) = (y_re[b], y_im[b]);
                            let (y1r, y1i) = (y_re[w + b], y_im[w + b]);
                            re[o0 + b] = y0r + y1r;
                            im[o0 + b] = y0i + y1i;
                            re[o1 + b] = y0r - y1r;
                            im[o1 + b] = y0i - y1i;
                        }
                    } else if r == 3 {
                        let (o0, o1, o2) = (o(0), o(1), o(2));
                        for b in 0..w {
                            let (y0r, y0i) = (y_re[b], y_im[b]);
                            let (y1r, y1i) = (y_re[w + b], y_im[w + b]);
                            let (y2r, y2i) = (y_re[2 * w + b], y_im[2 * w + b]);
                            let (sr, si) = (y1r + y2r, y1i + y2i);
                            let (dr, di) = (y1r - y2r, y1i - y2i);
                            re[o0 + b] = y0r + sr;
                            im[o0 + b] = y0i + si;
                            let (cr, ci) = (y0r + c3 * sr, y0i + c3 * si);
                            let (xr, xi) = (d3 * di, d3 * dr);
                            re[o1 + b] = cr - xr;
                            im[o1 + b] = ci + xi;
                            re[o2 + b] = cr + xr;
                            im[o2 + b] = ci - xi;
                        }
                    } else if r == 5 {
                        let (o0, o1, o2, o3, o4) = (o(0), o(1), o(2), o(3), o(4));
                        for b in 0..w {
                            let (y0r, y0i) = (y_re[b], y_im[b]);
                            let (y1r, y1i) = (y_re[w + b], y_im[w + b]);
                            let (y2r, y2i) = (y_re[2 * w + b], y_im[2 * w + b]);
                            let (y3r, y3i) = (y_re[3 * w + b], y_im[3 * w + b]);
                            let (y4r, y4i) = (y_re[4 * w + b], y_im[4 * w + b]);
                            let (t1r, t1i) = (y1r + y4r, y1i + y4i);
                            let (t1dr, t1di) = (y1r - y4r, y1i - y4i);
                            let (t2r, t2i) = (y2r + y3r, y2i + y3i);
                            let (t2dr, t2di) = (y2r - y3r, y2i - y3i);
                            re[o0 + b] = y0r + t1r + t2r;
                            im[o0 + b] = y0i + t1i + t2i;
                            let ar = y0r + c5a * t1r + c5b * t2r;
                            let ai = y0i + c5a * t1i + c5b * t2i;
                            let xr = d5a * t1di + d5b * t2di;
                            let xi = d5a * t1dr + d5b * t2dr;
                            re[o1 + b] = ar - xr;
                            im[o1 + b] = ai + xi;
                            re[o4 + b] = ar + xr;
                            im[o4 + b] = ai - xi;
                            let br = y0r + c5b * t1r + c5a * t2r;
                            let bi = y0i + c5b * t1i + c5a * t2i;
                            let yr = d5b * t1di - d5a * t2di;
                            let yi = d5b * t1dr - d5a * t2dr;
                            re[o2 + b] = br - yr;
                            im[o2 + b] = bi + yi;
                            re[o3 + b] = br + yr;
                            im[o3 + b] = bi - yi;
                        }
                    } else {
                        // General O(r^2) fallback (7/11/13).
                        for s in 0..r {
                            let obase = (start + j + s * len) * w;
                            for b in 0..w {
                                let mut ar = 0.0f64;
                                let mut ai = 0.0f64;
                                for t in 0..r {
                                    let (wr, wi) = roots[(ds * s * t) % n];
                                    let yr = y_re[t * w + b];
                                    let yi = y_im[t * w + b];
                                    ar += yr * wr - yi * wi;
                                    ai += yr * wi + yi * wr;
                                }
                                re[obase + b] = ar;
                                im[obase + b] = ai;
                            }
                        }
                    }
                }
                start += next;
            }
            len = next;
        }

        let inv_n = if inverse { 1.0 / (n as f64) } else { 1.0 };
        for b in 0..batch {
            let dst = &mut out[b * n..b * n + n];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = (re[k * w + b] * inv_n, im[k * w + b] * inv_n);
            }
        }
        out
    }

    /// The specialized-butterfly iterative kernel must match the independent O(n^2)
    /// `dft_1d`/`idft_1d` oracle to tolerance (it reorders the radix-3/5 arithmetic, so
    /// it is NOT bit-identical to the general kernel). Covers radix-2/3/5 + general
    /// (7/11/13) factors and pure odd-prime-powers, both directions.
    #[test]
    fn iterative_soa_specialized_matches_dft_oracle() {
        for &n in &[6usize, 10, 12, 14, 15, 21, 27, 30, 35, 49, 77, 121, 125, 143] {
            assert!(is_mixed_radix_smooth(n));
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let f = i as f64;
                    ((f * 0.031).cos() - 0.2, (f * 0.013).sin() * 0.8)
                })
                .collect();
            let fwd_roots = precompute_twiddles(n, false);
            let got_f = mixed_radix_iterative_soa_specialized(&input, n, 1, false, &fwd_roots);
            for (k, (a, b)) in dft_1d(&input).iter().zip(got_f.iter()).enumerate() {
                assert!(
                    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                    "specialized iterative != dft_1d (n={n} k={k})"
                );
            }
            let inv_roots = precompute_twiddles(n, true);
            let got_i = mixed_radix_iterative_soa_specialized(&input, n, 1, true, &inv_roots);
            for (k, (a, b)) in idft_1d(&input).iter().zip(got_i.iter()).enumerate() {
                assert!(
                    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                    "specialized iterative != idft_1d (n={n} k={k})"
                );
            }
        }
    }

    /// END-TO-END DISPATCH integration test: drives `transform_batches_dense` (the
    /// function `eval_fft`/`eval_ifft` call) at batch >= 8 across lengths that hit EVERY
    /// routing branch, and checks each row against the independent O(n^2) `dft_1d`/`idft_1d`
    /// oracle. Per-kernel tests validate each SoA kernel in isolation; this validates that
    /// the DISPATCH sends each length class to the right kernel AND returns correct output:
    ///   - pow2 (16, 256)                 -> `transform_batches_pow2_vectorized`
    ///   - smooth composite n<=1024 (12, 30, 49, 1000) -> iterative mixed-radix SoA
    ///   - smooth composite n>1024 (1080=2^3*3^3*5)     -> per-row `BatchFftPlan::Mixed`
    ///   - prime (13, 127)                -> Bluestein SoA
    ///   - non-smooth composite (46=2*23, 187=11*17)    -> Bluestein SoA
    ///     Magnitude-relative tolerance (robust for the larger n's FFT-vs-O(n^2)-DFT rounding).
    #[test]
    fn transform_batches_dense_dispatch_matches_dft_oracle() {
        for &n in &[16usize, 256, 12, 30, 49, 1000, 1080, 13, 127, 46, 187] {
            for inverse in [false, true] {
                let batch = 8usize; // >= every SoA gate's MIN_BATCH
                let elements: Vec<(f64, f64)> = (0..batch * n)
                    .map(|i| {
                        let f = (i % (3 * n)) as f64;
                        ((f * 0.017).sin() - 0.3, (f * 0.011).cos() * 0.7)
                    })
                    .collect();
                let got = transform_batches_dense(&elements, n, batch, inverse);
                for b in 0..batch {
                    let row = &elements[b * n..b * n + n];
                    let reference = if inverse { idft_1d(row) } else { dft_1d(row) };
                    for (k, (a, c)) in reference.iter().zip(got[b * n..b * n + n].iter()).enumerate()
                    {
                        let tol = 1e-9 * a.0.abs().max(a.1.abs()).max(1.0);
                        assert!(
                            (a.0 - c.0).abs() <= tol && (a.1 - c.1).abs() <= tol,
                            "dispatch != DFT oracle (n={n} inverse={inverse} row={b} k={k}): {a:?} vs {c:?}"
                        );
                    }
                }
            }
        }
    }

    /// THREADED-PATH validation for the shipped SoA routes. Every other SoA bit-identity
    /// test uses a small batch that stays single-threaded; this one forces batches large
    /// enough to cross the `1<<18`-element threading floor so `transform_batches_dense`
    /// fans the work across `thread::scope` chunks — exercising the chunk-boundary /
    /// row-offset / `split_at_mut` logic that would otherwise be untested in
    /// production-active code. Asserts bit-identity vs the per-row `BatchFftPlan` (the SoA
    /// kernels are bit-identical per row, so threading must preserve that exactly). Covers
    /// the full-complex pow2 SoA (n=256) and the Bluestein SoA (n=127, m=256), both dirs.
    #[test]
    fn threaded_soa_dispatch_bit_identical_to_per_row() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        // batch*work >= 1<<18 (262144) forces the threaded path: 1100*256 = 281600.
        for &(n, batch) in &[(256usize, 1100usize), (127usize, 1100usize)] {
            for inverse in [false, true] {
                let elements: Vec<(f64, f64)> = (0..batch * n)
                    .map(|i| {
                        let f = (i % (7 * n)) as f64;
                        ((f * 0.013).sin(), (f * 0.021).cos())
                    })
                    .collect();
                // Per-row reference via the non-SoA engine (radix-2 / Bluestein per row).
                let plan = BatchFftPlan::new(n, inverse);
                let mut reference = vec![(0.0, 0.0); batch * n];
                let mut scratch = BluesteinScratch::default();
                let mut mixed_scratch = Vec::new();
                let mut buf = Vec::new();
                for b in 0..batch {
                    plan.apply_into(
                        &elements[b * n..b * n + n],
                        &mut scratch,
                        &mut mixed_scratch,
                        inverse,
                        &mut buf,
                    );
                    reference[b * n..b * n + n].copy_from_slice(&buf);
                }
                let got = transform_batches_dense(&elements, n, batch, inverse);
                assert_eq!(
                    bits(&reference),
                    bits(&got),
                    "threaded SoA dispatch != per-row (n={n} batch={batch} inverse={inverse})"
                );
            }
        }
    }

    /// THREADED-PATH validation for the RFFT/IRFFT SoA routes, which thread separately in
    /// `eval_rfft`/`eval_irfft` (splitting the batch into row-blocks, each via
    /// `rfft_rows_into`/`irfft_rows_f64_into` with a non-zero `row_start`). Every other
    /// rfft/irfft test uses a single block at offset 0, so the `row_start` block-offset
    /// logic the threading relies on is otherwise untested — a bad offset would silently
    /// corrupt large real-FFT batches. Asserts that computing all rows in one call equals
    /// computing them in two offset blocks (which is exactly what the threads do).
    #[test]
    fn rfft_irfft_block_offset_matches_single_block() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        let fft_length = 256usize;
        let batch = 20usize; // each half-block (10) >= the 8-row SoA floor

        // RFFT: real input rows -> Hermitian-half complex output.
        {
            let real_plan = RealRfftPower2Plan::new(fft_length);
            let (input_last, copy_len, out_last) = (fft_length, fft_length, fft_length / 2 + 1);
            let elements: Vec<(f64, f64)> = (0..batch * input_last)
                .map(|i| ((i as f64 * 0.017).sin() - 0.3, 0.0))
                .collect();
            let mut whole = vec![(0.0, 0.0); batch * out_last];
            rfft_rows_into(
                &elements, None, Some(&real_plan), fft_length, input_last, copy_len, out_last, 0,
                batch, &mut whole,
            );
            let mut split = vec![(0.0, 0.0); batch * out_last];
            let half = batch / 2;
            let (s0, s1) = split.split_at_mut(half * out_last);
            rfft_rows_into(
                &elements, None, Some(&real_plan), fft_length, input_last, copy_len, out_last, 0,
                half, s0,
            );
            rfft_rows_into(
                &elements, None, Some(&real_plan), fft_length, input_last, copy_len, out_last,
                half, batch - half, s1,
            );
            assert_eq!(bits(&whole), bits(&split), "rfft block-offset != single-block");
        }

        // IRFFT: Hermitian-half complex input -> real output.
        {
            let plan = BatchFftPlan::new(fft_length, true);
            let input_last = fft_length / 2 + 1;
            let copy_len = fft_length.min(input_last);
            let elements: Vec<(f64, f64)> = (0..batch * input_last)
                .map(|i| {
                    let f = i as f64;
                    ((f * 0.013).sin(), (f * 0.021).cos())
                })
                .collect();
            let mut whole = vec![0.0f64; batch * fft_length];
            irfft_rows_f64_into(
                &elements, Some(&plan), fft_length, input_last, copy_len, 0, batch, &mut whole,
            );
            let mut split = vec![0.0f64; batch * fft_length];
            let half = batch / 2;
            let (s0, s1) = split.split_at_mut(half * fft_length);
            irfft_rows_f64_into(
                &elements, Some(&plan), fft_length, input_last, copy_len, 0, half, s0,
            );
            irfft_rows_f64_into(
                &elements, Some(&plan), fft_length, input_last, copy_len, half, batch - half, s1,
            );
            let wb: Vec<u64> = whole.iter().map(|x| x.to_bits()).collect();
            let sb: Vec<u64> = split.iter().map(|x| x.to_bits()).collect();
            assert_eq!(wb, sb, "irfft block-offset != single-block");
        }
    }

    /// INDEPENDENT (oracle-free) property check on the SoA batch dispatch: Parseval's
    /// theorem. For the unnormalized forward transform, energy is conserved as
    /// `sum_k |X[k]|^2 == n * sum_j |x[j]|^2` for every row. This holds for ANY correct
    /// DFT and uses ONLY the dispatch's input/output — it does not compare to `dft_1d`,
    /// so it catches a scaling/missing-term error even in the (unlikely) case such a bug
    /// were shared with the reference. Covers every routing branch (pow2 / iterative
    /// smooth / Bluestein prime+non-smooth) at batch >= 8.
    #[test]
    fn soa_dispatch_satisfies_parseval() {
        for &n in &[16usize, 256, 12, 30, 1000, 13, 127, 46] {
            let batch = 8usize;
            let elements: Vec<(f64, f64)> = (0..batch * n)
                .map(|i| {
                    let f = (i % (5 * n)) as f64;
                    ((f * 0.017).sin() - 0.2, (f * 0.029).cos() * 0.6)
                })
                .collect();
            let got = transform_batches_dense(&elements, n, batch, false); // forward
            for b in 0..batch {
                let energy_in: f64 = elements[b * n..b * n + n]
                    .iter()
                    .map(|&(r, i)| r * r + i * i)
                    .sum();
                let energy_out: f64 =
                    got[b * n..b * n + n].iter().map(|&(r, i)| r * r + i * i).sum();
                let expected = n as f64 * energy_in;
                assert!(
                    (energy_out - expected).abs() <= 1e-7 * expected.max(1.0),
                    "Parseval violated (n={n} row={b}): out-energy {energy_out} vs n*in-energy {expected}"
                );
            }
        }
    }

    /// Old inline-recurrence radix-2 (pre-frankenjax-* twiddle-hoist), kept only as the
    /// bench baseline. Bit-identical to the production `radix2_fft_1d_into`.
    #[cfg(test)]
    fn radix2_fft_serial_ref(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>) {
        let n = input.len();
        output.clear();
        output.extend_from_slice(input);
        if n <= 1 {
            return;
        }
        bit_reverse_permute(output);
        let mut len = 2_usize;
        while len <= n {
            let half = len / 2;
            let angle = -2.0 * std::f64::consts::PI / (len as f64);
            let (sin_step, cos_step) = angle.sin_cos();
            for start in (0..n).step_by(len) {
                let mut tr = 1.0;
                let mut ti = 0.0;
                for offset in 0..half {
                    let even = output[start + offset];
                    let odd = output[start + offset + half];
                    let rr = odd.0 * tr - odd.1 * ti;
                    let ri = odd.0 * ti + odd.1 * tr;
                    output[start + offset] = (even.0 + rr, even.1 + ri);
                    output[start + offset + half] = (even.0 - rr, even.1 - ri);
                    let nr = tr * cos_step - ti * sin_step;
                    let ni = tr * sin_step + ti * cos_step;
                    tr = nr;
                    ti = ni;
                }
            }
            len *= 2;
        }
    }

    /// Same-binary A/B for the batched-pow2 plan-caching win (frankenjax-murmw): the OLD
    /// path rebuilt the bit-reversal + twiddle table (incl. per-stage `sin_cos`) on EVERY
    /// row via `fft_1d_into`; the NEW path builds one `Radix2Plan` and reuses it. Output is
    /// bit-identical (asserted); run with `--ignored --nocapture` for the ratio.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_batched_pow2_plan_cache_vs_per_row_rebuild() {
        let n = 256usize;
        let rows = 2048usize;
        let row: Vec<(f64, f64)> = (0..n)
            .map(|i| ((i as f64 * 0.013).sin(), (i as f64 * 0.027).cos()))
            .collect();

        let best = |f: &dyn Fn() -> u64| {
            let mut b = u64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };

        // OLD: per-row rebuild (twiddles + bit-reversal recomputed each row).
        let old = best(&|| {
            let mut buf = Vec::with_capacity(n);
            let mut checksum = 0u64;
            for _ in 0..rows {
                fft_1d_into(&row, &mut buf);
                checksum ^= buf[0].0.to_bits() ^ buf[n / 2].1.to_bits();
            }
            checksum
        });
        // NEW: one shared plan, reused across rows.
        let plan = Radix2Plan::new(n, false);
        let new = best(&|| {
            let mut buf = Vec::with_capacity(n);
            let mut checksum = 0u64;
            for _ in 0..rows {
                plan.apply_into(&row, &mut buf);
                checksum ^= buf[0].0.to_bits() ^ buf[n / 2].1.to_bits();
            }
            checksum
        });

        // INVERSE direction (irfft pow2 plan-cache, murmw step 2): same mechanism with
        // inverse=true (ifft_1d_into per row vs cached Radix2Plan).
        let old_inv = best(&|| {
            let mut buf = Vec::with_capacity(n);
            let mut checksum = 0u64;
            for _ in 0..rows {
                ifft_1d_into(&row, &mut buf);
                checksum ^= buf[0].0.to_bits() ^ buf[n / 2].1.to_bits();
            }
            checksum
        });
        let inv_plan = Radix2Plan::new(n, true);
        let new_inv = best(&|| {
            let mut buf = Vec::with_capacity(n);
            let mut checksum = 0u64;
            for _ in 0..rows {
                inv_plan.apply_into(&row, &mut buf);
                checksum ^= buf[0].0.to_bits() ^ buf[n / 2].1.to_bits();
            }
            checksum
        });

        // Bit-identity guards (the whole win rests on these) — forward AND inverse.
        let mut a = Vec::new();
        let mut b = Vec::new();
        fft_1d_into(&row, &mut a);
        plan.apply_into(&row, &mut b);
        let abits: Vec<(u64, u64)> = a.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
        let bbits: Vec<(u64, u64)> = b.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
        assert_eq!(abits, bbits, "cached-plan FFT must be bit-identical to per-row rebuild");
        let mut ai = Vec::new();
        let mut bi = Vec::new();
        ifft_1d_into(&row, &mut ai);
        inv_plan.apply_into(&row, &mut bi);
        let aibits: Vec<(u64, u64)> = ai.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
        let bibits: Vec<(u64, u64)> = bi.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
        assert_eq!(aibits, bibits, "cached-plan IFFT must be bit-identical to per-row rebuild");

        eprintln!(
            "[batched pow2 {rows}x{n}] FWD per-row-rebuild={:.3}ms shared-plan={:.3}ms ratio={:.2}x | INV {:.3}ms->{:.3}ms ratio={:.2}x",
            old as f64 / 1e6,
            new as f64 / 1e6,
            old as f64 / new as f64,
            old_inv as f64 / 1e6,
            new_inv as f64 / 1e6,
            old_inv as f64 / new_inv as f64,
        );
    }

    /// The vectorized SoA real-FFT kernel must be bit-for-bit identical to the
    /// per-row `RealRfftPower2Plan`, across pow2 lengths, batch counts, and the
    /// exact / zero-padded / truncated input regimes.
    #[test]
    fn vectorized_rfft_pow2_bit_identical_to_per_row() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        for &fft_len in &[2usize, 4, 8, 16, 64, 256] {
            let half = fft_len / 2;
            let out_last = half + 1;
            let real_plan = RealRfftPower2Plan::new(fft_len);
            // exact (= fft_len), truncated (> fft_len), padded (< fft_len, >=1)
            let input_lasts = [fft_len, fft_len + 5, (fft_len / 2).max(1)];
            for &input_last in &input_lasts {
                let copy_len = fft_len.min(input_last);
                for &batch in &[1usize, 3, 8, 17, 40] {
                    let elements: Vec<(f64, f64)> = (0..batch * input_last)
                        .map(|i| {
                            let f = i as f64;
                            ((f * 0.017).sin() - 0.3 * (f * 0.005).cos(), 0.0)
                        })
                        .collect();
                    let mut reference = vec![(0.0, 0.0); batch * out_last];
                    let mut packed = Vec::new();
                    let mut transformed = Vec::new();
                    for b in 0..batch {
                        real_plan.apply_into(
                            &elements[b * input_last..b * input_last + input_last],
                            copy_len,
                            &mut packed,
                            &mut transformed,
                            &mut reference[b * out_last..b * out_last + out_last],
                        );
                    }
                    let mut got = vec![(0.0, 0.0); batch * out_last];
                    vectorized_rfft_pow2_tiled(
                        &real_plan, &elements, batch, input_last, copy_len, out_last, &mut got,
                    );
                    assert_eq!(
                        bits(&reference),
                        bits(&got),
                        "rfft SoA != per-row (fft_len={fft_len} input_last={input_last} batch={batch})"
                    );
                }
            }
        }
    }

    /// Same-binary A/B for the SoA real-FFT batch kernel (frankenjax-murmw): OLD =
    /// per-row `RealRfftPower2Plan`; NEW = cache-blocked SoA tiled. Bit-identical
    /// (asserted); run with `--ignored --nocapture` for the single-thread ratio
    /// (the trustworthy signal; threaded FFT A/B is contention-fragile).
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_vectorized_rfft_soa_vs_per_row_plan() {
        let fft_len = 256usize;
        let rows = 2048usize;
        let input_last = fft_len;
        let copy_len = fft_len;
        let out_last = fft_len / 2 + 1;
        let real_plan = RealRfftPower2Plan::new(fft_len);
        let elements: Vec<(f64, f64)> = (0..rows * input_last)
            .map(|i| ((i as f64 * 0.017).sin(), 0.0))
            .collect();

        // Interleave OLD and NEW per iteration so slow cross-run worker drift cancels
        // in the ratio, then take the min of each (least-contended iteration).
        let run_old = || -> (u64, u64) {
            let mut out = vec![(0.0, 0.0); rows * out_last];
            let mut packed = Vec::new();
            let mut transformed = Vec::new();
            let t0 = std::time::Instant::now();
            for r in 0..rows {
                real_plan.apply_into(
                    &elements[r * input_last..r * input_last + input_last],
                    copy_len,
                    &mut packed,
                    &mut transformed,
                    &mut out[r * out_last..r * out_last + out_last],
                );
            }
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * out_last / 2].1.to_bits())
        };
        let run_new = || -> (u64, u64) {
            let mut out = vec![(0.0, 0.0); rows * out_last];
            let t0 = std::time::Instant::now();
            vectorized_rfft_pow2_tiled(
                &real_plan, &elements, rows, input_last, copy_len, out_last, &mut out,
            );
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].0.to_bits() ^ out[rows * out_last / 2].1.to_bits())
        };
        let (mut old_min, mut new_min) = (u64::MAX, u64::MAX);
        let mut chk = 0u64;
        for _ in 0..9 {
            let (o, co) = run_old();
            let (n, cn) = run_new();
            chk ^= co ^ cn;
            old_min = old_min.min(o);
            new_min = new_min.min(n);
        }
        std::hint::black_box(chk);
        eprintln!(
            "[rfft SoA {rows}x{fft_len}] 1T per-row={:.3}ms soa={:.3}ms ratio={:.2}x (min of 9 interleaved)",
            old_min as f64 / 1e6,
            new_min as f64 / 1e6,
            old_min as f64 / new_min as f64,
        );
    }

    /// The vectorized SoA Bluestein kernel must be bit-for-bit identical to the
    /// per-row `BluesteinPlan::apply_into`, across prime / rough lengths, batch
    /// counts, and both FFT and IFFT.
    #[test]
    fn vectorized_bluestein_bit_identical_to_per_row() {
        let bits = |v: &[(f64, f64)]| -> Vec<(u64, u64)> {
            v.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect()
        };
        for &n in &[3usize, 7, 11, 13, 17, 23, 127, 257, 1009, 4099] {
            for inverse in [false, true] {
                let plan = BluesteinPlan::new(n, inverse);
                for &batch in &[1usize, 3, 4, 8, 11] {
                    let elements: Vec<(f64, f64)> = (0..batch * n)
                        .map(|i| {
                            let f = i as f64;
                            ((f * 0.023).sin() - 0.3, (f * 0.017).cos() * 0.7)
                        })
                        .collect();
                    let mut reference = vec![(0.0, 0.0); batch * n];
                    let mut scratch = BluesteinScratch::default();
                    let mut ob = Vec::new();
                    for b in 0..batch {
                        plan.apply_into(&elements[b * n..b * n + n], &mut scratch, &mut ob);
                        reference[b * n..b * n + n].copy_from_slice(&ob);
                    }
                    let got = transform_batches_bluestein_vectorized(&plan, &elements, n, batch);
                    assert_eq!(
                        bits(&reference),
                        bits(&got),
                        "bluestein SoA != per-row (n={n} batch={batch} inverse={inverse})"
                    );
                }
            }
        }
    }

    /// Same-binary A/B for the SoA Bluestein kernel (frankenjax-murmw): OLD = per-row
    /// `BluesteinPlan::apply_into`; NEW = SoA tiled. Interleaved min-of-9
    /// single-thread ratio at a small prime (m=256) and the benchmarked n=1009
    /// (m=2048); run with `--ignored --nocapture`.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_vectorized_bluestein_vs_per_row() {
        for &(n, rows) in &[(127usize, 2048usize), (1009usize, 256usize), (4099usize, 128usize)] {
            let plan = BluesteinPlan::new(n, false);
            let elements: Vec<(f64, f64)> = (0..rows * n)
                .map(|i| {
                    let f = i as f64;
                    ((f * 0.023).sin(), (f * 0.017).cos())
                })
                .collect();
            let run_old = || -> (u64, u64) {
                let mut out = vec![(0.0, 0.0); rows * n];
                let mut scratch = BluesteinScratch::default();
                let mut ob = Vec::new();
                let t0 = std::time::Instant::now();
                for b in 0..rows {
                    plan.apply_into(&elements[b * n..b * n + n], &mut scratch, &mut ob);
                    out[b * n..b * n + n].copy_from_slice(&ob);
                }
                let dt = t0.elapsed().as_nanos() as u64;
                (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
            };
            let run_new = || -> (u64, u64) {
                let t0 = std::time::Instant::now();
                let out = transform_batches_bluestein_vectorized(&plan, &elements, n, rows);
                let dt = t0.elapsed().as_nanos() as u64;
                (dt, out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits())
            };
            let (mut old_min, mut new_min) = (u64::MAX, u64::MAX);
            let mut chk = 0u64;
            for _ in 0..9 {
                let (o, co) = run_old();
                let (nn, cn) = run_new();
                chk ^= co ^ cn;
                old_min = old_min.min(o);
                new_min = new_min.min(nn);
            }
            std::hint::black_box(chk);
            let mm = (2 * n - 1).next_power_of_two();
            eprintln!(
                "[bluestein SoA {rows}x{n} m={mm}] 1T per-row={:.3}ms soa={:.3}ms ratio={:.2}x (min of 9 interleaved)",
                old_min as f64 / 1e6,
                new_min as f64 / 1e6,
                old_min as f64 / new_min as f64,
            );
        }
    }

    /// The vectorized SoA inverse-real-FFT kernel must be bit-for-bit identical to
    /// the per-row `irfft_rows_f64_into`, across pow2 lengths, batch counts, and the
    /// exact / oversized / undersized half-spectrum regimes.
    #[test]
    fn vectorized_irfft_pow2_bit_identical_to_per_row() {
        for &fft_len in &[2usize, 4, 8, 16, 64, 256] {
            let plan_batch = BatchFftPlan::new(fft_len, true);
            let r2 = plan_batch.as_pow2().expect("pow2 plan");
            let input_lasts = [fft_len / 2 + 1, fft_len / 2 + 3, (fft_len / 4).max(1)];
            for &input_last in &input_lasts {
                let copy_len = fft_len.min(input_last);
                for &batch in &[1usize, 3, 8, 17, 40] {
                    let elements: Vec<(f64, f64)> = (0..batch * input_last)
                        .map(|i| {
                            let f = i as f64;
                            ((f * 0.013).sin() - 0.2, (f * 0.021).cos() * 0.5)
                        })
                        .collect();
                    // Reference: per-row production path (rows=1 -> below the SoA floor).
                    let mut reference = vec![0.0f64; batch * fft_len];
                    for b in 0..batch {
                        irfft_rows_f64_into(
                            &elements,
                            Some(&plan_batch),
                            fft_len,
                            input_last,
                            copy_len,
                            b,
                            1,
                            &mut reference[b * fft_len..b * fft_len + fft_len],
                        );
                    }
                    let mut got = vec![0.0f64; batch * fft_len];
                    vectorized_irfft_pow2_tiled(
                        r2, &elements, batch, fft_len, input_last, copy_len, &mut got,
                    );
                    let rb: Vec<u64> = reference.iter().map(|x| x.to_bits()).collect();
                    let gb: Vec<u64> = got.iter().map(|x| x.to_bits()).collect();
                    assert_eq!(
                        rb, gb,
                        "irfft SoA != per-row (fft_len={fft_len} input_last={input_last} batch={batch})"
                    );
                }
            }
        }
    }

    /// Same-binary A/B for the SoA inverse-real-FFT batch kernel (frankenjax-murmw):
    /// OLD = per-row `irfft_rows_f64_into`; NEW = cache-blocked SoA tiled. Run with
    /// `--ignored --nocapture`; interleaved min-of-9 single-thread ratio.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_vectorized_irfft_soa_vs_per_row_plan() {
        let fft_len = 256usize;
        let rows = 2048usize;
        let input_last = fft_len / 2 + 1;
        let copy_len = fft_len.min(input_last);
        let plan_batch = BatchFftPlan::new(fft_len, true);
        let r2 = plan_batch.as_pow2().expect("pow2 plan");
        let elements: Vec<(f64, f64)> = (0..rows * input_last)
            .map(|i| {
                let f = i as f64;
                ((f * 0.013).sin(), (f * 0.021).cos())
            })
            .collect();

        let run_old = || -> (u64, u64) {
            let mut out = vec![0.0f64; rows * fft_len];
            let t0 = std::time::Instant::now();
            // rows=1 per call forces the per-row path (below the SoA floor).
            for b in 0..rows {
                irfft_rows_f64_into(
                    &elements,
                    Some(&plan_batch),
                    fft_len,
                    input_last,
                    copy_len,
                    b,
                    1,
                    &mut out[b * fft_len..b * fft_len + fft_len],
                );
            }
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].to_bits() ^ out[rows * fft_len / 2].to_bits())
        };
        let run_new = || -> (u64, u64) {
            let mut out = vec![0.0f64; rows * fft_len];
            let t0 = std::time::Instant::now();
            vectorized_irfft_pow2_tiled(r2, &elements, rows, fft_len, input_last, copy_len, &mut out);
            let dt = t0.elapsed().as_nanos() as u64;
            (dt, out[0].to_bits() ^ out[rows * fft_len / 2].to_bits())
        };
        let (mut old_min, mut new_min) = (u64::MAX, u64::MAX);
        let mut chk = 0u64;
        for _ in 0..9 {
            let (o, co) = run_old();
            let (n, cn) = run_new();
            chk ^= co ^ cn;
            old_min = old_min.min(o);
            new_min = new_min.min(n);
        }
        std::hint::black_box(chk);
        eprintln!(
            "[irfft SoA {rows}x{fft_len}] 1T per-row={:.3}ms soa={:.3}ms ratio={:.2}x (min of 9 interleaved)",
            old_min as f64 / 1e6,
            new_min as f64 / 1e6,
            old_min as f64 / new_min as f64,
        );
    }

    /// The vectorized SoA batch kernel must be bit-for-bit identical to applying
    /// the scalar `Radix2Plan` per row, for both FFT and IFFT, across sizes and
    /// batch counts (incl. signed zeros / negatives in the data).
    #[test]
    fn vectorized_pow2_batch_bit_identical_to_per_row() {
        for &n in &[2usize, 4, 8, 16, 64, 256] {
            for &batch in &[1usize, 3, 8, 17, 64] {
                let elements: Vec<(f64, f64)> = (0..batch * n)
                    .map(|i| {
                        let f = i as f64;
                        ((f * 0.013).sin() - 0.5, (f * 0.027).cos() * (if i % 3 == 0 { -1.0 } else { 1.0 }))
                    })
                    .collect();
                for inverse in [false, true] {
                    let plan = Radix2Plan::new(n, inverse);
                    // Reference: independent per-row Radix2Plan apply.
                    let mut reference: Vec<(f64, f64)> = vec![(0.0, 0.0); batch * n];
                    let mut buf = Vec::with_capacity(n);
                    for b in 0..batch {
                        plan.apply_into(&elements[b * n..b * n + n], &mut buf);
                        reference[b * n..b * n + n].copy_from_slice(&buf);
                    }
                    // Candidate: vectorized SoA block.
                    let mut re = vec![0.0f64; batch * n];
                    let mut im = vec![0.0f64; batch * n];
                    let mut got: Vec<(f64, f64)> = vec![(0.0, 0.0); batch * n];
                    vectorized_pow2_block(
                        &plan, &elements, batch, n, inverse, &mut re, &mut im, &mut got,
                    );
                    let rbits: Vec<(u64, u64)> =
                        reference.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
                    let gbits: Vec<(u64, u64)> =
                        got.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
                    assert_eq!(
                        rbits, gbits,
                        "vectorized SoA batch FFT must be bit-identical to per-row Radix2Plan (n={n}, batch={batch}, inverse={inverse})"
                    );
                    // And the dense batch dispatcher (which routes to the vectorized
                    // path above the gate) must agree too.
                    let dispatched = transform_batches_dense(&elements, n, batch, inverse);
                    let dbits: Vec<(u64, u64)> =
                        dispatched.iter().map(|&(r, i)| (r.to_bits(), i.to_bits())).collect();
                    assert_eq!(
                        rbits, dbits,
                        "transform_batches_dense must be bit-identical to per-row Radix2Plan (n={n}, batch={batch}, inverse={inverse})"
                    );
                }
            }
        }
    }

    /// Same-binary A/B for the vectorized SoA batch kernel (frankenjax-murmw): the
    /// OLD path runs the cached `Radix2Plan` per row into a row buffer + copy; the
    /// NEW path transposes a row-block to SoA and runs the butterflies vertically
    /// over the batch. Output is bit-identical (asserted); run with
    /// `--ignored --nocapture` for the ratio.
    #[test]
    #[ignore = "informational micro-bench; run with --ignored --nocapture"]
    fn bench_vectorized_soa_batch_vs_per_row_plan() {
        let n = 256usize;
        let rows = 2048usize;
        let elements: Vec<(f64, f64)> = (0..rows * n)
            .map(|i| ((i as f64 * 0.013).sin(), (i as f64 * 0.027).cos()))
            .collect();

        let best = |f: &dyn Fn() -> u64| {
            let mut b = u64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };

        for (label, inverse) in [("FWD", false), ("INV", true)] {
            let plan = Radix2Plan::new(n, inverse);
            // OLD: cached plan applied per row + copy into a full dense output.
            let old = best(&|| {
                let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * n];
                let mut buf = Vec::with_capacity(n);
                for b in 0..rows {
                    plan.apply_into(&elements[b * n..b * n + n], &mut buf);
                    out[b * n..b * n + n].copy_from_slice(&buf);
                }
                out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits()
            });
            // NEW: cache-blocked vectorized SoA batch (single-thread, tiled).
            let new = best(&|| {
                let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * n];
                vectorized_pow2_tiled(&plan, &elements, rows, n, inverse, &mut out);
                out[0].0.to_bits() ^ out[rows * n / 2].1.to_bits()
            });
            eprintln!(
                "[vectorized SoA {label} {rows}x{n}] per-row-plan={:.3}ms soa-batch={:.3}ms ratio={:.2}x",
                old as f64 / 1e6,
                new as f64 / 1e6,
                old as f64 / new as f64,
            );
        }
    }

    #[test]
    fn radix2_twiddle_hoist_bit_identical_to_serial() {
        // The production (twiddle-precomputed) radix-2 must equal the inline-recurrence
        // version bit-for-bit across sizes.
        for log2 in [1u32, 2, 5, 8, 10] {
            let n = 1usize << log2;
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| ((i as f64 * 0.013).sin(), (i as f64 * 0.027).cos()))
                .collect();
            let mut a = Vec::new();
            radix2_fft_1d_into(&input, &mut a, false);
            let mut b = Vec::new();
            radix2_fft_serial_ref(&input, &mut b);
            let bits: Vec<(u64, u64)> = a
                .iter()
                .map(|&(re, im)| (re.to_bits(), im.to_bits()))
                .collect();
            let bref: Vec<(u64, u64)> = b
                .iter()
                .map(|&(re, im)| (re.to_bits(), im.to_bits()))
                .collect();
            assert_eq!(bits, bref, "n={n} hoisted != serial");
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_radix2_twiddle_hoist_vs_serial() {
        use std::time::Instant;
        let n = 1usize << 20;
        let input: Vec<(f64, f64)> = (0..n)
            .map(|i| ((i as f64 * 0.001).sin(), (i as f64 * 0.002).cos()))
            .collect();
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let i1 = input.clone();
        let serial = best(Box::new(move || {
            let mut o = Vec::new();
            radix2_fft_serial_ref(&i1, &mut o);
            o.len()
        }));
        let i2 = input.clone();
        let hoist = best(Box::new(move || {
            let mut o = Vec::new();
            radix2_fft_1d_into(&i2, &mut o, false);
            o.len()
        }));
        println!(
            "BENCH radix2 FFT [2^20] twiddle-hoist: serial={:.2}ms hoist={:.2}ms speedup={:.2}x",
            serial * 1e3,
            hoist * 1e3,
            serial / hoist
        );
    }

    fn make_real_vector(data: &[f64]) -> Value {
        let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        Value::Tensor(TensorValue::new(DType::F64, shape, elements).unwrap())
    }

    fn make_complex_vector(data: &[(f64, f64)]) -> Value {
        let elements: Vec<Literal> = data
            .iter()
            .map(|&(re, im)| Literal::from_complex128(re, im))
            .collect();
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        // Boxed (Literal-backed) complex reference: `TensorValue::new` densifies
        // homogeneous complex literals (cbea72b3); `crate::new_boxed` keeps it
        // boxed so the dense-input path is compared against a true literal path.
        Value::Tensor(crate::new_boxed(DType::Complex128, shape, elements).unwrap())
    }

    fn make_complex64_vector(data: &[(f32, f32)]) -> Value {
        let elements: Vec<Literal> = data
            .iter()
            .map(|&(re, im)| Literal::from_complex64(re, im))
            .collect();
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        Value::Tensor(TensorValue::new(DType::Complex64, shape, elements).unwrap())
    }

    /// Build the same logical complex128 vector backed by dense `(f64, f64)`
    /// storage (the new `as_complex_slice` fast path) rather than `Literal`s.
    fn make_complex_vector_dense(data: &[(f64, f64)]) -> Value {
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        Value::Tensor(
            TensorValue::new_complex_values(DType::Complex128, shape, data.to_vec()).unwrap(),
        )
    }

    /// Read every element's (re, im) raw bits — the bit-exact fingerprint of a
    /// complex tensor, independent of storage representation.
    fn complex_bits(v: &Value) -> Vec<(u64, u64)> {
        v.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|lit| match lit {
                Literal::Complex128Bits(re, im) => (*re, *im),
                Literal::Complex64Bits(re, im) => (u64::from(*re), u64::from(*im)),
                other => panic!("expected complex literal, got {other:?}"),
            })
            .collect()
    }

    /// Isomorphism proof for frankenjax-6294o: the dense `as_complex_slice` input
    /// path and the `Literal`-backed input path must produce bit-identical FFT
    /// outputs. Covers fft / ifft / rfft over power-of-two and Bluestein lengths.
    #[test]
    fn dense_complex_input_bit_identical_to_literal_input() {
        let p = BTreeMap::new();
        // Lengths: 8 (radix-2) and 6 (Bluestein, non-power-of-two).
        for len in [8_usize, 6] {
            let data: Vec<(f64, f64)> = (0..len)
                .map(|i| {
                    let x = i as f64;
                    ((x * 0.125).sin(), (x * 0.25).cos() - 0.3 * x)
                })
                .collect();
            let lit = make_complex_vector(&data);
            let dense = make_complex_vector_dense(&data);
            assert!(
                dense
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_complex_slice()
                    .is_some(),
                "dense input should expose as_complex_slice"
            );
            assert!(
                lit.as_tensor()
                    .unwrap()
                    .elements
                    .as_complex_slice()
                    .is_none(),
                "literal input must not be dense-complex-backed"
            );
            for prim in [Primitive::Fft, Primitive::Ifft] {
                let from_lit = eval_fft_dispatch(prim, &lit, &p);
                let from_dense = eval_fft_dispatch(prim, &dense, &p);
                assert_eq!(
                    complex_bits(&from_lit),
                    complex_bits(&from_dense),
                    "{prim:?} len={len}: dense-input output must match literal-input bit-for-bit"
                );
            }
        }
    }

    fn eval_fft_dispatch(prim: Primitive, input: &Value, p: &BTreeMap<String, String>) -> Value {
        match prim {
            Primitive::Fft => eval_fft(std::slice::from_ref(input), p).unwrap(),
            Primitive::Ifft => eval_ifft(std::slice::from_ref(input), p).unwrap(),
            _ => unreachable!(),
        }
    }

    /// Threading isomorphism proof for frankenjax-6294o: a batched FFT large
    /// enough to fan out across threads (`rows*cols >= 1<<18`, `rows > 1`) must
    /// produce, row for row, exactly the same bits as transforming each row on
    /// its own (which always takes the serial path, `batch_size == 1`). Any
    /// row-partitioning or shared-state bug would diverge here.
    #[test]
    fn threaded_batch_fft_matches_per_row_serial() {
        let p = BTreeMap::new();
        let rows = 512_usize;
        let cols = 512_usize; // 512*512 = 262_144 = 1<<18 -> threaded path
        assert!(rows * cols >= (1usize << 18) && rows > 1);

        let row_data: Vec<Vec<(f64, f64)>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let x = (r * cols + c) as f64;
                        ((x * 0.0009765625).sin(), (x * 0.001953125).cos() - 0.1)
                    })
                    .collect()
            })
            .collect();

        for prim in [Primitive::Fft, Primitive::Ifft] {
            // Batched (threaded) evaluation over a dense rows×cols input.
            let flat: Vec<(f64, f64)> = row_data.iter().flatten().copied().collect();
            let batched_input = Value::Tensor(
                TensorValue::new_complex_values(
                    DType::Complex128,
                    Shape {
                        dims: vec![rows as u32, cols as u32],
                    },
                    flat,
                )
                .unwrap(),
            );
            let batched = eval_fft_dispatch(prim, &batched_input, &p);
            let batched_bits = complex_bits(&batched);

            // Per-row serial reference.
            let mut expected: Vec<(u64, u64)> = Vec::with_capacity(rows * cols);
            for data in &row_data {
                let single = make_complex_vector_dense(data);
                expected.extend(complex_bits(&eval_fft_dispatch(prim, &single, &p)));
            }

            assert_eq!(
                batched_bits, expected,
                "{prim:?}: threaded batch output must match per-row serial bit-for-bit"
            );
        }
    }

    /// Threading isomorphism proof for the dense RFFT: a batched real FFT large
    /// enough to fan out across threads must produce, row for row, exactly the
    /// same bits as transforming each real row on its own (serial path).
    #[test]
    fn threaded_batch_rfft_matches_per_row_serial() {
        let p = BTreeMap::new();
        let rows = 2048_usize;
        let cols = 256_usize; // out_last=129; 2048*129 = 264_192 >= 1<<18 -> threaded
        let row_data: Vec<Vec<f64>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let x = (r * cols + c) as f64;
                        (x * 0.0009765625).sin() + 0.3 * (x * 0.002).cos()
                    })
                    .collect()
            })
            .collect();
        let out_last = cols / 2 + 1;
        assert!(rows * out_last >= (1usize << 18) && rows > 1);

        let flat: Vec<f64> = row_data.iter().flatten().copied().collect();
        let batched_input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                flat,
            )
            .unwrap(),
        );
        let batched = eval_rfft(std::slice::from_ref(&batched_input), &p).unwrap();
        let batched_bits = complex_bits(&batched);

        let mut expected: Vec<(u64, u64)> = Vec::with_capacity(rows * out_last);
        for data in &row_data {
            let single = make_real_vector(data);
            let single_out = eval_rfft(std::slice::from_ref(&single), &p).unwrap();
            expected.extend(complex_bits(&single_out));
        }

        assert_eq!(
            batched_bits, expected,
            "threaded batch RFFT must match per-row serial bit-for-bit"
        );
    }

    #[test]
    fn threaded_bluestein_batch_rfft_matches_serial_row_block() {
        let p = BTreeMap::new();
        let rows = 256_usize;
        let cols = 257_usize;
        let out_last = cols / 2 + 1;
        let plan = BatchFftPlan::new(cols, false);
        assert!(rows * out_last < (1usize << 18));
        assert!(rows * plan.work_len(cols) >= (1usize << 18));

        let data: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let x = i as f64;
                (x * 0.001953125).sin() - 0.25 * (x * 0.00390625).cos()
            })
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let batched = eval_rfft(std::slice::from_ref(&input), &p).unwrap();
        let batched_bits = complex_bits(&batched);
        let batched_digest =
            fj_test_utils::fixture_id_from_json(&batched_bits).expect("RFFT digest should build");

        let elements: Vec<(f64, f64)> = data.into_iter().map(|value| (value, 0.0)).collect();
        let mut expected = vec![(0.0, 0.0); rows * out_last];
        rfft_rows_into(
            &elements,
            Some(&plan),
            None,
            cols,
            cols,
            cols,
            out_last,
            0,
            rows,
            &mut expected,
        );
        let expected_bits: Vec<(u64, u64)> = expected
            .iter()
            .map(|(re, im)| (re.to_bits(), im.to_bits()))
            .collect();

        assert_eq!(
            batched_bits, expected_bits,
            "threaded Bluestein RFFT must match the serial packed row block bit-for-bit"
        );
        assert_eq!(
            batched_digest, "11378141236573f4d7b1e8a2e85465e2798b1c608ed0dbc17c308e948628292e",
            "threaded Bluestein RFFT golden output digest changed"
        );
    }

    /// Threading isomorphism proof for the dense IRFFT: a batched inverse real FFT
    /// large enough to fan out across threads must produce, row for row, exactly
    /// the same bits as inverting each row's half-spectrum on its own (serial).
    #[test]
    fn threaded_batch_irfft_matches_per_row_serial() {
        let mut rp = BTreeMap::new();
        rp.insert("fft_length".to_owned(), "256".to_owned());
        let rows = 2048usize;
        let cols = 256usize; // output 2048*256 = 524288 >= 1<<17 -> threaded
        assert!(rows * cols >= (1usize << 17) && rows > 1);

        let real_rows: Vec<Vec<f64>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let x = (r * cols + c) as f64;
                        (x * 0.0009765625).sin() + 0.4 * (x * 0.002).cos()
                    })
                    .collect()
            })
            .collect();
        let flat: Vec<f64> = real_rows.iter().flatten().copied().collect();
        let real_mat = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                flat,
            )
            .unwrap(),
        );
        let batched_spectrum = eval_rfft(std::slice::from_ref(&real_mat), &rp).unwrap();
        let batched = eval_irfft(std::slice::from_ref(&batched_spectrum), &rp).unwrap();
        let got: Vec<u64> = batched
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap().to_bits())
            .collect();

        let mut expect: Vec<u64> = Vec::with_capacity(rows * cols);
        for data in &real_rows {
            let v = make_real_vector(data);
            let spec = eval_rfft(std::slice::from_ref(&v), &rp).unwrap();
            let inv = eval_irfft(std::slice::from_ref(&spec), &rp).unwrap();
            expect.extend(
                inv.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap().to_bits()),
            );
        }
        assert_eq!(
            got, expect,
            "threaded batch IRFFT must match per-row serial bit-for-bit"
        );
    }

    #[test]
    fn half_length_rfft_matches_full_fft_reference() {
        for &(fft_length, input_len) in &[(2_usize, 1_usize), (4, 4), (8, 5), (16, 20), (256, 193)]
        {
            let data: Vec<f64> = (0..input_len)
                .map(|i| {
                    let x = i as f64;
                    (x * 0.03125).sin() - 0.4 * (x * 0.046875).cos()
                })
                .collect();
            let input = make_real_vector(&data);
            let mut params = BTreeMap::new();
            params.insert("fft_length".to_owned(), fft_length.to_string());
            let actual = eval_rfft(std::slice::from_ref(&input), &params).unwrap();
            let actual = extract_complex_elements(&actual);

            let mut padded = vec![(0.0, 0.0); fft_length];
            for (slot, &value) in padded.iter_mut().zip(data.iter()) {
                slot.0 = value;
            }
            let expected = fft_1d(&padded);
            assert_complex_close(&actual, &expected[..fft_length / 2 + 1], 1e-9);
        }
    }

    /// Golden guard for the half-length real FFT path. The path intentionally
    /// reorders floating-point operations versus the old full-FFT implementation,
    /// so this pins the chosen safe-Rust algorithm rather than claiming bit-level
    /// isomorphism to the previous kernel.
    #[test]
    fn half_length_rfft_golden_output_digest() {
        let p = BTreeMap::new();
        let (rows, cols) = (64_usize, 256_usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let x = i as f64;
                (x * 0.0078125).sin() + 0.3 * (x * 0.015625).cos()
            })
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .unwrap(),
        );
        let out = eval_rfft(std::slice::from_ref(&input), &p).unwrap();
        let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
        for (re, im) in complex_bits(&out) {
            for byte in re.to_le_bytes().into_iter().chain(im.to_le_bytes()) {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(
            digest, 0xbd9e_108f_2d66_fe21,
            "RFFT half-length golden output digest changed"
        );
    }

    /// Golden-output regression guard for frankenjax-6294o. Pins the bit-exact
    /// output of a canonical 64×128 complex128 FFT through the dense+threaded
    /// path. The little-endian (re,im) bit stream of the 8192-element output has
    ///   sha256 = e08a6c8279950e499a2619baa424616e9a205eafd8727223aacdb633026a356a
    /// (131072 bytes); the inline FNV-1a-64 digest below is a self-contained
    /// proxy so the guard needs no `sha2` dependency. Any numeric drift in the
    /// transform, the dense storage round-trip, or threading changes this digest.
    #[test]
    fn fft_golden_output_digest() {
        let p = BTreeMap::new();
        let (rows, cols) = (64_usize, 128_usize);
        let data: Vec<(f64, f64)> = (0..rows * cols)
            .map(|i| {
                let x = i as f64;
                ((x * 0.0078125).sin(), (x * 0.015625).cos() - 0.2)
            })
            .collect();
        let input = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .unwrap(),
        );
        let out = eval_fft(std::slice::from_ref(&input), &p).unwrap();
        let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
        for (re, im) in complex_bits(&out) {
            for byte in re.to_le_bytes().into_iter().chain(im.to_le_bytes()) {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(
            digest, 0xe981_f3ae_df00_1c51,
            "FFT golden output digest changed — behavior is NOT isomorphic"
        );
    }

    fn oversized_fft_length_params() -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert(
            "fft_length".to_owned(),
            (u64::from(u32::MAX) + 1).to_string(),
        );
        params
    }

    fn assert_oversized_fft_length_error(err: EvalError, expected_primitive: Primitive) {
        match err {
            EvalError::Unsupported { primitive, detail } => {
                assert_eq!(primitive, expected_primitive);
                assert!(
                    detail.contains("fft_length") && detail.contains("exceeds u32"),
                    "unexpected detail: {detail}"
                );
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    fn extract_complex_elements(v: &Value) -> Vec<(f64, f64)> {
        match v {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|l| literal_to_complex(*l).unwrap())
                .collect(),
            Value::Scalar(l) => vec![literal_to_complex(*l).unwrap()],
        }
    }

    fn extract_f64_elements(v: &Value) -> Vec<f64> {
        match v {
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
            Value::Scalar(l) => vec![l.as_f64().unwrap()],
        }
    }

    fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&(ar, ai), &(er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ar - er).abs() < tol && (ai - ei).abs() < tol,
                "element {i}: got ({ar}, {ai}), expected ({er}, {ei})"
            );
        }
    }

    #[test]
    fn radix2_plan_matches_unplanned_path_by_bits() {
        for &n in &[1_usize, 2, 4, 8, 16, 256] {
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    ((x * 0.19).sin() - 0.25 * x, (x * 0.13).cos() + 0.5)
                })
                .collect();

            for inverse in [false, true] {
                let mut unplanned = Vec::new();
                radix2_fft_1d_into(&input, &mut unplanned, inverse);

                let plan = Radix2Plan::new(n, inverse);
                let mut planned = Vec::new();
                plan.apply_into(&input, &mut planned);

                assert_eq!(planned.len(), unplanned.len());
                for (idx, (&(ar, ai), &(er, ei))) in
                    planned.iter().zip(unplanned.iter()).enumerate()
                {
                    assert_eq!(
                        ar.to_bits(),
                        er.to_bits(),
                        "real mismatch at n={n} idx={idx}"
                    );
                    assert_eq!(
                        ai.to_bits(),
                        ei.to_bits(),
                        "imag mismatch at n={n} idx={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_row_smooth_fft_routes_mixed_radix_matching_dft() {
        // The single-row fft_1d/ifft_1d path now routes smooth composites through
        // mixed-radix (consistent with the batched path) instead of Bluestein. Verify
        // the public path matches the direct DFT to tolerance for sizes that exercise
        // mixed-radix (smooth) AND a prime that stays on Bluestein, plus round-trip.
        for &n in &[18usize, 24, 36, 50, 81, 125, 250, 1080] {
            assert!(is_mixed_radix_smooth(n), "n={n} should be smooth");
            let input: Vec<(f64, f64)> = (0..n)
                .map(|j| {
                    (
                        (j as f64 * 0.41).cos() * 3.0 - 0.7,
                        (j as f64 * 0.23).sin() + 0.1,
                    )
                })
                .collect();
            assert_complex_close(&fft_1d(&input), &dft_1d(&input), 1e-9);
            assert_complex_close(&ifft_1d(&input), &idft_1d(&input), 1e-9);
            assert_complex_close(&ifft_1d(&fft_1d(&input)), &input, 1e-9);
        }
        // A prime length stays on Bluestein and must still match the DFT.
        let p = 59usize;
        assert!(!is_mixed_radix_smooth(p));
        let pin: Vec<(f64, f64)> = (0..p)
            .map(|j| (j as f64 * 0.1, -(j as f64) * 0.05))
            .collect();
        assert_complex_close(&fft_1d(&pin), &dft_1d(&pin), 1e-9);
    }

    #[test]
    fn mixed_radix_matches_dft_for_smooth_sizes() {
        // The mixed-radix path must equal the O(n²) DFT reference (and round-trip
        // through the inverse) for smooth composite lengths.
        for &n in &[6usize, 12, 15, 63, 100, 720, 945, 1000] {
            assert!(
                is_mixed_radix_smooth(n),
                "n={n} should take the mixed-radix path"
            );
            let input: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    (
                        ((i as f64) * 0.3).sin() * 2.0,
                        ((i as f64) * 0.17).cos() - 0.2,
                    )
                })
                .collect();

            let mut scratch = Vec::new();
            let mut got = Vec::new();
            mixed_radix_into(
                &input,
                &precompute_twiddles(n, false),
                false,
                &mut got,
                &mut scratch,
            );
            assert_complex_close(&got, &dft_1d(&input), 1e-9);

            let mut back = Vec::new();
            mixed_radix_into(
                &got,
                &precompute_twiddles(n, true),
                true,
                &mut back,
                &mut scratch,
            );
            assert_complex_close(&back, &input, 1e-9);
        }
        // Prime / large-prime-factor / power-of-two lengths stay off the path.
        assert!(!is_mixed_radix_smooth(1009)); // prime
        assert!(!is_mixed_radix_smooth(34)); // 2·17, factor 17 > 13
        assert!(!is_mixed_radix_smooth(1024)); // power of two
    }

    #[test]
    fn fft_power_of_two_fast_path_matches_dft_reference() {
        let input = [
            (1.0, 0.5),
            (2.0, -1.0),
            (-0.25, 3.0),
            (0.0, 0.75),
            (4.0, -2.0),
            (-3.0, 1.25),
            (0.5, -0.5),
            (2.25, 0.0),
        ];
        assert_complex_close(&fft_1d(&input), &dft_1d(&input), 1e-10);
    }

    #[test]
    fn ifft_power_of_two_fast_path_matches_idft_reference() {
        let input = [
            (1.0, 0.5),
            (2.0, -1.0),
            (-0.25, 3.0),
            (0.0, 0.75),
            (4.0, -2.0),
            (-3.0, 1.25),
            (0.5, -0.5),
            (2.25, 0.0),
        ];
        assert_complex_close(&ifft_1d(&input), &idft_1d(&input), 1e-10);
    }

    #[test]
    fn non_power_of_two_lengths_match_dft_reference() {
        // Non-power-of-two now routes through Bluestein (O(n log n)); it agrees
        // with the direct O(n²) DFT/IDFT reference to floating tolerance.
        let input = [
            (1.0, 0.5),
            (2.0, -1.0),
            (-0.25, 3.0),
            (0.0, 0.75),
            (4.0, -2.0),
            (-3.0, 1.25),
        ];
        assert_complex_close(&fft_1d(&input), &dft_1d(&input), 1e-10);
        assert_complex_close(&ifft_1d(&input), &idft_1d(&input), 1e-10);
    }

    #[test]
    fn bluestein_matches_direct_dft_across_sizes() {
        // Bluestein vs the direct DFT/IDFT reference across assorted
        // non-power-of-two lengths, including primes and composites.
        for &n in &[3usize, 5, 6, 7, 9, 10, 11, 12, 15, 17, 100, 257, 1000] {
            let input: Vec<(f64, f64)> = (0..n)
                .map(|j| {
                    let j = j as f64;
                    ((j * 0.37).sin() * 2.0 - 0.5, (j * 0.11).cos() + 0.25 * j)
                })
                .collect();
            assert_complex_close(&fft_1d(&input), &dft_1d(&input), 1e-9);
            assert_complex_close(&ifft_1d(&input), &idft_1d(&input), 1e-9);
            // Round-trip: ifft(fft(x)) == x.
            let round = ifft_1d(&fft_1d(&input));
            assert_complex_close(&round, &input, 1e-9);
        }
    }

    // ── FFT tests ────────────────────────────────────────────

    #[test]
    fn fft_dc_signal() {
        // All-ones input: DFT should give [n, 0, 0, 0]
        let input = make_real_vector(&[1.0, 1.0, 1.0, 1.0]);
        let result = eval_fft(&[input], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        assert_complex_close(
            &elems,
            &[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1e-10,
        );
    }

    #[test]
    fn fft_impulse() {
        // Impulse [1, 0, 0, 0]: DFT should be all ones
        let input = make_real_vector(&[1.0, 0.0, 0.0, 0.0]);
        let result = eval_fft(&[input], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        assert_complex_close(
            &elems,
            &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
            1e-10,
        );
    }

    #[test]
    fn fft_known_4point() {
        // x = [1, 2, 3, 4]
        // X[0] = 10, X[1] = -2+2i, X[2] = -2, X[3] = -2-2i
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let result = eval_fft(&[input], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        assert_complex_close(
            &elems,
            &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            1e-10,
        );
    }

    #[test]
    fn fft_rejects_scalar() {
        let scalar = Value::Scalar(Literal::from_f64(1.0));
        assert!(eval_fft(&[scalar], &BTreeMap::new()).is_err());
    }

    #[test]
    fn fft_rejects_empty_last_axis_without_panicking() {
        let input = make_real_vector(&[]);
        let err = eval_fft(&[input], &BTreeMap::new()).expect_err("empty FFT input must fail");
        assert!(
            err.to_string().contains("last dimension > 0"),
            "unexpected error: {err}"
        );
    }

    // ── IFFT tests ───────────────────────────────────────────

    #[test]
    fn ifft_roundtrip() {
        // FFT then IFFT should recover original signal
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let fft_result = eval_fft(&[input], &BTreeMap::new()).unwrap();
        let ifft_result = eval_ifft(&[fft_result], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&ifft_result);
        assert_complex_close(
            &elems,
            &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
            1e-10,
        );
    }

    #[test]
    fn ifft_known_spectrum() {
        // IFFT of [4, 0, 0, 0] should give [1, 1, 1, 1]
        let input = make_complex_vector(&[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let result = eval_ifft(&[input], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        assert_complex_close(
            &elems,
            &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
            1e-10,
        );
    }

    #[test]
    fn ifft_rejects_real_input() {
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let err = eval_ifft(&[input], &BTreeMap::new()).expect_err("real IFFT input must fail");
        assert!(matches!(
            err,
            EvalError::Unsupported {
                primitive: Primitive::Ifft,
                ..
            }
        ));
        assert!(
            err.to_string().contains("complex-valued input"),
            "unexpected error: {err}"
        );
    }

    // ── RFFT tests ───────────────────────────────────────────

    #[test]
    fn rfft_basic() {
        // x = [1, 2, 3, 4], fft_length=4
        // Full FFT: [10, -2+2i, -2, -2-2i]
        // RFFT keeps first 3: [10, -2+2i, -2]
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), "4".to_owned());
        let result = eval_rfft(&[input], &params).unwrap();
        let elems = extract_complex_elements(&result);
        assert_eq!(elems.len(), 3);
        assert_complex_close(&elems, &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)], 1e-10);
    }

    #[test]
    fn rfft_paired_batch_matches_per_row_nonpow2() {
        // Non-power-of-two length (7 ⇒ Bluestein) with an ODD batch count (5) so both
        // the row-pairing path and the leftover single-row tail run. Each row's
        // spectrum must equal the independent single-row RFFT to tolerance — the
        // isomorphism guard for the paired non-pow2 batch path.
        let rows = 5usize;
        let n = 7usize;
        let data: Vec<f64> = (0..rows * n)
            .map(|i| (((i * 13 + 1) % 9) as f64) - 4.0)
            .collect();
        let batched = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, n as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        let out = eval_rfft(&[batched], &BTreeMap::new()).unwrap();
        let got = extract_complex_elements(&out);
        let out_last = n / 2 + 1;
        for r in 0..rows {
            let row = make_real_vector(&data[r * n..(r + 1) * n]);
            let row_out = eval_rfft(&[row], &BTreeMap::new()).unwrap();
            let row_got = extract_complex_elements(&row_out);
            assert_complex_close(&got[r * out_last..(r + 1) * out_last], &row_got, 1e-12);
        }
    }

    #[test]
    fn rfft_default_length() {
        // Without fft_length param, defaults to input length
        let input = make_real_vector(&[1.0, 0.0, 0.0, 0.0]);
        let result = eval_rfft(&[input], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        // Impulse FFT: all ones, keep first 3
        assert_eq!(elems.len(), 3);
        assert_complex_close(&elems, &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], 1e-10);
    }

    #[test]
    fn rfft_accepts_fft_lengths_alias() {
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("fft_lengths".to_owned(), "4".to_owned());
        let result = eval_rfft(&[input], &params).unwrap();
        let elems = extract_complex_elements(&result);
        assert_eq!(elems.len(), 3);
        assert_complex_close(&elems, &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)], 1e-10);
    }

    #[test]
    fn rfft_accepts_fft_lengths_list() {
        let input = make_real_vector(&[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("fft_lengths".to_owned(), "2, 4".to_owned());
        let result = eval_rfft(&[input], &params).unwrap();
        let elems = extract_complex_elements(&result);
        assert_eq!(elems.len(), 3);
        assert_complex_close(&elems, &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)], 1e-10);
    }

    #[test]
    fn rfft_rejects_complex_input() {
        let input = make_complex_vector(&[(1.0, 0.0), (0.0, 1.0), (2.0, -1.0), (3.0, 0.5)]);
        let err = eval_rfft(&[input], &BTreeMap::new()).expect_err("complex RFFT input must fail");
        assert!(matches!(
            err,
            EvalError::Unsupported {
                primitive: Primitive::Rfft,
                ..
            }
        ));
        assert!(
            err.to_string().contains("real-valued input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rfft_rejects_oversized_fft_length_before_allocation() {
        let input = make_real_vector(&[1.0]);
        let err = eval_rfft(&[input], &oversized_fft_length_params())
            .expect_err("oversized RFFT length must fail before allocation");
        assert_oversized_fft_length_error(err, Primitive::Rfft);
    }

    // ── IRFFT tests ──────────────────────────────────────────

    #[test]
    fn irfft_basic() {
        // Half-spectrum [10, -2+2i, -2] with fft_length=4
        // Full spectrum: [10, -2+2i, -2, -2-2i]
        // IDFT should recover [1, 2, 3, 4]
        let input = make_complex_vector(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]);
        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), "4".to_owned());
        let result = eval_irfft(&[input], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert_eq!(elems.len(), 4);
        for (i, (&got, &expected)) in elems.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-10,
                "element {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn rfft_irfft_roundtrip() {
        // RFFT then IRFFT should recover the original signal
        let original = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let input = make_real_vector(&original);

        let mut rfft_params = BTreeMap::new();
        rfft_params.insert("fft_length".to_owned(), "8".to_owned());
        let rfft_result = eval_rfft(&[input], &rfft_params).unwrap();

        let mut irfft_params = BTreeMap::new();
        irfft_params.insert("fft_length".to_owned(), "8".to_owned());
        let irfft_result = eval_irfft(&[rfft_result], &irfft_params).unwrap();

        let elems = extract_f64_elements(&irfft_result);
        assert_eq!(elems.len(), 8);
        for (i, (&got, &expected)) in elems.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-10,
                "element {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn irfft_rejects_real_input() {
        let input = make_real_vector(&[1.0, 2.0, 3.0]);
        let err = eval_irfft(&[input], &BTreeMap::new()).expect_err("real IRFFT input must fail");
        assert!(matches!(
            err,
            EvalError::Unsupported {
                primitive: Primitive::Irfft,
                ..
            }
        ));
        assert!(
            err.to_string().contains("complex-valued input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn irfft_rejects_oversized_fft_length_before_allocation() {
        let input = make_complex_vector(&[(1.0, 0.0)]);
        let err = eval_irfft(&[input], &oversized_fft_length_params())
            .expect_err("oversized IRFFT length must fail before allocation");
        assert_oversized_fft_length_error(err, Primitive::Irfft);
    }

    #[test]
    fn irfft_complex64_output_uses_f32_literals() {
        let input = make_complex64_vector(&[(1.0, 0.0), (0.0, 0.0)]);
        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), "2".to_owned());

        let result = eval_irfft(&[input], &params).unwrap();
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                tensor
                    .validate_dtype_consistency()
                    .expect("IRFFT Complex64 output dtype/element invariant");
            }
            Value::Scalar(_) => panic!("IRFFT must return a tensor"),
        }
    }

    #[test]
    fn irfft_accepts_fft_lengths_alias() {
        let input = make_complex_vector(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]);
        let mut params = BTreeMap::new();
        params.insert("fft_lengths".to_owned(), "4".to_owned());
        let result = eval_irfft(&[input], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert_eq!(elems.len(), 4);
        for (i, (&got, &expected)) in elems.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-10,
                "element {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn irfft_accepts_fft_lengths_list() {
        let input = make_complex_vector(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]);
        let mut params = BTreeMap::new();
        params.insert("fft_lengths".to_owned(), "2, 4".to_owned());
        let result = eval_irfft(&[input], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert_eq!(elems.len(), 4);
        for (i, (&got, &expected)) in elems.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-10,
                "element {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn fft_batched_2d() {
        // 2×4 tensor: FFT along last axis for each row independently
        let elements: Vec<Literal> = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            .iter()
            .map(|&v| Literal::from_f64(v))
            .collect();
        let shape = Shape { dims: vec![2, 4] };
        let tensor = Value::Tensor(TensorValue::new(DType::F64, shape, elements).unwrap());

        let result = eval_fft(&[tensor], &BTreeMap::new()).unwrap();
        let elems = extract_complex_elements(&result);
        assert_eq!(elems.len(), 8);

        // Row 0: [1,1,1,1] → [4, 0, 0, 0]
        assert_complex_close(
            &elems[0..4],
            &[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1e-10,
        );
        // Row 1: [1,0,0,0] → [1, 1, 1, 1]
        assert_complex_close(
            &elems[4..8],
            &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
            1e-10,
        );
    }
}
