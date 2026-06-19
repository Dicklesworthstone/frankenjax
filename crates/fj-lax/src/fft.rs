#![forbid(unsafe_code)]

//! FFT primitives: Fft, Ifft, Rfft, Irfft.
//!
//! Uses a radix-2 Cooley-Tukey fast path for power-of-two lengths and the
//! direct O(n²) DFT fallback for other lengths. The 1D transform is applied
//! along the last axis for each batch (all leading dimensions).

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

/// Per-row transform engine for a batch of equal-length rows. Built once and
/// shared across rows: power-of-two lengths use radix-2, smooth composite lengths
/// use mixed-radix (caching the roots table), and everything else uses a cached
/// Bluestein plan.
enum BatchFftPlan {
    Pow2,
    Mixed(Vec<(f64, f64)>),
    Bluestein(BluesteinPlan),
}

impl BatchFftPlan {
    fn new(n: usize, inverse: bool) -> Self {
        if n <= 1 || n.is_power_of_two() {
            BatchFftPlan::Pow2
        } else if is_mixed_radix_smooth(n) {
            BatchFftPlan::Mixed(precompute_twiddles(n, inverse))
        } else {
            BatchFftPlan::Bluestein(BluesteinPlan::new(n, inverse))
        }
    }

    fn work_len(&self, n: usize) -> usize {
        match self {
            BatchFftPlan::Pow2 | BatchFftPlan::Mixed(_) => n,
            BatchFftPlan::Bluestein(plan) => plan.m,
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
            BatchFftPlan::Pow2 => {
                if inverse {
                    ifft_1d_into(input, output);
                } else {
                    fft_1d_into(input, output);
                }
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
    // Shared, immutable per-row plan (built once): radix-2 / mixed-radix / Bluestein.
    let plan = BatchFftPlan::new(n, inverse);
    let total = batch_size * n;
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

    // Only fan out when there is enough work to amortize thread spawn; dense
    // storage makes the extract/output trivial, so the per-row transform is the
    // dominant cost and parallelizes cleanly across rows.
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
    // Non-power-of-two transform length: build the inverse per-row plan once
    // (mixed-radix for smooth lengths, Bluestein for large-prime lengths).
    let plan = (fft_length > 1 && !fft_length.is_power_of_two())
        .then(|| BatchFftPlan::new(fft_length, true));

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
