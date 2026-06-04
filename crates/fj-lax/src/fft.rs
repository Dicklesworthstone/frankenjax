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

/// Build a complex Literal from (re, im) in the given complex dtype.
fn make_complex_literal(re: f64, im: f64, dtype: DType) -> Literal {
    match dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        _ => Literal::from_complex128(re, im),
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
#[cfg(test)]
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
}

/// Reusable per-row work buffers for [`BluesteinPlan::apply_into`].
#[derive(Default)]
struct BluesteinScratch {
    a: Vec<(f64, f64)>,
    fa: Vec<(f64, f64)>,
    prod: Vec<(f64, f64)>,
    conv: Vec<(f64, f64)>,
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
        radix2_fft_1d_into(&b, &mut fb, false);

        Self {
            n,
            m,
            inverse,
            chirp,
            fb,
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

        radix2_fft_1d_into(a, fa, false);

        prod.clear();
        prod.resize(m, (0.0, 0.0));
        for (p, (&(ar, ai), &(br, bi))) in prod.iter_mut().zip(fa.iter().zip(self.fb.iter())) {
            *p = (ar * br - ai * bi, ar * bi + ai * br);
        }

        radix2_fft_1d_into(prod, conv, true);

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

fn radix2_fft_1d_into(input: &[(f64, f64)], output: &mut Vec<(f64, f64)>, inverse: bool) {
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
        let angle = if inverse {
            2.0 * PI / (len as f64)
        } else {
            -2.0 * PI / (len as f64)
        };
        let (sin_step, cos_step) = angle.sin_cos();

        for start in (0..n).step_by(len) {
            let mut twiddle_re = 1.0;
            let mut twiddle_im = 0.0;

            for offset in 0..half {
                let even = output[start + offset];
                let odd = output[start + offset + half];
                let rotated_re = odd.0 * twiddle_re - odd.1 * twiddle_im;
                let rotated_im = odd.0 * twiddle_im + odd.1 * twiddle_re;

                output[start + offset] = (even.0 + rotated_re, even.1 + rotated_im);
                output[start + offset + half] = (even.0 - rotated_re, even.1 - rotated_im);

                let next_re = twiddle_re * cos_step - twiddle_im * sin_step;
                let next_im = twiddle_re * sin_step + twiddle_im * cos_step;
                twiddle_re = next_re;
                twiddle_im = next_im;
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
    if input.len().is_power_of_two() {
        radix2_fft_1d_into(input, output, false);
    } else {
        // Non-power-of-two: Bluestein O(n log n) instead of the O(n²) direct DFT.
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
    if input.len().is_power_of_two() {
        radix2_fft_1d_into(input, output, true);
    } else {
        // Non-power-of-two: Bluestein O(n log n) instead of the O(n²) direct IDFT.
        bluestein_dft_into(input, output, true);
    }
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

    let mut out_elements = Vec::with_capacity(elements.len());
    let mut transform_buf = Vec::with_capacity(n);
    // Non-power-of-two: build the Bluestein plan once and reuse it across every
    // row, instead of rebuilding the chirp table + kernel FFT per row.
    let plan = (n > 1 && !n.is_power_of_two()).then(|| BluesteinPlan::new(n, false));
    let mut scratch = BluesteinScratch::default();
    for batch in 0..batch_size {
        let start = batch * n;
        let slice = &elements[start..start + n];
        match &plan {
            Some(plan) => plan.apply_into(slice, &mut scratch, &mut transform_buf),
            None => fft_1d_into(slice, &mut transform_buf),
        }
        for &(re, im) in &transform_buf {
            out_elements.push(make_complex_literal(re, im, out_dtype));
        }
    }

    let tensor =
        TensorValue::new(out_dtype, shape, out_elements).map_err(EvalError::InvalidTensor)?;
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

    let mut out_elements = Vec::with_capacity(elements.len());
    let mut transform_buf = Vec::with_capacity(n);
    let plan = (n > 1 && !n.is_power_of_two()).then(|| BluesteinPlan::new(n, true));
    let mut scratch = BluesteinScratch::default();
    for batch in 0..batch_size {
        let start = batch * n;
        let slice = &elements[start..start + n];
        match &plan {
            Some(plan) => plan.apply_into(slice, &mut scratch, &mut transform_buf),
            None => ifft_1d_into(slice, &mut transform_buf),
        }
        for &(re, im) in &transform_buf {
            out_elements.push(make_complex_literal(re, im, out_dtype));
        }
    }

    let tensor =
        TensorValue::new(out_dtype, shape, out_elements).map_err(EvalError::InvalidTensor)?;
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

    let mut out_elements = Vec::with_capacity(output_count);
    let copy_len = fft_length.min(input_last);

    // Reuse buffers across batch iterations to avoid O(batch_size) allocations
    let mut padded = vec![(0.0, 0.0); fft_length];
    let mut transformed = Vec::with_capacity(fft_length);

    for batch in 0..batch_size {
        let start = batch * input_last;
        let batch_slice = &elements[start..start + input_last];

        // Zero-pad or truncate to fft_length (reuse buffer)
        padded[..copy_len].copy_from_slice(&batch_slice[..copy_len]);
        padded[copy_len..].fill((0.0, 0.0));

        fft_1d_into(&padded, &mut transformed);

        // Keep only the first fft_length/2 + 1 elements
        for &(re, im) in &transformed[..out_last] {
            out_elements.push(make_complex_literal(re, im, out_dtype));
        }
    }

    let tensor =
        TensorValue::new(out_dtype, out_shape, out_elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

// ── IRFFT ────────────────────────────────────────────────────────────

/// Compute the 1D complex-to-real inverse FFT along the last axis.
///
/// Input: rank-1+ complex tensor with last dim = n/2 + 1 (half-spectrum).
/// Output: real tensor with last dim = fft_length.
///
/// Reconstructs the full spectrum from Hermitian symmetry, computes IDFT,
/// and takes the real part.
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

    let mut out_elements = Vec::with_capacity(output_count);
    let copy_len = fft_length.min(input_last);

    // Reuse buffers across batch iterations to avoid O(batch_size) allocations
    let mut full = vec![(0.0, 0.0); fft_length];
    let mut transformed = Vec::with_capacity(fft_length);

    for batch in 0..batch_size {
        let start = batch * input_last;
        let half_spectrum = &elements[start..start + input_last];

        // Reconstruct full spectrum using Hermitian symmetry (reuse buffer)
        // X[k] = conj(X[n-k]) for k = n/2+1..n-1
        full[..copy_len].copy_from_slice(&half_spectrum[..copy_len]);
        full[copy_len..].fill((0.0, 0.0));

        // Fill conjugate-symmetric part
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

        ifft_1d_into(&full, &mut transformed);

        // Take real part only
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
        Value::Tensor(TensorValue::new(DType::Complex128, shape, elements).unwrap())
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
