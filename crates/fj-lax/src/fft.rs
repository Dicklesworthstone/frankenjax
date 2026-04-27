#![forbid(unsafe_code)]

//! FFT primitives: Fft, Ifft, Rfft, Irfft.
//!
//! Uses the naive O(n²) DFT for correctness over speed. The 1D transform is
//! applied along the last axis for each batch (all leading dimensions).

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

/// Build a real Literal from an f64 value.
/// All real dtypes use F64Bits storage internally; the tensor's DType tracks the logical type.
fn make_real_literal(val: f64, _dtype: DType) -> Literal {
    Literal::from_f64(val)
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

/// Compute 1D DFT of a complex signal of length n.
/// X[k] = Σ_{j=0}^{n-1} x[j] * e^{-2πi·j·k/n}
fn dft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mut output = vec![(0.0, 0.0); n];
    for (k, out) in output.iter_mut().enumerate() {
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;
        for (j, &(xr, xi)) in input.iter().enumerate() {
            let angle = -2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            // (xr + i·xi) * (cos_a + i·sin_a) = (xr·cos_a - xi·sin_a) + i·(xr·sin_a + xi·cos_a)
            re_sum += xr * cos_a - xi * sin_a;
            im_sum += xr * sin_a + xi * cos_a;
        }
        *out = (re_sum, im_sum);
    }
    output
}

/// Compute 1D inverse DFT of a complex signal of length n.
/// x[j] = (1/n) Σ_{k=0}^{n-1} X[k] * e^{2πi·j·k/n}
fn idft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mut output = vec![(0.0, 0.0); n];
    let inv_n = 1.0 / (n as f64);
    for (j, out) in output.iter_mut().enumerate() {
        let mut re_sum = 0.0;
        let mut im_sum = 0.0;
        for (k, &(xr, xi)) in input.iter().enumerate() {
            let angle = 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            re_sum += xr * cos_a - xi * sin_a;
            im_sum += xr * sin_a + xi * cos_a;
        }
        *out = (re_sum * inv_n, im_sum * inv_n);
    }
    output
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
    for batch in 0..batch_size {
        let start = batch * n;
        let slice = &elements[start..start + n];
        let transformed = dft_1d(slice);
        for &(re, im) in &transformed {
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
    for batch in 0..batch_size {
        let start = batch * n;
        let slice = &elements[start..start + n];
        let transformed = idft_1d(slice);
        for &(re, im) in &transformed {
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

    let mut out_elements = Vec::with_capacity(batch_size * out_last);
    for batch in 0..batch_size {
        let start = batch * input_last;
        let batch_slice = &elements[start..start + input_last];

        // Zero-pad or truncate to fft_length
        let mut padded = vec![(0.0, 0.0); fft_length];
        let copy_len = fft_length.min(input_last);
        padded[..copy_len].copy_from_slice(&batch_slice[..copy_len]);

        let transformed = dft_1d(&padded);

        // Keep only the first fft_length/2 + 1 elements
        for &(re, im) in &transformed[..out_last] {
            out_elements.push(make_complex_literal(re, im, out_dtype));
        }
    }

    // Build output shape: replace last dim with out_last
    let mut out_dims = shape.dims;
    replace_last_axis_len(primitive, &mut out_dims, out_last)?;
    let out_shape = Shape { dims: out_dims };

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

    let mut out_elements = Vec::with_capacity(batch_size * fft_length);
    for batch in 0..batch_size {
        let start = batch * input_last;
        let half_spectrum = &elements[start..start + input_last];

        // Reconstruct full spectrum using Hermitian symmetry:
        // X[k] = conj(X[n-k]) for k = n/2+1..n-1
        let mut full = vec![(0.0, 0.0); fft_length];
        let copy_len = fft_length.min(input_last);
        full[..copy_len].copy_from_slice(&half_spectrum[..copy_len]);

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

        let transformed = idft_1d(&full);

        // Take real part only
        for &(re, _) in &transformed {
            out_elements.push(make_real_literal(re, out_dtype));
        }
    }

    // Build output shape: replace last dim with fft_length
    let mut out_dims = shape.dims;
    replace_last_axis_len(primitive, &mut out_dims, fft_length)?;
    let out_shape = Shape { dims: out_dims };

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
