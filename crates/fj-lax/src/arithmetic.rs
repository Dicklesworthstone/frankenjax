#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::collections::BTreeMap;

use crate::EvalError;
use crate::tensor_contraction::matmul_2d;
use crate::type_promotion::{binary_literal_op, promote_dtype};

/// Binary elementwise operation dispatching on int/float paths.
/// Supports full NumPy broadcasting: scalar-scalar, tensor-tensor (same shape),
/// scalar-tensor, tensor-scalar, and multi-dim broadcasting.
#[inline]
pub(crate) fn eval_binary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    if inputs.iter().any(value_contains_complex) {
        return eval_binary_elementwise_complex(primitive, inputs);
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs, *rhs, primitive, &int_op, &float_op,
        )?)),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape == rhs.shape {
                if lhs.dtype == DType::F64
                    && rhs.dtype == DType::F64
                    && let Some(value) = eval_same_shape_f64_binop(primitive, lhs, rhs)?
                {
                    return Ok(value);
                }
                if primitive == Primitive::Add
                    && lhs.dtype == DType::I64
                    && rhs.dtype == DType::I64
                    && let Some(value) = eval_same_shape_i64_add(lhs, rhs, &int_op)?
                {
                    return Ok(value);
                }

                let mut elements = Vec::with_capacity(lhs.elements.len());
                for (left, right) in lhs
                    .elements
                    .iter()
                    .copied()
                    .zip(rhs.elements.iter().copied())
                {
                    elements.push(binary_literal_op(
                        left, right, primitive, &int_op, &float_op,
                    )?);
                }

                let dtype = promote_dtype(lhs.dtype, rhs.dtype);
                Ok(Value::Tensor(TensorValue::new(
                    dtype,
                    lhs.shape.clone(),
                    elements,
                )?))
            } else {
                // Attempt NumPy multi-dim broadcasting
                broadcast_binary_tensors(primitive, lhs, rhs, &int_op, &float_op)
            }
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            if let Some(value) = eval_f64_scalar_broadcast_binop(primitive, *lhs, rhs, true)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(rhs.elements.len());
            for right in rhs.elements.iter().copied() {
                elements.push(binary_literal_op(
                    *lhs, right, primitive, &int_op, &float_op,
                )?);
            }

            let lhs_dtype = literal_dtype(*lhs);
            let dtype = promote_dtype(lhs_dtype, rhs.dtype);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            if let Some(value) = eval_f64_scalar_broadcast_binop(primitive, *rhs, lhs, false)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(lhs.elements.len());
            for left in lhs.elements.iter().copied() {
                elements.push(binary_literal_op(
                    left, *rhs, primitive, &int_op, &float_op,
                )?);
            }

            let rhs_dtype = literal_dtype(*rhs);
            let dtype = promote_dtype(lhs.dtype, rhs_dtype);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Same-shape F64⊗F64 elementwise fast path for the binops whose per-lane
/// operation is a plain `f64 -> f64` function (`+`, `-`, `*`, `/`, `max`, `min`).
///
/// This is bit-for-bit identical to the generic `binary_literal_op` path: for
/// `DType::F64` operands that path computes `Literal::from_f64(float_op(a, b))`
/// with the same closures/fns used here (see `lib.rs`: Add `|a,b| a+b`, Sub
/// `|a,b| a-b`, Mul `|a,b| a*b`, Div `|a,b| a/b`, Max `jax_max_f64`, Min
/// `jax_min_f64`). It only skips the per-element enum/promotion dispatch, so
/// output bits, ordering, NaN/inf behavior and errors are unchanged. Returns
/// `Ok(None)` for any primitive or element that is not the F64 fast case,
/// letting the caller fall through to the generic path.
#[inline]
fn eval_same_shape_f64_binop(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<Option<Value>, EvalError> {
    match primitive {
        Primitive::Add => eval_same_shape_f64_map(lhs, rhs, |a, b| a + b),
        Primitive::Sub => eval_same_shape_f64_map(lhs, rhs, |a, b| a - b),
        Primitive::Mul => eval_same_shape_f64_map(lhs, rhs, |a, b| a * b),
        Primitive::Div => eval_same_shape_f64_map(lhs, rhs, |a, b| a / b),
        // Max/Min must reuse the exact NaN-propagating helpers the generic path
        // passes to `eval_binary_elementwise` (NOT `f64::max`/`min`, which drop
        // NaN), so the fast path stays bit-for-bit identical.
        Primitive::Max => eval_same_shape_f64_map(lhs, rhs, crate::jax_max_f64),
        Primitive::Min => eval_same_shape_f64_map(lhs, rhs, crate::jax_min_f64),
        _ => Ok(None),
    }
}

#[inline]
fn eval_same_shape_f64_map(
    lhs: &TensorValue,
    rhs: &TensorValue,
    op: impl Fn(f64, f64) -> f64,
) -> Result<Option<Value>, EvalError> {
    if let (Some(lhs_values), Some(rhs_values)) =
        (lhs.elements.as_f64_slice(), rhs.elements.as_f64_slice())
    {
        let values = lhs_values
            .iter()
            .zip(rhs_values)
            .map(|(&left, &right)| op(left, right))
            .collect::<Vec<_>>();
        return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
            lhs.shape.clone(),
            values,
        )?)));
    }

    let mut elements = Vec::with_capacity(lhs.elements.len());
    for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::F64Bits(left_bits), Literal::F64Bits(right_bits)) = (*left, *right) else {
            return Ok(None);
        };
        let out = op(f64::from_bits(left_bits), f64::from_bits(right_bits));
        elements.push(Literal::from_f64(out));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::F64,
        lhs.shape.clone(),
        elements,
    )?)))
}

/// Same-shape I64 tensor Add fast path.
///
/// This preserves the public `Primitive::Add` integer semantics by applying the
/// same `int_op` closure as the generic `binary_literal_op` loop. The public
/// dispatcher supplies `i64::wrapping_add`, so overflow behavior is unchanged.
/// Returning `Ok(None)` for non-I64 elements preserves malformed-tensor fallback.
#[inline]
fn eval_same_shape_i64_add(
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_op: &impl Fn(i64, i64) -> i64,
) -> Result<Option<Value>, EvalError> {
    let mut elements = Vec::with_capacity(lhs.elements.len());
    for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::I64(left), Literal::I64(right)) = (*left, *right) else {
            return Ok(None);
        };
        elements.push(Literal::I64(int_op(left, right)));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::I64,
        lhs.shape.clone(),
        elements,
    )?)))
}

/// F64 scalar/tensor broadcast fast path for the arithmetic binops whose
/// per-lane operation is plain IEEE-754 f64 arithmetic (`+`, `-`, `*`, `/`).
///
/// `scalar_on_left` distinguishes `Scalar ⊗ Tensor` (`op(scalar, elem)`) from
/// `Tensor ⊗ Scalar` (`op(elem, scalar)`), preserving operand order for the
/// non-commutative `Sub`/`Div` cases. Bit-for-bit identical to the generic
/// `binary_literal_op` path: for F64 operands that path computes
/// `Literal::from_f64(float_op(a, b))` with the same closures (lib.rs). Returns
/// `Ok(None)` for any non-F64 scalar/element or non-fast-path primitive so the
/// caller falls through to the generic per-element loop.
#[inline]
fn eval_f64_scalar_broadcast_binop(
    primitive: Primitive,
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
) -> Result<Option<Value>, EvalError> {
    let Literal::F64Bits(scalar_bits) = scalar else {
        return Ok(None);
    };
    if tensor.dtype != DType::F64 {
        return Ok(None);
    }
    let scalar = f64::from_bits(scalar_bits);
    match primitive {
        Primitive::Add => f64_scalar_broadcast_map(scalar, tensor, scalar_on_left, |a, b| a + b),
        Primitive::Sub => f64_scalar_broadcast_map(scalar, tensor, scalar_on_left, |a, b| a - b),
        Primitive::Mul => f64_scalar_broadcast_map(scalar, tensor, scalar_on_left, |a, b| a * b),
        Primitive::Div => f64_scalar_broadcast_map(scalar, tensor, scalar_on_left, |a, b| a / b),
        _ => Ok(None),
    }
}

#[inline]
fn f64_scalar_broadcast_map(
    scalar: f64,
    tensor: &TensorValue,
    scalar_on_left: bool,
    op: impl Fn(f64, f64) -> f64,
) -> Result<Option<Value>, EvalError> {
    let mut elements = Vec::with_capacity(tensor.elements.len());
    for &elem in &tensor.elements {
        let Literal::F64Bits(bits) = elem else {
            return Ok(None);
        };
        let value = f64::from_bits(bits);
        let out = if scalar_on_left {
            op(scalar, value)
        } else {
            op(value, scalar)
        };
        elements.push(Literal::from_f64(out));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::F64,
        tensor.shape.clone(),
        elements,
    )?)))
}

fn value_contains_complex(value: &Value) -> bool {
    match value {
        Value::Scalar(literal) => literal.is_complex(),
        Value::Tensor(tensor) => matches!(tensor.dtype, DType::Complex64 | DType::Complex128),
    }
}

fn literal_to_complex_parts(
    _primitive: Primitive,
    literal: Literal,
) -> Result<(f64, f64), EvalError> {
    match literal {
        Literal::I64(v) => Ok((v as f64, 0.0)),
        Literal::U32(v) => Ok((v as f64, 0.0)),
        Literal::U64(v) => Ok((v as f64, 0.0)),
        Literal::Bool(v) => Ok((f64::from(u8::from(v)), 0.0)),
        Literal::BF16Bits(bits) => {
            let lit = Literal::BF16Bits(bits);
            Ok((lit.as_f64().unwrap_or_default(), 0.0))
        }
        Literal::F16Bits(bits) => {
            let lit = Literal::F16Bits(bits);
            Ok((lit.as_f64().unwrap_or_default(), 0.0))
        }
        Literal::F32Bits(bits) => Ok((f64::from(f32::from_bits(bits)), 0.0)),
        Literal::F64Bits(bits) => Ok((f64::from_bits(bits), 0.0)),
        Literal::Complex64Bits(re, im) => {
            Ok((f32::from_bits(re) as f64, f32::from_bits(im) as f64))
        }
        Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
    }
}

fn complex_literal_from_f64_parts(out_dtype: DType, re: f64, im: f64) -> Literal {
    match out_dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        DType::Complex128 => Literal::from_complex128(re, im),
        _ => Literal::from_complex128(re, im),
    }
}

fn real_literal_from_f64(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f32(value as f32),
        DType::F16 => Literal::from_f16_f32(value as f32),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

fn complex_binary_output_dtype(lhs: DType, rhs: DType) -> DType {
    match promote_dtype(lhs, rhs) {
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        _ => DType::Complex128,
    }
}

fn apply_complex_binary(
    primitive: Primitive,
    lhs: (f64, f64),
    rhs: (f64, f64),
) -> Result<(f64, f64), EvalError> {
    let (ar, ai) = lhs;
    let (br, bi) = rhs;
    match primitive {
        Primitive::Add => Ok((ar + br, ai + bi)),
        Primitive::Sub => Ok((ar - br, ai - bi)),
        Primitive::Mul => Ok((ar * br - ai * bi, ar * bi + ai * br)),
        Primitive::Div => Ok(complex_div(lhs, rhs)),
        Primitive::Rem => {
            let quotient = apply_complex_binary(Primitive::Div, lhs, rhs)?;
            let rounded = (quotient.0.round(), quotient.1.round());
            Ok(complex_sub((ar, ai), complex_mul(rhs, rounded)))
        }
        Primitive::Max => Ok(if complex_lex_ge(lhs, rhs) { lhs } else { rhs }),
        Primitive::Min => Ok(if complex_lex_ge(lhs, rhs) { rhs } else { lhs }),
        Primitive::Pow => apply_complex_pow(lhs, rhs),
        Primitive::Atan2 => Ok(complex_atan2(lhs, rhs)),
        Primitive::XLogY => {
            if ar == 0.0 && ai == 0.0 {
                Ok((0.0, 0.0))
            } else {
                Ok(complex_mul(lhs, complex_log(rhs)))
            }
        }
        Primitive::LogAddExp => {
            let exp_lhs = complex_exp(lhs);
            let exp_rhs = complex_exp(rhs);
            let sum = complex_add(exp_lhs, exp_rhs);
            Ok(complex_log(sum))
        }
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: complex_binary_unsupported_detail(primitive),
        }),
    }
}

fn complex_sub(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

fn complex_add(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    let (ar, ai) = lhs;
    let (br, bi) = rhs;
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_div(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    let (ar, ai) = lhs;
    let (br, bi) = rhs;
    let denom = br * br + bi * bi;
    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

fn complex_exp((re, im): (f64, f64)) -> (f64, f64) {
    let exp_re = re.exp();
    let (sin_im, cos_im) = im.sin_cos();
    (exp_re * cos_im, exp_re * sin_im)
}

fn complex_expm1((re, im): (f64, f64)) -> (f64, f64) {
    let expm1_re = re.exp_m1();
    let exp_re = expm1_re + 1.0;
    let (sin_im, cos_im) = im.sin_cos();
    (expm1_re * cos_im + cos_im - 1.0, exp_re * sin_im)
}

fn complex_log((re, im): (f64, f64)) -> (f64, f64) {
    (re.hypot(im).ln(), im.atan2(re))
}

fn complex_reciprocal((re, im): (f64, f64)) -> (f64, f64) {
    let denom = re * re + im * im;
    (re / denom, -im / denom)
}

fn complex_sqrt((re, im): (f64, f64)) -> (f64, f64) {
    let magnitude = re.hypot(im);
    let out_re = ((magnitude + re) * 0.5).sqrt();
    let out_im = ((magnitude - re) * 0.5).sqrt().copysign(im);
    (out_re, out_im)
}

fn complex_cbrt(input: (f64, f64)) -> (f64, f64) {
    let logged = complex_log(input);
    complex_exp((logged.0 / 3.0, logged.1 / 3.0))
}

fn complex_asin(input: (f64, f64)) -> (f64, f64) {
    let input_squared = complex_mul(input, input);
    let sqrt_term = complex_sqrt(complex_sub((1.0, 0.0), input_squared));
    let i_times_input = (-input.1, input.0);
    let logged = complex_log(complex_add(i_times_input, sqrt_term));
    (logged.1, -logged.0)
}

fn complex_acos(input: (f64, f64)) -> (f64, f64) {
    let asin = complex_asin(input);
    (std::f64::consts::FRAC_PI_2 - asin.0, -asin.1)
}

fn complex_atan(input: (f64, f64)) -> (f64, f64) {
    let i_times_input = (-input.1, input.0);
    let log_left = complex_log(complex_sub((1.0, 0.0), i_times_input));
    let log_right = complex_log(complex_add((1.0, 0.0), i_times_input));
    let diff = complex_sub(log_left, log_right);
    (-0.5 * diff.1, 0.5 * diff.0)
}

fn complex_atan2(y: (f64, f64), x: (f64, f64)) -> (f64, f64) {
    let numerator = complex_add(x, (-y.1, y.0));
    let denominator = complex_sqrt(complex_add(complex_mul(x, x), complex_mul(y, y)));
    let quotient = complex_div(numerator, denominator);
    let logged = complex_log(quotient);
    (logged.1, -logged.0)
}

fn complex_logistic(input: (f64, f64)) -> (f64, f64) {
    let neg_input = (-input.0, -input.1);
    complex_reciprocal(complex_add((1.0, 0.0), complex_exp(neg_input)))
}

fn complex_sin((re, im): (f64, f64)) -> (f64, f64) {
    (re.sin() * im.cosh(), re.cos() * im.sinh())
}

fn complex_sinc(input: (f64, f64)) -> (f64, f64) {
    if input.0 == 0.0 && input.1 == 0.0 {
        return (1.0, 0.0);
    }
    let pi = std::f64::consts::PI;
    let pi_z = (input.0 * pi, input.1 * pi);
    let sin_pi_z = complex_sin(pi_z);
    complex_div(sin_pi_z, pi_z)
}

fn complex_erf(z: (f64, f64)) -> (f64, f64) {
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let mut result = z;
    let mut z_power = z;
    let z_squared = complex_mul(z, z);
    let neg_z_squared = (-z_squared.0, -z_squared.1);
    let mut n_factorial = 1.0_f64;
    for n in 1..50 {
        z_power = complex_mul(z_power, neg_z_squared);
        n_factorial *= n as f64;
        let denom = n_factorial * (2 * n + 1) as f64;
        let term = (z_power.0 / denom, z_power.1 / denom);
        result = complex_add(result, term);
        if term.0.abs() < 1e-15 && term.1.abs() < 1e-15 {
            break;
        }
    }
    (result.0 * two_over_sqrt_pi, result.1 * two_over_sqrt_pi)
}

fn complex_erfc(z: (f64, f64)) -> (f64, f64) {
    let erf_z = complex_erf(z);
    (1.0 - erf_z.0, -erf_z.1)
}

fn complex_erf_inv(w: (f64, f64)) -> (f64, f64) {
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let mut z = if w.1.abs() < 1e-10 {
        let real_approx = if w.0.abs() < 0.5 {
            let x = w.0;
            let a = [0.886226899, -1.645349621, 0.914624893, -0.140543331];
            let b = [1.0, -2.118377725, 1.442710462, -0.329097515, 0.012229801];
            let num = a[0] + x * (a[1] + x * (a[2] + x * a[3]));
            let den = b[0] + x * (b[1] + x * (b[2] + x * (b[3] + x * b[4])));
            x * num / den
        } else {
            let t = (-2.0 * (1.0 - w.0.abs()).ln()).sqrt();
            let c = [2.515517, 0.802853, 0.010328];
            let d = [1.0, 1.432788, 0.189269, 0.001308];
            let approx =
                t - (c[0] + t * (c[1] + t * c[2])) / (d[0] + t * (d[1] + t * (d[2] + t * d[3])));
            if w.0 < 0.0 { -approx } else { approx }
        };
        (real_approx, w.1 * std::f64::consts::FRAC_PI_2.sqrt())
    } else {
        (
            w.0 * std::f64::consts::FRAC_PI_2.sqrt(),
            w.1 * std::f64::consts::FRAC_PI_2.sqrt(),
        )
    };
    for _ in 0..30 {
        let erf_z = complex_erf(z);
        let residual = complex_sub(erf_z, w);
        if residual.0.abs() < 1e-14 && residual.1.abs() < 1e-14 {
            break;
        }
        let neg_z_sq = complex_mul(z, z);
        let deriv = (
            two_over_sqrt_pi * (-neg_z_sq.0).exp() * neg_z_sq.1.cos(),
            two_over_sqrt_pi * (-neg_z_sq.0).exp() * (-neg_z_sq.1.sin()),
        );
        let step = complex_div(residual, deriv);
        z = complex_sub(z, step);
    }
    z
}

fn complex_lgamma(z: (f64, f64)) -> (f64, f64) {
    const G: f64 = 7.0;
    #[allow(clippy::excessive_precision)]
    const LANCZOS_COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let (re, im) = z;
    if re < 0.5 {
        let sin_pz = complex_sin((std::f64::consts::PI * re, std::f64::consts::PI * im));
        let ln_pi = (std::f64::consts::PI.ln(), 0.0);
        let ln_sin = complex_log(sin_pz);
        let lgamma_1mz = complex_lgamma((1.0 - re, -im));
        return complex_sub(complex_sub(ln_pi, ln_sin), lgamma_1mz);
    }
    let z_shifted = (re - 1.0, im);
    let mut x = (LANCZOS_COEFFS[0], 0.0);
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        let denom = complex_add(z_shifted, (i as f64, 0.0));
        x = complex_add(x, complex_div((coeff, 0.0), denom));
    }
    let t = complex_add(z_shifted, (G + 0.5, 0.0));
    let half_ln_2pi = (0.5 * (2.0 * std::f64::consts::PI).ln(), 0.0);
    let exp_term = complex_add(z_shifted, (0.5, 0.0));
    complex_add(
        complex_add(half_ln_2pi, complex_mul(exp_term, complex_log(t))),
        complex_sub(complex_log(x), t),
    )
}

fn complex_digamma(z: (f64, f64)) -> (f64, f64) {
    let (mut re, im) = z;
    let mut result = (0.0, 0.0);
    while re < 6.0 {
        result = complex_sub(result, complex_reciprocal((re, im)));
        re += 1.0;
    }
    let z2 = (re, im);
    let z2_inv = complex_reciprocal(z2);
    let z2_inv_sq = complex_mul(z2_inv, z2_inv);
    let mut term = z2_inv_sq;
    let bernoulli = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        5.0 / 660.0,
    ];
    let ln_z = complex_log(z2);
    result = complex_add(result, ln_z);
    result = complex_sub(result, (0.5 * z2_inv.0, 0.5 * z2_inv.1));
    for &b in &bernoulli {
        result = complex_sub(result, (b * term.0, b * term.1));
        term = complex_mul(term, z2_inv_sq);
    }
    result
}

fn complex_bessel_i0e(z: (f64, f64)) -> (f64, f64) {
    let z_sq_4 = complex_mul(complex_mul(z, z), (0.25, 0.0));
    let mut sum = (1.0, 0.0);
    let mut term = (1.0, 0.0);
    for k in 1..40 {
        term = complex_div(complex_mul(term, z_sq_4), ((k * k) as f64, 0.0));
        sum = complex_add(sum, term);
        if term.0.abs() < 1e-15 && term.1.abs() < 1e-15 {
            break;
        }
    }
    let scale = (-z.0.abs()).exp();
    (sum.0 * scale, sum.1 * scale)
}

fn complex_bessel_i1e(z: (f64, f64)) -> (f64, f64) {
    let z_sq_4 = complex_mul(complex_mul(z, z), (0.25, 0.0));
    let mut sum = (1.0, 0.0);
    let mut term = (1.0, 0.0);
    for k in 1..40 {
        term = complex_div(complex_mul(term, z_sq_4), ((k * (k + 1)) as f64, 0.0));
        sum = complex_add(sum, term);
        if term.0.abs() < 1e-15 && term.1.abs() < 1e-15 {
            break;
        }
    }
    let i1 = complex_mul((0.5, 0.0), complex_mul(z, sum));
    let scale = (-z.0.abs()).exp();
    (i1.0 * scale, i1.1 * scale)
}

fn complex_unary_elementwise(primitive: Primitive, input: (f64, f64)) -> Option<(f64, f64)> {
    match primitive {
        Primitive::Sqrt => Some(complex_sqrt(input)),
        Primitive::Rsqrt => Some(complex_reciprocal(complex_sqrt(input))),
        Primitive::Cbrt => Some(complex_cbrt(input)),
        Primitive::Asin => Some(complex_asin(input)),
        Primitive::Acos => Some(complex_acos(input)),
        Primitive::Atan => Some(complex_atan(input)),
        Primitive::Logistic => Some(complex_logistic(input)),
        Primitive::Expm1 => Some(complex_expm1(input)),
        Primitive::Log1p => Some(complex_log((input.0 + 1.0, input.1))),
        Primitive::Reciprocal => Some(complex_reciprocal(input)),
        Primitive::Exp2 => {
            let ln2 = std::f64::consts::LN_2;
            Some(complex_exp((input.0 * ln2, input.1 * ln2)))
        }
        Primitive::Log2 => {
            let result = complex_log(input);
            let ln2 = std::f64::consts::LN_2;
            Some((result.0 / ln2, result.1 / ln2))
        }
        Primitive::Sinc => Some(complex_sinc(input)),
        Primitive::Erf => Some(complex_erf(input)),
        Primitive::Erfc => Some(complex_erfc(input)),
        Primitive::ErfInv => Some(complex_erf_inv(input)),
        Primitive::Lgamma => Some(complex_lgamma(input)),
        Primitive::Digamma => Some(complex_digamma(input)),
        Primitive::BesselI0e => Some(complex_bessel_i0e(input)),
        Primitive::BesselI1e => Some(complex_bessel_i1e(input)),
        _ => None,
    }
}

fn complex_binary_unsupported_detail(_primitive: Primitive) -> &'static str {
    "operation is not supported for complex operands"
}

fn complex_unary_unsupported_detail(primitive: Primitive) -> &'static str {
    match primitive {
        Primitive::Floor => "floor is not supported for complex dtypes",
        Primitive::Ceil => "ceil is not supported for complex dtypes",
        Primitive::Round => "round is not supported for complex dtypes",
        _ => "operation is not supported for complex operands",
    }
}

fn select_literal_as_dtype(
    primitive: Primitive,
    value: Literal,
    dtype: DType,
    real_detail: &'static str,
    integer_detail: &'static str,
    unsigned_detail: &'static str,
) -> Result<Literal, EvalError> {
    match dtype {
        DType::F64 | DType::F32 | DType::BF16 | DType::F16 => {
            let f_val = value.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: real_detail,
            })?;
            Ok(real_literal_from_f64(dtype, f_val))
        }
        DType::I64 | DType::I32 => {
            let i_val = value.as_i64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: integer_detail,
            })?;
            Ok(Literal::I64(i_val))
        }
        DType::U32 => {
            let u_val = value.as_u64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: unsigned_detail,
            })?;
            Ok(Literal::U32(u32::try_from(u_val).map_err(|_| {
                EvalError::TypeMismatch {
                    primitive,
                    detail: "u32 overflow in select",
                }
            })?))
        }
        DType::U64 => {
            let u_val = value.as_u64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: unsigned_detail,
            })?;
            Ok(Literal::U64(u_val))
        }
        DType::Bool => Ok(value),
        DType::Complex64 | DType::Complex128 => {
            let (re, im) = literal_to_complex_parts(primitive, value)?;
            Ok(complex_literal_from_f64_parts(dtype, re, im))
        }
    }
}

fn select_bool_condition(primitive: Primitive, value: Literal) -> Result<bool, EvalError> {
    match value {
        Literal::Bool(flag) => Ok(flag),
        Literal::I64(v) => Ok(v != 0),
        Literal::U32(v) => Ok(v != 0),
        Literal::U64(v) => Ok(v != 0),
        Literal::BF16Bits(bits) => Ok(Literal::BF16Bits(bits).as_f64().is_some_and(|v| v != 0.0)),
        Literal::F16Bits(bits) => Ok(Literal::F16Bits(bits).as_f64().is_some_and(|v| v != 0.0)),
        Literal::F32Bits(bits) => Ok(f32::from_bits(bits) != 0.0),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits) != 0.0),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "select condition must be boolean or numeric",
        }),
    }
}

fn complex_lex_ge(lhs: (f64, f64), rhs: (f64, f64)) -> bool {
    lhs.0 > rhs.0 || (lhs.0 == rhs.0 && lhs.1 >= rhs.1)
}

fn complex_integer_pow(mut base: (f64, f64), mut exponent: u64) -> (f64, f64) {
    let mut acc = (1.0, 0.0);
    while exponent > 0 {
        if exponent & 1 == 1 {
            acc = complex_mul(acc, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = complex_mul(base, base);
        }
    }
    acc
}

fn apply_complex_pow(lhs: (f64, f64), rhs: (f64, f64)) -> Result<(f64, f64), EvalError> {
    let (base_re, base_im) = lhs;
    let (exp_re, exp_im) = rhs;

    if base_re == 0.0 && base_im == 0.0 {
        return if exp_re == 0.0 && exp_im == 0.0 {
            Ok((1.0, 0.0))
        } else if exp_im == 0.0 && exp_re > 0.0 {
            Ok((0.0, 0.0))
        } else {
            Ok((f64::NAN, f64::NAN))
        };
    }

    if exp_im == 0.0 && exp_re.is_finite() && exp_re.fract() == 0.0 {
        if exp_re >= 0.0 {
            return Ok(complex_integer_pow(lhs, exp_re as u64));
        }
        let positive = complex_integer_pow(lhs, (-exp_re) as u64);
        let denom = positive.0 * positive.0 + positive.1 * positive.1;
        return Ok((positive.0 / denom, -positive.1 / denom));
    }

    let radius = base_re.hypot(base_im);
    let angle = base_im.atan2(base_re);
    let log_base = (radius.ln(), angle);
    Ok(complex_exp(complex_mul((exp_re, exp_im), log_base)))
}

fn complex_binary_literal_op(
    primitive: Primitive,
    lhs: Literal,
    rhs: Literal,
    out_dtype: DType,
) -> Result<Literal, EvalError> {
    let lhs = literal_to_complex_parts(primitive, lhs)?;
    let rhs = literal_to_complex_parts(primitive, rhs)?;
    let (re, im) = apply_complex_binary(primitive, lhs, rhs)?;
    Ok(complex_literal_from_f64_parts(out_dtype, re, im))
}

fn eval_binary_elementwise_complex(
    primitive: Primitive,
    inputs: &[Value],
) -> Result<Value, EvalError> {
    match primitive {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Atan2
        | Primitive::XLogY
        | Primitive::LogAddExp => {}
        _ => {
            return Err(EvalError::TypeMismatch {
                primitive,
                detail: complex_binary_unsupported_detail(primitive),
            });
        }
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            let out_dtype = complex_binary_output_dtype(literal_dtype(*lhs), literal_dtype(*rhs));
            Ok(Value::Scalar(complex_binary_literal_op(
                primitive, *lhs, *rhs, out_dtype,
            )?))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            let out_dtype = complex_binary_output_dtype(lhs.dtype, rhs.dtype);
            if lhs.shape == rhs.shape {
                if primitive == Primitive::Mul
                    && let Some(value) = eval_same_shape_complex128_mul(lhs, rhs)?
                {
                    return Ok(value);
                }

                let mut elements = Vec::with_capacity(lhs.elements.len());
                for (left, right) in lhs
                    .elements
                    .iter()
                    .copied()
                    .zip(rhs.elements.iter().copied())
                {
                    elements.push(complex_binary_literal_op(
                        primitive, left, right, out_dtype,
                    )?);
                }

                Ok(Value::Tensor(TensorValue::new(
                    out_dtype,
                    lhs.shape.clone(),
                    elements,
                )?))
            } else {
                let out_shape =
                    broadcast_shape(&lhs.shape, &rhs.shape).ok_or(EvalError::ShapeMismatch {
                        primitive,
                        left: lhs.shape.clone(),
                        right: rhs.shape.clone(),
                    })?;

                let out_count = out_shape.element_count().unwrap_or(0) as usize;
                let out_strides = compute_strides(&out_shape.dims);
                let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
                let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

                let mut multi = Vec::with_capacity(out_strides.len());
                let mut elements = Vec::with_capacity(out_count);
                for flat_idx in 0..out_count {
                    flat_to_multi_into(flat_idx, &out_strides, &mut multi);
                    let lhs_idx = broadcast_flat_index(&multi, &lhs_strides);
                    let rhs_idx = broadcast_flat_index(&multi, &rhs_strides);
                    elements.push(complex_binary_literal_op(
                        primitive,
                        lhs.elements[lhs_idx],
                        rhs.elements[rhs_idx],
                        out_dtype,
                    )?);
                }

                Ok(Value::Tensor(TensorValue::new(
                    out_dtype, out_shape, elements,
                )?))
            }
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let out_dtype = complex_binary_output_dtype(literal_dtype(*lhs), rhs.dtype);
            let mut elements = Vec::with_capacity(rhs.elements.len());
            for right in rhs.elements.iter().copied() {
                elements.push(complex_binary_literal_op(
                    primitive, *lhs, right, out_dtype,
                )?);
            }

            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let out_dtype = complex_binary_output_dtype(lhs.dtype, literal_dtype(*rhs));
            let mut elements = Vec::with_capacity(lhs.elements.len());
            for left in lhs.elements.iter().copied() {
                elements.push(complex_binary_literal_op(primitive, left, *rhs, out_dtype)?);
            }

            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Same-shape Complex128 tensor multiply fast path.
///
/// The generic complex path converts each literal into `(f64, f64)`, dispatches
/// `Primitive::Mul`, then rebuilds `Literal::Complex128Bits`. This path applies
/// the identical formula (`ar*br - ai*bi`, `ar*bi + ai*br`) in the same element
/// order, skipping the per-element enum and primitive dispatch.
#[inline]
fn eval_same_shape_complex128_mul(
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<Option<Value>, EvalError> {
    if lhs.dtype != DType::Complex128 || rhs.dtype != DType::Complex128 {
        return Ok(None);
    }

    let mut elements = Vec::with_capacity(lhs.elements.len());
    for (&left, &right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::Complex128Bits(ar_bits, ai_bits), Literal::Complex128Bits(br_bits, bi_bits)) =
            (left, right)
        else {
            return Ok(None);
        };
        let ar = f64::from_bits(ar_bits);
        let ai = f64::from_bits(ai_bits);
        let br = f64::from_bits(br_bits);
        let bi = f64::from_bits(bi_bits);
        elements.push(Literal::from_complex128(
            ar * br - ai * bi,
            ar * bi + ai * br,
        ));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::Complex128,
        lhs.shape.clone(),
        elements,
    )?)))
}

/// Full NumPy multi-dim broadcasting for two tensors.
fn broadcast_binary_tensors(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_op: &impl Fn(i64, i64) -> i64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    let out_shape = broadcast_shape(&lhs.shape, &rhs.shape).ok_or(EvalError::ShapeMismatch {
        primitive,
        left: lhs.shape.clone(),
        right: rhs.shape.clone(),
    })?;

    let out_count = out_shape.element_count().unwrap_or(0) as usize;
    let out_strides = compute_strides(&out_shape.dims);

    // Compute broadcast-aware strides for lhs and rhs
    let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
    let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

    // F64⊗F64 fast path: the generic binary_literal_op reduces to
    // `Literal::from_f64(float_op(a, b))` for F64 operands (out dtype F64 hits
    // the float branch), so apply float_op directly to the broadcast-gathered
    // values and skip the per-element promote_dtype / literal_to_numeric_f64 /
    // literal_from_numeric_f64 dispatch. Same broadcast index math => identical
    // element order and bits. Bails to the generic path on any non-F64Bits
    // element. Works for every binary primitive since float_op carries the op.
    if lhs.dtype == DType::F64
        && rhs.dtype == DType::F64
        && let Some(value) = broadcast_binary_f64(
            lhs,
            rhs,
            &out_shape,
            out_count,
            &out_strides,
            &lhs_strides,
            &rhs_strides,
            float_op,
        )?
    {
        return Ok(value);
    }

    let mut multi = Vec::with_capacity(out_strides.len());
    let mut elements = Vec::with_capacity(out_count);
    for flat_idx in 0..out_count {
        flat_to_multi_into(flat_idx, &out_strides, &mut multi);
        let lhs_idx = broadcast_flat_index(&multi, &lhs_strides);
        let rhs_idx = broadcast_flat_index(&multi, &rhs_strides);

        let l = lhs.elements[lhs_idx];
        let r = rhs.elements[rhs_idx];
        elements.push(binary_literal_op(l, r, primitive, int_op, float_op)?);
    }

    let dtype = promote_dtype(lhs.dtype, rhs.dtype);
    Ok(Value::Tensor(TensorValue::new(dtype, out_shape, elements)?))
}

/// F64 broadcast binary fast path. Produces output elements in the same
/// row-major flat order as the generic broadcast loop using identical index
/// math, applying `float_op` directly to the gathered F64 values. Bit-for-bit
/// identical to the generic path for F64 operands (where binary_literal_op is
/// `from_f64(float_op(from_bits(l), from_bits(r)))`). Returns `Ok(None)` if any
/// gathered element is not `F64Bits`, so the caller falls through to generic.
#[inline]
#[allow(clippy::too_many_arguments)]
fn broadcast_binary_f64(
    lhs: &TensorValue,
    rhs: &TensorValue,
    out_shape: &Shape,
    out_count: usize,
    out_strides: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    float_op: &impl Fn(f64, f64) -> f64,
) -> Result<Option<Value>, EvalError> {
    let mut multi = Vec::with_capacity(out_strides.len());
    let mut elements = Vec::with_capacity(out_count);
    for flat_idx in 0..out_count {
        flat_to_multi_into(flat_idx, out_strides, &mut multi);
        let lhs_idx = broadcast_flat_index(&multi, lhs_strides);
        let rhs_idx = broadcast_flat_index(&multi, rhs_strides);

        let (Literal::F64Bits(lhs_bits), Literal::F64Bits(rhs_bits)) =
            (lhs.elements[lhs_idx], rhs.elements[rhs_idx])
        else {
            return Ok(None);
        };
        let out = float_op(f64::from_bits(lhs_bits), f64::from_bits(rhs_bits));
        elements.push(Literal::from_f64(out));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::F64,
        out_shape.clone(),
        elements,
    )?)))
}

fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    let max_rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for offset in 0..max_rank {
        let lhs_dim = if offset < lhs.rank() {
            lhs.dims[lhs.rank() - 1 - offset]
        } else {
            1
        };
        let rhs_dim = if offset < rhs.rank() {
            rhs.dims[rhs.rank() - 1 - offset]
        } else {
            1
        };

        let out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return None;
        };
        dims.push(out_dim);
    }

    dims.reverse();
    Some(Shape { dims })
}

fn compute_strides(dims: &[u32]) -> Vec<usize> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn flat_to_multi_into(flat: usize, strides: &[usize], out: &mut Vec<usize>) {
    out.clear();
    let mut remainder = flat;
    for &stride in strides {
        out.push(remainder / stride);
        remainder %= stride;
    }
}

/// Compute strides for a tensor being broadcast to out_shape.
/// Dimensions of size 1 get stride 0 (broadcast), left-padded with 0s.
fn broadcast_strides(shape: &Shape, out_shape: &Shape) -> Vec<usize> {
    let rank = shape.rank();
    let out_rank = out_shape.rank();

    // Compute real strides for the input tensor
    let real_strides = compute_strides(&shape.dims);

    let mut result = vec![0_usize; out_rank];
    for (i, &stride) in real_strides.iter().enumerate().take(rank) {
        let out_axis = out_rank - rank + i;
        if shape.dims[i] == 1 {
            result[out_axis] = 0; // broadcast
        } else {
            result[out_axis] = stride;
        }
    }
    result
}

fn broadcast_flat_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

fn literal_dtype(literal: Literal) -> DType {
    match literal {
        Literal::I64(_) => DType::I64,
        Literal::U32(_) => DType::U32,
        Literal::U64(_) => DType::U64,
        Literal::BF16Bits(_) => DType::BF16,
        Literal::F16Bits(_) => DType::F16,
        Literal::F32Bits(_) => DType::F32,
        Literal::F64Bits(_) => DType::F64,
        Literal::Bool(_) => DType::Bool,
        Literal::Complex64Bits(..) => DType::Complex64,
        Literal::Complex128Bits(..) => DType::Complex128,
    }
}

fn operand_dtype(value: &Value) -> DType {
    match value {
        Value::Scalar(literal) => literal_dtype(*literal),
        Value::Tensor(tensor) => tensor.dtype,
    }
}

fn literal_to_real_f64(primitive: Primitive, literal: Literal) -> Result<f64, EvalError> {
    match literal {
        Literal::I64(v) => Ok(v as f64),
        Literal::U32(v) => Ok(v as f64),
        Literal::U64(v) => Ok(v as f64),
        Literal::BF16Bits(bits) => {
            Literal::BF16Bits(bits)
                .as_f64()
                .ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected real numeric value, got bf16",
                })
        }
        Literal::F16Bits(bits) => Literal::F16Bits(bits)
            .as_f64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected real numeric value, got f16",
            }),
        Literal::F32Bits(bits) => Ok(f64::from(f32::from_bits(bits))),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits)),
        Literal::Bool(_) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "expected real numeric value, got bool",
        }),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "expected real numeric value, got complex",
        }),
    }
}

fn complex_output_dtype(lhs: DType, rhs: DType) -> DType {
    match (lhs, rhs) {
        (DType::F32, DType::F32) => DType::Complex64,
        _ => DType::Complex128,
    }
}

fn complex_literal_from_parts(
    primitive: Primitive,
    real: Literal,
    imag: Literal,
    out_dtype: DType,
) -> Result<Literal, EvalError> {
    let re = literal_to_real_f64(primitive, real)?;
    let im = literal_to_real_f64(primitive, imag)?;
    Ok(match out_dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        DType::Complex128 => Literal::from_complex128(re, im),
        _ => Literal::from_complex128(re, im),
    })
}

fn complex_literal_from_parts_fast(
    primitive: Primitive,
    real: Literal,
    imag: Literal,
    out_dtype: DType,
) -> Result<Literal, EvalError> {
    match (out_dtype, real, imag) {
        (DType::Complex64, Literal::F32Bits(re), Literal::F32Bits(im)) => {
            Ok(Literal::Complex64Bits(re, im))
        }
        (DType::Complex128, Literal::F64Bits(re), Literal::F64Bits(im)) => {
            Ok(Literal::Complex128Bits(re, im))
        }
        _ => complex_literal_from_parts(primitive, real, imag, out_dtype),
    }
}

pub(crate) fn eval_complex(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(real), Value::Scalar(imag)) => {
            let dtype = complex_output_dtype(literal_dtype(*real), literal_dtype(*imag));
            Ok(Value::Scalar(complex_literal_from_parts_fast(
                primitive, *real, *imag, dtype,
            )?))
        }
        (Value::Tensor(real), Value::Tensor(imag)) => {
            let out_dtype = complex_output_dtype(real.dtype, imag.dtype);
            if real.shape == imag.shape {
                let mut elements = Vec::with_capacity(real.elements.len());
                for (re, im) in real
                    .elements
                    .iter()
                    .copied()
                    .zip(imag.elements.iter().copied())
                {
                    elements.push(complex_literal_from_parts_fast(
                        primitive, re, im, out_dtype,
                    )?);
                }
                Ok(Value::Tensor(TensorValue::new(
                    out_dtype,
                    real.shape.clone(),
                    elements,
                )?))
            } else {
                let out_shape =
                    broadcast_shape(&real.shape, &imag.shape).ok_or(EvalError::ShapeMismatch {
                        primitive,
                        left: real.shape.clone(),
                        right: imag.shape.clone(),
                    })?;
                let out_count = out_shape.element_count().unwrap_or(0) as usize;
                let out_strides = compute_strides(&out_shape.dims);
                let real_strides = broadcast_strides(&real.shape, &out_shape);
                let imag_strides = broadcast_strides(&imag.shape, &out_shape);

                let mut multi = Vec::with_capacity(out_strides.len());
                let mut elements = Vec::with_capacity(out_count);
                for flat_idx in 0..out_count {
                    flat_to_multi_into(flat_idx, &out_strides, &mut multi);
                    let real_idx = broadcast_flat_index(&multi, &real_strides);
                    let imag_idx = broadcast_flat_index(&multi, &imag_strides);
                    elements.push(complex_literal_from_parts_fast(
                        primitive,
                        real.elements[real_idx],
                        imag.elements[imag_idx],
                        out_dtype,
                    )?);
                }
                Ok(Value::Tensor(TensorValue::new(
                    out_dtype, out_shape, elements,
                )?))
            }
        }
        (Value::Scalar(real), Value::Tensor(imag)) => {
            let out_dtype = complex_output_dtype(literal_dtype(*real), imag.dtype);
            let mut elements = Vec::with_capacity(imag.elements.len());
            for im in imag.elements.iter().copied() {
                elements.push(complex_literal_from_parts_fast(
                    primitive, *real, im, out_dtype,
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                imag.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(real), Value::Scalar(imag)) => {
            let out_dtype = complex_output_dtype(real.dtype, literal_dtype(*imag));
            let mut elements = Vec::with_capacity(real.elements.len());
            for re in real.elements.iter().copied() {
                elements.push(complex_literal_from_parts_fast(
                    primitive, re, *imag, out_dtype,
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                real.shape.clone(),
                elements,
            )?))
        }
    }
}

pub(crate) fn eval_conj(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let conj_literal = |lit: Literal| -> Result<Literal, EvalError> {
        match lit {
            Literal::Complex64Bits(re_bits, im_bits) => {
                Ok(Literal::Complex64Bits(re_bits, im_bits ^ 0x8000_0000))
            }
            Literal::Complex128Bits(re_bits, im_bits) => Ok(Literal::Complex128Bits(
                re_bits,
                im_bits ^ 0x8000_0000_0000_0000,
            )),
            _ => Err(EvalError::TypeMismatch {
                primitive,
                detail: "conj expects complex-valued input",
            }),
        }
    };

    match &inputs[0] {
        Value::Scalar(lit) => Ok(Value::Scalar(conj_literal(*lit)?)),
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for lit in tensor.elements.iter().copied() {
                elements.push(conj_literal(lit)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

fn real_dtype_from_complex(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => DType::F64,
    }
}

pub(crate) fn eval_real(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let real_part = |lit: Literal| -> Result<Literal, EvalError> {
        match lit {
            Literal::Complex64Bits(re_bits, _) => Ok(Literal::F32Bits(re_bits)),
            Literal::Complex128Bits(re_bits, _) => Ok(Literal::F64Bits(re_bits)),
            _ => Err(EvalError::TypeMismatch {
                primitive,
                detail: "real expects complex-valued input",
            }),
        }
    };

    match &inputs[0] {
        Value::Scalar(lit) => Ok(Value::Scalar(real_part(*lit)?)),
        Value::Tensor(tensor) => {
            let out_dtype = real_dtype_from_complex(tensor.dtype);
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for lit in tensor.elements.iter().copied() {
                elements.push(real_part(lit)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

pub(crate) fn eval_imag(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let imag_part = |lit: Literal| -> Result<Literal, EvalError> {
        match lit {
            Literal::Complex64Bits(_, im_bits) => Ok(Literal::F32Bits(im_bits)),
            Literal::Complex128Bits(_, im_bits) => Ok(Literal::F64Bits(im_bits)),
            _ => Err(EvalError::TypeMismatch {
                primitive,
                detail: "imag expects complex-valued input",
            }),
        }
    };

    match &inputs[0] {
        Value::Scalar(lit) => Ok(Value::Scalar(imag_part(*lit)?)),
        Value::Tensor(tensor) => {
            let out_dtype = real_dtype_from_complex(tensor.dtype);
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for lit in tensor.elements.iter().copied() {
                elements.push(imag_part(lit)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

fn eval_unary_complex_map(
    primitive: Primitive,
    inputs: &[Value],
    op: impl Fn(f64, f64) -> (f64, f64),
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let map_literal = |lit: Literal| -> Result<Literal, EvalError> {
        let (re, im) = literal_to_complex_parts(primitive, lit)?;
        let (re, im) = op(re, im);
        let out_dtype = match lit {
            Literal::Complex64Bits(..) => DType::Complex64,
            Literal::Complex128Bits(..) => DType::Complex128,
            _ => DType::Complex128,
        };
        Ok(complex_literal_from_f64_parts(out_dtype, re, im))
    };

    match &inputs[0] {
        Value::Scalar(lit) => Ok(Value::Scalar(map_literal(*lit)?)),
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for lit in tensor.elements.iter().copied() {
                elements.push(map_literal(lit)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

fn eval_unary_complex_abs(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let abs_literal = |lit: Literal| -> Result<Literal, EvalError> {
        match lit {
            Literal::Complex64Bits(re_bits, im_bits) => Ok(Literal::from_f32(
                f32::from_bits(re_bits).hypot(f32::from_bits(im_bits)),
            )),
            Literal::Complex128Bits(re_bits, im_bits) => Ok(Literal::from_f64(
                f64::from_bits(re_bits).hypot(f64::from_bits(im_bits)),
            )),
            _ => Err(EvalError::TypeMismatch {
                primitive,
                detail: "abs expects complex-valued input",
            }),
        }
    };

    match &inputs[0] {
        Value::Scalar(lit) => Ok(Value::Scalar(abs_literal(*lit)?)),
        Value::Tensor(tensor) => {
            let out_dtype = real_dtype_from_complex(tensor.dtype);
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for lit in tensor.elements.iter().copied() {
                elements.push(abs_literal(lit)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

pub(crate) fn eval_neg(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |re, im| (-re, -im))
    } else {
        eval_unary_int_or_float(
            primitive,
            inputs,
            i64::wrapping_neg,
            u32::wrapping_neg,
            u64::wrapping_neg,
            |x| -x,
        )
    }
}

pub(crate) fn eval_abs(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_abs(primitive, inputs)
    } else {
        eval_unary_int_or_float(
            primitive,
            inputs,
            i64::wrapping_abs,
            std::convert::identity,
            std::convert::identity,
            f64::abs,
        )
    }
}

pub(crate) fn eval_exp(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| complex_exp((a, b)))
    } else {
        eval_unary_elementwise(primitive, inputs, f64::exp)
    }
}

pub(crate) fn eval_log(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| complex_log((a, b)))
    } else {
        eval_unary_elementwise(primitive, inputs, f64::ln)
    }
}

pub(crate) fn eval_sin(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.sin() * b.cosh(), a.cos() * b.sinh())
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::sin)
    }
}

pub(crate) fn eval_cos(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.cos() * b.cosh(), -a.sin() * b.sinh())
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::cos)
    }
}

pub(crate) fn eval_tan(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let denom = (2.0 * a).cos() + (2.0 * b).cosh();
            ((2.0 * a).sin() / denom, (2.0 * b).sinh() / denom)
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::tan)
    }
}

pub(crate) fn eval_sinh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.sinh() * b.cos(), a.cosh() * b.sin())
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::sinh)
    }
}

pub(crate) fn eval_cosh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.cosh() * b.cos(), a.sinh() * b.sin())
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::cosh)
    }
}

pub(crate) fn eval_tanh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let denom = (2.0 * a).cosh() + (2.0 * b).cos();
            ((2.0 * a).sinh() / denom, (2.0 * b).sin() / denom)
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::tanh)
    }
}

pub(crate) fn eval_asinh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let z_sq = (a * a - b * b + 1.0, 2.0 * a * b);
            let sqrt = complex_sqrt(z_sq);
            let w = (a + sqrt.0, b + sqrt.1);
            complex_log(w)
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::asinh)
    }
}

pub(crate) fn eval_acosh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let z_sq = (a * a - b * b - 1.0, 2.0 * a * b);
            let sqrt = complex_sqrt(z_sq);
            let w = (a + sqrt.0, b + sqrt.1);
            complex_log(w)
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::acosh)
    }
}

pub(crate) fn eval_atanh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let numer = (1.0 + a, b);
            let denom = (1.0 - a, -b);
            let div = complex_div(numer, denom);
            let log = complex_log(div);
            (0.5 * log.0, 0.5 * log.1)
        })
    } else {
        eval_unary_elementwise(primitive, inputs, f64::atanh)
    }
}

/// Unary elementwise operation that converts to f64 first (exp, log, sqrt, etc.).
#[inline]
pub(crate) fn eval_unary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    op: impl Fn(f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            if literal.is_complex() {
                return {
                    let input = literal_to_complex_parts(primitive, *literal)?;
                    if let Some((out_re, out_im)) = complex_unary_elementwise(primitive, input) {
                        Ok(Value::Scalar(complex_literal_from_f64_parts(
                            literal_dtype(*literal),
                            out_re,
                            out_im,
                        )))
                    } else {
                        Err(EvalError::TypeMismatch {
                            primitive,
                            detail: complex_unary_unsupported_detail(primitive),
                        })
                    }
                };
            }
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric scalar",
            })?;
            // Preserve the input literal's dtype so unary transcendentals
            // (Exp, Log, Sin, Cos, …) on F32/BF16/F16 scalars don't get
            // silently widened to F64. Integer/Bool inputs fall back to
            // F64 since these primitives are not natively integer-typed.
            let result = op(value);
            let out_lit = match literal {
                Literal::F32Bits(_) => Literal::from_f32(result as f32),
                Literal::BF16Bits(_) => Literal::from_bf16_f32(result as f32),
                Literal::F16Bits(_) => Literal::from_f16_f32(result as f32),
                Literal::F64Bits(_) => Literal::from_f64(result),
                _ => Literal::from_f64(result),
            };
            Ok(Value::Scalar(out_lit))
        }
        Value::Tensor(tensor) => {
            if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) {
                return {
                    let mut elements = Vec::with_capacity(tensor.elements.len());
                    for literal in tensor.elements.iter().copied() {
                        let input = literal_to_complex_parts(primitive, literal)?;
                        let (out_re, out_im) = complex_unary_elementwise(primitive, input).ok_or(
                            EvalError::TypeMismatch {
                                primitive,
                                detail: complex_unary_unsupported_detail(primitive),
                            },
                        )?;
                        elements.push(complex_literal_from_f64_parts(tensor.dtype, out_re, out_im));
                    }

                    Ok(Value::Tensor(TensorValue::new(
                        tensor.dtype,
                        tensor.shape.clone(),
                        elements,
                    )?))
                };
            }
            let out_dtype = match tensor.dtype {
                DType::BF16 | DType::F16 | DType::F32 | DType::F64 => tensor.dtype,
                _ => DType::F64,
            };

            if let Some(result) = eval_unary_f64_tensor_fast_path(tensor, &op) {
                return result;
            }

            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in tensor.elements.iter().copied() {
                let mapped = literal.as_f64().map(&op).ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric tensor elements",
                })?;
                let out = match out_dtype {
                    DType::BF16 => Literal::from_bf16_f32(mapped as f32),
                    DType::F16 => Literal::from_f16_f32(mapped as f32),
                    DType::F32 => Literal::from_f32(mapped as f32),
                    _ => Literal::from_f64(mapped),
                };
                elements.push(out);
            }

            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

fn eval_unary_f64_tensor_fast_path(
    tensor: &TensorValue,
    op: &impl Fn(f64) -> f64,
) -> Option<Result<Value, EvalError>> {
    if tensor.dtype != DType::F64 {
        return None;
    }

    let mut elements = Vec::with_capacity(tensor.elements.len());
    for literal in tensor.elements.iter().copied() {
        let Literal::F64Bits(bits) = literal else {
            return None;
        };
        elements.push(Literal::F64Bits(op(f64::from_bits(bits)).to_bits()));
    }

    Some(
        TensorValue::new(DType::F64, tensor.shape.clone(), elements)
            .map(Value::Tensor)
            .map_err(EvalError::from),
    )
}

/// Round with JAX's `rounding_method` parameter.
pub(crate) fn eval_round(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let op = match params.get("rounding_method").map(|raw| raw.trim()) {
        None | Some("") | Some("0") | Some("AWAY_FROM_ZERO") | Some("away_from_zero") => f64::round,
        Some("1") | Some("TO_NEAREST_EVEN") | Some("to_nearest_even") | Some("nearest_even") => {
            f64::round_ties_even
        }
        Some(raw) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("unsupported rounding_method '{raw}'"),
            });
        }
    };

    eval_unary_elementwise(primitive, inputs, op)
}

/// Unary elementwise that preserves integer types (for neg, abs).
#[inline]
pub(crate) fn eval_unary_int_or_float(
    primitive: Primitive,
    inputs: &[Value],
    int_op: impl Fn(i64) -> i64,
    u32_op: impl Fn(u32) -> u32,
    u64_op: impl Fn(u64) -> u64,
    float_op: impl Fn(f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => match *literal {
            Literal::I64(v) => Ok(Value::scalar_i64(int_op(v))),
            Literal::U32(v) => Ok(Value::scalar_u32(u32_op(v))),
            Literal::U64(v) => Ok(Value::scalar_u64(u64_op(v))),
            Literal::BF16Bits(bits) => {
                let val = Literal::BF16Bits(bits)
                    .as_f64()
                    .ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar, got bf16",
                    })?;
                // Preserve BF16 dtype — F32/F64 arms already preserve.
                Ok(Value::Scalar(Literal::from_bf16_f32(float_op(val) as f32)))
            }
            Literal::F16Bits(bits) => {
                let val = Literal::F16Bits(bits)
                    .as_f64()
                    .ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar, got f16",
                    })?;
                // Preserve F16 dtype — F32/F64 arms already preserve.
                Ok(Value::Scalar(Literal::from_f16_f32(float_op(val) as f32)))
            }
            Literal::F32Bits(bits) => Ok(Value::scalar_f32(
                float_op(f64::from(f32::from_bits(bits))) as f32,
            )),
            Literal::F64Bits(bits) => Ok(Value::scalar_f64(float_op(f64::from_bits(bits)))),
            Literal::Bool(_) => Err(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric scalar, got bool",
            }),
            Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => match primitive {
                Primitive::Sign => {
                    let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                    let magnitude = re.hypot(im);
                    let (out_re, out_im) = if magnitude == 0.0 {
                        (0.0, 0.0)
                    } else {
                        (re / magnitude, im / magnitude)
                    };
                    Ok(Value::Scalar(complex_literal_from_f64_parts(
                        literal_dtype(*literal),
                        out_re,
                        out_im,
                    )))
                }
                Primitive::Square => {
                    let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                    let (out_re, out_im) = complex_mul((re, im), (re, im));
                    Ok(Value::Scalar(complex_literal_from_f64_parts(
                        literal_dtype(*literal),
                        out_re,
                        out_im,
                    )))
                }
                _ => Err(EvalError::TypeMismatch {
                    primitive,
                    detail: complex_unary_unsupported_detail(primitive),
                }),
            },
        },
        Value::Tensor(tensor) => {
            // F64 fast path (e.g. Square, Sign over F64 tensors): the generic
            // arm below computes `Literal::from_f64(float_op(f64::from_bits(b)))`
            // for F64Bits, which is exactly what eval_unary_f64_tensor_fast_path
            // does (from_f64(x) == F64Bits(x.to_bits())), so it is bit-for-bit
            // identical while skipping the per-element variant match.
            if let Some(result) = eval_unary_f64_tensor_fast_path(tensor, &float_op) {
                return result;
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in tensor.elements.iter().copied() {
                let out = match literal {
                    Literal::I64(v) => Literal::I64(int_op(v)),
                    Literal::U32(v) => Literal::U32(u32_op(v)),
                    Literal::U64(v) => Literal::U64(u64_op(v)),
                    Literal::BF16Bits(bits) => {
                        let val =
                            Literal::BF16Bits(bits)
                                .as_f64()
                                .ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor elements, got bf16",
                                })?;
                        Literal::from_bf16_f32(float_op(val) as f32)
                    }
                    Literal::F16Bits(bits) => {
                        let val =
                            Literal::F16Bits(bits)
                                .as_f64()
                                .ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor elements, got f16",
                                })?;
                        Literal::from_f16_f32(float_op(val) as f32)
                    }
                    Literal::F32Bits(bits) => {
                        Literal::from_f32(float_op(f64::from(f32::from_bits(bits))) as f32)
                    }
                    Literal::F64Bits(bits) => Literal::from_f64(float_op(f64::from_bits(bits))),
                    Literal::Bool(_) => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements, got bool",
                        });
                    }
                    Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => match primitive {
                        Primitive::Sign => {
                            let (re, im) = literal_to_complex_parts(primitive, literal)?;
                            let magnitude = re.hypot(im);
                            let (out_re, out_im) = if magnitude == 0.0 {
                                (0.0, 0.0)
                            } else {
                                (re / magnitude, im / magnitude)
                            };
                            complex_literal_from_f64_parts(tensor.dtype, out_re, out_im)
                        }
                        Primitive::Square => {
                            let (re, im) = literal_to_complex_parts(primitive, literal)?;
                            let (out_re, out_im) = complex_mul((re, im), (re, im));
                            complex_literal_from_f64_parts(tensor.dtype, out_re, out_im)
                        }
                        _ => {
                            return Err(EvalError::TypeMismatch {
                                primitive,
                                detail: complex_unary_unsupported_detail(primitive),
                            });
                        }
                    },
                };
                elements.push(out);
            }

            let dtype = tensor.dtype;
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Select operation: select(cond, on_true, on_false) -> on_true where cond else on_false.
/// Same-shape `select(Bool cond, F64 on_true, F64 on_false)` fast path.
///
/// Bit-for-bit identical to the generic path: for F64 on_true/on_false the
/// generic path computes `select_literal_as_dtype(selected, F64)` which, for an
/// F64Bits value, is `Literal::from_f64(f64::from_bits(bits))` — an identity on
/// the bits. This picks the selected operand's bits directly, skipping the
/// per-element bool-condition coercion and dtype-conversion dispatch. Returns
/// `Ok(None)` if any condition element is not `Bool` or any operand element is
/// not `F64Bits`, so the caller falls through to the generic path.
#[inline]
fn select_f64_same_shape_fast_path(
    cond: &TensorValue,
    on_true: &TensorValue,
    on_false: &TensorValue,
) -> Result<Option<Value>, EvalError> {
    let mut elements = Vec::with_capacity(cond.elements.len());
    for ((c, t), f) in cond
        .elements
        .iter()
        .zip(&on_true.elements)
        .zip(&on_false.elements)
    {
        let Literal::Bool(flag) = *c else {
            return Ok(None);
        };
        let (Literal::F64Bits(true_bits), Literal::F64Bits(false_bits)) = (*t, *f) else {
            return Ok(None);
        };
        elements.push(Literal::F64Bits(if flag { true_bits } else { false_bits }));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::F64,
        cond.shape.clone(),
        elements,
    )?)))
}

pub(crate) fn eval_select(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1], &inputs[2]) {
        (Value::Scalar(cond), Value::Scalar(on_true), Value::Scalar(on_false)) => {
            let c = select_bool_condition(primitive, *cond)?;
            let val = if c { *on_true } else { *on_false };
            let lhs_dtype = literal_dtype(*on_true);
            let rhs_dtype = literal_dtype(*on_false);
            let dtype = promote_dtype(lhs_dtype, rhs_dtype);
            let promoted_val = select_literal_as_dtype(
                primitive,
                val,
                dtype,
                "expected numeric scalar for select",
                "expected integer scalar for select",
                "expected unsigned scalar for select",
            )?;
            Ok(Value::Scalar(promoted_val))
        }
        (Value::Tensor(cond), Value::Tensor(on_true), Value::Tensor(on_false)) => {
            if cond.shape != on_true.shape || cond.shape != on_false.shape {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "select requires all inputs to have the same shape".to_owned(),
                });
            }
            if cond.dtype == DType::Bool
                && on_true.dtype == DType::F64
                && on_false.dtype == DType::F64
                && let Some(value) = select_f64_same_shape_fast_path(cond, on_true, on_false)?
            {
                return Ok(value);
            }
            let dtype = promote_dtype(on_true.dtype, on_false.dtype);
            let mut elements = Vec::with_capacity(cond.elements.len());
            for ((c, t), f) in cond
                .elements
                .iter()
                .copied()
                .zip(on_true.elements.iter().copied())
                .zip(on_false.elements.iter().copied())
            {
                let flag = select_bool_condition(primitive, c)?;
                let val = if flag { t } else { f };
                elements.push(select_literal_as_dtype(
                    primitive,
                    val,
                    dtype,
                    "expected numeric tensor elements for select",
                    "expected integer tensor elements for select",
                    "expected unsigned tensor elements for select",
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                cond.shape.clone(),
                elements,
            )?))
        }
        // Tensor cond + scalar on_true + scalar on_false: broadcast scalars
        (Value::Tensor(cond), Value::Scalar(on_true), Value::Scalar(on_false)) => {
            let dtype = promote_dtype(literal_dtype(*on_true), literal_dtype(*on_false));
            let mut elements = Vec::with_capacity(cond.elements.len());
            for c in cond.elements.iter().copied() {
                let flag = select_bool_condition(primitive, c)?;
                let val = if flag { *on_true } else { *on_false };
                elements.push(select_literal_as_dtype(
                    primitive,
                    val,
                    dtype,
                    "expected numeric scalar for select",
                    "expected integer scalar for select",
                    "expected unsigned scalar for select",
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                cond.shape.clone(),
                elements,
            )?))
        }
        // Scalar cond + tensor on_true + tensor on_false
        (Value::Scalar(cond), Value::Tensor(on_true), Value::Tensor(on_false)) => {
            if on_true.shape != on_false.shape {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "select requires on_true and on_false to have the same shape"
                        .to_owned(),
                });
            }
            let flag = select_bool_condition(primitive, *cond)?;
            if flag {
                Ok(Value::Tensor(on_true.clone()))
            } else {
                Ok(Value::Tensor(on_false.clone()))
            }
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "select requires matching scalar/tensor kinds".to_owned(),
        }),
    }
}

/// Convert a SelectN index literal to a usize.
/// Supports both integer indices and boolean indices (when n_operands <= 2).
fn select_n_index_to_usize(
    idx_lit: Literal,
    n_operands: usize,
    primitive: Primitive,
) -> Result<usize, EvalError> {
    let idx = match idx_lit {
        Literal::Bool(b) => {
            if n_operands > 2 {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "select_n with boolean index requires at most 2 operands",
                });
            }
            if b { 1 } else { 0 }
        }
        _ => idx_lit.as_i64().ok_or(EvalError::TypeMismatch {
            primitive,
            detail: "select_n index must be an integer or boolean",
        })? as usize,
    };
    if idx >= n_operands {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("select_n index {idx} out of bounds for {n_operands} operands"),
        });
    }
    Ok(idx)
}

/// SelectN: select from N operands based on integer index.
///
/// inputs[0] is the index (integer values 0..N-1), inputs[1..] are the N operands.
/// For each element position i, output[i] = operands[index[i]][i].
/// Also supports boolean indices when there are at most 2 operands (false=0, true=1).
pub(crate) fn eval_select_n(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() < 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let n_operands = inputs.len() - 1;
    let index = &inputs[0];
    let operands = &inputs[1..];

    // Validate all operands have the same dtype (per upstream check_same_dtypes)
    let first_dtype = operand_dtype(&operands[0]);
    for (i, op) in operands[1..].iter().enumerate() {
        let op_dtype = operand_dtype(op);
        if op_dtype != first_dtype {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "select_n case dtypes must match: case 0 has {:?}, case {} has {:?}",
                    first_dtype,
                    i + 1,
                    op_dtype
                ),
            });
        }
    }

    match index {
        Value::Scalar(idx_lit) => {
            let idx = select_n_index_to_usize(*idx_lit, n_operands, primitive)?;
            if idx >= n_operands {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("select_n index {idx} out of bounds for {n_operands} operands"),
                });
            }
            Ok(operands[idx].clone())
        }
        Value::Tensor(idx_tensor) => {
            let first_operand = match &operands[0] {
                Value::Tensor(t) => t,
                Value::Scalar(_) => {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "select_n with tensor index requires tensor operands".into(),
                    });
                }
            };

            for op in &operands[1..] {
                match op {
                    Value::Tensor(t) => {
                        if t.shape != first_operand.shape {
                            return Err(EvalError::Unsupported {
                                primitive,
                                detail: "select_n operands must have matching shapes".into(),
                            });
                        }
                    }
                    Value::Scalar(_) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "select_n operands must all be tensors when index is tensor"
                                .into(),
                        });
                    }
                }
            }

            if idx_tensor.shape != first_operand.shape {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "select_n index shape must match operand shapes".into(),
                });
            }

            let dtype = first_operand.dtype;
            let mut elements = Vec::with_capacity(idx_tensor.elements.len());

            for (i, idx_lit) in idx_tensor.elements.iter().enumerate() {
                let idx = select_n_index_to_usize(*idx_lit, n_operands, primitive)?;

                let operand = match &operands[idx] {
                    Value::Tensor(t) => t,
                    Value::Scalar(_) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "select_n operands must all be tensors when index is tensor"
                                .into(),
                        });
                    }
                };
                elements.push(operand.elements[i]);
            }

            Ok(Value::Tensor(TensorValue::new(
                dtype,
                idx_tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// FMA: fused multiply-add: fma(a, b, c) = a * b + c with single rounding.
/// Supports NumPy-style broadcasting for all three operands.
pub(crate) fn eval_fma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    fn fma_literal(a: Literal, b: Literal, c: Literal) -> Result<Literal, &'static str> {
        match (a, b, c) {
            (Literal::I64(av), Literal::I64(bv), Literal::I64(cv)) => {
                Ok(Literal::I64(av.wrapping_mul(bv).wrapping_add(cv)))
            }
            (Literal::F64Bits(ab), Literal::F64Bits(bb), Literal::F64Bits(cb)) => {
                let af = f64::from_bits(ab);
                let bf = f64::from_bits(bb);
                let cf = f64::from_bits(cb);
                Ok(Literal::from_f64(af.mul_add(bf, cf)))
            }
            (Literal::F32Bits(ab), Literal::F32Bits(bb), Literal::F32Bits(cb)) => {
                let af = f32::from_bits(ab);
                let bf = f32::from_bits(bb);
                let cf = f32::from_bits(cb);
                Ok(Literal::from_f32(af.mul_add(bf, cf)))
            }
            (
                Literal::Complex64Bits(a_re, a_im),
                Literal::Complex64Bits(b_re, b_im),
                Literal::Complex64Bits(c_re, c_im),
            ) => {
                let a = (f32::from_bits(a_re) as f64, f32::from_bits(a_im) as f64);
                let b = (f32::from_bits(b_re) as f64, f32::from_bits(b_im) as f64);
                let c = (f32::from_bits(c_re) as f64, f32::from_bits(c_im) as f64);
                let prod = complex_mul(a, b);
                let result = complex_add(prod, c);
                Ok(Literal::from_complex64(result.0 as f32, result.1 as f32))
            }
            (
                Literal::Complex128Bits(a_re, a_im),
                Literal::Complex128Bits(b_re, b_im),
                Literal::Complex128Bits(c_re, c_im),
            ) => {
                let a = (f64::from_bits(a_re), f64::from_bits(a_im));
                let b = (f64::from_bits(b_re), f64::from_bits(b_im));
                let c = (f64::from_bits(c_re), f64::from_bits(c_im));
                let prod = complex_mul(a, b);
                let result = complex_add(prod, c);
                Ok(Literal::from_complex128(result.0, result.1))
            }
            _ => {
                let af = a.as_f64().ok_or("expected numeric")?;
                let bf = b.as_f64().ok_or("expected numeric")?;
                let cf = c.as_f64().ok_or("expected numeric")?;
                Ok(Literal::from_f64(af.mul_add(bf, cf)))
            }
        }
    }

    fn shape_of(v: &Value) -> Shape {
        match v {
            Value::Scalar(_) => Shape { dims: vec![] },
            Value::Tensor(t) => t.shape.clone(),
        }
    }

    fn get_literal(v: &Value, idx: usize) -> Literal {
        match v {
            Value::Scalar(l) => *l,
            Value::Tensor(t) => t.elements[idx],
        }
    }

    fn get_dtype(v: &Value) -> DType {
        match v {
            Value::Scalar(l) => literal_dtype(*l),
            Value::Tensor(t) => t.dtype,
        }
    }

    let shape_a = shape_of(&inputs[0]);
    let shape_b = shape_of(&inputs[1]);
    let shape_c = shape_of(&inputs[2]);

    let out_shape_ab = broadcast_shape(&shape_a, &shape_b).ok_or(EvalError::ShapeMismatch {
        primitive,
        left: shape_a.clone(),
        right: shape_b.clone(),
    })?;

    let out_shape = broadcast_shape(&out_shape_ab, &shape_c).ok_or(EvalError::ShapeMismatch {
        primitive,
        left: out_shape_ab,
        right: shape_c.clone(),
    })?;

    if out_shape.rank() == 0 {
        let a = get_literal(&inputs[0], 0);
        let b = get_literal(&inputs[1], 0);
        let c = get_literal(&inputs[2], 0);
        return fma_literal(a, b, c)
            .map(Value::Scalar)
            .map_err(|e| EvalError::TypeMismatch {
                primitive,
                detail: e,
            });
    }

    let out_count = out_shape.element_count().unwrap_or(0) as usize;
    let out_strides = compute_strides(&out_shape.dims);
    let a_strides = broadcast_strides(&shape_a, &out_shape);
    let b_strides = broadcast_strides(&shape_b, &out_shape);
    let c_strides = broadcast_strides(&shape_c, &out_shape);

    let out_dtype = promote_dtype(
        promote_dtype(get_dtype(&inputs[0]), get_dtype(&inputs[1])),
        get_dtype(&inputs[2]),
    );

    let mut multi = Vec::with_capacity(out_strides.len());
    let mut elements = Vec::with_capacity(out_count);
    for flat_idx in 0..out_count {
        flat_to_multi_into(flat_idx, &out_strides, &mut multi);
        let a_idx = broadcast_flat_index(&multi, &a_strides);
        let b_idx = broadcast_flat_index(&multi, &b_strides);
        let c_idx = broadcast_flat_index(&multi, &c_strides);

        let a = get_literal(&inputs[0], a_idx);
        let b = get_literal(&inputs[1], b_idx);
        let c = get_literal(&inputs[2], c_idx);

        elements.push(fma_literal(a, b, c).map_err(|e| EvalError::TypeMismatch {
            primitive,
            detail: e,
        })?);
    }

    Ok(Value::Tensor(TensorValue::new(
        out_dtype, out_shape, elements,
    )?))
}

/// Clamp: clamp(lo, x, hi) = min(max(lo, x), hi).
/// Supports scalar and tensor inputs with broadcasting.
pub(crate) fn eval_clamp(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    fn clamp_f64(lo: f64, x: f64, hi: f64) -> f64 {
        if lo.is_nan() || x.is_nan() || hi.is_nan() {
            return f64::NAN;
        }
        let lower_bounded = if x < lo { lo } else { x };
        if lower_bounded > hi {
            hi
        } else {
            lower_bounded
        }
    }

    fn clamp_f32(lo: f32, x: f32, hi: f32) -> f32 {
        if lo.is_nan() || x.is_nan() || hi.is_nan() {
            return f32::NAN;
        }
        let lower_bounded = if x < lo { lo } else { x };
        if lower_bounded > hi {
            hi
        } else {
            lower_bounded
        }
    }

    fn clamp_literal(
        lo: Literal,
        x: Literal,
        hi: Literal,
        target_dtype: Option<DType>,
    ) -> Result<Literal, &'static str> {
        match (lo, x, hi) {
            (Literal::I64(lov), Literal::I64(xv), Literal::I64(hiv)) => {
                Ok(Literal::I64(lov.max(xv).min(hiv)))
            }
            (Literal::F64Bits(lob), Literal::F64Bits(xb), Literal::F64Bits(hib)) => {
                let lof = f64::from_bits(lob);
                let xf = f64::from_bits(xb);
                let hif = f64::from_bits(hib);
                Ok(Literal::from_f64(clamp_f64(lof, xf, hif)))
            }
            (Literal::F32Bits(lob), Literal::F32Bits(xb), Literal::F32Bits(hib)) => {
                let lof = f32::from_bits(lob);
                let xf = f32::from_bits(xb);
                let hif = f32::from_bits(hib);
                Ok(Literal::from_f32(clamp_f32(lof, xf, hif)))
            }
            _ => {
                // Mixed types: promote to f64
                let lof = match lo {
                    Literal::I64(v) => v as f64,
                    Literal::U32(v) => v as f64,
                    Literal::U64(v) => v as f64,
                    Literal::BF16Bits(bits) => Literal::BF16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F16Bits(bits) => Literal::F16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F32Bits(b) => f64::from(f32::from_bits(b)),
                    Literal::F64Bits(b) => f64::from_bits(b),
                    Literal::Bool(_) => return Err("clamp does not support bool"),
                    Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => {
                        return Err("clamp is not supported for complex dtypes");
                    }
                };
                let xf = match x {
                    Literal::I64(v) => v as f64,
                    Literal::U32(v) => v as f64,
                    Literal::U64(v) => v as f64,
                    Literal::BF16Bits(bits) => Literal::BF16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F16Bits(bits) => Literal::F16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F32Bits(b) => f64::from(f32::from_bits(b)),
                    Literal::F64Bits(b) => f64::from_bits(b),
                    Literal::Bool(_) => return Err("clamp does not support bool"),
                    Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => {
                        return Err("clamp is not supported for complex dtypes");
                    }
                };
                let hif = match hi {
                    Literal::I64(v) => v as f64,
                    Literal::U32(v) => v as f64,
                    Literal::U64(v) => v as f64,
                    Literal::BF16Bits(bits) => Literal::BF16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F16Bits(bits) => Literal::F16Bits(bits).as_f64().unwrap_or_default(),
                    Literal::F32Bits(b) => f64::from(f32::from_bits(b)),
                    Literal::F64Bits(b) => f64::from_bits(b),
                    Literal::Bool(_) => return Err("clamp does not support bool"),
                    Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => {
                        return Err("clamp is not supported for complex dtypes");
                    }
                };
                let clamped = clamp_f64(lof, xf, hif);
                match target_dtype {
                    Some(dt) => Ok(real_literal_from_f64(dt, clamped)),
                    None => Ok(Literal::from_f64(clamped)),
                }
            }
        }
    }

    // Fast path for an F64 tensor with F64 scalar bounds (the common
    // clamp(min, x, max) pattern). Bit-for-bit identical to the generic
    // per-element path: for all-F64Bits operands `clamp_literal` computes
    // `Literal::from_f64(clamp_f64(lo, x, hi))` (ignoring target_dtype), which
    // is exactly what this does, but it extracts the bounds once and skips the
    // per-element 3-tuple match. Returns None for non-F64 inputs.
    fn clamp_f64_scalar_bounds(
        x: &TensorValue,
        lo: Literal,
        hi: Literal,
    ) -> Result<Option<Value>, EvalError> {
        if x.dtype != DType::F64 {
            return Ok(None);
        }
        let (Literal::F64Bits(lo_bits), Literal::F64Bits(hi_bits)) = (lo, hi) else {
            return Ok(None);
        };
        let lof = f64::from_bits(lo_bits);
        let hif = f64::from_bits(hi_bits);
        let mut elements = Vec::with_capacity(x.elements.len());
        for &elem in &x.elements {
            let Literal::F64Bits(xb) = elem else {
                return Ok(None);
            };
            elements.push(Literal::from_f64(clamp_f64(lof, f64::from_bits(xb), hif)));
        }
        Ok(Some(Value::Tensor(TensorValue::new(
            DType::F64,
            x.shape.clone(),
            elements,
        )?)))
    }

    match (&inputs[0], &inputs[1], &inputs[2]) {
        (Value::Scalar(lo), Value::Scalar(x), Value::Scalar(hi)) => {
            let result = clamp_literal(*lo, *x, *hi, None)
                .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?;
            Ok(Value::Scalar(result))
        }
        // JAX order: clamp(min, x, max) with scalar bounds
        (Value::Scalar(lo), Value::Tensor(x), Value::Scalar(hi)) => {
            if let Some(value) = clamp_f64_scalar_bounds(x, *lo, *hi)? {
                return Ok(value);
            }
            let mut elements = Vec::with_capacity(x.elements.len());
            for elem in x.elements.iter().copied() {
                elements.push(
                    clamp_literal(*lo, elem, *hi, Some(x.dtype))
                        .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?,
                );
            }
            Ok(Value::Tensor(TensorValue::new(
                x.dtype,
                x.shape.clone(),
                elements,
            )?))
        }
        // Legacy (x, lo, hi) order kept for compatibility
        (Value::Tensor(x), Value::Scalar(lo), Value::Scalar(hi)) => {
            if let Some(value) = clamp_f64_scalar_bounds(x, *lo, *hi)? {
                return Ok(value);
            }
            let mut elements = Vec::with_capacity(x.elements.len());
            for elem in x.elements.iter().copied() {
                elements.push(
                    clamp_literal(*lo, elem, *hi, Some(x.dtype))
                        .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?,
                );
            }
            Ok(Value::Tensor(TensorValue::new(
                x.dtype,
                x.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lo), Value::Tensor(x), Value::Tensor(hi)) => {
            // Support broadcasting for tensor bounds
            let out_shape = broadcast_shape(&x.shape, &lo.shape)
                .and_then(|s| broadcast_shape(&s, &hi.shape))
                .ok_or(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "clamp shapes not broadcast-compatible: min={:?}, x={:?}, max={:?}",
                        lo.shape, x.shape, hi.shape
                    ),
                })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = compute_strides(&out_shape.dims);
            let lo_strides = broadcast_strides(&lo.shape, &out_shape);
            let x_strides = broadcast_strides(&x.shape, &out_shape);
            let hi_strides = broadcast_strides(&hi.shape, &out_shape);

            let mut elements = Vec::with_capacity(out_count);
            let mut multi = Vec::with_capacity(out_strides.len());
            for flat_idx in 0..out_count {
                flat_to_multi_into(flat_idx, &out_strides, &mut multi);
                let lo_idx = broadcast_flat_index(&multi, &lo_strides);
                let x_idx = broadcast_flat_index(&multi, &x_strides);
                let hi_idx = broadcast_flat_index(&multi, &hi_strides);

                let lov = lo.elements[lo_idx];
                let xv = x.elements[x_idx];
                let hiv = hi.elements[hi_idx];
                elements.push(
                    clamp_literal(lov, xv, hiv, Some(x.dtype))
                        .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?,
                );
            }
            Ok(Value::Tensor(TensorValue::new(
                x.dtype, out_shape, elements,
            )?))
        }
        // Scalar lo with tensor x and tensor hi (broadcasts lo)
        (Value::Scalar(lo), Value::Tensor(x), Value::Tensor(hi)) => {
            let out_shape = broadcast_shape(&x.shape, &hi.shape).ok_or(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "clamp shapes not broadcast-compatible: x={:?}, max={:?}",
                    x.shape, hi.shape
                ),
            })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = compute_strides(&out_shape.dims);
            let x_strides = broadcast_strides(&x.shape, &out_shape);
            let hi_strides = broadcast_strides(&hi.shape, &out_shape);

            let mut elements = Vec::with_capacity(out_count);
            let mut multi = Vec::with_capacity(out_strides.len());
            for flat_idx in 0..out_count {
                flat_to_multi_into(flat_idx, &out_strides, &mut multi);
                let x_idx = broadcast_flat_index(&multi, &x_strides);
                let hi_idx = broadcast_flat_index(&multi, &hi_strides);

                let xv = x.elements[x_idx];
                let hiv = hi.elements[hi_idx];
                elements.push(
                    clamp_literal(*lo, xv, hiv, Some(x.dtype))
                        .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?,
                );
            }
            Ok(Value::Tensor(TensorValue::new(
                x.dtype, out_shape, elements,
            )?))
        }
        // Tensor lo with tensor x and scalar hi (broadcasts hi)
        (Value::Tensor(lo), Value::Tensor(x), Value::Scalar(hi)) => {
            let out_shape = broadcast_shape(&x.shape, &lo.shape).ok_or(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "clamp shapes not broadcast-compatible: min={:?}, x={:?}",
                    lo.shape, x.shape
                ),
            })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = compute_strides(&out_shape.dims);
            let lo_strides = broadcast_strides(&lo.shape, &out_shape);
            let x_strides = broadcast_strides(&x.shape, &out_shape);

            let mut elements = Vec::with_capacity(out_count);
            let mut multi = Vec::with_capacity(out_strides.len());
            for flat_idx in 0..out_count {
                flat_to_multi_into(flat_idx, &out_strides, &mut multi);
                let lo_idx = broadcast_flat_index(&multi, &lo_strides);
                let x_idx = broadcast_flat_index(&multi, &x_strides);

                let lov = lo.elements[lo_idx];
                let xv = x.elements[x_idx];
                elements.push(
                    clamp_literal(lov, xv, *hi, Some(x.dtype))
                        .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?,
                );
            }
            Ok(Value::Tensor(TensorValue::new(
                x.dtype, out_shape, elements,
            )?))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "clamp requires broadcast-compatible inputs".to_owned(),
        }),
    }
}

/// High-accuracy `erf` (agrees with JAX/scipy to ~1e-12 or better, vs the old
/// Abramowitz & Stegun 7.1.26 form's ~1.5e-7).
///
/// Two stable elementary series (no magic rational-Chebyshev constants):
/// - `|x| < 3.5`: the Maclaurin series `erf(x) = (2/√π) Σ (-1)ⁿ x^(2n+1)/(n!(2n+1))`,
///   excellent for the common range (`|x| ≤ 2` reaches ~1e-15; cancellation grows
///   to ~3e-12 by 3.5).
/// - `3.5 ≤ |x| < 6`: `erf = 1 - erfc` with `erfc` from its asymptotic series
///   `e^{-x²}/(x√π) Σ (-1)ⁿ (2n-1)!!/(2x²)ⁿ`, summed until the terms stop
///   shrinking (asymptotic divergence) — accurate to ~f64 there.
/// - `|x| ≥ 6`: `erf` is `±1` to f64 precision (`erfc(6) ≈ 2e-17`).
pub(crate) fn erf_approx(x: f64) -> f64 {
    use std::f64::consts::FRAC_2_SQRT_PI; // 2/√π
    if x == 0.0 {
        return x; // preserve signed zero
    }
    if x.is_nan() {
        return f64::NAN;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax < 3.5 {
        let x2 = ax * ax;
        let mut term = ax; // n = 0: x^1 / (0! · 1)
        let mut sum = ax;
        let mut n = 1.0_f64;
        loop {
            // tₙ = tₙ₋₁ · (-x²/n) · (2n-1)/(2n+1)
            term *= -x2 / n * (2.0 * n - 1.0) / (2.0 * n + 1.0);
            sum += term;
            if term.abs() <= sum.abs() * f64::EPSILON || n > 200.0 {
                break;
            }
            n += 1.0;
        }
        sign * FRAC_2_SQRT_PI * sum
    } else if ax < 6.0 {
        let x2 = ax * ax;
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        let mut prev_abs = f64::INFINITY;
        let mut n = 1.0_f64;
        loop {
            // tₙ = tₙ₋₁ · -(2n-1)/(2x²)
            term *= -(2.0 * n - 1.0) / (2.0 * x2);
            let mag = term.abs();
            if mag > prev_abs {
                break; // asymptotic series started to diverge
            }
            sum += term;
            prev_abs = mag;
            if mag <= f64::EPSILON || n > 100.0 {
                break;
            }
            n += 1.0;
        }
        let erfc = (-x2).exp() / (ax * std::f64::consts::PI.sqrt()) * sum;
        sign * (1.0 - erfc)
    } else {
        sign // erf(±6) == ±1.0 to f64 precision
    }
}

const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

const LANCZOS_G: f64 = 7.0;

#[inline]
fn is_near_integer(x: f64) -> bool {
    (x - x.round()).abs() < 1e-14
}

pub(crate) fn lgamma_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if x <= 0.0 && is_near_integer(x) {
        return f64::INFINITY;
    }

    if x < 0.5 {
        let sin_term = (std::f64::consts::PI * x).sin().abs();
        if sin_term == 0.0 {
            return f64::INFINITY;
        }
        return std::f64::consts::PI.ln() - sin_term.ln() - lgamma_approx(1.0 - x);
    }

    let z = x - 1.0;
    let mut acc = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (z + idx as f64);
    }

    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + acc.ln()
}

pub(crate) fn digamma_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if x <= 0.0 && is_near_integer(x) {
        return f64::NEG_INFINITY;
    }

    if x < 0.5 {
        let pi_x = std::f64::consts::PI * x;
        return digamma_approx(1.0 - x) - std::f64::consts::PI / pi_x.tan();
    }

    let mut shifted = x;
    let mut result = 0.0;
    while shifted < 8.0 {
        result -= 1.0 / shifted;
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv4 * inv4;
    let inv10 = inv8 * inv2;

    result + shifted.ln() - 0.5 * inv - inv2 / 12.0 + inv4 / 120.0 - inv6 / 252.0 + inv8 / 240.0
        - 5.0 * inv10 / 660.0
}

pub(crate) fn polygamma_approx(n: i64, x: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return digamma_approx(x);
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if x <= 0.0 && is_near_integer(x) {
        return if n % 2 == 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        let cot_deriv = polygamma_cot_derivative(n, pi * x) * pi.powi(n as i32 + 1);
        return sign * polygamma_approx(n, 1.0 - x) + cot_deriv;
    }

    let mut shifted = x;
    let mut result = 0.0;
    let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
    let n_fact = factorial(n as u32) as f64;
    // Shift to larger x for better asymptotic convergence
    while shifted < 100.0 {
        result += sign * n_fact / shifted.powi(n as i32 + 1);
        shifted += 1.0;
    }

    result + polygamma_asymptotic(n, shifted)
}

fn polygamma_cot_derivative(n: i64, x: f64) -> f64 {
    let sin_x = x.sin();
    let cos_x = x.cos();
    if sin_x.abs() < 1e-15 {
        return if n % 2 == 0 { f64::NAN } else { f64::INFINITY };
    }
    match n {
        1 => -1.0 / (sin_x * sin_x),
        2 => 2.0 * cos_x / sin_x.powi(3),
        3 => -2.0 * (1.0 + 3.0 * cos_x * cos_x) / sin_x.powi(4),
        _ => {
            let h = 1e-6;
            (polygamma_cot_derivative(n - 1, x + h) - polygamma_cot_derivative(n - 1, x - h))
                / (2.0 * h)
        }
    }
}

fn polygamma_asymptotic(n: i64, x: f64) -> f64 {
    let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
    let n_fact = factorial(n as u32) as f64;
    let inv = 1.0 / x;

    // Bernoulli numbers B_2k for k=0..6 (indices 0,2,4,6,8,10,12)
    // B_0=1, B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66, B_12=-691/2730
    let bernoulli: [f64; 13] = [
        1.0,
        -0.5,
        1.0 / 6.0,
        0.0,
        -1.0 / 30.0,
        0.0,
        1.0 / 42.0,
        0.0,
        -1.0 / 30.0,
        0.0,
        5.0 / 66.0,
        0.0,
        -691.0 / 2730.0,
    ];

    let mut sum = 0.0;
    let mut pow = inv.powi(n as i32);
    sum += sign * factorial((n - 1) as u32) as f64 * pow;

    pow *= inv;
    sum += sign * n_fact * 0.5 * pow; // half-term, pow = inv^(n+1)

    // Bernoulli corrections (DLMF 5.15.8): Σ_k B_2k · (2k+n-1)!/(2k)! · z^{-(2k+n)}
    // for k=1..6. The coefficient is rising_factorial(2k+1, n-1) = (2k+n-1)!/(2k)!,
    // and term k has power n+2k. The previous code used the wrong power (n+2k+1,
    // because it multiplied by inv² *before* the k=1 term) and the wrong factorial
    // rising_factorial(n+1, 2k-1), which dropped the leading correction entirely —
    // e.g. ~1.6e-7 error for trigamma(1) at the x=100 shift point.
    let inv2 = inv * inv;
    let mut bpow = pow * inv; // inv^{n+2}, the k=1 power
    for k in 1..=6 {
        let rising = rising_factorial(2 * k as u32 + 1, n as u32 - 1);
        sum += sign * bernoulli[2 * k] * rising as f64 * bpow;
        bpow *= inv2;
    }
    sum
}

fn factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

fn rising_factorial(x: u32, n: u32) -> u64 {
    (0..n).map(|i| (x + i) as u64).product()
}

#[cfg(test)]
pub(crate) fn trigamma_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if x <= 0.0 && is_near_integer(x) {
        return f64::INFINITY;
    }

    if x < 0.5 {
        let sin_px = (std::f64::consts::PI * x).sin();
        if sin_px == 0.0 {
            return f64::INFINITY;
        }
        let csc2 = 1.0 / (sin_px * sin_px);
        return std::f64::consts::PI.powi(2) * csc2 - trigamma_approx(1.0 - x);
    }

    let mut shifted = x;
    let mut result = 0.0;
    while shifted < 8.0 {
        result += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;

    result + inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + 5.0 * inv11 / 66.0
}

pub(crate) fn erf_inv_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= -1.0 {
        return if x == -1.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if x >= 1.0 {
        return if x == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    if x == 0.0 {
        return x;
    }

    // Winitzki initial approximation with Newton refinement.
    let a = 0.147_f64;
    let ln_term = (1.0 - x * x).ln();
    let t = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let mut y = x.signum() * ((t * t - ln_term / a).sqrt() - t).sqrt();

    let coeff = 2.0 / std::f64::consts::PI.sqrt();
    for _ in 0..3 {
        let err = erf_approx(y) - x;
        let deriv = coeff * (-y * y).exp();
        y -= err / deriv;
    }
    y
}

pub(crate) fn eval_lgamma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise(primitive, inputs, lgamma_approx)
}

pub(crate) fn eval_digamma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise(primitive, inputs, digamma_approx)
}

pub(crate) fn eval_polygamma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    let n_val = &inputs[0];
    let x_val = &inputs[1];

    match (n_val, x_val) {
        (Value::Scalar(n_lit), Value::Scalar(x_lit)) => {
            let n = polygamma_literal_to_i64(*n_lit, primitive)?;
            let x = polygamma_literal_to_f64(*x_lit, primitive)?;
            Ok(Value::Scalar(Literal::from_f64(polygamma_approx(n, x))))
        }
        (Value::Scalar(n_lit), Value::Tensor(x_tensor)) => {
            let n = polygamma_literal_to_i64(*n_lit, primitive)?;
            let mut elements = Vec::with_capacity(x_tensor.elements.len());
            for x_elem in &x_tensor.elements {
                let x = polygamma_literal_to_f64(*x_elem, primitive)?;
                elements.push(Literal::from_f64(polygamma_approx(n, x)));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::F64,
                x_tensor.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(n_tensor), Value::Scalar(x_lit)) => {
            let x = polygamma_literal_to_f64(*x_lit, primitive)?;
            let mut elements = Vec::with_capacity(n_tensor.elements.len());
            for n_elem in &n_tensor.elements {
                let n = polygamma_literal_to_i64(*n_elem, primitive)?;
                elements.push(Literal::from_f64(polygamma_approx(n, x)));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::F64,
                n_tensor.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(n_tensor), Value::Tensor(x_tensor)) => {
            let out_shape = broadcast_shape(&n_tensor.shape, &x_tensor.shape).ok_or(
                EvalError::ShapeMismatch {
                    primitive,
                    left: n_tensor.shape.clone(),
                    right: x_tensor.shape.clone(),
                },
            )?;
            let n_strides = broadcast_strides(&n_tensor.shape, &out_shape);
            let x_strides = broadcast_strides(&x_tensor.shape, &out_shape);
            let out_strides = compute_strides(&out_shape.dims);
            let total: usize = out_shape.dims.iter().map(|&d| d as usize).product();
            let mut elements = Vec::with_capacity(total.max(1));
            let mut multi = vec![0usize; out_shape.dims.len()];
            for flat in 0..total.max(1) {
                flat_to_multi_into(flat, &out_strides, &mut multi);
                let n_idx = broadcast_flat_index(&multi, &n_strides);
                let x_idx = broadcast_flat_index(&multi, &x_strides);
                let n = polygamma_literal_to_i64(n_tensor.elements[n_idx], primitive)?;
                let x = polygamma_literal_to_f64(x_tensor.elements[x_idx], primitive)?;
                elements.push(Literal::from_f64(polygamma_approx(n, x)));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::F64,
                out_shape,
                elements,
            )?))
        }
    }
}

fn polygamma_literal_to_i64(lit: Literal, primitive: Primitive) -> Result<i64, EvalError> {
    match lit {
        Literal::I64(v) => Ok(v),
        Literal::U32(v) => Ok(i64::from(v)),
        Literal::U64(v) => Ok(v as i64),
        Literal::F32Bits(bits) => Ok(f32::from_bits(bits) as i64),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits) as i64),
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "polygamma order must be numeric".to_string(),
        }),
    }
}

fn polygamma_literal_to_f64(lit: Literal, primitive: Primitive) -> Result<f64, EvalError> {
    match lit {
        Literal::I64(v) => Ok(v as f64),
        Literal::U32(v) => Ok(f64::from(v)),
        Literal::U64(v) => Ok(v as f64),
        Literal::F32Bits(bits) => Ok(f64::from(f32::from_bits(bits))),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits)),
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "polygamma argument must be numeric".to_string(),
        }),
    }
}

pub(crate) fn eval_erf_inv(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise(primitive, inputs, erf_inv_approx)
}

fn igamma_series(a: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut n = 1.0;
    while term.abs() > sum.abs() * 1e-15 && n < 1000.0 {
        term *= x / (a + n);
        sum += term;
        n += 1.0;
    }
    sum * (-x + a * x.ln() - lgamma_approx(a)).exp()
}

fn igammac_cf(a: f64, x: f64) -> f64 {
    let mut f = 1.0e-30_f64;
    let mut c = f;
    let mut d = 0.0;
    for n in 1..=1000 {
        let an = if n == 1 {
            1.0
        } else if n % 2 == 0 {
            let k = (n / 2) as f64;
            k * (a - k) / ((a + 2.0 * k - 1.0) * (a + 2.0 * k))
        } else {
            let k = ((n - 1) / 2) as f64;
            -((a + k) * (a + k + 1.0) / ((a + 2.0 * k) * (a + 2.0 * k + 1.0)))
        };
        let bn = if n == 1 { x - a + 1.0 } else { 2.0 };
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    f * (-x + a * x.ln() - lgamma_approx(a)).exp()
}

/// ∂/∂a of the regularized lower incomplete gamma P(a, x) = igamma(a, x).
///
/// Translated from JAX's `igamma_grad_a_impl` (lax/special.py): a power
/// series for the small-x regime and a continued fraction for the large-x
/// regime, each carrying the derivative recurrence alongside the value. The
/// branch split (`x > 1 && x > a` → continued fraction) matches JAX exactly.
pub(crate) fn igamma_grad_a_approx(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    // domain error: x < 0 or a <= 0 → NaN (matches JAX)
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let ax_exponent = a * x.ln() - x - lgamma_approx(a);
    // underflow: ax ≈ 0 makes the whole derivative ≈ 0
    if ax_exponent < -f64::MAX.ln() {
        return 0.0;
    }
    let ax = ax_exponent.exp();
    if x > 1.0 && x > a {
        -igammac_cf_grad_a(ax, x, a)
    } else {
        igamma_series_grad_a(ax, x, a)
    }
}

/// Power-series branch of [`igamma_grad_a_approx`] (JAX `_igamma_series`,
/// DERIVATIVE mode).
fn igamma_series_grad_a(ax: f64, x: f64, a: f64) -> f64 {
    let eps = f64::EPSILON;
    let mut r = a;
    let mut c = 1.0_f64;
    let mut ans = 1.0_f64;
    let mut dc_da = 0.0_f64;
    let mut dans_da = 0.0_f64;
    for _ in 0..2000 {
        r += 1.0;
        dc_da = dc_da * (x / r) - (c * x) / (r * r);
        dans_da += dc_da;
        c *= x / r;
        ans += c;
        if dans_da != 0.0 && (dc_da / dans_da).abs() <= eps {
            break;
        }
    }
    let dlogax_da = x.ln() - digamma_approx(a + 1.0);
    ax * (ans * dlogax_da + dans_da) / a
}

/// Continued-fraction branch of [`igamma_grad_a_approx`] (JAX
/// `_igammac_continued_fraction`, DERIVATIVE mode). Returns ∂/∂a of the
/// *upper* regularized incomplete gamma; the caller negates it.
fn igammac_cf_grad_a(ax: f64, x: f64, a: f64) -> f64 {
    let eps = f64::EPSILON;
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0_f64;
    let mut pkm2 = 1.0_f64;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    let mut dpkm2_da = 0.0_f64;
    let mut dqkm2_da = 0.0_f64;
    let mut dpkm1_da = 0.0_f64;
    let mut dqkm1_da = -x;
    let mut dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
    while c < 2000.0 {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        let dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
        let dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;

        let grad_conditional;
        if qk != 0.0 {
            let r = pk / qk;
            ans = r;
            let new_dans_da = (dpk_da - ans * dqk_da) / qk;
            grad_conditional = (new_dans_da - dans_da).abs();
            dans_da = new_dans_da;
        } else {
            grad_conditional = 1.0;
        }

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        dpkm2_da = dpkm1_da;
        dqkm2_da = dqkm1_da;
        dpkm1_da = dpk_da;
        dqkm1_da = dqk_da;

        // Rescale to avoid overflow once the numerator grows large.
        if pk.abs() > 1.0 / eps {
            pkm2 *= eps;
            pkm1 *= eps;
            qkm2 *= eps;
            qkm1 *= eps;
            dpkm2_da *= eps;
            dqkm2_da *= eps;
            dpkm1_da *= eps;
            dqkm1_da *= eps;
        }

        if grad_conditional <= eps {
            break;
        }
    }
    let dlogax_da = x.ln() - digamma_approx(a);
    ax * (ans * dlogax_da + dans_da)
}

pub(crate) fn igamma_approx(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    if x < a + 1.0 {
        igamma_series(a, x)
    } else {
        1.0 - igammac_cf(a, x)
    }
}

pub(crate) fn igammac_approx(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - igamma_series(a, x)
    } else {
        igammac_cf(a, x)
    }
}

pub(crate) fn eval_igamma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    eval_binary_elementwise(
        primitive,
        inputs,
        |a, x| igamma_approx(a as f64, x as f64) as i64,
        igamma_approx,
    )
}

pub(crate) fn eval_igammac(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    eval_binary_elementwise(
        primitive,
        inputs,
        |a, x| igammac_approx(a as f64, x as f64) as i64,
        igammac_approx,
    )
}

/// Elementwise ∂/∂a of the regularized lower incomplete gamma P(a, x).
/// Exposed for the AD layer's igamma/igammac VJP and JVP rules (JAX's
/// `igamma_grad_a` primitive). `Q(a,x) = 1 - P(a,x)`, so igammac's
/// derivative wrt `a` is the negation of this.
pub fn eval_igamma_grad_a(inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive: Primitive::Igamma,
            expected: 2,
            actual: inputs.len(),
        });
    }
    eval_binary_elementwise(
        Primitive::Igamma,
        inputs,
        |a, x| igamma_grad_a_approx(a as f64, x as f64) as i64,
        igamma_grad_a_approx,
    )
}

fn betainc_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=200 {
        let m = m as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

pub(crate) fn betainc_approx(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }

    let bt = (lgamma_approx(a + b) - lgamma_approx(a) - lgamma_approx(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betainc_cf(a, b, x) / a
    } else {
        1.0 - bt * betainc_cf(b, a, 1.0 - x) / b
    }
}

pub(crate) fn eval_betainc(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }
    eval_ternary_elementwise(primitive, inputs, betainc_approx)
}

fn eval_ternary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    op: impl Fn(f64, f64, f64) -> f64,
) -> Result<Value, EvalError> {
    let a_val = &inputs[0];
    let b_val = &inputs[1];
    let x_val = &inputs[2];

    // All scalars -> scalar output
    if let (Value::Scalar(a), Value::Scalar(b), Value::Scalar(x)) = (a_val, b_val, x_val) {
        let a_f = a.as_f64().unwrap_or(0.0);
        let b_f = b.as_f64().unwrap_or(0.0);
        let x_f = x.as_f64().unwrap_or(0.0);
        return Ok(Value::Scalar(Literal::from_f64(op(a_f, b_f, x_f))));
    }

    // Convert scalars to 0-d tensors for uniform handling
    let scalar_to_tensor = |v: &Value| -> TensorValue {
        match v {
            Value::Scalar(lit) => TensorValue::new(DType::F64, Shape { dims: vec![] }, vec![*lit])
                .expect("scalar->tensor conversion"),
            Value::Tensor(t) => t.clone(),
        }
    };

    let t_a = scalar_to_tensor(a_val);
    let t_b = scalar_to_tensor(b_val);
    let t_x = scalar_to_tensor(x_val);

    // Broadcast first two shapes, then result with third
    let ab_shape = broadcast_shape(&t_a.shape, &t_b.shape).ok_or(EvalError::ShapeMismatch {
        primitive,
        left: t_a.shape.clone(),
        right: t_b.shape.clone(),
    })?;
    let out_shape = broadcast_shape(&ab_shape, &t_x.shape).ok_or(EvalError::ShapeMismatch {
        primitive,
        left: ab_shape.clone(),
        right: t_x.shape.clone(),
    })?;

    let total_elements: usize = out_shape.dims.iter().map(|&d| d as usize).product();
    if total_elements == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            DType::F64,
            out_shape,
            vec![],
        )?));
    }

    let out_strides = compute_strides(&out_shape.dims);
    let a_strides = broadcast_strides(&t_a.shape, &out_shape);
    let b_strides = broadcast_strides(&t_b.shape, &out_shape);
    let x_strides = broadcast_strides(&t_x.shape, &out_shape);

    let mut elements = Vec::with_capacity(total_elements);
    let mut multi = Vec::with_capacity(out_shape.rank());

    for flat in 0..total_elements {
        flat_to_multi_into(flat, &out_strides, &mut multi);
        let a_idx = broadcast_flat_index(&multi, &a_strides);
        let b_idx = broadcast_flat_index(&multi, &b_strides);
        let x_idx = broadcast_flat_index(&multi, &x_strides);

        let a_f = t_a.elements[a_idx].as_f64().unwrap_or(0.0);
        let b_f = t_b.elements[b_idx].as_f64().unwrap_or(0.0);
        let x_f = t_x.elements[x_idx].as_f64().unwrap_or(0.0);
        elements.push(Literal::from_f64(op(a_f, b_f, x_f)));
    }

    Ok(Value::Tensor(TensorValue::new(
        DType::F64,
        out_shape,
        elements,
    )?))
}

pub(crate) fn hurwitz_zeta_approx(x: f64, q: f64) -> f64 {
    if x.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    if q <= 0.0 {
        return f64::NAN;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        return f64::NAN;
    }

    // Euler-Maclaurin summation for the Hurwitz zeta ζ(s, q) = Σ (n+q)^{-s}:
    // sum the first N terms directly, then approximate the tail Σ_{n≥N} with the
    // integral + half-term + Bernoulli corrections. This reaches ~1e-13 for s>1
    // with N=10, M=8, whereas the previous naive 10000-term truncation left an
    // O(N^{1-s}/(s-1)) tail error (~1e-4 at s=2, far worse as s→1).
    const N: usize = 10;
    // c_j = B_{2j} / (2j)! for j = 1..=8 (Bernoulli numbers over factorials).
    const C: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30240.0,
        -1.0 / 1_209_600.0,
        1.0 / 47_900_160.0,
        -691.0 / 1_307_674_368_000.0,
        1.0 / 74_724_249_600.0,
        -3617.0 / 10_670_622_842_880_000.0,
    ];

    let s = x;
    let mut sum = 0.0;
    for n in 0..N {
        sum += (n as f64 + q).powf(-s);
    }

    let a = N as f64 + q; // (N + q)
    let a_pow = a.powf(-s); // (N+q)^{-s}
    sum += a * a_pow / (s - 1.0); // integral term (N+q)^{1-s}/(s-1)
    sum += 0.5 * a_pow; // half-term

    // Bernoulli corrections: Σ_j c_j · (s)_{2j-1} · (N+q)^{-s-2j+1}
    let a_inv2 = 1.0 / (a * a);
    let mut poch = s; // rising factorial (s)_1
    let mut a_factor = a_pow / a; // (N+q)^{-s-1}
    for (idx, &c) in C.iter().enumerate() {
        sum += c * poch * a_factor;
        let j = (idx + 1) as f64;
        poch *= (s + 2.0 * j - 1.0) * (s + 2.0 * j);
        a_factor *= a_inv2;
    }

    sum
}

pub(crate) fn eval_zeta(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    eval_binary_elementwise(primitive, inputs, |_, _| 0, hurwitz_zeta_approx)
}

pub(crate) fn bessel_i0e_approx(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        let i0 = 1.0
            + t * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        i0 * (-ax).exp()
    } else {
        let t = 3.75 / ax;
        (0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
            / ax.sqrt()
    }
}

pub(crate) fn bessel_i1e_approx(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        let i1 = ax
            * (0.5
                + t * (0.87890594
                    + t * (0.51498869
                        + t * (0.15084934
                            + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        let result = i1 * (-ax).exp();
        if x < 0.0 { -result } else { result }
    } else {
        let t = 3.75 / ax;
        let i1e = (0.39894228
            + t * (-0.03988024
                + t * (-0.00362018
                    + t * (0.00163801
                        + t * (-0.01031555
                            + t * (0.02282967
                                + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))))
            / ax.sqrt();
        if x < 0.0 { -i1e } else { i1e }
    }
}

pub(crate) fn eval_bessel_i0e(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise(primitive, inputs, bessel_i0e_approx)
}

pub(crate) fn eval_bessel_i1e(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise(primitive, inputs, bessel_i1e_approx)
}

fn dot_result_is_integral(lhs: &TensorValue, rhs: &TensorValue) -> bool {
    lhs.elements.iter().all(|literal| literal.is_integral())
        && rhs.elements.iter().all(|literal| literal.is_integral())
}

#[derive(Clone, Copy)]
enum DotOutputKind {
    Integral(DType),
    /// Real dot product; payload is the JAX-promoted real dtype to emit
    /// (BF16 / F16 / F32 / F64) so F32×F32 doesn't silently widen to F64
    /// (parity with `lax.dot_general` real-input promotion).
    Real(DType),
    Complex(DType),
}

impl DotOutputKind {
    fn tensor_dtype(self) -> DType {
        match self {
            Self::Integral(dtype) => dtype,
            Self::Real(dtype) => dtype,
            Self::Complex(dtype) => dtype,
        }
    }
}

fn dot_output_kind(lhs: &TensorValue, rhs: &TensorValue) -> DotOutputKind {
    if matches!(lhs.dtype, DType::Complex64 | DType::Complex128)
        || matches!(rhs.dtype, DType::Complex64 | DType::Complex128)
    {
        DotOutputKind::Complex(complex_binary_output_dtype(lhs.dtype, rhs.dtype))
    } else {
        // Honour JAX promotion: F32×F32 → F32, BF16×BF16 → BF16,
        // F16×F16 → F16, half + F32 → F32, anything with F64 → F64.
        let promoted = promote_dtype(lhs.dtype, rhs.dtype);
        if dot_result_is_integral(lhs, rhs)
            && matches!(promoted, DType::I32 | DType::I64 | DType::U32 | DType::U64)
        {
            DotOutputKind::Integral(promoted)
        } else {
            let real_dtype = match promoted {
                DType::BF16 | DType::F16 | DType::F32 | DType::F64 => promoted,
                _ => DType::F64,
            };
            DotOutputKind::Real(real_dtype)
        }
    }
}

fn integral_dot_literal_from_i64(dtype: DType, value: i64) -> Result<Literal, EvalError> {
    match dtype {
        DType::I32 | DType::I64 => Ok(Literal::I64(value)),
        _ => Err(EvalError::Unsupported {
            primitive: Primitive::Dot,
            detail: format!("unsupported signed dot output dtype {dtype:?}"),
        }),
    }
}

fn integral_dot_literal_from_u64(dtype: DType, value: u64) -> Result<Literal, EvalError> {
    match dtype {
        DType::U32 => Ok(Literal::U32(value as u32)),
        DType::U64 => Ok(Literal::U64(value)),
        _ => Err(EvalError::Unsupported {
            primitive: Primitive::Dot,
            detail: format!("unsupported unsigned dot output dtype {dtype:?}"),
        }),
    }
}

fn dot_accumulate(
    primitive: Primitive,
    output_kind: DotOutputKind,
    len: usize,
    mut pair_at: impl FnMut(usize) -> (Literal, Literal),
) -> Result<Literal, EvalError> {
    match output_kind {
        DotOutputKind::Integral(dtype) => {
            if matches!(dtype, DType::U32 | DType::U64) {
                let mut sum = 0_u64;
                for index in 0..len {
                    let (left, right) = pair_at(index);
                    let left_u = left.as_u64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "unsigned dot expected unsigned/integral lhs elements",
                    })?;
                    let right_u = right.as_u64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "unsigned dot expected unsigned/integral rhs elements",
                    })?;
                    sum = sum.wrapping_add(left_u.wrapping_mul(right_u));
                }
                integral_dot_literal_from_u64(dtype, sum)
            } else {
                let mut sum = 0_i64;
                for index in 0..len {
                    let (left, right) = pair_at(index);
                    let left_i = left.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "signed dot expected signed/integral lhs elements",
                    })?;
                    let right_i = right.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "signed dot expected signed/integral rhs elements",
                    })?;
                    sum = sum.wrapping_add(left_i.wrapping_mul(right_i));
                }
                integral_dot_literal_from_i64(dtype, sum)
            }
        }
        DotOutputKind::Real(dtype) => {
            let mut sum = 0.0_f64;
            for index in 0..len {
                let (left, right) = pair_at(index);
                let left_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric lhs tensor",
                })?;
                let right_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric rhs tensor",
                })?;
                sum += left_f * right_f;
            }
            Ok(real_literal_from_f64(dtype, sum))
        }
        DotOutputKind::Complex(dtype) => {
            let mut sum = (0.0_f64, 0.0_f64);
            for index in 0..len {
                let (left, right) = pair_at(index);
                let product = complex_mul(
                    literal_to_complex_parts(primitive, left)?,
                    literal_to_complex_parts(primitive, right)?,
                );
                sum.0 += product.0;
                sum.1 += product.1;
            }
            Ok(complex_literal_from_f64_parts(dtype, sum.0, sum.1))
        }
    }
}

fn dot_accumulate_contiguous(
    primitive: Primitive,
    output_kind: DotOutputKind,
    lhs: &[Literal],
    rhs: &[Literal],
) -> Result<Literal, EvalError> {
    match output_kind {
        DotOutputKind::Integral(dtype) => {
            if matches!(dtype, DType::U32 | DType::U64) {
                let mut sum = 0_u64;
                for (left, right) in lhs.iter().zip(rhs) {
                    let left_u = left.as_u64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "unsigned dot expected unsigned/integral lhs elements",
                    })?;
                    let right_u = right.as_u64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "unsigned dot expected unsigned/integral rhs elements",
                    })?;
                    sum = sum.wrapping_add(left_u.wrapping_mul(right_u));
                }
                integral_dot_literal_from_u64(dtype, sum)
            } else {
                let mut sum = 0_i64;
                for (left, right) in lhs.iter().zip(rhs) {
                    let left_i = left.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "signed dot expected signed/integral lhs elements",
                    })?;
                    let right_i = right.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "signed dot expected signed/integral rhs elements",
                    })?;
                    sum = sum.wrapping_add(left_i.wrapping_mul(right_i));
                }
                integral_dot_literal_from_i64(dtype, sum)
            }
        }
        DotOutputKind::Real(dtype) => {
            let mut sum = 0.0_f64;
            for (left, right) in lhs.iter().zip(rhs) {
                let left_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric lhs tensor",
                })?;
                let right_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric rhs tensor",
                })?;
                sum += left_f * right_f;
            }
            Ok(real_literal_from_f64(dtype, sum))
        }
        DotOutputKind::Complex(dtype) => {
            let mut sum = (0.0_f64, 0.0_f64);
            for (left, right) in lhs.iter().zip(rhs) {
                let product = complex_mul(
                    literal_to_complex_parts(primitive, *left)?,
                    literal_to_complex_parts(primitive, *right)?,
                );
                sum.0 += product.0;
                sum.1 += product.1;
            }
            Ok(complex_literal_from_f64_parts(dtype, sum.0, sum.1))
        }
    }
}

fn element_count_from_dims(dims: &[u32]) -> usize {
    dims.iter()
        .fold(1_usize, |count, &dim| count * dim as usize)
}

fn dot_output_value(
    dtype: DType,
    shape_dims: Vec<u32>,
    elements: Vec<Literal>,
) -> Result<Value, EvalError> {
    if shape_dims.is_empty() {
        Ok(Value::Scalar(elements[0]))
    } else {
        Ok(Value::Tensor(TensorValue::new(
            dtype,
            Shape { dims: shape_dims },
            elements,
        )?))
    }
}

fn rank2_f64_matmul(
    lhs: &TensorValue,
    rhs: &TensorValue,
    output_dims: &[u32],
) -> Result<Option<Value>, EvalError> {
    if lhs.dtype != DType::F64 || rhs.dtype != DType::F64 || lhs.rank() != 2 || rhs.rank() != 2 {
        return Ok(None);
    }

    let m = lhs.shape.dims[0] as usize;
    let k = lhs.shape.dims[1] as usize;
    let n = rhs.shape.dims[1] as usize;
    let (Some(lhs_values), Some(rhs_values)) = (dot_f64_elements(lhs), dot_f64_elements(rhs))
    else {
        return Ok(None);
    };
    let elements = matmul_2d(&lhs_values, m, k, &rhs_values, n)
        .into_iter()
        .map(Literal::from_f64)
        .collect();

    dot_output_value(DType::F64, output_dims.to_vec(), elements).map(Some)
}

fn dot_f64_elements(tensor: &TensorValue) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(tensor.elements.len());
    for &literal in &tensor.elements {
        let Literal::F64Bits(bits) = literal else {
            return None;
        };
        values.push(f64::from_bits(bits));
    }
    Some(values)
}

fn eval_tensor_dot(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<Value, EvalError> {
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    if lhs_rank == 0 || rhs_rank == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "tensor dot requires rank >= 1 tensors".to_owned(),
        });
    }

    let output_kind = dot_output_kind(lhs, rhs);
    let dtype = output_kind.tensor_dtype();
    let lhs_inner = lhs.shape.dims[lhs_rank - 1] as usize;

    if rhs_rank == 1 {
        let rhs_inner = rhs.shape.dims[0] as usize;
        if lhs_inner != rhs_inner {
            return Err(EvalError::ShapeMismatch {
                primitive,
                left: lhs.shape.clone(),
                right: rhs.shape.clone(),
            });
        }

        let output_dims = lhs.shape.dims[..lhs_rank - 1].to_vec();
        let output_count = element_count_from_dims(&output_dims);
        let mut elements = Vec::with_capacity(output_count);
        for lhs_prefix in 0..output_count {
            let lhs_base = lhs_prefix * lhs_inner;
            elements.push(dot_accumulate_contiguous(
                primitive,
                output_kind,
                &lhs.elements[lhs_base..lhs_base + lhs_inner],
                &rhs.elements[..rhs_inner],
            )?);
        }

        return dot_output_value(dtype, output_dims, elements);
    }

    let rhs_inner = rhs.shape.dims[rhs_rank - 2] as usize;
    if lhs_inner != rhs_inner {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: lhs.shape.clone(),
            right: rhs.shape.clone(),
        });
    }

    let columns = rhs.shape.dims[rhs_rank - 1] as usize;
    let lhs_prefix_dims = &lhs.shape.dims[..lhs_rank - 1];
    let rhs_prefix_dims = &rhs.shape.dims[..rhs_rank - 2];
    let lhs_prefix_count = element_count_from_dims(lhs_prefix_dims);
    let rhs_prefix_count = element_count_from_dims(rhs_prefix_dims);

    let mut output_dims = Vec::with_capacity(lhs_prefix_dims.len() + rhs_prefix_dims.len() + 1);
    output_dims.extend_from_slice(lhs_prefix_dims);
    output_dims.extend_from_slice(rhs_prefix_dims);
    output_dims.push(columns as u32);

    if let Some(value) = rank2_f64_matmul(lhs, rhs, &output_dims)? {
        return Ok(value);
    }

    let mut elements = Vec::with_capacity(lhs_prefix_count * rhs_prefix_count * columns);
    for lhs_prefix in 0..lhs_prefix_count {
        let lhs_base = lhs_prefix * lhs_inner;
        for rhs_prefix in 0..rhs_prefix_count {
            let rhs_base = rhs_prefix * rhs_inner * columns;
            for column in 0..columns {
                elements.push(dot_accumulate(
                    primitive,
                    output_kind,
                    lhs_inner,
                    |index| {
                        (
                            lhs.elements[lhs_base + index],
                            rhs.elements[rhs_base + index * columns + column],
                        )
                    },
                )?);
            }
        }
    }

    dot_output_value(dtype, output_dims, elements)
}

/// Dot product: scalar multiply plus tensor dot contraction.
pub(crate) fn eval_dot(inputs: &[Value]) -> Result<Value, EvalError> {
    let primitive = Primitive::Dot;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) if lhs.is_complex() || rhs.is_complex() => {
            let output_kind = DotOutputKind::Complex(complex_binary_output_dtype(
                literal_dtype(*lhs),
                literal_dtype(*rhs),
            ));
            Ok(Value::Scalar(dot_accumulate(
                primitive,
                output_kind,
                1,
                |_| (*lhs, *rhs),
            )?))
        }
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs,
            *rhs,
            primitive,
            &|a, b| a * b,
            &|a, b| a * b,
        )?)),
        (Value::Scalar(_), Value::Tensor(_)) | (Value::Tensor(_), Value::Scalar(_)) => {
            eval_binary_elementwise(Primitive::Mul, inputs, |a, b| a * b, |a, b| a * b)
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => eval_tensor_dot(primitive, lhs, rhs),
    }
}

fn parse_dim_list(s: &str) -> Vec<usize> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    s.split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect()
}

pub(crate) fn eval_dot_general(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::DotGeneral;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let lhs_contracting = parse_dim_list(params.get("lhs_contracting_dims").map_or("", |s| s));
    let rhs_contracting = parse_dim_list(params.get("rhs_contracting_dims").map_or("", |s| s));
    let lhs_batch = parse_dim_list(params.get("lhs_batch_dims").map_or("", |s| s));
    let rhs_batch = parse_dim_list(params.get("rhs_batch_dims").map_or("", |s| s));

    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "lhs and rhs must have same number of contracting dims".into(),
        });
    }
    if lhs_batch.len() != rhs_batch.len() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "lhs and rhs must have same number of batch dims".into(),
        });
    }

    let (lhs, rhs) = match (&inputs[0], &inputs[1]) {
        (Value::Tensor(l), Value::Tensor(r)) => (l, r),
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "dot_general requires tensor inputs".into(),
            });
        }
    };

    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();

    for &d in &lhs_contracting {
        if d >= lhs_rank {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("lhs contracting dim {d} out of range for rank {lhs_rank}"),
            });
        }
    }
    for &d in &rhs_contracting {
        if d >= rhs_rank {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("rhs contracting dim {d} out of range for rank {rhs_rank}"),
            });
        }
    }
    for &d in &lhs_batch {
        if d >= lhs_rank {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("lhs batch dim {d} out of range for rank {lhs_rank}"),
            });
        }
    }
    for &d in &rhs_batch {
        if d >= rhs_rank {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("rhs batch dim {d} out of range for rank {rhs_rank}"),
            });
        }
    }

    for (i, (&ld, &rd)) in lhs_contracting
        .iter()
        .zip(rhs_contracting.iter())
        .enumerate()
    {
        if lhs.shape.dims[ld] != rhs.shape.dims[rd] {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "contracting dim size mismatch at pair {i}: lhs[{ld}]={} != rhs[{rd}]={}",
                    lhs.shape.dims[ld], rhs.shape.dims[rd]
                ),
            });
        }
    }
    for (i, (&ld, &rd)) in lhs_batch.iter().zip(rhs_batch.iter()).enumerate() {
        if lhs.shape.dims[ld] != rhs.shape.dims[rd] {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "batch dim size mismatch at pair {i}: lhs[{ld}]={} != rhs[{rd}]={}",
                    lhs.shape.dims[ld], rhs.shape.dims[rd]
                ),
            });
        }
    }

    let mut lhs_free_dims = Vec::new();
    for d in 0..lhs_rank {
        if !lhs_contracting.contains(&d) && !lhs_batch.contains(&d) {
            lhs_free_dims.push(d);
        }
    }
    let mut rhs_free_dims = Vec::new();
    for d in 0..rhs_rank {
        if !rhs_contracting.contains(&d) && !rhs_batch.contains(&d) {
            rhs_free_dims.push(d);
        }
    }

    let mut output_dims = Vec::new();
    for &d in &lhs_batch {
        output_dims.push(lhs.shape.dims[d]);
    }
    for &d in &lhs_free_dims {
        output_dims.push(lhs.shape.dims[d]);
    }
    for &d in &rhs_free_dims {
        output_dims.push(rhs.shape.dims[d]);
    }

    let contracting_size: usize = lhs_contracting
        .iter()
        .map(|&d| lhs.shape.dims[d] as usize)
        .product();
    if contracting_size == 0 && !lhs_contracting.is_empty() {
        let output_count = output_dims.iter().map(|&d| d as usize).product::<usize>();
        let output_kind = dot_output_kind(lhs, rhs);
        let dtype = output_kind.tensor_dtype();
        let zero = match output_kind {
            DotOutputKind::Real(dt) => real_literal_from_f64(dt, 0.0),
            DotOutputKind::Complex(dt) => match dt {
                DType::Complex64 => Literal::from_complex64(0.0, 0.0),
                _ => Literal::from_complex128(0.0, 0.0),
            },
            DotOutputKind::Integral(dt) => match dt {
                DType::U32 => Literal::U32(0),
                DType::U64 => Literal::U64(0),
                _ => Literal::I64(0),
            },
        };
        return dot_output_value(dtype, output_dims, vec![zero; output_count]);
    }

    let batch_size: usize = lhs_batch
        .iter()
        .map(|&d| lhs.shape.dims[d] as usize)
        .product();
    let lhs_free_size: usize = lhs_free_dims
        .iter()
        .map(|&d| lhs.shape.dims[d] as usize)
        .product();
    let rhs_free_size: usize = rhs_free_dims
        .iter()
        .map(|&d| rhs.shape.dims[d] as usize)
        .product();

    let output_kind = dot_output_kind(lhs, rhs);
    let dtype = output_kind.tensor_dtype();
    let output_count = batch_size.max(1) * lhs_free_size.max(1) * rhs_free_size.max(1);

    let standard_rank2_matmul = lhs_rank == 2
        && rhs_rank == 2
        && lhs_batch.is_empty()
        && rhs_batch.is_empty()
        && lhs_contracting.as_slice() == [1usize]
        && rhs_contracting.as_slice() == [0usize]
        && lhs_free_dims.as_slice() == [0usize]
        && rhs_free_dims.as_slice() == [1usize];
    if standard_rank2_matmul && let Some(value) = rank2_f64_matmul(lhs, rhs, &output_dims)? {
        return Ok(value);
    }

    let mut elements = Vec::with_capacity(output_count);

    let lhs_strides = compute_strides(&lhs.shape.dims);
    let rhs_strides = compute_strides(&rhs.shape.dims);

    let batch_ranges: Vec<u32> = lhs_batch.iter().map(|&d| lhs.shape.dims[d]).collect();
    let lhs_free_ranges: Vec<u32> = lhs_free_dims.iter().map(|&d| lhs.shape.dims[d]).collect();
    let rhs_free_ranges: Vec<u32> = rhs_free_dims.iter().map(|&d| rhs.shape.dims[d]).collect();
    let contract_ranges: Vec<u32> = lhs_contracting.iter().map(|&d| lhs.shape.dims[d]).collect();

    for batch_idx in MultiIndexIterator::new(&batch_ranges) {
        for lhs_free_idx in MultiIndexIterator::new(&lhs_free_ranges) {
            for rhs_free_idx in MultiIndexIterator::new(&rhs_free_ranges) {
                let acc = dot_accumulate(primitive, output_kind, contracting_size.max(1), |k| {
                    let contract_idx = linear_to_multi_index(k, &contract_ranges);

                    let mut lhs_index = 0usize;
                    for (i, &d) in lhs_batch.iter().enumerate() {
                        lhs_index += batch_idx.get(i).copied().unwrap_or(0) * lhs_strides[d];
                    }
                    for (i, &d) in lhs_free_dims.iter().enumerate() {
                        lhs_index += lhs_free_idx.get(i).copied().unwrap_or(0) * lhs_strides[d];
                    }
                    for (i, &d) in lhs_contracting.iter().enumerate() {
                        lhs_index += contract_idx.get(i).copied().unwrap_or(0) * lhs_strides[d];
                    }

                    let mut rhs_index = 0usize;
                    for (i, &d) in rhs_batch.iter().enumerate() {
                        rhs_index += batch_idx.get(i).copied().unwrap_or(0) * rhs_strides[d];
                    }
                    for (i, &d) in rhs_free_dims.iter().enumerate() {
                        rhs_index += rhs_free_idx.get(i).copied().unwrap_or(0) * rhs_strides[d];
                    }
                    for (i, &d) in rhs_contracting.iter().enumerate() {
                        rhs_index += contract_idx.get(i).copied().unwrap_or(0) * rhs_strides[d];
                    }

                    (lhs.elements[lhs_index], rhs.elements[rhs_index])
                })?;
                elements.push(acc);
            }
        }
    }

    dot_output_value(dtype, output_dims, elements)
}

fn linear_to_multi_index(mut linear: usize, dims: &[u32]) -> Vec<usize> {
    let mut result = vec![0usize; dims.len()];
    for i in (0..dims.len()).rev() {
        let size = dims[i] as usize;
        if size > 0 {
            result[i] = linear % size;
            linear /= size;
        }
    }
    result
}

struct MultiIndexIterator {
    dims: Vec<u32>,
    current: Vec<usize>,
    done: bool,
}

impl MultiIndexIterator {
    fn new(dims: &[u32]) -> Self {
        let done = dims.contains(&0);
        Self {
            dims: dims.to_vec(),
            current: vec![0; dims.len()],
            done: done || dims.is_empty(),
        }
    }
}

impl Iterator for MultiIndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            if self.dims.is_empty() && self.current.is_empty() {
                self.done = true;
                self.current.push(0);
                return Some(Vec::new());
            }
            return None;
        }

        let result = self.current.clone();

        for i in (0..self.dims.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.dims[i] as usize {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

/// IsFinite: returns Bool indicating whether each element is finite (not NaN or Inf).
pub(crate) fn eval_is_finite(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let is_finite = match *literal {
                Literal::Complex64Bits(re, im) => {
                    f32::from_bits(re).is_finite() && f32::from_bits(im).is_finite()
                }
                Literal::Complex128Bits(re, im) => {
                    f64::from_bits(re).is_finite() && f64::from_bits(im).is_finite()
                }
                _ => {
                    let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar",
                    })?;
                    value.is_finite()
                }
            };
            Ok(Value::Scalar(Literal::Bool(is_finite)))
        }
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let is_finite = match *literal {
                    Literal::Complex64Bits(re, im) => Ok(Literal::Bool(
                        f32::from_bits(re).is_finite() && f32::from_bits(im).is_finite(),
                    )),
                    Literal::Complex128Bits(re, im) => Ok(Literal::Bool(
                        f64::from_bits(re).is_finite() && f64::from_bits(im).is_finite(),
                    )),
                    _ => literal
                        .as_f64()
                        .map(|v| Literal::Bool(v.is_finite()))
                        .ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        }),
                }?;
                elements.push(is_finite);
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// IsNan: returns Bool indicating whether each element is NaN.
pub(crate) fn eval_is_nan(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let is_nan = match *literal {
                Literal::Complex64Bits(re, im) => {
                    f32::from_bits(re).is_nan() || f32::from_bits(im).is_nan()
                }
                Literal::Complex128Bits(re, im) => {
                    f64::from_bits(re).is_nan() || f64::from_bits(im).is_nan()
                }
                _ => {
                    let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar",
                    })?;
                    value.is_nan()
                }
            };
            Ok(Value::Scalar(Literal::Bool(is_nan)))
        }
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let is_nan = match *literal {
                    Literal::Complex64Bits(re, im) => Ok(Literal::Bool(
                        f32::from_bits(re).is_nan() || f32::from_bits(im).is_nan(),
                    )),
                    Literal::Complex128Bits(re, im) => Ok(Literal::Bool(
                        f64::from_bits(re).is_nan() || f64::from_bits(im).is_nan(),
                    )),
                    _ => literal.as_f64().map(|v| Literal::Bool(v.is_nan())).ok_or(
                        EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        },
                    ),
                }?;
                elements.push(is_nan);
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// IsInf: returns Bool indicating whether each element is infinite.
pub(crate) fn eval_is_inf(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let is_inf = match *literal {
                Literal::Complex64Bits(re, im) => {
                    f32::from_bits(re).is_infinite() || f32::from_bits(im).is_infinite()
                }
                Literal::Complex128Bits(re, im) => {
                    f64::from_bits(re).is_infinite() || f64::from_bits(im).is_infinite()
                }
                _ => {
                    let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar",
                    })?;
                    value.is_infinite()
                }
            };
            Ok(Value::Scalar(Literal::Bool(is_inf)))
        }
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let is_inf = match *literal {
                    Literal::Complex64Bits(re, im) => Ok(Literal::Bool(
                        f32::from_bits(re).is_infinite() || f32::from_bits(im).is_infinite(),
                    )),
                    Literal::Complex128Bits(re, im) => Ok(Literal::Bool(
                        f64::from_bits(re).is_infinite() || f64::from_bits(im).is_infinite(),
                    )),
                    _ => literal
                        .as_f64()
                        .map(|v| Literal::Bool(v.is_infinite()))
                        .ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        }),
                }?;
                elements.push(is_inf);
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Signbit: returns Bool indicating whether the sign bit is set (true for negative, including -0.0).
pub(crate) fn eval_signbit(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let signbit = match *literal {
                Literal::I64(v) => v < 0,
                Literal::F64Bits(b) => f64::from_bits(b).is_sign_negative(),
                Literal::F32Bits(b) => f32::from_bits(b).is_sign_negative(),
                _ => {
                    let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar",
                    })?;
                    value.is_sign_negative()
                }
            };
            Ok(Value::Scalar(Literal::Bool(signbit)))
        }
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let signbit = match *literal {
                    Literal::I64(v) => Literal::Bool(v < 0),
                    Literal::F64Bits(b) => Literal::Bool(f64::from_bits(b).is_sign_negative()),
                    Literal::F32Bits(b) => Literal::Bool(f32::from_bits(b).is_sign_negative()),
                    _ => literal
                        .as_f64()
                        .map(|v| Literal::Bool(v.is_sign_negative()))
                        .ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        })?,
                };
                elements.push(signbit);
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// IntegerPow: x.powi(n) where n is an integer exponent from params.
pub(crate) fn eval_integer_pow(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let exponent: i32 = params
        .get("exponent")
        .and_then(|s| s.trim().parse().ok())
        .ok_or(EvalError::Unsupported {
            primitive,
            detail: "integer_pow requires 'exponent' param".to_owned(),
        })?;

    fn complex_powi(z: (f64, f64), n: i32) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        }
        if n < 0 {
            let inv = complex_powi(z, -n);
            return complex_div((1.0, 0.0), inv);
        }
        let mut result = (1.0, 0.0);
        let mut base = z;
        let mut exp = n as u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = complex_mul(result, base);
            }
            base = complex_mul(base, base);
            exp >>= 1;
        }
        result
    }

    fn integer_pow_literal(literal: Literal, exponent: i32) -> Result<Literal, &'static str> {
        match literal {
            Literal::Complex64Bits(re_bits, im_bits) => {
                let z = (
                    f32::from_bits(re_bits) as f64,
                    f32::from_bits(im_bits) as f64,
                );
                let result = complex_powi(z, exponent);
                Ok(Literal::from_complex64(result.0 as f32, result.1 as f32))
            }
            Literal::Complex128Bits(re_bits, im_bits) => {
                let z = (f64::from_bits(re_bits), f64::from_bits(im_bits));
                let result = complex_powi(z, exponent);
                Ok(Literal::from_complex128(result.0, result.1))
            }
            _ => {
                let value = literal.as_f64().ok_or("expected numeric")?;
                let in_dtype = literal_dtype(literal);
                Ok(real_literal_from_f64(in_dtype, value.powi(exponent)))
            }
        }
    }

    match &inputs[0] {
        Value::Scalar(literal) => integer_pow_literal(*literal, exponent)
            .map(Value::Scalar)
            .map_err(|e| EvalError::TypeMismatch {
                primitive,
                detail: e,
            }),
        Value::Tensor(tensor) => {
            let out_dtype = tensor.dtype;
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                elements.push(integer_pow_literal(*literal, exponent).map_err(|e| {
                    EvalError::TypeMismatch {
                        primitive,
                        detail: e,
                    }
                })?);
            }

            Ok(Value::Tensor(TensorValue::new(
                out_dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Nextafter: IEEE 754 next representable float value from x towards y.
pub(crate) fn eval_nextafter(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    fn next_after_f64(x: f64, y: f64) -> f64 {
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        if x == y {
            return y;
        }
        if x == 0.0 {
            // Smallest subnormal towards y's sign
            if y > 0.0 {
                return f64::from_bits(1);
            }
            return f64::from_bits(1 | (1_u64 << 63));
        }
        let bits = x.to_bits();
        let result_bits = if (x < y) == (x > 0.0) {
            bits + 1
        } else {
            bits - 1
        };
        f64::from_bits(result_bits)
    }

    fn next_after_f32(x: f32, y: f32) -> f32 {
        if x.is_nan() || y.is_nan() {
            return f32::NAN;
        }
        if x == y {
            return y;
        }
        if x == 0.0 {
            if y > 0.0 {
                return f32::from_bits(1);
            }
            return f32::from_bits(1 | (1_u32 << 31));
        }
        let bits = x.to_bits();
        let result_bits = if (x < y) == (x > 0.0) {
            bits + 1
        } else {
            bits - 1
        };
        f32::from_bits(result_bits)
    }

    fn next_after_literal(
        primitive: Primitive,
        lhs: Literal,
        rhs: Literal,
    ) -> Result<Literal, EvalError> {
        match (lhs, rhs) {
            (Literal::F32Bits(left), Literal::F32Bits(right)) => Ok(Literal::from_f32(
                next_after_f32(f32::from_bits(left), f32::from_bits(right)),
            )),
            (Literal::F64Bits(left), Literal::F64Bits(right)) => Ok(Literal::from_f64(
                next_after_f64(f64::from_bits(left), f64::from_bits(right)),
            )),
            (left, right) => {
                let x = left.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric nextafter lhs",
                })?;
                let y = right.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric nextafter rhs",
                })?;
                Ok(Literal::from_f64(next_after_f64(x, y)))
            }
        }
    }

    #[inline]
    fn next_after_same_shape_f64_tensor(
        lhs: &TensorValue,
        rhs: &TensorValue,
    ) -> Result<Option<Value>, EvalError> {
        let mut elements = Vec::with_capacity(lhs.elements.len());
        for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
            let (Literal::F64Bits(left), Literal::F64Bits(right)) = (*left, *right) else {
                return Ok(None);
            };
            elements.push(Literal::from_f64(next_after_f64(
                f64::from_bits(left),
                f64::from_bits(right),
            )));
        }

        Ok(Some(Value::Tensor(TensorValue::new(
            DType::F64,
            lhs.shape.clone(),
            elements,
        )?)))
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            Ok(Value::Scalar(next_after_literal(primitive, *lhs, *rhs)?))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.dtype == DType::F64
                && rhs.dtype == DType::F64
                && lhs.shape == rhs.shape
                && let Some(value) = next_after_same_shape_f64_tensor(lhs, rhs)?
            {
                return Ok(value);
            }

            let out_shape =
                broadcast_shape(&lhs.shape, &rhs.shape).ok_or(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                })?;

            let total_elements: usize = out_shape.dims.iter().map(|&d| d as usize).product();
            let mut elements = Vec::with_capacity(total_elements);

            let out_strides = compute_strides(&out_shape.dims);
            let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
            let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

            let mut multi = Vec::with_capacity(out_shape.rank());
            for flat in 0..total_elements {
                flat_to_multi_into(flat, &out_strides, &mut multi);
                let lhs_idx = broadcast_flat_index(&multi, &lhs_strides);
                let rhs_idx = broadcast_flat_index(&multi, &rhs_strides);
                elements.push(next_after_literal(
                    primitive,
                    lhs.elements[lhs_idx],
                    rhs.elements[rhs_idx],
                )?);
            }

            let dtype = match (lhs.dtype, rhs.dtype) {
                (DType::F32, DType::F32) => DType::F32,
                (DType::F64, DType::F64) => DType::F64,
                _ => DType::F64,
            };
            Ok(Value::Tensor(TensorValue::new(dtype, out_shape, elements)?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let mut elements = Vec::with_capacity(rhs.elements.len());
            for r in rhs.elements.iter() {
                elements.push(next_after_literal(primitive, *lhs, *r)?);
            }
            let dtype = match rhs.dtype {
                DType::F32 => DType::F32,
                _ => DType::F64,
            };
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let mut elements = Vec::with_capacity(lhs.elements.len());
            for l in lhs.elements.iter() {
                elements.push(next_after_literal(primitive, *l, *rhs)?);
            }
            let dtype = match lhs.dtype {
                DType::F32 => DType::F32,
                _ => DType::F64,
            };
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_test_utils::fixture_id_from_json;
    use std::f64::consts::PI;

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn s_f32(v: f32) -> Value {
        Value::Scalar(Literal::from_f32(v))
    }
    fn s_i64(v: i64) -> Value {
        Value::Scalar(Literal::I64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn v_f32(data: &[f32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f32(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn matrix_f64(rows: u32, columns: u32, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows, columns],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn matrix_i64(rows: u32, columns: u32, data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![rows, columns],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn tensor_f64(dims: Vec<u32>, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn tensor_i64(dims: Vec<u32>, data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn v_complex64(data: &[(f32, f32)]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter()
                    .map(|&(re, im)| Literal::from_complex64(re, im))
                    .collect(),
            )
            .unwrap(),
        )
    }
    fn v_complex128(data: &[(f64, f64)]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        )
    }
    fn matrix_complex128(rows: u32, columns: u32, data: &[(f64, f64)]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![rows, columns],
                },
                data.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        )
    }
    fn extract_f64(val: &Value) -> f64 {
        val.as_f64_scalar().unwrap()
    }
    fn extract_f64_vec(val: &Value) -> Vec<f64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    }
    fn extract_f64_bits_vec(val: &Value) -> Vec<u64> {
        let tensor = val.as_tensor().unwrap();
        let bits = tensor
            .elements
            .iter()
            .filter_map(|literal| match *literal {
                Literal::F64Bits(bits) => Some(bits),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(bits.len(), tensor.elements.len());
        bits
    }
    fn reference_matmul_bits(lhs: &[f64], m: usize, k: usize, rhs: &[f64], n: usize) -> Vec<u64> {
        let mut bits = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for l in 0..k {
                    sum += lhs[i * k + l] * rhs[l * n + j];
                }
                bits.push(sum.to_bits());
            }
        }
        bits
    }
    fn extract_i64_vec(val: &Value) -> Vec<i64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect()
    }
    fn extract_complex_vec(val: &Value) -> Vec<(f64, f64)> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|literal| {
                literal.as_complex128().unwrap_or_else(|| {
                    let (re, im) = literal.as_complex64().unwrap();
                    (f64::from(re), f64::from(im))
                })
            })
            .collect()
    }
    fn assert_complex_close(actual: (f64, f64), expected: (f64, f64)) {
        assert!(
            (actual.0 - expected.0).abs() < 1e-9,
            "real mismatch: actual={actual:?} expected={expected:?}"
        );
        assert!(
            (actual.1 - expected.1).abs() < 1e-9,
            "imag mismatch: actual={actual:?} expected={expected:?}"
        );
    }
    // ── Binary elementwise: scalar-scalar ──

    #[test]
    fn binary_add_scalars() {
        let result = eval_binary_elementwise(
            Primitive::Add,
            &[s_f64(2.0), s_f64(3.0)],
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn binary_mul_int_scalars() {
        let result = eval_binary_elementwise(
            Primitive::Mul,
            &[s_i64(7), s_i64(6)],
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::I64(_))),
            "expected i64 scalar"
        );
        let Value::Scalar(Literal::I64(v)) = result else {
            return;
        };
        assert_eq!(v, 42);
    }

    #[test]
    fn binary_arity_mismatch() {
        let result =
            eval_binary_elementwise(Primitive::Add, &[s_f64(1.0)], |a, b| a + b, |a, b| a + b);
        assert!(result.is_err());
    }

    // ── Binary elementwise: tensor-tensor same shape ──

    #[test]
    fn binary_add_tensors_same_shape() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[4.0, 5.0, 6.0]);
        let result =
            eval_binary_elementwise(Primitive::Add, &[a, b], |a, b| a + b, |a, b| a + b).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn dense_f64_pass44_same_shape_arithmetic_fast_path_bit_identical_to_scalar() {
        // Adversarial F64 inputs: signed zero, infinities, NaN, div-by-zero,
        // 0/0, and ordinary values. The same-shape F64 fast path must produce
        // bits identical to the per-element scalar op (`Literal::from_f64`).
        let lhs_data = [1.5, -0.0, f64::INFINITY, f64::NAN, 7.0, -3.25, 0.0];
        let rhs_data = [2.0, 3.0, -4.0, 5.0, 0.0, f64::NEG_INFINITY, 0.0];
        for primitive in [
            Primitive::Add,
            Primitive::Sub,
            Primitive::Mul,
            Primitive::Div,
        ] {
            let scalar = |x: f64, y: f64| match primitive {
                Primitive::Add => x + y,
                Primitive::Sub => x - y,
                Primitive::Mul => x * y,
                Primitive::Div => x / y,
                _ => unreachable!(),
            };
            let a = Value::vector_f64(&lhs_data).unwrap();
            let b = Value::vector_f64(&rhs_data).unwrap();
            let result = eval_binary_elementwise(primitive, &[a, b], |x, y| x + y, scalar).unwrap();
            let Value::Tensor(tensor) = result else {
                panic!("expected tensor for {primitive:?}");
            };
            assert_eq!(tensor.dtype, DType::F64);
            assert!(
                tensor.elements.as_f64_slice().is_some(),
                "{primitive:?} should keep dense F64 output"
            );
            let expected: Vec<Literal> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(&x, &y)| Literal::from_f64(scalar(x, y)))
                .collect();
            // Compare raw bits so NaN payloads and -0.0 are distinguished.
            assert_eq!(tensor.elements, expected, "{primitive:?} bit mismatch");
        }
    }

    #[test]
    fn dense_f64_pass44_same_shape_max_min_fast_path_bit_identical_to_scalar() {
        // Max/Min route through the same-shape F64 fast path using the crate's
        // NaN-propagating `jax_max_f64`/`jax_min_f64`. The result must be bit-
        // identical to the per-element scalar op, including NaN propagation
        // (where `f64::max`/`min` would wrongly drop the NaN) and signed zero.
        let lhs_data = [1.5, -0.0, f64::INFINITY, f64::NAN, 7.0, -3.25, 0.0, -2.0];
        let rhs_data = [2.0, 0.0, -4.0, 5.0, f64::NAN, f64::NEG_INFINITY, 0.0, -2.0];
        for (primitive, scalar) in [
            (Primitive::Max, crate::jax_max_f64 as fn(f64, f64) -> f64),
            (Primitive::Min, crate::jax_min_f64 as fn(f64, f64) -> f64),
        ] {
            let a = Value::vector_f64(&lhs_data).unwrap();
            let b = Value::vector_f64(&rhs_data).unwrap();
            // Pass the real dispatch ops so the generic fallback (if hit) would
            // match too; the fast path is what actually runs here.
            let int_op = |x: i64, y: i64| {
                if primitive == Primitive::Max {
                    x.max(y)
                } else {
                    x.min(y)
                }
            };
            let result = eval_binary_elementwise(primitive, &[a, b], int_op, scalar).unwrap();
            let Value::Tensor(tensor) = result else {
                panic!("expected tensor for {primitive:?}");
            };
            assert_eq!(tensor.dtype, DType::F64);
            assert!(
                tensor.elements.as_f64_slice().is_some(),
                "{primitive:?} should keep dense F64 output"
            );
            let expected: Vec<Literal> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(&x, &y)| Literal::from_f64(scalar(x, y)))
                .collect();
            assert_eq!(tensor.elements, expected, "{primitive:?} bit mismatch");
        }
    }

    #[test]
    fn dense_f64_pass44_declared_f64_malformed_tensor_still_falls_back() {
        let lhs = Value::Tensor(
            TensorValue::new(DType::F64, Shape::vector(1), vec![Literal::I64(2)]).unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(DType::F64, Shape::vector(1), vec![Literal::I64(5)]).unwrap(),
        );
        let result =
            eval_binary_elementwise(Primitive::Add, &[lhs, rhs], |a, b| a + b, |a, b| a + b)
                .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.elements, vec![Literal::I64(7)]);
        assert!(
            tensor.elements.as_f64_slice().is_none(),
            "malformed literal-backed tensor must not become dense"
        );
    }

    #[test]
    fn same_shape_i64_add_fast_path_matches_generic_edge_values() -> Result<(), String> {
        let lhs_data = [0, 1, -1, i64::MAX, i64::MIN, 1_234_567_890_123_456_789];
        let rhs_data = [0, -1, i64::MAX, 1, -1, -987_654_321_098_765_432];
        let lhs = matrix_i64(2, 3, &lhs_data);
        let rhs = matrix_i64(2, 3, &rhs_data);
        let int_op = |a: i64, b: i64| a.wrapping_add(b);
        let float_op = |a: f64, b: f64| a + b;
        let result = eval_binary_elementwise(Primitive::Add, &[lhs, rhs], int_op, float_op)
            .map_err(|err| format!("{err:?}"))?;
        let Value::Tensor(tensor) = result else {
            return Err("expected tensor".to_owned());
        };
        let expected = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&left, &right)| {
                binary_literal_op(
                    Literal::I64(left),
                    Literal::I64(right),
                    Primitive::Add,
                    &int_op,
                    &float_op,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("{err:?}"))?;
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(tensor.elements, expected);

        let malformed_lhs = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(1.5), Literal::I64(2)],
            )
            .map_err(|err| format!("{err:?}"))?,
        );
        let malformed_rhs = tensor_i64(vec![2], &[3, 4]);
        let result = eval_binary_elementwise(
            Primitive::Add,
            &[malformed_lhs, malformed_rhs],
            int_op,
            float_op,
        )
        .map_err(|err| format!("{err:?}"))?;
        let Value::Tensor(tensor) = result else {
            return Err("expected tensor".to_owned());
        };
        assert_eq!(
            tensor.elements,
            vec![Literal::from_f64(4.5), Literal::I64(6)]
        );
        Ok(())
    }

    #[test]
    fn unary_int_or_float_f64_fast_path_bit_identical() {
        // Square and Sign over an F64 tensor must match the per-element generic
        // arm bit-for-bit, including -0.0 (Sign preserves it), NaN and +-inf.
        let data = [
            -2.0,
            0.0,
            -0.0,
            3.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let params = BTreeMap::new();

        // Square: float_op = |x| x * x
        let result = crate::eval_primitive(Primitive::Square, &[v_f64(&data)], &params).unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::F64);
        let expected: Vec<Literal> = data.iter().map(|&x| Literal::from_f64(x * x)).collect();
        assert_eq!(tensor.elements, expected, "square");

        // Sign: NaN -> NaN, x==0.0 -> x (keeps -0.0), else x.signum()
        let sign = |x: f64| {
            if x.is_nan() {
                f64::NAN
            } else if x == 0.0 {
                x
            } else {
                x.signum()
            }
        };
        let result = crate::eval_primitive(Primitive::Sign, &[v_f64(&data)], &params).unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        let expected: Vec<Literal> = data.iter().map(|&x| Literal::from_f64(sign(x))).collect();
        assert_eq!(tensor.elements, expected, "sign");
    }

    #[test]
    fn same_shape_complex128_mul_fast_path_bit_identical() {
        let lhs = [
            (1.5, -2.0),
            (-0.0, 3.0),
            (f64::INFINITY, -4.0),
            (f64::from_bits(0x7ff8_0000_0000_1234), 5.0),
        ];
        // Note rhs[3] is finite: multiplying it against lhs[3]'s NaN payload
        // exercises deterministic single-NaN × finite propagation. A NaN × NaN
        // product (two distinct payloads) is intentionally avoided — IEEE-754
        // leaves which input payload survives implementation-defined, and LLVM
        // may commute the multiply, so its result bits are not a stable contract.
        let rhs = [
            (2.0, 0.5),
            (f64::NEG_INFINITY, -0.0),
            (-1.25, f64::INFINITY),
            (6.0, 7.0),
        ];
        // `black_box` the constant operands so the compiler cannot const-fold the
        // `expected` reference arithmetic. Under some nightlies LLVM folds the
        // literal `ar*br - ai*bi` to the compiler's +NaN while the runtime fast
        // path yields the CPU's -NaN, a spurious NaN-sign mismatch (the e7ej
        // unpinned-nightly fragility). Both paths are bit-identical at runtime.
        let lhs = std::hint::black_box(lhs);
        let rhs = std::hint::black_box(rhs);
        let params = BTreeMap::new();
        let result = crate::eval_primitive(
            Primitive::Mul,
            &[v_complex128(&lhs), v_complex128(&rhs)],
            &params,
        )
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::Complex128);
        let expected: Vec<Literal> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&(ar, ai), &(br, bi))| {
                Literal::from_complex128(ar * br - ai * bi, ar * bi + ai * br)
            })
            .collect();
        assert_eq!(tensor.elements, expected);
    }

    #[test]
    fn broadcast_binary_f64_fast_path_bit_identical() {
        // 2x3 broadcast against a row-vector [3] and a column-vector [2,1],
        // across add/sub/mul/div with NaN / -0.0 / +-inf. Each output element
        // must match lhs op rhs computed with the same broadcasting.
        fn t(dims: Vec<u32>, data: &[f64]) -> (Value, Vec<f64>) {
            let v = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims },
                    data.iter().map(|&x| Literal::from_f64(x)).collect(),
                )
                .unwrap(),
            );
            (v, data.to_vec())
        }
        let lhs_data = [1.5, -0.0, f64::INFINITY, f64::NAN, 7.0, -3.25]; // 2x3
        let row = [2.0, -4.0, 0.0]; // [3]
        let col = [5.0, f64::NEG_INFINITY]; // [2,1]
        for primitive in [
            Primitive::Add,
            Primitive::Sub,
            Primitive::Mul,
            Primitive::Div,
        ] {
            let op = |a: f64, b: f64| match primitive {
                Primitive::Add => a + b,
                Primitive::Sub => a - b,
                Primitive::Mul => a * b,
                Primitive::Div => a / b,
                _ => unreachable!(),
            };
            let int_op = |a: i64, b: i64| match primitive {
                Primitive::Add => a.wrapping_add(b),
                Primitive::Sub => a.wrapping_sub(b),
                Primitive::Mul => a.wrapping_mul(b),
                _ => a.checked_div(b).unwrap_or(0),
            };
            // row broadcast: out[r,c] = lhs[r*3+c] op row[c]
            let (lhs, _) = t(vec![2, 3], &lhs_data);
            let (rhs, _) = t(vec![3], &row);
            let result = eval_binary_elementwise(primitive, &[lhs, rhs], int_op, op).unwrap();
            let Value::Tensor(tensor) = result else {
                panic!("expected tensor");
            };
            let expected: Vec<Literal> = (0..6)
                .map(|i| Literal::from_f64(op(lhs_data[i], row[i % 3])))
                .collect();
            assert_eq!(tensor.elements, expected, "{primitive:?} row-broadcast");
            assert_eq!(tensor.shape.dims, vec![2, 3]);

            // column broadcast: out[r,c] = lhs[r*3+c] op col[r]
            let (lhs, _) = t(vec![2, 3], &lhs_data);
            let (rhs, _) = t(vec![2, 1], &col);
            let result = eval_binary_elementwise(primitive, &[lhs, rhs], int_op, op).unwrap();
            let Value::Tensor(tensor) = result else {
                panic!("expected tensor");
            };
            let expected: Vec<Literal> = (0..6)
                .map(|i| Literal::from_f64(op(lhs_data[i], col[i / 3])))
                .collect();
            assert_eq!(tensor.elements, expected, "{primitive:?} col-broadcast");
        }
    }

    #[test]
    fn f64_scalar_broadcast_fast_path_bit_identical_to_scalar() {
        // Both operand orders for the non-commutative ops, with adversarial
        // values, must match the per-element scalar op bit-for-bit.
        let tensor_data = [1.5, -0.0, f64::INFINITY, f64::NAN, 7.0, -3.25, 0.0];
        let scalar = 2.5_f64;
        for primitive in [
            Primitive::Add,
            Primitive::Sub,
            Primitive::Mul,
            Primitive::Div,
        ] {
            let op = |x: f64, y: f64| match primitive {
                Primitive::Add => x + y,
                Primitive::Sub => x - y,
                Primitive::Mul => x * y,
                Primitive::Div => x / y,
                _ => unreachable!(),
            };
            for scalar_on_left in [true, false] {
                let scalar_val = Value::Scalar(Literal::from_f64(scalar));
                let tensor_val = v_f64(&tensor_data);
                let inputs = if scalar_on_left {
                    [scalar_val, tensor_val]
                } else {
                    [tensor_val, scalar_val]
                };
                let result = eval_binary_elementwise(primitive, &inputs, |x, y| x + y, op).unwrap();
                let Value::Tensor(tensor) = result else {
                    panic!("expected tensor for {primitive:?}");
                };
                assert_eq!(tensor.dtype, DType::F64);
                let expected: Vec<Literal> = tensor_data
                    .iter()
                    .map(|&e| {
                        let out = if scalar_on_left {
                            op(scalar, e)
                        } else {
                            op(e, scalar)
                        };
                        Literal::from_f64(out)
                    })
                    .collect();
                assert_eq!(
                    tensor.elements, expected,
                    "{primitive:?} scalar_on_left={scalar_on_left} bit mismatch"
                );
            }
        }
    }

    #[test]
    fn binary_mul_f64_tensors_same_shape_preserves_bits() {
        let a = v_f64(&[1.5, -0.0, -2.0]);
        let b = v_f64(&[2.0, 3.0, -4.0]);
        let result =
            eval_binary_elementwise(Primitive::Mul, &[a, b], |a, b| a * b, |a, b| a * b).unwrap();
        assert!(matches!(&result, Value::Tensor(_)));
        let Value::Tensor(tensor) = result else {
            return;
        };
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(
            tensor.elements,
            vec![
                Literal::from_f64(3.0),
                Literal::from_f64(-0.0),
                Literal::from_f64(8.0)
            ]
        );
    }

    // ── Binary elementwise: scalar-tensor broadcasting ──

    #[test]
    fn binary_mul_scalar_tensor() {
        let a = s_f64(2.0);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result =
            eval_binary_elementwise(Primitive::Mul, &[a, b], |a, b| a * b, |a, b| a * b).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn binary_mul_tensor_scalar() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = s_f64(10.0);
        let result =
            eval_binary_elementwise(Primitive::Mul, &[a, b], |a, b| a * b, |a, b| a * b).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    // ── Unary elementwise ──

    #[test]
    fn unary_neg_scalar() {
        let result = eval_neg(Primitive::Neg, &[s_f64(3.0)]).unwrap();
        assert!((extract_f64(&result) + 3.0).abs() < 1e-12);
    }

    #[test]
    fn unary_neg_tensor() {
        let result = eval_neg(Primitive::Neg, &[v_f64(&[1.0, -2.0, 0.0])]).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn unary_abs_scalar() {
        let result = eval_abs(Primitive::Abs, &[s_f64(-5.0)]).unwrap();
        assert!((extract_f64(&result) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn unary_abs_tensor() {
        let result = eval_abs(Primitive::Abs, &[v_f64(&[-1.0, 2.0, -3.0])]).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unary_exp_scalar() {
        let result = eval_exp(Primitive::Exp, &[s_f64(0.0)]).unwrap();
        assert!((extract_f64(&result) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unary_log_scalar() {
        let result = eval_log(Primitive::Log, &[s_f64(1.0)]).unwrap();
        assert!(extract_f64(&result).abs() < 1e-12);
    }

    #[test]
    fn unary_sin_cos_identity() {
        let x = 0.7;
        let sin_val = extract_f64(&eval_sin(Primitive::Sin, &[s_f64(x)]).unwrap());
        let cos_val = extract_f64(&eval_cos(Primitive::Cos, &[s_f64(x)]).unwrap());
        assert!((sin_val * sin_val + cos_val * cos_val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unary_exp_log_roundtrip() {
        let x = 2.5;
        let exp_val = eval_exp(Primitive::Exp, &[s_f64(x)]).unwrap();
        let roundtrip = eval_log(Primitive::Log, &[exp_val]).unwrap();
        assert!((extract_f64(&roundtrip) - x).abs() < 1e-12);
    }

    #[test]
    fn unary_exp_f32_tensor_preserves_dtype() {
        // Regression test for frankenjax-eldm: eval_unary_elementwise tensor
        // arm previously emitted F64Bits element literals in an F32-declared
        // tensor, silently corrupting the dtype/element invariant.
        let input = v_f32(&[0.0, 1.0, 2.0]);
        let result = eval_exp(Primitive::Exp, &[input]).unwrap();
        let t = match result {
            Value::Tensor(t) => t,
            other => {
                assert!(matches!(other, Value::Tensor(_)), "expected tensor");
                return;
            }
        };
        assert_eq!(t.dtype, DType::F32);
        t.validate_dtype_consistency()
            .expect("F32 tensor must contain only F32Bits elements");
    }

    #[test]
    fn unary_sin_f32_tensor_preserves_dtype() {
        // Regression test for frankenjax-eldm.
        let input = v_f32(&[0.0, 1.0, 2.0]);
        let result = eval_sin(Primitive::Sin, &[input]).unwrap();
        let t = match result {
            Value::Tensor(t) => t,
            other => {
                assert!(matches!(other, Value::Tensor(_)), "expected tensor");
                return;
            }
        };
        assert_eq!(t.dtype, DType::F32);
        t.validate_dtype_consistency()
            .expect("F32 tensor must contain only F32Bits elements");
    }

    #[test]
    fn unary_f64_tensor_fast_path_preserves_output_bits_and_golden() {
        let data = [
            -0.0,
            0.0,
            0.125,
            -1.5,
            PI,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0042),
        ];
        let input = v_f64(&data);

        let sin_result = eval_sin(Primitive::Sin, std::slice::from_ref(&input)).unwrap();
        let exp_result = eval_exp(Primitive::Exp, std::slice::from_ref(&input)).unwrap();

        let sin_bits = extract_f64_bits_vec(&sin_result);
        let exp_bits = extract_f64_bits_vec(&exp_result);
        let expected_sin_bits = data
            .iter()
            .copied()
            .map(|value| value.sin().to_bits())
            .collect::<Vec<_>>();
        let expected_exp_bits = data
            .iter()
            .copied()
            .map(|value| value.exp().to_bits())
            .collect::<Vec<_>>();

        assert_eq!(sin_bits, expected_sin_bits);
        assert_eq!(exp_bits, expected_exp_bits);

        let mut golden_bits = sin_bits;
        golden_bits.extend(exp_bits);
        assert_eq!(
            fixture_id_from_json(&golden_bits).unwrap(),
            "05f74679299f58d4736eaa85fee8265afdc992a428f9cf4e260fd36b1297e2c7"
        );
    }

    #[test]
    fn unary_f64_tensor_fast_path_falls_through_for_malformed_literals() {
        let input = Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape { dims: vec![2] },
            elements: vec![Literal::from_f64(0.0), Literal::Bool(true)].into(),
        });

        let err = eval_exp(Primitive::Exp, &[input]).unwrap_err();
        assert_eq!(
            err,
            EvalError::TypeMismatch {
                primitive: Primitive::Exp,
                detail: "expected numeric tensor elements",
            }
        );
    }

    // ── Complex operations ──

    #[test]
    fn complex_construction() {
        let result = eval_complex(Primitive::Complex, &[s_f64(3.0), s_f64(4.0)]).unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::Complex128Bits(..))),
            "expected complex scalar"
        );
        let Value::Scalar(Literal::Complex128Bits(re, im)) = result else {
            return;
        };
        assert!((f64::from_bits(re) - 3.0).abs() < 1e-12);
        assert!((f64::from_bits(im) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn complex_conj() {
        let z = Value::Scalar(Literal::from_complex128(3.0, 4.0));
        let result = eval_conj(Primitive::Conj, &[z]).unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::Complex128Bits(..))),
            "expected complex scalar"
        );
        let Value::Scalar(Literal::Complex128Bits(re, im)) = result else {
            return;
        };
        assert!((f64::from_bits(re) - 3.0).abs() < 1e-12);
        assert!((f64::from_bits(im) + 4.0).abs() < 1e-12);
    }

    #[test]
    fn complex_real_imag() {
        let z = Value::Scalar(Literal::from_complex128(3.0, 4.0));
        let re = eval_real(Primitive::Real, std::slice::from_ref(&z)).unwrap();
        let im = eval_imag(Primitive::Imag, std::slice::from_ref(&z)).unwrap();
        assert!((extract_f64(&re) - 3.0).abs() < 1e-12);
        assert!((extract_f64(&im) - 4.0).abs() < 1e-12);
    }

    // ── Special math functions ──

    #[test]
    fn erf_at_zero() {
        assert!(erf_approx(0.0).abs() < 1e-12);
    }

    #[test]
    fn erf_symmetry() {
        let x = 1.5;
        assert!((erf_approx(x) + erf_approx(-x)).abs() < 1e-10);
    }

    #[test]
    fn erf_saturation() {
        assert!((erf_approx(5.0) - 1.0).abs() < 1e-6);
        assert!((erf_approx(-5.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn erf_high_accuracy_and_seam_continuity() {
        // Confirmed scipy/JAX values (also pinned in fj-conformance erf_oracle).
        // The new erf agrees to ~1e-12; the old A&S 7.1.26 form was only ~1.5e-7.
        for (x, expected) in [
            (0.5_f64, 0.5204998778130465_f64),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
        ] {
            let got = erf_approx(x);
            assert!(
                (got - expected).abs() < 1e-12,
                "erf({x}) = {got}, want {expected}, diff {}",
                (got - expected).abs()
            );
            assert!((erf_approx(-x) + got).abs() < 1e-15, "odd symmetry at {x}");
        }
        // The Maclaurin branch (just below 3.5) and the asymptotic-erfc branch
        // (>= 3.5) must agree at the seam — catches a wrong large-x series. The
        // points straddle 3.5 by only 1e-7, so the true erf change there
        // (~erf'(3.5)*1e-7 ≈ 5e-13) is far below the 1e-9 agreement bound.
        assert!(
            (erf_approx(3.4999999) - erf_approx(3.5)).abs() < 1e-9,
            "seam discontinuity at |x|=3.5"
        );
        // Monotone toward 1; exactly ±1 past ~6.
        assert!(erf_approx(3.0) < erf_approx(4.0) && erf_approx(4.0) < erf_approx(5.0));
        assert!(erf_approx(5.0) > 0.999_999_999_9 && erf_approx(5.0) < 1.0);
        assert_eq!(erf_approx(6.5), 1.0);
        assert_eq!(erf_approx(-6.5), -1.0);
        assert_eq!(erf_approx(0.0), 0.0);
        assert!(erf_approx(f64::NAN).is_nan());
    }

    #[test]
    fn lgamma_at_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0 (since Gamma(1) = Gamma(2) = 1)
        assert!(lgamma_approx(1.0).abs() < 1e-6);
        assert!(lgamma_approx(2.0).abs() < 1e-6);
        // lgamma(0.5) = ln(sqrt(pi)) ≈ 0.5723649...
        assert!((lgamma_approx(0.5) - 0.5 * PI.ln()).abs() < 1e-4);
    }

    #[test]
    fn digamma_at_one() {
        // ψ(1) = -γ ≈ -0.5772156649
        let euler_gamma = 0.5772156649;
        assert!((digamma_approx(1.0) + euler_gamma).abs() < 1e-4);
    }

    #[test]
    fn hurwitz_zeta_high_accuracy() {
        // Euler-Maclaurin zeta agrees with true scipy/JAX values to ~1e-12
        // (the old naive truncated sum was only ~1e-4 at s=2). Riemann zeta is
        // ζ(s) = hurwitz_zeta(s, 1); also check genuine Hurwitz cases (q ≠ 1).
        let cases = [
            (2.0, 1.0, PI * PI / 6.0),        // ζ(2) = π²/6
            (3.0, 1.0, 1.2020569031595943),   // Apéry's constant
            (4.0, 1.0, PI.powi(4) / 90.0),    // ζ(4) = π⁴/90
            (5.0, 1.0, 1.036_927_755_143_37), // ζ(5)
            (6.0, 1.0, PI.powi(6) / 945.0),   // ζ(6) = π⁶/945
            (2.0, 2.0, PI * PI / 6.0 - 1.0),  // ζ(2,2) = ζ(2) - 1
            (2.0, 0.5, PI * PI / 2.0),        // ζ(2,1/2) = π²/2
        ];
        for (s, q, expected) in cases {
            let got = hurwitz_zeta_approx(s, q);
            assert!(
                (got - expected).abs() < 1e-11,
                "zeta({s},{q}) = {got}, want {expected}, diff {}",
                (got - expected).abs()
            );
        }
        assert!(hurwitz_zeta_approx(1.0, 1.0).is_infinite()); // pole at s=1
        assert!(hurwitz_zeta_approx(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn polygamma_n0_is_digamma() {
        let x = 2.0;
        let pg0 = polygamma_approx(0, x);
        let dg = digamma_approx(x);
        assert!(
            (pg0 - dg).abs() < 1e-10,
            "polygamma(0,x) should equal digamma(x)"
        );
    }

    #[test]
    fn polygamma_n1_is_trigamma() {
        let x = 2.0;
        let pg1 = polygamma_approx(1, x);
        let tg = trigamma_approx(x);
        assert!(
            (pg1 - tg).abs() < 1e-2,
            "polygamma(1,x) should equal trigamma(x): got {} vs {}",
            pg1,
            tg
        );
    }

    #[test]
    fn polygamma_recurrence() {
        // ψ^(n)(x+1) = ψ^(n)(x) + (-1)^n * n! / x^(n+1)
        let x = 1.5;
        let n = 2_i64;
        let pg_x = polygamma_approx(n, x);
        let pg_x1 = polygamma_approx(n, x + 1.0);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let n_fact = (1..=n as u64).product::<u64>() as f64;
        let expected = pg_x + sign * n_fact / x.powi(n as i32 + 1);
        assert!(
            (pg_x1 - expected).abs() < 1e-6,
            "recurrence failed: {} vs {}",
            pg_x1,
            expected
        );
    }

    #[test]
    fn eval_polygamma_scalar() {
        let result = eval_polygamma(
            Primitive::Polygamma,
            &[
                Value::Scalar(Literal::I64(0)),
                Value::Scalar(Literal::from_f64(1.0)),
            ],
        )
        .unwrap();
        let expected = digamma_approx(1.0);
        let got = match result {
            Value::Scalar(Literal::F64Bits(bits)) => f64::from_bits(bits),
            other => {
                assert!(
                    matches!(other, Value::Scalar(Literal::F64Bits(_))),
                    "expected F64Bits scalar"
                );
                return;
            }
        };
        assert!((got - expected).abs() < 1e-10);
    }

    #[test]
    fn erf_inv_roundtrip() {
        let x = 0.5;
        let y = erf_approx(x);
        let x_recovered = erf_inv_approx(y);
        assert!((x_recovered - x).abs() < 1e-4);
    }

    // ── Select ──

    #[test]
    fn select_scalar_true() {
        let result = eval_select(
            Primitive::Select,
            &[Value::Scalar(Literal::Bool(true)), s_f64(10.0), s_f64(20.0)],
        )
        .unwrap();
        assert!((extract_f64(&result) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn select_scalar_false() {
        let result = eval_select(
            Primitive::Select,
            &[
                Value::Scalar(Literal::Bool(false)),
                s_f64(10.0),
                s_f64(20.0),
            ],
        )
        .unwrap();
        assert!((extract_f64(&result) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn select_f64_same_shape_fast_path_bit_identical() {
        // Bool cond + F64/F64 same-shape select must pick the selected operand's
        // bits exactly (including -0.0, NaN, +-inf) and match the generic path.
        let cond_flags = [true, false, true, false, true, false];
        let true_data = [1.5, -0.0, f64::INFINITY, f64::NAN, -2.5, 0.0];
        let false_data = [-9.0, f64::NEG_INFINITY, 3.0, -0.0, f64::NAN, 8.25];
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape {
                    dims: vec![cond_flags.len() as u32],
                },
                cond_flags.iter().map(|&b| Literal::Bool(b)).collect(),
            )
            .unwrap(),
        );
        let result = eval_select(
            Primitive::Select,
            &[cond, v_f64(&true_data), v_f64(&false_data)],
        )
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.dtype, DType::F64);
        let expected: Vec<Literal> = cond_flags
            .iter()
            .zip(true_data.iter().zip(false_data.iter()))
            .map(|(&flag, (&t, &f))| Literal::from_f64(if flag { t } else { f }))
            .collect();
        // Raw-bit comparison distinguishes -0.0 and NaN payloads.
        assert_eq!(tensor.elements, expected);
    }

    // ── Clamp ──

    #[test]
    fn clamp_scalar_within_bounds() {
        let result = eval_clamp(Primitive::Clamp, &[s_f64(5.0), s_f64(0.0), s_f64(10.0)]).unwrap();
        assert!((extract_f64(&result) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_scalar_below_min() {
        let result = eval_clamp(Primitive::Clamp, &[s_f64(-5.0), s_f64(0.0), s_f64(10.0)]).unwrap();
        assert!((extract_f64(&result) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_scalar_above_max() {
        let result = eval_clamp(Primitive::Clamp, &[s_f64(15.0), s_f64(0.0), s_f64(10.0)]).unwrap();
        assert!((extract_f64(&result) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_f64_tensor_scalar_bounds_fast_path_bit_identical() {
        // F64 tensor with scalar bounds, both operand orders, with NaN / -0.0 /
        // +-inf, must match the scalar clamp_f64 reference bit-for-bit.
        let x_data = [-1.0, 0.5, 2.0, f64::NAN, -0.0, 1.0, f64::INFINITY];
        let lo = 0.0_f64;
        let hi = 1.0_f64;
        let reference = |lo: f64, x: f64, hi: f64| -> f64 {
            if lo.is_nan() || x.is_nan() || hi.is_nan() {
                return f64::NAN;
            }
            let lower_bounded = if x < lo { lo } else { x };
            if lower_bounded > hi {
                hi
            } else {
                lower_bounded
            }
        };
        let expected: Vec<Literal> = x_data
            .iter()
            .map(|&x| Literal::from_f64(reference(lo, x, hi)))
            .collect();

        // JAX order: clamp(min, x, max)
        let r1 = eval_clamp(Primitive::Clamp, &[s_f64(lo), v_f64(&x_data), s_f64(hi)]).unwrap();
        // Legacy order: clamp(x, min, max)
        let r2 = eval_clamp(Primitive::Clamp, &[v_f64(&x_data), s_f64(lo), s_f64(hi)]).unwrap();
        for r in [r1, r2] {
            let Value::Tensor(tensor) = r else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.dtype, DType::F64);
            // Raw-bit comparison distinguishes -0.0 and NaN payloads.
            assert_eq!(tensor.elements, expected);
        }
    }

    // ── Dot product ──

    #[test]
    fn dot_scalar_scalar() {
        let result = eval_dot(&[s_f64(3.0), s_f64(4.0)]).unwrap();
        assert!((extract_f64(&result) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn dot_scalar_vector_matches_multiply() {
        let result = eval_dot(&[s_f64(2.0), v_f64(&[1.0, -2.0, 3.5])]).unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(extract_f64_vec(&result), vec![2.0, -4.0, 7.0]);
    }

    #[test]
    fn dot_matrix_scalar_preserves_integral_output() {
        let result = eval_dot(&[matrix_i64(2, 2, &[1, 2, 3, 4]), s_i64(3)]).unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(extract_i64_vec(&result), vec![3, 6, 9, 12]);
    }

    #[test]
    fn dot_complex_scalar_vector_matches_multiply() {
        let result = eval_dot(&[
            Value::scalar_complex128(2.0, -1.0),
            v_complex128(&[(1.0, 3.0), (-2.0, 0.5)]),
        ])
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::Complex128);
        assert_eq!(tensor.shape.dims, vec![2]);
        let values = extract_complex_vec(&result);
        assert_complex_close(values[0], (5.0, 5.0));
        assert_complex_close(values[1], (-3.5, 3.0));
    }

    #[test]
    fn dot_vector_vector() {
        // [1, 2, 3] · [4, 5, 6] = 4 + 10 + 18 = 32
        let result = eval_dot(&[v_f64(&[1.0, 2.0, 3.0]), v_f64(&[4.0, 5.0, 6.0])]).unwrap();
        assert!((extract_f64(&result) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn dot_orthogonal_vectors() {
        // [1, 0] · [0, 1] = 0
        let result = eval_dot(&[v_f64(&[1.0, 0.0]), v_f64(&[0.0, 1.0])]).unwrap();
        assert!(extract_f64(&result).abs() < 1e-12);
    }

    #[test]
    fn dot_matrix_vector() {
        // [[1, 2, 3], [4, 5, 6]] · [10, 20, 30] = [140, 320]
        let result = eval_dot(&[
            matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            v_f64(&[10.0, 20.0, 30.0]),
        ])
        .unwrap();
        assert_eq!(result.as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(extract_f64_vec(&result), vec![140.0, 320.0]);
    }

    #[test]
    fn dot_vector_matrix() {
        // [1, 2] · [[10, 20, 30], [40, 50, 60]] = [90, 120, 150]
        let result = eval_dot(&[
            v_f64(&[1.0, 2.0]),
            matrix_f64(2, 3, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ])
        .unwrap();
        assert_eq!(result.as_tensor().unwrap().shape.dims, vec![3]);
        assert_eq!(extract_f64_vec(&result), vec![90.0, 120.0, 150.0]);
    }

    #[test]
    fn dot_matrix_matrix() {
        // [[1, 2, 3], [4, 5, 6]] · [[7, 8], [9, 10], [11, 12]]
        // = [[58, 64], [139, 154]]
        let result = eval_dot(&[
            matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            matrix_f64(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ])
        .unwrap();
        assert_eq!(result.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(extract_f64_vec(&result), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn dot_rank2_matmul_f64_matches_row_major_ijk_bits() {
        let lhs = [
            0.125,
            -3.5,
            f64::from_bits(0x8000_0000_0000_0000),
            7.25,
            11.0,
            -13.75,
            17.5,
            -19.25,
            23.0,
            29.125,
            -31.5,
            37.75,
        ];
        let rhs = [2.0, -5.5, 7.0, 11.25, -13.0, 17.5, 19.0, -23.25];
        let result = eval_dot(&[matrix_f64(3, 4, &lhs), matrix_f64(4, 2, &rhs)]).unwrap();
        assert_eq!(result.as_tensor().unwrap().shape.dims, vec![3, 2]);
        assert_eq!(
            extract_f64_bits_vec(&result),
            reference_matmul_bits(&lhs, 3, 4, &rhs, 2)
        );
    }

    #[test]
    fn dot_rank2_f64_fast_path_falls_back_for_malformed_literals() -> Result<(), String> {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .map_err(|err| format!("{err:?}"))?,
        );
        let rhs = matrix_f64(2, 2, &[5.0, 6.0, 7.0, 8.0]);

        let result = eval_dot(&[lhs, rhs]).map_err(|err| format!("{err:?}"))?;
        let Some(tensor) = result.as_tensor() else {
            return Err("expected tensor".to_owned());
        };
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(
            tensor.elements,
            vec![
                Literal::from_f64(19.0),
                Literal::from_f64(22.0),
                Literal::from_f64(43.0),
                Literal::from_f64(50.0),
            ]
        );
        Ok(())
    }

    #[test]
    fn dot_matrix_matrix_i64_preserves_integral_output() {
        let result = eval_dot(&[
            matrix_i64(2, 3, &[1, 2, 3, 4, 5, 6]),
            matrix_i64(3, 2, &[7, 8, 9, 10, 11, 12]),
        ])
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(extract_i64_vec(&result), vec![58, 64, 139, 154]);
    }

    #[test]
    fn dot_rank3_rhs_vector_keeps_lhs_prefix_axes() {
        let result = eval_dot(&[
            tensor_f64(
                vec![2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            ),
            v_f64(&[1.0, 10.0, 100.0]),
        ])
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(extract_f64_vec(&result), vec![321.0, 654.0, 987.0, 1320.0]);
    }

    #[test]
    fn dot_rank3_rank3_stacks_batch_axes() {
        let result = eval_dot(&[
            tensor_i64(vec![2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]),
            tensor_i64(vec![2, 2, 2], &[10, 20, 30, 40, 1, 2, 3, 4]),
        ])
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 2, 2, 2]);
        assert_eq!(
            extract_i64_vec(&result),
            vec![
                70, 100, 7, 10, 150, 220, 15, 22, 230, 340, 23, 34, 310, 460, 31, 46
            ]
        );
    }

    #[test]
    fn dot_complex_scalar_scalar() {
        let result = eval_dot(&[
            Value::scalar_complex128(1.0, 2.0),
            Value::scalar_complex128(3.0, 4.0),
        ])
        .unwrap();
        assert_complex_close(result.as_complex128_scalar().unwrap(), (-5.0, 10.0));
    }

    #[test]
    fn dot_complex_vector_vector_does_not_conjugate() {
        let result = eval_dot(&[
            v_complex128(&[(1.0, 2.0), (3.0, -1.0)]),
            v_complex128(&[(4.0, -2.0), (-1.0, 0.5)]),
        ])
        .unwrap();
        assert_complex_close(result.as_complex128_scalar().unwrap(), (5.5, 8.5));
    }

    #[test]
    fn dot_complex_matrix_vector() {
        let result = eval_dot(&[
            matrix_complex128(2, 2, &[(1.0, 1.0), (2.0, -1.0), (0.0, 3.0), (-1.0, 0.5)]),
            v_complex128(&[(2.0, 0.0), (1.0, -1.0)]),
        ])
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::Complex128);
        assert_eq!(tensor.shape.dims, vec![2]);
        let values = extract_complex_vec(&result);
        assert_complex_close(values[0], (3.0, -1.0));
        assert_complex_close(values[1], (-0.5, 7.5));
    }

    #[test]
    fn dot_complex64_vector_vector_preserves_complex64_output() {
        let result = eval_dot(&[
            v_complex64(&[(1.0, 2.0), (3.0, -1.0)]),
            v_complex64(&[(4.0, -2.0), (-1.0, 0.5)]),
        ])
        .unwrap();
        let literal = result.as_scalar_literal().unwrap();
        assert!(matches!(literal, Literal::Complex64Bits(..)));
        let (re, im) = literal.as_complex64().unwrap();
        assert!((re - 5.5).abs() < 1e-6);
        assert!((im - 8.5).abs() < 1e-6);
    }

    #[test]
    fn dot_rank2_shape_mismatch_error() {
        let result = eval_dot(&[
            matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            v_f64(&[10.0, 20.0]),
        ]);
        assert!(matches!(result, Err(EvalError::ShapeMismatch { .. })));
    }

    // ── IsFinite ──

    #[test]
    fn is_finite_normal() {
        let result = eval_is_finite(Primitive::IsFinite, &[s_f64(42.0)]).unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::Bool(_))),
            "expected bool"
        );
        let Value::Scalar(Literal::Bool(v)) = result else {
            return;
        };
        assert!(v);
    }

    #[test]
    fn is_finite_nan() {
        let result = eval_is_finite(Primitive::IsFinite, &[s_f64(f64::NAN)]).unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::Bool(_))),
            "expected bool"
        );
        let Value::Scalar(Literal::Bool(v)) = result else {
            return;
        };
        assert!(!v);
    }

    #[test]
    fn is_finite_inf() {
        let result = eval_is_finite(Primitive::IsFinite, &[s_f64(f64::INFINITY)]).unwrap();
        assert!(
            matches!(result, Value::Scalar(Literal::Bool(_))),
            "expected bool"
        );
        let Value::Scalar(Literal::Bool(v)) = result else {
            return;
        };
        assert!(!v);
    }

    // ── IntegerPow ──

    #[test]
    fn integer_pow_positive() {
        let mut params = BTreeMap::new();
        params.insert("exponent".to_owned(), "3".to_owned());
        let result = eval_integer_pow(Primitive::IntegerPow, &[s_f64(2.0)], &params).unwrap();
        assert!((extract_f64(&result) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn integer_pow_zero() {
        let mut params = BTreeMap::new();
        params.insert("exponent".to_owned(), "0".to_owned());
        let result = eval_integer_pow(Primitive::IntegerPow, &[s_f64(5.0)], &params).unwrap();
        assert!((extract_f64(&result) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn integer_pow_negative() {
        let mut params = BTreeMap::new();
        params.insert("exponent".to_owned(), "-2".to_owned());
        let result = eval_integer_pow(Primitive::IntegerPow, &[s_f64(2.0)], &params).unwrap();
        assert!((extract_f64(&result) - 0.25).abs() < 1e-12);
    }

    // ── Nextafter ──

    #[test]
    fn nextafter_upward() {
        let result = eval_nextafter(Primitive::Nextafter, &[s_f64(1.0), s_f64(2.0)]).unwrap();
        let val = extract_f64(&result);
        assert!(val > 1.0);
        assert!(val < 1.0 + 1e-10);
    }

    #[test]
    fn nextafter_downward() {
        let result = eval_nextafter(Primitive::Nextafter, &[s_f64(1.0), s_f64(0.0)]).unwrap();
        let val = extract_f64(&result);
        assert!(val < 1.0);
        assert!(val > 1.0 - 1e-10);
    }

    #[test]
    fn nextafter_f32_scalar_preserves_dtype() {
        let result = eval_nextafter(Primitive::Nextafter, &[s_f32(1.0), s_f32(2.0)]).unwrap();
        let Value::Scalar(Literal::F32Bits(bits)) = result else {
            assert!(
                matches!(result, Value::Scalar(Literal::F32Bits(_))),
                "expected F32 scalar nextafter output"
            );
            return;
        };
        let val = f32::from_bits(bits);
        assert!(val > 1.0);
        assert!(val < 1.0 + 1e-5);
    }

    #[test]
    fn nextafter_f32_tensor_preserves_dtype() {
        let lhs = v_f32(&[1.0, 2.0, 0.0]);
        let rhs = v_f32(&[2.0, 1.0, -1.0]);
        let result = eval_nextafter(Primitive::Nextafter, &[lhs, rhs]).unwrap();
        let Value::Tensor(tensor) = result else {
            assert!(matches!(result, Value::Tensor(_)), "expected tensor output");
            return;
        };
        assert_eq!(tensor.dtype, DType::F32);
        let mut vals = Vec::with_capacity(tensor.elements.len());
        for literal in &tensor.elements {
            assert!(
                matches!(literal, Literal::F32Bits(_)),
                "expected F32Bits element"
            );
            let Literal::F32Bits(bits) = literal else {
                return;
            };
            vals.push(f32::from_bits(*bits));
        }
        assert!(vals[0] > 1.0);
        assert!(vals[1] < 2.0);
        assert!(vals[2] < 0.0);
    }

    #[test]
    fn nextafter_same_shape_f64_tensor_matches_scalar_bits() {
        let lhs = [1.0, 1.0, 0.0, -0.0, 5.0, f64::NAN, f64::from_bits(1)];
        let rhs = [2.0, 0.0, 1.0, -1.0, 5.0, 1.0, 0.0];
        let result = match eval_nextafter(Primitive::Nextafter, &[v_f64(&lhs), v_f64(&rhs)]) {
            Ok(result) => result,
            Err(err) => {
                assert_eq!(err.to_string(), "", "unexpected tensor nextafter error");
                return;
            }
        };
        let result_bits = extract_f64_bits_vec(&result);
        let expected_bits = lhs
            .iter()
            .zip(rhs)
            .map(|(&left, right)| {
                match eval_nextafter(Primitive::Nextafter, &[s_f64(left), s_f64(right)]) {
                    Ok(Value::Scalar(Literal::F64Bits(bits))) => bits,
                    Ok(result) => {
                        assert!(
                            matches!(result, Value::Scalar(Literal::F64Bits(_))),
                            "expected scalar F64Bits output"
                        );
                        0
                    }
                    Err(err) => {
                        assert_eq!(err.to_string(), "", "unexpected scalar nextafter error");
                        0
                    }
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(result_bits, expected_bits);
    }

    // ── SelectN ──

    #[test]
    fn select_n_scalar_index_picks_first() {
        let result = eval_select_n(
            Primitive::SelectN,
            &[s_i64(0), s_f64(10.0), s_f64(20.0), s_f64(30.0)],
        )
        .unwrap();
        assert!((extract_f64(&result) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn select_n_scalar_index_picks_second() {
        let result = eval_select_n(
            Primitive::SelectN,
            &[s_i64(1), s_f64(10.0), s_f64(20.0), s_f64(30.0)],
        )
        .unwrap();
        assert!((extract_f64(&result) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn select_n_scalar_index_picks_last() {
        let result = eval_select_n(
            Primitive::SelectN,
            &[s_i64(2), s_f64(10.0), s_f64(20.0), s_f64(30.0)],
        )
        .unwrap();
        assert!((extract_f64(&result) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn select_n_scalar_index_tensor_operands() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[4.0, 5.0, 6.0]);
        let result = eval_select_n(Primitive::SelectN, &[s_i64(1), a, b]).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn select_n_tensor_index_elementwise() {
        let idx = tensor_i64(vec![3], &[0, 1, 0]);
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[10.0, 20.0, 30.0]);
        let result = eval_select_n(Primitive::SelectN, &[idx, a, b]).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 20.0, 3.0]);
    }

    #[test]
    fn select_n_tensor_index_three_operands() {
        let idx = tensor_i64(vec![4], &[0, 1, 2, 1]);
        let a = v_f64(&[1.0, 2.0, 3.0, 4.0]);
        let b = v_f64(&[10.0, 20.0, 30.0, 40.0]);
        let c = v_f64(&[100.0, 200.0, 300.0, 400.0]);
        let result = eval_select_n(Primitive::SelectN, &[idx, a, b, c]).unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 20.0, 300.0, 40.0]);
    }

    #[test]
    fn select_n_out_of_bounds() {
        let result = eval_select_n(
            Primitive::SelectN,
            &[s_i64(3), s_f64(10.0), s_f64(20.0), s_f64(30.0)],
        );
        assert!(result.is_err());
    }

    // ── DotGeneral ──

    #[test]
    fn dot_general_vector_vector_contract() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[4.0, 5.0, 6.0]);
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "0".to_string());
        params.insert("rhs_contracting_dims".to_string(), "0".to_string());
        params.insert("lhs_batch_dims".to_string(), "".to_string());
        params.insert("rhs_batch_dims".to_string(), "".to_string());
        let result = eval_dot_general(&[a, b], &params).unwrap();
        let val = extract_f64(&result);
        assert!((val - 32.0).abs() < 1e-12);
    }

    #[test]
    fn dot_general_matmul_like() {
        let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = matrix_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "1".to_string());
        params.insert("rhs_contracting_dims".to_string(), "0".to_string());
        params.insert("lhs_batch_dims".to_string(), "".to_string());
        params.insert("rhs_batch_dims".to_string(), "".to_string());
        let result = eval_dot_general(&[a, b], &params).unwrap();
        let t = match result {
            Value::Tensor(t) => t,
            other => {
                assert!(matches!(other, Value::Tensor(_)), "expected tensor");
                return;
            }
        };
        assert_eq!(t.shape.dims, vec![2, 2]);
        let vals = extract_f64_vec(&Value::Tensor(t));
        assert_eq!(vals, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn dot_general_rank2_matmul_f64_matches_row_major_ijk_bits() {
        let lhs = [
            1.25, -2.5, 3.75, -4.0, 5.5, 6.25, -7.75, 8.125, -9.5, 10.25, 11.75, -12.5,
        ];
        let rhs = [-0.5, 2.0, 3.5, -4.25, 5.75, 6.5, -7.25, 8.0];
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "1".to_string());
        params.insert("rhs_contracting_dims".to_string(), "0".to_string());
        params.insert("lhs_batch_dims".to_string(), "".to_string());
        params.insert("rhs_batch_dims".to_string(), "".to_string());
        let result =
            eval_dot_general(&[matrix_f64(3, 4, &lhs), matrix_f64(4, 2, &rhs)], &params).unwrap();
        assert_eq!(result.as_tensor().unwrap().shape.dims, vec![3, 2]);
        assert_eq!(
            extract_f64_bits_vec(&result),
            reference_matmul_bits(&lhs, 3, 4, &rhs, 2)
        );
    }

    #[test]
    fn dot_general_outer_product() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[3.0, 4.0, 5.0]);
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "".to_string());
        params.insert("rhs_contracting_dims".to_string(), "".to_string());
        params.insert("lhs_batch_dims".to_string(), "".to_string());
        params.insert("rhs_batch_dims".to_string(), "".to_string());
        let result = eval_dot_general(&[a, b], &params).unwrap();
        let t = match result {
            Value::Tensor(t) => t,
            other => {
                assert!(matches!(other, Value::Tensor(_)), "expected tensor");
                return;
            }
        };
        assert_eq!(t.shape.dims, vec![2, 3]);
        let vals = extract_f64_vec(&Value::Tensor(t));
        assert_eq!(vals, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn dot_general_batched_matmul() {
        let a = tensor_f64(
            vec![2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let b = tensor_f64(vec![2, 3, 1], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "2".to_string());
        params.insert("rhs_contracting_dims".to_string(), "1".to_string());
        params.insert("lhs_batch_dims".to_string(), "0".to_string());
        params.insert("rhs_batch_dims".to_string(), "0".to_string());
        let result = eval_dot_general(&[a, b], &params).unwrap();
        let t = match result {
            Value::Tensor(t) => t,
            other => {
                assert!(matches!(other, Value::Tensor(_)), "expected tensor");
                return;
            }
        };
        assert_eq!(t.shape.dims, vec![2, 2, 1]);
        let vals = extract_f64_vec(&Value::Tensor(t));
        assert_eq!(vals, vec![1.0, 4.0, 8.0, 11.0]);
    }

    // ── Igamma/Igammac ──

    #[test]
    fn igamma_scalars() {
        let result = eval_igamma(Primitive::Igamma, &[s_f64(1.0), s_f64(1.0)]).unwrap();
        let val = extract_f64(&result);
        let expected = 1.0 - (-1.0_f64).exp();
        assert!(
            (val - expected).abs() < 1e-10,
            "igamma(1,1) should be 1-e^-1"
        );
    }

    #[test]
    fn igamma_at_zero() {
        let result = eval_igamma(Primitive::Igamma, &[s_f64(2.0), s_f64(0.0)]).unwrap();
        let val = extract_f64(&result);
        assert!(val.abs() < 1e-12, "igamma(a, 0) = 0");
    }

    #[test]
    fn igammac_scalars() {
        let result = eval_igammac(Primitive::Igammac, &[s_f64(1.0), s_f64(1.0)]).unwrap();
        let val = extract_f64(&result);
        let expected = (-1.0_f64).exp();
        assert!(
            (val - expected).abs() < 1e-10,
            "igammac(1,1) should be e^-1"
        );
    }

    #[test]
    fn igamma_igammac_sum_to_one() {
        let a = s_f64(2.5);
        let x = s_f64(1.5);
        let igamma_val =
            extract_f64(&eval_igamma(Primitive::Igamma, &[a.clone(), x.clone()]).unwrap());
        let igammac_val = extract_f64(&eval_igammac(Primitive::Igammac, &[a, x]).unwrap());
        assert!((igamma_val + igammac_val - 1.0).abs() < 1e-10, "P + Q = 1");
    }

    #[test]
    fn igamma_grad_a_matches_scipy_golden_both_regimes() {
        // Golden ∂P(a,x)/∂a values from scipy.special.gammainc (central
        // finite difference, h=1e-7), covering both the series regime
        // (x <= 1 or x <= a) and the continued-fraction regime (x > 1 && x > a).
        // Validating against an external reference rather than a finite
        // difference of fj-lax's own igamma_approx — the latter is slightly
        // imprecise in the CF regime and would bias the check.
        let golden = [
            // (a, x, dP/da) — series regime
            (2.0, 1.0, -0.2761960455),
            (3.0, 2.0, -0.2318486705),
            (4.0, 0.5, -0.0038892172),
            (1.5, 1.2, -0.3702955070),
            // continued-fraction regime: x > 1 && x > a
            (1.0, 3.0, -0.0964829422),
            (2.0, 5.0, -0.0558598962),
            (0.7, 4.0, -0.0245665260),
            (2.5, 8.0, -0.0103088638),
        ];
        for (a, x, expected) in golden {
            let analytic = igamma_grad_a_approx(a, x);
            assert!(
                (analytic - expected).abs() < 1e-7,
                "igamma_grad_a({a},{x}): analytic={analytic}, scipy={expected}, diff={}",
                (analytic - expected).abs()
            );
        }
    }

    #[test]
    fn igamma_grad_a_domain_and_edges() {
        assert!(igamma_grad_a_approx(f64::NAN, 1.0).is_nan());
        assert!(igamma_grad_a_approx(2.0, -1.0).is_nan(), "x<0 → NaN");
        assert!(igamma_grad_a_approx(-1.0, 1.0).is_nan(), "a<=0 → NaN");
        assert_eq!(igamma_grad_a_approx(2.0, 0.0), 0.0, "x==0 → 0");
    }

    // ── Betainc ──

    #[test]
    fn betainc_boundary_zero() {
        let result =
            eval_betainc(Primitive::Betainc, &[s_f64(1.0), s_f64(1.0), s_f64(0.0)]).unwrap();
        let val = extract_f64(&result);
        assert!((val - 0.0).abs() < 1e-10, "I_0(a,b) = 0");
    }

    #[test]
    fn betainc_boundary_one() {
        let result =
            eval_betainc(Primitive::Betainc, &[s_f64(1.0), s_f64(1.0), s_f64(1.0)]).unwrap();
        let val = extract_f64(&result);
        assert!((val - 1.0).abs() < 1e-10, "I_1(a,b) = 1");
    }

    #[test]
    fn betainc_uniform() {
        let result =
            eval_betainc(Primitive::Betainc, &[s_f64(1.0), s_f64(1.0), s_f64(0.5)]).unwrap();
        let val = extract_f64(&result);
        assert!((val - 0.5).abs() < 1e-10, "I_0.5(1,1) = 0.5 (uniform)");
    }

    #[test]
    fn betainc_half_half() {
        let result =
            eval_betainc(Primitive::Betainc, &[s_f64(0.5), s_f64(0.5), s_f64(0.5)]).unwrap();
        let val = extract_f64(&result);
        assert!((val - 0.5).abs() < 1e-10, "I_0.5(0.5,0.5) = 0.5 (arcsine)");
    }

    // ── BesselI0e / BesselI1e ──

    #[test]
    fn bessel_i0e_at_zero() {
        let result = eval_bessel_i0e(Primitive::BesselI0e, &[s_f64(0.0)]).unwrap();
        let val = extract_f64(&result);
        assert!((val - 1.0).abs() < 1e-10, "I0e(0) = 1");
    }

    #[test]
    fn bessel_i0e_small() {
        let result = eval_bessel_i0e(Primitive::BesselI0e, &[s_f64(1.0)]).unwrap();
        let val = extract_f64(&result);
        let expected = 0.4657596;
        assert!((val - expected).abs() < 1e-4, "I0e(1) ≈ 0.4658");
    }

    #[test]
    fn bessel_i1e_at_zero() {
        let result = eval_bessel_i1e(Primitive::BesselI1e, &[s_f64(0.0)]).unwrap();
        let val = extract_f64(&result);
        assert!(val.abs() < 1e-10, "I1e(0) = 0");
    }

    #[test]
    fn bessel_i1e_small() {
        let result = eval_bessel_i1e(Primitive::BesselI1e, &[s_f64(1.0)]).unwrap();
        let val = extract_f64(&result);
        let expected = 0.2079104;
        assert!((val - expected).abs() < 1e-4, "I1e(1) ≈ 0.2079");
    }

    #[test]
    fn bessel_i1e_negative() {
        let pos = eval_bessel_i1e(Primitive::BesselI1e, &[s_f64(2.0)]).unwrap();
        let neg = eval_bessel_i1e(Primitive::BesselI1e, &[s_f64(-2.0)]).unwrap();
        assert!(
            (extract_f64(&pos) + extract_f64(&neg)).abs() < 1e-10,
            "I1e(-x) = -I1e(x)"
        );
    }
}
