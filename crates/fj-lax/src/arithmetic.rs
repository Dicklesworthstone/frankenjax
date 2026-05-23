#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::collections::BTreeMap;

use crate::EvalError;
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
        Primitive::Lgamma => "lgamma is not supported for complex dtypes",
        Primitive::Digamma => "digamma is not supported for complex dtypes",
        Primitive::Erf => "erf is not supported for complex dtypes",
        Primitive::Erfc => "erfc is not supported for complex dtypes",
        Primitive::ErfInv => "erf_inv is not supported for complex dtypes",
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
        | Primitive::Atan2 => {}
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

    let out_dtype = promote_dtype(promote_dtype(get_dtype(&inputs[0]), get_dtype(&inputs[1])), get_dtype(&inputs[2]));

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

        elements.push(
            fma_literal(a, b, c).map_err(|e| EvalError::TypeMismatch {
                primitive,
                detail: e,
            })?,
        );
    }

    Ok(Value::Tensor(TensorValue::new(out_dtype, out_shape, elements)?))
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

    match (&inputs[0], &inputs[1], &inputs[2]) {
        (Value::Scalar(lo), Value::Scalar(x), Value::Scalar(hi)) => {
            let result = clamp_literal(*lo, *x, *hi, None)
                .map_err(|detail| EvalError::TypeMismatch { primitive, detail })?;
            Ok(Value::Scalar(result))
        }
        // JAX order: clamp(min, x, max) with scalar bounds
        (Value::Scalar(lo), Value::Tensor(x), Value::Scalar(hi)) => {
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
                x.dtype,
                out_shape,
                elements,
            )?))
        }
        // Scalar lo with tensor x and tensor hi (broadcasts lo)
        (Value::Scalar(lo), Value::Tensor(x), Value::Tensor(hi)) => {
            let out_shape =
                broadcast_shape(&x.shape, &hi.shape).ok_or(EvalError::Unsupported {
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
                x.dtype,
                out_shape,
                elements,
            )?))
        }
        // Tensor lo with tensor x and scalar hi (broadcasts hi)
        (Value::Tensor(lo), Value::Tensor(x), Value::Scalar(hi)) => {
            let out_shape =
                broadcast_shape(&x.shape, &lo.shape).ok_or(EvalError::Unsupported {
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
                x.dtype,
                out_shape,
                elements,
            )?))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "clamp requires broadcast-compatible inputs".to_owned(),
        }),
    }
}

/// Approximate erf using Abramowitz & Stegun formula (max error ~1.5e-7).
pub(crate) fn erf_approx(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
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
    sum += sign * n_fact * 0.5 * pow;

    // Use more terms (k=1..6) for better accuracy
    for k in 1..=6 {
        let rising = rising_factorial(n as u32 + 1, 2 * k as u32 - 1);
        pow *= inv * inv;
        sum += sign * bernoulli[2 * k] * rising as f64 * pow;
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
            let out_shape =
                broadcast_shape(&n_tensor.shape, &x_tensor.shape).ok_or(EvalError::ShapeMismatch {
                    primitive,
                    left: n_tensor.shape.clone(),
                    right: x_tensor.shape.clone(),
                })?;
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
            Ok(Value::Tensor(TensorValue::new(DType::F64, out_shape, elements)?))
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
        return Ok(Value::Tensor(TensorValue::new(DType::F64, out_shape, vec![])?));
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

    Ok(Value::Tensor(TensorValue::new(DType::F64, out_shape, elements)?))
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

    let mut sum = 0.0;
    for n in 0..10000 {
        let term = 1.0 / (n as f64 + q).powf(x);
        if term.is_nan() || term.is_infinite() {
            break;
        }
        sum += term;
        if term < sum.abs() * 1e-15 {
            break;
        }
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
                let z = (f32::from_bits(re_bits) as f64, f32::from_bits(im_bits) as f64);
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
        Value::Scalar(literal) => {
            integer_pow_literal(*literal, exponent)
                .map(Value::Scalar)
                .map_err(|e| EvalError::TypeMismatch {
                    primitive,
                    detail: e,
                })
        }
        Value::Tensor(tensor) => {
            let out_dtype = tensor.dtype;
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                elements.push(
                    integer_pow_literal(*literal, exponent).map_err(|e| EvalError::TypeMismatch {
                        primitive,
                        detail: e,
                    })?,
                );
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

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            Ok(Value::Scalar(next_after_literal(primitive, *lhs, *rhs)?))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
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
