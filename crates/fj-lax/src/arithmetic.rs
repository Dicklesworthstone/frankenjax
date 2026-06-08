#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::EvalError;
use crate::tensor_contraction::{batched_matmul_2d, matmul_2d};
use crate::type_promotion::{binary_literal_op, promote_dtype};

/// Expensive per-element cost amortizes the thread fan-out at a lower element
/// count than cheap ops.
const EXPENSIVE_BINARY_PARALLEL_MIN: usize = 1 << 16; // 65_536

/// Right-size a parallel fan-out to the WORK, not the core count. `std::thread::
/// scope` spawns each OS thread sequentially (~tens of µs apiece), so a flat
/// all-core fan-out is spawn-overhead-dominated at moderate element counts (an
/// elementwise 512k exp was ~3x slower at 64 threads than at 8). Give every
/// thread at least `ELEMS_PER_THREAD` elements; full core-count parallelism is
/// still reached once `elems ≥ cores·ELEMS_PER_THREAD`. Thread count never
/// affects results, so every caller stays bit-identical.
pub(crate) fn work_scaled_threads(elems: usize) -> usize {
    const ELEMS_PER_THREAD: usize = 1 << 16; // 65_536
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (elems / ELEMS_PER_THREAD).clamp(1, cores)
}

#[inline]
fn is_expensive_binary(primitive: Primitive) -> bool {
    matches!(
        primitive,
        Primitive::Pow
            | Primitive::Atan2
            | Primitive::Hypot
            | Primitive::LogAddExp
            | Primitive::XLogY
            // Heavy binary special functions (series / continued-fraction
            // approximations) routed through eval_binary_elementwise — even more
            // compute-bound per element than pow/atan2.
            | Primitive::Igamma
            | Primitive::Igammac
            | Primitive::Zeta
    )
}

/// Complex binary ops whose per-element cost is several complex transcendentals
/// (so threading the dense same-shape case pays). `apply_complex_binary` is
/// infallible for each of these.
#[inline]
fn is_expensive_complex_binary(primitive: Primitive) -> bool {
    matches!(
        primitive,
        Primitive::Pow | Primitive::Atan2 | Primitive::LogAddExp | Primitive::XLogY
    )
}

/// Complex unary/binary transcendentals cost several real transcendentals per
/// element, so they amortize the thread fan-out at a low element count.
const COMPLEX_UNARY_PARALLEL_MIN: usize = 1 << 13; // 8_192

/// Threaded scalar⊗tensor (or tensor⊗scalar) fast path for the expensive binary
/// ops. Mirrors [`eval_same_shape_f64_expensive_parallel`] for the broadcast case
/// (e.g. `x ** 2.0`), applying the identical `float_op` so it is bit-for-bit
/// identical to the serial generic path. `scalar_on_left` selects operand order.
fn eval_f64_scalar_expensive_parallel(
    primitive: Primitive,
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Option<Value> {
    if !is_expensive_binary(primitive) {
        return None;
    }
    let Literal::F64Bits(scalar_bits) = scalar else {
        return None;
    };
    if tensor.dtype != DType::F64 {
        return None;
    }
    let src = tensor.elements.as_f64_slice()?;
    let n = src.len();
    if n < EXPENSIVE_BINARY_PARALLEL_MIN {
        return None;
    }
    let scalar = f64::from_bits(scalar_bits);
    let threads = work_scaled_threads(n);
    if threads <= 1 {
        return None;
    }
    let mut out = vec![0.0f64; n];
    let chunk = n.div_ceil(threads);
    let op_ref = float_op;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < n {
            let len = chunk.min(n - start);
            let (blk, tail) = rest.split_at_mut(len);
            rest = tail;
            let s = start;
            scope.spawn(move || {
                for (i, o) in blk.iter_mut().enumerate() {
                    let v = src[s + i];
                    *o = if scalar_on_left {
                        op_ref(scalar, v)
                    } else {
                        op_ref(v, scalar)
                    };
                }
            });
            start += len;
        }
    });
    TensorValue::new_f64_values(tensor.shape.clone(), out)
        .ok()
        .map(Value::Tensor)
}

/// Expensive elementwise binary ops are dominated by per-element transcendental
/// cost, not memory traffic, so threading over elements scales (unlike cheap
/// memory-bound add/sub/mul). These currently fall through to the generic
/// per-`Literal` path; this routes the same-shape dense-F64 case onto scoped
/// threads applying the exact same `float_op`. Cheap ops (add/sub/mul/div/max/min)
/// are intentionally excluded — they are memory-bound and threading regresses
/// them on large-L3 hosts.
fn eval_same_shape_f64_expensive_parallel(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Option<Value> {
    if !is_expensive_binary(primitive) {
        return None;
    }
    let (a, b) = (lhs.elements.as_f64_slice()?, rhs.elements.as_f64_slice()?);
    let n = a.len();
    if n < EXPENSIVE_BINARY_PARALLEL_MIN {
        return None;
    }
    let threads = work_scaled_threads(n);
    if threads <= 1 {
        return None;
    }
    let mut out = vec![0.0f64; n];
    let chunk = n.div_ceil(threads);
    let op_ref = float_op;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < n {
            let len = chunk.min(n - start);
            let (blk, tail) = rest.split_at_mut(len);
            rest = tail;
            let s = start;
            scope.spawn(move || {
                for (i, o) in blk.iter_mut().enumerate() {
                    *o = op_ref(a[s + i], b[s + i]);
                }
            });
            start += len;
        }
    });
    TensorValue::new_f64_values(lhs.shape.clone(), out)
        .ok()
        .map(Value::Tensor)
}

/// Binary elementwise operation dispatching on int/float paths.
/// Supports full NumPy broadcasting: scalar-scalar, tensor-tensor (same shape),
/// scalar-tensor, tensor-scalar, and multi-dim broadcasting.
#[inline]
pub(crate) fn eval_binary_elementwise(
    primitive: Primitive,
    inputs: &[Value],
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64 + Sync,
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
                    && let Some(value) =
                        eval_same_shape_f64_expensive_parallel(primitive, lhs, rhs, &float_op)
                {
                    return Ok(value);
                }
                if lhs.dtype == DType::F64
                    && rhs.dtype == DType::F64
                    && let Some(value) = eval_same_shape_f64_binop(primitive, lhs, rhs)?
                {
                    return Ok(value);
                }
                if lhs.dtype == DType::I64
                    && rhs.dtype == DType::I64
                    && let Some(value) = eval_same_shape_i64_binop(lhs, rhs, &int_op)?
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
            if let Some(value) =
                eval_f64_scalar_expensive_parallel(primitive, *lhs, rhs, true, &float_op)
            {
                return Ok(value);
            }
            if let Some(value) = eval_f64_scalar_broadcast_binop(primitive, *lhs, rhs, true)? {
                return Ok(value);
            }
            if let Some(value) = eval_i64_scalar_broadcast_binop(*lhs, rhs, true, &int_op)? {
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
            if let Some(value) =
                eval_f64_scalar_expensive_parallel(primitive, *rhs, lhs, false, &float_op)
            {
                return Ok(value);
            }
            if let Some(value) = eval_f64_scalar_broadcast_binop(primitive, *rhs, lhs, false)? {
                return Ok(value);
            }
            if let Some(value) = eval_i64_scalar_broadcast_binop(*rhs, lhs, false, &int_op)? {
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
/// Same-shape I64⊗I64 elementwise fast path for any binop whose I64⊗I64 result
/// is `Literal::I64(int_op(a, b))` — i.e. every primitive routed through
/// `eval_binary_elementwise` (Add/Sub/Mul/Div/Rem/Max/Min/Pow); the dispatcher's
/// `int_op` carries the exact per-primitive semantics (e.g. `wrapping_add`,
/// `checked_div().unwrap_or(0)`). Comparisons take a separate path and never
/// reach here.
fn eval_same_shape_i64_binop(
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_op: &impl Fn(i64, i64) -> i64,
) -> Result<Option<Value>, EvalError> {
    // Dense i64 fast path: fold the two contiguous `i64` backing slices directly
    // and emit a dense `i64` output. Bit-for-bit identical to the generic
    // `Vec<Literal>` loop below — same `int_op`, same element order, same I64
    // output — but skips the per-element `Literal::I64` match and the 24-byte
    // enum stride (8 vs 24 bytes/element). `as_i64_slice()` is `Some` only for
    // I64 dense storage.
    if let (Some(left_values), Some(right_values)) =
        (lhs.elements.as_i64_slice(), rhs.elements.as_i64_slice())
    {
        let values: Vec<i64> = left_values
            .iter()
            .zip(right_values)
            .map(|(&left, &right)| int_op(left, right))
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_i64_values(
            lhs.shape.clone(),
            values,
        )?)));
    }

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

/// I64 scalar⊗tensor broadcast fast path. `scalar_on_left` distinguishes
/// `Scalar ⊗ Tensor` (`int_op(scalar, elem)`) from `Tensor ⊗ Scalar`
/// (`int_op(elem, scalar)`), preserving operand order for non-commutative ops.
/// Bit-for-bit identical to the generic `binary_literal_op` path: for I64⊗I64 it
/// returns `Literal::I64(int_op(a, b))`, and `promote_dtype(I64, I64) == I64`.
/// Returns `Ok(None)` (falling back to the generic loop) unless the scalar is
/// `I64` and the tensor is I64 dense storage.
#[inline]
fn eval_i64_scalar_broadcast_binop(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    int_op: &impl Fn(i64, i64) -> i64,
) -> Result<Option<Value>, EvalError> {
    let Literal::I64(scalar) = scalar else {
        return Ok(None);
    };
    if tensor.dtype != DType::I64 {
        return Ok(None);
    }
    let Some(values) = tensor.elements.as_i64_slice() else {
        return Ok(None);
    };
    let out: Vec<i64> = if scalar_on_left {
        values.iter().map(|&v| int_op(scalar, v)).collect()
    } else {
        values.iter().map(|&v| int_op(v, scalar)).collect()
    };
    Ok(Some(Value::Tensor(TensorValue::new_i64_values(
        tensor.shape.clone(),
        out,
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
        Primitive::Add => f64_scalar_broadcast_map(
            scalar,
            tensor,
            scalar_on_left,
            crate::dense::ArithOp::Add,
            |a, b| a + b,
        ),
        Primitive::Sub => f64_scalar_broadcast_map(
            scalar,
            tensor,
            scalar_on_left,
            crate::dense::ArithOp::Sub,
            |a, b| a - b,
        ),
        Primitive::Mul => f64_scalar_broadcast_map(
            scalar,
            tensor,
            scalar_on_left,
            crate::dense::ArithOp::Mul,
            |a, b| a * b,
        ),
        Primitive::Div => f64_scalar_broadcast_map(
            scalar,
            tensor,
            scalar_on_left,
            crate::dense::ArithOp::Div,
            |a, b| a / b,
        ),
        _ => Ok(None),
    }
}

#[inline]
fn f64_scalar_broadcast_map(
    scalar: f64,
    tensor: &TensorValue,
    scalar_on_left: bool,
    dense_op: crate::dense::ArithOp,
    op: impl Fn(f64, f64) -> f64,
) -> Result<Option<Value>, EvalError> {
    if let Some(values) = tensor.elements.as_f64_slice() {
        return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
            tensor.shape.clone(),
            crate::dense::scalar_op(values, scalar, dense_op, scalar_on_left),
        )?)));
    }

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
        DType::BF16 => Literal::from_bf16_f64(value),
        DType::F16 => Literal::from_f16_f64(value),
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
    // erf is odd — reduce to Re(z) ≥ 0 so the large-|z| asymptotic branch (valid for
    // |arg z| < 3π/4) always applies after the reduction.
    if z.0 < 0.0 {
        let e = complex_erf((-z.0, -z.1));
        return (-e.0, -e.1);
    }
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let mag_sq = z.0 * z.0 + z.1 * z.1;
    if mag_sq < 16.0 {
        // |z| < 4: Maclaurin series erf(z) = (2/√π) Σ_{n≥0} (−1)ⁿ z^{2n+1}/(n!(2n+1)).
        // Accurate for small |z|; catastrophic cancellation only bites past |z|≈4, where
        // the asymptotic branch below takes over (the old code ran this series for ALL z,
        // returning garbage once the alternating terms dwarfed the ~O(1) result).
        let mut result = z;
        let mut z_power = z;
        let z_squared = complex_mul(z, z);
        let neg_z_squared = (-z_squared.0, -z_squared.1);
        let mut n_factorial = 1.0_f64;
        for n in 1..60 {
            z_power = complex_mul(z_power, neg_z_squared);
            n_factorial *= n as f64;
            let denom = n_factorial * (2 * n + 1) as f64;
            let term = (z_power.0 / denom, z_power.1 / denom);
            result = complex_add(result, term);
            if term.0.abs() < 1e-17 && term.1.abs() < 1e-17 {
                break;
            }
        }
        return (result.0 * two_over_sqrt_pi, result.1 * two_over_sqrt_pi);
    }
    // |z| ≥ 4, Re(z) ≥ 0: erf(z) = 1 − erfc(z) with the asymptotic expansion
    //   erfc(z) ~ e^{−z²}/(z√π) · Σ_{k≥0} (−1)ᵏ (2k−1)!! / (2z²)ᵏ.
    // The series is divergent (asymptotic) — truncate at the smallest term.
    let z2 = complex_mul(z, z);
    let exp_neg_z2 = complex_exp((-z2.0, -z2.1));
    let inv_2z2 = complex_reciprocal((2.0 * z2.0, 2.0 * z2.1));
    let mut zpow = (1.0_f64, 0.0_f64); // (1/(2z²))ᵏ
    let mut sum = (1.0_f64, 0.0_f64); // k = 0 term
    let mut prev_mag = 1.0_f64;
    let mut dfact = 1.0_f64; // (2k−1)!!
    let mut sign = -1.0_f64;
    for k in 1..=40 {
        dfact *= (2 * k - 1) as f64;
        zpow = complex_mul(zpow, inv_2z2);
        let tk = (sign * dfact * zpow.0, sign * dfact * zpow.1);
        let tmag = tk.0 * tk.0 + tk.1 * tk.1;
        if tmag > prev_mag {
            break; // asymptotic series began to diverge — stop at the smallest term
        }
        sum = complex_add(sum, tk);
        prev_mag = tmag;
        sign = -sign;
    }
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let erfc = complex_mul(
        complex_mul(exp_neg_z2, complex_reciprocal((z.0 * sqrt_pi, z.1 * sqrt_pi))),
        sum,
    );
    (1.0 - erfc.0, -erfc.1)
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

                // Threaded dense fast path for the EXPENSIVE complex binary ops
                // (Pow/Atan2/XLogY/LogAddExp — each several complex transcendentals
                // per element). `apply_complex_binary` is infallible for these, and
                // the packed values + dtype narrowing match the serial path, so this
                // is bit-for-bit identical.
                if is_expensive_complex_binary(primitive)
                    && let (Some(a), Some(b)) = (
                        lhs.elements.as_complex_slice(),
                        rhs.elements.as_complex_slice(),
                    )
                    && a.len() >= COMPLEX_UNARY_PARALLEL_MIN
                {
                    let n = a.len();
                    let threads = std::thread::available_parallelism()
                        .map(|p| p.get())
                        .unwrap_or(1)
                        .min(n);
                    if threads > 1 {
                        let mut out = vec![(0.0f64, 0.0f64); n];
                        let chunk = n.div_ceil(threads);
                        std::thread::scope(|scope| {
                            let mut rest: &mut [(f64, f64)] = out.as_mut_slice();
                            let mut start = 0usize;
                            while start < n {
                                let len = chunk.min(n - start);
                                let (blk, tail) = rest.split_at_mut(len);
                                rest = tail;
                                let s = start;
                                scope.spawn(move || {
                                    for (i, o) in blk.iter_mut().enumerate() {
                                        *o = apply_complex_binary(primitive, a[s + i], b[s + i])
                                            .unwrap_or((f64::NAN, f64::NAN));
                                    }
                                });
                                start += len;
                            }
                        });
                        return Ok(Value::Tensor(TensorValue::new_complex_values(
                            out_dtype,
                            lhs.shape.clone(),
                            out,
                        )?));
                    }
                }

                // Dense same-shape fast path for the CHEAP complex binary ops
                // (Add/Sub/Div, plus any non-Complex128 Mul that skipped the
                // dtype-specialized path above). These are arithmetic-light, so
                // the per-`Literal` unpack (4 f64 from bits) + repack dominates;
                // reading the packed (re,im) slices and applying
                // `apply_complex_binary` straight into dense complex storage
                // removes it. `apply_complex_binary` is exactly what
                // `complex_binary_literal_op` delegates to, and
                // `new_complex_values` applies the same `out_dtype` narrowing,
                // so this is bit-for-bit identical to the per-`Literal` loop
                // below (same proof as the Mul and broadcast dense paths). Serial
                // (no threads): these ops are memory-bound, where fan-out regresses.
                if let (Some(a), Some(b)) = (
                    lhs.elements.as_complex_slice(),
                    rhs.elements.as_complex_slice(),
                ) {
                    let n = a.len();
                    let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
                    for i in 0..n {
                        out.push(apply_complex_binary(primitive, a[i], b[i])?);
                    }
                    return Ok(Value::Tensor(TensorValue::new_complex_values(
                        out_dtype,
                        lhs.shape.clone(),
                        out,
                    )?));
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
                let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
                let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

                // Dense Complex broadcast fast path: read the contiguous (re,im)
                // backings and apply the op directly, replacing the per-element
                // flat→multi decode + Literal unpack/repack. Uses the same
                // outer-odometer + contiguous-inner traversal as the f64/i64
                // broadcast fast paths, so the (lhs_idx, rhs_idx) visited per
                // output element is the same row-major sequence. Bit-for-bit
                // identical: complex_binary_literal_op already delegates to
                // apply_complex_binary, and new_complex_values applies the same
                // out_dtype narrowing (see eval_same_shape / the threaded path).
                if let (Some(a), Some(b)) = (
                    lhs.elements.as_complex_slice(),
                    rhs.elements.as_complex_slice(),
                ) {
                    let rank = out_shape.dims.len();
                    let mut out: Vec<(f64, f64)> = Vec::with_capacity(out_count);
                    if rank >= 1 && out_count > 0 {
                        let inner = out_shape.dims[rank - 1] as usize;
                        let inner_ls = lhs_strides[rank - 1];
                        let inner_rs = rhs_strides[rank - 1];
                        let outer = out_count / inner;
                        let mut coord = vec![0usize; rank.saturating_sub(1)];
                        let mut lb = 0usize;
                        let mut rb = 0usize;
                        for _ in 0..outer {
                            match (inner_ls, inner_rs) {
                                (1, 1) => {
                                    for k in 0..inner {
                                        out.push(apply_complex_binary(
                                            primitive,
                                            a[lb + k],
                                            b[rb + k],
                                        )?);
                                    }
                                }
                                (1, 0) => {
                                    let rv = b[rb];
                                    for k in 0..inner {
                                        out.push(apply_complex_binary(primitive, a[lb + k], rv)?);
                                    }
                                }
                                (0, 1) => {
                                    let lv = a[lb];
                                    for k in 0..inner {
                                        out.push(apply_complex_binary(primitive, lv, b[rb + k])?);
                                    }
                                }
                                _ => {
                                    for k in 0..inner {
                                        out.push(apply_complex_binary(
                                            primitive,
                                            a[lb + k * inner_ls],
                                            b[rb + k * inner_rs],
                                        )?);
                                    }
                                }
                            }
                            if rank >= 2 {
                                let mut ax = rank - 2;
                                loop {
                                    coord[ax] += 1;
                                    lb += lhs_strides[ax];
                                    rb += rhs_strides[ax];
                                    if coord[ax] < out_shape.dims[ax] as usize {
                                        break;
                                    }
                                    coord[ax] = 0;
                                    lb -= lhs_strides[ax] * out_shape.dims[ax] as usize;
                                    rb -= rhs_strides[ax] * out_shape.dims[ax] as usize;
                                    if ax == 0 {
                                        break;
                                    }
                                    ax -= 1;
                                }
                            }
                        }
                    } else {
                        let mut odometer =
                            BroadcastOdometer::new(&out_shape.dims, &lhs_strides, &rhs_strides);
                        for _ in 0..out_count {
                            let (li, ri) = odometer.next();
                            out.push(apply_complex_binary(primitive, a[li], b[ri])?);
                        }
                    }
                    return Ok(Value::Tensor(TensorValue::new_complex_values(
                        out_dtype, out_shape, out,
                    )?));
                }

                let out_strides = compute_strides(&out_shape.dims);
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

    // Dense fast path: when both operands expose packed `(re, im)` storage we
    // multiply straight from the slices into a dense output — no per-`Literal`
    // extraction or rebuild — and fan out across threads for large arrays. The
    // packed Complex128 values are the exact f64 bits a `Literal::Complex128Bits`
    // would carry, so this is bit-identical to the serial Literal path below.
    if let (Some(a), Some(b)) = (
        lhs.elements.as_complex_slice(),
        rhs.elements.as_complex_slice(),
    ) {
        let out = complex128_mul_dense(a, b);
        return Ok(Some(Value::Tensor(TensorValue::new_complex_values(
            DType::Complex128,
            lhs.shape.clone(),
            out,
        )?)));
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

/// Elementwise complex multiply of two equal-length packed `(re, im)` slices into
/// a dense output. Each output element depends only on the same-index inputs, so
/// large arrays are split into disjoint chunks across threads — bit-identical to
/// the serial path (`out[i] = (ar*br - ai*bi, ar*bi + ai*br)`).
fn complex128_mul_dense(a: &[(f64, f64)], b: &[(f64, f64)]) -> Vec<(f64, f64)> {
    debug_assert_eq!(a.len(), b.len());

    #[inline]
    fn mul_one((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
        (ar * br - ai * bi, ar * bi + ai * br)
    }

    // Single-pass build (no zero-init): complex multiply is memory-bandwidth
    // bound, so reading packed `(re, im)` slices and collecting straight into a
    // dense output already saturates the win — no per-`Literal` extract/rebuild.
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| mul_one(av, bv))
        .collect()
}

/// Full NumPy multi-dim broadcasting for two tensors.
/// Incremental dual-index odometer for NumPy multi-dim broadcasting. `next()`
/// returns `(lhs_idx, rhs_idx)` — the broadcast-gathered source offsets for the
/// output element at the current ascending flat position — then advances,
/// reproducing `broadcast_flat_index(flat_to_multi(flat), strides)` for both
/// operands without the per-element `Vec` decode + two stride dot-products. The
/// broadcast-aware strides are 0 on broadcast axes, so an axis that broadcasts
/// simply leaves that operand's index unchanged as it varies. Must be stepped
/// exactly `product(out_dims)` times; the final step harmlessly wraps to 0.
pub(crate) struct BroadcastOdometer {
    dims: Vec<usize>,
    lhs_strides: Vec<usize>,
    rhs_strides: Vec<usize>,
    coord: Vec<usize>,
    lhs_idx: usize,
    rhs_idx: usize,
}

impl BroadcastOdometer {
    pub(crate) fn new(out_dims: &[u32], lhs_strides: &[usize], rhs_strides: &[usize]) -> Self {
        Self {
            dims: out_dims.iter().map(|&d| d as usize).collect(),
            lhs_strides: lhs_strides.to_vec(),
            rhs_strides: rhs_strides.to_vec(),
            coord: vec![0_usize; out_dims.len()],
            lhs_idx: 0,
            rhs_idx: 0,
        }
    }

    #[inline]
    pub(crate) fn next(&mut self) -> (usize, usize) {
        let current = (self.lhs_idx, self.rhs_idx);
        let rank = self.dims.len();
        if rank == 0 {
            return current;
        }
        let mut ax = rank - 1;
        loop {
            self.coord[ax] += 1;
            self.lhs_idx += self.lhs_strides[ax];
            self.rhs_idx += self.rhs_strides[ax];
            if self.coord[ax] < self.dims[ax] {
                break;
            }
            self.coord[ax] = 0;
            self.lhs_idx -= self.lhs_strides[ax] * self.dims[ax];
            self.rhs_idx -= self.rhs_strides[ax] * self.dims[ax];
            if ax == 0 {
                break;
            }
            ax -= 1;
        }
        current
    }
}

/// Visit every output element of a broadcast in row-major order, calling
/// `f(lhs_idx, rhs_idx)` once per element. An outer odometer over the leading
/// dims plus a tight contiguous inner loop over the last axis (the inner stride
/// is always 0 or 1 for a valid broadcast) replaces the per-element
/// [`BroadcastOdometer`] carry, which vectorizes far better. The visited
/// `(lhs_idx, rhs_idx)` sequence is identical to `BroadcastOdometer`'s, so any
/// caller stays bit-for-bit identical to the per-element decode.
#[inline]
pub(crate) fn broadcast_visit_row_major(
    out_dims: &[u32],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    mut f: impl FnMut(usize, usize),
) {
    let rank = out_dims.len();
    let out_count: usize = out_dims.iter().map(|&d| d as usize).product();
    if out_count == 0 {
        return;
    }
    if rank == 0 {
        f(0, 0);
        return;
    }
    let inner = out_dims[rank - 1] as usize;
    let inner_ls = lhs_strides[rank - 1];
    let inner_rs = rhs_strides[rank - 1];
    let outer = out_count / inner;
    let mut coord = vec![0usize; rank - 1];
    let mut lb = 0usize;
    let mut rb = 0usize;
    for _ in 0..outer {
        match (inner_ls, inner_rs) {
            (1, 1) => {
                for k in 0..inner {
                    f(lb + k, rb + k);
                }
            }
            (1, 0) => {
                for k in 0..inner {
                    f(lb + k, rb);
                }
            }
            (0, 1) => {
                for k in 0..inner {
                    f(lb, rb + k);
                }
            }
            _ => {
                for k in 0..inner {
                    f(lb + k * inner_ls, rb + k * inner_rs);
                }
            }
        }
        if rank >= 2 {
            let mut ax = rank - 2;
            loop {
                coord[ax] += 1;
                lb += lhs_strides[ax];
                rb += rhs_strides[ax];
                if coord[ax] < out_dims[ax] as usize {
                    break;
                }
                coord[ax] = 0;
                lb -= lhs_strides[ax] * out_dims[ax] as usize;
                rb -= rhs_strides[ax] * out_dims[ax] as usize;
                if ax == 0 {
                    break;
                }
                ax -= 1;
            }
        }
    }
}

fn broadcast_binary_tensors(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_op: &impl Fn(i64, i64) -> i64,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
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

    // Expensive binary ops (Pow/Atan2/Hypot/…) over a large broadcast: each output
    // element is an independent transcendental on broadcast-gathered operands, so
    // thread over the output flat range. Per-thread per-element index decode is
    // amortized by the expensive float op. Bit-identical to the serial gather.
    if is_expensive_binary(primitive)
        && lhs.dtype == DType::F64
        && rhs.dtype == DType::F64
        && out_count >= EXPENSIVE_BINARY_PARALLEL_MIN
        && let Some(value) = broadcast_binary_f64_expensive_parallel(
            lhs,
            rhs,
            &out_shape,
            out_count,
            &out_strides,
            &lhs_strides,
            &rhs_strides,
            float_op,
        )
    {
        return Ok(value);
    }

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

    // I64⊗I64 dense broadcast fast path: for I64 operands binary_literal_op
    // returns Literal::I64(int_op(a, b)) and promote_dtype(I64, I64) == I64, so
    // a dense int_op fold over the broadcast-gathered i64 values is identical.
    if lhs.dtype == DType::I64
        && rhs.dtype == DType::I64
        && let Some(value) = broadcast_binary_i64(
            lhs,
            rhs,
            &out_shape,
            out_count,
            &lhs_strides,
            &rhs_strides,
            int_op,
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

/// I64 dense broadcast binary fast path mirroring [`broadcast_binary_f64`]: gather
/// the broadcast source offsets from the contiguous `as_i64_slice()` backings via
/// a [`BroadcastOdometer`] (no per-element multi-index decode), apply `int_op`,
/// and emit dense i64. Bit-for-bit identical to the generic broadcast loop for
/// I64⊗I64 (same gather indices in the same row-major order, same `int_op`,
/// `promote_dtype(I64, I64) == I64`). Returns `Ok(None)` unless both operands are
/// I64 dense storage, so the caller falls through to the generic path.
#[inline]
fn broadcast_binary_i64(
    lhs: &TensorValue,
    rhs: &TensorValue,
    out_shape: &Shape,
    out_count: usize,
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    int_op: &impl Fn(i64, i64) -> i64,
) -> Result<Option<Value>, EvalError> {
    let (Some(lhs_values), Some(rhs_values)) =
        (lhs.elements.as_i64_slice(), rhs.elements.as_i64_slice())
    else {
        return Ok(None);
    };
    let rank = out_shape.dims.len();
    let mut values = Vec::with_capacity(out_count);
    if rank >= 1 && out_count > 0 {
        // Contiguous-inner fast path (see broadcast_binary_f64): outer odometer
        // over leading dims + a tight inner loop over the contiguous last axis,
        // branched on the inner strides. The (lhs_idx, rhs_idx) per element is
        // identical to the generic odometer, so `int_op`, operands and order are
        // unchanged — bit-for-bit identical — while the inner run vectorizes.
        let inner = out_shape.dims[rank - 1] as usize;
        let inner_ls = lhs_strides[rank - 1];
        let inner_rs = rhs_strides[rank - 1];
        let outer = out_count / inner;
        let mut coord = vec![0usize; rank.saturating_sub(1)];
        let mut lb = 0usize;
        let mut rb = 0usize;
        for _ in 0..outer {
            match (inner_ls, inner_rs) {
                (1, 1) => {
                    let l = &lhs_values[lb..lb + inner];
                    let r = &rhs_values[rb..rb + inner];
                    for k in 0..inner {
                        values.push(int_op(l[k], r[k]));
                    }
                }
                (1, 0) => {
                    let l = &lhs_values[lb..lb + inner];
                    let rv = rhs_values[rb];
                    for &lv in l {
                        values.push(int_op(lv, rv));
                    }
                }
                (0, 1) => {
                    let lv = lhs_values[lb];
                    let r = &rhs_values[rb..rb + inner];
                    for &rv in r {
                        values.push(int_op(lv, rv));
                    }
                }
                _ => {
                    for k in 0..inner {
                        values.push(int_op(
                            lhs_values[lb + k * inner_ls],
                            rhs_values[rb + k * inner_rs],
                        ));
                    }
                }
            }
            if rank >= 2 {
                let mut ax = rank - 2;
                loop {
                    coord[ax] += 1;
                    lb += lhs_strides[ax];
                    rb += rhs_strides[ax];
                    if coord[ax] < out_shape.dims[ax] as usize {
                        break;
                    }
                    coord[ax] = 0;
                    lb -= lhs_strides[ax] * out_shape.dims[ax] as usize;
                    rb -= rhs_strides[ax] * out_shape.dims[ax] as usize;
                    if ax == 0 {
                        break;
                    }
                    ax -= 1;
                }
            }
        }
    } else {
        let mut odometer = BroadcastOdometer::new(&out_shape.dims, lhs_strides, rhs_strides);
        for _ in 0..out_count {
            let (lhs_idx, rhs_idx) = odometer.next();
            values.push(int_op(lhs_values[lhs_idx], rhs_values[rhs_idx]));
        }
    }
    Ok(Some(Value::Tensor(TensorValue::new_i64_values(
        out_shape.clone(),
        values,
    )?)))
}

/// F64 broadcast binary fast path. Produces output elements in the same
/// row-major flat order as the generic broadcast loop using identical index
/// math, applying `float_op` directly to the gathered F64 values. Bit-for-bit
/// identical to the generic path for F64 operands (where binary_literal_op is
/// `from_f64(float_op(from_bits(l), from_bits(r)))`). Returns `Ok(None)` if any
/// gathered element is not `F64Bits`, so the caller falls through to generic.
#[inline]
/// Threaded broadcast fast path for the expensive binary ops. Returns `None`
/// unless both operands are dense F64 (caller already checked the op + size).
/// Each thread decodes its own output flat-index range to broadcast-gathered
/// operand indices and applies the identical `float_op` — bit-for-bit identical
/// to the serial gather, just split across the output space.
#[allow(clippy::too_many_arguments)]
fn broadcast_binary_f64_expensive_parallel(
    lhs: &TensorValue,
    rhs: &TensorValue,
    out_shape: &Shape,
    out_count: usize,
    out_strides: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Option<Value> {
    let (lhs_values, rhs_values) = (lhs.elements.as_f64_slice()?, rhs.elements.as_f64_slice()?);
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(out_count);
    if threads <= 1 {
        return None;
    }
    let mut out = vec![0.0f64; out_count];
    let chunk = out_count.div_ceil(threads);
    let op_ref = float_op;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < out_count {
            let len = chunk.min(out_count - start);
            let (blk, tail) = rest.split_at_mut(len);
            rest = tail;
            let s = start;
            scope.spawn(move || {
                let mut multi: Vec<usize> = Vec::with_capacity(out_strides.len());
                for (i, o) in blk.iter_mut().enumerate() {
                    flat_to_multi_into(s + i, out_strides, &mut multi);
                    let lhs_idx = broadcast_flat_index(&multi, lhs_strides);
                    let rhs_idx = broadcast_flat_index(&multi, rhs_strides);
                    *o = op_ref(lhs_values[lhs_idx], rhs_values[rhs_idx]);
                }
            });
            start += len;
        }
    });
    TensorValue::new_f64_values(out_shape.clone(), out)
        .ok()
        .map(Value::Tensor)
}

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
    // Dense tier: gather from the contiguous f64 backings via the incremental
    // BroadcastOdometer — same row-major gather order and indices as the generic
    // decode, but no per-element Vec decode or 24-byte Literal materialization.
    if let (Some(lhs_values), Some(rhs_values)) =
        (lhs.elements.as_f64_slice(), rhs.elements.as_f64_slice())
    {
        let rank = out_shape.dims.len();
        let mut values = Vec::with_capacity(out_count);
        if rank >= 1 && out_count > 0 {
            // Contiguous-inner fast path. The generic BroadcastOdometer runs a
            // branchy per-element carry that defeats autovectorization; instead
            // iterate the OUTER dims with the same row-major carry, then run a
            // tight inner loop over the last (contiguous) axis. The (lhs_idx,
            // rhs_idx) visited per element is identical to the odometer's, so the
            // gathered operands, the `float_op`, and the order are unchanged —
            // bit-for-bit identical. Branching on the inner strides lets the
            // (1,1)/(1,0)/(0,1) hot cases (row/col broadcast, bias-add) vectorize.
            let inner = out_shape.dims[rank - 1] as usize;
            let inner_ls = lhs_strides[rank - 1];
            let inner_rs = rhs_strides[rank - 1];
            let outer = out_count / inner;
            let mut coord = vec![0usize; rank.saturating_sub(1)];
            let mut lb = 0usize;
            let mut rb = 0usize;
            for _ in 0..outer {
                match (inner_ls, inner_rs) {
                    (1, 1) => {
                        let l = &lhs_values[lb..lb + inner];
                        let r = &rhs_values[rb..rb + inner];
                        for k in 0..inner {
                            values.push(float_op(l[k], r[k]));
                        }
                    }
                    (1, 0) => {
                        let l = &lhs_values[lb..lb + inner];
                        let rv = rhs_values[rb];
                        for &lv in l {
                            values.push(float_op(lv, rv));
                        }
                    }
                    (0, 1) => {
                        let lv = lhs_values[lb];
                        let r = &rhs_values[rb..rb + inner];
                        for &rv in r {
                            values.push(float_op(lv, rv));
                        }
                    }
                    _ => {
                        for k in 0..inner {
                            values.push(float_op(
                                lhs_values[lb + k * inner_ls],
                                rhs_values[rb + k * inner_rs],
                            ));
                        }
                    }
                }
                // Advance the outer odometer over dims[0..rank-1] (row-major),
                // mirroring BroadcastOdometer's carry exactly.
                if rank >= 2 {
                    let mut ax = rank - 2;
                    loop {
                        coord[ax] += 1;
                        lb += lhs_strides[ax];
                        rb += rhs_strides[ax];
                        if coord[ax] < out_shape.dims[ax] as usize {
                            break;
                        }
                        coord[ax] = 0;
                        lb -= lhs_strides[ax] * out_shape.dims[ax] as usize;
                        rb -= rhs_strides[ax] * out_shape.dims[ax] as usize;
                        if ax == 0 {
                            break;
                        }
                        ax -= 1;
                    }
                }
            }
        } else {
            let mut odometer = BroadcastOdometer::new(&out_shape.dims, lhs_strides, rhs_strides);
            for _ in 0..out_count {
                let (lhs_idx, rhs_idx) = odometer.next();
                values.push(float_op(lhs_values[lhs_idx], rhs_values[rhs_idx]));
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
            out_shape.clone(),
            values,
        )?)));
    }

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
    op: impl Fn(f64, f64) -> (f64, f64) + Sync,
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
            // Dense + threaded fast path: complex transcendentals (exp/log/sin/
            // tanh/…) cost several real transcendentals per element. When the
            // operand is dense-complex-backed and large, apply `op` across scoped
            // threads into a dense output — bit-for-bit identical to the serial map.
            if matches!(tensor.dtype, DType::Complex64 | DType::Complex128)
                && let Some(dense) = tensor.elements.as_complex_slice()
                && dense.len() >= COMPLEX_UNARY_PARALLEL_MIN
            {
                let n = dense.len();
                let threads = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
                    .min(n);
                if threads > 1 {
                    let mut out = vec![(0.0f64, 0.0f64); n];
                    let chunk = n.div_ceil(threads);
                    let op_ref = &op;
                    std::thread::scope(|scope| {
                        let mut rest: &mut [(f64, f64)] = out.as_mut_slice();
                        let mut start = 0usize;
                        while start < n {
                            let len = chunk.min(n - start);
                            let (blk, tail) = rest.split_at_mut(len);
                            rest = tail;
                            let s = start;
                            scope.spawn(move || {
                                for (i, o) in blk.iter_mut().enumerate() {
                                    let (re, im) = dense[s + i];
                                    *o = op_ref(re, im);
                                }
                            });
                            start += len;
                        }
                    });
                    return Ok(Value::Tensor(TensorValue::new_complex_values(
                        tensor.dtype,
                        tensor.shape.clone(),
                        out,
                    )?));
                }
            }

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
        eval_unary_elementwise_parallel(primitive, inputs, f64::exp)
    }
}

pub(crate) fn eval_log(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| complex_log((a, b)))
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::ln)
    }
}

pub(crate) fn eval_sin(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.sin() * b.cosh(), a.cos() * b.sinh())
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::sin)
    }
}

pub(crate) fn eval_cos(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.cos() * b.cosh(), -a.sin() * b.sinh())
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::cos)
    }
}

pub(crate) fn eval_tan(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let denom = (2.0 * a).cos() + (2.0 * b).cosh();
            ((2.0 * a).sin() / denom, (2.0 * b).sinh() / denom)
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::tan)
    }
}

pub(crate) fn eval_sinh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.sinh() * b.cos(), a.cosh() * b.sin())
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::sinh)
    }
}

pub(crate) fn eval_cosh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            (a.cosh() * b.cos(), a.sinh() * b.sin())
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::cosh)
    }
}

pub(crate) fn eval_tanh(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.first().is_some_and(value_contains_complex) {
        eval_unary_complex_map(primitive, inputs, |a, b| {
            let denom = (2.0 * a).cosh() + (2.0 * b).cos();
            ((2.0 * a).sinh() / denom, (2.0 * b).sin() / denom)
        })
    } else {
        eval_unary_elementwise_parallel(primitive, inputs, f64::tanh)
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
        eval_unary_elementwise_parallel(primitive, inputs, f64::asinh)
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
        eval_unary_elementwise_parallel(primitive, inputs, f64::acosh)
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
        eval_unary_elementwise_parallel(primitive, inputs, f64::atanh)
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
                Literal::BF16Bits(_) => Literal::from_bf16_f64(result),
                Literal::F16Bits(_) => Literal::from_f16_f64(result),
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
                    DType::BF16 => Literal::from_bf16_f64(mapped),
                    DType::F16 => Literal::from_f16_f64(mapped),
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

/// Threaded variant of [`eval_unary_elementwise`] for the COMPUTE-bound
/// transcendentals (lgamma, digamma, erf_inv, …) whose per-element cost
/// dominates memory traffic, so threading over elements scales (unlike cheap
/// memory-bound ops such as neg/abs, which stay on the serial path). Threads
/// `op` across a large dense-F64 tensor with scoped threads — each element is
/// independent, so the result is bit-for-bit identical to the serial map (see
/// lgamma_parallel_bit_identical). Falls back to the serial path for scalars,
/// non-F64, complex, Literal-backed, or small inputs.
pub(crate) fn eval_unary_elementwise_parallel(
    primitive: Primitive,
    inputs: &[Value],
    op: impl Fn(f64) -> f64 + Sync,
) -> Result<Value, EvalError> {
    if let [Value::Tensor(tensor)] = inputs
        && tensor.dtype == DType::F64
        && let Some(src) = tensor.elements.as_f64_slice()
    {
        const PARALLEL_MIN_ELEMS: usize = 1 << 18; // 262_144 — enough work to amortize the thread fan-out even for the cheaper kept ops
        let n = src.len();
        let threads = if n >= PARALLEL_MIN_ELEMS {
            work_scaled_threads(n)
        } else {
            1
        };
        if threads > 1 {
            let mut out = vec![0.0f64; n];
            let chunk = n.div_ceil(threads);
            let op_ref = &op;
            std::thread::scope(|scope| {
                let mut rest: &mut [f64] = out.as_mut_slice();
                let mut start = 0usize;
                while start < n {
                    let len = chunk.min(n - start);
                    let (blk, tail) = rest.split_at_mut(len);
                    rest = tail;
                    let s = start;
                    scope.spawn(move || {
                        for (i, o) in blk.iter_mut().enumerate() {
                            *o = op_ref(src[s + i]);
                        }
                    });
                    start += len;
                }
            });
            return Ok(Value::Tensor(TensorValue::new_f64_values(
                tensor.shape.clone(),
                out,
            )?));
        }
    }
    eval_unary_elementwise(primitive, inputs, op)
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
                Ok(Value::Scalar(Literal::from_bf16_f64(float_op(val))))
            }
            Literal::F16Bits(bits) => {
                let val = Literal::F16Bits(bits)
                    .as_f64()
                    .ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric scalar, got f16",
                    })?;
                // Preserve F16 dtype — F32/F64 arms already preserve.
                Ok(Value::Scalar(Literal::from_f16_f64(float_op(val))))
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
                        Literal::from_bf16_f64(float_op(val))
                    }
                    Literal::F16Bits(bits) => {
                        let val =
                            Literal::F16Bits(bits)
                                .as_f64()
                                .ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor elements, got f16",
                                })?;
                        Literal::from_f16_f64(float_op(val))
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
    // Dense path: read both branches from their contiguous f64 backings and pick
    // per the bool cond into a dense f64 output. When the cond is dense Bool too
    // this is a branchless contiguous select the compiler vectorizes, and avoids
    // materializing any branch/cond/output as a 24-byte Literal — the binding cost
    // for jnp.where(mask, a, b) where the mask comes from a (now dense-Bool)
    // comparison. Bit-identical: for dense f64, the stored value round-trips its
    // bits exactly (incl. NaN payloads), and the chosen value matches the prior
    // `Literal::F64Bits(if flag { t } else { f })`.
    if let (Some(t), Some(f)) = (
        on_true.elements.as_f64_slice(),
        on_false.elements.as_f64_slice(),
    ) {
        if let Some(conds) = cond.elements.as_bool_slice() {
            let out: Vec<f64> = conds
                .iter()
                .zip(t)
                .zip(f)
                .map(|((&c, &tv), &fv)| if c { tv } else { fv })
                .collect();
            return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
                cond.shape.clone(),
                out,
            )?)));
        }
        let mut out = Vec::with_capacity(t.len());
        for (i, c) in cond.elements.iter().enumerate() {
            let Literal::Bool(flag) = *c else {
                return Ok(None);
            };
            out.push(if flag { t[i] } else { f[i] });
        }
        return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
            cond.shape.clone(),
            out,
        )?)));
    }

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

/// Dense I64 same-shape `select` fast path: pick `on_true[i]` / `on_false[i]`
/// per the bool `cond[i]`, reading both branches straight from their contiguous
/// `as_i64_slice()` backings and emitting a dense i64 output. Bit-for-bit
/// identical to the generic path: for I64/I64 `promote_dtype` is I64 and
/// `select_literal_as_dtype` returns the chosen `Literal::I64` unchanged. It
/// skips the per-element promote/literal_to_i128 machinery and the 24-byte enum
/// stride on the branches. `cond` is still read per element because no dense Bool
/// storage exists. Returns `Ok(None)` unless both branches are dense I64 storage
/// and every `cond` element is `Literal::Bool`.
#[inline]
fn select_i64_same_shape_fast_path(
    cond: &TensorValue,
    on_true: &TensorValue,
    on_false: &TensorValue,
) -> Result<Option<Value>, EvalError> {
    let (Some(true_values), Some(false_values)) = (
        on_true.elements.as_i64_slice(),
        on_false.elements.as_i64_slice(),
    ) else {
        return Ok(None);
    };
    // Dense Bool cond: branchless contiguous select over the two i64 backings,
    // no per-element Literal materialization of the mask.
    if let Some(conds) = cond.elements.as_bool_slice() {
        let out: Vec<i64> = conds
            .iter()
            .zip(true_values)
            .zip(false_values)
            .map(|((&c, &tv), &fv)| if c { tv } else { fv })
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_i64_values(
            cond.shape.clone(),
            out,
        )?)));
    }
    let conds = cond.elements.as_slice();
    let mut out = Vec::with_capacity(conds.len());
    for (i, c) in conds.iter().enumerate() {
        let Literal::Bool(flag) = *c else {
            return Ok(None);
        };
        out.push(if flag {
            true_values[i]
        } else {
            false_values[i]
        });
    }
    Ok(Some(Value::Tensor(TensorValue::new_i64_values(
        cond.shape.clone(),
        out,
    )?)))
}

/// Dense fast path for `select(tensor_cond, scalar_true, scalar_false)` — the
/// `jnp.where(mask, a, b)` masking idiom with scalar branches. Reads the dense
/// Bool cond slice and writes the chosen scalar straight into a dense f64/i64
/// output, skipping the per-element `Literal` match + `select_literal_as_dtype`
/// rebuild. Returns `None` (caller falls back to the generic loop) when the cond
/// is not dense-Bool or the two scalars are not a matching F64/F64 or I64/I64
/// pair. Bit-identical: `promote_dtype` of equal dtypes is that dtype, and for an
/// F64/I64 value `select_literal_as_dtype` is the identity, so the dense buffer
/// stores exactly the bits the generic path would (incl. F64 NaN payloads).
fn select_scalar_branches_fast_path(
    cond: &TensorValue,
    on_true: Literal,
    on_false: Literal,
) -> Result<Option<Value>, EvalError> {
    let Some(conds) = cond.elements.as_bool_slice() else {
        return Ok(None);
    };
    match (on_true, on_false) {
        (Literal::F64Bits(tb), Literal::F64Bits(fb)) => {
            let t = f64::from_bits(tb);
            let f = f64::from_bits(fb);
            let out: Vec<f64> = conds.iter().map(|&c| if c { t } else { f }).collect();
            Ok(Some(Value::Tensor(TensorValue::new_f64_values(
                cond.shape.clone(),
                out,
            )?)))
        }
        (Literal::I64(tv), Literal::I64(fv)) => {
            let out: Vec<i64> = conds.iter().map(|&c| if c { tv } else { fv }).collect();
            Ok(Some(Value::Tensor(TensorValue::new_i64_values(
                cond.shape.clone(),
                out,
            )?)))
        }
        _ => Ok(None),
    }
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
            if cond.dtype == DType::Bool
                && on_true.dtype == DType::I64
                && on_false.dtype == DType::I64
                && let Some(value) = select_i64_same_shape_fast_path(cond, on_true, on_false)?
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
            if let Some(value) = select_scalar_branches_fast_path(cond, *on_true, *on_false)? {
                return Ok(value);
            }
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
        // Dense path: read the packed f64 backing and clamp straight into dense
        // f64 storage, skipping the per-`Literal` unpack + the 24-byte
        // `Vec<Literal>` output. Bit-identical to the per-element path below:
        // `as_f64_slice` round-trips each `F64Bits` exactly, `clamp_f64`
        // normalizes any-NaN to canonical NaN (so `Literal::from_f64` and the
        // dense store agree bit-for-bit), and `new_f64_values` keeps the f64 bits.
        if let Some(xs) = x.elements.as_f64_slice() {
            let out: Vec<f64> = xs.iter().map(|&xv| clamp_f64(lof, xv, hif)).collect();
            return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
                x.shape.clone(),
                out,
            )?)));
        }
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

/// Complementary error function `erfc(x) = 1 − erf(x)`, accurate even in the far tail.
/// For `|x| < 3.5` the subtraction `1 − erf(x)` is harmless (erf is well below 1). For
/// `|x| ≥ 3.5` erfc is computed DIRECTLY from the asymptotic expansion
///   `erfc(t) ~ e^{−t²}/(t√π) · Σ_k (−1)ᵏ (2k−1)!!/(2t²)ᵏ`  (truncated at the smallest term),
/// so the tiny tail survives. The previous dispatch `1 − erf_approx(x)` returned exactly
/// `0` for `|x| ≥ 6` (erf_approx saturates to ±1 there) — e.g. erfc(8) was 0 vs the true
/// ~1.1e-29, a parity gap vs XLA's directly-evaluated erfc (the tail feeds log-survival /
/// normal-CDF tails). `erfc(−|x|) = 2 − erfc(|x|)`.
pub(crate) fn erfc_approx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        // erfc(+∞) = 0, erfc(−∞) = 2 (the continued fraction below would give ∞·0 = NaN).
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    let ax = x.abs();
    if ax < 3.5 {
        return 1.0 - erf_approx(x);
    }
    // |x| ≥ 3.5: Laplace continued fraction (DLMF 7.9.3), evaluated by modified Lentz —
    //   √π·e^{x²}·erfc(x) = 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + …))))),
    // i.e. the CF a₁/(b₁ + a₂/(b₂ + …)) with bⱼ = x, a₁ = 1, aⱼ = (j−1)/2 for j ≥ 2.
    // Constant-free and accurate to ~1e-15 across the whole tail, unlike the asymptotic
    // series whose relative error floored at ~√ε near x = 3.5. erfc(−x) = 2 − erfc(x).
    const TINY: f64 = 1e-300;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0_f64;
    for j in 1..=300 {
        let aj = if j == 1 { 1.0 } else { (j - 1) as f64 / 2.0 };
        d = ax + aj * d;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;
        c = ax + aj / c;
        if c.abs() < TINY {
            c = TINY;
        }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    let erfc_pos = (-ax * ax).exp() / std::f64::consts::PI.sqrt() * f;
    if x >= 0.0 {
        erfc_pos
    } else {
        2.0 - erfc_pos
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
    eval_unary_elementwise_parallel(primitive, inputs, lgamma_approx)
}

pub(crate) fn eval_digamma(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise_parallel(primitive, inputs, digamma_approx)
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
            // Dense + threaded fast path: polygamma_approx is a heavy series/
            // asymptotic evaluation per element, so a dense-F64 `x` tensor fans out
            // across threads. For F64Bits, polygamma_literal_to_f64 == from_bits ==
            // the slice value, so this is bit-identical to the serial loop below.
            if x_tensor.dtype == DType::F64
                && let Some(xs) = x_tensor.elements.as_f64_slice()
                && xs.len() >= (1 << 13)
            {
                let len = xs.len();
                let threads = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
                    .min(len);
                if threads > 1 {
                    let mut out = vec![0.0f64; len];
                    let chunk = len.div_ceil(threads);
                    std::thread::scope(|scope| {
                        let mut rest: &mut [f64] = out.as_mut_slice();
                        let mut start = 0usize;
                        while start < len {
                            let take = chunk.min(len - start);
                            let (blk, tail) = rest.split_at_mut(take);
                            rest = tail;
                            let s = start;
                            scope.spawn(move || {
                                for (i, o) in blk.iter_mut().enumerate() {
                                    *o = polygamma_approx(n, xs[s + i]);
                                }
                            });
                            start += take;
                        }
                    });
                    return Ok(Value::Tensor(TensorValue::new_f64_values(
                        x_tensor.shape.clone(),
                        out,
                    )?));
                }
            }

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
    eval_unary_elementwise_parallel(primitive, inputs, erf_inv_approx)
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

    // JAX's betainc uses `log1p(-x)` for the b·log(1-x) term (lax/special.py),
    // which stays accurate as x→0 where `(1.0 - x).ln()` loses precision to the
    // rounding of `1 - x` near 1.0.
    let bt = (lgamma_approx(a + b) - lgamma_approx(a) - lgamma_approx(b)
        + a * x.ln()
        + b * (-x).ln_1p())
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

/// Evaluate a Chebyshev series at `x` using Clenshaw recurrence — the Cephes
/// `chbevl` routine. The coefficient arrays are ordered from highest to lowest
/// Chebyshev order and the series is evaluated as `0.5*(b0 - b2)` over a shifted
/// argument. This is the exact primitive XLA/JAX use for `bessel_i0e`/`i1e`.
fn chbevl(x: f64, coeffs: &[f64]) -> f64 {
    let mut b0 = coeffs[0];
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for &c in &coeffs[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

/// Chebyshev coefficients for `exp(-x) I0(x)` on `[0, 8]` (Cephes `i0.c` A[]).
const BESSEL_I0E_A: [f64; 30] = [
    -4.4153416464793395e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728896e-05,
    -5.754195010082104e-05,
    0.00018850288509584165,
    -0.0005763755745385824,
    0.0016394756169413357,
    -0.004324309995050576,
    0.010546460394594998,
    -0.02373741480589947,
    0.04930528423967071,
    -0.09490109704804764,
    0.17162090152220877,
    -0.3046826723431984,
    0.6767952744094761,
];

/// Chebyshev coefficients for `exp(-x) sqrt(x) I0(x)` on `[8, inf]` (Cephes B[]).
const BESSEL_I0E_B: [f64; 25] = [
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565263e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.54008621752141e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916825e-05,
    0.0033691164782556943,
    0.8044904110141088,
];

/// Chebyshev coefficients for `exp(-x) I1(x) / x` on `[0, 8]` (Cephes `i1.c` A[]).
const BESSEL_I1E_A: [f64; 29] = [
    2.7779141127610464e-18,
    -2.111421214358166e-17,
    1.5536319577362005e-16,
    -1.1055969477353862e-15,
    7.600684294735408e-15,
    -5.042185504727912e-14,
    3.223793365945575e-13,
    -1.9839743977649436e-12,
    1.1736186298890901e-11,
    -6.663489723502027e-11,
    3.625590281552117e-10,
    -1.8872497517228294e-09,
    9.381537386495773e-09,
    -4.445059128796328e-08,
    2.0032947535521353e-07,
    -8.568720264695455e-07,
    3.4702513081376785e-06,
    -1.3273163656039436e-05,
    4.781565107550054e-05,
    -0.00016176081582589674,
    0.0005122859561685758,
    -0.0015135724506312532,
    0.004156422944312888,
    -0.010564084894626197,
    0.024726449030626516,
    -0.05294598120809499,
    0.1026436586898471,
    -0.17641651835783406,
    0.25258718644363365,
];

/// Chebyshev coefficients for `exp(-x) sqrt(x) I1(x)` on `[8, inf]` (Cephes B[]).
const BESSEL_I1E_B: [f64; 25] = [
    7.517296310842105e-18,
    4.414348323071708e-18,
    -4.6503053684893586e-17,
    -3.209525921993424e-17,
    2.96262899764595e-16,
    3.3082023109209285e-16,
    -1.8803547755107825e-15,
    -3.8144030724370075e-15,
    1.0420276984128802e-14,
    4.272440016711951e-14,
    -2.1015418427726643e-14,
    -4.0835511110921974e-13,
    -7.198551776245908e-13,
    2.0356285441470896e-12,
    1.4125807436613782e-11,
    3.2526035830154884e-11,
    -1.8974958123505413e-11,
    -5.589743462196584e-10,
    -3.835380385964237e-09,
    -2.6314688468895196e-08,
    -2.512236237870209e-07,
    -3.882564808877691e-06,
    -0.00011058893876262371,
    -0.009761097491361469,
    0.7785762350182801,
];

/// `bessel_i0e(x) = exp(-|x|) * I0(x)`, the exponentially-scaled modified Bessel
/// function of the first kind, order 0. Matches JAX/XLA bit-for-bit via Cephes'
/// Chebyshev expansion (`chbevl`), accurate to ~1e-16. (The previous
/// Abramowitz & Stegun polynomial form was only ~1e-7 accurate.)
pub(crate) fn bessel_i0e_approx(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 8.0 {
        chbevl(ax / 2.0 - 2.0, &BESSEL_I0E_A)
    } else {
        chbevl(32.0 / ax - 2.0, &BESSEL_I0E_B) / ax.sqrt()
    }
}

/// `bessel_i1e(x) = exp(-|x|) * I1(x)`, exponentially-scaled modified Bessel
/// function of the first kind, order 1 (odd in `x`). Matches JAX/XLA bit-for-bit
/// via Cephes' Chebyshev expansion (~1e-16); the previous Abramowitz & Stegun
/// form was only ~1e-7 accurate.
pub(crate) fn bessel_i1e_approx(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax <= 8.0 {
        chbevl(ax / 2.0 - 2.0, &BESSEL_I1E_A) * ax
    } else {
        chbevl(32.0 / ax - 2.0, &BESSEL_I1E_B) / ax.sqrt()
    };
    if x < 0.0 { -result } else { result }
}

pub(crate) fn eval_bessel_i0e(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise_parallel(primitive, inputs, bessel_i0e_approx)
}

pub(crate) fn eval_bessel_i1e(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    eval_unary_elementwise_parallel(primitive, inputs, bessel_i1e_approx)
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

/// Fast path for the canonical **batched** f64 matmul `lhs[B..,M,K]·rhs[B..,K,N]
/// -> out[B..,M,N]` (leading batch dims in order, single contracting axis at the
/// M|K / K|N boundary). Routes each contiguous batch slice through the
/// contiguous, multi-threaded `batched_matmul_2d` kernel instead of the generic
/// per-(output,contracted)-pair `dot_general` loop. Bit-exact (ascending-k,
/// batch-major output) — see dot_general_batched_fast_path_bit_identical.
/// Returns `None` for any non-canonical layout / dtype / sub-threshold size,
/// leaving the generic path untouched.
#[allow(clippy::too_many_arguments)]
fn batched_standard_f64_matmul(
    lhs: &TensorValue,
    rhs: &TensorValue,
    lhs_batch: &[usize],
    rhs_batch: &[usize],
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_free_dims: &[usize],
    rhs_free_dims: &[usize],
    output_dims: &[u32],
) -> Result<Option<Value>, EvalError> {
    if lhs.dtype != DType::F64 || rhs.dtype != DType::F64 {
        return Ok(None);
    }
    let nb = lhs_batch.len();
    if nb == 0
        || rhs_batch.len() != nb
        || lhs.rank() != nb + 2
        || rhs.rank() != nb + 2
        || (0..nb).any(|i| lhs_batch[i] != i || rhs_batch[i] != i)
        || lhs_free_dims != [nb]
        || lhs_contracting != [nb + 1]
        || rhs_contracting != [nb]
        || rhs_free_dims != [nb + 1]
    {
        return Ok(None);
    }
    let m = lhs.shape.dims[nb] as usize;
    let k = lhs.shape.dims[nb + 1] as usize;
    let n = rhs.shape.dims[nb + 1] as usize;
    if k == 0 || rhs.shape.dims[nb] as usize != k || m == 0 || n == 0 {
        return Ok(None);
    }
    let mut batch = 1usize;
    for i in 0..nb {
        if lhs.shape.dims[i] != rhs.shape.dims[i] || lhs.shape.dims[i] == 0 {
            return Ok(None);
        }
        batch = batch.saturating_mul(lhs.shape.dims[i] as usize);
    }
    // Only worth the contiguous-kernel route for enough work; small batched
    // matmuls stay on the LLVM-optimized generic loop (no regression).
    const BATCHED_FASTPATH_MIN_OPS: usize = 1 << 20; // ~1M FMAs
    if batch.saturating_mul(m).saturating_mul(k).saturating_mul(n) < BATCHED_FASTPATH_MIN_OPS {
        return Ok(None);
    }
    let (Some(lhs_values), Some(rhs_values)) = (dot_f64_elements(lhs), dot_f64_elements(rhs))
    else {
        return Ok(None);
    };
    let values = batched_matmul_2d(&lhs_values, batch, m, k, &rhs_values, n);
    Ok(Some(Value::Tensor(TensorValue::new_f64_values(
        Shape {
            dims: output_dims.to_vec(),
        },
        values,
    )?)))
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
    let values = matmul_2d(&lhs_values, m, k, &rhs_values, n);
    if output_dims.is_empty() {
        return Ok(Some(Value::Scalar(Literal::from_f64(values[0]))));
    }

    Ok(Some(Value::Tensor(TensorValue::new_f64_values(
        Shape {
            dims: output_dims.to_vec(),
        },
        values,
    )?)))
}

fn dot_f64_elements(tensor: &TensorValue) -> Option<Cow<'_, [f64]>> {
    if let Some(values) = tensor.elements.as_f64_slice() {
        return Some(Cow::Borrowed(values));
    }

    let mut values = Vec::with_capacity(tensor.elements.len());
    for &literal in &tensor.elements {
        let Literal::F64Bits(bits) = literal else {
            return None;
        };
        values.push(f64::from_bits(bits));
    }
    Some(Cow::Owned(values))
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

    // Canonical batched f64 matmul -> contiguous multi-threaded kernel (fast i-k-j
    // per slice, parallelized over the flattened batch×row space) instead of the
    // generic per-element loop. Bit-exact; falls through for non-canonical cases.
    if let Some(value) = batched_standard_f64_matmul(
        lhs,
        rhs,
        &lhs_batch,
        &rhs_batch,
        &lhs_contracting,
        &rhs_contracting,
        &lhs_free_dims,
        &rhs_free_dims,
        &output_dims,
    )? {
        return Ok(value);
    }

    let lhs_strides = compute_strides(&lhs.shape.dims);
    let rhs_strides = compute_strides(&rhs.shape.dims);

    let batch_ranges: Vec<u32> = lhs_batch.iter().map(|&d| lhs.shape.dims[d]).collect();
    let lhs_free_ranges: Vec<u32> = lhs_free_dims.iter().map(|&d| lhs.shape.dims[d]).collect();
    let rhs_free_ranges: Vec<u32> = rhs_free_dims.iter().map(|&d| rhs.shape.dims[d]).collect();
    let contract_ranges: Vec<u32> = lhs_contracting.iter().map(|&d| lhs.shape.dims[d]).collect();

    // Parallelize large contractions (batched matmul / attention, higher-rank
    // tensordot) across the output index space. Each output element is an
    // independent ascending-`k` reduction with the same index math as the serial
    // nested loop, computed at the same flat position, so the result is
    // bit-for-bit identical (see dot_general_parallel_bit_identical). Gated to
    // large total work so small/normal contractions keep the serial path below.
    const DG_PARALLEL_MIN_OPS: usize = 1 << 26; // ~67M multiply-adds
    let total_ops = output_count.saturating_mul(contracting_size.max(1));
    let dg_threads = if total_ops >= DG_PARALLEL_MIN_OPS {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(output_count.max(1))
    } else {
        1
    };
    if dg_threads > 1 {
        let lhs_el = lhs.elements.as_slice();
        let rhs_el = rhs.elements.as_slice();
        let bs = batch_size.max(1);
        let lfs = lhs_free_size.max(1);
        let rfs = rhs_free_size.max(1);
        let csz = contracting_size.max(1);
        // Per-output-element reduction, addressed by flat output index `o`
        // (decomposing as batch-major × lhs_free × rhs_free, matching the serial
        // push order). Shared immutably across worker threads.
        let compute = |o: usize| -> Result<Literal, EvalError> {
            let rhs_free_flat = o % rfs;
            let t = o / rfs;
            let lhs_free_flat = t % lfs;
            let batch_flat = (t / lfs) % bs;
            let batch_idx = linear_to_multi_index(batch_flat, &batch_ranges);
            let lhs_free_idx = linear_to_multi_index(lhs_free_flat, &lhs_free_ranges);
            let rhs_free_idx = linear_to_multi_index(rhs_free_flat, &rhs_free_ranges);
            dot_accumulate(primitive, output_kind, csz, |k| {
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
                (lhs_el[lhs_index], rhs_el[rhs_index])
            })
        };

        let mut elements = vec![Literal::I64(0); output_count];
        let chunk = output_count.div_ceil(dg_threads);
        let compute_ref = &compute;
        std::thread::scope(|scope| -> Result<(), EvalError> {
            let mut handles = Vec::new();
            let mut rest = elements.as_mut_slice();
            let mut start = 0usize;
            while start < output_count {
                let len = chunk.min(output_count - start);
                let (blk, tail) = rest.split_at_mut(len);
                rest = tail;
                let s = start;
                handles.push(scope.spawn(move || -> Result<(), EvalError> {
                    for (i, slot) in blk.iter_mut().enumerate() {
                        *slot = compute_ref(s + i)?;
                    }
                    Ok(())
                }));
                start += len;
            }
            for h in handles {
                h.join().map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: "dot_general worker thread panicked".into(),
                })??;
            }
            Ok(())
        })?;
        return dot_output_value(dtype, output_dims, elements);
    }

    let mut elements = Vec::with_capacity(output_count);
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
/// Dense f64 fast path for the elementwise float predicates (`is_finite`,
/// `is_nan`, `is_inf`, `signbit`). When the operand is a dense F64 tensor, read
/// its packed f64 slice and apply `pred` straight into dense Bool storage,
/// skipping the per-`Literal` unpack and the boxed `Vec<Literal::Bool>` (24
/// bytes/elem vs 1). Returns `None` (caller falls back to the generic loop) for
/// non-dense or non-F64 operands. Bit-identical: for an F64 element the generic
/// path computes `literal.as_f64()` (= `f64::from_bits`) then the same predicate,
/// which is exactly what `as_f64_slice` + `pred` does; `new_bool_values` stores
/// the same bools.
fn f64_predicate_dense(
    tensor: &TensorValue,
    pred: impl Fn(f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    if tensor.dtype != DType::F64 {
        return Ok(None);
    }
    let Some(xs) = tensor.elements.as_f64_slice() else {
        return Ok(None);
    };
    let out: Vec<bool> = xs.iter().map(|&v| pred(v)).collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        tensor.shape.clone(),
        out,
    )?)))
}

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
            if let Some(value) = f64_predicate_dense(tensor, f64::is_finite)? {
                return Ok(value);
            }
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
            if let Some(value) = f64_predicate_dense(tensor, f64::is_nan)? {
                return Ok(value);
            }
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
            if let Some(value) = f64_predicate_dense(tensor, f64::is_infinite)? {
                return Ok(value);
            }
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
            if let Some(value) = f64_predicate_dense(tensor, f64::is_sign_negative)? {
                return Ok(value);
            }
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

    fn integer_pow_literal(
        literal: Literal,
        exponent: i32,
        dtype: DType,
    ) -> Result<Literal, &'static str> {
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
            // Integer bases with a non-negative exponent use EXACT wrapping integer
            // power (two's-complement, per dtype width), matching XLA/`lax.integer_
            // pow`. The old f64 `powi` round-trip lost precision above 2^53 and
            // saturated on overflow instead of wrapping. (Negative exponents on an
            // integer base — a TypeError in JAX — fall through to the float path.)
            Literal::I64(base) if exponent >= 0 => {
                let e = exponent as u32;
                Ok(match dtype {
                    DType::I32 => Literal::I64(i64::from((base as i32).wrapping_pow(e))),
                    _ => Literal::I64(base.wrapping_pow(e)),
                })
            }
            Literal::U32(base) if exponent >= 0 => {
                Ok(Literal::U32(base.wrapping_pow(exponent as u32)))
            }
            Literal::U64(base) if exponent >= 0 => {
                Ok(Literal::U64(base.wrapping_pow(exponent as u32)))
            }
            _ => {
                let value = literal.as_f64().ok_or("expected numeric")?;
                let in_dtype = literal_dtype(literal);
                Ok(real_literal_from_f64(in_dtype, value.powi(exponent)))
            }
        }
    }

    match &inputs[0] {
        Value::Scalar(literal) => integer_pow_literal(*literal, exponent, literal_dtype(*literal))
            .map(Value::Scalar)
            .map_err(|e| EvalError::TypeMismatch {
                primitive,
                detail: e,
            }),
        Value::Tensor(tensor) => {
            let out_dtype = tensor.dtype;
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                elements.push(
                    integer_pow_literal(*literal, exponent, out_dtype).map_err(|e| {
                        EvalError::TypeMismatch {
                            primitive,
                            detail: e,
                        }
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

    /// Isomorphism + golden proof for threading Sinc: the parallel path must be
    /// BIT-FOR-BIT identical to the serial map (each element independent), and the
    /// same-binary A/B speedup is printed.
    #[test]
    fn sinc_parallel_bit_identical_and_faster() {
        use std::time::Instant;

        let sinc = |x: f64| {
            if x == 0.0 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            }
        };
        let n: u32 = 1 << 20;
        // Include an exact zero (the x==0 branch) plus a wide range.
        let xs: Vec<f64> = (0..n)
            .map(|i| if i == 7 { 0.0 } else { (i as f64) * 1e-3 - 500.0 })
            .collect();
        let shape = Shape { dims: vec![n] };
        let x = Value::Tensor(TensorValue::new_f64_values(shape.clone(), xs.clone()).unwrap());

        let par =
            eval_unary_elementwise_parallel(Primitive::Sinc, std::slice::from_ref(&x), sinc).unwrap();
        let ser = eval_unary_elementwise(Primitive::Sinc, std::slice::from_ref(&x), sinc).unwrap();
        let par_t = par.as_tensor().unwrap();
        let ser_t = ser.as_tensor().unwrap();

        let mut golden: u64 = 0xcbf29ce484222325;
        for k in 0..n as usize {
            assert_eq!(par_t.elements[k], ser_t.elements[k], "sinc mismatch at {k}");
            if let Literal::F64Bits(b) = par_t.elements[k] {
                for byte in b.to_le_bytes() {
                    golden ^= byte as u64;
                    golden = golden.wrapping_mul(0x100000001b3);
                }
            }
        }

        let reps = 30u32;
        let t0 = Instant::now();
        for _ in 0..reps {
            let o =
                eval_unary_elementwise_parallel(Primitive::Sinc, std::slice::from_ref(&x), sinc)
                    .unwrap();
            std::hint::black_box(&o);
        }
        let par_ns = t0.elapsed().as_nanos().max(1);

        let t1 = Instant::now();
        for _ in 0..reps {
            let o = eval_unary_elementwise(Primitive::Sinc, std::slice::from_ref(&x), sinc).unwrap();
            std::hint::black_box(&o);
        }
        let ser_ns = t1.elapsed().as_nanos().max(1);

        let ratio = ser_ns as f64 / par_ns as f64;
        println!(
            "[sinc] parallel={:.3}ms serial={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
            par_ns as f64 / reps as f64 / 1e6,
            ser_ns as f64 / reps as f64 / 1e6,
        );
    }

    /// Isomorphism + golden proof for threading sigmoid (Logistic): the parallel
    /// path must be BIT-FOR-BIT identical to the serial map (each element is
    /// independent), and the same-binary A/B speedup is printed.
    #[test]
    fn logistic_parallel_bit_identical_and_faster() {
        use std::time::Instant;

        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let n: u32 = 1 << 20;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-4 - 50.0).collect();
        let shape = Shape { dims: vec![n] };
        let x = Value::Tensor(TensorValue::new_f64_values(shape.clone(), xs.clone()).unwrap());

        let par = eval_unary_elementwise_parallel(
            Primitive::Logistic,
            std::slice::from_ref(&x),
            sigmoid,
        )
        .unwrap();
        let ser =
            eval_unary_elementwise(Primitive::Logistic, std::slice::from_ref(&x), sigmoid).unwrap();
        let par_t = par.as_tensor().unwrap();
        let ser_t = ser.as_tensor().unwrap();

        let mut golden: u64 = 0xcbf29ce484222325;
        for k in 0..n as usize {
            assert_eq!(par_t.elements[k], ser_t.elements[k], "logistic mismatch at {k}");
            if let Literal::F64Bits(b) = par_t.elements[k] {
                for byte in b.to_le_bytes() {
                    golden ^= byte as u64;
                    golden = golden.wrapping_mul(0x100000001b3);
                }
            }
        }

        let reps = 30u32;
        let t0 = Instant::now();
        for _ in 0..reps {
            let o = eval_unary_elementwise_parallel(
                Primitive::Logistic,
                std::slice::from_ref(&x),
                sigmoid,
            )
            .unwrap();
            std::hint::black_box(&o);
        }
        let par_ns = t0.elapsed().as_nanos().max(1);

        let t1 = Instant::now();
        for _ in 0..reps {
            let o = eval_unary_elementwise(Primitive::Logistic, std::slice::from_ref(&x), sigmoid)
                .unwrap();
            std::hint::black_box(&o);
        }
        let ser_ns = t1.elapsed().as_nanos().max(1);

        let ratio = ser_ns as f64 / par_ns as f64;
        println!(
            "[logistic] parallel={:.3}ms serial={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
            par_ns as f64 / reps as f64 / 1e6,
            ser_ns as f64 / reps as f64 / 1e6,
        );
    }

    /// Isomorphism + golden proof for the dense f64 float-predicate fast paths
    /// (is_finite / is_nan / is_inf / signbit). Output must be BIT-FOR-BIT
    /// identical to the per-`Literal` path; same-binary A/B timing is printed.
    #[test]
    fn f64_float_predicates_dense_path_bit_identical_to_literal() {
        use std::time::Instant;

        let n: u32 = 1 << 20;
        // Cycle through normal, +/-0, +/-inf, NaN, subnormal, negative.
        let specials = [
            1.0f64,
            -2.5,
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE / 2.0,
            -1e300,
            123.456,
        ];
        let xs: Vec<f64> = (0..n as usize).map(|i| specials[i % specials.len()]).collect();
        let shape = Shape { dims: vec![n] };
        let x_tensor = TensorValue::new_f64_values(shape.clone(), xs.clone()).unwrap();
        let x = Value::Tensor(x_tensor.clone());

        type EvalFn = fn(Primitive, &[Value]) -> Result<Value, EvalError>;
        type PredicateCase = (&'static str, Primitive, EvalFn, fn(f64) -> bool);
        let cases: [PredicateCase; 4] = [
            (
                "is_finite",
                Primitive::IsFinite,
                eval_is_finite,
                f64::is_finite,
            ),
            ("is_nan", Primitive::IsNan, eval_is_nan, f64::is_nan),
            ("is_inf", Primitive::IsInf, eval_is_inf, f64::is_infinite),
            (
                "signbit",
                Primitive::Signbit,
                eval_signbit,
                f64::is_sign_negative,
            ),
        ];

        let reps = 40u32;
        for (label, prim, eval_fn, pred) in cases {
            // NEW dense path.
            let dense = eval_fn(prim, std::slice::from_ref(&x)).unwrap();
            let dense_t = dense.as_tensor().unwrap();

            // Reference: the predicate applied directly (what the per-Literal path
            // computes via literal.as_f64()).
            let mut golden: u64 = 0xcbf29ce484222325;
            for (k, &v) in xs.iter().enumerate() {
                assert_eq!(
                    dense_t.elements[k],
                    Literal::Bool(pred(v)),
                    "{label} mismatch at {k} (v={v})"
                );
                golden ^= u64::from(pred(v));
                golden = golden.wrapping_mul(0x100000001b3);
            }

            // Same-binary A/B timing: dense eval vs the per-Literal loop.
            let t0 = Instant::now();
            for _ in 0..reps {
                let out = eval_fn(prim, std::slice::from_ref(&x)).unwrap();
                std::hint::black_box(&out);
            }
            let dense_ns = t0.elapsed().as_nanos().max(1);

            let t1 = Instant::now();
            for _ in 0..reps {
                let mut elements = Vec::with_capacity(n as usize);
                for &v in &xs {
                    elements.push(Literal::Bool(pred(v)));
                }
                let out = TensorValue::new(DType::Bool, shape.clone(), elements).unwrap();
                std::hint::black_box(&out);
            }
            let lit_ns = t1.elapsed().as_nanos().max(1);

            let ratio = lit_ns as f64 / dense_ns as f64;
            println!(
                "[{label}] dense={:.3}ms literal={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
                dense_ns as f64 / reps as f64 / 1e6,
                lit_ns as f64 / reps as f64 / 1e6,
            );
        }
    }

    /// Isomorphism + golden proof for the dense `clamp(scalar_lo, tensor_x,
    /// scalar_hi)` fast path (`jnp.clip`). Every output element must be
    /// BIT-FOR-BIT identical to the old per-`Literal` loop; same-binary A/B
    /// timing is printed (run with `--nocapture`).
    #[test]
    fn clamp_f64_scalar_bounds_dense_path_bit_identical_to_literal() {
        use std::time::Instant;

        // Mirror eval_clamp's nested clamp_f64 exactly (any-NaN -> canonical NaN).
        fn clamp_ref(lo: f64, x: f64, hi: f64) -> f64 {
            if lo.is_nan() || x.is_nan() || hi.is_nan() {
                return f64::NAN;
            }
            let lower_bounded = if x < lo { lo } else { x };
            if lower_bounded > hi { hi } else { lower_bounded }
        }

        let n: u32 = 1 << 20; // 1,048,576 elements
        let lo = -1.0f64;
        let hi = 1.0f64;
        // Values spanning below-lo, in-range, above-hi, plus periodic NaN.
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                if i % 997 == 0 {
                    f64::NAN
                } else {
                    (i as f64) * 1e-4 - 50.0
                }
            })
            .collect();
        let shape = Shape { dims: vec![n] };
        let x_tensor = TensorValue::new_f64_values(shape.clone(), xs.clone()).unwrap();
        let x = Value::Tensor(x_tensor.clone());
        let vlo = Value::Scalar(Literal::F64Bits(lo.to_bits()));
        let vhi = Value::Scalar(Literal::F64Bits(hi.to_bits()));

        // NEW dense path.
        let dense = eval_clamp(Primitive::Clamp, &[vlo.clone(), x.clone(), vhi.clone()]).unwrap();
        let dense_t = dense.as_tensor().unwrap();

        // OLD per-`Literal` path (the loop the dense path replaced).
        let mut ref_elems = Vec::with_capacity(n as usize);
        for elem in x_tensor.elements.iter().copied() {
            let Literal::F64Bits(xb) = elem else { unreachable!() };
            ref_elems.push(Literal::from_f64(clamp_ref(lo, f64::from_bits(xb), hi)));
        }

        // Isomorphism: bit-for-bit identical, every element.
        let mut golden: u64 = 0xcbf29ce484222325;
        for (k, re) in ref_elems.iter().enumerate() {
            assert_eq!(dense_t.elements[k], *re, "clamp bit mismatch at {k}");
            if let Literal::F64Bits(b) = dense_t.elements[k] {
                for byte in b.to_le_bytes() {
                    golden ^= byte as u64;
                    golden = golden.wrapping_mul(0x100000001b3);
                }
            }
        }

        let reps = 40u32;
        let t0 = Instant::now();
        for _ in 0..reps {
            let out =
                eval_clamp(Primitive::Clamp, &[vlo.clone(), x.clone(), vhi.clone()]).unwrap();
            std::hint::black_box(&out);
        }
        let dense_ns = t0.elapsed().as_nanos().max(1);

        let t1 = Instant::now();
        for _ in 0..reps {
            let mut elements = Vec::with_capacity(n as usize);
            for elem in x_tensor.elements.iter().copied() {
                let Literal::F64Bits(xb) = elem else { unreachable!() };
                elements.push(Literal::from_f64(clamp_ref(lo, f64::from_bits(xb), hi)));
            }
            let out = TensorValue::new(DType::F64, shape.clone(), elements).unwrap();
            std::hint::black_box(&out);
        }
        let lit_ns = t1.elapsed().as_nanos().max(1);

        let ratio = lit_ns as f64 / dense_ns as f64;
        println!(
            "[clamp f64] dense={:.3}ms literal={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
            dense_ns as f64 / reps as f64 / 1e6,
            lit_ns as f64 / reps as f64 / 1e6,
        );
    }

    /// Isomorphism + golden proof for the dense `select(tensor_cond, scalar,
    /// scalar)` masking fast path (`jnp.where(mask, a, b)`). Every output element
    /// must be BIT-FOR-BIT identical to the old per-`Literal` loop
    /// (`select_bool_condition` + `select_literal_as_dtype`); same-binary A/B
    /// timing is printed (run with `--nocapture`).
    #[test]
    fn select_scalar_branches_dense_path_bit_identical_to_literal() {
        use std::time::Instant;

        let n: u32 = 1 << 20; // 1,048,576 elements
        let conds: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let shape = Shape { dims: vec![n] };
        let cond_tensor =
            TensorValue::new_bool_values(shape.clone(), conds.clone()).unwrap();
        let cond = Value::Tensor(cond_tensor.clone());

        let reps = 40u32;
        // (label, on_true, on_false, expected output dtype)
        let cases: [(&str, Literal, Literal); 2] = [
            (
                "f64",
                Literal::F64Bits(2.5f64.to_bits()),
                Literal::F64Bits((-7.25f64).to_bits()),
            ),
            ("i64", Literal::I64(7), Literal::I64(-3)),
        ];

        for (label, on_true, on_false) in cases {
            let vt = Value::Scalar(on_true);
            let vf = Value::Scalar(on_false);

            // NEW dense path.
            let dense = eval_select(Primitive::Select, &[cond.clone(), vt.clone(), vf.clone()])
                .unwrap();
            let dense_t = dense.as_tensor().unwrap();

            // OLD per-`Literal` path (the loop the fast path replaced).
            let dtype = promote_dtype(literal_dtype(on_true), literal_dtype(on_false));
            let mut ref_elems = Vec::with_capacity(n as usize);
            for c in cond_tensor.elements.iter().copied() {
                let flag = select_bool_condition(Primitive::Select, c).unwrap();
                let val = if flag { on_true } else { on_false };
                ref_elems.push(
                    select_literal_as_dtype(
                        Primitive::Select,
                        val,
                        dtype,
                        "numeric",
                        "integer",
                        "unsigned",
                    )
                    .unwrap(),
                );
            }

            // Isomorphism: bit-for-bit identical, every element.
            let mut golden: u64 = 0xcbf29ce484222325;
            for (k, re) in ref_elems.iter().enumerate() {
                assert_eq!(dense_t.elements[k], *re, "{label} bit mismatch at {k}");
                let bits = match dense_t.elements[k] {
                    Literal::F64Bits(b) => b,
                    Literal::I64(v) => v as u64,
                    other => panic!("unexpected output literal {other:?}"),
                };
                for byte in bits.to_le_bytes() {
                    golden ^= byte as u64;
                    golden = golden.wrapping_mul(0x100000001b3);
                }
            }

            // Same-binary A/B timing.
            let t0 = Instant::now();
            for _ in 0..reps {
                let out = eval_select(Primitive::Select, &[cond.clone(), vt.clone(), vf.clone()])
                    .unwrap();
                std::hint::black_box(&out);
            }
            let dense_ns = t0.elapsed().as_nanos().max(1);

            let t1 = Instant::now();
            for _ in 0..reps {
                let mut elements = Vec::with_capacity(n as usize);
                for c in cond_tensor.elements.iter().copied() {
                    let flag = select_bool_condition(Primitive::Select, c).unwrap();
                    let val = if flag { on_true } else { on_false };
                    elements.push(
                        select_literal_as_dtype(
                            Primitive::Select,
                            val,
                            dtype,
                            "numeric",
                            "integer",
                            "unsigned",
                        )
                        .unwrap(),
                    );
                }
                let out = TensorValue::new(dtype, shape.clone(), elements).unwrap();
                std::hint::black_box(&out);
            }
            let lit_ns = t1.elapsed().as_nanos().max(1);

            let ratio = lit_ns as f64 / dense_ns as f64;
            println!(
                "[select {label}] dense={:.3}ms literal={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
                dense_ns as f64 / reps as f64 / 1e6,
                lit_ns as f64 / reps as f64 / 1e6,
            );
        }
    }

    /// Isomorphism + golden proof for the same-shape dense complex Add/Sub/Div
    /// fast path: every output element must be BIT-FOR-BIT identical to the old
    /// per-`Literal` loop (`complex_binary_literal_op`), and the same-binary A/B
    /// timing is printed (run with `--nocapture`). Bit-identity is a strictly
    /// stronger proof than a digest match; a dependency-free rolling checksum of
    /// the dense output is also printed as the golden record.
    #[test]
    fn complex_addsubdiv_dense_path_bit_identical_to_literal() {
        use std::time::Instant;

        let n: u32 = 1 << 20; // 1,048,576 complex elements
        let a_pairs: Vec<(f64, f64)> = (0..n)
            .map(|i| (i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
            .collect();
        let b_pairs: Vec<(f64, f64)> = (0..n)
            .map(|i| (i as f64 * 0.125 + 2.0, i as f64 * 0.75 - 4.0))
            .collect();
        let shape = Shape { dims: vec![n] };

        let va = Value::Tensor(
            TensorValue::new_complex_values(DType::Complex128, shape.clone(), a_pairs.clone())
                .unwrap(),
        );
        let vb = Value::Tensor(
            TensorValue::new_complex_values(DType::Complex128, shape.clone(), b_pairs.clone())
                .unwrap(),
        );

        // Reference Literal buffers for the OLD per-`Literal` path.
        let lit_a: Vec<Literal> = a_pairs
            .iter()
            .map(|&(r, i)| Literal::Complex128Bits(r.to_bits(), i.to_bits()))
            .collect();
        let lit_b: Vec<Literal> = b_pairs
            .iter()
            .map(|&(r, i)| Literal::Complex128Bits(r.to_bits(), i.to_bits()))
            .collect();

        let reps = 30u32;
        let out_dtype = DType::Complex128;
        for op in [Primitive::Add, Primitive::Sub, Primitive::Div] {
            // NEW dense path output.
            let dense = eval_binary_elementwise_complex(op, &[va.clone(), vb.clone()]).unwrap();
            let dense_t = dense.as_tensor().unwrap();

            // OLD per-`Literal` path output (the loop the dense path replaced).
            let mut ref_elems = Vec::with_capacity(n as usize);
            for (l, r) in lit_a.iter().copied().zip(lit_b.iter().copied()) {
                ref_elems.push(complex_binary_literal_op(op, l, r, out_dtype).unwrap());
            }

            // Isomorphism: bit-for-bit identical, every element.
            let mut golden: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
            for (k, re) in ref_elems.iter().enumerate() {
                assert_eq!(dense_t.elements[k], *re, "op {op:?} bit mismatch at {k}");
                if let Literal::Complex128Bits(rb, ib) = dense_t.elements[k] {
                    for byte in rb.to_le_bytes().iter().chain(ib.to_le_bytes().iter()) {
                        golden ^= *byte as u64;
                        golden = golden.wrapping_mul(0x100000001b3);
                    }
                }
            }

            // Same-binary A/B timing.
            let t0 = Instant::now();
            for _ in 0..reps {
                let out = eval_binary_elementwise_complex(op, &[va.clone(), vb.clone()]).unwrap();
                std::hint::black_box(&out);
            }
            let dense_ns = t0.elapsed().as_nanos().max(1);

            let t1 = Instant::now();
            for _ in 0..reps {
                let mut elements = Vec::with_capacity(n as usize);
                for (l, r) in lit_a.iter().copied().zip(lit_b.iter().copied()) {
                    elements.push(complex_binary_literal_op(op, l, r, out_dtype).unwrap());
                }
                let out = TensorValue::new(out_dtype, shape.clone(), elements).unwrap();
                std::hint::black_box(&out);
            }
            let lit_ns = t1.elapsed().as_nanos().max(1);

            let ratio = lit_ns as f64 / dense_ns as f64;
            println!(
                "[complex {op:?}] dense={:.3}ms literal={:.3}ms ratio={ratio:.2}x golden={golden:016x}",
                dense_ns as f64 / reps as f64 / 1e6,
                lit_ns as f64 / reps as f64 / 1e6,
            );
        }
    }

    /// Bit-exact parity for the dense-i64 same-shape Add fast path: a dense
    /// `vector_i64` input (folds two i64 slices) must produce element-for-element
    /// identical results to the `Vec<Literal>`-backed tensor (generic loop),
    /// including wrapping at i64::MIN/MAX, which the dispatcher's
    /// `i64::wrapping_add` must reproduce on both paths.
    #[test]
    fn dense_i64_same_shape_add_bit_identical_to_literal_path() {
        let data: Vec<i64> = vec![7, -3, i64::MAX, i64::MIN, 0, -1, 123456789];
        let n = data.len() as u32;

        let dense_lhs = Value::vector_i64(&data).unwrap();
        let dense_rhs = Value::vector_i64(&data).unwrap();
        assert!(
            dense_lhs
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_some()
        );

        let lit = |d: &[i64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: vec![n] },
                    d.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            )
        };
        let literal_lhs = lit(&data);
        let literal_rhs = lit(&data);
        assert!(
            literal_lhs
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_none()
        );

        let p = std::collections::BTreeMap::new();
        let dense_out = crate::eval_primitive(Primitive::Add, &[dense_lhs, dense_rhs], &p).unwrap();
        let literal_out =
            crate::eval_primitive(Primitive::Add, &[literal_lhs, literal_rhs], &p).unwrap();

        let dense_t = dense_out.as_tensor().unwrap();
        let literal_t = literal_out.as_tensor().unwrap();
        let dense_vals: Vec<i64> = dense_t
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let literal_vals: Vec<i64> = literal_t
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(dense_vals, literal_vals);
        // Spot-check the wrapping semantics are preserved (not saturated/panicked).
        assert_eq!(dense_vals[2], i64::MAX.wrapping_add(i64::MAX));
        assert_eq!(dense_vals[3], i64::MIN.wrapping_add(i64::MIN));
    }

    /// Bit-exact parity for the generalized dense-i64 elementwise fast paths:
    /// same-shape Sub/Mul (non-commutative + wrapping) and i64 scalar broadcast
    /// in both operand orders, dense (vector_i64) vs Vec<Literal>-backed.
    #[test]
    fn dense_i64_sub_mul_and_scalar_broadcast_bit_identical_to_literal_path() {
        let data: Vec<i64> = vec![7, -3, i64::MAX, i64::MIN, 0, -1, 5, 123456789];
        let n = data.len() as u32;
        let lit_vec = |d: &[i64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: vec![n] },
                    d.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            )
        };
        let ints = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        let p = std::collections::BTreeMap::new();

        // Same-shape Sub and Mul.
        for prim in [Primitive::Sub, Primitive::Mul] {
            let dense = crate::eval_primitive(
                prim,
                &[
                    Value::vector_i64(&data).unwrap(),
                    Value::vector_i64(&data).unwrap(),
                ],
                &p,
            )
            .unwrap();
            let literal =
                crate::eval_primitive(prim, &[lit_vec(&data), lit_vec(&data)], &p).unwrap();
            assert_eq!(ints(&dense), ints(&literal), "same-shape {prim:?}");
        }

        // Scalar broadcast, both operand orders, non-commutative Sub.
        let scalar = Value::Scalar(Literal::I64(1_000));
        for prim in [Primitive::Sub, Primitive::Mul, Primitive::Add] {
            // Tensor ⊗ Scalar
            let dense_ts = crate::eval_primitive(
                prim,
                &[Value::vector_i64(&data).unwrap(), scalar.clone()],
                &p,
            )
            .unwrap();
            let lit_ts =
                crate::eval_primitive(prim, &[lit_vec(&data), scalar.clone()], &p).unwrap();
            assert_eq!(ints(&dense_ts), ints(&lit_ts), "tensor⊗scalar {prim:?}");
            // Scalar ⊗ Tensor
            let dense_st = crate::eval_primitive(
                prim,
                &[scalar.clone(), Value::vector_i64(&data).unwrap()],
                &p,
            )
            .unwrap();
            let lit_st =
                crate::eval_primitive(prim, &[scalar.clone(), lit_vec(&data)], &p).unwrap();
            assert_eq!(ints(&dense_st), ints(&lit_st), "scalar⊗tensor {prim:?}");
        }
    }

    /// Bit-exact parity for the dense multi-dim broadcast fast paths (i64 + f64)
    /// via the BroadcastOdometer, across several broadcast shapes and ops, vs the
    /// Vec<Literal>-backed generic broadcast loop. Covers row-broadcast,
    /// col-broadcast, rank expansion, and a 3-D case.
    #[test]
    fn dense_broadcast_bit_identical_to_literal_path() {
        let shapes: [(Vec<u32>, Vec<u32>); 5] = [
            (vec![4, 5], vec![5]),    // row vector broadcast over rows
            (vec![4, 5], vec![4, 1]), // column broadcast
            (vec![3, 1], vec![1, 6]), // outer-product style
            (vec![2, 3, 4], vec![4]), // rank expansion + trailing broadcast
            (vec![2, 1, 4], vec![1, 3, 1]),
        ];
        let prod = |d: &[u32]| d.iter().map(|&x| x as usize).product::<usize>();

        for (ls, rs) in shapes {
            let ln = prod(&ls);
            let rn = prod(&rs);
            let lf: Vec<f64> = (0..ln).map(|i| (i as f64 - 3.5) * 0.25).collect();
            let rf: Vec<f64> = (0..rn).map(|i| (i as f64 + 1.0) * 0.5).collect();
            let li: Vec<i64> = (0..ln as i64).map(|i| i - 3).collect();
            let ri: Vec<i64> = (0..rn as i64).map(|i| i + 1).collect();

            let dense_f = |d: &[f64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new_f64_values(Shape { dims: s.to_vec() }, d.to_vec()).unwrap(),
                )
            };
            let lit_f = |d: &[f64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims: s.to_vec() },
                        d.iter().copied().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                )
            };
            let dense_i = |d: &[i64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new_i64_values(Shape { dims: s.to_vec() }, d.to_vec()).unwrap(),
                )
            };
            let lit_i = |d: &[i64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::I64,
                        Shape { dims: s.to_vec() },
                        d.iter().copied().map(Literal::I64).collect(),
                    )
                    .unwrap(),
                )
            };
            let bits = |v: &Value| -> Vec<u64> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| l.as_f64().map(|f| f.to_bits()).unwrap_or(0))
                    .collect()
            };
            let ints = |v: &Value| -> Vec<i64> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| l.as_i64().unwrap())
                    .collect()
            };
            let p = std::collections::BTreeMap::new();

            for prim in [Primitive::Add, Primitive::Sub, Primitive::Mul] {
                let df = crate::eval_primitive(prim, &[dense_f(&lf, &ls), dense_f(&rf, &rs)], &p)
                    .unwrap();
                let lfr =
                    crate::eval_primitive(prim, &[lit_f(&lf, &ls), lit_f(&rf, &rs)], &p).unwrap();
                assert_eq!(
                    df.as_tensor().unwrap().shape.dims,
                    lfr.as_tensor().unwrap().shape.dims
                );
                assert_eq!(bits(&df), bits(&lfr), "f64 {prim:?} {ls:?} {rs:?}");

                let di = crate::eval_primitive(prim, &[dense_i(&li, &ls), dense_i(&ri, &rs)], &p)
                    .unwrap();
                let lir =
                    crate::eval_primitive(prim, &[lit_i(&li, &ls), lit_i(&ri, &rs)], &p).unwrap();
                assert_eq!(ints(&di), ints(&lir), "i64 {prim:?} {ls:?} {rs:?}");
            }
        }
    }

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

    /// Same logical complex128 vector backed by dense packed `(re, im)` storage
    /// (`as_complex_slice`) — exercises the dense complex-multiply fast path.
    fn v_complex128_dense(data: &[(f64, f64)]) -> Value {
        Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.to_vec(),
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

    /// Isomorphism proof for the threaded expensive-binary fast path: a same-shape
    /// dense-F64 Pow/Atan2/Hypot/LogAddExp large enough to fan out across threads
    /// (>= 1<<16) must equal, bit-for-bit, the element-wise reference op applied
    /// in order (each element is independent and uses the identical float op).
    #[test]
    fn threaded_expensive_binary_bit_identical_to_reference() {
        let n = 1usize << 16; // 65_536 -> threaded path
        let a: Vec<f64> = (0..n).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| 0.5 + (i % 13) as f64 * 0.1).collect();
        let va = v_f64(&a);
        let vb = v_f64(&b);
        type BinaryRef = fn(f64, f64) -> f64;
        let cases: [(Primitive, BinaryRef); 3] = [
            (Primitive::Pow, |x, y| x.powf(y)),
            (Primitive::Atan2, |x, y| x.atan2(y)),
            (Primitive::Hypot, |x, y| x.hypot(y)),
        ];
        for (prim, refop) in cases {
            let got =
                crate::eval_primitive(prim, &[va.clone(), vb.clone()], &BTreeMap::new()).unwrap();
            let got: Vec<u64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let expect: Vec<u64> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| refop(x, y).to_bits())
                .collect();
            assert_eq!(got, expect, "{prim:?} threaded != element-wise reference");
        }
    }

    /// Isomorphism proof for the heavy binary special functions newly routed onto
    /// the threaded expensive-binary path (Igamma/Igammac/Zeta): a large same-shape
    /// dense-F64 evaluation must equal the element-wise reference bit-for-bit.
    #[test]
    fn threaded_special_binary_bit_identical_to_reference() {
        let n = 1usize << 16; // > 1<<16 -> threaded
        let a: Vec<f64> = (0..n).map(|i| 1.0 + (i % 97) as f64 * 0.05).collect();
        let x: Vec<f64> = (0..n).map(|i| 0.5 + (i % 211) as f64 * 0.02).collect();
        let va = v_f64(&a);
        let vx = v_f64(&x);
        type SpecialBinaryCase = (Primitive, fn(f64, f64) -> f64);
        let cases: [SpecialBinaryCase; 3] = [
            (Primitive::Igamma, igamma_approx),
            (Primitive::Igammac, igammac_approx),
            (Primitive::Zeta, hurwitz_zeta_approx),
        ];
        for (prim, refop) in cases {
            let got =
                crate::eval_primitive(prim, &[va.clone(), vx.clone()], &BTreeMap::new()).unwrap();
            let got: Vec<u64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let expect: Vec<u64> = a
                .iter()
                .zip(x.iter())
                .map(|(&av, &xv)| refop(av, xv).to_bits())
                .collect();
            assert_eq!(got, expect, "{prim:?} threaded != element-wise reference");
        }
    }

    /// Isomorphism proof for the threaded scalar-broadcast expensive path: a
    /// large `tensor op scalar` and `scalar op tensor` must equal the element-wise
    /// reference (same float op, same operand order). Covers `x ** 2.5`.
    #[test]
    fn threaded_expensive_binary_scalar_bit_identical_to_reference() {
        let n = 1usize << 16;
        let t: Vec<f64> = (0..n).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
        let vt = v_f64(&t);
        let s = 2.5f64;
        let vs = Value::scalar_f64(s);
        // tensor op scalar
        for (prim, refop) in [
            (
                Primitive::Pow,
                (|x: f64, y: f64| x.powf(y)) as fn(f64, f64) -> f64,
            ),
            (Primitive::Atan2, (|x, y| x.atan2(y)) as fn(f64, f64) -> f64),
        ] {
            let got =
                crate::eval_primitive(prim, &[vt.clone(), vs.clone()], &BTreeMap::new()).unwrap();
            let got: Vec<u64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let expect: Vec<u64> = t.iter().map(|&x| refop(x, s).to_bits()).collect();
            assert_eq!(got, expect, "{prim:?} tensor⊗scalar threaded != reference");

            // scalar op tensor
            let got2 =
                crate::eval_primitive(prim, &[vs.clone(), vt.clone()], &BTreeMap::new()).unwrap();
            let got2: Vec<u64> = got2
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let expect2: Vec<u64> = t.iter().map(|&x| refop(s, x).to_bits()).collect();
            assert_eq!(
                got2, expect2,
                "{prim:?} scalar⊗tensor threaded != reference"
            );
        }
    }

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
            assert!(matches!(result, Value::Tensor(_)));
            let Value::Tensor(tensor) = result else {
                return;
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
            assert!(matches!(result, Value::Tensor(_)));
            let Value::Tensor(tensor) = result else {
                return;
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

    /// Isomorphism proof for the threaded dense complex-unary path: a large
    /// dense-complex128 Exp/Log/Tanh (>= 1<<13, threaded) must equal the
    /// Literal-backed serial map bit-for-bit.
    #[test]
    fn threaded_dense_complex_unary_bit_identical_to_literal() {
        let n = 1usize << 13; // 8192 -> threaded
        let data: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let x = i as f64;
                ((x * 0.013).sin() * 2.0, (x * 0.0071).cos() - 0.5)
            })
            .collect();
        let lit = v_complex128(&data);
        let dense = v_complex128_dense(&data);
        assert!(
            dense
                .as_tensor()
                .unwrap()
                .elements
                .as_complex_slice()
                .is_some()
        );
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    other => panic!("expected c128, got {other:?}"),
                })
                .collect()
        };
        let prims = [
            Primitive::Exp,
            Primitive::Log,
            Primitive::Tanh,
            Primitive::Sin,
        ];
        for prim in prims {
            let from_lit =
                crate::eval_primitive(prim, std::slice::from_ref(&lit), &BTreeMap::new()).unwrap();
            let from_dense =
                crate::eval_primitive(prim, std::slice::from_ref(&dense), &BTreeMap::new())
                    .unwrap();
            assert_eq!(
                bits(&from_lit),
                bits(&from_dense),
                "{prim:?} dense threaded != literal serial"
            );
        }
    }

    /// Isomorphism proof for the threaded dense complex-binary path: a large
    /// same-shape dense-c128 Pow/Atan2/XLogY/LogAddExp (>= 1<<13, threaded) must
    /// equal the Literal-backed serial map bit-for-bit.
    #[test]
    fn threaded_dense_complex_binary_bit_identical_to_literal() {
        let n = 1usize << 13;
        let a: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let x = i as f64;
                (1.0 + (x * 0.011).sin(), (x * 0.0073).cos() - 0.3)
            })
            .collect();
        let b: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let x = i as f64;
                (0.5 + (x * 0.017).cos(), (x * 0.0091).sin() + 0.2)
            })
            .collect();
        let la = v_complex128(&a);
        let lb = v_complex128(&b);
        let da = v_complex128_dense(&a);
        let db = v_complex128_dense(&b);
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    other => panic!("expected c128, got {other:?}"),
                })
                .collect()
        };
        let prims = [
            Primitive::Pow,
            Primitive::Atan2,
            Primitive::XLogY,
            Primitive::LogAddExp,
        ];
        for prim in prims {
            let from_lit =
                crate::eval_primitive(prim, &[la.clone(), lb.clone()], &BTreeMap::new()).unwrap();
            let from_dense =
                crate::eval_primitive(prim, &[da.clone(), db.clone()], &BTreeMap::new()).unwrap();
            assert_eq!(
                bits(&from_lit),
                bits(&from_dense),
                "{prim:?} dense threaded != literal serial"
            );
        }
    }

    /// Isomorphism proof: the dense `as_complex_slice` multiply fast path must
    /// produce bit-identical results to the Literal-backed path (and to the
    /// reference formula), including a length above the parallel threshold so the
    /// single-pass dense kernel is exercised at scale.
    #[test]
    fn dense_complex128_mul_bit_identical_to_literal() {
        let params = BTreeMap::new();
        // Small hand-picked edge cases + a large run (> 1<<18) of varied values.
        let mut lhs: Vec<(f64, f64)> = vec![
            (1.5, -2.0),
            (-0.0, 3.0),
            (2.25, -1.75),
            (f64::INFINITY, -4.0),
        ];
        let mut rhs: Vec<(f64, f64)> = vec![(2.0, 0.5), (-1.25, 0.0), (3.5, 6.0), (6.0, 7.0)];
        for i in 0..300_000_usize {
            let x = i as f64;
            lhs.push(((x * 0.013).sin(), (x * 0.027).cos() - 0.4));
            rhs.push(((x * 0.019).cos(), (x * 0.011).sin() + 0.2));
        }
        let lhs = std::hint::black_box(lhs);
        let rhs = std::hint::black_box(rhs);

        let from_lit = crate::eval_primitive(
            Primitive::Mul,
            &[v_complex128(&lhs), v_complex128(&rhs)],
            &params,
        )
        .unwrap();
        let from_dense = crate::eval_primitive(
            Primitive::Mul,
            &[v_complex128_dense(&lhs), v_complex128_dense(&rhs)],
            &params,
        )
        .unwrap();

        let lit_t = from_lit.as_tensor().unwrap();
        let dense_t = from_dense.as_tensor().unwrap();
        assert_eq!(dense_t.dtype, DType::Complex128);
        // Bit-exact across representations (compare materialized literals).
        assert_eq!(
            lit_t.elements.as_slice(),
            dense_t.elements.as_slice(),
            "dense complex-mul output must match literal path bit-for-bit"
        );
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
    fn complex_broadcast_dense_bit_identical_to_literal_path() {
        // The dense Complex128 broadcast fast path must equal the literal-backed
        // fallback (per-element complex_binary_literal_op) bit-for-bit, across
        // broadcast shapes covering (1,1)/(1,0)/(0,1) + rank-3 carries, with
        // NaN / -0.0 / ±inf operands and add/sub/mul/div.
        let shapes: [(Vec<u32>, Vec<u32>); 4] = [
            (vec![4, 5], vec![5]),
            (vec![4, 5], vec![4, 1]),
            (vec![3, 1], vec![1, 6]),
            (vec![2, 3, 4], vec![4]),
        ];
        let edge = [
            (1.5, -2.0),
            (-0.0, 0.0),
            (f64::INFINITY, -1.0),
            (f64::NAN, 3.0),
            (0.0, f64::NEG_INFINITY),
            (-3.25, -0.0),
        ];
        let prod = |d: &[u32]| d.iter().map(|&x| x as usize).product::<usize>();
        let p = std::collections::BTreeMap::new();
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    other => panic!("expected Complex128, got {other:?}"),
                })
                .collect()
        };
        for (ls, rs) in shapes {
            let lf: Vec<(f64, f64)> = (0..prod(&ls)).map(|i| edge[i % edge.len()]).collect();
            let rf: Vec<(f64, f64)> = (0..prod(&rs)).map(|i| edge[(i + 2) % edge.len()]).collect();
            let dense = |d: &[(f64, f64)], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new_complex_values(
                        DType::Complex128,
                        Shape { dims: s.to_vec() },
                        d.to_vec(),
                    )
                    .unwrap(),
                )
            };
            let lit = |d: &[(f64, f64)], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::Complex128,
                        Shape { dims: s.to_vec() },
                        d.iter()
                            .map(|&(re, im)| Literal::Complex128Bits(re.to_bits(), im.to_bits()))
                            .collect(),
                    )
                    .unwrap(),
                )
            };
            for prim in [
                Primitive::Add,
                Primitive::Sub,
                Primitive::Mul,
                Primitive::Div,
            ] {
                let d =
                    crate::eval_primitive(prim, &[dense(&lf, &ls), dense(&rf, &rs)], &p).unwrap();
                let l = crate::eval_primitive(prim, &[lit(&lf, &ls), lit(&rf, &rs)], &p).unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().shape.dims,
                    l.as_tensor().unwrap().shape.dims,
                    "{prim:?} {ls:?}/{rs:?} shape"
                );
                assert_eq!(bits(&d), bits(&l), "{prim:?} {ls:?}/{rs:?} bits");
            }
        }
    }

    /// Isomorphism proof for the threaded expensive-broadcast path: a large
    /// `[N,1] op [1,M]` Pow/Atan2 (output >= 1<<16) must equal the element-wise
    /// reference `out[i*M+j] = op(a[i], b[j])`, bit-for-bit (same broadcast indices,
    /// same float op, just split across the output space).
    #[test]
    fn threaded_expensive_broadcast_bit_identical_to_reference() {
        let (n, m) = (512usize, 256usize); // 131072 outputs -> threaded
        assert!(n * m >= (1usize << 16));
        let a: Vec<f64> = (0..n).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
        let b: Vec<f64> = (0..m).map(|j| 0.25 + (j % 13) as f64 * 0.1).collect();
        let lhs = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32, 1],
                },
                a.clone(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![1, m as u32],
                },
                b.clone(),
            )
            .unwrap(),
        );
        for (prim, refop) in [
            (
                Primitive::Pow,
                (|x: f64, y: f64| x.powf(y)) as fn(f64, f64) -> f64,
            ),
            (Primitive::Atan2, (|x, y| x.atan2(y)) as fn(f64, f64) -> f64),
        ] {
            let got =
                crate::eval_primitive(prim, &[lhs.clone(), rhs.clone()], &BTreeMap::new()).unwrap();
            let got: Vec<u64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let mut expect: Vec<u64> = Vec::with_capacity(n * m);
            for &ai in &a {
                for &bj in &b {
                    expect.push(refop(ai, bj).to_bits());
                }
            }
            assert_eq!(got, expect, "{prim:?} broadcast threaded != reference");
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
                let tensor_val = Value::vector_f64(&tensor_data).unwrap();
                let inputs = if scalar_on_left {
                    [scalar_val, tensor_val]
                } else {
                    [tensor_val, scalar_val]
                };
                let result = eval_binary_elementwise(primitive, &inputs, |x, y| x + y, op).unwrap();
                let Value::Tensor(tensor) = result else {
                    assert!(matches!(result, Value::Tensor(_)));
                    return;
                };
                assert_eq!(tensor.dtype, DType::F64);
                assert!(
                    tensor.elements.as_f64_slice().is_some(),
                    "dense F64 scalar broadcast output should remain dense"
                );
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
    fn f64_scalar_broadcast_fast_path_preserves_literal_backed_fallback() {
        let tensor_data = [1.5, -0.0, f64::NAN, f64::INFINITY];
        let tensor_val = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![4] },
                tensor_data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let scalar_val = Value::Scalar(Literal::from_f64(2.5));

        let result = eval_binary_elementwise(
            Primitive::Sub,
            &[tensor_val, scalar_val],
            |x, y| x - y,
            |x, y| x - y,
        )
        .unwrap();
        let Value::Tensor(tensor) = result else {
            assert!(matches!(result, Value::Tensor(_)));
            return;
        };
        assert!(
            tensor.elements.as_f64_slice().is_none(),
            "literal-backed fallback should remain literal-backed"
        );
        let expected = tensor_data
            .iter()
            .copied()
            .map(|value| Literal::from_f64(value - 2.5))
            .collect::<Vec<_>>();
        assert_eq!(tensor.elements, expected);
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

    /// Isomorphism proof for the threaded dense polygamma path: polygamma(n, x)
    /// over a large dense-F64 tensor (>= 1<<13, threaded) must equal the
    /// element-wise polygamma_approx(n, x) reference bit-for-bit.
    #[test]
    fn threaded_dense_polygamma_bit_identical_to_reference() {
        let len = 1usize << 13;
        let x: Vec<f64> = (0..len).map(|i| 0.5 + (i % 4096) as f64 * 0.01).collect();
        let xt = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![len as u32],
                },
                x.clone(),
            )
            .unwrap(),
        );
        for n in [0i64, 1, 2, 3] {
            let got = crate::eval_primitive(
                Primitive::Polygamma,
                &[Value::scalar_i64(n), xt.clone()],
                &BTreeMap::new(),
            )
            .unwrap();
            let got: Vec<u64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let expect: Vec<u64> = x
                .iter()
                .map(|&v| polygamma_approx(n, v).to_bits())
                .collect();
            assert_eq!(
                got, expect,
                "polygamma n={n} threaded != element-wise reference"
            );
        }
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

    #[test]
    fn select_i64_dense_fast_path_matches_generic() {
        // Dense i64 select (vector_i64 branches) must match the Literal-backed
        // generic path element-for-element, incl i64::MIN/MAX and both cond flags.
        let n = 600usize;
        let cond_flags: Vec<bool> = (0..n).map(|i| i % 3 == 0 || i % 7 == 1).collect();
        let t: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(2_654_435_761) - 5)
            .collect();
        let f: Vec<i64> = (0..n)
            .map(|i| match i % 5 {
                0 => i64::MIN,
                1 => i64::MAX,
                _ => -(i as i64) * 3,
            })
            .collect();
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape {
                    dims: vec![n as u32],
                },
                cond_flags.iter().map(|&b| Literal::Bool(b)).collect(),
            )
            .unwrap(),
        );
        let lit = |d: &[i64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: vec![n as u32],
                    },
                    d.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            )
        };
        let dense_t = Value::vector_i64(&t).unwrap();
        let dense_f = Value::vector_i64(&f).unwrap();
        assert!(
            dense_t
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_some()
        );

        let dense = eval_select(Primitive::Select, &[cond.clone(), dense_t, dense_f]).unwrap();
        let generic = eval_select(Primitive::Select, &[cond, lit(&t), lit(&f)]).unwrap();
        let ints = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        assert_eq!(dense.as_tensor().unwrap().dtype, DType::I64);
        assert_eq!(ints(&dense), ints(&generic));
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

    #[test]
    fn integer_pow_i64_exact_and_wrapping() {
        // lax.integer_pow on integers is exact wrapping arithmetic, not an f64
        // round-trip. 3^39 exceeds 2^53, where the old `(x as f64).powi(e) as i64`
        // lost precision.
        let p = |e: &str| {
            let mut m = BTreeMap::new();
            m.insert("exponent".to_owned(), e.to_owned());
            m
        };
        let r = eval_integer_pow(Primitive::IntegerPow, &[s_i64(3)], &p("39")).unwrap();
        assert_eq!(r.as_i64_scalar().unwrap(), 4_052_555_153_018_976_267);
        assert_eq!(r.as_i64_scalar().unwrap(), 3i64.wrapping_pow(39));

        // 2^63 overflows i64 → two's-complement wrap (XLA), not saturation.
        let r = eval_integer_pow(Primitive::IntegerPow, &[s_i64(2)], &p("63")).unwrap();
        assert_eq!(r.as_i64_scalar().unwrap(), i64::MIN);

        // Small powers stay exact.
        let r = eval_integer_pow(Primitive::IntegerPow, &[s_i64(7)], &p("2")).unwrap();
        assert_eq!(r.as_i64_scalar().unwrap(), 49);
        let r = eval_integer_pow(Primitive::IntegerPow, &[s_i64(-3)], &p("3")).unwrap();
        assert_eq!(r.as_i64_scalar().unwrap(), -27);
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
    fn erfc_approx_far_tail_accurate() {
        // The erfc tail (|x| ≥ 3.5) is now a constant-free Laplace continued fraction,
        // accurate to ~1e-14 RELATIVE across the whole tail. Two distinct prior gaps are
        // closed: (a) |x| ≥ 6 returned exactly 0 (erf_approx saturates to ±1, so the old
        // dispatch 1 − erf gave 0); (b) the [3.5, 6) asymptotic floored at ~√ε relative
        // (erfc(3.5) was only ~3.5e-6 accurate). All match scipy to rel < 1e-12.
        // Mid-range [3.5, 6]: the constant-free continued fraction matches scipy to
        // ~1e-13 RELATIVE — the asymptotic series it replaced floored at ~√ε (erfc(3.5)
        // was only ~3.5e-6 accurate). These reference values are cross-validated (the CF,
        // an independent method, agrees to < 1e-12).
        let mid = [
            (3.5_f64, 7.430_983_723_414_312e-7),
            (4.0_f64, 1.541_725_790_028_002e-8),
            (4.5_f64, 1.966_160_441_542_887e-10),
            (5.0_f64, 1.537_459_794_428_035e-12),
            (6.0_f64, 2.151_973_671_249_891_4e-17),
        ];
        for (x, expected) in mid {
            let got = erfc_approx(x);
            let rel = (got - expected).abs() / expected.abs();
            assert!(
                rel < 1e-12,
                "erfc({x}) = {got:e}, expected {expected:e}, rel err {rel:e}"
            );
        }
        // Far tail: the old dispatch 1 − erf_approx returned exactly 0 for |x| ≥ 6. Now
        // nonzero and order-correct (loose tol: 8 sig figs — references are uncertain in
        // the 11th digit, and the win here is "nonzero", proven last tick).
        for (x, expected) in [(8.0_f64, 1.1224e-29), (10.0_f64, 2.0885e-45)] {
            let got = erfc_approx(x);
            assert!(got > 0.0, "erfc({x}) must be nonzero (was 0 in the old code)");
            assert!(
                (got - expected).abs() / expected.abs() < 1e-4,
                "erfc({x}) = {got:e}, expected ~{expected:e}"
            );
        }
        // erfc(1), erfc(2) use the exact 1 − erf path (no cancellation; erf well below 1).
        assert!((erfc_approx(1.0) - 0.157_299_207_050_285_13).abs() < 1e-13);
        assert!((erfc_approx(2.0) - 4.677_734_981_047_266e-3).abs() < 1e-14);
        // Negative tail: erfc(−x) = 2 − erfc(x) → ≈ 2 for large x.
        assert!((erfc_approx(-8.0) - (2.0 - 1.122_429_717_205_234_2e-29)).abs() < 1e-12);
        assert!((erfc_approx(0.0) - 1.0).abs() < 1e-15);
        // Small-|x| path unchanged: still exactly 1 − erf_approx (preserves the
        // metamorphic erfc==1-erf invariant tested over x ∈ [−3, 3]).
        for x in [-2.0, -0.5, 0.3, 1.5, 3.0] {
            assert_eq!(erfc_approx(x), 1.0 - erf_approx(x), "erfc==1-erf for |x|<3.5");
        }
    }

    #[test]
    fn complex_erf_large_argument_accurate() {
        // The old Maclaurin-only complex_erf returned garbage for |z|≳5 (alternating
        // terms dwarf the ~O(1) result). The asymptotic-erfc branch fixes it. Real-axis
        // values must match the known real erf; the new branch (|z|≥4) is exercised by
        // erf(4), erf(5), and the complex consistency checks.
        let cases: [((f64, f64), (f64, f64)); 4] = [
            ((3.0, 0.0), (0.999_977_909_503, 0.0)), // Maclaurin branch boundary
            ((4.0, 0.0), (0.999_999_984_583, 0.0)), // asymptotic branch
            ((5.0, 0.0), (1.0, 0.0)),               // erfc(5) ≈ 1.5e-12
            ((6.0, 0.0), (1.0, 0.0)),
        ];
        for ((re, im), (ere, eim)) in cases {
            let got = complex_erf((re, im));
            assert!(
                (got.0 - ere).abs() < 1e-9 && (got.1 - eim).abs() < 1e-9,
                "erf({re}+{im}i) = {got:?}, expected ({ere}, {eim})"
            );
        }
        // erf is odd and conjugate-symmetric — both must hold on the asymptotic branch.
        let z = (4.5, 1.2);
        let ez = complex_erf(z);
        let enz = complex_erf((-z.0, -z.1));
        assert!(
            (enz.0 + ez.0).abs() < 1e-9 && (enz.1 + ez.1).abs() < 1e-9,
            "erf(-z) = -erf(z) failed: ez={ez:?} enz={enz:?}"
        );
        let ecz = complex_erf((z.0, -z.1));
        assert!(
            (ecz.0 - ez.0).abs() < 1e-9 && (ecz.1 + ez.1).abs() < 1e-9,
            "erf(conj z) = conj(erf z) failed: ez={ez:?} ecz={ecz:?}"
        );
    }

    #[test]
    fn lgamma_parallel_bit_identical() {
        // A large dense-F64 tensor (>65_536) engages the threaded transcendental
        // path; it must equal the serial elementwise map bit-for-bit.
        let n = 70_000usize;
        let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 997) as f64 * 0.01).collect();
        let input = tensor_f64(vec![n as u32], &data);
        let parallel =
            extract_f64_vec(&eval_lgamma(Primitive::Lgamma, std::slice::from_ref(&input)).unwrap());
        let serial = extract_f64_vec(
            &eval_unary_elementwise(Primitive::Lgamma, &[input], lgamma_approx).unwrap(),
        );
        assert_eq!(parallel.len(), serial.len());
        for idx in 0..n {
            assert_eq!(
                parallel[idx].to_bits(),
                serial[idx].to_bits(),
                "mismatch at {idx}"
            );
        }
        // also spot-check vs a direct reference value
        assert_eq!(parallel[0].to_bits(), lgamma_approx(data[0]).to_bits());
    }

    /// Erf and Cbrt were routed onto the threaded transcendental path; a large
    /// dense-F64 tensor must equal the serial elementwise map bit-for-bit.
    #[test]
    fn erf_cbrt_parallel_bit_identical() {
        let n = 300_000usize; // > 1<<18 -> threaded
        let data: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let input = tensor_f64(vec![n as u32], &data);

        let erf_par = extract_f64_vec(
            &crate::eval_primitive(
                Primitive::Erf,
                std::slice::from_ref(&input),
                &BTreeMap::new(),
            )
            .unwrap(),
        );
        let erf_ser = extract_f64_vec(
            &eval_unary_elementwise(Primitive::Erf, std::slice::from_ref(&input), erf_approx)
                .unwrap(),
        );
        for idx in 0..n {
            assert_eq!(
                erf_par[idx].to_bits(),
                erf_ser[idx].to_bits(),
                "erf at {idx}"
            );
        }

        let cbrt_par = extract_f64_vec(
            &crate::eval_primitive(
                Primitive::Cbrt,
                std::slice::from_ref(&input),
                &BTreeMap::new(),
            )
            .unwrap(),
        );
        let cbrt_ser = extract_f64_vec(
            &eval_unary_elementwise(Primitive::Cbrt, std::slice::from_ref(&input), f64::cbrt)
                .unwrap(),
        );
        for idx in 0..n {
            assert_eq!(
                cbrt_par[idx].to_bits(),
                cbrt_ser[idx].to_bits(),
                "cbrt at {idx}"
            );
        }
    }

    #[test]
    fn dot_general_parallel_bit_identical() {
        // A batched matmul large enough (4·256·256·256 = 67.1M FMAs) to engage
        // the threaded output-space path must equal the textbook per-batch,
        // ascending-k reference bit-for-bit.
        let (bt, m, k, n) = (4usize, 256usize, 256usize, 256usize);
        let a_data: Vec<f64> = (0..bt * m * k)
            .map(|i| ((i % 97) as f64 * 0.013).sin())
            .collect();
        let b_data: Vec<f64> = (0..bt * k * n)
            .map(|i| ((i % 89) as f64 * 0.017).cos())
            .collect();
        let a = tensor_f64(vec![bt as u32, m as u32, k as u32], &a_data);
        let b = tensor_f64(vec![bt as u32, k as u32, n as u32], &b_data);
        let mut params = BTreeMap::new();
        params.insert("lhs_contracting_dims".to_string(), "2".to_string());
        params.insert("rhs_contracting_dims".to_string(), "1".to_string());
        params.insert("lhs_batch_dims".to_string(), "0".to_string());
        params.insert("rhs_batch_dims".to_string(), "0".to_string());
        let got = extract_f64_vec(&eval_dot_general(&[a, b], &params).unwrap());

        let mut want = vec![0.0f64; bt * m * n];
        for bb in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for l in 0..k {
                        s += a_data[(bb * m + i) * k + l] * b_data[(bb * k + l) * n + j];
                    }
                    want[(bb * m + i) * n + j] = s;
                }
            }
        }
        assert_eq!(got.len(), want.len());
        for idx in 0..want.len() {
            assert_eq!(got[idx].to_bits(), want[idx].to_bits(), "mismatch at {idx}");
        }
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
