#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::{binary_literal_op, infer_dtype, promote_dtype};

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

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs, *rhs, primitive, &int_op, &float_op,
        )?)),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape == rhs.shape {
                // Same shape: elementwise
                let elements = lhs
                    .elements
                    .iter()
                    .copied()
                    .zip(rhs.elements.iter().copied())
                    .map(|(left, right)| {
                        binary_literal_op(left, right, primitive, &int_op, &float_op)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

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
            let elements = rhs
                .elements
                .iter()
                .copied()
                .map(|right| binary_literal_op(*lhs, right, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let lhs_dtype = match lhs {
                Literal::I64(_) => DType::I64,
                Literal::F64Bits(_) => DType::F64,
                Literal::Bool(_) => DType::Bool,
            };
            let dtype = promote_dtype(lhs_dtype, rhs.dtype);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let elements = lhs
                .elements
                .iter()
                .copied()
                .map(|left| binary_literal_op(left, *rhs, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let rhs_dtype = match rhs {
                Literal::I64(_) => DType::I64,
                Literal::F64Bits(_) => DType::F64,
                Literal::Bool(_) => DType::Bool,
            };
            let dtype = promote_dtype(lhs.dtype, rhs_dtype);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
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

    let mut elements = Vec::with_capacity(out_count);
    for flat_idx in 0..out_count {
        // Convert output flat index to multi-index
        let multi = flat_to_multi(flat_idx, &out_strides);
        // Map to input indices
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

fn flat_to_multi(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for &stride in strides {
        multi.push(remainder / stride);
        remainder %= stride;
    }
    multi
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
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric scalar",
            })?;
            Ok(Value::scalar_f64(op(value)))
        }
        Value::Tensor(tensor) => {
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| {
                    literal.as_f64().map(&op).map(Literal::from_f64).ok_or(
                        EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor elements",
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Value::Tensor(TensorValue::new(
                DType::F64,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Unary elementwise that preserves integer types (for neg, abs).
#[inline]
pub(crate) fn eval_unary_int_or_float(
    primitive: Primitive,
    inputs: &[Value],
    int_op: impl Fn(i64) -> i64,
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
            Literal::F64Bits(bits) => Ok(Value::scalar_f64(float_op(f64::from_bits(bits)))),
            Literal::Bool(_) => Err(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric scalar, got bool",
            }),
        },
        Value::Tensor(tensor) => {
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| match literal {
                    Literal::I64(v) => Ok(Literal::I64(int_op(v))),
                    Literal::F64Bits(bits) => Ok(Literal::from_f64(float_op(f64::from_bits(bits)))),
                    Literal::Bool(_) => Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric tensor elements, got bool",
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;

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
            let c = match cond {
                Literal::Bool(b) => *b,
                Literal::I64(v) => *v != 0,
                Literal::F64Bits(bits) => f64::from_bits(*bits) != 0.0,
            };
            Ok(Value::Scalar(if c { *on_true } else { *on_false }))
        }
        (Value::Tensor(cond), Value::Tensor(on_true), Value::Tensor(on_false)) => {
            if cond.shape != on_true.shape || cond.shape != on_false.shape {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "select requires all inputs to have the same shape".to_owned(),
                });
            }
            let elements: Vec<Literal> = cond
                .elements
                .iter()
                .zip(on_true.elements.iter())
                .zip(on_false.elements.iter())
                .map(|((c, t), f)| {
                    let flag = match c {
                        Literal::Bool(b) => *b,
                        Literal::I64(v) => *v != 0,
                        Literal::F64Bits(bits) => f64::from_bits(*bits) != 0.0,
                    };
                    if flag { *t } else { *f }
                })
                .collect();
            let dtype = promote_dtype(on_true.dtype, on_false.dtype);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                cond.shape.clone(),
                elements,
            )?))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "select requires matching scalar/tensor kinds".to_owned(),
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

/// Dot product: scalar-scalar, vector-vector.
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
        (Value::Scalar(lhs), Value::Scalar(rhs)) => Ok(Value::Scalar(binary_literal_op(
            *lhs,
            *rhs,
            primitive,
            &|a, b| a * b,
            &|a, b| a * b,
        )?)),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.rank() != 1 || rhs.rank() != 1 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "dot currently supports only rank-1 tensors".to_owned(),
                });
            }
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }

            if lhs.elements.iter().all(|literal| literal.is_integral())
                && rhs.elements.iter().all(|literal| literal.is_integral())
            {
                let mut sum = 0_i64;
                for (left, right) in lhs.elements.iter().zip(rhs.elements.iter()) {
                    let left_i = left.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "integral dot expected i64 elements",
                    })?;
                    let right_i = right.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "integral dot expected i64 elements",
                    })?;
                    sum += left_i * right_i;
                }
                return Ok(Value::scalar_i64(sum));
            }

            let mut sum = 0.0_f64;
            for (left, right) in lhs.elements.iter().zip(rhs.elements.iter()) {
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
            Ok(Value::scalar_f64(sum))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "dot expects either two scalars or two vectors".to_owned(),
        }),
    }
}
