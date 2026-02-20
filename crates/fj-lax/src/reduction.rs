#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;

/// Generic reduction: reduces elements of a tensor along specified axes (or all axes).
pub(crate) fn eval_reduce(
    primitive: Primitive,
    inputs: &[Value],
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            // For backward compat: no axes param means reduce all
            // Axis-specific reduction is needed for partial reduction
            let rank = tensor.shape.rank();

            // Full reduction (all elements to scalar)
            if rank == 0 {
                return Ok(inputs[0].clone());
            }

            let is_integral = tensor.elements.iter().all(|literal| literal.is_integral());

            // Full reduction: flatten to scalar
            if is_integral {
                let mut acc = int_init;
                for literal in &tensor.elements {
                    let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected i64 tensor",
                    })?;
                    acc = int_op(acc, val);
                }
                Ok(Value::scalar_i64(acc))
            } else {
                let mut acc = float_init;
                for literal in &tensor.elements {
                    let val = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric tensor",
                    })?;
                    acc = float_op(acc, val);
                }
                Ok(Value::scalar_f64(acc))
            }
        }
    }
}

/// Axis-aware reduction: reduces tensor along specified axes, producing a tensor output.
pub(crate) fn eval_reduce_axes(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    // If no axes param, fall back to full reduction
    let axes_str = match params.get("axes") {
        Some(s) if !s.trim().is_empty() => s,
        _ => return eval_reduce(primitive, inputs, int_init, float_init, int_op, float_op),
    };

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            let axes: Vec<usize> = axes_str
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<usize>()
                        .map_err(|_| EvalError::Unsupported {
                            primitive,
                            detail: format!("invalid axis value: {}", s.trim()),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Validate axes
            for &axis in &axes {
                if axis >= rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("axis {} out of bounds for rank {}", axis, rank),
                    });
                }
            }

            let mut axes_sorted = axes.clone();
            axes_sorted.sort_unstable();
            axes_sorted.dedup();

            // If reducing all axes, just do full reduction
            if axes_sorted.len() == rank {
                return eval_reduce(primitive, inputs, int_init, float_init, int_op, float_op);
            }

            // Compute output shape (remove reduced axes)
            let out_dims: Vec<u32> = tensor
                .shape
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes_sorted.contains(i))
                .map(|(_, d)| *d)
                .collect();

            let is_integral = tensor.elements.iter().all(|literal| literal.is_integral());

            // Compute strides for the input tensor (row-major)
            let strides = compute_strides(&tensor.shape.dims);

            // Total number of output elements
            let out_count: usize = out_dims.iter().map(|d| *d as usize).product();
            if out_count == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    vec![],
                )?));
            }

            // For each output element, iterate over the reduced axes and accumulate
            let kept_axes: Vec<usize> = (0..rank).filter(|i| !axes_sorted.contains(i)).collect();

            if is_integral {
                let mut result = vec![int_init; out_count];
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    // Compute multi-index from flat index
                    let multi = flat_to_multi(flat_idx, &strides, &tensor.shape.dims);
                    // Compute output flat index from kept dimensions
                    let out_idx = multi_to_out_flat(&multi, &kept_axes, &out_dims);
                    let val =
                        tensor.elements[flat_idx]
                            .as_i64()
                            .ok_or(EvalError::TypeMismatch {
                                primitive,
                                detail: "expected i64 tensor",
                            })?;
                    result[out_idx] = int_op(result[out_idx], val);
                }
                let elements: Vec<Literal> = result.into_iter().map(Literal::I64).collect();
                Ok(Value::Tensor(TensorValue::new(
                    DType::I64,
                    Shape { dims: out_dims },
                    elements,
                )?))
            } else {
                let mut result = vec![float_init; out_count];
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    let multi = flat_to_multi(flat_idx, &strides, &tensor.shape.dims);
                    let out_idx = multi_to_out_flat(&multi, &kept_axes, &out_dims);
                    let val =
                        tensor.elements[flat_idx]
                            .as_f64()
                            .ok_or(EvalError::TypeMismatch {
                                primitive,
                                detail: "expected numeric tensor",
                            })?;
                    result[out_idx] = float_op(result[out_idx], val);
                }
                let elements: Vec<Literal> = result.into_iter().map(Literal::from_f64).collect();
                Ok(Value::Tensor(TensorValue::new(
                    DType::F64,
                    Shape { dims: out_dims },
                    elements,
                )?))
            }
        }
    }
}

fn compute_strides(dims: &[u32]) -> Vec<usize> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn flat_to_multi(flat: usize, strides: &[usize], _dims: &[u32]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for &stride in strides {
        multi.push(remainder / stride);
        remainder %= stride;
    }
    multi
}

fn multi_to_out_flat(multi: &[usize], kept_axes: &[usize], out_dims: &[u32]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..kept_axes.len()).rev() {
        idx += multi[kept_axes[i]] * stride;
        stride *= out_dims[i] as usize;
    }
    idx
}
