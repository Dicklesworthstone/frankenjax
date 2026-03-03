#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;

fn literal_to_complex_parts(
    primitive: Primitive,
    literal: Literal,
) -> Result<(f64, f64), EvalError> {
    match literal {
        Literal::Complex64Bits(re, im) => {
            Ok((f32::from_bits(re) as f64, f32::from_bits(im) as f64))
        }
        Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "expected complex tensor elements",
        }),
    }
}

fn complex_literal_from_parts(dtype: DType, re: f64, im: f64) -> Literal {
    match dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        DType::Complex128 => Literal::from_complex128(re, im),
        _ => Literal::from_complex128(re, im),
    }
}

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
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) {
                if primitive != Primitive::ReduceSum {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "complex reduction is currently supported only for reduce_sum"
                            .to_owned(),
                    });
                }

                let mut re_acc = 0.0_f64;
                let mut im_acc = 0.0_f64;
                for literal in &tensor.elements {
                    let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                    re_acc += re;
                    im_acc += im;
                }
                return Ok(Value::Scalar(complex_literal_from_parts(
                    tensor.dtype,
                    re_acc,
                    im_acc,
                )));
            }

            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

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

            let is_complex = matches!(tensor.dtype, DType::Complex64 | DType::Complex128);
            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

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

            if is_complex {
                if primitive != Primitive::ReduceSum {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "complex reduction is currently supported only for reduce_sum"
                            .to_owned(),
                    });
                }

                let mut result_re = vec![float_init; out_count];
                let mut result_im = vec![float_init; out_count];
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    let multi = flat_to_multi(flat_idx, &strides, &tensor.shape.dims);
                    let out_idx = multi_to_out_flat(&multi, &kept_axes, &out_dims);
                    let (re, im) = literal_to_complex_parts(primitive, tensor.elements[flat_idx])?;
                    result_re[out_idx] = float_op(result_re[out_idx], re);
                    result_im[out_idx] = float_op(result_im[out_idx], im);
                }

                let elements: Vec<Literal> = result_re
                    .into_iter()
                    .zip(result_im)
                    .map(|(re, im)| complex_literal_from_parts(tensor.dtype, re, im))
                    .collect();

                Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    elements,
                )?))
            } else if is_integral {
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

/// Cumulative scan along a specified axis. Output shape matches input.
/// If no axis param, defaults to axis 0.
pub(crate) fn eval_cumulative(
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

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let axis: usize = params
                .get("axis")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);

            if axis >= rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {} out of bounds for rank {}", axis, rank),
                });
            }

            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

            // Compute strides
            let strides = compute_strides(&tensor.shape.dims);
            let axis_dim = tensor.shape.dims[axis] as usize;
            let axis_stride = strides[axis];

            let total = tensor.elements.len();
            let mut elements = tensor.elements.clone();

            // For each "line" along the axis, do a prefix scan
            // A line is identified by all indices except the axis index
            let outer_count = total / axis_dim;

            for outer in 0..outer_count {
                // Compute the flat index of the first element of this line (axis index = 0)
                // outer is the index in the "all-but-axis" space
                let base = {
                    let mut idx = outer;
                    let mut flat = 0_usize;
                    for ax in (0..rank).rev() {
                        if ax == axis {
                            continue;
                        }
                        let dim = tensor.shape.dims[ax] as usize;
                        flat += (idx % dim) * strides[ax];
                        idx /= dim;
                    }
                    flat
                };

                if is_integral {
                    let mut acc = int_init;
                    for i in 0..axis_dim {
                        let flat_idx = base + i * axis_stride;
                        let val = elements[flat_idx].as_i64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected i64 tensor",
                        })?;
                        acc = int_op(acc, val);
                        elements[flat_idx] = Literal::I64(acc);
                    }
                } else {
                    let mut acc = float_init;
                    for i in 0..axis_dim {
                        let flat_idx = base + i * axis_stride;
                        let val = elements[flat_idx].as_f64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor",
                        })?;
                        acc = float_op(acc, val);
                        elements[flat_idx] = Literal::from_f64(acc);
                    }
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}
