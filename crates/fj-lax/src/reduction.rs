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

fn parse_reduction_axes(
    primitive: Primitive,
    raw_axes: &str,
    rank: usize,
) -> Result<Vec<usize>, EvalError> {
    let mut axes = Vec::new();
    for raw_axis in raw_axes.split(',') {
        let trimmed = raw_axis.trim();
        let axis = trimmed.parse::<i64>().map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid axis value: {trimmed}"),
        })?;
        let normalized = if axis < 0 { rank as i64 + axis } else { axis };
        if normalized < 0 || normalized >= rank as i64 {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("axis {axis} out of bounds for rank {rank}"),
            });
        }

        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("duplicate value in axes: {axis}"),
            });
        }
        axes.push(normalized);
    }
    Ok(axes)
}

fn parse_bool_param(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
    key: &str,
    default: bool,
) -> Result<bool, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };
    let trimmed = raw.trim();
    if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") || trimmed.eq_ignore_ascii_case("yes")
    {
        Ok(true)
    } else if trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
    {
        Ok(false)
    } else {
        Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid boolean parameter {key}={raw:?}"),
        })
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

    // If no axes param, fall back to full reduction. Empty list means identity.
    let axes_str = match params.get("axes") {
        Some(s) => s,
        None => return eval_reduce(primitive, inputs, int_init, float_init, int_op, float_op),
    };
    if axes_str.trim().is_empty() {
        return Ok(inputs[0].clone());
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            let mut axes_sorted = parse_reduction_axes(primitive, axes_str, rank)?;
            axes_sorted.sort_unstable();

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

            // Total number of output elements
            let out_count = checked_shape_element_count(primitive, "reduction output", &out_dims)?;
            if out_count == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    vec![],
                )?));
            }

            // For each output element, iterate over the reduced axes and accumulate
            let kept_axes: Vec<usize> = (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
            let strides = checked_strides(primitive, "reduction input", &tensor.shape.dims)?;

            if is_complex {
                if primitive != Primitive::ReduceSum {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "complex reduction is currently supported only for reduce_sum"
                            .to_owned(),
                    });
                }

                let mut result_re = try_filled_vec(
                    primitive,
                    "reduction real accumulator",
                    out_count,
                    float_init,
                )?;
                let mut result_im = try_filled_vec(
                    primitive,
                    "reduction imaginary accumulator",
                    out_count,
                    float_init,
                )?;
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    let multi = flat_to_multi(flat_idx, &strides);
                    let out_idx = multi_to_out_flat(
                        primitive,
                        "reduction output",
                        &multi,
                        &kept_axes,
                        &out_dims,
                    )?;
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
                let mut result = try_filled_vec(
                    primitive,
                    "reduction integer accumulator",
                    out_count,
                    int_init,
                )?;
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    // Compute multi-index from flat index
                    let multi = flat_to_multi(flat_idx, &strides);
                    // Compute output flat index from kept dimensions
                    let out_idx = multi_to_out_flat(
                        primitive,
                        "reduction output",
                        &multi,
                        &kept_axes,
                        &out_dims,
                    )?;
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
                let mut result = try_filled_vec(
                    primitive,
                    "reduction float accumulator",
                    out_count,
                    float_init,
                )?;
                let total = tensor.elements.len();
                for flat_idx in 0..total {
                    let multi = flat_to_multi(flat_idx, &strides);
                    let out_idx = multi_to_out_flat(
                        primitive,
                        "reduction output",
                        &multi,
                        &kept_axes,
                        &out_dims,
                    )?;
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

/// Axis-aware bitwise reduction over bool/i64 tensors.
///
/// If `axes` is omitted, performs a full reduction to scalar.
pub(crate) fn eval_reduce_bitwise_axes(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
    int_init: i64,
    bool_init: bool,
    int_op: impl Fn(i64, i64) -> i64,
    bool_op: impl Fn(bool, bool) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let axes_param = params.get("axes");
    if axes_param.is_some_and(|raw_axes| raw_axes.trim().is_empty()) {
        return Ok(inputs[0].clone());
    }
    let reduce_all = axes_param.is_none();

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let mut axes_sorted = if reduce_all {
                Vec::new()
            } else {
                let raw_axes = params.get("axes").ok_or(EvalError::Unsupported {
                    primitive,
                    detail: "missing axes parameter".to_owned(),
                })?;
                parse_reduction_axes(primitive, raw_axes, rank)?
            };
            axes_sorted.sort_unstable();

            let reduce_all_axes = reduce_all || axes_sorted.len() == rank;

            match tensor.dtype {
                DType::Bool => {
                    if reduce_all_axes {
                        let mut acc = bool_init;
                        for literal in &tensor.elements {
                            let val = match literal {
                                Literal::Bool(v) => *v,
                                _ => {
                                    return Err(EvalError::TypeMismatch {
                                        primitive,
                                        detail: "expected bool tensor",
                                    });
                                }
                            };
                            acc = bool_op(acc, val);
                        }
                        return Ok(Value::scalar_bool(acc));
                    }

                    let out_dims: Vec<u32> = tensor
                        .shape
                        .dims
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes_sorted.contains(i))
                        .map(|(_, d)| *d)
                        .collect();
                    let out_count = checked_shape_element_count(
                        primitive,
                        "bitwise reduction output",
                        &out_dims,
                    )?;
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::Bool,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let strides =
                        checked_strides(primitive, "bitwise reduction input", &tensor.shape.dims)?;
                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = try_filled_vec(
                        primitive,
                        "bitwise reduction bool accumulator",
                        out_count,
                        bool_init,
                    )?;
                    for flat_idx in 0..tensor.elements.len() {
                        let multi = flat_to_multi(flat_idx, &strides);
                        let out_idx = multi_to_out_flat(
                            primitive,
                            "bitwise reduction output",
                            &multi,
                            &kept_axes,
                            &out_dims,
                        )?;
                        let val = match tensor.elements[flat_idx] {
                            Literal::Bool(v) => v,
                            _ => {
                                return Err(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected bool tensor",
                                });
                            }
                        };
                        result[out_idx] = bool_op(result[out_idx], val);
                    }

                    let elements = result.into_iter().map(Literal::Bool).collect();
                    Ok(Value::Tensor(TensorValue::new(
                        DType::Bool,
                        Shape { dims: out_dims },
                        elements,
                    )?))
                }
                DType::I64 => {
                    if reduce_all_axes {
                        let mut acc = int_init;
                        for literal in &tensor.elements {
                            let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                                primitive,
                                detail: "expected i64 tensor",
                            })?;
                            acc = int_op(acc, val);
                        }
                        return Ok(Value::scalar_i64(acc));
                    }

                    let out_dims: Vec<u32> = tensor
                        .shape
                        .dims
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes_sorted.contains(i))
                        .map(|(_, d)| *d)
                        .collect();
                    let out_count = checked_shape_element_count(
                        primitive,
                        "bitwise reduction output",
                        &out_dims,
                    )?;
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::I64,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let strides =
                        checked_strides(primitive, "bitwise reduction input", &tensor.shape.dims)?;
                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = try_filled_vec(
                        primitive,
                        "bitwise reduction integer accumulator",
                        out_count,
                        int_init,
                    )?;
                    for flat_idx in 0..tensor.elements.len() {
                        let multi = flat_to_multi(flat_idx, &strides);
                        let out_idx = multi_to_out_flat(
                            primitive,
                            "bitwise reduction output",
                            &multi,
                            &kept_axes,
                            &out_dims,
                        )?;
                        let val =
                            tensor.elements[flat_idx]
                                .as_i64()
                                .ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                        result[out_idx] = int_op(result[out_idx], val);
                    }

                    let elements = result.into_iter().map(Literal::I64).collect();
                    Ok(Value::Tensor(TensorValue::new(
                        DType::I64,
                        Shape { dims: out_dims },
                        elements,
                    )?))
                }
                _ => Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitwise reduction expects bool or i64 tensor",
                }),
            }
        }
    }
}

fn checked_shape_element_count(
    primitive: Primitive,
    context: &str,
    dims: &[u32],
) -> Result<usize, EvalError> {
    if dims.contains(&0) {
        return Ok(0);
    }

    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} shape overflows usize"),
            })
    })
}

fn checked_strides(
    primitive: Primitive,
    context: &str,
    dims: &[u32],
) -> Result<Vec<usize>, EvalError> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1] as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} strides overflow usize"),
            })?;
    }
    Ok(strides)
}

fn try_filled_vec<T: Clone>(
    primitive: Primitive,
    context: &str,
    len: usize,
    value: T,
) -> Result<Vec<T>, EvalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|err| EvalError::Unsupported {
            primitive,
            detail: format!("{context} allocation failed for {len} elements: {err}"),
        })?;
    values.resize(len, value);
    Ok(values)
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

fn multi_to_out_flat(
    primitive: Primitive,
    context: &str,
    multi: &[usize],
    kept_axes: &[usize],
    out_dims: &[u32],
) -> Result<usize, EvalError> {
    let mut idx = 0_usize;
    let mut stride = 1_usize;
    for i in (0..kept_axes.len()).rev() {
        let offset =
            multi[kept_axes[i]]
                .checked_mul(stride)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("{context} flat index overflows usize"),
                })?;
        idx = idx
            .checked_add(offset)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} flat index overflows usize"),
            })?;
        stride =
            stride
                .checked_mul(out_dims[i] as usize)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("{context} stride overflows usize"),
                })?;
    }
    Ok(idx)
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

            let axis: usize = {
                let raw: i64 = params
                    .get("axis")
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                if raw < 0 {
                    (rank as i64 + raw) as usize
                } else {
                    raw as usize
                }
            };

            if axis >= rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {} out of bounds for rank {}", axis, rank),
                });
            }
            let reverse = parse_bool_param(primitive, params, "reverse", false)?;

            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

            let axis_dim = tensor.shape.dims[axis] as usize;

            let total = tensor.elements.len();
            let mut elements = tensor.elements.clone();
            if axis_dim == 0 || total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    tensor.shape.clone(),
                    elements,
                )?));
            }

            let strides = checked_strides(primitive, "cumulative input", &tensor.shape.dims)?;
            let axis_stride = strides[axis];

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
                    if reverse {
                        for i in (0..axis_dim).rev() {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_i64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                            acc = int_op(acc, val);
                            elements[flat_idx] = Literal::I64(acc);
                        }
                    } else {
                        for i in 0..axis_dim {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_i64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                            acc = int_op(acc, val);
                            elements[flat_idx] = Literal::I64(acc);
                        }
                    }
                } else {
                    let mut acc = float_init;
                    if reverse {
                        for i in (0..axis_dim).rev() {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_f64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor",
                                })?;
                            acc = float_op(acc, val);
                            elements[flat_idx] = Literal::from_f64(acc);
                        }
                    } else {
                        for i in 0..axis_dim {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_f64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor",
                                })?;
                            acc = float_op(acc, val);
                            elements[flat_idx] = Literal::from_f64(acc);
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
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
    fn v_i64(data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn empty_i64(dims: Vec<u32>) -> Value {
        Value::Tensor(TensorValue::new(DType::I64, Shape { dims }, vec![]).unwrap())
    }
    fn empty_bool(dims: Vec<u32>) -> Value {
        Value::Tensor(TensorValue::new(DType::Bool, Shape { dims }, vec![]).unwrap())
    }
    fn mat_f64(rows: u32, cols: u32, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows, cols],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn extract_f64(val: &Value) -> f64 {
        val.as_f64_scalar().unwrap()
    }
    fn extract_i64(val: &Value) -> i64 {
        val.as_i64_scalar().expect("expected i64 scalar")
    }
    fn extract_f64_vec(val: &Value) -> Vec<f64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    }

    // ── Reduce all axes ──

    #[test]
    fn reduce_sum_scalar() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[s_f64(42.0)],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_sum_vector() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_max_vector() {
        let result = eval_reduce(
            Primitive::ReduceMax,
            &[v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0])],
            i64::MIN,
            f64::NEG_INFINITY,
            |a, b| a.max(b),
            |a, b| a.max(b),
        )
        .unwrap();
        assert!((extract_f64(&result) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_min_vector() {
        let result = eval_reduce(
            Primitive::ReduceMin,
            &[v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0])],
            i64::MAX,
            f64::INFINITY,
            |a, b| a.min(b),
            |a, b| a.min(b),
        )
        .unwrap();
        assert!((extract_f64(&result) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_prod_vector() {
        let result = eval_reduce(
            Primitive::ReduceProd,
            &[v_f64(&[2.0, 3.0, 4.0])],
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 24.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_sum_integer() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[v_i64(&[10, 20, 30])],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert_eq!(extract_i64(&result), 60);
    }

    #[test]
    fn reduce_arity_error() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        );
        assert!(result.is_err());
    }

    // ── Reduce along axes ──

    #[test]
    fn reduce_axes_sum_rows() {
        // [[1, 2], [3, 4]] reduced along axis 0 (rows) → [4, 6]
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_axes_sum_cols() {
        // [[1, 2], [3, 4]] reduced along axis 1 (cols) → [3, 7]
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 3.0).abs() < 1e-12);
        assert!((vals[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_axes_rejects_duplicate_axes_after_normalization() {
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1,-1".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("duplicate canonical axes should be rejected")
        .to_string();

        assert!(
            err.contains("duplicate value in axes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bitwise_reduce_axes_rejects_duplicate_axes_after_normalization() {
        let x = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1,-1".to_owned());

        let err = eval_reduce_bitwise_axes(
            Primitive::ReduceOr,
            &[x],
            &params,
            0,
            false,
            |a, b| a | b,
            |a, b| a || b,
        )
        .expect_err("duplicate canonical bitwise axes should be rejected")
        .to_string();

        assert!(
            err.contains("duplicate value in axes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_axes_empty_kept_zero_dim_short_circuits_huge_product() {
        let huge = u32::MAX;
        let x = empty_i64(vec![1, huge, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect("zero-sized output should return before huge shape products overflow");

        let tensor = result.as_tensor().expect("tensor output");
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn reduce_axes_rejects_overflowing_kept_shape_without_panicking() {
        let huge = u32::MAX;
        let x = empty_i64(vec![0, huge, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("overflowing non-empty reduction output shape should be rejected")
        .to_string();

        assert!(
            err.contains("reduction output shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_axes_rejects_impossible_identity_output_allocation() {
        let huge = u32::MAX;
        let x = empty_i64(vec![0, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("huge identity-filled reduction output should fail with a typed error")
        .to_string();

        assert!(
            err.contains("reduction integer accumulator allocation failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bitwise_reduce_axes_rejects_overflowing_kept_shape_without_panicking() {
        let huge = u32::MAX;
        let x = empty_bool(vec![0, huge, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_bitwise_axes(
            Primitive::ReduceAnd,
            &[x],
            &params,
            0,
            true,
            |a, b| a & b,
            |a, b| a & b,
        )
        .expect_err("overflowing non-empty bitwise reduction output shape should be rejected")
        .to_string();

        assert!(
            err.contains("bitwise reduction output shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    // ── Cumulative ──

    #[test]
    fn cumsum_vector() {
        // cumsum([1, 2, 3, 4]) = [1, 3, 6, 10]
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &BTreeMap::new(),
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn cumsum_empty_huge_shape_returns_empty_before_stride_overflow() {
        let huge = u32::MAX;
        let dims = vec![0, huge, huge, huge, huge];
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "0".to_owned());

        let result = eval_cumulative(
            Primitive::Cumsum,
            &[empty_i64(dims.clone())],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect("empty cumulative output should not need huge strides");

        let tensor = result.as_tensor().expect("tensor output");
        assert_eq!(tensor.shape.dims, dims);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumsum_reverse_vector() {
        let mut params = BTreeMap::new();
        params.insert("reverse".to_owned(), "true".to_owned());
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![10.0, 9.0, 7.0, 4.0]);
    }

    #[test]
    fn cumprod_vector() {
        // cumprod([1, 2, 3, 4]) = [1, 2, 6, 24]
        let result = eval_cumulative(
            Primitive::Cumprod,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &BTreeMap::new(),
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn cumsum_empty_selected_axis_returns_empty_tensor() {
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "-1".to_owned());
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[empty_i64(vec![0])],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumprod_empty_selected_axis_returns_empty_matrix() {
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "1".to_owned());
        let result = eval_cumulative(
            Primitive::Cumprod,
            &[empty_i64(vec![2, 0])],
            &params,
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumsum_scalar() {
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[s_f64(7.0)],
            &BTreeMap::new(),
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 7.0).abs() < 1e-12);
    }
}
