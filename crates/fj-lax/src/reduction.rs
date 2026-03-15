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

    let reduce_all = !matches!(params.get("axes"), Some(s) if !s.trim().is_empty());

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
                raw_axes
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| EvalError::Unsupported {
                                primitive,
                                detail: format!("invalid axis value: {}", s.trim()),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };

            for &axis in &axes_sorted {
                if axis >= rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("axis {} out of bounds for rank {}", axis, rank),
                    });
                }
            }
            axes_sorted.sort_unstable();
            axes_sorted.dedup();

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
                    let out_count: usize = out_dims.iter().map(|d| *d as usize).product();
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::Bool,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let strides = compute_strides(&tensor.shape.dims);
                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = vec![bool_init; out_count];
                    for flat_idx in 0..tensor.elements.len() {
                        let multi = flat_to_multi(flat_idx, &strides, &tensor.shape.dims);
                        let out_idx = multi_to_out_flat(&multi, &kept_axes, &out_dims);
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
                    let out_count: usize = out_dims.iter().map(|d| *d as usize).product();
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::I64,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let strides = compute_strides(&tensor.shape.dims);
                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = vec![int_init; out_count];
                    for flat_idx in 0..tensor.elements.len() {
                        let multi = flat_to_multi(flat_idx, &strides, &tensor.shape.dims);
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
        match val {
            Value::Scalar(Literal::I64(v)) => *v,
            _ => panic!("expected i64 scalar"),
        }
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
