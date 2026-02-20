#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value, ValueError};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    ArityMismatch {
        primitive: Primitive,
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        primitive: Primitive,
        detail: &'static str,
    },
    ShapeMismatch {
        primitive: Primitive,
        left: Shape,
        right: Shape,
    },
    Unsupported {
        primitive: Primitive,
        detail: String,
    },
    InvalidTensor(ValueError),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch {
                primitive,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "arity mismatch for {}: expected {}, got {}",
                    primitive.as_str(),
                    expected,
                    actual
                )
            }
            Self::TypeMismatch { primitive, detail } => {
                write!(f, "type mismatch for {}: {}", primitive.as_str(), detail)
            }
            Self::ShapeMismatch {
                primitive,
                left,
                right,
            } => {
                write!(
                    f,
                    "shape mismatch for {}: left={:?} right={:?}",
                    primitive.as_str(),
                    left.dims,
                    right.dims
                )
            }
            Self::Unsupported { primitive, detail } => {
                write!(f, "unsupported {} behavior: {}", primitive.as_str(), detail)
            }
            Self::InvalidTensor(err) => write!(f, "invalid tensor: {err}"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<ValueError> for EvalError {
    fn from(value: ValueError) -> Self {
        Self::InvalidTensor(value)
    }
}

#[inline]
pub fn eval_primitive(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    match primitive {
        // Binary arithmetic
        Primitive::Add => eval_binary_elementwise(primitive, inputs, |a, b| a + b, |a, b| a + b),
        Primitive::Sub => eval_binary_elementwise(primitive, inputs, |a, b| a - b, |a, b| a - b),
        Primitive::Mul => eval_binary_elementwise(primitive, inputs, |a, b| a * b, |a, b| a * b),
        Primitive::Max => eval_binary_elementwise(primitive, inputs, |a, b| a.max(b), f64::max),
        Primitive::Min => eval_binary_elementwise(primitive, inputs, |a, b| a.min(b), f64::min),
        Primitive::Pow => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| (a as f64).powf(b as f64) as i64,
            f64::powf,
        ),
        // Unary arithmetic
        Primitive::Neg => eval_unary_int_or_float(primitive, inputs, |x| -x, |x| -x),
        Primitive::Abs => eval_unary_int_or_float(primitive, inputs, i64::abs, f64::abs),
        Primitive::Exp => eval_unary_elementwise(primitive, inputs, f64::exp),
        Primitive::Log => eval_unary_elementwise(primitive, inputs, f64::ln),
        Primitive::Sqrt => eval_unary_elementwise(primitive, inputs, f64::sqrt),
        Primitive::Rsqrt => eval_unary_elementwise(primitive, inputs, |x| 1.0 / x.sqrt()),
        Primitive::Floor => eval_unary_elementwise(primitive, inputs, f64::floor),
        Primitive::Ceil => eval_unary_elementwise(primitive, inputs, f64::ceil),
        Primitive::Round => eval_unary_elementwise(primitive, inputs, f64::round),
        // Trigonometric
        Primitive::Sin => eval_unary_elementwise(primitive, inputs, f64::sin),
        Primitive::Cos => eval_unary_elementwise(primitive, inputs, f64::cos),
        // Dot product
        Primitive::Dot => eval_dot(inputs),
        // Comparison
        Primitive::Eq => eval_comparison(primitive, inputs, |a, b| a == b, |a, b| a == b),
        Primitive::Ne => eval_comparison(primitive, inputs, |a, b| a != b, |a, b| a != b),
        Primitive::Lt => eval_comparison(primitive, inputs, |a, b| a < b, |a, b| a < b),
        Primitive::Le => eval_comparison(primitive, inputs, |a, b| a <= b, |a, b| a <= b),
        Primitive::Gt => eval_comparison(primitive, inputs, |a, b| a > b, |a, b| a > b),
        Primitive::Ge => eval_comparison(primitive, inputs, |a, b| a >= b, |a, b| a >= b),
        // Reductions
        Primitive::ReduceSum => {
            eval_reduce(primitive, inputs, 0_i64, 0.0, |a, b| a + b, |a, b| a + b)
        }
        Primitive::ReduceMax => eval_reduce(
            primitive,
            inputs,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            f64::max,
        ),
        Primitive::ReduceMin => eval_reduce(
            primitive,
            inputs,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            f64::min,
        ),
        Primitive::ReduceProd => {
            eval_reduce(primitive, inputs, 1_i64, 1.0, |a, b| a * b, |a, b| a * b)
        }
        // Shape manipulation
        Primitive::Reshape => eval_reshape(inputs, params),
        Primitive::Transpose => eval_transpose(inputs, params),
        Primitive::BroadcastInDim => eval_broadcast_in_dim(inputs, params),
        Primitive::Concatenate => eval_concatenate(inputs, params),
        Primitive::Slice => eval_slice(inputs, params),
        // Not yet implemented
        Primitive::Gather | Primitive::Scatter => Err(EvalError::Unsupported {
            primitive,
            detail: "runtime kernel not implemented yet for this primitive".to_owned(),
        }),
    }
}

#[inline]
fn eval_binary_elementwise(
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
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }

            let elements = lhs
                .elements
                .iter()
                .copied()
                .zip(rhs.elements.iter().copied())
                .map(|(left, right)| binary_literal_op(left, right, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let elements = rhs
                .elements
                .iter()
                .copied()
                .map(|right| binary_literal_op(*lhs, right, primitive, &int_op, &float_op))
                .collect::<Result<Vec<_>, _>>()?;

            let dtype = infer_dtype(&elements);
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

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

#[inline]
fn eval_unary_elementwise(
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

fn eval_dot(inputs: &[Value]) -> Result<Value, EvalError> {
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

/// Generic reduction: reduces all elements of a tensor to a scalar.
fn eval_reduce(
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
            if tensor.elements.iter().all(|literal| literal.is_integral()) {
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

// ---------------------------------------------------------------------------
// Shape manipulation kernels
// ---------------------------------------------------------------------------

/// Parse a comma-separated list of i64 values from a param string.
fn parse_i64_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<Vec<i64>, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    raw.split(',')
        .map(|s| {
            s.trim().parse::<i64>().map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid integer in param '{key}': '{s}'"),
            })
        })
        .collect()
}

/// Parse a comma-separated list of usize values from a param string.
fn parse_usize_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<Vec<usize>, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    raw.split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid usize in param '{key}': '{s}'"),
                })
        })
        .collect()
}

/// Reshape: change the shape of a tensor without changing its data.
/// Params: `new_shape` (comma-separated dims, -1 for a single inferred axis).
fn eval_reshape(inputs: &[Value], params: &BTreeMap<String, String>) -> Result<Value, EvalError> {
    let primitive = Primitive::Reshape;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let shape_spec = parse_i64_param(primitive, "new_shape", params)?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Scalar → rank-N tensor (all dims must multiply to 1)
            let mut dims = Vec::with_capacity(shape_spec.len());
            for d in &shape_spec {
                if *d == -1 {
                    dims.push(1_u32);
                } else if *d >= 0 {
                    dims.push(*d as u32);
                } else {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("invalid dim {d} in new_shape"),
                    });
                }
            }
            let total: u64 = dims.iter().map(|d| u64::from(*d)).product();
            if total != 1 {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: Shape::scalar(),
                    right: Shape { dims },
                });
            }
            Ok(Value::Tensor(TensorValue::new(
                match lit {
                    Literal::I64(_) => DType::I64,
                    Literal::F64Bits(_) => DType::F64,
                    Literal::Bool(_) => DType::Bool,
                },
                Shape { dims },
                vec![*lit],
            )?))
        }
        Value::Tensor(tensor) => {
            let elem_count = tensor.elements.len() as u64;
            let mut inferred_axis: Option<usize> = None;
            let mut known_product = 1_u64;
            let mut dims = Vec::with_capacity(shape_spec.len());

            for (idx, d) in shape_spec.iter().enumerate() {
                if *d == -1 {
                    if inferred_axis.is_some() {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "only one -1 inferred axis allowed in new_shape".into(),
                        });
                    }
                    inferred_axis = Some(idx);
                    dims.push(0_u32); // placeholder
                } else if *d >= 0 {
                    let du = *d as u32;
                    known_product *= u64::from(du);
                    dims.push(du);
                } else {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("invalid dim {d} in new_shape"),
                    });
                }
            }

            if let Some(axis) = inferred_axis {
                if known_product == 0 || elem_count % known_product != 0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "cannot infer dim: elem_count={elem_count} known_product={known_product}"
                        ),
                    });
                }
                dims[axis] = (elem_count / known_product) as u32;
            } else {
                let total: u64 = dims.iter().map(|d| u64::from(*d)).product();
                if total != elem_count {
                    return Err(EvalError::ShapeMismatch {
                        primitive,
                        left: tensor.shape.clone(),
                        right: Shape { dims },
                    });
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims },
                tensor.elements.clone(),
            )?))
        }
    }
}

/// Transpose: permute the axes of a tensor.
/// Params: `permutation` (comma-separated axis indices). If absent, reverses axes.
fn eval_transpose(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Transpose;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()), // scalar transpose is identity
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            let permutation = if params.contains_key("permutation") {
                parse_usize_param(primitive, "permutation", params)?
            } else {
                (0..rank).rev().collect()
            };

            if permutation.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "permutation length {} does not match rank {}",
                        permutation.len(),
                        rank
                    ),
                });
            }

            // Validate permutation is valid
            let mut seen = vec![false; rank];
            for &p in &permutation {
                if p >= rank || seen[p] {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("invalid permutation: {permutation:?} for rank {rank}"),
                    });
                }
                seen[p] = true;
            }

            let old_dims = &tensor.shape.dims;
            let new_dims: Vec<u32> = permutation.iter().map(|&p| old_dims[p]).collect();

            // Compute strides for the original tensor (row-major).
            let mut old_strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                old_strides[i] = old_strides[i + 1] * old_dims[i + 1] as usize;
            }

            let total = tensor.elements.len();
            let mut new_elements = vec![Literal::I64(0); total];

            for flat_idx in 0..total {
                // Convert flat index to multi-index in new layout.
                let mut remaining = flat_idx;
                let mut old_flat = 0_usize;
                for (new_axis, &perm_axis) in permutation.iter().enumerate() {
                    let new_dim = new_dims[new_axis] as usize;
                    let coord = remaining / {
                        let mut stride = 1;
                        for d in &new_dims[(new_axis + 1)..] {
                            stride *= *d as usize;
                        }
                        stride
                    };
                    remaining %= {
                        let mut stride = 1;
                        for d in &new_dims[(new_axis + 1)..] {
                            stride *= *d as usize;
                        }
                        stride
                    };
                    let _ = new_dim; // used for bounds
                    old_flat += coord * old_strides[perm_axis];
                }
                new_elements[flat_idx] = tensor.elements[old_flat];
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims: new_dims },
                new_elements,
            )?))
        }
    }
}

/// BroadcastInDim: broadcast a tensor to a larger shape.
/// Params: `shape` (target dims), `broadcast_dimensions` (mapping of input axes to output axes).
fn eval_broadcast_in_dim(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BroadcastInDim;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let target_dims = parse_i64_param(primitive, "shape", params)?;
    let target_dims: Vec<u32> = target_dims
        .into_iter()
        .map(|d| {
            if d >= 0 {
                Ok(d as u32)
            } else {
                Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid target dim {d}"),
                })
            }
        })
        .collect::<Result<_, _>>()?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Broadcast scalar to target shape.
            let total: u64 = target_dims.iter().map(|d| u64::from(*d)).product();
            let elements = vec![*lit; total as usize];
            let dtype = match lit {
                Literal::I64(_) => DType::I64,
                Literal::F64Bits(_) => DType::F64,
                Literal::Bool(_) => DType::Bool,
            };
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                Shape {
                    dims: target_dims,
                },
                elements,
            )?))
        }
        Value::Tensor(tensor) => {
            let broadcast_dims = if params.contains_key("broadcast_dimensions") {
                parse_usize_param(primitive, "broadcast_dimensions", params)?
            } else {
                // Default: map input axes to trailing output axes.
                let out_rank = target_dims.len();
                let in_rank = tensor.shape.rank();
                (out_rank - in_rank..out_rank).collect()
            };

            if broadcast_dims.len() != tensor.shape.rank() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "broadcast_dimensions length {} must equal input rank {}",
                        broadcast_dims.len(),
                        tensor.shape.rank()
                    ),
                });
            }

            let out_rank = target_dims.len();
            let total: usize = target_dims.iter().map(|d| *d as usize).product();

            // Build mapping: for each output axis, which input axis maps to it (if any).
            let mut out_to_in: Vec<Option<usize>> = vec![None; out_rank];
            for (in_axis, &out_axis) in broadcast_dims.iter().enumerate() {
                out_to_in[out_axis] = Some(in_axis);
            }

            // Compute input strides (row-major).
            let in_dims = &tensor.shape.dims;
            let mut in_strides = vec![1_usize; tensor.shape.rank()];
            for i in (0..tensor.shape.rank().saturating_sub(1)).rev() {
                in_strides[i] = in_strides[i + 1] * in_dims[i + 1] as usize;
            }

            let mut elements = Vec::with_capacity(total);
            let mut out_coords = vec![0_usize; out_rank];

            for _ in 0..total {
                // Map output coords to input index.
                let mut in_flat = 0_usize;
                for (out_axis, mapping) in out_to_in.iter().enumerate() {
                    if let Some(in_axis) = mapping {
                        let in_dim = in_dims[*in_axis] as usize;
                        // If input dim is 1, broadcast (coord maps to 0).
                        let coord = if in_dim == 1 {
                            0
                        } else {
                            out_coords[out_axis]
                        };
                        in_flat += coord * in_strides[*in_axis];
                    }
                }
                elements.push(tensor.elements[in_flat]);

                // Increment output coordinates (row-major order).
                for axis in (0..out_rank).rev() {
                    out_coords[axis] += 1;
                    if out_coords[axis] < target_dims[axis] as usize {
                        break;
                    }
                    out_coords[axis] = 0;
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape {
                    dims: target_dims,
                },
                elements,
            )?))
        }
    }
}

/// Concatenate: join multiple tensors along an axis.
/// Params: `dimension` (axis index, default 0).
fn eval_concatenate(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Concatenate;
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: 0,
        });
    }

    let axis: usize = params
        .get("dimension")
        .map(|s| {
            s.parse::<usize>().map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid dimension param: '{s}'"),
            })
        })
        .transpose()?
        .unwrap_or(0);

    // Collect all inputs as tensors. Scalars are invalid for concatenation.
    let tensors: Vec<&TensorValue> = inputs
        .iter()
        .map(|v| match v {
            Value::Tensor(t) => Ok(t),
            Value::Scalar(_) => Err(EvalError::Unsupported {
                primitive,
                detail: "cannot concatenate scalars".into(),
            }),
        })
        .collect::<Result<_, _>>()?;

    let rank = tensors[0].shape.rank();
    if axis >= rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("concat axis {axis} out of bounds for rank {rank}"),
        });
    }

    // Validate: all tensors must have the same rank and matching dims on non-concat axes.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape.rank() != rank {
            return Err(EvalError::ShapeMismatch {
                primitive,
                left: tensors[0].shape.clone(),
                right: t.shape.clone(),
            });
        }
        for (ax, (d0, di)) in tensors[0]
            .shape
            .dims
            .iter()
            .zip(t.shape.dims.iter())
            .enumerate()
        {
            if ax != axis && d0 != di {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "tensor {i} dim mismatch on axis {ax}: expected {d0}, got {di}"
                    ),
                });
            }
        }
    }

    // Build output shape.
    let mut out_dims = tensors[0].shape.dims.clone();
    let concat_size: u32 = tensors.iter().map(|t| t.shape.dims[axis]).sum();
    out_dims[axis] = concat_size;

    // For concatenation, iterate over all output coordinates.
    let total: usize = out_dims.iter().map(|d| *d as usize).product();
    let mut elements = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; rank];

    for _ in 0..total {
        // Determine which input tensor and its local coord on the concat axis.
        let mut concat_coord = out_coords[axis];
        let mut tensor_idx = 0;
        for (i, t) in tensors.iter().enumerate() {
            let dim = t.shape.dims[axis] as usize;
            if concat_coord < dim {
                tensor_idx = i;
                break;
            }
            concat_coord -= dim;
        }

        // Compute flat index into the selected tensor.
        let t = tensors[tensor_idx];
        let mut flat = 0_usize;
        let mut stride = 1_usize;
        for ax in (0..rank).rev() {
            let coord = if ax == axis {
                concat_coord
            } else {
                out_coords[ax]
            };
            flat += coord * stride;
            stride *= t.shape.dims[ax] as usize;
        }
        elements.push(t.elements[flat]);

        // Increment output coordinates.
        for ax in (0..rank).rev() {
            out_coords[ax] += 1;
            if out_coords[ax] < out_dims[ax] as usize {
                break;
            }
            out_coords[ax] = 0;
        }
    }

    let dtype = tensors[0].dtype;
    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Slice: extract a sub-tensor.
/// Params: `start_indices` and `limit_indices` (comma-separated u32 lists).
fn eval_slice(inputs: &[Value], params: &BTreeMap<String, String>) -> Result<Value, EvalError> {
    let primitive = Primitive::Slice;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "cannot slice a scalar".into(),
        }),
        Value::Tensor(tensor) => {
            let starts = parse_usize_param(primitive, "start_indices", params)?;
            let limits = parse_usize_param(primitive, "limit_indices", params)?;

            let rank = tensor.shape.rank();
            if starts.len() != rank || limits.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "start/limit rank mismatch: starts={} limits={} rank={}",
                        starts.len(),
                        limits.len(),
                        rank
                    ),
                });
            }

            // Validate and compute output dims.
            let mut out_dims = Vec::with_capacity(rank);
            for ax in 0..rank {
                let dim = tensor.shape.dims[ax] as usize;
                if starts[ax] > limits[ax] || limits[ax] > dim {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "invalid slice on axis {ax}: start={} limit={} dim={}",
                            starts[ax], limits[ax], dim
                        ),
                    });
                }
                out_dims.push((limits[ax] - starts[ax]) as u32);
            }

            // Compute input strides (row-major).
            let in_dims = &tensor.shape.dims;
            let mut in_strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                in_strides[i] = in_strides[i + 1] * in_dims[i + 1] as usize;
            }

            let total: usize = out_dims.iter().map(|d| *d as usize).product();
            let mut elements = Vec::with_capacity(total);
            let mut out_coords = vec![0_usize; rank];

            for _ in 0..total {
                // Map output coords to input coords by adding start offsets.
                let mut in_flat = 0_usize;
                for ax in 0..rank {
                    in_flat += (out_coords[ax] + starts[ax]) * in_strides[ax];
                }
                elements.push(tensor.elements[in_flat]);

                // Increment output coordinates.
                for ax in (0..rank).rev() {
                    out_coords[ax] += 1;
                    if out_coords[ax] < out_dims[ax] as usize {
                        break;
                    }
                    out_coords[ax] = 0;
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims: out_dims },
                elements,
            )?))
        }
    }
}

/// Unary elementwise that preserves integer types (for neg, abs).
#[inline]
fn eval_unary_int_or_float(
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

            let dtype = infer_dtype(&elements);
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Comparison operators: return Bool scalars/tensors.
#[inline]
fn eval_comparison(
    primitive: Primitive,
    inputs: &[Value],
    int_cmp: impl Fn(i64, i64) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            let result = compare_literals(*lhs, *rhs, primitive, &int_cmp, &float_cmp)?;
            Ok(Value::scalar_bool(result))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }
            let elements = lhs
                .elements
                .iter()
                .copied()
                .zip(rhs.elements.iter().copied())
                .map(|(l, r)| {
                    compare_literals(l, r, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let elements = rhs
                .elements
                .iter()
                .copied()
                .map(|r| {
                    compare_literals(*lhs, r, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let elements = lhs
                .elements
                .iter()
                .copied()
                .map(|l| {
                    compare_literals(l, *rhs, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

#[inline]
fn compare_literals(
    lhs: Literal,
    rhs: Literal,
    primitive: Primitive,
    int_cmp: &impl Fn(i64, i64) -> bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<bool, EvalError> {
    match (lhs, rhs) {
        (Literal::I64(a), Literal::I64(b)) => Ok(int_cmp(a, b)),
        (Literal::Bool(a), Literal::Bool(b)) => Ok(int_cmp(a as i64, b as i64)),
        (left, right) => {
            let lhs_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs for comparison",
            })?;
            let rhs_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs for comparison",
            })?;
            Ok(float_cmp(lhs_f, rhs_f))
        }
    }
}

#[inline]
fn binary_literal_op(
    lhs: Literal,
    rhs: Literal,
    primitive: Primitive,
    int_op: &impl Fn(i64, i64) -> i64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Result<Literal, EvalError> {
    match (lhs, rhs) {
        (Literal::I64(left), Literal::I64(right)) => Ok(Literal::I64(int_op(left, right))),
        (left, right) => {
            let lhs_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs",
            })?;
            let rhs_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs",
            })?;
            Ok(Literal::from_f64(float_op(lhs_f, rhs_f)))
        }
    }
}

#[inline]
fn infer_dtype(elements: &[Literal]) -> DType {
    if elements
        .iter()
        .all(|literal| matches!(literal, Literal::I64(_)))
    {
        DType::I64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Bool(_)))
    {
        DType::Bool
    } else {
        DType::F64
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalError, eval_primitive};
    use fj_core::{Primitive, Value};

    #[test]
    fn add_i64_scalars() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_i64(5)],
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn add_vector_and_scalar_broadcasts() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let out = eval_primitive(Primitive::Add, &[input, Value::scalar_i64(2)])
            .expect("broadcasted add should succeed");

        let expected = Value::vector_i64(&[3, 4, 5]).expect("vector value should build");
        assert_eq!(out, expected);
    }

    #[test]
    fn sub_i64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn sub_f64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_f64(5.5), Value::scalar_f64(2.0)],
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.5).abs() < 1e-10);
    }

    #[test]
    fn neg_i64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_i64(7)]);
        assert_eq!(out, Ok(Value::scalar_i64(-7)));
    }

    #[test]
    fn neg_f64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(3.5)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - (-3.5)).abs() < 1e-10);
    }

    #[test]
    fn abs_negative_i64() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_i64(-42)]);
        assert_eq!(out, Ok(Value::scalar_i64(42)));
    }

    #[test]
    fn abs_negative_f64() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_f64(-3.14)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.14).abs() < 1e-10);
    }

    #[test]
    fn max_i64_scalars() {
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn min_i64_scalars() {
        let out = eval_primitive(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
        );
        assert_eq!(out, Ok(Value::scalar_i64(3)));
    }

    #[test]
    fn exp_scalar() {
        let out = eval_primitive(Primitive::Exp, &[Value::scalar_f64(1.0)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar() {
        let out =
            eval_primitive(Primitive::Log, &[Value::scalar_f64(std::f64::consts::E)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_scalar() {
        let out = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(9.0)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn rsqrt_scalar() {
        let out = eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(4.0)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn floor_scalar() {
        let out = eval_primitive(Primitive::Floor, &[Value::scalar_f64(3.7)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn ceil_scalar() {
        let out = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(3.2)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn round_scalar() {
        let out = eval_primitive(Primitive::Round, &[Value::scalar_f64(3.5)]).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn pow_f64_scalars() {
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eq_i64_scalars() {
        let out = eval_primitive(Primitive::Eq, &[Value::scalar_i64(3), Value::scalar_i64(3)]);
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(Primitive::Eq, &[Value::scalar_i64(3), Value::scalar_i64(4)]);
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_i64_scalars() {
        let out = eval_primitive(Primitive::Ne, &[Value::scalar_i64(3), Value::scalar_i64(4)]);
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_i64_scalars() {
        let out = eval_primitive(Primitive::Lt, &[Value::scalar_i64(3), Value::scalar_i64(5)]);
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(Primitive::Lt, &[Value::scalar_i64(5), Value::scalar_i64(3)]);
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn le_ge_i64_scalars() {
        let out = eval_primitive(Primitive::Le, &[Value::scalar_i64(3), Value::scalar_i64(3)]);
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(Primitive::Ge, &[Value::scalar_i64(3), Value::scalar_i64(3)]);
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn gt_f64_scalars() {
        let out = eval_primitive(
            Primitive::Gt,
            &[Value::scalar_f64(3.5), Value::scalar_f64(2.0)],
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn comparison_on_vectors() {
        let lhs = Value::vector_i64(&[1, 2, 3]).unwrap();
        let rhs = Value::vector_i64(&[2, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::Lt, &[lhs, rhs]).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 3);
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true));
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false));
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false));
        } else {
            panic!("expected tensor output for vector comparison");
        }
    }

    #[test]
    fn reduce_max_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input]).unwrap();
        assert_eq!(out, Value::scalar_i64(9));
    }

    #[test]
    fn reduce_min_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input]).unwrap();
        assert_eq!(out, Value::scalar_i64(1));
    }

    #[test]
    fn reduce_prod_vector() {
        let input = Value::vector_i64(&[2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input]).unwrap();
        assert_eq!(out, Value::scalar_i64(24));
    }

    #[test]
    fn neg_vector() {
        let input = Value::vector_i64(&[1, -2, 3]).unwrap();
        let out = eval_primitive(Primitive::Neg, &[input]).unwrap();
        let expected = Value::vector_i64(&[-1, 2, -3]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_vector_works() {
        let lhs = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let rhs = Value::vector_i64(&[4, 5, 6]).expect("vector value should build");
        let out = eval_primitive(Primitive::Dot, &[lhs, rhs]).expect("dot should succeed");
        assert_eq!(out, Value::scalar_i64(32));
    }

    #[test]
    fn reduce_sum_requires_single_argument() {
        let err = eval_primitive(Primitive::ReduceSum, &[]).expect_err("should fail");
        assert_eq!(
            err,
            EvalError::ArityMismatch {
                primitive: Primitive::ReduceSum,
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn test_lax_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("lax", "add")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_lax_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }
}
