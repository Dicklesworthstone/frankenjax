#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::EvalError;
use crate::type_promotion::{binary_literal_op, promote_dtype};

/// Parse a comma-separated list of i64 values from a param string.
pub(crate) fn parse_i64_param(
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

fn parse_pad_i64_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
    rank: usize,
    missing_default: Option<i64>,
) -> Result<Vec<i64>, EvalError> {
    let Some(raw) = params.get(key) else {
        if let Some(default) = missing_default {
            return Ok(vec![default; rank]);
        }
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("missing required param '{key}'"),
        });
    };

    if raw.trim().is_empty() {
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid empty param '{key}' for rank {rank}"),
        });
    }

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
pub(crate) fn parse_usize_param(
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

fn parse_axis_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
    rank: usize,
    default: usize,
) -> Result<usize, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };

    let axis = raw
        .trim()
        .parse::<i64>()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid integer in param '{key}': '{raw}'"),
        })?;
    let normalized = if axis < 0 { rank as i64 + axis } else { axis };

    if normalized < 0 || normalized >= rank as i64 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("axis {axis} out of bounds for rank {rank}"),
        });
    }

    Ok(normalized as usize)
}

fn parse_dtype_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<DType, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    parse_dtype_name(primitive, key, raw)
}

fn parse_dtype_name(primitive: Primitive, key: &str, raw: &str) -> Result<DType, EvalError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" => Ok(DType::F16),
        "f32" | "float32" => Ok(DType::F32),
        "f64" | "float64" => Ok(DType::F64),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "complex64" => Ok(DType::Complex64),
        "complex128" => Ok(DType::Complex128),
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported dtype in '{key}': '{raw}'"),
        }),
    }
}

const fn dtype_bit_width(dtype: DType) -> u32 {
    match dtype {
        DType::BF16 | DType::F16 => 16,
        DType::F32 | DType::I32 | DType::U32 => 32,
        DType::F64 | DType::I64 | DType::U64 | DType::Complex64 => 64,
        DType::Complex128 => 128,
        DType::Bool => 8,
    }
}

fn literal_to_bytes(
    primitive: Primitive,
    dtype: DType,
    literal: Literal,
) -> Result<Vec<u8>, EvalError> {
    let bytes = match dtype {
        DType::I64 => literal
            .as_i64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as i64",
            })?
            .to_le_bytes()
            .to_vec(),
        DType::I32 => {
            let value = literal.as_i64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as i32",
            })?;
            i32::try_from(value)
                .map_err(|_| EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is out of i32 range",
                })?
                .to_le_bytes()
                .to_vec()
        }
        DType::U32 => u32::try_from(literal.as_u64().ok_or(EvalError::TypeMismatch {
            primitive,
            detail: "bitcast source literal is not representable as u32",
        })?)
        .map_err(|_| EvalError::TypeMismatch {
            primitive,
            detail: "bitcast source literal is out of u32 range",
        })?
        .to_le_bytes()
        .to_vec(),
        DType::U64 => literal
            .as_u64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as u64",
            })?
            .to_le_bytes()
            .to_vec(),
        DType::Bool => match literal {
            Literal::Bool(value) => vec![u8::from(value)],
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not a bool",
                });
            }
        },
        DType::BF16 => match literal {
            Literal::BF16Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not bf16",
                });
            }
        },
        DType::F16 => match literal {
            Literal::F16Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not f16",
                });
            }
        },
        DType::F32 => match literal {
            Literal::F32Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not f32",
                });
            }
        },
        DType::F64 => literal
            .as_f64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as f64",
            })?
            .to_bits()
            .to_le_bytes()
            .to_vec(),
        DType::Complex64 => match literal {
            Literal::Complex64Bits(re, im) => {
                let mut out = Vec::with_capacity(8);
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
                out
            }
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not complex64",
                });
            }
        },
        DType::Complex128 => match literal {
            Literal::Complex128Bits(re, im) => {
                let mut out = Vec::with_capacity(16);
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
                out
            }
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not complex128",
                });
            }
        },
    };
    Ok(bytes)
}

fn bytes_to_literal(
    primitive: Primitive,
    dtype: DType,
    bytes: &[u8],
) -> Result<Literal, EvalError> {
    match dtype {
        DType::I64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for i64".to_owned(),
            })?;
            Ok(Literal::I64(i64::from_le_bytes(array)))
        }
        DType::I32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for i32".to_owned(),
            })?;
            Ok(Literal::I64(i64::from(i32::from_le_bytes(array))))
        }
        DType::U32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for u32".to_owned(),
            })?;
            Ok(Literal::U32(u32::from_le_bytes(array)))
        }
        DType::U64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for u64".to_owned(),
            })?;
            Ok(Literal::U64(u64::from_le_bytes(array)))
        }
        DType::Bool => {
            let array = <[u8; 1]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for bool".to_owned(),
            })?;
            Ok(Literal::Bool(array[0] != 0))
        }
        DType::BF16 => {
            let array = <[u8; 2]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for bf16".to_owned(),
            })?;
            Ok(Literal::BF16Bits(u16::from_le_bytes(array)))
        }
        DType::F16 => {
            let array = <[u8; 2]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f16".to_owned(),
            })?;
            Ok(Literal::F16Bits(u16::from_le_bytes(array)))
        }
        DType::F32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f32".to_owned(),
            })?;
            Ok(Literal::F32Bits(u32::from_le_bytes(array)))
        }
        DType::F64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f64".to_owned(),
            })?;
            Ok(Literal::F64Bits(u64::from_le_bytes(array)))
        }
        DType::Complex64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for complex64".to_owned(),
            })?;
            let re = u32::from_le_bytes([array[0], array[1], array[2], array[3]]);
            let im = u32::from_le_bytes([array[4], array[5], array[6], array[7]]);
            Ok(Literal::Complex64Bits(re, im))
        }
        DType::Complex128 => {
            let array = <[u8; 16]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for complex128".to_owned(),
            })?;
            let re = u64::from_le_bytes([
                array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7],
            ]);
            let im = u64::from_le_bytes([
                array[8], array[9], array[10], array[11], array[12], array[13], array[14],
                array[15],
            ]);
            Ok(Literal::Complex128Bits(re, im))
        }
    }
}

/// Reshape: change the shape of a tensor without changing its data.
/// Params: `new_shape` (comma-separated dims, -1 for a single inferred axis).
fn resolve_reshape_dims(
    primitive: Primitive,
    shape_spec: &[i64],
    elem_count: usize,
    input_shape: Shape,
) -> Result<Vec<u32>, EvalError> {
    let mut inferred_axis: Option<usize> = None;
    let mut known_product = 1_usize;
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
            dims.push(0_u32); // temporary slot filled after the inferred extent is known
        } else if *d >= 0 {
            let du = u32::try_from(*d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("new_shape dim {d} exceeds u32 range"),
            })?;
            known_product =
                known_product
                    .checked_mul(du as usize)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "reshape known dimension product overflows usize".to_owned(),
                    })?;
            dims.push(du);
        } else {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("invalid dim {d} in new_shape"),
            });
        }
    }

    if let Some(axis) = inferred_axis {
        if known_product == 0 || !elem_count.is_multiple_of(known_product) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "cannot infer dim: elem_count={elem_count} known_product={known_product}"
                ),
            });
        }
        let inferred = elem_count / known_product;
        dims[axis] = u32::try_from(inferred).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("inferred reshape dim {inferred} exceeds u32 range"),
        })?;
    } else if known_product != elem_count {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: input_shape,
            right: Shape { dims },
        });
    }

    Ok(dims)
}

pub(crate) fn eval_reshape(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
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
            let dims = resolve_reshape_dims(primitive, &shape_spec, 1, Shape::scalar())?;
            Ok(Value::Tensor(TensorValue::new(
                match lit {
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
                },
                Shape { dims },
                vec![*lit],
            )?))
        }
        Value::Tensor(tensor) => {
            let dims = resolve_reshape_dims(
                primitive,
                &shape_spec,
                tensor.elements.len(),
                tensor.shape.clone(),
            )?;

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
pub(crate) fn eval_transpose(
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

            for (flat_idx, elem) in new_elements.iter_mut().enumerate() {
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
                *elem = tensor.elements[old_flat];
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
pub(crate) fn eval_broadcast_in_dim(
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

    let target_dims = parse_broadcast_target_dims(primitive, params)?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Broadcast scalar to target shape.
            let total = checked_shape_element_count(primitive, "broadcast_in_dim", &target_dims)?;
            let elements = vec![*lit; total];
            let dtype = match lit {
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
            };
            Ok(Value::Tensor(TensorValue::new(
                dtype,
                Shape { dims: target_dims },
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
                if in_rank > out_rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("input rank {} exceeds output rank {}", in_rank, out_rank),
                    });
                }
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
            let mut seen = BTreeSet::new();
            for (in_axis, &out_axis) in broadcast_dims.iter().enumerate() {
                if out_axis >= out_rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "broadcast_dimensions axis {} out of range for output rank {}",
                            out_axis, out_rank
                        ),
                    });
                }
                if !seen.insert(out_axis) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "broadcast_dimensions must be unique".to_owned(),
                    });
                }
                let in_dim = tensor.shape.dims[in_axis];
                let target_dim = target_dims[out_axis];
                if in_dim != 1 && in_dim != target_dim {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "cannot broadcast input dim {} into target dim {} at axis {}",
                            in_dim, target_dim, out_axis
                        ),
                    });
                }
            }
            let total = checked_shape_element_count(primitive, "broadcast_in_dim", &target_dims)?;

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
                        let coord = if in_dim == 1 { 0 } else { out_coords[out_axis] };
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
                Shape { dims: target_dims },
                elements,
            )?))
        }
    }
}

fn parse_broadcast_target_dims(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<Vec<u32>, EvalError> {
    parse_i64_param(primitive, "shape", params)?
        .into_iter()
        .map(|d| {
            if d < 0 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid target dim {d}"),
                });
            }
            u32::try_from(d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("target dim {d} exceeds u32 range"),
            })
        })
        .collect()
}

fn checked_shape_element_count(
    primitive: Primitive,
    op_name: &str,
    dims: &[u32],
) -> Result<usize, EvalError> {
    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{op_name} shape overflows usize"),
            })
    })
}

fn checked_row_major_strides(
    primitive: Primitive,
    op_name: &str,
    dims: &[u32],
) -> Result<Vec<usize>, EvalError> {
    let rank = dims.len();
    let mut strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1] as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{op_name} input strides overflow usize"),
            })?;
    }
    Ok(strides)
}

/// Concatenate: join multiple tensors along an axis.
/// Params: `dimension` (axis index, default 0).
pub(crate) fn eval_concatenate(
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
    let concat_size = tensors.iter().try_fold(0_u32, |acc, tensor| {
        acc.checked_add(tensor.shape.dims[axis])
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "concatenate axis size overflows u32".to_owned(),
            })
    })?;
    out_dims[axis] = concat_size;

    // For concatenation, iterate over all output coordinates.
    let total = checked_shape_element_count(primitive, "concatenate", &out_dims)?;
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

/// Pad: add low/high edge padding and interior padding between elements.
///
/// Inputs: `[operand, pad_value]`
/// Params:
/// - `padding_low`: comma-separated integers (one per axis; negative crops low edge)
/// - `padding_high`: comma-separated integers (one per axis; negative crops high edge)
/// - `padding_interior`: comma-separated non-negative integers (one per axis, optional; defaults to 0)
pub(crate) fn eval_pad(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Pad;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let operand_dtype = inputs[0].dtype();
    let pad_value_dtype = inputs[1].dtype();
    if pad_value_dtype != operand_dtype {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "pad value dtype {:?} must match operand dtype {:?}",
                pad_value_dtype, operand_dtype
            ),
        });
    }

    let pad_literal = match &inputs[1] {
        Value::Scalar(lit) => *lit,
        Value::Tensor(tensor) if tensor.elements.len() == 1 => tensor.elements[0],
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "pad value must be a scalar literal".into(),
            });
        }
    };

    let Value::Tensor(operand) = &inputs[0] else {
        let rank = 0;
        let lows_raw = parse_pad_i64_param(primitive, "padding_low", params, rank, None)?;
        let highs_raw = parse_pad_i64_param(primitive, "padding_high", params, rank, None)?;
        let interiors_raw =
            parse_pad_i64_param(primitive, "padding_interior", params, rank, Some(0))?;
        if !lows_raw.is_empty() || !highs_raw.is_empty() || !interiors_raw.is_empty() {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "padding params must match rank {rank}: low={} high={} interior={}",
                    lows_raw.len(),
                    highs_raw.len(),
                    interiors_raw.len()
                ),
            });
        }
        return Ok(inputs[0].clone());
    };

    let rank = operand.shape.rank();
    let lows_raw = parse_pad_i64_param(primitive, "padding_low", params, rank, None)?;
    let highs_raw = parse_pad_i64_param(primitive, "padding_high", params, rank, None)?;
    let interiors_raw =
        parse_pad_i64_param(primitive, "padding_interior", params, rank, Some(0_i64))?;

    if lows_raw.len() != rank || highs_raw.len() != rank || interiors_raw.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "padding params must match rank {rank}: low={} high={} interior={}",
                lows_raw.len(),
                highs_raw.len(),
                interiors_raw.len()
            ),
        });
    }

    let mut lows = Vec::with_capacity(rank);
    let mut interiors = Vec::with_capacity(rank);
    let mut out_dims = Vec::with_capacity(rank);

    for ax in 0..rank {
        let low = lows_raw[ax];
        let high = highs_raw[ax];
        let interior = interiors_raw[ax];
        if interior < 0 {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("interior padding must be non-negative on axis {ax}: {interior}"),
            });
        }

        let dim = i64::from(operand.shape.dims[ax]);
        let interior_span = if dim == 0 {
            0
        } else {
            (dim - 1)
                .checked_mul(interior)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("padded dimension overflow on axis {ax}"),
                })?
        };
        let out_dim = low
            .checked_add(dim)
            .and_then(|value| value.checked_add(interior_span))
            .and_then(|value| value.checked_add(high))
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("padded dimension overflow on axis {ax}"),
            })?;
        if out_dim < 0 || out_dim > i64::from(u32::MAX) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("padded dimension overflow on axis {ax}: {out_dim}"),
            });
        }

        lows.push(low);
        interiors.push(interior as usize);
        out_dims.push(out_dim as u32);
    }

    let out_total = checked_shape_element_count(primitive, "pad", &out_dims)?;
    let mut out_elements = vec![pad_literal; out_total];

    if rank == 0 {
        if !out_elements.is_empty() {
            out_elements[0] = operand.elements[0];
        }
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            out_elements,
        )?));
    }
    if operand.elements.is_empty() {
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            out_elements,
        )?));
    }

    // Row-major strides.
    let in_dims = &operand.shape.dims;
    let mut out_strides = vec![1_usize; rank];
    for ax in (0..rank.saturating_sub(1)).rev() {
        out_strides[ax] = out_strides[ax + 1] * out_dims[ax + 1] as usize;
    }

    let mut in_coords = vec![0_usize; rank];
    for element in &operand.elements {
        let mut out_flat = 0_usize;
        let mut place_element = true;
        for ax in 0..rank {
            let coord = lows[ax] + (in_coords[ax] * (interiors[ax] + 1)) as i64;
            if coord < 0 || coord >= i64::from(out_dims[ax]) {
                place_element = false;
                break;
            }
            out_flat += coord as usize * out_strides[ax];
        }
        if place_element {
            out_elements[out_flat] = *element;
        }

        for ax in (0..rank).rev() {
            in_coords[ax] += 1;
            if in_coords[ax] < in_dims[ax] as usize {
                break;
            }
            in_coords[ax] = 0;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        Shape { dims: out_dims },
        out_elements,
    )?))
}

/// Slice: extract a sub-tensor.
/// Params: `start_indices` and `limit_indices` (comma-separated u32 lists).
pub(crate) fn eval_slice(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
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
            let slice_strides = if params.contains_key("strides") {
                parse_usize_param(primitive, "strides", params)?
            } else {
                vec![1; rank]
            };
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
            if slice_strides.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "strides rank mismatch: strides={} rank={}",
                        slice_strides.len(),
                        rank
                    ),
                });
            }

            // Validate and compute output dims.
            let mut out_dims = Vec::with_capacity(rank);
            for ax in 0..rank {
                let dim = tensor.shape.dims[ax] as usize;
                let stride = slice_strides[ax];
                if stride == 0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("slice stride on axis {ax} must be positive"),
                    });
                }
                if starts[ax] > limits[ax] || limits[ax] > dim {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "invalid slice on axis {ax}: start={} limit={} dim={}",
                            starts[ax], limits[ax], dim
                        ),
                    });
                }
                let span = limits[ax] - starts[ax];
                out_dims.push(span.div_ceil(stride) as u32);
            }

            let total = checked_shape_element_count(primitive, "slice", &out_dims)?;
            if total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    Vec::new(),
                )?));
            }

            // Compute input strides (row-major).
            let in_dims = &tensor.shape.dims;
            let in_strides = checked_row_major_strides(primitive, "slice", in_dims)?;

            let has_contiguous_trailing_slice = rank > 0
                && slice_strides.iter().all(|&stride| stride == 1)
                && starts.iter().skip(1).all(|&start| start == 0)
                && limits
                    .iter()
                    .skip(1)
                    .zip(in_dims.iter().skip(1))
                    .all(|(&limit, &dim)| limit == dim as usize);
            if has_contiguous_trailing_slice {
                let start_offset =
                    starts[0]
                        .checked_mul(in_strides[0])
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "slice start offset overflows usize".to_owned(),
                        })?;
                let end_offset =
                    start_offset
                        .checked_add(total)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "slice end offset overflows usize".to_owned(),
                        })?;
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    tensor.elements[start_offset..end_offset].to_vec(),
                )?));
            }

            let mut elements = Vec::with_capacity(total);
            let mut out_coords = vec![0_usize; rank];

            for _ in 0..total {
                // Map output coords to input coords by adding start offsets.
                let mut in_flat = 0_usize;
                for ax in 0..rank {
                    let stepped =
                        out_coords[ax]
                            .checked_mul(slice_strides[ax])
                            .ok_or_else(|| EvalError::Unsupported {
                                primitive,
                                detail: format!("slice coordinate overflows usize on axis {ax}"),
                            })?;
                    let coord =
                        starts[ax]
                            .checked_add(stepped)
                            .ok_or_else(|| EvalError::Unsupported {
                                primitive,
                                detail: format!("slice coordinate overflows usize on axis {ax}"),
                            })?;
                    let offset = coord.checked_mul(in_strides[ax]).ok_or_else(|| {
                        EvalError::Unsupported {
                            primitive,
                            detail: format!("slice offset overflows usize on axis {ax}"),
                        }
                    })?;
                    in_flat =
                        in_flat
                            .checked_add(offset)
                            .ok_or_else(|| EvalError::Unsupported {
                                primitive,
                                detail: "slice flat offset overflows usize".to_owned(),
                            })?;
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

/// Gather: index into an operand tensor using an indices tensor.
///
/// Simplified semantics (axis-0 index gather):
///   operand: tensor of any rank
///   indices: integer tensor of gather indices (into axis 0 of operand)
///   params:  `slice_sizes` — comma-separated sizes for each axis of the gathered slice
///
/// For each index i in `indices`, extracts a slice of shape `slice_sizes` starting
/// at position `[indices[i], 0, 0, ...]` from `operand`.
/// Constraint: `slice_sizes[0]` must be 1 (axis-0 indexing only).
/// Output shape: `indices.shape ++ slice_sizes[1..]` (the leading axis is replaced by
/// the indices tensor shape, remaining axes keep their slice sizes).
pub(crate) fn eval_gather(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Gather;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot gather from a scalar".into(),
            });
        }
    };

    if operand.shape.rank() == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "cannot gather from a rank-0 tensor".into(),
        });
    }

    let slice_sizes = parse_usize_param(primitive, "slice_sizes", params)?;
    if slice_sizes.len() != operand.shape.rank() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes length {} must equal operand rank {}",
                slice_sizes.len(),
                operand.shape.rank()
            ),
        });
    }

    if slice_sizes[0] != 1 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes[0] must be 1 for gather, got {}",
                slice_sizes[0]
            ),
        });
    }

    // Extract indices as flat i64 values + capture indices shape
    let (index_vals, index_shape): (Vec<usize>, Vec<u32>) = match &inputs[1] {
        Value::Scalar(lit) => (vec![lit_to_usize(lit, primitive)?], Vec::new()),
        Value::Tensor(t) => (
            t.elements
                .iter()
                .map(|lit| lit_to_usize(lit, primitive))
                .collect::<Result<_, _>>()?,
            t.shape.dims.clone(),
        ),
    };

    let num_indices = index_vals.len();
    let rank = operand.shape.rank();

    // Compute operand strides (row-major)
    let op_dims = &operand.shape.dims;

    for (ax, (&ss, &dim)) in slice_sizes.iter().zip(op_dims.iter()).enumerate() {
        if ss > dim as usize {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("slice_sizes[{ax}] = {ss} exceeds operand dimension {dim}"),
            });
        }
    }

    let mut op_strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        op_strides[i] = op_strides[i + 1] * op_dims[i + 1] as usize;
    }

    // Output shape: indices.shape ++ slice_sizes[1..]
    let mut out_dims: Vec<u32> = index_shape;
    out_dims.extend(slice_sizes.iter().skip(1).map(|&s| s as u32));

    // Number of elements per gathered slice (product of slice_sizes[1..])
    let slice_elems: usize = slice_sizes[1..].iter().product::<usize>();

    let total = num_indices * slice_elems;
    let mut elements = Vec::with_capacity(total);
    let trailing_slice_is_contiguous = slice_sizes
        .iter()
        .skip(1)
        .zip(op_dims.iter().skip(1))
        .all(|(&slice_size, &dim)| slice_size == dim as usize);

    for &idx in &index_vals {
        if idx >= op_dims[0] as usize {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "gather index {} out of bounds for axis 0 dim {}",
                    idx, op_dims[0]
                ),
            });
        }

        // Base offset for this index along axis 0
        let base_offset = idx * op_strides[0];
        if trailing_slice_is_contiguous {
            let end = base_offset + slice_elems;
            elements.extend_from_slice(&operand.elements[base_offset..end]);
            continue;
        }

        // Iterate over all positions within the slice (axes 1..rank)
        let mut slice_coords = vec![0_usize; rank.saturating_sub(1)];
        for _ in 0..slice_elems {
            let mut flat = base_offset;
            for (ax, &coord) in slice_coords.iter().enumerate() {
                flat += coord * op_strides[ax + 1];
            }
            elements.push(operand.elements[flat]);

            // Increment slice coordinates
            if rank > 1 {
                for ax in (0..rank - 1).rev() {
                    slice_coords[ax] += 1;
                    if slice_coords[ax] < slice_sizes[ax + 1] {
                        break;
                    }
                    slice_coords[ax] = 0;
                }
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Scatter: update positions in an operand tensor using indices and update values.
///
/// Simplified semantics (1-D index scatter):
///   operand: tensor to scatter into (cloned, not mutated)
///   indices: 1-D integer tensor of scatter indices (into axis 0 of operand)
///   updates: tensor of values to scatter; shape = [num_indices] ++ operand.shape[1..]
///
/// For each index i in `indices`, overwrites the slice at `operand[indices[i], ...]`
/// with the corresponding slice from `updates[i, ...]`.
pub(crate) fn eval_scatter(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Scatter;
    let _ = params; // No params needed for basic scatter
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot scatter into a scalar".into(),
            });
        }
    };

    if operand.shape.rank() == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "cannot scatter into a rank-0 tensor".into(),
        });
    }

    let (index_vals, index_shape): (Vec<usize>, Vec<u32>) = match &inputs[1] {
        Value::Scalar(lit) => (vec![lit_to_usize(lit, primitive)?], Vec::new()),
        Value::Tensor(t) => (
            t.elements
                .iter()
                .map(|lit| lit_to_usize(lit, primitive))
                .collect::<Result<_, _>>()?,
            t.shape.dims.clone(),
        ),
    };

    let updates_dtype = inputs[2].dtype();
    if updates_dtype != operand.dtype {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "scatter updates dtype must match operand dtype",
        });
    }

    let mut expected_update_dims = index_shape;
    expected_update_dims.extend(operand.shape.dims.iter().skip(1).copied());
    let expected_update_shape = Shape {
        dims: expected_update_dims.clone(),
    };

    let scalar_updates_storage = match &inputs[2] {
        Value::Scalar(lit) => {
            if expected_update_dims.is_empty() {
                Some(TensorValue::new(
                    operand.dtype,
                    Shape::scalar(),
                    vec![*lit],
                )?)
            } else {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "updates must be a tensor for non-scalar scatter slices".into(),
                });
            }
        }
        Value::Tensor(_) => None,
    };
    let updates = match &inputs[2] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => match scalar_updates_storage.as_ref() {
            Some(tensor) => tensor,
            None => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "scalar updates tensor construction failed".into(),
                });
            }
        },
    };

    let rank = operand.shape.rank();
    let op_dims = &operand.shape.dims;

    // Compute operand strides (row-major)
    let mut op_strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        op_strides[i] = op_strides[i + 1] * op_dims[i + 1] as usize;
    }

    // Number of elements per slice (product of dims[1..])
    let slice_elems: usize = op_dims[1..].iter().map(|d| *d as usize).product::<usize>();

    // Clone operand elements to create output
    let mut result_elements = operand.elements.clone();

    let mode = params
        .get("mode")
        .map(|s| s.as_str())
        .unwrap_or("overwrite");

    if mode != "overwrite" && mode != "add" {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("unknown scatter mode \"{mode}\", expected \"overwrite\" or \"add\""),
        });
    }
    let add_mode = mode == "add";

    if updates.shape != expected_update_shape {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: updates.shape.clone(),
            right: expected_update_shape,
        });
    }

    let expected_update_elems = index_vals.len() * slice_elems;
    if updates.elements.len() != expected_update_elems {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "updates has {} elements but expected {} (num_indices={} * slice_elems={slice_elems})",
                updates.elements.len(),
                expected_update_elems,
                index_vals.len()
            ),
        });
    }

    for (i, &idx) in index_vals.iter().enumerate() {
        if idx >= op_dims[0] as usize {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "scatter index {} out of bounds for axis 0 dim {}",
                    idx, op_dims[0]
                ),
            });
        }

        let base_offset = idx * op_strides[0];
        let update_offset = i * slice_elems;

        if !add_mode {
            let result_end = base_offset + slice_elems;
            let update_end = update_offset + slice_elems;
            result_elements[base_offset..result_end]
                .copy_from_slice(&updates.elements[update_offset..update_end]);
            continue;
        }

        for j in 0..slice_elems {
            let current = &result_elements[base_offset + j];
            let update = &updates.elements[update_offset + j];
            result_elements[base_offset + j] = binary_literal_op(
                *current,
                *update,
                Primitive::Add,
                &|a, b| a.wrapping_add(b),
                &|a, b| a + b,
            )?;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        operand.shape.clone(),
        result_elements,
    )?))
}

fn lit_to_usize(lit: &Literal, primitive: Primitive) -> Result<usize, EvalError> {
    match lit {
        Literal::I64(n) => {
            if *n >= 0 {
                Ok(*n as usize)
            } else {
                Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("negative index {n}"),
                })
            }
        }
        Literal::U32(n) => Ok(*n as usize),
        Literal::U64(n) => usize::try_from(*n).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("index {n} exceeds usize range"),
        }),
        Literal::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) => {
            Err(EvalError::Unsupported {
                primitive,
                detail: "float indices not supported".into(),
            })
        }
        Literal::F64Bits(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "float indices not supported".into(),
        }),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::Unsupported {
            primitive,
            detail: "complex indices not supported".into(),
        }),
    }
}

fn normalize_dynamic_start(raw: i64, dim: i64, window: i64) -> usize {
    let adjusted = if raw < 0 { raw + dim } else { raw };
    adjusted.max(0).min(dim - window) as usize
}

/// Dynamic slice: like slice but start indices are dynamic (runtime) values.
///
/// Inputs: [operand, start_0, start_1, ...] where start_i are scalar indices.
/// Params: `slice_sizes` — comma-separated sizes for the output slice along each axis.
///
/// JAX semantics: negative start indices are interpreted relative to the end,
/// then clamped to valid range [0, dim - size].
pub(crate) fn eval_dynamic_slice(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::DynamicSlice;
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: 0,
        });
    }

    let tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot dynamic_slice a scalar".into(),
            });
        }
    };

    let rank = tensor.shape.rank();
    let slice_sizes = parse_usize_param(primitive, "slice_sizes", params)?;

    if slice_sizes.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes length {} does not match operand rank {}",
                slice_sizes.len(),
                rank
            ),
        });
    }
    for (ax, &size) in slice_sizes.iter().enumerate() {
        let dim = tensor.shape.dims[ax] as usize;
        if size > dim {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("slice size {size} exceeds dimension {dim} on axis {ax}"),
            });
        }
    }

    // Start indices come from remaining inputs (one scalar per axis)
    if inputs.len() != 1 + rank {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1 + rank,
            actual: inputs.len(),
        });
    }

    let mut starts = Vec::with_capacity(rank);
    for ax in 0..rank {
        let start_val = match &inputs[1 + ax] {
            Value::Scalar(lit) => {
                let raw = match lit {
                    Literal::I64(v) => *v,
                    Literal::U32(v) => i64::from(*v),
                    Literal::U64(v) => i64::try_from(*v).unwrap_or(i64::MAX),
                    Literal::Bool(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
                    Literal::BF16Bits(_)
                    | Literal::F16Bits(_)
                    | Literal::F32Bits(_)
                    | Literal::F64Bits(_)
                    | Literal::Complex64Bits(..)
                    | Literal::Complex128Bits(..) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: format!("start index for axis {ax} must be integral dtype"),
                        });
                    }
                };
                let dim = tensor.shape.dims[ax] as i64;
                let size = slice_sizes[ax] as i64;
                normalize_dynamic_start(raw, dim, size)
            }
            Value::Tensor(_) => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("start index for axis {ax} must be a scalar"),
                });
            }
        };
        starts.push(start_val);
    }

    // Validate slice sizes
    for ax in 0..rank {
        let dim = tensor.shape.dims[ax] as usize;
        let max_start = dim - slice_sizes[ax];
        if starts[ax] > max_start {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "dynamic_slice out of bounds on axis {ax}: start={} max_start={}",
                    starts[ax], max_start
                ),
            });
        }
    }

    let out_dims: Vec<u32> = slice_sizes.iter().map(|&s| s as u32).collect();
    let total = checked_shape_element_count(primitive, "dynamic_slice", &out_dims)?;
    if total == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            tensor.dtype,
            Shape { dims: out_dims },
            Vec::new(),
        )?));
    }

    // Compute strides
    let in_strides = checked_row_major_strides(primitive, "dynamic_slice", &tensor.shape.dims)?;

    let has_contiguous_trailing_slice = rank > 0
        && slice_sizes
            .iter()
            .skip(1)
            .zip(tensor.shape.dims.iter().skip(1))
            .all(|(&slice_size, &dim)| slice_size == dim as usize)
        && starts.iter().skip(1).all(|&start| start == 0);
    if has_contiguous_trailing_slice {
        let start_offset =
            starts[0]
                .checked_mul(in_strides[0])
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "dynamic_slice start offset overflows usize".to_owned(),
                })?;
        let end_offset = start_offset
            .checked_add(total)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "dynamic_slice end offset overflows usize".to_owned(),
            })?;
        return Ok(Value::Tensor(TensorValue::new(
            tensor.dtype,
            Shape { dims: out_dims },
            tensor.elements[start_offset..end_offset].to_vec(),
        )?));
    }

    let mut elements = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; rank];

    for _ in 0..total {
        let mut in_flat = 0_usize;
        for ax in 0..rank {
            let coord =
                out_coords[ax]
                    .checked_add(starts[ax])
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: format!("dynamic_slice coordinate overflows usize on axis {ax}"),
                    })?;
            let offset =
                coord
                    .checked_mul(in_strides[ax])
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: format!("dynamic_slice offset overflows usize on axis {ax}"),
                    })?;
            in_flat = in_flat
                .checked_add(offset)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "dynamic_slice flat offset overflows usize".to_owned(),
                })?;
        }
        elements.push(tensor.elements[in_flat]);

        // Increment output coordinates
        for ax in (0..rank).rev() {
            out_coords[ax] += 1;
            if out_coords[ax] < slice_sizes[ax] {
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

/// DynamicUpdateSlice: write `update` into `operand` at position given by start indices.
///
/// Inputs: [operand, update, start_0, start_1, ..., start_{rank-1}]
/// Output: tensor with same shape as operand, with the update region overwritten.
pub(crate) fn eval_dynamic_update_slice(
    inputs: &[Value],
    _params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::DynamicUpdateSlice;
    if inputs.len() < 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot dynamic_update_slice a scalar".into(),
            });
        }
    };

    let update = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "update must be a tensor".into(),
            });
        }
    };
    if update.dtype != operand.dtype {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "update dtype must match operand dtype",
        });
    }

    let rank = operand.shape.rank();
    if update.shape.rank() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "update rank {} != operand rank {}",
                update.shape.rank(),
                rank
            ),
        });
    }
    for ax in 0..rank {
        let dim = operand.shape.dims[ax];
        let upd = update.shape.dims[ax];
        if upd > dim {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("update dim {upd} exceeds operand dim {dim} on axis {ax}"),
            });
        }
    }

    if inputs.len() != 2 + rank {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2 + rank,
            actual: inputs.len(),
        });
    }

    // Parse start indices (one scalar per axis), adjusted then clamped to valid range.
    let mut starts = Vec::with_capacity(rank);
    for ax in 0..rank {
        let start_val = match &inputs[2 + ax] {
            Value::Scalar(lit) => {
                let raw = match lit {
                    Literal::I64(v) => *v,
                    Literal::U32(v) => i64::from(*v),
                    Literal::U64(v) => i64::try_from(*v).unwrap_or(i64::MAX),
                    Literal::Bool(b) => i64::from(*b),
                    Literal::BF16Bits(_)
                    | Literal::F16Bits(_)
                    | Literal::F32Bits(_)
                    | Literal::F64Bits(_)
                    | Literal::Complex64Bits(..)
                    | Literal::Complex128Bits(..) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: format!("start index for axis {ax} must be integral dtype"),
                        });
                    }
                };
                let dim = operand.shape.dims[ax] as i64;
                let upd_size = update.shape.dims[ax] as i64;
                normalize_dynamic_start(raw, dim, upd_size)
            }
            Value::Tensor(_) => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("start index for axis {ax} must be a scalar"),
                });
            }
        };
        starts.push(start_val);
    }

    // Copy operand elements, overwriting the update region
    let mut elements = operand.elements.clone();

    let mut op_strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        op_strides[i] = op_strides[i + 1] * operand.shape.dims[i + 1] as usize;
    }

    let upd_total = update.elements.len();
    let has_contiguous_trailing_update = rank > 0
        && update
            .shape
            .dims
            .iter()
            .skip(1)
            .zip(operand.shape.dims.iter().skip(1))
            .all(|(&update_dim, &operand_dim)| update_dim == operand_dim)
        && starts.iter().skip(1).all(|&start| start == 0);
    if has_contiguous_trailing_update {
        let start_offset = starts[0] * op_strides[0];
        let end_offset = start_offset + upd_total;
        elements[start_offset..end_offset].copy_from_slice(&update.elements);
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            operand.shape.clone(),
            elements,
        )?));
    }

    let mut upd_coords = vec![0_usize; rank];

    for upd_flat in 0..upd_total {
        let mut op_flat = 0_usize;
        for ax in 0..rank {
            op_flat += (upd_coords[ax] + starts[ax]) * op_strides[ax];
        }
        if op_flat < elements.len() {
            elements[op_flat] = update.elements[upd_flat];
        }

        // Increment update coordinates
        for ax in (0..rank).rev() {
            upd_coords[ax] += 1;
            if upd_coords[ax] < update.shape.dims[ax] as usize {
                break;
            }
            upd_coords[ax] = 0;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        operand.shape.clone(),
        elements,
    )?))
}

/// Copy: explicit identity operation that returns an independent cloned value.
pub(crate) fn eval_copy(inputs: &[Value]) -> Result<Value, EvalError> {
    let primitive = Primitive::Copy;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }
    Ok(inputs[0].clone())
}

/// BitcastConvertType: reinterpret element bit patterns as a new dtype.
///
/// Params:
/// - `new_dtype`: destination dtype string (e.g. `i32`, `f32`, `i64`, `f64`)
///
/// Constraint:
/// - Source and destination element widths must match.
pub(crate) fn eval_bitcast_convert_type(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BitcastConvertType;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let target_dtype = parse_dtype_param(primitive, "new_dtype", params)?;
    let source_dtype = inputs[0].dtype();

    if dtype_bit_width(source_dtype) != dtype_bit_width(target_dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "bitcast requires equal element widths: source={source_dtype:?}({} bits) target={target_dtype:?}({} bits)",
                dtype_bit_width(source_dtype),
                dtype_bit_width(target_dtype)
            ),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let raw = literal_to_bytes(primitive, source_dtype, *literal)?;
            let converted = bytes_to_literal(primitive, target_dtype, &raw)?;
            Ok(Value::Scalar(converted))
        }
        Value::Tensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let raw = literal_to_bytes(primitive, source_dtype, *literal)?;
                out.push(bytes_to_literal(primitive, target_dtype, &raw)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                target_dtype,
                tensor.shape.clone(),
                out,
            )?))
        }
    }
}

/// Iota: generate a 1-D index tensor of length `length` with dtype `dtype`.
///
/// Inputs: [] (no inputs — iota is a nullary operation)
/// Params: `length` — number of elements, `dtype` — element type (default: I64)
pub(crate) fn eval_iota(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Iota;
    if !inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 0,
            actual: inputs.len(),
        });
    }

    let length_str = params.get("length").ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: "missing required param 'length'".into(),
    })?;
    let length: u32 = length_str
        .trim()
        .parse()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid length: '{length_str}'"),
        })?;

    let dtype_str = params.get("dtype").map(String::as_str).unwrap_or("I64");
    let (dtype, elements) = match dtype_str {
        "I64" | "i64" => {
            let elems: Vec<Literal> = (0..length as i64).map(Literal::I64).collect();
            (DType::I64, elems)
        }
        "I32" | "i32" => {
            let elems: Vec<Literal> = (0..length as i64).map(Literal::I64).collect();
            (DType::I32, elems)
        }
        "F64" | "f64" => {
            let elems: Vec<Literal> = (0..length)
                .map(|i| Literal::from_f64(f64::from(i)))
                .collect();
            (DType::F64, elems)
        }
        "F32" | "f32" => {
            let elems: Vec<Literal> = (0..length)
                .map(|i| Literal::from_f64(f64::from(i)))
                .collect();
            (DType::F32, elems)
        }
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("unsupported dtype for iota: '{dtype_str}'"),
            });
        }
    };

    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape::vector(length),
        elements,
    )?))
}

fn literal_from_index_for_dtype(
    primitive: Primitive,
    dtype: DType,
    index: usize,
) -> Result<Literal, EvalError> {
    match dtype {
        DType::I64 => Ok(Literal::I64(index as i64)),
        DType::I32 => {
            let value = i32::try_from(index).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("index {index} exceeds i32 range"),
            })?;
            Ok(Literal::I64(i64::from(value)))
        }
        DType::U32 => {
            let value = u32::try_from(index).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("index {index} exceeds u32 range"),
            })?;
            Ok(Literal::U32(value))
        }
        DType::U64 => Ok(Literal::U64(index as u64)),
        DType::F64 => Ok(Literal::from_f64(index as f64)),
        DType::F32 => Ok(Literal::from_f64(f64::from(index as f32))),
        DType::BF16 => Ok(Literal::from_bf16_f32(index as f32)),
        DType::F16 => Ok(Literal::from_f16_f32(index as f32)),
        DType::Bool => Ok(Literal::Bool(index != 0)),
        DType::Complex64 | DType::Complex128 => Err(EvalError::Unsupported {
            primitive,
            detail: "broadcasted_iota does not support complex dtypes".to_owned(),
        }),
    }
}

/// BroadcastedIota: iota over `dimension` and broadcast across full `shape`.
///
/// Inputs: none.
/// Params:
/// - `shape`: comma-separated output dimensions
/// - `dimension`: axis carrying monotonically increasing indices
/// - `dtype`: optional output dtype (default `i64`)
pub(crate) fn eval_broadcasted_iota(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BroadcastedIota;
    if !inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 0,
            actual: inputs.len(),
        });
    }

    let shape_usize = parse_usize_param(primitive, "shape", params)?;
    let dimension = params
        .get("dimension")
        .map(|raw| {
            raw.trim()
                .parse::<usize>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid dimension: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(0);
    let dtype = if let Some(raw) = params.get("dtype") {
        parse_dtype_name(primitive, "dtype", raw)?
    } else {
        DType::I64
    };

    let shape_u32: Vec<u32> = shape_usize
        .iter()
        .map(|&d| {
            u32::try_from(d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("shape dimension {d} exceeds u32 range"),
            })
        })
        .collect::<Result<_, _>>()?;

    if shape_usize.is_empty() {
        return Ok(Value::Scalar(literal_from_index_for_dtype(
            primitive, dtype, 0,
        )?));
    }

    if dimension >= shape_usize.len() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "dimension {dimension} out of bounds for rank {}",
                shape_usize.len()
            ),
        });
    }

    let total = shape_usize.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or(EvalError::Unsupported {
            primitive,
            detail: "broadcasted_iota shape overflows usize".to_owned(),
        })
    })?;
    let stride = shape_usize[(dimension + 1)..]
        .iter()
        .fold(1_usize, |acc, dim| acc.saturating_mul(*dim));
    let axis_extent = shape_usize[dimension];

    let mut elements = Vec::with_capacity(total);
    for flat in 0..total {
        let axis_index = (flat / stride) % axis_extent;
        elements.push(literal_from_index_for_dtype(primitive, dtype, axis_index)?);
    }

    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape { dims: shape_u32 },
        elements,
    )?))
}

fn quantize_f64(value: f64, exponent_bits: u32, mantissa_bits: u32) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }

    let exp_bits = exponent_bits.clamp(1, 11);
    let mant_bits = mantissa_bits.min(52);
    if exp_bits == 11 && mant_bits == 52 {
        return value;
    }

    let bits = value.to_bits();
    let sign = bits & (1_u64 << 63);
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mut mant = bits & ((1_u64 << 52) - 1);

    if exp == 0 {
        return f64::from_bits(sign);
    }

    let unbiased = exp - 1023;
    if exp_bits < 11 {
        let bias_new = (1_i32 << (exp_bits - 1)) - 1;
        let min_unbiased = 1 - bias_new;
        let max_unbiased = bias_new;
        if unbiased > max_unbiased {
            return f64::from_bits(sign | (0x7ff_u64 << 52));
        }
        if unbiased < min_unbiased {
            return f64::from_bits(sign);
        }
    }

    if mant_bits < 52 {
        let drop = 52 - mant_bits;
        mant = (mant >> drop) << drop;
    }

    f64::from_bits(sign | ((exp as u64) << 52) | mant)
}

fn quantize_f32(value: f32, exponent_bits: u32, mantissa_bits: u32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }

    let exp_bits = exponent_bits.clamp(1, 8);
    let mant_bits = mantissa_bits.min(23);
    if exp_bits == 8 && mant_bits == 23 {
        return value;
    }

    let bits = value.to_bits();
    let sign = bits & (1_u32 << 31);
    let exp = ((bits >> 23) & 0xff) as i32;
    let mut mant = bits & ((1_u32 << 23) - 1);

    if exp == 0 {
        return f32::from_bits(sign);
    }

    let unbiased = exp - 127;
    if exp_bits < 8 {
        let bias_new = (1_i32 << (exp_bits - 1)) - 1;
        let min_unbiased = 1 - bias_new;
        let max_unbiased = bias_new;
        if unbiased > max_unbiased {
            return f32::from_bits(sign | (0xff_u32 << 23));
        }
        if unbiased < min_unbiased {
            return f32::from_bits(sign);
        }
    }

    if mant_bits < 23 {
        let drop = 23 - mant_bits;
        mant = (mant >> drop) << drop;
    }

    f32::from_bits(sign | ((exp as u32) << 23) | mant)
}

fn reduce_precision_literal(
    primitive: Primitive,
    dtype: DType,
    literal: Literal,
    exponent_bits: u32,
    mantissa_bits: u32,
) -> Result<Literal, EvalError> {
    match dtype {
        DType::F64 => {
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected an f64 value",
            })?;
            Ok(Literal::from_f64(quantize_f64(
                value,
                exponent_bits,
                mantissa_bits,
            )))
        }
        DType::F32 => {
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected an f32-compatible value",
            })? as f32;
            Ok(Literal::from_f64(f64::from(quantize_f32(
                value,
                exponent_bits,
                mantissa_bits,
            ))))
        }
        DType::BF16 => {
            let value = literal.as_bf16_f32().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected bf16 literal payload",
            })?;
            Ok(Literal::from_bf16_f32(quantize_f32(
                value,
                exponent_bits.min(8),
                mantissa_bits.min(23),
            )))
        }
        DType::F16 => {
            let value = literal.as_f16_f32().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected f16 literal payload",
            })?;
            Ok(Literal::from_f16_f32(quantize_f32(
                value,
                exponent_bits.min(8),
                mantissa_bits.min(23),
            )))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "reduce_precision supports floating-point dtypes only".to_owned(),
        }),
    }
}

/// ReducePrecision: simulate reduced floating-point exponent/mantissa precision.
///
/// Params:
/// - `exponent_bits` (default: native exponent width)
/// - `mantissa_bits` (default: native mantissa width)
pub(crate) fn eval_reduce_precision(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ReducePrecision;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let source_dtype = inputs[0].dtype();
    let (default_exp_bits, default_mant_bits) = match source_dtype {
        DType::F64 => (11_u32, 52_u32),
        DType::F32 => (8_u32, 23_u32),
        DType::BF16 => (8_u32, 7_u32),
        DType::F16 => (5_u32, 10_u32),
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "reduce_precision supports floating-point dtypes only".to_owned(),
            });
        }
    };

    let exponent_bits = params
        .get("exponent_bits")
        .map(|raw| {
            raw.trim()
                .parse::<u32>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid exponent_bits: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(default_exp_bits);
    let mantissa_bits = params
        .get("mantissa_bits")
        .map(|raw| {
            raw.trim()
                .parse::<u32>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid mantissa_bits: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(default_mant_bits);

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(reduce_precision_literal(
            primitive,
            source_dtype,
            *literal,
            exponent_bits,
            mantissa_bits,
        )?)),
        Value::Tensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                out.push(reduce_precision_literal(
                    primitive,
                    source_dtype,
                    *literal,
                    exponent_bits,
                    mantissa_bits,
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                source_dtype,
                tensor.shape.clone(),
                out,
            )?))
        }
    }
}

/// One-hot encoding: given integer indices, produces a tensor with a trailing
/// `num_classes` dimension where position `index` is `on_value` and the rest
/// are `off_value`.
pub(crate) fn eval_one_hot(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::OneHot;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let num_classes_str = params
        .get("num_classes")
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "missing required param 'num_classes'".into(),
        })?;
    let num_classes: u32 = num_classes_str
        .trim()
        .parse()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid num_classes: '{num_classes_str}'"),
        })?;

    let on_value: f64 = params
        .get("on_value")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);
    let off_value: f64 = params
        .get("off_value")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    let dtype_str = params.get("dtype").map(String::as_str).unwrap_or("F64");

    // Collect flat index values from input
    let indices: Vec<i64> = match &inputs[0] {
        Value::Scalar(lit) => {
            vec![lit.as_i64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "one_hot requires integer indices",
            })?]
        }
        Value::Tensor(t) => {
            let mut idxs = Vec::with_capacity(t.elements.len());
            for lit in &t.elements {
                idxs.push(lit.as_i64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "one_hot requires integer indices",
                })?);
            }
            idxs
        }
    };

    // Build output shape: input_shape ++ [num_classes]
    let input_shape = match &inputs[0] {
        Value::Scalar(_) => vec![],
        Value::Tensor(t) => t.shape.dims.clone(),
    };
    let mut out_dims = input_shape;
    out_dims.push(num_classes);

    let nc = num_classes as usize;
    let total = indices.len() * nc;
    let mut elements = Vec::with_capacity(total);

    let use_int = matches!(dtype_str, "I64" | "i64" | "I32" | "i32");

    for &idx in &indices {
        for c in 0..nc {
            let val = if idx >= 0 && (idx as usize) == c {
                on_value
            } else {
                off_value
            };
            if use_int {
                elements.push(Literal::I64(val as i64));
            } else {
                elements.push(Literal::from_f64(val));
            }
        }
    }

    let dtype = match dtype_str {
        "I64" | "i64" => DType::I64,
        "I32" | "i32" => DType::I32,
        "F32" | "f32" => DType::F32,
        _ => DType::F64,
    };

    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Sort: sort elements along a specified axis (default: last axis).
pub(crate) fn eval_sort(
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

    let descending = params
        .get("descending")
        .map(|s| s.trim() == "true")
        .unwrap_or(false);

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let axis = parse_axis_param(primitive, "axis", params, rank, rank - 1)?;

            sort_along_axis(tensor, axis, descending, false).map_err(|e| EvalError::Unsupported {
                primitive,
                detail: e,
            })
        }
    }
}

/// Argsort: return indices that would sort along a specified axis.
pub(crate) fn eval_argsort(
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

    let descending = params
        .get("descending")
        .map(|s| s.trim() == "true")
        .unwrap_or(false);

    match &inputs[0] {
        Value::Scalar(_) => Ok(Value::scalar_i64(0)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::scalar_i64(0));
            }

            let axis = parse_axis_param(primitive, "axis", params, rank, rank - 1)?;

            sort_along_axis(tensor, axis, descending, true).map_err(|e| EvalError::Unsupported {
                primitive,
                detail: e,
            })
        }
    }
}

/// Sort or argsort a tensor along a given axis.
fn sort_along_axis(
    tensor: &TensorValue,
    axis: usize,
    descending: bool,
    return_indices: bool,
) -> Result<Value, String> {
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;

    if axis_dim == 0 || tensor.elements.is_empty() {
        let result_dtype = if return_indices {
            DType::I64
        } else {
            tensor.dtype
        };
        return Ok(Value::Tensor(
            TensorValue::new(result_dtype, tensor.shape.clone(), vec![])
                .map_err(|e| format!("empty sort result: {e}"))?,
        ));
    }

    let mut strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tensor.shape.dims[i + 1] as usize;
    }
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    let outer_count = total / axis_dim;

    let mut result_elements = if return_indices {
        vec![Literal::I64(0); total]
    } else {
        tensor.elements.clone()
    };

    for outer in 0..outer_count {
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

        let mut indexed: Vec<(usize, SortKey)> = (0..axis_dim)
            .map(|i| {
                let flat_idx = base + i * axis_stride;
                sort_key(tensor.elements[flat_idx]).map(|key| (i, key))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if descending {
            indexed.sort_by(|a, b| compare_sort_keys(b.1, a.1));
        } else {
            indexed.sort_by(|a, b| compare_sort_keys(a.1, b.1));
        }

        for (out_pos, &(orig_idx, _)) in indexed.iter().enumerate() {
            let flat_idx = base + out_pos * axis_stride;
            if return_indices {
                result_elements[flat_idx] = Literal::I64(orig_idx as i64);
            } else {
                let source_idx = base + orig_idx * axis_stride;
                result_elements[flat_idx] = tensor.elements[source_idx];
            }
        }
    }

    let out_dtype = if return_indices {
        DType::I64
    } else {
        tensor.dtype
    };
    Ok(Value::Tensor(
        TensorValue::new(out_dtype, tensor.shape.clone(), result_elements)
            .map_err(|e| e.to_string())?,
    ))
}

#[derive(Clone, Copy)]
enum SortKey {
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

fn sort_key(literal: Literal) -> Result<SortKey, String> {
    match literal {
        Literal::Bool(value) => Ok(SortKey::Bool(value)),
        Literal::I64(value) => Ok(SortKey::Signed(value)),
        Literal::U32(value) => Ok(SortKey::Unsigned(u64::from(value))),
        Literal::U64(value) => Ok(SortKey::Unsigned(value)),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) | Literal::F64Bits(_) => {
            literal
                .as_f64()
                .map(SortKey::Float)
                .ok_or_else(|| format!("sort requires orderable numeric literals, got {literal:?}"))
        }
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(format!(
            "sort does not support complex literals: {literal:?}"
        )),
    }
}

fn compare_sort_keys(lhs: SortKey, rhs: SortKey) -> Ordering {
    match (lhs, rhs) {
        (SortKey::Bool(lhs), SortKey::Bool(rhs)) => lhs.cmp(&rhs),
        (SortKey::Signed(lhs), SortKey::Signed(rhs)) => lhs.cmp(&rhs),
        (SortKey::Unsigned(lhs), SortKey::Unsigned(rhs)) => lhs.cmp(&rhs),
        (SortKey::Float(lhs), SortKey::Float(rhs)) => lhs.total_cmp(&rhs),
        (lhs, rhs) => sort_key_rank(lhs).cmp(&sort_key_rank(rhs)),
    }
}

fn sort_key_rank(key: SortKey) -> u8 {
    match key {
        SortKey::Bool(_) => 0,
        SortKey::Signed(_) => 1,
        SortKey::Unsigned(_) => 2,
        SortKey::Float(_) => 3,
    }
}

/// Conv: N-dimensional convolution.
/// Layout: lhs=[batch, spatial..., in_channels], rhs=[kernel_spatial..., in_channels, out_channels]
/// Params: strides (comma-sep), padding ("valid" or "same")
pub(crate) fn eval_conv(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let lhs = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "conv requires tensor inputs".into(),
            });
        }
    };
    let rhs = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "conv requires tensor kernel".into(),
            });
        }
    };

    let is_float =
        |dtype: DType| matches!(dtype, DType::BF16 | DType::F16 | DType::F32 | DType::F64);
    if !is_float(lhs.dtype) || !is_float(rhs.dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "conv requires floating dtypes, got lhs {:?}, rhs {:?}",
                lhs.dtype, rhs.dtype
            ),
        });
    }

    let lhs_rank = lhs.shape.rank();
    if lhs_rank < 3 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("conv lhs must have rank >= 3, got {lhs_rank}"),
        });
    }

    if lhs_rank > 4 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("conv supports rank 3 (1D) and rank 4 (2D), got rank {lhs_rank}"),
        });
    }

    let padding_mode = params.get("padding").map(String::as_str).unwrap_or("valid");

    if lhs_rank == 3 {
        eval_conv_1d(primitive, lhs, rhs, params, padding_mode)
    } else {
        eval_conv_2d(primitive, lhs, rhs, params, padding_mode)
    }
}

/// 1D convolution: lhs=[N, W, C_in], rhs=[K, C_in, C_out]
fn eval_conv_1d(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    params: &BTreeMap<String, String>,
    padding_mode: &str,
) -> Result<Value, EvalError> {
    if rhs.shape.rank() != 3 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "1D conv kernel must have rank 3 [K, C_in, C_out], got rank {}",
                rhs.shape.rank()
            ),
        });
    }

    let batch = lhs.shape.dims[0] as usize;
    let width = lhs.shape.dims[1] as usize;
    let c_in = lhs.shape.dims[2] as usize;

    let kernel_w = rhs.shape.dims[0] as usize;
    let rhs_c_in = rhs.shape.dims[1] as usize;
    let c_out = rhs.shape.dims[2] as usize;

    if c_in != rhs_c_in {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("channel mismatch: lhs c_in={c_in}, rhs c_in={rhs_c_in}"),
        });
    }

    let stride = parse_positive_stride(primitive, params.get("strides").map(String::as_str))?;

    let (out_w, pad_left) = match padding_mode {
        "same" | "SAME" => {
            let out_w = width.div_ceil(stride);
            let pad_total = ((out_w - 1) * stride + kernel_w).saturating_sub(width);
            (out_w, pad_total / 2)
        }
        _ => {
            if width < kernel_w {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("input width {width} < kernel {kernel_w} with valid padding"),
                });
            }
            ((width - kernel_w) / stride + 1, 0)
        }
    };

    let total = batch * out_w * c_out;
    let mut elements = Vec::with_capacity(total);

    for n in 0..batch {
        for w in 0..out_w {
            for co in 0..c_out {
                let mut acc = 0.0_f64;
                for k in 0..kernel_w {
                    let in_pos = (w * stride + k) as isize - pad_left as isize;
                    if in_pos >= 0 && (in_pos as usize) < width {
                        for ci in 0..c_in {
                            let lhs_idx = n * width * c_in + (in_pos as usize) * c_in + ci;
                            let rhs_idx = k * c_in * c_out + ci * c_out + co;
                            let lhs_val = lhs.elements[lhs_idx].as_f64().unwrap_or(0.0);
                            let rhs_val = rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                            acc += lhs_val * rhs_val;
                        }
                    }
                }
                elements.push(Literal::from_f64(acc));
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        promote_dtype(lhs.dtype, rhs.dtype),
        Shape {
            dims: vec![batch as u32, out_w as u32, c_out as u32],
        },
        elements,
    )?))
}

/// 2D convolution: lhs=[N, H, W, C_in], rhs=[KH, KW, C_in, C_out]
fn eval_conv_2d(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    params: &BTreeMap<String, String>,
    padding_mode: &str,
) -> Result<Value, EvalError> {
    if rhs.shape.rank() != 4 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "2D conv kernel must have rank 4 [KH, KW, C_in, C_out], got rank {}",
                rhs.shape.rank()
            ),
        });
    }

    let batch = lhs.shape.dims[0] as usize;
    let height = lhs.shape.dims[1] as usize;
    let width = lhs.shape.dims[2] as usize;
    let c_in = lhs.shape.dims[3] as usize;

    let kernel_h = rhs.shape.dims[0] as usize;
    let kernel_w = rhs.shape.dims[1] as usize;
    let rhs_c_in = rhs.shape.dims[2] as usize;
    let c_out = rhs.shape.dims[3] as usize;

    if c_in != rhs_c_in {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("channel mismatch: lhs c_in={c_in}, rhs c_in={rhs_c_in}"),
        });
    }

    // Parse strides: either single value or "h,w" pair
    let (stride_h, stride_w) = parse_stride_pair(primitive, params)?;

    let (out_h, pad_top) = compute_output_and_pad(height, kernel_h, stride_h, padding_mode);
    let (out_w, pad_left) = compute_output_and_pad(width, kernel_w, stride_w, padding_mode);

    if padding_mode != "same" && padding_mode != "SAME" {
        if height < kernel_h {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("input height {height} < kernel {kernel_h} with valid padding"),
            });
        }
        if width < kernel_w {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("input width {width} < kernel {kernel_w} with valid padding"),
            });
        }
    }

    let total = batch * out_h * out_w * c_out;
    let mut elements = Vec::with_capacity(total);

    for n in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for co in 0..c_out {
                    let mut acc = 0.0_f64;
                    for kh in 0..kernel_h {
                        let in_h = (oh * stride_h + kh) as isize - pad_top as isize;
                        if in_h < 0 || (in_h as usize) >= height {
                            continue;
                        }
                        for kw in 0..kernel_w {
                            let in_w = (ow * stride_w + kw) as isize - pad_left as isize;
                            if in_w < 0 || (in_w as usize) >= width {
                                continue;
                            }
                            for ci in 0..c_in {
                                let lhs_idx = n * height * width * c_in
                                    + (in_h as usize) * width * c_in
                                    + (in_w as usize) * c_in
                                    + ci;
                                let rhs_idx = kh * kernel_w * c_in * c_out
                                    + kw * c_in * c_out
                                    + ci * c_out
                                    + co;
                                let lhs_val = lhs.elements[lhs_idx].as_f64().unwrap_or(0.0);
                                let rhs_val = rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                                acc += lhs_val * rhs_val;
                            }
                        }
                    }
                    elements.push(Literal::from_f64(acc));
                }
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        promote_dtype(lhs.dtype, rhs.dtype),
        Shape {
            dims: vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
        },
        elements,
    )?))
}

fn parse_positive_stride(primitive: Primitive, stride: Option<&str>) -> Result<usize, EvalError> {
    let raw = stride.unwrap_or("1").trim();
    let parsed = raw.parse::<usize>().map_err(|_| EvalError::Unsupported {
        primitive,
        detail: format!("invalid conv stride {raw:?}"),
    })?;
    if parsed == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "conv stride must be positive".to_owned(),
        });
    }
    Ok(parsed)
}

fn parse_stride_pair(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<(usize, usize), EvalError> {
    let strides_str = params.get("strides").map(String::as_str).unwrap_or("1");
    let mut parts = strides_str.split(',');
    let first = parts.next().unwrap_or("1");
    let second = parts.next();
    if parts.next().is_some() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid conv strides {strides_str:?}"),
        });
    }
    match second {
        Some(width) => Ok((
            parse_positive_stride(primitive, Some(first))?,
            parse_positive_stride(primitive, Some(width))?,
        )),
        None => {
            let stride = parse_positive_stride(primitive, Some(first))?;
            Ok((stride, stride))
        }
    }
}

fn compute_output_and_pad(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding_mode: &str,
) -> (usize, usize) {
    match padding_mode {
        "same" | "SAME" => {
            let out = input_size.div_ceil(stride);
            let pad_total = ((out - 1) * stride + kernel_size).saturating_sub(input_size);
            (out, pad_total / 2)
        }
        _ => {
            if input_size < kernel_size {
                (0, 0)
            } else {
                ((input_size - kernel_size) / stride + 1, 0)
            }
        }
    }
}

// ── Rev: reverse elements along specified axes ─────────────────

pub(crate) fn eval_rev(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Rev;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let axes = parse_usize_param(primitive, "axes", params)?;

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let dims = &tensor.shape.dims;
            let rank = dims.len();

            for &a in &axes {
                if a >= rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("axis {a} out of range for rank {rank}"),
                    });
                }
            }

            // Compute strides (row-major)
            let mut strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * dims[i + 1] as usize;
            }

            let total = tensor.elements.len();
            let mut result = vec![Literal::I64(0); total];

            for (flat_idx, elem) in result.iter_mut().enumerate() {
                // Decompose flat_idx into multi-index
                let mut remaining = flat_idx;
                let mut coords = vec![0_usize; rank];
                for d in 0..rank {
                    coords[d] = remaining / strides[d];
                    remaining %= strides[d];
                }

                // Reverse specified axes
                let mut src_coords = coords;
                for &a in &axes {
                    src_coords[a] = (dims[a] as usize) - 1 - src_coords[a];
                }

                // Compute source flat index
                let src_flat: usize = src_coords
                    .iter()
                    .zip(strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();

                *elem = tensor.elements[src_flat];
            }

            Ok(Value::Tensor(
                TensorValue::new(tensor.dtype, tensor.shape.clone(), result).map_err(|e| {
                    EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    }
                })?,
            ))
        }
    }
}

// ── Squeeze: remove singleton dimensions ───────────────────────

pub(crate) fn eval_squeeze(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Squeeze;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let dims = &tensor.shape.dims;

            let squeeze_dims = if params.contains_key("dimensions") {
                parse_usize_param(primitive, "dimensions", params)?
            } else {
                // If no dimensions specified, squeeze all size-1 dims
                dims.iter()
                    .enumerate()
                    .filter(|&(_, &d)| d == 1)
                    .map(|(i, _)| i)
                    .collect()
            };

            for &d in &squeeze_dims {
                if d >= dims.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("dimension {d} out of range for rank {}", dims.len()),
                    });
                }
                if dims[d] != 1 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "cannot squeeze dimension {d} with size {} (must be 1)",
                            dims[d]
                        ),
                    });
                }
            }

            let new_dims: Vec<u32> = dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !squeeze_dims.contains(i))
                .map(|(_, &d)| d)
                .collect();

            if new_dims.is_empty() {
                // All dims squeezed — return scalar
                Ok(Value::Scalar(tensor.elements[0]))
            } else {
                Ok(Value::Tensor(
                    TensorValue::new(
                        tensor.dtype,
                        Shape { dims: new_dims },
                        tensor.elements.clone(),
                    )
                    .map_err(|e| EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    })?,
                ))
            }
        }
    }
}

// ── Split: split array along an axis ───────────────────────────

pub(crate) fn eval_split(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Split;
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
            detail: "cannot split a scalar".into(),
        }),
        Value::Tensor(tensor) => {
            let axis = params
                .get("axis")
                .map(|raw| {
                    raw.trim()
                        .parse::<usize>()
                        .map_err(|_| EvalError::Unsupported {
                            primitive,
                            detail: format!("invalid integer in param 'axis': '{raw}'"),
                        })
                })
                .transpose()?
                .unwrap_or(0);
            let dims = &tensor.shape.dims;
            let rank = dims.len();

            if axis >= rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {axis} out of range for rank {rank}"),
                });
            }

            let axis_size = dims[axis] as usize;

            // Determine split sizes
            let sizes: Vec<usize> = if params.contains_key("sizes") {
                parse_usize_param(primitive, "sizes", params)?
            } else if params.contains_key("num_sections") {
                let num_sections_vec = parse_usize_param(primitive, "num_sections", params)?;
                let num_sections = num_sections_vec[0];
                if !axis_size.is_multiple_of(num_sections) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "axis size {axis_size} not evenly divisible by {num_sections}"
                        ),
                    });
                }
                let section_size = axis_size / num_sections;
                vec![section_size; num_sections]
            } else {
                vec![axis_size]
            };

            // Validate sizes sum to axis_size
            let total: usize = sizes.iter().sum();
            if total != axis_size {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("split sizes sum to {total} but axis size is {axis_size}"),
                });
            }

            // Compute strides (row-major)
            let mut strides = vec![1_usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * dims[i + 1] as usize;
            }

            // For now, Split returns the first section as a single Value.
            // Multi-output support: we return a Tensor containing concatenated sections.
            // Since our Value type doesn't support tuples, we return the first section only
            // (consistent with how Scan and other multi-output ops work — the caller
            // handles unpacking through the equation's output vars).

            // Actually, the simplest approach: return all sections concatenated as a single
            // tensor with an extra leading dimension = num_sections.
            // But the bead says "Returns multiple outputs" which our Value can't directly do.
            // For eval_primitive which returns a single Value, we'll return a reshaped tensor.
            // The correct approach: reshape to [num_sections, section_size, ...rest_dims].

            let num_sections = sizes.len();
            if sizes.windows(2).all(|w| w[0] == w[1]) || sizes.len() == 1 {
                // Equal split — reshape into [num_sections, section_size, ...rest]
                let section_size = sizes[0];
                let mut new_dims = Vec::with_capacity(rank + 1);
                for (i, &d) in dims.iter().enumerate() {
                    if i == axis {
                        new_dims.push(num_sections as u32);
                        new_dims.push(section_size as u32);
                    } else {
                        new_dims.push(d);
                    }
                }
                Ok(Value::Tensor(
                    TensorValue::new(
                        tensor.dtype,
                        Shape { dims: new_dims },
                        tensor.elements.clone(),
                    )
                    .map_err(|e| EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    })?,
                ))
            } else {
                // Unequal split — extract the sections using slicing
                // For simplicity, return the first section
                let section_size = sizes[0];
                let mut new_dims = dims.to_vec();
                new_dims[axis] = section_size as u32;

                let elements_per_slice: usize = strides[axis] * section_size;
                let outer_size: usize = if axis == 0 {
                    1
                } else {
                    dims[..axis].iter().map(|&d| d as usize).product()
                };

                let mut result = Vec::new();
                for outer in 0..outer_size {
                    let base = outer * strides[axis] * axis_size;
                    for i in 0..elements_per_slice {
                        result.push(tensor.elements[base + i]);
                    }
                }

                Ok(Value::Tensor(
                    TensorValue::new(tensor.dtype, Shape { dims: new_dims }, result).map_err(
                        |e| EvalError::Unsupported {
                            primitive,
                            detail: e.to_string(),
                        },
                    )?,
                ))
            }
        }
    }
}

// ── ExpandDims: add a singleton dimension ──────────────────────

pub(crate) fn eval_expand_dims(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ExpandDims;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let axis_vec = parse_usize_param(primitive, "axis", params)?;
    let axis = axis_vec[0];

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Scalar -> rank-1 tensor of shape [1]
            Ok(Value::Tensor(
                TensorValue::new(
                    match lit {
                        Literal::I64(_) => DType::I64,
                        Literal::U32(_) => DType::U32,
                        Literal::U64(_) => DType::U64,
                        Literal::Bool(_) => DType::Bool,
                        Literal::BF16Bits(_) => DType::BF16,
                        Literal::F16Bits(_) => DType::F16,
                        Literal::F32Bits(_) => DType::F32,
                        Literal::F64Bits(_) => DType::F64,
                        Literal::Complex64Bits(..) => DType::Complex64,
                        Literal::Complex128Bits(..) => DType::Complex128,
                    },
                    Shape { dims: vec![1] },
                    vec![*lit],
                )
                .map_err(|e| EvalError::Unsupported {
                    primitive,
                    detail: e.to_string(),
                })?,
            ))
        }
        Value::Tensor(tensor) => {
            let rank = tensor.shape.dims.len();
            if axis > rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {axis} out of range for rank {rank} (max is {rank})"),
                });
            }

            let mut new_dims = tensor.shape.dims.clone();
            new_dims.insert(axis, 1);

            Ok(Value::Tensor(
                TensorValue::new(
                    tensor.dtype,
                    Shape { dims: new_dims },
                    tensor.elements.clone(),
                )
                .map_err(|e| EvalError::Unsupported {
                    primitive,
                    detail: e.to_string(),
                })?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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
    fn extract_f64_vec(val: &Value) -> Vec<f64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    }
    fn extract_shape(val: &Value) -> Vec<u32> {
        val.as_tensor().unwrap().shape.dims.clone()
    }
    fn params(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    // ── Reshape ──

    #[test]
    fn reshape_1d_to_2d() {
        let x = v_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("new_shape", "2,3")]);
        let result = eval_reshape(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![2, 3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_2d_to_1d() {
        let x = mat_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("new_shape", "6")]);
        let result = eval_reshape(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![6]);
    }

    #[test]
    fn reshape_rejects_tensor_dim_above_u32() {
        let x =
            Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap());
        let p = params(&[("new_shape", "4294967296")]);
        let err = eval_reshape(&[x], &p).unwrap_err().to_string();

        assert!(
            err.contains("new_shape dim 4294967296 exceeds u32 range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reshape_rejects_scalar_dim_above_u32() {
        let p = params(&[("new_shape", "4294967296")]);
        let err = eval_reshape(&[Value::scalar_i64(1)], &p)
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("new_shape dim 4294967296 exceeds u32 range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reshape_rejects_shape_product_overflow() {
        let x =
            Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap());
        let p = params(&[("new_shape", "4294967295,4294967295,4294967295")]);
        let err = eval_reshape(&[x], &p).unwrap_err().to_string();

        assert!(
            err.contains("reshape known dimension product overflows usize"),
            "unexpected error: {err}"
        );
    }

    // ── Transpose ──

    #[test]
    fn transpose_2x3() {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let x = mat_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("permutation", "1,0")]);
        let result = eval_transpose(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3, 2]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity() {
        let x = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let p = params(&[("permutation", "0,1")]);
        let result = eval_transpose(std::slice::from_ref(&x), &p).unwrap();
        assert_eq!(extract_f64_vec(&result), extract_f64_vec(&x));
    }

    // ── Slice ──

    #[test]
    fn slice_1d() {
        let x = v_f64(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let p = params(&[("start_indices", "1"), ("limit_indices", "4")]);
        let result = eval_slice(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_1d_from_start() {
        let x = v_f64(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let p = params(&[("start_indices", "0"), ("limit_indices", "2")]);
        let result = eval_slice(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![10.0, 20.0]);
    }

    #[test]
    fn slice_empty_huge_trailing_shape_returns_empty() {
        let huge = u32::MAX;
        let x = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let p = params(&[
            ("start_indices", "0,0,0,0,0"),
            (
                "limit_indices",
                "0,4294967295,4294967295,4294967295,4294967295",
            ),
        ]);

        let result = eval_slice(&[x], &p).unwrap();

        assert_eq!(extract_shape(&result), vec![0, huge, huge, huge, huge]);
        assert!(result.as_tensor().unwrap().elements.is_empty());
    }

    // ── Concatenate ──

    #[test]
    fn concatenate_1d() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[3.0, 4.0, 5.0]);
        let p = params(&[("dimension", "0")]);
        let result = eval_concatenate(&[a, b], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ── Reverse ──

    #[test]
    fn rev_1d() {
        let x = v_f64(&[1.0, 2.0, 3.0, 4.0]);
        let p = params(&[("axes", "0")]);
        let result = eval_rev(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 2.0, 1.0]);
    }

    // ── Iota ──

    #[test]
    fn iota_1d() {
        let p = params(&[("length", "5"), ("dtype", "F64")]);
        let result = eval_iota(&[], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // ── One-hot ──

    #[test]
    fn one_hot_basic() {
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3] },
                vec![Literal::I64(0), Literal::I64(1), Literal::I64(2)],
            )
            .unwrap(),
        );
        let p = params(&[("num_classes", "4"), ("dtype", "f64")]);
        let result = eval_one_hot(&[indices], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3, 4]);
        let vals = extract_f64_vec(&result);
        // Row 0: [1,0,0,0], Row 1: [0,1,0,0], Row 2: [0,0,1,0]
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[4], 0.0);
        assert_eq!(vals[5], 1.0);
        assert_eq!(vals[10], 1.0);
    }

    // ── Scatter ──

    #[test]
    fn scatter_scalar_index_accepts_scalar_update() {
        let operand = v_f64(&[1.0, 2.0, 3.0]);
        let index = Value::Scalar(Literal::I64(1));
        let update = Value::Scalar(Literal::from_f64(9.0));
        let result = eval_scatter(&[operand, index, update], &BTreeMap::new()).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 9.0, 3.0]);
    }

    #[test]
    fn scatter_scalar_update_rejects_non_scalar_slice() {
        let operand = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let index = Value::Scalar(Literal::I64(1));
        let update = Value::Scalar(Literal::from_f64(9.0));
        let err = eval_scatter(&[operand, index, update], &BTreeMap::new())
            .expect_err("scalar update should not fill non-scalar scatter slices");
        assert!(
            err.to_string().contains("updates must be a tensor"),
            "unexpected error: {err}"
        );
    }

    // ── Sort ──

    #[test]
    fn sort_1d_ascending() {
        let x = v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "false"),
        ]);
        let result = eval_sort(Primitive::Sort, &[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_1d_descending() {
        let x = v_f64(&[3.0, 1.0, 4.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "true"),
        ]);
        let result = eval_sort(Primitive::Sort, &[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 1.0]);
    }

    // ── Argsort ──

    #[test]
    fn argsort_1d() {
        let x = v_f64(&[30.0, 10.0, 20.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "false"),
        ]);
        let result = eval_argsort(Primitive::Argsort, &[x], &p).unwrap();
        let indices: Vec<i64> = result
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().expect("expected integer index"))
            .collect();
        assert_eq!(indices, vec![1, 2, 0]);
    }

    // ── Copy ──

    #[test]
    fn copy_preserves_value() {
        let x = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_copy(std::slice::from_ref(&x)).unwrap();
        assert_eq!(extract_f64_vec(&result), extract_f64_vec(&x));
    }

    // ── Squeeze ──

    #[test]
    fn squeeze_removes_unit_dim() {
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![1, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let p = params(&[("dimensions", "0")]);
        let result = eval_squeeze(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
    }

    // ── ExpandDims ──

    #[test]
    fn expand_dims_adds_unit_dim() {
        let x = v_f64(&[1.0, 2.0, 3.0]);
        let p = params(&[("axis", "0")]);
        let result = eval_expand_dims(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![1, 3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn expand_dims_trailing() {
        let x = v_f64(&[1.0, 2.0]);
        let p = params(&[("axis", "1")]);
        let result = eval_expand_dims(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![2, 1]);
    }
}
