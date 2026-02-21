#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use std::collections::BTreeMap;

use crate::EvalError;

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

/// Reshape: change the shape of a tensor without changing its data.
/// Params: `new_shape` (comma-separated dims, -1 for a single inferred axis).
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
            // Scalar -> rank-N tensor (all dims must multiply to 1)
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
                if known_product == 0 || !elem_count.is_multiple_of(known_product) {
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

/// Gather: index into an operand tensor using an indices tensor.
///
/// Simplified semantics (1-D index gather):
///   operand: tensor of any rank
///   indices: 1-D integer tensor of gather indices (into axis 0 of operand)
///   params:  `slice_sizes` — comma-separated sizes for each axis of the gathered slice
///
/// For each index i in `indices`, extracts a slice of shape `slice_sizes` starting
/// at position `[indices[i], 0, 0, ...]` from `operand`.
/// Output shape: `[num_indices] ++ slice_sizes[1..]` (the leading axis is replaced by
/// the number of gathered indices, remaining axes keep their slice sizes).
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

    // Extract indices as flat i64 values
    let index_vals: Vec<usize> = match &inputs[1] {
        Value::Scalar(lit) => {
            vec![lit_to_usize(lit, primitive)?]
        }
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|lit| lit_to_usize(lit, primitive))
            .collect::<Result<_, _>>()?,
    };

    let num_indices = index_vals.len();
    let rank = operand.shape.rank();

    // Compute operand strides (row-major)
    let op_dims = &operand.shape.dims;
    let mut op_strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        op_strides[i] = op_strides[i + 1] * op_dims[i + 1] as usize;
    }

    // Output shape: [num_indices, slice_sizes[1], slice_sizes[2], ...]
    let mut out_dims: Vec<u32> = vec![num_indices as u32];
    for &s in &slice_sizes[1..] {
        out_dims.push(s as u32);
    }

    // Number of elements per gathered slice (product of slice_sizes[1..])
    let slice_elems: usize = slice_sizes[1..].iter().product::<usize>();

    let total = num_indices * slice_elems;
    let mut elements = Vec::with_capacity(total);

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

    let index_vals: Vec<usize> = match &inputs[1] {
        Value::Scalar(lit) => vec![lit_to_usize(lit, primitive)?],
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|lit| lit_to_usize(lit, primitive))
            .collect::<Result<_, _>>()?,
    };

    let updates = match &inputs[2] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "updates must be a tensor".into(),
            });
        }
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

        for j in 0..slice_elems {
            if update_offset + j < updates.elements.len() {
                let current = &result_elements[base_offset + j];
                let update = &updates.elements[update_offset + j];
                if mode == "add" {
                    let c_val = current.as_f64().unwrap_or(0.0);
                    let u_val = update.as_f64().unwrap_or(0.0);
                    result_elements[base_offset + j] = Literal::from_f64(c_val + u_val);
                } else {
                    result_elements[base_offset + j] = *update;
                }
            }
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
        Literal::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Literal::F64Bits(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "float indices not supported".into(),
        }),
    }
}
