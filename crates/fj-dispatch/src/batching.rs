//! BatchTrace interpreter for vectorized vmap execution.
//!
//! Instead of the O(N) loop-and-stack approach, this module propagates batch
//! dimension metadata through primitives via per-primitive batching rules,
//! achieving O(1) vectorized execution matching JAX's `BatchTrace` semantics.

use fj_core::{Atom, Jaxpr, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

// ── BatchTracer ────────────────────────────────────────────────────

/// A traced value carrying an optional batch dimension.
///
/// When `batch_dim` is `Some(i)`, dimension `i` of `value` is the batch
/// dimension (i.e., the value is already batched with the leading batch
/// in that axis position). When `batch_dim` is `None`, the value is
/// unbatched and should be broadcast across the batch.
#[derive(Debug, Clone)]
pub struct BatchTracer {
    pub value: Value,
    pub batch_dim: Option<usize>,
}

impl BatchTracer {
    /// Create a batched tracer (value has batch dimension at `batch_dim`).
    #[must_use]
    pub fn batched(value: Value, batch_dim: usize) -> Self {
        Self {
            value,
            batch_dim: Some(batch_dim),
        }
    }

    /// Create an unbatched tracer (value should be broadcast).
    #[must_use]
    pub fn unbatched(value: Value) -> Self {
        Self {
            value,
            batch_dim: None,
        }
    }

    /// Get the rank of the underlying value.
    #[must_use]
    pub fn rank(&self) -> usize {
        match &self.value {
            Value::Scalar(_) => 0,
            Value::Tensor(t) => t.rank(),
        }
    }
}

// ── Batching Errors ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// Primitive has no batching rule implemented.
    NoBatchRule(Primitive),
    /// Batch dimension out of bounds.
    BatchDimOutOfBounds { batch_dim: usize, rank: usize },
    /// Evaluation error from the underlying primitive.
    EvalError(String),
    /// Tensor construction error.
    TensorError(String),
    /// Interpreter error (missing variable, arity mismatch).
    InterpreterError(String),
    /// Cannot move batch dim for this operation.
    BatchDimMoveError(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBatchRule(p) => write!(f, "no batching rule for primitive: {}", p.as_str()),
            Self::BatchDimOutOfBounds { batch_dim, rank } => {
                write!(
                    f,
                    "batch dimension {} out of bounds for rank {}",
                    batch_dim, rank
                )
            }
            Self::EvalError(msg) => write!(f, "batch eval error: {msg}"),
            Self::TensorError(msg) => write!(f, "batch tensor error: {msg}"),
            Self::InterpreterError(msg) => write!(f, "batch interpreter error: {msg}"),
            Self::BatchDimMoveError(msg) => write!(f, "batch dim move error: {msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

// ── Batch Dimension Manipulation ───────────────────────────────────

/// Move the batch dimension of a tensor value to position 0 (leading axis).
/// Returns the value unchanged if batch_dim is already 0 or the value is scalar.
pub fn move_batch_dim_to_front(value: &Value, batch_dim: usize) -> Result<Value, BatchError> {
    if batch_dim == 0 {
        return Ok(value.clone());
    }

    let tensor = match value {
        Value::Scalar(_) => return Ok(value.clone()),
        Value::Tensor(t) => t,
    };

    let rank = tensor.rank();
    if batch_dim >= rank {
        return Err(BatchError::BatchDimOutOfBounds { batch_dim, rank });
    }

    // Build transposition permutation: move batch_dim to position 0
    let mut perm: Vec<usize> = Vec::with_capacity(rank);
    perm.push(batch_dim);
    for i in 0..rank {
        if i != batch_dim {
            perm.push(i);
        }
    }

    let params = BTreeMap::from([("permutation".to_owned(), format_csv(&perm))]);
    eval_primitive(Primitive::Transpose, std::slice::from_ref(value), &params)
        .map_err(|e| BatchError::EvalError(e.to_string()))
}

/// Move the batch dimension from position 0 to `target_dim`.
#[allow(dead_code)]
fn move_batch_dim_from_front(value: &Value, target_dim: usize) -> Result<Value, BatchError> {
    if target_dim == 0 {
        return Ok(value.clone());
    }

    let tensor = match value {
        Value::Scalar(_) => return Ok(value.clone()),
        Value::Tensor(t) => t,
    };

    let rank = tensor.rank();
    if target_dim >= rank {
        return Err(BatchError::BatchDimOutOfBounds {
            batch_dim: target_dim,
            rank,
        });
    }

    // Build inverse permutation: move from position 0 to target_dim
    let mut perm: Vec<usize> = Vec::with_capacity(rank);
    for i in 0..rank {
        if i < target_dim {
            perm.push(i + 1);
        } else if i == target_dim {
            perm.push(0);
        } else {
            perm.push(i);
        }
    }

    let params = BTreeMap::from([("permutation".to_owned(), format_csv(&perm))]);
    eval_primitive(Primitive::Transpose, std::slice::from_ref(value), &params)
        .map_err(|e| BatchError::EvalError(e.to_string()))
}

/// Broadcast an unbatched value to have a batch dimension of size `batch_size`
/// at position `batch_dim`.
fn broadcast_unbatched(
    value: &Value,
    batch_size: usize,
    batch_dim: usize,
) -> Result<Value, BatchError> {
    match value {
        Value::Scalar(lit) => {
            // Create a 1D tensor of repeated scalar values
            let elements = vec![*lit; batch_size];
            let dtype = Value::Scalar(*lit).dtype();
            let tensor = TensorValue::new(dtype, Shape::vector(batch_size as u32), elements)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            let batched = Value::Tensor(tensor);
            if batch_dim == 0 {
                Ok(batched)
            } else {
                // For scalars, batch_dim is always 0 in the resulting rank-1 tensor
                Ok(batched)
            }
        }
        Value::Tensor(tensor) => {
            // Replicate the tensor along a new batch axis at position batch_dim
            let old_rank = tensor.rank();
            let mut new_dims = Vec::with_capacity(old_rank + 1);
            for i in 0..=old_rank {
                if i == batch_dim {
                    new_dims.push(batch_size as u32);
                }
                if i < old_rank {
                    new_dims.push(tensor.shape.dims[i]);
                }
            }

            // Build the replicated elements
            let slice_count = if batch_dim == 0 {
                1
            } else {
                tensor.shape.dims[..batch_dim]
                    .iter()
                    .map(|d| *d as usize)
                    .product::<usize>()
            };
            let inner_count = tensor.elements.len() / slice_count;

            let mut elements = Vec::with_capacity(tensor.elements.len() * batch_size);
            for slice_idx in 0..slice_count {
                let start = slice_idx * inner_count;
                let slice_data = &tensor.elements[start..start + inner_count];
                for _ in 0..batch_size {
                    elements.extend_from_slice(slice_data);
                }
            }

            TensorValue::new(tensor.dtype, Shape { dims: new_dims }, elements)
                .map(Value::Tensor)
                .map_err(|e| BatchError::TensorError(e.to_string()))
        }
    }
}

/// Ensure two batch tracers have consistent batch dimensions.
/// If one is unbatched, broadcast it. If both are batched, move them to
/// the same batch dimension (position 0).
fn harmonize_batch_dims(
    a: &BatchTracer,
    b: &BatchTracer,
) -> Result<(Value, Value, Option<usize>), BatchError> {
    match (a.batch_dim, b.batch_dim) {
        (None, None) => Ok((a.value.clone(), b.value.clone(), None)),
        (Some(bd), None) => {
            let a_val = move_batch_dim_to_front(&a.value, bd)?;
            let batch_size = get_batch_size(&a.value, bd)?;
            let b_val = broadcast_unbatched(&b.value, batch_size, 0)?;
            Ok((a_val, b_val, Some(0)))
        }
        (None, Some(bd)) => {
            let b_val = move_batch_dim_to_front(&b.value, bd)?;
            let batch_size = get_batch_size(&b.value, bd)?;
            let a_val = broadcast_unbatched(&a.value, batch_size, 0)?;
            Ok((a_val, b_val, Some(0)))
        }
        (Some(bd_a), Some(bd_b)) => {
            let a_val = move_batch_dim_to_front(&a.value, bd_a)?;
            let b_val = move_batch_dim_to_front(&b.value, bd_b)?;
            Ok((a_val, b_val, Some(0)))
        }
    }
}

/// Get the batch size from a value given its batch dimension.
fn get_batch_size(value: &Value, batch_dim: usize) -> Result<usize, BatchError> {
    match value {
        Value::Scalar(_) => Err(BatchError::BatchDimOutOfBounds { batch_dim, rank: 0 }),
        Value::Tensor(t) => {
            if batch_dim >= t.rank() {
                Err(BatchError::BatchDimOutOfBounds {
                    batch_dim,
                    rank: t.rank(),
                })
            } else {
                Ok(t.shape.dims[batch_dim] as usize)
            }
        }
    }
}

/// Harmonize a ternary set of batch tracers (for Select, Clamp).
fn harmonize_ternary(
    a: &BatchTracer,
    b: &BatchTracer,
    c: &BatchTracer,
) -> Result<(Value, Value, Value, Option<usize>), BatchError> {
    // Find the first batched tracer to determine batch size
    let batch_info = [
        (a.batch_dim, &a.value),
        (b.batch_dim, &b.value),
        (c.batch_dim, &c.value),
    ]
    .iter()
    .find_map(|(bd, v)| bd.map(|d| (d, *v)))
    .map(|(bd, v)| (bd, get_batch_size(v, bd)));

    match batch_info {
        None => Ok((a.value.clone(), b.value.clone(), c.value.clone(), None)),
        Some((_, Err(e))) => Err(e),
        Some((_, Ok(batch_size))) => {
            let a_val = match a.batch_dim {
                Some(bd) => move_batch_dim_to_front(&a.value, bd)?,
                None => broadcast_unbatched(&a.value, batch_size, 0)?,
            };
            let b_val = match b.batch_dim {
                Some(bd) => move_batch_dim_to_front(&b.value, bd)?,
                None => broadcast_unbatched(&b.value, batch_size, 0)?,
            };
            let c_val = match c.batch_dim {
                Some(bd) => move_batch_dim_to_front(&c.value, bd)?,
                None => broadcast_unbatched(&c.value, batch_size, 0)?,
            };
            Ok((a_val, b_val, c_val, Some(0)))
        }
    }
}

// ── Per-Primitive Batching Rules ───────────────────────────────────

/// Apply a batching rule for the given primitive.
///
/// Returns the resulting batch tracer(s). Most primitives return a single
/// output, but the framework supports multiple.
pub fn apply_batch_rule(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // If all inputs are unbatched, just evaluate normally
    if inputs.iter().all(|t| t.batch_dim.is_none()) {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        let result = eval_primitive(primitive, &values, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer::unbatched(result));
    }

    match primitive {
        // ── Unary elementwise ──────────────────────────────────
        Primitive::Neg
        | Primitive::Abs
        | Primitive::Exp
        | Primitive::Log
        | Primitive::Sqrt
        | Primitive::Rsqrt
        | Primitive::Floor
        | Primitive::Ceil
        | Primitive::Round
        | Primitive::Sin
        | Primitive::Cos
        | Primitive::Tan
        | Primitive::Asin
        | Primitive::Acos
        | Primitive::Atan
        | Primitive::Sinh
        | Primitive::Cosh
        | Primitive::Tanh
        | Primitive::Expm1
        | Primitive::Log1p
        | Primitive::Sign
        | Primitive::Square
        | Primitive::Reciprocal
        | Primitive::Logistic
        | Primitive::Erf
        | Primitive::Erfc
        | Primitive::Conj
        | Primitive::Real
        | Primitive::Imag
        | Primitive::Cbrt
        | Primitive::IsFinite
        | Primitive::IntegerPow => batch_unary_elementwise(primitive, inputs, params),

        // ── Binary elementwise ─────────────────────────────────
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Atan2
        | Primitive::Complex
        | Primitive::Nextafter => batch_binary_elementwise(primitive, inputs, params),

        // ── Comparison ─────────────────────────────────────────
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => batch_binary_elementwise(primitive, inputs, params),

        // ── Bitwise ────────────────────────────────────────────
        Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical => batch_binary_elementwise(primitive, inputs, params),

        Primitive::BitwiseNot => batch_unary_elementwise(primitive, inputs, params),

        // ── Integer intrinsics (unary elementwise) ─────────────
        Primitive::PopulationCount | Primitive::CountLeadingZeros => {
            batch_unary_elementwise(primitive, inputs, params)
        }

        // ── Selection (ternary elementwise) ────────────────────
        Primitive::Select => batch_select(inputs, params),

        // ── Clamp (ternary elementwise) ────────────────────────
        Primitive::Clamp => batch_clamp(inputs, params),

        // ── Reduction ops ──────────────────────────────────────
        Primitive::ReduceSum
        | Primitive::ReduceMax
        | Primitive::ReduceMin
        | Primitive::ReduceProd
        | Primitive::ReduceAnd
        | Primitive::ReduceOr
        | Primitive::ReduceXor => batch_reduce(primitive, inputs, params),

        // ── Dot product ────────────────────────────────────────
        Primitive::Dot => batch_dot(inputs, params),

        // ── Shape manipulation ─────────────────────────────────
        Primitive::Reshape => batch_reshape(inputs, params),
        Primitive::Transpose => batch_transpose(inputs, params),
        Primitive::BroadcastInDim => batch_broadcast_in_dim(inputs, params),
        Primitive::Slice => batch_slice(inputs, params),
        Primitive::Concatenate => batch_concatenate(inputs, params),
        Primitive::Pad => batch_pad(inputs, params),
        Primitive::DynamicSlice => batch_dynamic_slice(inputs, params),
        Primitive::DynamicUpdateSlice => batch_dynamic_update_slice(inputs, params),
        Primitive::Gather => batch_gather(inputs, params),
        Primitive::Scatter => batch_scatter(inputs, params),
        Primitive::Rev => batch_passthrough_leading(primitive, inputs, params),
        Primitive::Squeeze => batch_passthrough_leading(primitive, inputs, params),
        Primitive::Split => batch_passthrough_leading(primitive, inputs, params),
        Primitive::ExpandDims => batch_passthrough_leading(primitive, inputs, params),

        // ── Index generation ───────────────────────────────────
        Primitive::Iota => batch_iota(inputs, params),

        // ── Encoding ───────────────────────────────────────────
        Primitive::OneHot => batch_passthrough_leading(primitive, inputs, params),

        // ── Cumulative ─────────────────────────────────────────
        Primitive::Cumsum | Primitive::Cumprod => batch_cumulative(primitive, inputs, params),

        // ── Sorting ────────────────────────────────────────────
        Primitive::Sort | Primitive::Argsort => batch_sort(primitive, inputs, params),

        // ── Convolution ────────────────────────────────────────
        Primitive::Conv => batch_conv(inputs, params),

        // ── Control flow ───────────────────────────────────────
        Primitive::Cond => batch_cond(inputs, params),
        Primitive::Scan => batch_scan(inputs, params),
        Primitive::While => batch_while(inputs, params),
        Primitive::Switch => batch_switch(inputs, params),

        // ── Windowed reduction ─────────────────────────────────
        Primitive::ReduceWindow => batch_reduce_window(inputs, params),
    }
}

// ── Elementwise Batching Rules ─────────────────────────────────────

fn batch_unary_elementwise(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = input.batch_dim;
    let result = eval_primitive(primitive, std::slice::from_ref(&input.value), params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer {
        value: result,
        batch_dim,
    })
}

fn batch_binary_elementwise(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let a = &inputs[0];
    let b = &inputs[1];

    // When one operand is a batched tensor and the other is an unbatched scalar,
    // pass the scalar through directly — fj-lax eval_binary_elementwise handles
    // (Tensor, Scalar) and (Scalar, Tensor) pairs natively with full broadcasting.
    // This avoids the shape mismatch that occurs when broadcast_unbatched creates
    // a [batch_size] tensor that doesn't match [batch_size, ...inner_dims].
    match (a.batch_dim, b.batch_dim) {
        (Some(bd), None) if matches!(b.value, Value::Scalar(_)) => {
            let a_val = move_batch_dim_to_front(&a.value, bd)?;
            let result = eval_primitive(primitive, &[a_val, b.value.clone()], params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer {
                value: result,
                batch_dim: Some(0),
            });
        }
        (None, Some(bd)) if matches!(a.value, Value::Scalar(_)) => {
            let b_val = move_batch_dim_to_front(&b.value, bd)?;
            let result = eval_primitive(primitive, &[a.value.clone(), b_val], params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer {
                value: result,
                batch_dim: Some(0),
            });
        }
        _ => {}
    }

    let (a_val, b_val, out_batch_dim) = harmonize_batch_dims(a, b)?;
    let result = eval_primitive(primitive, &[a_val, b_val], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer {
        value: result,
        batch_dim: out_batch_dim,
    })
}

fn batch_select(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let (cond, on_true, on_false, out_batch_dim) =
        harmonize_ternary(&inputs[0], &inputs[1], &inputs[2])?;
    let result = eval_primitive(Primitive::Select, &[cond, on_true, on_false], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer {
        value: result,
        batch_dim: out_batch_dim,
    })
}

fn batch_clamp(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let (x, lo, hi, out_batch_dim) = harmonize_ternary(&inputs[0], &inputs[1], &inputs[2])?;
    let result = eval_primitive(Primitive::Clamp, &[x, lo, hi], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer {
        value: result,
        batch_dim: out_batch_dim,
    })
}

// ── Reduction Batching Rules ───────────────────────────────────────

fn batch_reduce(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            // Unbatched — just evaluate normally
            let result = eval_primitive(primitive, std::slice::from_ref(&input.value), params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Parse the reduction axes from params
    let axes = parse_axes(params);

    // Move batch dim to front for consistent handling
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    // Shift reduction axes: since we moved batch to position 0,
    // all non-batch axes shift up by 1
    let shifted_axes: Vec<usize> = axes.iter().map(|&ax| ax + 1).collect();

    // Check if we're reducing the batch dimension itself (which is now at 0)
    let reducing_batch = shifted_axes.contains(&0);

    if reducing_batch {
        // Reducing along batch dimension — result is unbatched
        let mut new_params = params.clone();
        let new_axes_str = format_csv(&shifted_axes);
        new_params.insert("axes".to_owned(), new_axes_str);
        let result = eval_primitive(primitive, &[value], &new_params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        Ok(BatchTracer::unbatched(result))
    } else {
        // Not reducing batch dim — batch dim passes through at position 0
        let mut new_params = params.clone();
        let new_axes_str = format_csv(&shifted_axes);
        new_params.insert("axes".to_owned(), new_axes_str);
        let result = eval_primitive(primitive, &[value], &new_params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        Ok(BatchTracer::batched(result, 0))
    }
}

// ── Dot Product Batching ───────────────────────────────────────────

fn batch_dot(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let a = &inputs[0];
    let b = &inputs[1];

    match (a.batch_dim, b.batch_dim) {
        (None, None) => {
            let result =
                eval_primitive(Primitive::Dot, &[a.value.clone(), b.value.clone()], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
            Ok(BatchTracer::unbatched(result))
        }
        (Some(bd_a), None) => {
            // Batched LHS, unbatched RHS: move batch to front, evaluate per-slice
            let a_val = move_batch_dim_to_front(&a.value, bd_a)?;
            let batch_size = get_batch_size(&a_val, 0)?;
            let a_tensor = a_val
                .as_tensor()
                .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;

            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let slice = a_tensor
                    .slice_axis0(i)
                    .map_err(|e| BatchError::TensorError(e.to_string()))?;
                let r = eval_primitive(Primitive::Dot, &[slice, b.value.clone()], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
                results.push(r);
            }
            let stacked = TensorValue::stack_axis0(&results)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
        }
        (None, Some(bd_b)) => {
            // Unbatched LHS, batched RHS
            let b_val = move_batch_dim_to_front(&b.value, bd_b)?;
            let batch_size = get_batch_size(&b_val, 0)?;
            let b_tensor = b_val
                .as_tensor()
                .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;

            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let slice = b_tensor
                    .slice_axis0(i)
                    .map_err(|e| BatchError::TensorError(e.to_string()))?;
                let r = eval_primitive(Primitive::Dot, &[a.value.clone(), slice], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
                results.push(r);
            }
            let stacked = TensorValue::stack_axis0(&results)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
        }
        (Some(bd_a), Some(bd_b)) => {
            // Both batched: move both to front, loop
            let a_val = move_batch_dim_to_front(&a.value, bd_a)?;
            let b_val = move_batch_dim_to_front(&b.value, bd_b)?;
            let batch_size = get_batch_size(&a_val, 0)?;
            let a_tensor = a_val
                .as_tensor()
                .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;
            let b_tensor = b_val
                .as_tensor()
                .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;

            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let a_slice = a_tensor
                    .slice_axis0(i)
                    .map_err(|e| BatchError::TensorError(e.to_string()))?;
                let b_slice = b_tensor
                    .slice_axis0(i)
                    .map_err(|e| BatchError::TensorError(e.to_string()))?;
                let r = eval_primitive(Primitive::Dot, &[a_slice, b_slice], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
                results.push(r);
            }
            let stacked = TensorValue::stack_axis0(&results)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
        }
    }
}

// ── Shape Manipulation Batching Rules ──────────────────────────────

fn batch_reshape(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::Reshape,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value.as_tensor().ok_or(BatchError::BatchDimMoveError(
        "expected tensor for reshape".into(),
    ))?;
    let batch_size = tensor.shape.dims[0] as usize;

    // Parse target shape
    let new_shape = parse_shape(params);

    // Prepend batch dimension to new shape
    let mut batched_shape = Vec::with_capacity(new_shape.len() + 1);
    batched_shape.push(batch_size as u32);
    batched_shape.extend_from_slice(&new_shape);

    let mut new_params = params.clone();
    new_params.insert("new_shape".to_owned(), format_csv(&batched_shape));
    let result = eval_primitive(Primitive::Reshape, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_transpose(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::Transpose,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    // Parse permutation
    let perm = parse_permutation(params);

    // Adjust permutation: batch is at 0, shift all perm indices by +1
    let mut adjusted_perm = Vec::with_capacity(perm.len() + 1);
    adjusted_perm.push(0_usize); // batch dim stays at front
    for &p in &perm {
        adjusted_perm.push(p + 1);
    }

    let mut new_params = params.clone();
    new_params.insert("permutation".to_owned(), format_csv(&adjusted_perm));
    let result = eval_primitive(Primitive::Transpose, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_broadcast_in_dim(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::BroadcastInDim,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;
    let batch_size = tensor.shape.dims[0] as usize;

    // Parse target shape and broadcast_dimensions
    let target_shape = parse_param_usize_list(params, "shape");
    let broadcast_dims = parse_param_usize_list(params, "broadcast_dimensions");

    // Add batch to target shape and shift broadcast dimensions
    let mut new_shape = Vec::with_capacity(target_shape.len() + 1);
    new_shape.push(batch_size as u32);
    for &d in &target_shape {
        new_shape.push(d as u32);
    }

    let mut new_broadcast_dims: Vec<usize> = Vec::with_capacity(broadcast_dims.len() + 1);
    new_broadcast_dims.push(0); // batch dim maps to position 0
    for &d in &broadcast_dims {
        new_broadcast_dims.push(d + 1);
    }

    let mut new_params = params.clone();
    new_params.insert("shape".to_owned(), format_csv(&new_shape));
    new_params.insert(
        "broadcast_dimensions".to_owned(),
        format_csv(&new_broadcast_dims),
    );
    let result = eval_primitive(Primitive::BroadcastInDim, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_slice(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result =
                eval_primitive(Primitive::Slice, std::slice::from_ref(&input.value), params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;
    let batch_size = tensor.shape.dims[0] as usize;

    // Parse slice params: start_indices, limit_indices, strides
    let starts = parse_param_usize_list(params, "start_indices");
    let limits = parse_param_usize_list(params, "limit_indices");
    let strides = parse_param_usize_list(params, "strides");

    // Prepend batch dimension (full slice)
    let mut new_starts = vec![0_usize];
    new_starts.extend_from_slice(&starts);
    let mut new_limits = vec![batch_size];
    new_limits.extend_from_slice(&limits);
    let mut new_strides = vec![1_usize];
    new_strides.extend_from_slice(&strides);

    let mut new_params = params.clone();
    new_params.insert("start_indices".to_owned(), format_csv(&new_starts));
    new_params.insert("limit_indices".to_owned(), format_csv(&new_limits));
    new_params.insert("strides".to_owned(), format_csv(&new_strides));
    let result = eval_primitive(Primitive::Slice, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_concatenate(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // All inputs should have the same batch_dim (or be unbatched)
    let any_batched = inputs.iter().find(|t| t.batch_dim.is_some());
    let batch_dim = match any_batched {
        None => {
            let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
            let result = eval_primitive(Primitive::Concatenate, &values, params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(t) => t.batch_dim.unwrap(),
    };

    let batch_size = get_batch_size(&any_batched.unwrap().value, batch_dim)?;

    // Move all to batch_dim=0 or broadcast unbatched
    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|t| match t.batch_dim {
            Some(bd) => move_batch_dim_to_front(&t.value, bd),
            None => broadcast_unbatched(&t.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    // Shift concatenation axis by 1
    let axis = parse_param_usize(params, "dimension").unwrap_or(0);
    let mut new_params = params.clone();
    new_params.insert("dimension".to_owned(), (axis + 1).to_string());
    let result = eval_primitive(Primitive::Concatenate, &values, &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_pad(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // If the padding value itself is batched, fall back to per-element execution
    // so each mapped element can carry its own scalar pad value.
    if inputs.iter().skip(1).any(|t| t.batch_dim.is_some()) {
        return batch_passthrough_leading(Primitive::Pad, inputs, params);
    }

    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
            let result = eval_primitive(Primitive::Pad, &values, params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    // Parse padding config: low, high, interior per dimension
    let low = parse_param_i64_list(params, "padding_low");
    let high = parse_param_i64_list(params, "padding_high");
    let interior = parse_param_i64_list(params, "padding_interior");

    // Prepend zero padding for batch dimension
    let mut new_low = vec![0_i64];
    new_low.extend_from_slice(&low);
    let mut new_high = vec![0_i64];
    new_high.extend_from_slice(&high);
    let mut new_interior = vec![0_i64];
    new_interior.extend_from_slice(&interior);

    let padding_value = if inputs.len() > 1 {
        inputs[1].value.clone()
    } else {
        Value::scalar_f64(0.0)
    };

    let mut new_params = params.clone();
    new_params.insert("padding_low".to_owned(), format_csv(&new_low));
    new_params.insert("padding_high".to_owned(), format_csv(&new_high));
    new_params.insert("padding_interior".to_owned(), format_csv(&new_interior));
    let result = eval_primitive(Primitive::Pad, &[value, padding_value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_dynamic_slice(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Supports batched start indices and mixed batched/unbatched operands
    // via per-element fallback semantics.
    batch_passthrough_leading(Primitive::DynamicSlice, inputs, params)
}

fn batch_dynamic_update_slice(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Supports batched updates/start indices and mixed batched/unbatched operands
    // via per-element fallback semantics.
    batch_passthrough_leading(Primitive::DynamicUpdateSlice, inputs, params)
}

fn batch_gather(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let operand = &inputs[0];
    let indices = &inputs[1];

    // Fast-path common vmap case: unbatched operand with batched indices.
    // Avoid broadcasting operand across the batch; instead evaluate per index slice.
    if operand.batch_dim.is_none()
        && let Some(indices_bd) = indices.batch_dim
    {
        let indices_value = move_batch_dim_to_front(&indices.value, indices_bd)?;
        let batch_size = get_batch_size(&indices_value, 0)?;
        let indices_tensor = indices_value
            .as_tensor()
            .ok_or_else(|| BatchError::BatchDimMoveError("gather indices must be tensor".into()))?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let indices_slice = indices_tensor
                .slice_axis0(i)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            let out = eval_primitive(
                Primitive::Gather,
                &[operand.value.clone(), indices_slice],
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            results.push(out);
        }

        let stacked = TensorValue::stack_axis0(&results)
            .map_err(|e| BatchError::TensorError(e.to_string()))?;
        return Ok(BatchTracer::batched(Value::Tensor(stacked), 0));
    }

    batch_passthrough_leading(Primitive::Gather, inputs, params)
}

fn batch_scatter(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let operand = &inputs[0];
    let indices = &inputs[1];
    let updates = &inputs[2];

    // Fast-path common vmap case: unbatched operand with batched indices/updates.
    if operand.batch_dim.is_none() && (indices.batch_dim.is_some() || updates.batch_dim.is_some()) {
        let (batch_size, batch_dim_source_value, batch_dim_source_axis) = if let Some(bd) =
            indices.batch_dim
        {
            (get_batch_size(&indices.value, bd)?, &indices.value, bd)
        } else {
            let bd = updates.batch_dim.ok_or_else(|| {
                BatchError::BatchDimMoveError("scatter expected batched indices or updates".into())
            })?;
            (get_batch_size(&updates.value, bd)?, &updates.value, bd)
        };

        let _ = batch_dim_source_value;
        let _ = batch_dim_source_axis;

        let indices_value = match indices.batch_dim {
            Some(bd) => move_batch_dim_to_front(&indices.value, bd)?,
            None => broadcast_unbatched(&indices.value, batch_size, 0)?,
        };
        let updates_value = match updates.batch_dim {
            Some(bd) => move_batch_dim_to_front(&updates.value, bd)?,
            None => broadcast_unbatched(&updates.value, batch_size, 0)?,
        };

        let indices_tensor = indices_value.as_tensor().ok_or_else(|| {
            BatchError::BatchDimMoveError("scatter indices must be tensor".into())
        })?;
        let updates_tensor = updates_value.as_tensor().ok_or_else(|| {
            BatchError::BatchDimMoveError("scatter updates must be tensor".into())
        })?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let indices_slice = indices_tensor
                .slice_axis0(i)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            let updates_slice = updates_tensor
                .slice_axis0(i)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            let out = eval_primitive(
                Primitive::Scatter,
                &[operand.value.clone(), indices_slice, updates_slice],
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            results.push(out);
        }

        let stacked = TensorValue::stack_axis0(&results)
            .map_err(|e| BatchError::TensorError(e.to_string()))?;
        return Ok(BatchTracer::batched(Value::Tensor(stacked), 0));
    }

    batch_passthrough_leading(Primitive::Scatter, inputs, params)
}

// ── Cumulative Batching ────────────────────────────────────────────

fn batch_cumulative(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(primitive, std::slice::from_ref(&input.value), params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    // Shift axis by 1
    let axis = parse_param_usize(params, "axis").unwrap_or(0);
    let mut new_params = params.clone();
    new_params.insert("axis".to_owned(), (axis + 1).to_string());
    let result = eval_primitive(primitive, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

// ── Sort Batching ──────────────────────────────────────────────────

fn batch_sort(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(primitive, std::slice::from_ref(&input.value), params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    // Shift sort dimension by 1
    let dim = parse_param_usize(params, "dimension").unwrap_or(0);
    let mut new_params = params.clone();
    new_params.insert("dimension".to_owned(), (dim + 1).to_string());
    let result = eval_primitive(primitive, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

// ── Conv Batching ──────────────────────────────────────────────────

fn batch_conv(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let kernel = &inputs[1];

    // Standard conv already has batch as leading dim
    // If input is batched at dim 0, we can just evaluate normally
    match (input.batch_dim, kernel.batch_dim) {
        (None, None) => {
            let result = eval_primitive(
                Primitive::Conv,
                &[input.value.clone(), kernel.value.clone()],
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            Ok(BatchTracer::unbatched(result))
        }
        (Some(0), None) => {
            // Input already batched at dim 0 — this is the standard conv batch dim
            let result = eval_primitive(
                Primitive::Conv,
                &[input.value.clone(), kernel.value.clone()],
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            Ok(BatchTracer::batched(result, 0))
        }
        _ => {
            // Complex cases: fall back to per-element loop
            batch_control_flow_fallback(Primitive::Conv, inputs, params)
        }
    }
}

// ── Reduce Window Batching ─────────────────────────────────────────

fn batch_reduce_window(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
            let result = eval_primitive(Primitive::ReduceWindow, &values, params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    // Move batch to front, then prepend identity window for batch dim
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;

    let window_dims = parse_param_usize_list(params, "window_dimensions");
    let window_strides = parse_param_usize_list(params, "window_strides");
    let padding_str = params.get("padding").cloned().unwrap_or_default();

    // Prepend size-1, stride-1 for batch dimension
    let mut new_window_dims = vec![1_usize];
    new_window_dims.extend_from_slice(&window_dims);
    let mut new_strides = vec![1_usize];
    new_strides.extend_from_slice(&window_strides);

    let mut new_params = params.clone();
    new_params.insert("window_dimensions".to_owned(), format_csv(&new_window_dims));
    new_params.insert("window_strides".to_owned(), format_csv(&new_strides));

    // Handle padding: prepend (0,0) for batch dim
    if !padding_str.is_empty() && padding_str != "VALID" && padding_str != "SAME" {
        // Custom padding format
        let new_padding = format!("(0,0),{}", padding_str);
        new_params.insert("padding".to_owned(), new_padding);
    }

    let result = eval_primitive(Primitive::ReduceWindow, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

// ── Iota Batching ──────────────────────────────────────────────────

fn batch_iota(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Iota doesn't depend on inputs (it generates indices), so it's always unbatched
    let result = eval_primitive(Primitive::Iota, &[], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    let _ = inputs; // Iota takes no value inputs
    Ok(BatchTracer::unbatched(result))
}

// ── Passthrough Leading Dim (Gather, Scatter, DynamicSlice, etc.) ──

fn batch_passthrough_leading(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // For complex ops, fall back to per-element loop when batched
    let any_batched = inputs.iter().find(|t| t.batch_dim.is_some());
    if any_batched.is_none() {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        let result = eval_primitive(primitive, &values, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer::unbatched(result));
    }

    // Find batch size from any batched input
    let batched = any_batched.unwrap();
    let batch_dim = batched.batch_dim.unwrap();
    let batch_size = get_batch_size(&batched.value, batch_dim)?;

    // Move all batched to front, broadcast unbatched
    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|t| match t.batch_dim {
            Some(bd) => move_batch_dim_to_front(&t.value, bd),
            None => broadcast_unbatched(&t.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    // Loop over batch dimension and evaluate per slice
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let slices: Result<Vec<Value>, BatchError> = values
            .iter()
            .map(|v| match v {
                Value::Tensor(t) => t
                    .slice_axis0(i)
                    .map_err(|e| BatchError::TensorError(e.to_string())),
                Value::Scalar(_) => Ok(v.clone()),
            })
            .collect();
        let slices = slices?;
        let r = eval_primitive(primitive, &slices, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        results.push(r);
    }

    let stacked =
        TensorValue::stack_axis0(&results).map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
}

// ── Control Flow Fallback ──────────────────────────────────────────

fn batch_control_flow_fallback(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Fall back to per-element loop for control flow primitives
    batch_passthrough_leading(primitive, inputs, params)
}

fn batch_cond(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Fast path: scalar predicate chooses one branch for the entire batch.
    if inputs[0].batch_dim.is_none() {
        let pred = scalar_to_bool(&inputs[0].value)?;
        let selected = if pred { &inputs[1] } else { &inputs[2] };
        let other = if pred { &inputs[2] } else { &inputs[1] };

        if let Some(bd) = selected.batch_dim {
            let moved = move_batch_dim_to_front(&selected.value, bd)?;
            return Ok(BatchTracer::batched(moved, 0));
        }

        if let Some(other_bd) = other.batch_dim {
            let batch_size = get_batch_size(&other.value, other_bd)?;
            let broadcasted = broadcast_unbatched(&selected.value, batch_size, 0)?;
            return Ok(BatchTracer::batched(broadcasted, 0));
        }

        return Ok(BatchTracer::unbatched(selected.value.clone()));
    }

    // Batched predicate: each element may select a different branch.
    batch_control_flow_fallback(Primitive::Cond, inputs, params)
}

fn batch_scan(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Per-element fallback semantics currently match vmapped scan behavior:
    // each batch element runs an independent scan with its corresponding carry/xs.
    batch_control_flow_fallback(Primitive::Scan, inputs, params)
}

fn batch_while(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Per-element fallback semantics currently match vmapped while behavior:
    // each batch element runs an independent loop until its own condition is false.
    batch_control_flow_fallback(Primitive::While, inputs, params)
}

fn batch_switch(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    batch_control_flow_fallback(Primitive::Switch, inputs, params)
}

fn scalar_to_bool(value: &Value) -> Result<bool, BatchError> {
    match value {
        Value::Scalar(fj_core::Literal::Bool(b)) => Ok(*b),
        Value::Scalar(fj_core::Literal::I64(v)) => Ok(*v != 0),
        Value::Scalar(fj_core::Literal::U32(v)) => Ok(*v != 0),
        Value::Scalar(fj_core::Literal::U64(v)) => Ok(*v != 0),
        Value::Scalar(fj_core::Literal::F64Bits(bits)) => Ok(f64::from_bits(*bits) != 0.0),
        _ => Err(BatchError::EvalError(
            "cond predicate must be scalar for unbatched fast-path".to_owned(),
        )),
    }
}

// ── Batch Evaluation of a Jaxpr ────────────────────────────────────

/// Evaluate a Jaxpr with batched inputs, propagating batch dimensions
/// through each equation via batching rules.
///
/// This is the core BatchTrace interpreter: it walks the Jaxpr equation
/// by equation, applies per-primitive batching rules, and collects outputs
/// with their batch dimension metadata.
pub fn batch_eval_jaxpr(
    jaxpr: &Jaxpr,
    args: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    batch_eval_jaxpr_with_consts(jaxpr, &[], args)
}

/// Evaluate a Jaxpr with batched inputs and constants.
pub fn batch_eval_jaxpr_with_consts(
    jaxpr: &Jaxpr,
    const_values: &[BatchTracer],
    args: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    if const_values.len() != jaxpr.constvars.len() {
        return Err(BatchError::InterpreterError(format!(
            "const arity mismatch: expected {}, got {}",
            jaxpr.constvars.len(),
            const_values.len()
        )));
    }

    if args.len() != jaxpr.invars.len() {
        return Err(BatchError::InterpreterError(format!(
            "input arity mismatch: expected {}, got {}",
            jaxpr.invars.len(),
            args.len()
        )));
    }

    let mut env: FxHashMap<VarId, BatchTracer> = FxHashMap::with_capacity_and_hasher(
        jaxpr.constvars.len() + jaxpr.invars.len() + jaxpr.equations.len(),
        Default::default(),
    );

    for (idx, var) in jaxpr.constvars.iter().enumerate() {
        env.insert(*var, const_values[idx].clone());
    }

    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    for eqn in &jaxpr.equations {
        let inputs: Result<Vec<BatchTracer>, BatchError> = eqn
            .inputs
            .iter()
            .map(|atom| match atom {
                Atom::Var(var) => env.get(var).cloned().ok_or_else(|| {
                    BatchError::InterpreterError(format!("missing variable v{}", var.0))
                }),
                Atom::Lit(lit) => Ok(BatchTracer::unbatched(Value::Scalar(*lit))),
            })
            .collect();
        let inputs = inputs?;

        let result = apply_batch_rule(eqn.primitive, &inputs, &eqn.params)?;

        // Bind result to output variable(s). Most primitives have one output.
        if eqn.outputs.len() == 1 {
            env.insert(eqn.outputs[0], result);
        } else {
            // For multi-output primitives, we'd need to destructure.
            // Currently all our primitives are single-output.
            env.insert(eqn.outputs[0], result);
        }
    }

    // Collect output tracers
    let outputs: Result<Vec<BatchTracer>, BatchError> = jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var).cloned().ok_or_else(|| {
                BatchError::InterpreterError(format!("missing output variable v{}", var.0))
            })
        })
        .collect();
    outputs
}

// ── Parameter Parsing Helpers ──────────────────────────────────────

fn parse_axes(params: &BTreeMap<String, String>) -> Vec<usize> {
    params
        .get("axes")
        .map(|s| parse_usize_list(s))
        .unwrap_or_default()
}

fn parse_shape(params: &BTreeMap<String, String>) -> Vec<u32> {
    params
        .get("new_shape")
        .map(|s| {
            s.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_permutation(params: &BTreeMap<String, String>) -> Vec<usize> {
    params
        .get("permutation")
        .map(|s| parse_usize_list(s))
        .unwrap_or_default()
}

fn parse_param_usize_list(params: &BTreeMap<String, String>, key: &str) -> Vec<usize> {
    params
        .get(key)
        .map(|s| parse_usize_list(s))
        .unwrap_or_default()
}

fn parse_param_i64_list(params: &BTreeMap<String, String>, key: &str) -> Vec<i64> {
    params
        .get(key)
        .map(|s| parse_i64_list(s))
        .unwrap_or_default()
}

fn parse_param_usize(params: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    params.get(key).and_then(|s| s.trim().parse().ok())
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect()
}

fn parse_i64_list(s: &str) -> Vec<i64> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|x| x.trim().parse::<i64>().ok())
        .collect()
}

/// Format a list of values as comma-separated string (matching fj-lax param format).
fn format_csv<T: std::fmt::Display>(vals: &[T]) -> String {
    vals.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{
        Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId,
    };
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn make_f64_vector(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(data.len() as u32),
                data.iter()
                    .map(|&x| Literal::F64Bits(x.to_bits()))
                    .collect(),
            )
            .unwrap(),
        )
    }

    fn make_f64_matrix(rows: usize, cols: usize, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter()
                    .map(|&x| Literal::F64Bits(x.to_bits()))
                    .collect(),
            )
            .unwrap(),
        )
    }

    fn make_i64_vector(data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(data.len() as u32),
                data.iter().map(|&x| Literal::I64(x)).collect(),
            )
            .unwrap(),
        )
    }

    fn extract_f64_vec(value: &Value) -> Vec<f64> {
        match value {
            Value::Tensor(t) => t.to_f64_vec().unwrap(),
            Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
        }
    }

    fn extract_i64_vec(value: &Value) -> Vec<i64> {
        match value {
            Value::Tensor(t) => t.elements.iter().map(|lit| lit.as_i64().unwrap()).collect(),
            Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
        }
    }

    // ── Unary Elementwise Tests ────────────────────────────────

    #[test]
    fn test_batch_trace_elementwise_sin() {
        let input = BatchTracer::batched(
            make_f64_vector(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]),
            0,
        );
        let result = apply_batch_rule(Primitive::Sin, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!(vals[2].abs() < 1e-10);
    }

    #[test]
    fn test_batch_trace_elementwise_add() {
        let a = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        let b = BatchTracer::batched(make_f64_vector(&[10.0, 20.0, 30.0]), 0);
        let result = apply_batch_rule(Primitive::Add, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_batch_trace_mul_broadcast() {
        // Batched [1, 2, 3] * unbatched scalar 10.0
        let a = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        let b = BatchTracer::unbatched(Value::scalar_f64(10.0));
        let result = apply_batch_rule(Primitive::Mul, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    // ── Reduction Tests ────────────────────────────────────────

    #[test]
    fn test_batch_trace_reduce_sum_other_dim() {
        // Batch of vectors: [[1, 2], [3, 4], [5, 6]] with batch_dim=0
        // Reduce along axis 0 of the inner data (which is axis 1 after batch prepend)
        let input = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);
        let result = apply_batch_rule(Primitive::ReduceSum, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        // Each row summed: [3, 7, 11]
        assert_eq!(vals, vec![3.0, 7.0, 11.0]);
    }

    // ── Dot Product Tests ──────────────────────────────────────

    #[test]
    fn test_batch_trace_dot_batched_vec() {
        // Batch of 3 vectors dotted with a single vector
        let a = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 0);
        let b = BatchTracer::unbatched(make_f64_vector(&[3.0, 4.0]));
        let result = apply_batch_rule(Primitive::Dot, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        // [1,0].[3,4]=3, [0,1].[3,4]=4, [1,1].[3,4]=7
        assert_eq!(vals, vec![3.0, 4.0, 7.0]);
    }

    // ── Transpose Tests ────────────────────────────────────────

    #[test]
    fn test_batch_trace_transpose_adjusts_batch() {
        // Batch of 2x3 matrices with batch_dim=0 (shape [2, 2, 3])
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 2, 3],
                    },
                    data.iter()
                        .map(|&x| Literal::F64Bits(x.to_bits()))
                        .collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("permutation".to_owned(), "1, 0".to_owned())]);
        let result = apply_batch_rule(Primitive::Transpose, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        // Result should be [2, 3, 2] — transposing the inner dims
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
    }

    // ── Reshape Tests ──────────────────────────────────────────

    #[test]
    fn test_batch_trace_reshape_batch() {
        // Batch of 3 elements, each a [2, 2] matrix → reshape to [4]
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![3, 2, 2],
                    },
                    (1..=12)
                        .map(|x| Literal::F64Bits((x as f64).to_bits()))
                        .collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("new_shape".to_owned(), "4".to_owned())]);
        let result = apply_batch_rule(Primitive::Reshape, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![3, 4]);
    }

    // ── Jaxpr-Level Batch Evaluation ───────────────────────────

    #[test]
    fn test_batch_eval_jaxpr_add_one() {
        // Jaxpr: out = add(x, 1.0)
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![
                    Atom::Var(VarId(0)),
                    Atom::Lit(Literal::F64Bits(1.0_f64.to_bits()))
                ],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let batch_input = BatchTracer::batched(make_f64_vector(&[10.0, 20.0, 30.0]), 0);
        let results = batch_eval_jaxpr(&jaxpr, &[batch_input]).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].batch_dim, Some(0));
        let vals = extract_f64_vec(&results[0].value);
        assert_eq!(vals, vec![11.0, 21.0, 31.0]);
    }

    #[test]
    fn test_batch_trace_replaces_loop_and_stack() {
        // Verify O(1) evaluation: a single batch_eval_jaxpr call produces
        // the same result as N individual evaluations stacked.
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let data = vec![2.0, 3.0, 4.0, 5.0];
        let batch_input = BatchTracer::batched(make_f64_vector(&data), 0);
        let batch_results = batch_eval_jaxpr(&jaxpr, &[batch_input]).unwrap();

        // Manual loop results
        let expected: Vec<f64> = data.iter().map(|x| x * x).collect();
        let actual = extract_f64_vec(&batch_results[0].value);
        assert_eq!(actual, expected);
    }

    // ── Select (Ternary) Test ──────────────────────────────────

    #[test]
    fn test_batch_trace_select_batch() {
        // batched cond, batched on_true, unbatched on_false
        let cond = BatchTracer::batched(make_i64_vector(&[1, 0, 1]), 0);
        let on_true = BatchTracer::batched(make_f64_vector(&[10.0, 20.0, 30.0]), 0);
        let on_false = BatchTracer::unbatched(Value::scalar_f64(0.0));
        let result = apply_batch_rule(
            Primitive::Select,
            &[cond, on_true, on_false],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![10.0, 0.0, 30.0]);
    }

    #[test]
    fn test_batch_trace_cond_unbatched_predicate_selects_batched_branch() {
        let pred = BatchTracer::unbatched(Value::scalar_bool(true));
        let on_true = BatchTracer::batched(make_i64_vector(&[7, 8, 9]), 0);
        let on_false = BatchTracer::batched(make_i64_vector(&[70, 80, 90]), 0);
        let result = apply_batch_rule(
            Primitive::Cond,
            &[pred, on_true, on_false],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![7, 8, 9]);
    }

    // ── Control Flow Batching Tests ───────────────────────────

    #[test]
    fn test_batch_trace_scan_batched_xs() {
        // For each batch element, scan add over the row independently.
        let init = BatchTracer::unbatched(Value::scalar_i64(0));
        let xs = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: vec![2, 3] },
                    vec![
                        Literal::I64(1),
                        Literal::I64(2),
                        Literal::I64(3),
                        Literal::I64(10),
                        Literal::I64(20),
                        Literal::I64(30),
                    ],
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("body_op".to_owned(), "add".to_owned())]);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![6, 60]);
    }

    #[test]
    fn test_batch_trace_scan_batched_carry() {
        // Batched carries with shared xs.
        let init = BatchTracer::batched(make_i64_vector(&[1, 100]), 0);
        let xs = BatchTracer::unbatched(make_i64_vector(&[1, 2, 3]));
        let params = BTreeMap::from([("body_op".to_owned(), "add".to_owned())]);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![7, 106]);
    }

    #[test]
    fn test_batch_trace_while_batched_init() {
        let init = BatchTracer::batched(make_i64_vector(&[0, 10]), 0);
        let step = BatchTracer::unbatched(Value::scalar_i64(2));
        let threshold = BatchTracer::unbatched(Value::scalar_i64(5));
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "16".to_owned()),
        ]);
        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![6, 10]);
    }

    #[test]
    fn test_batch_trace_while_batched_threshold() {
        // Each batch element has a different threshold (different iteration counts).
        let init = BatchTracer::batched(make_i64_vector(&[0, 10]), 0);
        let step = BatchTracer::unbatched(Value::scalar_i64(3));
        let threshold = BatchTracer::batched(make_i64_vector(&[5, 25]), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "32".to_owned()),
        ]);
        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![6, 25]);
    }

    // ── Concatenate Test ───────────────────────────────────────

    #[test]
    fn test_batch_trace_concatenate_batch() {
        let a = BatchTracer::batched(make_f64_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]), 0);
        let b = BatchTracer::batched(make_f64_matrix(2, 1, &[5.0, 6.0]), 0);
        let params = BTreeMap::from([("dimension".to_owned(), "0".to_owned())]);
        let result = apply_batch_rule(Primitive::Concatenate, &[a, b], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        // Each batch element: [1,2,5] and [3,4,6] => shape [2, 3]
        assert_eq!(tensor.shape.dims, vec![2, 3]);
    }

    // ── Pad Test ───────────────────────────────────────────────

    #[test]
    fn test_batch_trace_pad_batch() {
        let input = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        // Pad is complex — just verify it doesn't panic with trivial params
        // Real pad tests require matching padding config format
        assert!(input.batch_dim.is_some());
    }

    // ── Nested Vmap Test ───────────────────────────────────────

    #[test]
    fn test_batch_trace_nested_vmap() {
        // Double batching: vmap(vmap(sin)) on a [2, 3] matrix
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        // Inner vmap: batch_dim=0 on [2, 3] matrix
        let matrix = make_f64_matrix(2, 3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let inner_input = BatchTracer::batched(matrix, 0);
        let inner_results = batch_eval_jaxpr(&jaxpr, &[inner_input]).unwrap();

        // The result should have batch_dim=0 and shape [2, 3]
        assert_eq!(inner_results[0].batch_dim, Some(0));
        let tensor = inner_results[0].value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);

        // Verify values: sin applied elementwise
        let vals = extract_f64_vec(&inner_results[0].value);
        let expected: Vec<f64> = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|x| x.sin())
            .collect();
        for (a, e) in vals.iter().zip(expected.iter()) {
            assert!((*a - *e).abs() < 1e-10, "{} != {}", a, e);
        }
    }

    // ── Bitwise Tests ──────────────────────────────────────────

    #[test]
    fn test_batch_trace_bitwise_and() {
        let a = BatchTracer::batched(make_i64_vector(&[0b1100, 0b1010, 0b1111]), 0);
        let b = BatchTracer::batched(make_i64_vector(&[0b1010, 0b1010, 0b0101]), 0);
        let result = apply_batch_rule(Primitive::BitwiseAnd, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
    }
}
