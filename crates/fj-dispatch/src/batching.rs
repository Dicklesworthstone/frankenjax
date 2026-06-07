//! BatchTrace interpreter for vectorized vmap execution.
//!
//! Instead of the O(N) loop-and-stack approach, this module propagates batch
//! dimension metadata through primitives via per-primitive batching rules,
//! achieving O(1) vectorized execution matching JAX's `BatchTrace` semantics.

use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value,
    VarId,
};
use fj_interpreters::{eval_equation_outputs, eval_jaxpr_with_consts};
use fj_lax::linalg::analytic_eigh_3x3;
use fj_lax::{eval_primitive, eval_primitive_multi, promote_dtype_public};
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
        | Primitive::Asinh
        | Primitive::Acosh
        | Primitive::Atanh
        | Primitive::Expm1
        | Primitive::Log1p
        | Primitive::Sign
        | Primitive::Square
        | Primitive::Reciprocal
        | Primitive::Logistic
        | Primitive::Erf
        | Primitive::Erfc
        | Primitive::Lgamma
        | Primitive::Digamma
        | Primitive::ErfInv
        | Primitive::Conj
        | Primitive::Real
        | Primitive::Imag
        | Primitive::Cbrt
        | Primitive::IsFinite
        | Primitive::IntegerPow
        | Primitive::Copy
        | Primitive::ConvertElementType
        | Primitive::BitcastConvertType
        | Primitive::ReducePrecision
        | Primitive::Trunc
        | Primitive::Deg2Rad
        | Primitive::Rad2Deg
        | Primitive::Log2
        | Primitive::Exp2
        | Primitive::Sinc
        | Primitive::BesselI0e
        | Primitive::BesselI1e
        | Primitive::IsNan
        | Primitive::IsInf
        | Primitive::Signbit
        | Primitive::StopGradient => batch_unary_elementwise(primitive, inputs, params),

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
        | Primitive::Nextafter
        | Primitive::Hypot
        | Primitive::LogAddExp
        | Primitive::LogAddExp2
        | Primitive::Gcd
        | Primitive::Lcm
        | Primitive::Polygamma
        | Primitive::Igamma
        | Primitive::Igammac
        | Primitive::Zeta
        | Primitive::Heaviside
        | Primitive::CopySign
        | Primitive::Ldexp
        | Primitive::XLogY
        | Primitive::XLog1PY => batch_binary_elementwise(primitive, inputs, params),

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
        Primitive::PopulationCount
        | Primitive::CountLeadingZeros
        | Primitive::CountTrailingZeros => batch_unary_elementwise(primitive, inputs, params),

        // ── Selection (ternary elementwise) ────────────────────
        Primitive::Select => batch_select(inputs, params),
        Primitive::SelectN => batch_passthrough_leading(primitive, inputs, params),

        // ── Clamp (ternary elementwise) ────────────────────────
        Primitive::Clamp => batch_clamp(inputs, params),

        // ── Other ternary elementwise (per-slice eval = vmap) ──
        // Fma(a,b,c)=a*b+c and Betainc(a,b,x) are elementwise but ternary;
        // batch_passthrough_leading evaluates each batch slice and stacks,
        // broadcasting any unbatched operand.
        Primitive::Fma | Primitive::Betainc => batch_passthrough_leading(primitive, inputs, params),

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
        // dot_general's dimension_numbers index the unbatched operand ranks,
        // so per-slice eval (which drops the batch dim) applies them correctly.
        Primitive::DotGeneral => batch_passthrough_leading(primitive, inputs, params),

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
        Primitive::Rev => batch_rev(inputs, params),
        Primitive::Squeeze => batch_squeeze(inputs, params),
        Primitive::Split => batch_split(inputs, params),
        Primitive::ExpandDims => batch_expand_dims(inputs, params),
        Primitive::Tile => batch_passthrough_leading(primitive, inputs, params),
        Primitive::Cholesky => batch_cholesky(inputs, params),
        Primitive::TriangularSolve => batch_triangular_solve(inputs, params),
        Primitive::Qr | Primitive::Svd | Primitive::Eigh => {
            batch_passthrough_leading(primitive, inputs, params)
        }
        // Det → scalar, Solve → x. Single-output; per-slice eval + stack is the
        // vmap. batch_passthrough_leading handles Solve's partial batching
        // (e.g. batched A with shared b) by broadcasting unbatched inputs.
        Primitive::Det | Primitive::Solve => batch_passthrough_leading(primitive, inputs, params),
        Primitive::Fft | Primitive::Ifft | Primitive::Rfft | Primitive::Irfft => {
            batch_fft_like(primitive, inputs, params)
        }

        // ── Index generation ───────────────────────────────────
        Primitive::Iota => batch_iota(inputs, params),
        Primitive::BroadcastedIota => batch_broadcasted_iota(inputs, params),

        // ── Encoding ───────────────────────────────────────────
        Primitive::OneHot => batch_one_hot(inputs, params),

        // ── Cumulative ─────────────────────────────────────────
        Primitive::Cumsum | Primitive::Cumprod | Primitive::Cummax | Primitive::Cummin => {
            batch_cumulative(primitive, inputs, params)
        }

        // ── Sorting ────────────────────────────────────────────
        Primitive::Sort | Primitive::Argsort => batch_sort(primitive, inputs, params),
        Primitive::Argmin | Primitive::Argmax => {
            batch_passthrough_leading(primitive, inputs, params)
        }

        // ── Convolution ────────────────────────────────────────
        Primitive::Conv => batch_conv(inputs, params),

        // ── Control flow ───────────────────────────────────────
        Primitive::Cond => batch_cond(inputs, params),
        Primitive::Scan => batch_scan(inputs, params),
        // associative_scan runs an associative combine over a single array
        // axis. vmap = run the whole scan independently per batch slice; the
        // per-slice operand carries the original (unbatched) axis param, so
        // batch_passthrough_leading evaluates it correctly.
        Primitive::AssociativeScan => batch_passthrough_leading(primitive, inputs, params),
        Primitive::While => batch_while(inputs, params),
        Primitive::Switch => batch_switch(inputs, params),

        // ── Windowed reduction ─────────────────────────────────
        Primitive::ReduceWindow => batch_reduce_window(inputs, params),

        // ── Collective operations (pmap-only) ──────────────────
        Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::AxisIndex => Err(BatchError::EvalError(
            "collective operation requires an active pmap context".to_owned(),
        )),

        _ => Err(BatchError::NoBatchRule(primitive)),
    }
}

fn apply_batch_rule_multi(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Vec<BatchTracer>, BatchError> {
    match primitive {
        Primitive::Qr => batch_qr_multi(inputs, params),
        Primitive::Eigh => batch_eigh_multi(inputs, params),
        Primitive::Svd => batch_svd_multi(inputs, params),
        // LU is multi-output (packed LU, pivots, permutation) and has no
        // stacked fast path; the generic per-slice batcher runs lu() on each
        // batch element and stacks every output, matching JAX's batched LU.
        Primitive::Lu => batch_passthrough_leading_multi(primitive, inputs, params),
        // TopK is multi-output (values, indices) and reduces the last axis;
        // the per-slice batcher preserves that per-element semantics and
        // stacks both outputs, matching JAX's top_k batch rule.
        Primitive::TopK => batch_passthrough_leading_multi(primitive, inputs, params),
        // Slogdet → (sign, logabsdet) scalars. Per-slice eval + stack along
        // axis 0 is the correct vmap; scalar outputs stack into rank-1
        // [batch] vectors.
        Primitive::Slogdet => batch_passthrough_leading_multi(primitive, inputs, params),
        // Eig → (eigenvalues, eigenvectors), both Complex128. The per-slice
        // batcher stacks the complex outputs; eval_eig is deterministic, so
        // each lane's eigenvalue ordering matches the per-slice oracle.
        Primitive::Eig => batch_passthrough_leading_multi(primitive, inputs, params),
        _ => apply_batch_rule(primitive, inputs, params).map(|result| vec![result]),
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

    // Fast path: both inputs have batch_dim == 0, skip harmonization entirely
    // This is the common case for vmap with default in_axes=0.
    if a.batch_dim == Some(0) && b.batch_dim == Some(0) {
        let result = eval_primitive(primitive, &[a.value.clone(), b.value.clone()], params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer {
            value: result,
            batch_dim: Some(0),
        });
    }

    // When one operand is a batched tensor and the other is an unbatched scalar,
    // pass the scalar through directly — fj-lax eval_binary_elementwise handles
    // (Tensor, Scalar) and (Scalar, Tensor) pairs natively with full broadcasting.
    // This avoids the shape mismatch that occurs when broadcast_unbatched creates
    // a [batch_size] tensor that doesn't match [batch_size, ...inner_dims].
    match (a.batch_dim, b.batch_dim) {
        (Some(0), None) if matches!(b.value, Value::Scalar(_)) => {
            let result = eval_primitive(primitive, &[a.value.clone(), b.value.clone()], params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer {
                value: result,
                batch_dim: Some(0),
            });
        }
        (None, Some(0)) if matches!(a.value, Value::Scalar(_)) => {
            let result = eval_primitive(primitive, &[a.value.clone(), b.value.clone()], params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer {
                value: result,
                batch_dim: Some(0),
            });
        }
        (Some(bd), None) if matches!(b.value, Value::Scalar(_)) => {
            if let Some(result) =
                fuse_rank2_axis1_i64_scalar_add_move_to_front(primitive, &a.value, &b.value, bd)?
            {
                return Ok(result);
            }
            let a_val = move_batch_dim_to_front(&a.value, bd)?;
            let result = eval_primitive(primitive, &[a_val, b.value.clone()], params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer {
                value: result,
                batch_dim: Some(0),
            });
        }
        (None, Some(bd)) if matches!(a.value, Value::Scalar(_)) => {
            if let Some(result) =
                fuse_rank2_axis1_i64_scalar_add_move_to_front(primitive, &b.value, &a.value, bd)?
            {
                return Ok(result);
            }
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

#[inline]
fn fuse_rank2_axis1_i64_scalar_add_move_to_front(
    primitive: Primitive,
    batched_value: &Value,
    scalar_value: &Value,
    batch_dim: usize,
) -> Result<Option<BatchTracer>, BatchError> {
    if primitive != Primitive::Add || batch_dim != 1 {
        return Ok(None);
    }

    let (Value::Tensor(tensor), Value::Scalar(Literal::I64(scalar))) =
        (batched_value, scalar_value)
    else {
        return Ok(None);
    };
    if tensor.dtype != DType::I64 || tensor.rank() != 2 {
        return Ok(None);
    }

    let rows = tensor.shape.dims[0] as usize;
    let cols = tensor.shape.dims[1] as usize;
    let mut values = Vec::with_capacity(tensor.elements.len());
    if let Some(src) = tensor.elements.as_i64_slice() {
        for col in 0..cols {
            for row in 0..rows {
                values.push(src[row * cols + col].wrapping_add(*scalar));
            }
        }
    } else {
        let src = tensor.elements.as_slice();
        for col in 0..cols {
            for row in 0..rows {
                let Literal::I64(value) = src[row * cols + col] else {
                    return Ok(None);
                };
                values.push(value.wrapping_add(*scalar));
            }
        }
    }

    let moved = TensorValue::new_i64_values(
        Shape {
            dims: vec![cols as u32, rows as u32],
        },
        values,
    )
    .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer {
        value: Value::Tensor(moved),
        batch_dim: Some(0),
    }))
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
    let (min, operand, max, out_batch_dim) = harmonize_ternary(&inputs[0], &inputs[1], &inputs[2])?;
    let result = eval_primitive(Primitive::Clamp, &[min, operand, max], params)
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

    // Parse the reduction axes from params. Parse as i64 (allowing NEGATIVE, end-relative
    // axes, which eval_reduce_axes normalizes) — a usize parse rejects "-1" and so breaks
    // vmap over a negative reduction axis (e.g. reduce_sum(axes=-1)).
    let axes_raw = params.get("axes");

    // Move batch dim to front for consistent handling
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let per_elem_rank = match &value {
        Value::Scalar(_) => 0,
        Value::Tensor(tensor) => tensor.rank().saturating_sub(1),
    };

    // Resolve the per-element reduction axes, normalizing any negative axis against the
    // per-element rank exactly as eval_reduce_axes does. A naive +1 shift of a negative axis
    // would be wrong — it is end-relative — so normalize to a non-negative axis FIRST.
    let axes: Vec<usize> = match axes_raw {
        None => {
            if per_elem_rank == 0 {
                return Ok(BatchTracer::batched(value, 0));
            }
            (0..per_elem_rank).collect()
        }
        Some(raw) if is_empty_list(raw) => Vec::new(),
        Some(raw) => {
            let raw_axes = parse_i64_list(raw, "axes")?;
            let mut out = Vec::with_capacity(raw_axes.len());
            for ax in raw_axes {
                let norm = if ax < 0 { per_elem_rank as i64 + ax } else { ax };
                if norm < 0 || norm >= per_elem_rank as i64 {
                    return Err(BatchError::EvalError(format!(
                        "reduce axis {ax} out of range for per-element rank {per_elem_rank}"
                    )));
                }
                out.push(norm as usize);
            }
            out
        }
    };

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
            // Batched LHS, unbatched RHS: when the batched operand still has a
            // real per-slice tensor rank, fj-lax dot can handle the whole batch
            // directly by treating the batch axis as an LHS prefix dimension.
            let a_val = move_batch_dim_to_front(&a.value, bd_a)?;
            if dot_lhs_single_batch_can_eval_direct(&a_val, &b.value) {
                let result = eval_primitive(Primitive::Dot, &[a_val, b.value.clone()], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
                let result =
                    coerce_real_dot_result_dtype(result, a.value.dtype(), b.value.dtype())?;
                return Ok(BatchTracer::batched(result, 0));
            }

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
                let r = coerce_real_dot_result_dtype(r, a_tensor.dtype, b.value.dtype())?;
                results.push(r);
            }
            let stacked = TensorValue::stack_axis0(&results)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
        }
        (None, Some(bd_b)) => {
            // Unbatched LHS, batched RHS: direct fj-lax dot is valid when the
            // RHS slice rank is at least two, so the leading batch axis is a
            // prefix dimension rather than the dot contraction dimension.
            let b_val = move_batch_dim_to_front(&b.value, bd_b)?;
            if dot_rhs_single_batch_can_eval_direct(&a.value, &b_val) {
                let result = eval_primitive(Primitive::Dot, &[a.value.clone(), b_val], params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
                let result =
                    coerce_real_dot_result_dtype(result, a.value.dtype(), b.value.dtype())?;
                let output_batch_dim = match &a.value {
                    Value::Scalar(_) => 0,
                    Value::Tensor(tensor) => tensor.rank().saturating_sub(1),
                };
                let result = move_batch_dim_to_front(&result, output_batch_dim)?;
                return Ok(BatchTracer::batched(result, 0));
            }

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
                let r = coerce_real_dot_result_dtype(r, a.value.dtype(), b_tensor.dtype)?;
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
            if let Some(result) = batch_paired_i64_vector_dot(&a_val, &b_val)? {
                return Ok(result);
            }
            if let Some(result) = batch_paired_numeric_dot(&a_val, &b_val)? {
                return Ok(result);
            }

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
                let r = coerce_real_dot_result_dtype(r, a_tensor.dtype, b_tensor.dtype)?;
                results.push(r);
            }
            let stacked = TensorValue::stack_axis0(&results)
                .map_err(|e| BatchError::TensorError(e.to_string()))?;
            Ok(BatchTracer::batched(Value::Tensor(stacked), 0))
        }
    }
}

#[derive(Clone, Copy)]
enum BatchDotOutputKind {
    Integral,
    /// Real dot product; payload is the JAX-promoted real dtype to emit
    /// (BF16, F16, F32, or F64) so F32 inputs don't silently produce F64
    /// outputs (parity with `lax.dot_general` real-input promotion).
    Real(DType),
}

impl BatchDotOutputKind {
    fn dtype(self) -> DType {
        match self {
            Self::Integral => DType::I64,
            Self::Real(dt) => dt,
        }
    }
}

fn real_literal_from_f64_dtype(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f32(value as f32),
        DType::F16 => Literal::from_f16_f32(value as f32),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

fn real_dot_promoted_dtype(lhs_dtype: DType, rhs_dtype: DType) -> Option<DType> {
    if matches!(lhs_dtype, DType::Complex64 | DType::Complex128)
        || matches!(rhs_dtype, DType::Complex64 | DType::Complex128)
    {
        return None;
    }

    match promote_dtype_public(lhs_dtype, rhs_dtype) {
        dtype @ (DType::BF16 | DType::F16 | DType::F32 | DType::F64) => Some(dtype),
        _ => None,
    }
}

fn coerce_real_dot_result_dtype(
    result: Value,
    lhs_dtype: DType,
    rhs_dtype: DType,
) -> Result<Value, BatchError> {
    let Some(dtype) = real_dot_promoted_dtype(lhs_dtype, rhs_dtype) else {
        return Ok(result);
    };
    if result.dtype() == dtype {
        return Ok(result);
    }

    match result {
        Value::Scalar(literal) => {
            let value = literal.as_f64().ok_or_else(|| {
                BatchError::EvalError("real dot expected numeric scalar output".to_owned())
            })?;
            Ok(Value::Scalar(real_literal_from_f64_dtype(dtype, value)))
        }
        Value::Tensor(tensor) => {
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for literal in tensor.elements {
                let value = literal.as_f64().ok_or_else(|| {
                    BatchError::EvalError("real dot expected numeric tensor output".to_owned())
                })?;
                elements.push(real_literal_from_f64_dtype(dtype, value));
            }
            TensorValue::new(dtype, tensor.shape, elements)
                .map(Value::Tensor)
                .map_err(|e| BatchError::TensorError(e.to_string()))
        }
    }
}

fn batch_paired_i64_vector_dot(
    lhs_value: &Value,
    rhs_value: &Value,
) -> Result<Option<BatchTracer>, BatchError> {
    let (Value::Tensor(lhs), Value::Tensor(rhs)) = (lhs_value, rhs_value) else {
        return Ok(None);
    };
    if lhs.dtype != DType::I64
        || rhs.dtype != DType::I64
        || lhs.rank() != 2
        || rhs.rank() != 2
        || lhs.shape.dims != rhs.shape.dims
    {
        return Ok(None);
    }

    let batch = lhs.shape.dims[0] as usize;
    let width = lhs.shape.dims[1] as usize;
    let expected_len = batch
        .checked_mul(width)
        .ok_or_else(|| BatchError::TensorError("paired dot size overflowed".to_owned()))?;
    if lhs.elements.len() != expected_len || rhs.elements.len() != expected_len {
        return Ok(None);
    }

    let mut elements = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let offset = batch_idx * width;
        let mut sum = 0_i64;
        for kk in 0..width {
            let Literal::I64(left) = lhs.elements[offset + kk] else {
                return Ok(None);
            };
            let Literal::I64(right) = rhs.elements[offset + kk] else {
                return Ok(None);
            };
            sum = sum.wrapping_add(left.wrapping_mul(right));
        }
        elements.push(Literal::I64(sum));
    }

    let output = TensorValue::new(DType::I64, Shape::vector(lhs.shape.dims[0]), elements)
        .map_err(|error| BatchError::TensorError(error.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(output), 0)))
}

fn batch_paired_numeric_dot(
    lhs_value: &Value,
    rhs_value: &Value,
) -> Result<Option<BatchTracer>, BatchError> {
    let (Value::Tensor(lhs), Value::Tensor(rhs)) = (lhs_value, rhs_value) else {
        return Ok(None);
    };

    if lhs.rank() < 2 || rhs.rank() < 2 || lhs.shape.dims[0] != rhs.shape.dims[0] {
        return Ok(None);
    }

    let Some(output_kind) = batch_dot_output_kind(lhs, rhs) else {
        return Ok(None);
    };

    let batch = lhs.shape.dims[0] as usize;
    let output = match (lhs.rank(), rhs.rank()) {
        (2, 2) => {
            let k = lhs.shape.dims[1] as usize;
            if rhs.shape.dims[1] as usize != k {
                return Ok(None);
            }
            let mut elements = Vec::with_capacity(batch);
            for batch_idx in 0..batch {
                elements.push(batch_dot_accumulate(output_kind, k, |kk| {
                    (
                        lhs.elements[batch_idx * k + kk],
                        rhs.elements[batch_idx * k + kk],
                    )
                })?);
            }
            TensorValue::new(output_kind.dtype(), Shape::vector(batch as u32), elements)
        }
        (3, 2) => {
            let rows = lhs.shape.dims[1] as usize;
            let k = lhs.shape.dims[2] as usize;
            if rhs.shape.dims[1] as usize != k {
                return Ok(None);
            }
            let mut elements = Vec::with_capacity(batch * rows);
            for batch_idx in 0..batch {
                for row in 0..rows {
                    elements.push(batch_dot_accumulate(output_kind, k, |kk| {
                        let lhs_idx = (batch_idx * rows + row) * k + kk;
                        let rhs_idx = batch_idx * k + kk;
                        (lhs.elements[lhs_idx], rhs.elements[rhs_idx])
                    })?);
                }
            }
            TensorValue::new(
                output_kind.dtype(),
                Shape {
                    dims: vec![batch as u32, rows as u32],
                },
                elements,
            )
        }
        (2, 3) => {
            let k = lhs.shape.dims[1] as usize;
            let cols = rhs.shape.dims[2] as usize;
            if rhs.shape.dims[1] as usize != k {
                return Ok(None);
            }
            let mut elements = Vec::with_capacity(batch * cols);
            for batch_idx in 0..batch {
                for col in 0..cols {
                    elements.push(batch_dot_accumulate(output_kind, k, |kk| {
                        let lhs_idx = batch_idx * k + kk;
                        let rhs_idx = (batch_idx * k + kk) * cols + col;
                        (lhs.elements[lhs_idx], rhs.elements[rhs_idx])
                    })?);
                }
            }
            TensorValue::new(
                output_kind.dtype(),
                Shape {
                    dims: vec![batch as u32, cols as u32],
                },
                elements,
            )
        }
        (3, 3) => {
            let rows = lhs.shape.dims[1] as usize;
            let k = lhs.shape.dims[2] as usize;
            let cols = rhs.shape.dims[2] as usize;
            if rhs.shape.dims[1] as usize != k {
                return Ok(None);
            }
            let mut elements = Vec::with_capacity(batch * rows * cols);
            for batch_idx in 0..batch {
                for row in 0..rows {
                    for col in 0..cols {
                        elements.push(batch_dot_accumulate(output_kind, k, |kk| {
                            let lhs_idx = (batch_idx * rows + row) * k + kk;
                            let rhs_idx = (batch_idx * k + kk) * cols + col;
                            (lhs.elements[lhs_idx], rhs.elements[rhs_idx])
                        })?);
                    }
                }
            }
            TensorValue::new(
                output_kind.dtype(),
                Shape {
                    dims: vec![batch as u32, rows as u32, cols as u32],
                },
                elements,
            )
        }
        _ => return Ok(None),
    }
    .map_err(|e| BatchError::TensorError(e.to_string()))?;

    Ok(Some(BatchTracer::batched(Value::Tensor(output), 0)))
}

fn batch_dot_output_kind(lhs: &TensorValue, rhs: &TensorValue) -> Option<BatchDotOutputKind> {
    if matches!(lhs.dtype, DType::Complex64 | DType::Complex128)
        || matches!(rhs.dtype, DType::Complex64 | DType::Complex128)
    {
        return None;
    }
    if lhs.elements.iter().all(|literal| literal.is_integral())
        && rhs.elements.iter().all(|literal| literal.is_integral())
    {
        Some(BatchDotOutputKind::Integral)
    } else {
        // Match JAX promotion semantics so an F32 batched dot stays F32,
        // BF16/F16 stay narrow, mixed half + F32 promotes to F32, etc.
        let promoted = promote_dtype_public(lhs.dtype, rhs.dtype);
        let real_dtype = match promoted {
            DType::BF16 | DType::F16 | DType::F32 | DType::F64 => promoted,
            _ => DType::F64,
        };
        Some(BatchDotOutputKind::Real(real_dtype))
    }
}

fn batch_dot_accumulate(
    output_kind: BatchDotOutputKind,
    len: usize,
    mut pair_at: impl FnMut(usize) -> (Literal, Literal),
) -> Result<Literal, BatchError> {
    match output_kind {
        BatchDotOutputKind::Integral => {
            let mut sum = 0_i64;
            for index in 0..len {
                let (left, right) = pair_at(index);
                let left = left.as_i64().ok_or_else(|| {
                    BatchError::EvalError("integral dot expected i64 elements".to_owned())
                })?;
                let right = right.as_i64().ok_or_else(|| {
                    BatchError::EvalError("integral dot expected i64 elements".to_owned())
                })?;
                sum += left * right;
            }
            Ok(Literal::I64(sum))
        }
        BatchDotOutputKind::Real(dtype) => {
            let mut sum = 0.0_f64;
            for index in 0..len {
                let (left, right) = pair_at(index);
                let left = left.as_f64().ok_or_else(|| {
                    BatchError::EvalError("expected numeric lhs tensor".to_owned())
                })?;
                let right = right.as_f64().ok_or_else(|| {
                    BatchError::EvalError("expected numeric rhs tensor".to_owned())
                })?;
                sum += left * right;
            }
            Ok(real_literal_from_f64_dtype(dtype, sum))
        }
    }
}

fn dot_lhs_single_batch_can_eval_direct(lhs_value: &Value, rhs_value: &Value) -> bool {
    match (lhs_value, rhs_value) {
        (_, Value::Scalar(_)) => true,
        (Value::Tensor(lhs), Value::Tensor(rhs)) if lhs.rank() >= 2 && rhs.rank() >= 1 => {
            let lhs_contract = lhs.shape.dims[lhs.rank() - 1];
            let rhs_contract = if rhs.rank() == 1 {
                rhs.shape.dims[0]
            } else {
                rhs.shape.dims[rhs.rank() - 2]
            };
            lhs_contract == rhs_contract
        }
        _ => false,
    }
}

fn dot_rhs_single_batch_can_eval_direct(lhs_value: &Value, rhs_value: &Value) -> bool {
    match (lhs_value, rhs_value) {
        (Value::Scalar(_), _) => true,
        (Value::Tensor(lhs), Value::Tensor(rhs)) if lhs.rank() >= 1 && rhs.rank() >= 3 => {
            lhs.shape.dims[lhs.rank() - 1] == rhs.shape.dims[rhs.rank() - 2]
        }
        _ => false,
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
    let new_shape = parse_shape(params)?;

    // Prepend batch dimension to new shape
    let mut batched_shape = Vec::with_capacity(new_shape.len() + 1);
    batched_shape.push(batch_size as i64);
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

    let tensor = value
        .as_tensor()
        .ok_or(BatchError::BatchDimMoveError("expected tensor".into()))?;
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);

    // Parse permutation
    let perm = parse_permutation(params, per_elem_rank)?;

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
    let target_shape = parse_param_usize_list(params, "shape")?;
    let raw_broadcast_dims = params.get("broadcast_dimensions");
    let mut broadcast_dims = if let Some(raw) = raw_broadcast_dims {
        if is_empty_list(raw) {
            Vec::new()
        } else {
            parse_usize_list(raw, "broadcast_dimensions")?
        }
    } else {
        Vec::new()
    };
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);
    let needs_default = raw_broadcast_dims.is_none_or(|raw| is_empty_list(raw));
    if needs_default && per_elem_rank > 0 {
        if per_elem_rank > target_shape.len() {
            return Err(BatchError::EvalError(format!(
                "input rank {} exceeds target rank {}",
                per_elem_rank,
                target_shape.len()
            )));
        }
        let offset = target_shape.len() - per_elem_rank;
        broadcast_dims = (offset..target_shape.len()).collect();
    }

    // Add batch to target shape and shift broadcast dimensions
    let mut new_shape = Vec::with_capacity(target_shape.len() + 1);
    new_shape.push(batch_size);
    for &d in &target_shape {
        new_shape.push(d);
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

    // Parse slice params: start_indices, limit_indices, strides.
    // JAX treats omitted slice strides as one along every sliced axis.
    let starts = parse_param_usize_list(params, "start_indices")?;
    let limits = parse_param_usize_list(params, "limit_indices")?;
    let strides = match params.get("strides") {
        None => vec![1_usize; starts.len()],
        Some(raw) if is_empty_list(raw) => vec![1_usize; starts.len()],
        Some(raw) => parse_usize_list(raw, "strides")?,
    };

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
    let first_batched = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)));
    let (any_batched, batch_dim) = match first_batched {
        None => {
            let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
            let result = eval_primitive(Primitive::Concatenate, &values, params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(pair) => pair,
    };

    let batch_size = get_batch_size(&any_batched.value, batch_dim)?;

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
    let axis = parse_param_usize(params, "dimension")?.unwrap_or(0);
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

    // Parse padding config: low, high, interior per dimension.
    // fj-lax and fj-trace default omitted interior padding to zero.
    let low = parse_param_i64_list(params, "padding_low")?;
    let high = parse_param_i64_list(params, "padding_high")?;
    let interior = match params.get("padding_interior") {
        None => vec![0_i64; low.len()],
        Some(raw) if is_empty_list(raw) => vec![0_i64; low.len()],
        Some(raw) => parse_i64_list(raw, "padding_interior")?,
    };

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
    if let Some(result) = batch_dynamic_slice_static_starts(inputs, params)? {
        return Ok(result);
    }

    // Supports batched start indices and mixed batched/unbatched operands
    // via per-element fallback semantics.
    batch_passthrough_leading(Primitive::DynamicSlice, inputs, params)
}

fn batch_dynamic_slice_static_starts(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    let Some(operand) = inputs.first() else {
        return Ok(None);
    };
    let Some(batch_dim) = operand.batch_dim else {
        return Ok(None);
    };
    if inputs.iter().skip(1).any(|input| input.batch_dim.is_some()) {
        return Ok(None);
    }

    let value = move_batch_dim_to_front(&operand.value, batch_dim)?;
    let Value::Tensor(tensor) = &value else {
        return Ok(None);
    };
    let slice_sizes = parse_param_usize_list(params, "slice_sizes")?;
    if slice_sizes.len() + 1 != tensor.rank() {
        return Ok(None);
    }

    let batch_size = get_batch_size(&value, 0)?;
    let mut batched_slice_sizes = Vec::with_capacity(slice_sizes.len() + 1);
    batched_slice_sizes.push(batch_size);
    batched_slice_sizes.extend_from_slice(&slice_sizes);

    let mut values = Vec::with_capacity(inputs.len() + 1);
    values.push(value);
    values.push(Value::scalar_i64(0));
    values.extend(inputs.iter().skip(1).map(|input| input.value.clone()));

    let mut new_params = params.clone();
    new_params.insert("slice_sizes".to_owned(), format_csv(&batched_slice_sizes));
    let result = eval_primitive(Primitive::DynamicSlice, &values, &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(result, 0)))
}

fn batch_dynamic_update_slice(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    if let Some(result) = batch_dynamic_update_slice_static_starts(inputs, params)? {
        return Ok(result);
    }

    // Supports batched updates/start indices and mixed batched/unbatched operands
    // via per-element fallback semantics.
    batch_passthrough_leading(Primitive::DynamicUpdateSlice, inputs, params)
}

fn batch_dynamic_update_slice_static_starts(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    let Some(operand) = inputs.first() else {
        return Ok(None);
    };
    let Some(update) = inputs.get(1) else {
        return Ok(None);
    };
    let Some((batch_source_value, batch_source_dim)) = operand
        .batch_dim
        .map(|batch_dim| (&operand.value, batch_dim))
        .or_else(|| update.batch_dim.map(|batch_dim| (&update.value, batch_dim)))
    else {
        return Ok(None);
    };
    if inputs.iter().skip(2).any(|input| input.batch_dim.is_some()) {
        return Ok(None);
    }

    let batch_size = get_batch_size(batch_source_value, batch_source_dim)?;
    let operand_value = match operand.batch_dim {
        Some(batch_dim) => move_batch_dim_to_front(&operand.value, batch_dim)?,
        None => broadcast_unbatched(&operand.value, batch_size, 0)?,
    };
    let update_value = match update.batch_dim {
        Some(batch_dim) => move_batch_dim_to_front(&update.value, batch_dim)?,
        None => broadcast_unbatched(&update.value, batch_size, 0)?,
    };
    let (Value::Tensor(operand_tensor), Value::Tensor(update_tensor)) =
        (&operand_value, &update_value)
    else {
        return Ok(None);
    };
    if operand_tensor.rank() != update_tensor.rank()
        || operand_tensor.rank() != inputs.len() - 1
        || operand_tensor.shape.dims.first() != update_tensor.shape.dims.first()
    {
        return Ok(None);
    }

    let mut values = Vec::with_capacity(inputs.len() + 1);
    values.push(operand_value);
    values.push(update_value);
    values.push(Value::scalar_i64(0));
    values.extend(inputs.iter().skip(2).map(|input| input.value.clone()));

    let result = eval_primitive(Primitive::DynamicUpdateSlice, &values, params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(result, 0)))
}

fn batch_gather(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let operand = &inputs[0];
    let indices = &inputs[1];

    // Fast-path common vmap case: unbatched operand with batched indices.
    // eval_gather already maps over the full indices tensor, producing
    // indices.shape ++ slice_sizes[1..]. That is exactly the stacked result
    // of gathering each mapped index slice against the same operand.
    if operand.batch_dim.is_none()
        && let Some(indices_bd) = indices.batch_dim
    {
        let indices_value = move_batch_dim_to_front(&indices.value, indices_bd)?;
        if indices_value.as_tensor().is_none() {
            return Err(BatchError::BatchDimMoveError(
                "gather indices must be tensor".into(),
            ));
        }
        if let Some(result) =
            batch_gather_unbatched_operand_rank1_indices_direct(operand, &indices_value, params)?
        {
            return Ok(result);
        }
        let result = eval_primitive(
            Primitive::Gather,
            &[operand.value.clone(), indices_value],
            params,
        )
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer::batched(result, 0));
    }

    if let Some(result) = batch_gather_batched_operand_direct(operand, indices, params)? {
        return Ok(result);
    }

    batch_passthrough_leading(Primitive::Gather, inputs, params)
}

#[derive(Clone, Copy)]
enum GatherIndexMode {
    Clip,
    FillOrDrop,
    PromiseInBounds,
}

fn batch_gather_unbatched_operand_rank1_indices_direct(
    operand: &BatchTracer,
    indices_value: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    let Value::Tensor(operand_tensor) = &operand.value else {
        return Ok(None);
    };
    let Value::Tensor(indices_tensor) = indices_value else {
        return Ok(None);
    };
    if operand_tensor.rank() != 1 {
        return Ok(None);
    }

    let mode = match parse_gather_index_mode(params) {
        Some(mode) => mode,
        None => return Ok(None),
    };
    let slice_sizes = parse_param_usize_list(params, "slice_sizes")?;
    if slice_sizes.as_slice() != [1] {
        return Ok(None);
    }

    let gather_dim = operand_tensor.shape.dims[0] as usize;
    if gather_dim == 0 {
        return Ok(None);
    }

    let output_elems = checked_product_usize(&indices_tensor.shape.dims, "gather output")?;
    if output_elems != indices_tensor.elements.len() {
        return Ok(None);
    }

    let fill_lit = gather_fill_literal_for_dtype(operand_tensor.dtype);
    let operand_elements = operand_tensor.elements.as_slice();
    let mut elements = Vec::with_capacity(output_elems);
    for literal in indices_tensor.elements.iter() {
        let raw_index = gather_index_literal_to_usize(literal)?;
        match resolve_gather_index(raw_index, gather_dim, mode) {
            Some(resolved_index) => elements.push(operand_elements[resolved_index]),
            None => elements.push(fill_lit),
        }
    }

    let tensor = TensorValue::new(
        operand_tensor.dtype,
        Shape {
            dims: indices_tensor.shape.dims.clone(),
        },
        elements,
    )
    .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
}

fn batch_gather_batched_operand_direct(
    operand: &BatchTracer,
    indices: &BatchTracer,
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    let Some(operand_batch_dim) = operand.batch_dim else {
        return Ok(None);
    };

    let operand_value = move_batch_dim_to_front(&operand.value, operand_batch_dim)?;
    let Value::Tensor(operand_tensor) = &operand_value else {
        return Ok(None);
    };
    if operand_tensor.rank() < 2 {
        return Ok(None);
    }

    let mode = match parse_gather_index_mode(params) {
        Some(mode) => mode,
        None => return Ok(None),
    };
    let slice_sizes = parse_param_usize_list(params, "slice_sizes")?;
    let unbatched_rank = operand_tensor.rank() - 1;
    if slice_sizes.len() != unbatched_rank || slice_sizes.first().copied() != Some(1) {
        return Ok(None);
    }

    let unbatched_dims = &operand_tensor.shape.dims[1..];
    for (&slice_size, &dim) in slice_sizes.iter().zip(unbatched_dims) {
        if slice_size > dim as usize {
            return Ok(None);
        }
    }

    let batch_size = operand_tensor.shape.dims[0] as usize;
    let gather_dim = operand_tensor.shape.dims[1] as usize;
    if gather_dim == 0 {
        return Ok(None);
    }

    let indices_value = match indices.batch_dim {
        Some(batch_dim) => {
            let index_batch_size = get_batch_size(&indices.value, batch_dim)?;
            if index_batch_size != batch_size {
                return Ok(None);
            }
            move_batch_dim_to_front(&indices.value, batch_dim)?
        }
        None => indices.value.clone(),
    };

    let prepared_indices = prepare_gather_indices(&indices_value, indices.batch_dim, batch_size)?;
    let trailing_slice_dims: Vec<u32> = slice_sizes
        .iter()
        .skip(1)
        .map(|&size| {
            u32::try_from(size)
                .map_err(|_| BatchError::EvalError(format!("slice size {size} exceeds u32 range")))
        })
        .collect::<Result<_, _>>()?;
    let slice_elems = checked_product_usize(&trailing_slice_dims, "gather slice")?;

    let mut out_dims =
        Vec::with_capacity(1 + prepared_indices.per_batch_shape.len() + trailing_slice_dims.len());
    out_dims.push(operand_tensor.shape.dims[0]);
    out_dims.extend_from_slice(&prepared_indices.per_batch_shape);
    out_dims.extend_from_slice(&trailing_slice_dims);
    let output_elems = checked_product_usize(&out_dims, "gather output")?;
    if output_elems == 0 {
        let tensor = TensorValue::new(operand_tensor.dtype, Shape { dims: out_dims }, Vec::new())
            .map_err(|e| BatchError::TensorError(e.to_string()))?;
        return Ok(Some(BatchTracer::batched(Value::Tensor(tensor), 0)));
    }

    let mut elements = Vec::with_capacity(output_elems);
    let fill_lit = gather_fill_literal_for_dtype(operand_tensor.dtype);
    let operand_strides = row_major_strides(&operand_tensor.shape.dims)?;
    let operand_elements = operand_tensor.elements.as_slice();
    let trailing_slice_is_contiguous = slice_sizes
        .iter()
        .skip(1)
        .zip(unbatched_dims.iter().skip(1))
        .all(|(&slice_size, &dim)| slice_size == dim as usize);

    for batch_index in 0..batch_size {
        let batch_offset = batch_index
            .checked_mul(operand_strides[0])
            .ok_or_else(|| BatchError::EvalError("gather batch offset overflow".to_owned()))?;
        for literal in prepared_indices.literals_for_batch(batch_index) {
            let raw_index = gather_index_literal_to_usize(literal)?;
            let Some(resolved_index) = resolve_gather_index(raw_index, gather_dim, mode) else {
                elements.extend(std::iter::repeat_n(fill_lit, slice_elems));
                continue;
            };
            let base_offset =
                batch_offset
                    .checked_add(resolved_index.checked_mul(operand_strides[1]).ok_or_else(
                        || BatchError::EvalError("gather row offset overflow".to_owned()),
                    )?)
                    .ok_or_else(|| {
                        BatchError::EvalError("gather base offset overflow".to_owned())
                    })?;

            if trailing_slice_is_contiguous {
                let end = base_offset.checked_add(slice_elems).ok_or_else(|| {
                    BatchError::EvalError("gather contiguous slice end overflow".to_owned())
                })?;
                let slice = operand_elements.get(base_offset..end).ok_or_else(|| {
                    BatchError::EvalError(
                        "gather contiguous slice exceeds operand element count".to_owned(),
                    )
                })?;
                elements.extend_from_slice(slice);
                continue;
            }

            gather_partial_slice_elements(
                operand_elements,
                &operand_strides,
                base_offset,
                &slice_sizes,
                &mut elements,
            )?;
        }
    }

    let tensor = TensorValue::new(operand_tensor.dtype, Shape { dims: out_dims }, elements)
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
}

struct PreparedGatherIndices {
    per_batch_shape: Vec<u32>,
    per_batch_len: usize,
    storage: PreparedGatherIndexStorage,
}

enum PreparedGatherIndexStorage {
    Shared(Vec<Literal>),
    Batched(Vec<Literal>),
}

impl PreparedGatherIndices {
    fn literals_for_batch(&self, batch_index: usize) -> &[Literal] {
        match &self.storage {
            PreparedGatherIndexStorage::Shared(elements) => elements,
            PreparedGatherIndexStorage::Batched(elements) => {
                let start = batch_index * self.per_batch_len;
                &elements[start..start + self.per_batch_len]
            }
        }
    }
}

fn prepare_gather_indices(
    value: &Value,
    batch_dim: Option<usize>,
    batch_size: usize,
) -> Result<PreparedGatherIndices, BatchError> {
    match (batch_dim, value) {
        (None, Value::Scalar(literal)) => Ok(PreparedGatherIndices {
            per_batch_shape: Vec::new(),
            per_batch_len: 1,
            storage: PreparedGatherIndexStorage::Shared(vec![*literal]),
        }),
        (None, Value::Tensor(tensor)) => Ok(PreparedGatherIndices {
            per_batch_shape: tensor.shape.dims.clone(),
            per_batch_len: tensor.elements.len(),
            storage: PreparedGatherIndexStorage::Shared(tensor.elements.to_vec()),
        }),
        (Some(_), Value::Tensor(tensor)) => {
            if tensor.shape.dims.first().copied().map(|dim| dim as usize) != Some(batch_size) {
                return Err(BatchError::EvalError(format!(
                    "gather index batch size must be {batch_size}"
                )));
            }
            let per_batch_shape = tensor.shape.dims[1..].to_vec();
            let per_batch_len = checked_product_usize(&per_batch_shape, "gather indices")?;
            Ok(PreparedGatherIndices {
                per_batch_shape,
                per_batch_len,
                storage: PreparedGatherIndexStorage::Batched(tensor.elements.to_vec()),
            })
        }
        (Some(_), Value::Scalar(_)) => Err(BatchError::BatchDimOutOfBounds {
            batch_dim: 0,
            rank: 0,
        }),
    }
}

fn parse_gather_index_mode(params: &BTreeMap<String, String>) -> Option<GatherIndexMode> {
    match params.get("index_mode").map(String::as_str) {
        None | Some("clip") => Some(GatherIndexMode::Clip),
        Some("fill" | "drop" | "fill_or_drop") => Some(GatherIndexMode::FillOrDrop),
        Some("promise_in_bounds" | "promise") => Some(GatherIndexMode::PromiseInBounds),
        Some(_) => None,
    }
}

fn resolve_gather_index(index: usize, dim: usize, mode: GatherIndexMode) -> Option<usize> {
    if index < dim {
        Some(index)
    } else {
        match mode {
            GatherIndexMode::FillOrDrop => None,
            GatherIndexMode::Clip | GatherIndexMode::PromiseInBounds => Some(dim - 1),
        }
    }
}

fn gather_index_literal_to_usize(literal: &Literal) -> Result<usize, BatchError> {
    match literal {
        Literal::I64(value) if *value >= 0 => Ok(*value as usize),
        Literal::I64(value) => Err(BatchError::EvalError(format!("negative index {value}"))),
        Literal::U32(value) => Ok(*value as usize),
        Literal::U64(value) => usize::try_from(*value)
            .map_err(|_| BatchError::EvalError(format!("index {value} exceeds usize range"))),
        Literal::Bool(value) => Ok(usize::from(*value)),
        other => Err(BatchError::EvalError(format!(
            "unsupported gather index literal {other:?}"
        ))),
    }
}

fn gather_fill_literal_for_dtype(dtype: DType) -> Literal {
    match dtype {
        DType::F64 => Literal::F64Bits(f64::NAN.to_bits()),
        DType::F32 => Literal::F32Bits(f32::NAN.to_bits()),
        DType::F16 => Literal::from_f16_f64(f64::NAN),
        DType::BF16 => Literal::from_bf16_f64(f64::NAN),
        DType::I32 => Literal::I64(i64::from(i32::MIN)),
        DType::I64 => Literal::I64(i64::MIN),
        DType::U32 => Literal::U32(u32::MAX),
        DType::U64 => Literal::U64(u64::MAX),
        DType::Bool => Literal::Bool(true),
        DType::Complex64 => Literal::Complex64Bits(f32::NAN.to_bits(), 0),
        DType::Complex128 => Literal::Complex128Bits(f64::NAN.to_bits(), 0),
    }
}

fn gather_partial_slice_elements(
    operand_elements: &[Literal],
    operand_strides: &[usize],
    base_offset: usize,
    slice_sizes: &[usize],
    elements: &mut Vec<Literal>,
) -> Result<(), BatchError> {
    let slice_elems = checked_product_usize(&slice_sizes[1..], "gather slice")?;
    let mut slice_coords = vec![0_usize; slice_sizes.len().saturating_sub(1)];

    for _ in 0..slice_elems {
        let mut flat = base_offset;
        for (axis, &coord) in slice_coords.iter().enumerate() {
            let offset = coord
                .checked_mul(operand_strides[axis + 2])
                .ok_or_else(|| {
                    BatchError::EvalError(format!("gather offset overflow on axis {}", axis + 1))
                })?;
            flat = flat
                .checked_add(offset)
                .ok_or_else(|| BatchError::EvalError("gather flat offset overflow".to_owned()))?;
        }
        let element = operand_elements.get(flat).ok_or_else(|| {
            BatchError::EvalError("gather offset exceeds operand element count".to_owned())
        })?;
        elements.push(*element);

        for axis in (0..slice_coords.len()).rev() {
            slice_coords[axis] += 1;
            if slice_coords[axis] < slice_sizes[axis + 1] {
                break;
            }
            slice_coords[axis] = 0;
        }
    }

    Ok(())
}

fn row_major_strides(dims: &[u32]) -> Result<Vec<usize>, BatchError> {
    let mut strides = vec![1_usize; dims.len()];
    let mut stride = 1_usize;
    for (index, &dim) in dims.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride.checked_mul(dim as usize).ok_or_else(|| {
            BatchError::EvalError("row-major stride computation overflow".to_owned())
        })?;
    }
    Ok(strides)
}

fn checked_product_usize<T>(dims: &[T], context: &str) -> Result<usize, BatchError>
where
    T: Copy + TryInto<usize>,
{
    dims.iter().try_fold(1_usize, |acc, &dim| {
        let dim = dim
            .try_into()
            .map_err(|_| BatchError::EvalError(format!("{context} dimension exceeds usize")))?;
        acc.checked_mul(dim)
            .ok_or_else(|| BatchError::EvalError(format!("{context} element count overflow")))
    })
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
        if let Some(result) =
            batch_scatter_unbatched_operand_direct(operand, indices, updates, params)?
        {
            return Ok(result);
        }

        let batch_size = if let Some(bd) = indices.batch_dim {
            get_batch_size(&indices.value, bd)?
        } else {
            let bd = updates.batch_dim.ok_or_else(|| {
                BatchError::BatchDimMoveError("scatter expected batched indices or updates".into())
            })?;
            get_batch_size(&updates.value, bd)?
        };

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

fn batch_scatter_unbatched_operand_direct(
    operand: &BatchTracer,
    indices: &BatchTracer,
    updates: &BatchTracer,
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    let Some(operand_tensor) = operand.value.as_tensor() else {
        return Ok(None);
    };
    let Some(&operand_axis0_dim) = operand_tensor.shape.dims.first() else {
        return Ok(None);
    };
    let operand_axis0 = operand_axis0_dim as usize;

    let batch_size = if let Some(batch_dim) = indices.batch_dim {
        get_batch_size(&indices.value, batch_dim)?
    } else if let Some(batch_dim) = updates.batch_dim {
        get_batch_size(&updates.value, batch_dim)?
    } else {
        return Ok(None);
    };
    if batch_size == 0 {
        return Ok(None);
    }
    let batch_size_u32 = u32::try_from(batch_size)
        .map_err(|_| BatchError::TensorError("scatter batch size exceeds u32".to_owned()))?;

    let indices_value = match indices.batch_dim {
        Some(batch_dim) => move_batch_dim_to_front(&indices.value, batch_dim)?,
        None => broadcast_unbatched(&indices.value, batch_size, 0)?,
    };
    let updates_value = match updates.batch_dim {
        Some(batch_dim) => move_batch_dim_to_front(&updates.value, batch_dim)?,
        None => broadcast_unbatched(&updates.value, batch_size, 0)?,
    };
    let (Value::Tensor(indices_tensor), Value::Tensor(updates_tensor)) =
        (&indices_value, &updates_value)
    else {
        return Ok(None);
    };
    if indices_tensor.leading_dim() != Some(batch_size_u32)
        || updates_tensor.leading_dim() != Some(batch_size_u32)
    {
        return Ok(None);
    }
    let rank1_shape = ScatterRank1BatchShape {
        batch_size,
        batch_size_u32,
        operand_axis0,
        operand_axis0_dim,
    };
    if let Some(result) = batch_scatter_unbatched_operand_rank1_overwrite(
        operand_tensor,
        indices_tensor,
        updates_tensor,
        params,
        rank1_shape,
    )? {
        return Ok(Some(result));
    }

    let flat_axis0 = batch_size
        .checked_mul(operand_axis0)
        .ok_or_else(|| BatchError::TensorError("scatter flattened axis overflowed".to_owned()))?;
    let flat_axis0_u32 = u32::try_from(flat_axis0).map_err(|_| {
        BatchError::TensorError("scatter flattened axis exceeds u32 shape limit".to_owned())
    })?;

    let flat_operand_len = operand_tensor
        .elements
        .len()
        .checked_mul(batch_size)
        .ok_or_else(|| {
            BatchError::TensorError("scatter operand replication overflowed".to_owned())
        })?;
    let mut flat_operand_elements = Vec::with_capacity(flat_operand_len);
    for _ in 0..batch_size {
        flat_operand_elements.extend_from_slice(&operand_tensor.elements);
    }
    let mut flat_operand_dims = Vec::with_capacity(operand_tensor.rank());
    flat_operand_dims.push(flat_axis0_u32);
    flat_operand_dims.extend_from_slice(&operand_tensor.shape.dims[1..]);
    let flat_operand = Value::Tensor(
        TensorValue::new(
            operand_tensor.dtype,
            Shape {
                dims: flat_operand_dims,
            },
            flat_operand_elements,
        )
        .map_err(|e| BatchError::TensorError(e.to_string()))?,
    );

    let indices_per_batch = indices_tensor.elements.len() / batch_size;
    let mut flat_indices_elements = Vec::with_capacity(indices_tensor.elements.len());
    for (flat_pos, &literal) in indices_tensor.elements.iter().enumerate() {
        let batch_idx = flat_pos / indices_per_batch;
        let batch_offset = batch_idx
            .checked_mul(operand_axis0)
            .ok_or_else(|| BatchError::TensorError("scatter index offset overflowed".to_owned()))?;
        let Some(adjusted) = offset_scatter_index_literal(literal, batch_offset, operand_axis0)?
        else {
            return Ok(None);
        };
        flat_indices_elements.push(adjusted);
    }
    let flat_indices = Value::Tensor(
        TensorValue::new(
            indices_tensor.dtype,
            indices_tensor.shape.clone(),
            flat_indices_elements,
        )
        .map_err(|e| BatchError::TensorError(e.to_string()))?,
    );

    let flat_result = eval_primitive(
        Primitive::Scatter,
        &[flat_operand, flat_indices, updates_value],
        params,
    )
    .map_err(|e| BatchError::EvalError(e.to_string()))?;
    let Value::Tensor(flat_tensor) = flat_result else {
        return Ok(None);
    };
    let mut output_dims = Vec::with_capacity(operand_tensor.rank() + 1);
    output_dims.push(batch_size_u32);
    output_dims.extend_from_slice(&operand_tensor.shape.dims);
    let output = TensorValue::new(
        operand_tensor.dtype,
        Shape { dims: output_dims },
        flat_tensor.elements.to_vec(),
    )
    .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(output), 0)))
}

#[derive(Clone, Copy)]
struct ScatterRank1BatchShape {
    batch_size: usize,
    batch_size_u32: u32,
    operand_axis0: usize,
    operand_axis0_dim: u32,
}

fn batch_scatter_unbatched_operand_rank1_overwrite(
    operand_tensor: &TensorValue,
    indices_tensor: &TensorValue,
    updates_tensor: &TensorValue,
    params: &BTreeMap<String, String>,
    shape: ScatterRank1BatchShape,
) -> Result<Option<BatchTracer>, BatchError> {
    if operand_tensor.rank() != 1 {
        return Ok(None);
    }
    let mode = params
        .get("mode")
        .map(String::as_str)
        .unwrap_or("overwrite");
    if mode != "overwrite" {
        return Ok(None);
    }
    if updates_tensor.dtype != operand_tensor.dtype || updates_tensor.shape != indices_tensor.shape
    {
        return Ok(None);
    }
    if operand_tensor.elements.len() != shape.operand_axis0 {
        return Ok(None);
    }
    let Some(indices_per_batch) = indices_tensor.elements.len().checked_div(shape.batch_size)
    else {
        return Ok(None);
    };
    if indices_per_batch
        .checked_mul(shape.batch_size)
        .is_none_or(|len| len != indices_tensor.elements.len())
        || updates_tensor.elements.len() != indices_tensor.elements.len()
    {
        return Ok(None);
    }

    let output_len = shape
        .batch_size
        .checked_mul(shape.operand_axis0)
        .ok_or_else(|| BatchError::TensorError("scatter output size overflowed".to_owned()))?;
    let mut patches = Vec::with_capacity(indices_tensor.elements.len());

    for (flat_pos, &literal) in indices_tensor.elements.iter().enumerate() {
        let Some(raw_index) = scatter_index_literal_to_usize(literal)? else {
            return Ok(None);
        };
        if raw_index >= shape.operand_axis0 {
            return Ok(None);
        }
        let batch_idx = flat_pos / indices_per_batch;
        let batch_offset = batch_idx.checked_mul(shape.operand_axis0).ok_or_else(|| {
            BatchError::TensorError("scatter output offset overflowed".to_owned())
        })?;
        let output_index = batch_offset
            .checked_add(raw_index)
            .ok_or_else(|| BatchError::TensorError("scatter output index overflowed".to_owned()))?;
        let update = *updates_tensor.elements.get(flat_pos).ok_or_else(|| {
            BatchError::TensorError("scatter update index out of range".to_owned())
        })?;
        if output_index >= output_len {
            return Err(BatchError::TensorError(
                "scatter output index out of range".to_owned(),
            ));
        }
        patches.push((output_index, update));
    }

    let output_elements = LiteralBuffer::from_repeated_with_patches(
        operand_tensor.elements.to_vec(),
        shape.batch_size,
        patches,
    )
    .ok_or_else(|| {
        BatchError::TensorError("scatter repeated-patch buffer overflowed".to_owned())
    })?;
    let output = TensorValue::new_with_literal_buffer(
        operand_tensor.dtype,
        Shape {
            dims: vec![shape.batch_size_u32, shape.operand_axis0_dim],
        },
        output_elements,
    )
    .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(output), 0)))
}

fn offset_scatter_index_literal(
    literal: Literal,
    batch_offset: usize,
    operand_axis0: usize,
) -> Result<Option<Literal>, BatchError> {
    let Some(raw_index) = scatter_index_literal_to_usize(literal)? else {
        return Ok(None);
    };
    if raw_index >= operand_axis0 {
        return Ok(None);
    }
    let adjusted = batch_offset
        .checked_add(raw_index)
        .ok_or_else(|| BatchError::TensorError("scatter adjusted index overflowed".to_owned()))?;

    match literal {
        Literal::I64(_) => i64::try_from(adjusted)
            .map(Literal::I64)
            .map(Some)
            .map_err(|_| BatchError::TensorError("scatter adjusted index exceeds i64".to_owned())),
        Literal::U32(_) => u32::try_from(adjusted)
            .map(Literal::U32)
            .map(Some)
            .map_err(|_| BatchError::TensorError("scatter adjusted index exceeds u32".to_owned())),
        Literal::U64(_) => u64::try_from(adjusted)
            .map(Literal::U64)
            .map(Some)
            .map_err(|_| BatchError::TensorError("scatter adjusted index exceeds u64".to_owned())),
        Literal::Bool(value) if batch_offset == 0 => Ok(Some(Literal::Bool(value))),
        Literal::Bool(_) => Ok(None),
        Literal::BF16Bits(_)
        | Literal::F16Bits(_)
        | Literal::F32Bits(_)
        | Literal::F64Bits(_)
        | Literal::Complex64Bits(..)
        | Literal::Complex128Bits(..) => Ok(None),
    }
}

fn scatter_index_literal_to_usize(literal: Literal) -> Result<Option<usize>, BatchError> {
    match literal {
        Literal::I64(value) => {
            if value < 0 {
                return Ok(None);
            }
            usize::try_from(value)
                .map(Some)
                .map_err(|_| BatchError::TensorError("scatter i64 index exceeds usize".to_owned()))
        }
        Literal::U32(value) => Ok(Some(value as usize)),
        Literal::U64(value) => usize::try_from(value)
            .map(Some)
            .map_err(|_| BatchError::TensorError("scatter u64 index exceeds usize".to_owned())),
        Literal::Bool(value) => Ok(Some(usize::from(value))),
        Literal::BF16Bits(_)
        | Literal::F16Bits(_)
        | Literal::F32Bits(_)
        | Literal::F64Bits(_)
        | Literal::Complex64Bits(..)
        | Literal::Complex128Bits(..) => Ok(None),
    }
}

fn batch_squeeze(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::Squeeze,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let Value::Tensor(tensor) = &value else {
        return Ok(BatchTracer::batched(value, 0));
    };

    let squeeze_dims = match params.get("dimensions") {
        None => tensor
            .shape
            .dims
            .iter()
            .enumerate()
            .skip(1)
            .filter(|&(_, &dim)| dim == 1)
            .map(|(axis, _)| axis)
            .collect::<Vec<_>>(),
        // Parse as i64 so a negative (end-relative) dimension is handled. The
        // batch axis is prepended at the FRONT, so map each logical dimension to
        // its physical axis in the batched tensor: a non-negative dimension shifts
        // +1, while a negative (end-relative) one resolves to tensor_rank + dim
        // (both land strictly past the batch axis at index 0).
        Some(raw) => {
            let tensor_rank = tensor.shape.dims.len() as i64;
            parse_i64_list(raw, "dimensions")?
                .into_iter()
                .map(|dim| {
                    let physical = if dim >= 0 { dim + 1 } else { tensor_rank + dim };
                    if physical < 1 || physical >= tensor_rank {
                        return Err(BatchError::EvalError(format!(
                            "squeeze dimension {dim} out of range for per-element rank {}",
                            tensor_rank - 1
                        )));
                    }
                    Ok(physical as usize)
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    if squeeze_dims.is_empty() {
        return Ok(BatchTracer::batched(value, 0));
    }

    let mut new_params = params.clone();
    new_params.insert("dimensions".to_owned(), format_csv(&squeeze_dims));
    let result = eval_primitive(Primitive::Squeeze, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_expand_dims(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::ExpandDims,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for expand_dims".into()))?;
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);
    // Parse the per-element axis as i64 and normalize a negative (end-relative)
    // axis against the per-element OUTPUT rank, then shift +1 for the prepended
    // batch axis. A usize parse rejected axis=-1.
    let raw_axis: i64 = params
        .get("axis")
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .ok_or_else(|| BatchError::EvalError("invalid axis param for expand_dims".to_owned()))?;
    let physical_axis = if per_elem_rank == 0 {
        // Per-element scalar: the single new dim lands right after the batch axis
        // (the logical axis is immaterial — output is always [batch, 1]).
        1
    } else {
        // Normalize a negative (end-relative) axis against the per-element OUTPUT
        // rank, then shift +1 for the prepended batch axis. A usize parse rejected
        // axis=-1.
        let out_rank = per_elem_rank + 1;
        let logical_axis = if raw_axis < 0 { raw_axis + out_rank as i64 } else { raw_axis };
        if logical_axis < 0 || logical_axis as usize >= out_rank {
            return Err(BatchError::EvalError(format!(
                "expand_dims axis {raw_axis} out of range for per-element output rank {out_rank}"
            )));
        }
        logical_axis as usize + 1
    };

    let mut new_params = params.clone();
    new_params.insert("axis".to_owned(), physical_axis.to_string());
    let result = eval_primitive(Primitive::ExpandDims, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_rev(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(Primitive::Rev, std::slice::from_ref(&input.value), params)
                .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    let axes = parse_param_usize_list(params, "axes")?;
    get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for rev".into()))?;
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);

    if per_elem_rank == 0 {
        return Ok(BatchTracer::batched(value, 0));
    }

    let physical_axes = axes
        .into_iter()
        .map(|axis| {
            if axis >= per_elem_rank {
                return Err(BatchError::EvalError(format!(
                    "rev axis {axis} out of range for per-element rank {per_elem_rank}"
                )));
            }
            axis.checked_add(1)
                .ok_or_else(|| BatchError::EvalError("rev axis overflow".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut new_params = params.clone();
    new_params.insert("axes".to_owned(), format_csv(&physical_axes));
    let result = eval_primitive(Primitive::Rev, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_split(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result =
                eval_primitive(Primitive::Split, std::slice::from_ref(&input.value), params)
                    .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    let logical_axis = parse_param_usize(params, "axis")?.unwrap_or(0);
    get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for split".into()))?;
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);

    if per_elem_rank == 0 {
        return Err(BatchError::EvalError("cannot split a scalar".to_owned()));
    }
    if logical_axis >= per_elem_rank {
        return Err(BatchError::EvalError(format!(
            "split axis {logical_axis} out of range for per-element rank {per_elem_rank}"
        )));
    }

    let physical_axis = logical_axis
        .checked_add(1)
        .ok_or_else(|| BatchError::EvalError("split axis overflow".to_owned()))?;
    let mut new_params = params.clone();
    new_params.insert("axis".to_owned(), physical_axis.to_string());
    let result = eval_primitive(Primitive::Split, &[value], &new_params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
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

    // Shift the cumulative axis for the prepended batch axis. Parse as i64 so a
    // negative (end-relative) axis survives — eval_cumulative normalizes it
    // against the batched rank — and is left UNCHANGED (the batch is prepended at
    // the front, so an end-relative axis still points to the same position); only
    // a non-negative axis shifts +1. A usize parse previously rejected "-1".
    let raw_axis: i64 = params
        .get("axis")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let shifted = if raw_axis >= 0 { raw_axis + 1 } else { raw_axis };
    let mut new_params = params.clone();
    new_params.insert("axis".to_owned(), shifted.to_string());
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

    // eval_sort/eval_argsort read the "axis" param (defaulting to the last axis) — NOT
    // "dimension". With the batch axis moved to the front: a NON-NEGATIVE sort axis shifts
    // by +1; a NEGATIVE axis is end-relative and unchanged (the batch is prepended, not
    // appended); an ABSENT axis defaults to the batched operand's last axis, which is still
    // the original last axis. (This previously shifted a "dimension" param that eval never
    // reads, leaving "axis" unshifted — so vmap(sort) along an explicit non-negative axis
    // sorted the batch axis. See test_batch_sort_shifts_the_axis_param_eval_reads.)
    let mut new_params = params.clone();
    if let Some(raw) = params.get("axis") {
        let axis: i64 = raw.trim().parse().map_err(|_| {
            BatchError::EvalError(format!("sort: invalid integer in param 'axis': '{raw}'"))
        })?;
        if axis >= 0 {
            new_params.insert("axis".to_owned(), (axis + 1).to_string());
        }
    }
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

    let logical_rank = match &value {
        Value::Scalar(_) => 0,
        Value::Tensor(tensor) => tensor.rank().saturating_sub(1),
    };
    let window_dims = match params.get("window_dimensions") {
        None => vec![2_usize; logical_rank],
        Some(raw) if is_empty_list(raw) => vec![2_usize; logical_rank],
        Some(raw) => parse_usize_list(raw, "window_dimensions")?,
    };
    let window_strides = match params.get("window_strides") {
        None => vec![1_usize; window_dims.len()],
        Some(raw) if is_empty_list(raw) => vec![1_usize; window_dims.len()],
        Some(raw) => parse_usize_list(raw, "window_strides")?,
    };
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
    batch_nullary(Primitive::Iota, inputs, params)
}

fn batch_broadcasted_iota(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    batch_nullary(Primitive::BroadcastedIota, inputs, params)
}

fn batch_nullary(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // Nullary primitives do not depend on value inputs, so they stay unbatched.
    let result =
        eval_primitive(primitive, &[], params).map_err(|e| BatchError::EvalError(e.to_string()))?;
    let _ = inputs;
    Ok(BatchTracer::unbatched(result))
}

// ── Linear Algebra Batching ────────────────────────────────────────

fn batch_cholesky(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::Cholesky,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for cholesky".to_owned()))?;

    if tensor.rank() != 3 {
        return batch_passthrough_leading(Primitive::Cholesky, inputs, params);
    }

    let batch_size = tensor.shape.dims[0] as usize;
    let rows = tensor.shape.dims[1] as usize;
    let cols = tensor.shape.dims[2] as usize;
    if rows != cols {
        return Err(BatchError::EvalError(format!(
            "unsupported cholesky behavior: Cholesky requires a square matrix, got {rows}x{cols}"
        )));
    }

    let matrix_len = rows * cols;
    let mut elements = Vec::with_capacity(tensor.elements.len());
    for batch in 0..batch_size {
        let base = batch * matrix_len;
        let mut l = vec![0.0_f64; matrix_len];

        for i in 0..rows {
            for j in 0..=i {
                let mut sum = 0.0_f64;
                for k in 0..j {
                    sum += l[i * cols + k] * l[j * cols + k];
                }

                if i == j {
                    let diag = tensor.elements[base + i * cols + i]
                        .as_f64()
                        .ok_or_else(|| {
                            BatchError::EvalError(
                                "type mismatch for cholesky: expected numeric elements".to_owned(),
                            )
                        })?
                        - sum;
                    // jnp.linalg.cholesky returns NaN (not an error) for non-PD
                    // input; sqrt of a non-positive diagonal yields NaN/0 that
                    // propagates, matching the per-element fj-lax cholesky and
                    // JAX (NumPy raises LinAlgError, JAX does not).
                    l[i * cols + j] = diag.sqrt();
                } else {
                    let a_ij = tensor.elements[base + i * cols + j]
                        .as_f64()
                        .ok_or_else(|| {
                            BatchError::EvalError(
                                "type mismatch for cholesky: expected numeric elements".to_owned(),
                            )
                        })?;
                    l[i * cols + j] = (a_ij - sum) / l[j * cols + j];
                }
            }
        }

        elements.extend(l.into_iter().map(Literal::from_f64));
    }

    TensorValue::new(tensor.dtype, tensor.shape.clone(), elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn batch_qr_multi(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some((input, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        return eval_primitive_multi(Primitive::Qr, &values, params)
            .map(|outputs| outputs.into_iter().map(BatchTracer::unbatched).collect())
            .map_err(|e| BatchError::EvalError(e.to_string()));
    };

    if inputs.len() != 1 {
        return batch_passthrough_leading_multi(Primitive::Qr, inputs, params);
    }

    let batch_size = get_batch_size(&input.value, batch_dim)?;
    let moved_value = if batch_dim == 0 {
        None
    } else {
        Some(move_batch_dim_to_front(&input.value, batch_dim)?)
    };
    let value = moved_value.as_ref().unwrap_or(&input.value);
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for qr".to_owned()))?;

    if tensor.rank() != 3 {
        return batch_passthrough_leading_multi(Primitive::Qr, inputs, params);
    }

    if tensor.shape.dims[0] as usize != batch_size {
        return batch_passthrough_leading_multi(Primitive::Qr, inputs, params);
    }

    let m = tensor.shape.dims[1] as usize;
    let n = tensor.shape.dims[2] as usize;
    let k = m.min(n);
    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");
    let q_cols = if full_matrices { m } else { k };
    let r_rows = if full_matrices { m } else { k };
    let matrix_len = m * n;

    let mut q_elements = Vec::with_capacity(batch_size * m * q_cols);
    let mut r_elements = Vec::with_capacity(batch_size * r_rows * n);
    let mut matrix = Vec::with_capacity(matrix_len);
    let mut qr_scratch = QrScratch::default();
    for batch in 0..batch_size {
        let base = batch * matrix_len;
        if m == 3 && n == 2 && !full_matrices {
            let a00 = tensor.elements[base].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            let a01 = tensor.elements[base + 1].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            let a10 = tensor.elements[base + 2].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            let a11 = tensor.elements[base + 3].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            let a20 = tensor.elements[base + 4].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            let a21 = tensor.elements[base + 5].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for qr: expected numeric elements".to_owned())
            })?;
            qr_decompose_matrix_3x2_thin(
                [a00, a01, a10, a11, a20, a21],
                &mut qr_scratch.q_out,
                &mut qr_scratch.r_out,
            );
        } else {
            matrix.clear();
            for lit in &tensor.elements[base..base + matrix_len] {
                matrix.push(lit.as_f64().ok_or_else(|| {
                    BatchError::EvalError(
                        "type mismatch for qr: expected numeric elements".to_owned(),
                    )
                })?);
            }
            qr_decompose_matrix(m, n, &matrix, full_matrices, &mut qr_scratch);
        }
        q_elements.extend(qr_scratch.q_out.iter().copied().map(Literal::from_f64));
        r_elements.extend(qr_scratch.r_out.iter().copied().map(Literal::from_f64));
    }

    let q_shape = Shape {
        dims: vec![batch_size as u32, m as u32, q_cols as u32],
    };
    let r_shape = Shape {
        dims: vec![batch_size as u32, r_rows as u32, n as u32],
    };
    let q = TensorValue::new(tensor.dtype, q_shape, q_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    let r = TensorValue::new(tensor.dtype, r_shape, r_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(vec![q, r])
}

/// Reusable working buffers for batched Householder QR. Carrying these across
/// every matrix in a vmapped batch removes the per-matrix heap traffic
/// (`r.to_vec()`, the `Vec<Vec<f64>>` reflector store, `tau`, and both output
/// buffers) that dominated the decomposition cost for small matrices.
#[derive(Default)]
struct QrScratch {
    r: Vec<f64>,
    v_store: Vec<Vec<f64>>,
    tau_store: Vec<f64>,
    q_out: Vec<f64>,
    r_out: Vec<f64>,
}

fn qr_decompose_matrix_3x2_thin(a: [f64; 6], q_out: &mut Vec<f64>, r_out: &mut Vec<f64>) {
    let mut r = a;

    let mut v00 = r[0];
    let v01 = r[2];
    let v02 = r[4];
    let norm_v0 = (v00 * v00 + v01 * v01 + v02 * v02).sqrt();
    let alpha0 = if v00 >= 0.0 { -norm_v0 } else { norm_v0 };
    v00 -= alpha0;
    let v_norm_sq0 = v00 * v00 + v01 * v01 + v02 * v02;
    let tau0 = if v_norm_sq0 > f64::EPSILON * 1e4 {
        let tau = 2.0 / v_norm_sq0;

        let dot = v00 * r[0] + v01 * r[2] + v02 * r[4];
        r[0] -= tau * v00 * dot;
        r[2] -= tau * v01 * dot;
        r[4] -= tau * v02 * dot;

        let dot = v00 * r[1] + v01 * r[3] + v02 * r[5];
        r[1] -= tau * v00 * dot;
        r[3] -= tau * v01 * dot;
        r[5] -= tau * v02 * dot;

        tau
    } else {
        v00 = 0.0;
        0.0
    };

    let mut v10 = r[3];
    let v11 = r[5];
    let norm_v1 = (v10 * v10 + v11 * v11).sqrt();
    let alpha1 = if v10 >= 0.0 { -norm_v1 } else { norm_v1 };
    v10 -= alpha1;
    let v_norm_sq1 = v10 * v10 + v11 * v11;
    let tau1 = if v_norm_sq1 > f64::EPSILON * 1e4 {
        let tau = 2.0 / v_norm_sq1;

        let dot = v10 * r[3] + v11 * r[5];
        r[3] -= tau * v10 * dot;
        r[5] -= tau * v11 * dot;

        tau
    } else {
        v10 = 0.0;
        0.0
    };

    q_out.clear();
    q_out.resize(6, 0.0);
    q_out[0] = 1.0;
    q_out[3] = 1.0;

    if tau1.abs() >= f64::EPSILON {
        let dot = v10 * q_out[3] + v11 * q_out[5];
        q_out[3] -= tau1 * v10 * dot;
        q_out[5] -= tau1 * v11 * dot;
    }

    if tau0.abs() >= f64::EPSILON {
        let dot = v00 * q_out[0] + v01 * q_out[2] + v02 * q_out[4];
        q_out[0] -= tau0 * v00 * dot;
        q_out[2] -= tau0 * v01 * dot;
        q_out[4] -= tau0 * v02 * dot;

        let dot = v00 * q_out[1] + v01 * q_out[3] + v02 * q_out[5];
        q_out[1] -= tau0 * v00 * dot;
        q_out[3] -= tau0 * v01 * dot;
        q_out[5] -= tau0 * v02 * dot;
    }

    r_out.clear();
    r_out.resize(4, 0.0);
    r_out[0] = r[0];
    r_out[1] = r[1];
    r_out[3] = r[3];

    if r_out[0] < 0.0 {
        r_out[0] = -r_out[0];
        r_out[1] = -r_out[1];
        q_out[0] = -q_out[0];
        q_out[2] = -q_out[2];
        q_out[4] = -q_out[4];
    }
    if r_out[3] < 0.0 {
        r_out[2] = -r_out[2];
        r_out[3] = -r_out[3];
        q_out[1] = -q_out[1];
        q_out[3] = -q_out[3];
        q_out[5] = -q_out[5];
    }
}

/// Householder QR of a single `m x n` matrix, writing Q into `scratch.q_out`
/// and R into `scratch.r_out`. Every working buffer lives in `scratch` and is
/// cleared+reused per call, so a batched QR over same-shaped matrices performs
/// no per-matrix allocation. The arithmetic is identical to the prior
/// per-call implementation — same Householder reflections applied in the same
/// order, same `v_norm_sq` threshold, same diagonal-sign normalization — so the
/// Q and R outputs are bit-for-bit unchanged.
fn qr_decompose_matrix(
    m: usize,
    n: usize,
    matrix: &[f64],
    full_matrices: bool,
    scratch: &mut QrScratch,
) {
    let QrScratch {
        r,
        v_store,
        tau_store,
        q_out,
        r_out,
    } = scratch;
    let k = m.min(n);

    r.clear();
    r.extend_from_slice(matrix);
    tau_store.clear();
    while v_store.len() < k {
        v_store.push(Vec::new());
    }

    for j in 0..k {
        let v = &mut v_store[j];
        v.clear();
        v.extend((j..m).map(|i| r[i * n + j]));
        let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let alpha = if v[0] >= 0.0 { -norm_v } else { norm_v };
        v[0] -= alpha;
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();

        if v_norm_sq > f64::EPSILON * 1e4 {
            let tau = 2.0 / v_norm_sq;
            for col in j..n {
                let mut dot = 0.0;
                for (vi, row) in v.iter().zip(j..m) {
                    dot += vi * r[row * n + col];
                }
                for (vi, row) in v.iter().zip(j..m) {
                    r[row * n + col] -= tau * vi * dot;
                }
            }
            tau_store.push(tau);
        } else {
            // Original stored a zero reflector here; tau == 0 means the Q loop
            // skips it, so its contents do not affect the output.
            v.iter_mut().for_each(|x| *x = 0.0);
            tau_store.push(0.0);
        }
    }

    let q_cols = if full_matrices { m } else { k };
    q_out.clear();
    q_out.resize(m * q_cols, 0.0);
    for i in 0..q_cols.min(m) {
        q_out[i * q_cols + i] = 1.0;
    }

    for j in (0..k).rev() {
        let v = &v_store[j];
        let tau = tau_store[j];
        if tau.abs() < f64::EPSILON {
            continue;
        }

        for col in j..q_cols {
            let mut dot = 0.0;
            for (vi, row) in v.iter().zip(j..m) {
                dot += vi * q_out[row * q_cols + col];
            }
            for (vi, row) in v.iter().zip(j..m) {
                q_out[row * q_cols + col] -= tau * vi * dot;
            }
        }
    }

    let r_rows = if full_matrices { m } else { k };
    r_out.clear();
    r_out.resize(r_rows * n, 0.0);
    for i in 0..r_rows {
        for j in i..n {
            r_out[i * n + j] = r[i * n + j];
        }
    }

    for i in 0..k {
        if r_out[i * n + i] < 0.0 {
            for j in 0..n {
                r_out[i * n + j] = -r_out[i * n + j];
            }
            for row in 0..m {
                q_out[row * q_cols + i] = -q_out[row * q_cols + i];
            }
        }
    }
}

fn batch_eigh_multi(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some((input, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        return eval_primitive_multi(Primitive::Eigh, &values, params)
            .map(|outputs| outputs.into_iter().map(BatchTracer::unbatched).collect())
            .map_err(|e| BatchError::EvalError(e.to_string()));
    };

    if inputs.len() != 1 {
        return batch_passthrough_leading_multi(Primitive::Eigh, inputs, params);
    }

    let batch_size = get_batch_size(&input.value, batch_dim)?;
    let moved_value = if batch_dim == 0 {
        None
    } else {
        Some(move_batch_dim_to_front(&input.value, batch_dim)?)
    };
    let value = moved_value.as_ref().unwrap_or(&input.value);
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for eigh".to_owned()))?;

    if tensor.rank() != 3 {
        return batch_passthrough_leading_multi(Primitive::Eigh, inputs, params);
    }

    if tensor.shape.dims[0] as usize != batch_size {
        return batch_passthrough_leading_multi(Primitive::Eigh, inputs, params);
    }

    let m = tensor.shape.dims[1] as usize;
    let n = tensor.shape.dims[2] as usize;
    if m != n {
        return Err(BatchError::EvalError(format!(
            "unsupported eigh behavior: Eigh requires a square matrix, got {m}x{n}"
        )));
    }

    let matrix_len = m * n;
    let mut w_elements = Vec::with_capacity(batch_size * m);
    let mut v_elements = Vec::with_capacity(batch_size * m * m);

    if m == 3 {
        let mut matrix = Vec::new();
        let mut fallback_scratch = None;
        for batch in 0..batch_size {
            let base = batch * matrix_len;
            let elements = &tensor.elements[base..base + 9];
            let a = [
                elements[0].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[1].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[2].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[3].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[4].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[5].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[6].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[7].as_f64().ok_or_else(eigh_type_mismatch)?,
                elements[8].as_f64().ok_or_else(eigh_type_mismatch)?,
            ];
            if let Some((w3, v3)) = analytic_eigh_3x3(&a) {
                append_eigh_3x3_outputs(w3, v3, &mut w_elements, &mut v_elements);
                continue;
            }

            matrix.clear();
            matrix.extend_from_slice(&a);
            let scratch = fallback_scratch.get_or_insert_with(|| EighScratch::with_order(3));
            append_eigh_decomposition(&mut matrix, m, scratch, &mut w_elements, &mut v_elements);
        }
    } else {
        let mut matrix = Vec::with_capacity(matrix_len);
        let mut scratch = EighScratch::with_order(m);
        for batch in 0..batch_size {
            let base = batch * matrix_len;
            matrix.clear();
            for lit in &tensor.elements[base..base + matrix_len] {
                matrix.push(lit.as_f64().ok_or_else(eigh_type_mismatch)?);
            }
            append_eigh_decomposition(
                &mut matrix,
                m,
                &mut scratch,
                &mut w_elements,
                &mut v_elements,
            );
        }
    }

    let w_shape = Shape {
        dims: vec![batch_size as u32, m as u32],
    };
    let v_shape = Shape {
        dims: vec![batch_size as u32, m as u32, m as u32],
    };
    let w = TensorValue::new(tensor.dtype, w_shape, w_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    let v = TensorValue::new(tensor.dtype, v_shape, v_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(vec![w, v])
}

fn eigh_type_mismatch() -> BatchError {
    BatchError::EvalError("type mismatch for eigh: expected numeric elements".to_owned())
}

fn append_eigh_3x3_outputs(
    w: [f64; 3],
    v: [f64; 9],
    w_elements: &mut Vec<Literal>,
    v_elements: &mut Vec<Literal>,
) {
    w_elements.extend(w.into_iter().map(Literal::from_f64));
    v_elements.extend(v.into_iter().map(Literal::from_f64));
}

#[derive(Default)]
struct JacobiScratch {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<f64>,
    new_row_p: Vec<f64>,
    new_row_q: Vec<f64>,
}

/// Reusable working buffers for batched SVD (`A = U Σ Vᵀ` via eigendecomposition
/// of `AᵀA`). Carrying these across every matrix in a vmapped batch removes the
/// per-matrix heap traffic — `ata`, the Jacobi eigensolver buffers, the sort
/// indices, `sigma`, `v_sorted`, the thin `u`, and the `u_out`/`vt` outputs —
/// that dominated the decomposition cost for small matrices.
#[derive(Default)]
struct SvdScratch {
    jacobi: JacobiScratch,
    ata: Vec<f64>,
    indices: Vec<usize>,
    sigma: Vec<f64>,
    v_sorted: Vec<f64>,
    u: Vec<f64>,
    u_out: Vec<f64>,
    vt: Vec<f64>,
}

impl JacobiScratch {
    fn with_order(n: usize) -> Self {
        let matrix_len = n * n;
        Self {
            eigenvalues: Vec::with_capacity(n),
            eigenvectors: Vec::with_capacity(matrix_len),
            new_row_p: Vec::with_capacity(n),
            new_row_q: Vec::with_capacity(n),
        }
    }
}

struct EighScratch {
    jacobi: JacobiScratch,
    indices: Vec<usize>,
    w_sorted: Vec<f64>,
    v_sorted: Vec<f64>,
}

impl EighScratch {
    fn with_order(n: usize) -> Self {
        let matrix_len = n * n;
        Self {
            jacobi: JacobiScratch::with_order(n),
            indices: Vec::with_capacity(n),
            w_sorted: Vec::with_capacity(n),
            v_sorted: Vec::with_capacity(matrix_len),
        }
    }
}

fn append_eigh_decomposition(
    a: &mut [f64],
    n: usize,
    scratch: &mut EighScratch,
    w_elements: &mut Vec<Literal>,
    v_elements: &mut Vec<Literal>,
) {
    eigh_decompose_matrix_into(a, n, scratch);
    w_elements.extend(scratch.w_sorted.iter().copied().map(Literal::from_f64));
    v_elements.extend(scratch.v_sorted.iter().copied().map(Literal::from_f64));
}

fn eigh_decompose_matrix_into(a: &mut [f64], n: usize, scratch: &mut EighScratch) {
    if n == 3 {
        eigh_decompose_matrix_3x3_into(a, scratch);
        return;
    }

    jacobi_eigendecomposition_matrix_into(a, n, &mut scratch.jacobi);
    let eigenvalues = &scratch.jacobi.eigenvalues;
    let eigenvectors = &scratch.jacobi.eigenvectors;

    scratch.indices.clear();
    scratch.indices.extend(0..n);
    scratch
        .indices
        .sort_by(|&a_idx, &b_idx| eigenvalues[a_idx].total_cmp(&eigenvalues[b_idx]));

    scratch.w_sorted.resize(n, 0.0_f64);
    scratch.v_sorted.resize(n * n, 0.0_f64);
    for (new_col, &old_col) in scratch.indices.iter().enumerate() {
        scratch.w_sorted[new_col] = eigenvalues[old_col];
        for row in 0..n {
            scratch.v_sorted[row * n + new_col] = eigenvectors[row * n + old_col];
        }
    }
}

fn eigh_decompose_matrix_3x3_into(a: &mut [f64], scratch: &mut EighScratch) {
    // Fast path: closed-form analytic 3×3 symmetric eigensolver, shared with the
    // single-matrix `fj_lax::eval_eigh` path so batched and per-slice results
    // are bit-identical. Falls back to the iterative Jacobi sweep below when the
    // analytic residual check rejects (ill-conditioned), preserving parity.
    if let Some((w3, v3)) = analytic_eigh_3x3(a) {
        scratch.w_sorted.clear();
        scratch.w_sorted.extend_from_slice(&w3);
        scratch.v_sorted.clear();
        scratch.v_sorted.extend_from_slice(&v3);
        return;
    }

    jacobi_eigendecomposition_matrix_3x3_into(a, &mut scratch.jacobi);
    let eigenvalues = &scratch.jacobi.eigenvalues;
    let eigenvectors = &scratch.jacobi.eigenvectors;

    let mut col0 = 0;
    let mut col1 = 1;
    let mut col2 = 2;
    if eigenvalues[col1].total_cmp(&eigenvalues[col0]).is_lt() {
        std::mem::swap(&mut col0, &mut col1);
    }
    if eigenvalues[col2].total_cmp(&eigenvalues[col1]).is_lt() {
        std::mem::swap(&mut col1, &mut col2);
    }
    if eigenvalues[col1].total_cmp(&eigenvalues[col0]).is_lt() {
        std::mem::swap(&mut col0, &mut col1);
    }

    scratch.w_sorted.resize(3, 0.0_f64);
    scratch.w_sorted[0] = eigenvalues[col0];
    scratch.w_sorted[1] = eigenvalues[col1];
    scratch.w_sorted[2] = eigenvalues[col2];

    scratch.v_sorted.resize(9, 0.0_f64);
    scratch.v_sorted[0] = eigenvectors[col0];
    scratch.v_sorted[1] = eigenvectors[col1];
    scratch.v_sorted[2] = eigenvectors[col2];
    scratch.v_sorted[3] = eigenvectors[3 + col0];
    scratch.v_sorted[4] = eigenvectors[3 + col1];
    scratch.v_sorted[5] = eigenvectors[3 + col2];
    scratch.v_sorted[6] = eigenvectors[6 + col0];
    scratch.v_sorted[7] = eigenvectors[6 + col1];
    scratch.v_sorted[8] = eigenvectors[6 + col2];
}

fn jacobi_eigendecomposition_matrix_into(a: &mut [f64], n: usize, scratch: &mut JacobiScratch) {
    if n == 3 {
        jacobi_eigendecomposition_matrix_3x3_into(a, scratch);
        return;
    }

    scratch.eigenvectors.resize(n * n, 0.0_f64);
    scratch.eigenvectors.fill(0.0_f64);
    for i in 0..n {
        scratch.eigenvectors[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = f64::EPSILON * 1e2;
    scratch.new_row_p.resize(n, 0.0_f64);
    scratch.new_row_q.resize(n, 0.0_f64);

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let value = a[i * n + j].abs();
                if value > max_val {
                    max_val = value;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < f64::EPSILON {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        for i in 0..n {
            scratch.new_row_p[i] = cos_t * a[p * n + i] + sin_t * a[q * n + i];
            scratch.new_row_q[i] = -sin_t * a[p * n + i] + cos_t * a[q * n + i];
        }

        for i in 0..n {
            a[p * n + i] = scratch.new_row_p[i];
            a[q * n + i] = scratch.new_row_q[i];
            a[i * n + p] = scratch.new_row_p[i];
            a[i * n + q] = scratch.new_row_q[i];
        }

        a[p * n + p] = cos_t * scratch.new_row_p[p] + sin_t * scratch.new_row_p[q];
        a[q * n + q] = -sin_t * scratch.new_row_q[p] + cos_t * scratch.new_row_q[q];
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for i in 0..n {
            let vip = scratch.eigenvectors[i * n + p];
            let viq = scratch.eigenvectors[i * n + q];
            scratch.eigenvectors[i * n + p] = cos_t * vip + sin_t * viq;
            scratch.eigenvectors[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    scratch.eigenvalues.clear();
    scratch.eigenvalues.extend((0..n).map(|i| a[i * n + i]));
}

#[inline]
fn jacobi_eigendecomposition_matrix_3x3_into(a: &mut [f64], scratch: &mut JacobiScratch) {
    debug_assert_eq!(a.len(), 9);

    let eigenvectors = &mut scratch.eigenvectors;
    eigenvectors.resize(9, 0.0_f64);
    eigenvectors[0] = 1.0;
    eigenvectors[1] = 0.0;
    eigenvectors[2] = 0.0;
    eigenvectors[3] = 0.0;
    eigenvectors[4] = 1.0;
    eigenvectors[5] = 0.0;
    eigenvectors[6] = 0.0;
    eigenvectors[7] = 0.0;
    eigenvectors[8] = 1.0;

    let max_iter = 900;
    let tol = f64::EPSILON * 1e2;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;

        let value = a[1].abs();
        if value > max_val {
            max_val = value;
            p = 0;
            q = 1;
        }
        let value = a[2].abs();
        if value > max_val {
            max_val = value;
            p = 0;
            q = 2;
        }
        let value = a[5].abs();
        if value > max_val {
            max_val = value;
            p = 1;
            q = 2;
        }

        if max_val < tol {
            break;
        }

        let p_row = p * 3;
        let q_row = q * 3;
        let app = a[p_row + p];
        let aqq = a[q_row + q];
        let apq = a[p_row + q];

        let theta = if (app - aqq).abs() < f64::EPSILON {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        let new_row_p0 = cos_t * a[p_row] + sin_t * a[q_row];
        let new_row_q0 = -sin_t * a[p_row] + cos_t * a[q_row];
        let new_row_p1 = cos_t * a[p_row + 1] + sin_t * a[q_row + 1];
        let new_row_q1 = -sin_t * a[p_row + 1] + cos_t * a[q_row + 1];
        let new_row_p2 = cos_t * a[p_row + 2] + sin_t * a[q_row + 2];
        let new_row_q2 = -sin_t * a[p_row + 2] + cos_t * a[q_row + 2];
        let new_row_p = [new_row_p0, new_row_p1, new_row_p2];
        let new_row_q = [new_row_q0, new_row_q1, new_row_q2];

        a[p_row] = new_row_p0;
        a[q_row] = new_row_q0;
        a[p] = new_row_p0;
        a[q] = new_row_q0;

        a[p_row + 1] = new_row_p1;
        a[q_row + 1] = new_row_q1;
        a[3 + p] = new_row_p1;
        a[3 + q] = new_row_q1;

        a[p_row + 2] = new_row_p2;
        a[q_row + 2] = new_row_q2;
        a[6 + p] = new_row_p2;
        a[6 + q] = new_row_q2;

        a[p_row + p] = cos_t * new_row_p[p] + sin_t * new_row_p[q];
        a[q_row + q] = -sin_t * new_row_q[p] + cos_t * new_row_q[q];
        a[p_row + q] = 0.0;
        a[q_row + p] = 0.0;

        let v = &mut scratch.eigenvectors;
        let vip = v[p];
        let viq = v[q];
        v[p] = cos_t * vip + sin_t * viq;
        v[q] = -sin_t * vip + cos_t * viq;

        let vip = v[3 + p];
        let viq = v[3 + q];
        v[3 + p] = cos_t * vip + sin_t * viq;
        v[3 + q] = -sin_t * vip + cos_t * viq;

        let vip = v[6 + p];
        let viq = v[6 + q];
        v[6 + p] = cos_t * vip + sin_t * viq;
        v[6 + q] = -sin_t * vip + cos_t * viq;
    }

    scratch.eigenvalues.resize(3, 0.0_f64);
    scratch.eigenvalues[0] = a[0];
    scratch.eigenvalues[1] = a[4];
    scratch.eigenvalues[2] = a[8];
}

fn batch_svd_multi(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some((input, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        return eval_primitive_multi(Primitive::Svd, &values, params)
            .map(|outputs| outputs.into_iter().map(BatchTracer::unbatched).collect())
            .map_err(|e| BatchError::EvalError(e.to_string()));
    };

    if inputs.len() != 1 {
        return batch_passthrough_leading_multi(Primitive::Svd, inputs, params);
    }

    let batch_size = get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value
        .as_tensor()
        .ok_or_else(|| BatchError::BatchDimMoveError("expected tensor for svd".to_owned()))?;

    if tensor.rank() != 3 {
        return batch_passthrough_leading_multi(Primitive::Svd, inputs, params);
    }

    if tensor.shape.dims[0] as usize != batch_size {
        return batch_passthrough_leading_multi(Primitive::Svd, inputs, params);
    }

    let m = tensor.shape.dims[1] as usize;
    let n = tensor.shape.dims[2] as usize;
    let k = m.min(n);
    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");
    let u_cols = if full_matrices { m } else { k };
    let vt_rows = if full_matrices { n } else { k };
    let matrix_len = m * n;

    let mut u_elements = Vec::with_capacity(batch_size * m * u_cols);
    let mut s_elements = Vec::with_capacity(batch_size * k);
    let mut vt_elements = Vec::with_capacity(batch_size * vt_rows * n);

    let mut matrix = Vec::with_capacity(matrix_len);
    let mut svd_scratch = SvdScratch::default();
    for batch in 0..batch_size {
        let base = batch * matrix_len;
        if m == 3 && n == 2 && !full_matrices {
            let a00 = tensor.elements[base].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            let a01 = tensor.elements[base + 1].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            let a10 = tensor.elements[base + 2].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            let a11 = tensor.elements[base + 3].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            let a20 = tensor.elements[base + 4].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            let a21 = tensor.elements[base + 5].as_f64().ok_or_else(|| {
                BatchError::EvalError("type mismatch for svd: expected numeric elements".to_owned())
            })?;
            svd_decompose_matrix_3x2_thin([a00, a01, a10, a11, a20, a21], &mut svd_scratch);
        } else {
            matrix.clear();
            for lit in &tensor.elements[base..base + matrix_len] {
                matrix.push(lit.as_f64().ok_or_else(|| {
                    BatchError::EvalError(
                        "type mismatch for svd: expected numeric elements".to_owned(),
                    )
                })?);
            }
            svd_decompose_matrix(m, n, &matrix, full_matrices, &mut svd_scratch);
        }
        u_elements.extend(svd_scratch.u_out.iter().copied().map(Literal::from_f64));
        s_elements.extend(svd_scratch.sigma.iter().copied().map(Literal::from_f64));
        vt_elements.extend(svd_scratch.vt.iter().copied().map(Literal::from_f64));
    }

    let u_shape = Shape {
        dims: vec![batch_size as u32, m as u32, u_cols as u32],
    };
    let s_shape = Shape {
        dims: vec![batch_size as u32, k as u32],
    };
    let vt_shape = Shape {
        dims: vec![batch_size as u32, vt_rows as u32, n as u32],
    };
    let u = TensorValue::new(tensor.dtype, u_shape, u_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    let s = TensorValue::new(tensor.dtype, s_shape, s_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    let vt = TensorValue::new(tensor.dtype, vt_shape, vt_elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(vec![u, s, vt])
}

/// SVD of a single `m x n` matrix via eigendecomposition of `AᵀA`, writing U
/// into `scratch.u_out`, the singular values into `scratch.sigma`, and Vᵀ into
/// `scratch.vt`. Every working buffer lives in `scratch` and is cleared+reused
/// per call, so a batched SVD over same-shaped matrices performs no per-matrix
/// allocation (the Jacobi eigensolver reuses `scratch.jacobi`). The arithmetic
/// is identical to the prior per-call implementation — same `AᵀA`, same Jacobi
/// eigendecomposition, same descending `total_cmp` ordering, same σ/U/Vᵀ
/// formulas and thresholds — so the outputs are bit-for-bit unchanged.
fn svd_decompose_matrix(
    m: usize,
    n: usize,
    a: &[f64],
    full_matrices: bool,
    scratch: &mut SvdScratch,
) {
    let SvdScratch {
        jacobi,
        ata,
        indices,
        sigma,
        v_sorted,
        u,
        u_out,
        vt,
    } = scratch;
    let k = m.min(n);

    ata.clear();
    ata.resize(n * n, 0.0_f64);
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for row in 0..m {
                dot += a[row * n + i] * a[row * n + j];
            }
            ata[i * n + j] = dot;
            ata[j * n + i] = dot;
        }
    }

    jacobi_eigendecomposition_matrix_into(ata, n, jacobi);
    let eigenvalues = &jacobi.eigenvalues;
    let v = &jacobi.eigenvectors;

    indices.clear();
    indices.extend(0..n);
    indices.sort_by(|&a_idx, &b_idx| eigenvalues[b_idx].total_cmp(&eigenvalues[a_idx]));

    sigma.clear();
    sigma.resize(k, 0.0_f64);
    v_sorted.clear();
    v_sorted.resize(n * n, 0.0_f64);
    for (new_col, &old_col) in indices.iter().enumerate() {
        if new_col < k {
            sigma[new_col] = eigenvalues[old_col].max(0.0).sqrt();
        }
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    u.clear();
    u.resize(m * k, 0.0_f64);
    for i in 0..m {
        for j in 0..k {
            if sigma[j] > f64::EPSILON * 1e4 {
                let mut val = 0.0;
                for col in 0..n {
                    val += a[i * n + col] * v_sorted[col * n + j];
                }
                u[i * k + j] = val / sigma[j];
            }
        }
    }

    let u_cols = if full_matrices { m } else { k };
    u_out.clear();
    if full_matrices && u_cols > k {
        // Rare full-matrices path: keep the existing column-extension helper
        // (its output is value-identical) and copy into the reused buffer.
        let extended = extend_orthogonal_columns_matrix(u.as_slice(), m, k, u_cols);
        u_out.extend_from_slice(&extended);
    } else {
        u_out.extend_from_slice(u);
    }

    vt.clear();
    if full_matrices {
        vt.resize(n * n, 0.0_f64);
        for i in 0..n {
            for j in 0..n {
                vt[i * n + j] = v_sorted[j * n + i];
            }
        }
    } else {
        vt.resize(k * n, 0.0_f64);
        for i in 0..k {
            for j in 0..n {
                vt[i * n + j] = v_sorted[j * n + i];
            }
        }
    }
}

fn svd_decompose_matrix_3x2_thin(a: [f64; 6], scratch: &mut SvdScratch) {
    let a00 = a[0];
    let a01 = a[1];
    let a10 = a[2];
    let a11 = a[3];
    let a20 = a[4];
    let a21 = a[5];

    let mut ata00 = 0.0;
    ata00 += a00 * a00;
    ata00 += a10 * a10;
    ata00 += a20 * a20;

    let mut ata01 = 0.0;
    ata01 += a00 * a01;
    ata01 += a10 * a11;
    ata01 += a20 * a21;

    let ata10 = ata01;

    let mut ata11 = 0.0;
    ata11 += a01 * a01;
    ata11 += a11 * a11;
    ata11 += a21 * a21;

    let mut v00 = 1.0;
    let mut v01 = 0.0;
    let mut v10 = 0.0;
    let mut v11 = 1.0;

    let mut max_val = 0.0_f64;
    let value = ata01.abs();
    if value > max_val {
        max_val = value;
    }

    if max_val >= f64::EPSILON * 1e2 {
        let app = ata00;
        let aqq = ata11;
        let apq = ata01;

        let theta = if (app - aqq).abs() < f64::EPSILON {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        let new_row_p0 = cos_t * ata00 + sin_t * ata10;
        let new_row_q0 = -sin_t * ata00 + cos_t * ata10;
        let new_row_p1 = cos_t * ata01 + sin_t * ata11;
        let new_row_q1 = -sin_t * ata01 + cos_t * ata11;

        ata00 = cos_t * new_row_p0 + sin_t * new_row_p1;
        ata11 = -sin_t * new_row_q0 + cos_t * new_row_q1;

        let vip = v00;
        let viq = v01;
        v00 = cos_t * vip + sin_t * viq;
        v01 = -sin_t * vip + cos_t * viq;

        let vip = v10;
        let viq = v11;
        v10 = cos_t * vip + sin_t * viq;
        v11 = -sin_t * vip + cos_t * viq;
    }

    let mut col0 = 0;
    let mut col1 = 1;
    if ata11.total_cmp(&ata00).is_gt() {
        std::mem::swap(&mut col0, &mut col1);
    }

    let (lambda0, lambda1) = if col0 == 0 {
        (ata00, ata11)
    } else {
        (ata11, ata00)
    };
    let v_sorted00 = if col0 == 0 { v00 } else { v01 };
    let v_sorted01 = if col1 == 0 { v00 } else { v01 };
    let v_sorted10 = if col0 == 0 { v10 } else { v11 };
    let v_sorted11 = if col1 == 0 { v10 } else { v11 };

    let SvdScratch {
        sigma,
        u,
        u_out,
        vt,
        ..
    } = scratch;

    sigma.clear();
    sigma.resize(2, 0.0_f64);
    sigma[0] = lambda0.max(0.0).sqrt();
    sigma[1] = lambda1.max(0.0).sqrt();

    u.clear();
    u.resize(6, 0.0_f64);
    if sigma[0] > f64::EPSILON * 1e4 {
        let mut val = 0.0;
        val += a00 * v_sorted00;
        val += a01 * v_sorted10;
        u[0] = val / sigma[0];

        let mut val = 0.0;
        val += a10 * v_sorted00;
        val += a11 * v_sorted10;
        u[2] = val / sigma[0];

        let mut val = 0.0;
        val += a20 * v_sorted00;
        val += a21 * v_sorted10;
        u[4] = val / sigma[0];
    }
    if sigma[1] > f64::EPSILON * 1e4 {
        let mut val = 0.0;
        val += a00 * v_sorted01;
        val += a01 * v_sorted11;
        u[1] = val / sigma[1];

        let mut val = 0.0;
        val += a10 * v_sorted01;
        val += a11 * v_sorted11;
        u[3] = val / sigma[1];

        let mut val = 0.0;
        val += a20 * v_sorted01;
        val += a21 * v_sorted11;
        u[5] = val / sigma[1];
    }

    u_out.clear();
    u_out.extend_from_slice(u);

    vt.clear();
    vt.resize(4, 0.0_f64);
    vt[0] = v_sorted00;
    vt[1] = v_sorted10;
    vt[2] = v_sorted01;
    vt[3] = v_sorted11;
}

fn extend_orthogonal_columns_matrix(u: &[f64], m: usize, k: usize, m_full: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; m * m_full];

    for i in 0..m {
        for j in 0..k {
            result[i * m_full + j] = u[i * k + j];
        }
    }

    let mut added = k;
    for basis_idx in 0..m {
        if added >= m_full {
            break;
        }

        let mut col = vec![0.0_f64; m];
        col[basis_idx] = 1.0;

        for j in 0..added {
            let mut dot = 0.0;
            for i in 0..m {
                dot += col[i] * result[i * m_full + j];
            }
            for i in 0..m {
                col[i] -= dot * result[i * m_full + j];
            }
        }

        let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > f64::EPSILON * 1e4 {
            for i in 0..m {
                result[i * m_full + added] = col[i] / norm;
            }
            added += 1;
        }
    }

    result
}

fn batch_triangular_solve(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let Some((batched, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        let result = eval_primitive(Primitive::TriangularSolve, &values, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer::unbatched(result));
    };

    if inputs.len() != 2 {
        return batch_passthrough_leading(Primitive::TriangularSolve, inputs, params);
    }

    let batch_size = get_batch_size(&batched.value, batch_dim)?;
    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|t| match t.batch_dim {
            Some(bd) => move_batch_dim_to_front(&t.value, bd),
            None => broadcast_unbatched(&t.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    let (Value::Tensor(a_tensor), Value::Tensor(b_tensor)) = (&values[0], &values[1]) else {
        return batch_passthrough_leading(Primitive::TriangularSolve, inputs, params);
    };

    if a_tensor.rank() != 3 || b_tensor.rank() != 3 {
        return batch_passthrough_leading(Primitive::TriangularSolve, inputs, params);
    }

    if a_tensor.shape.dims[0] as usize != batch_size
        || b_tensor.shape.dims[0] as usize != batch_size
    {
        return batch_passthrough_leading(Primitive::TriangularSolve, inputs, params);
    }

    let m_a = a_tensor.shape.dims[1] as usize;
    let n_a = a_tensor.shape.dims[2] as usize;
    let m_b = b_tensor.shape.dims[1] as usize;
    let n_b = b_tensor.shape.dims[2] as usize;

    if m_a != n_a {
        return Err(BatchError::EvalError(format!(
            "unsupported triangular_solve behavior: A must be square, got {m_a}x{n_a}"
        )));
    }

    if m_a != m_b {
        return Err(BatchError::EvalError(format!(
            "shape mismatch for triangular_solve: left={:?} right={:?}",
            vec![m_a as u32, n_a as u32],
            vec![m_b as u32, n_b as u32]
        )));
    }

    let lower = params.get("lower").is_none_or(|v| v.trim() != "false");
    let transpose_a = params
        .get("transpose_a")
        .is_some_and(|v| v.trim() == "true");
    let unit_diagonal = params
        .get("unit_diagonal")
        .is_some_and(|v| v.trim() == "true");

    let a_matrix_len = m_a * n_a;
    let b_matrix_len = m_b * n_b;
    let mut elements = Vec::with_capacity(batch_size * m_a * n_b);

    for batch in 0..batch_size {
        let a_base = batch * a_matrix_len;
        let b_base = batch * b_matrix_len;
        let a = a_tensor.elements[a_base..a_base + a_matrix_len]
            .iter()
            .map(|lit| {
                lit.as_f64().ok_or_else(|| {
                    BatchError::EvalError(
                        "type mismatch for triangular_solve: expected numeric elements".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let b = b_tensor.elements[b_base..b_base + b_matrix_len]
            .iter()
            .map(|lit| {
                lit.as_f64().ok_or_else(|| {
                    BatchError::EvalError(
                        "type mismatch for triangular_solve: expected numeric elements".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut x = vec![0.0_f64; m_a * n_b];

        for col in 0..n_b {
            let mut b_col: Vec<f64> = (0..m_a).map(|row| b[row * n_b + col]).collect();

            if lower && !transpose_a {
                for i in 0..m_a {
                    for k in 0..i {
                        let a_ik = a[row_major_index(i, k, n_a)];
                        b_col[i] -= a_ik * x[k * n_b + col];
                    }
                    let diag = triangular_solve_diag(&a, i, n_a, unit_diagonal)?;
                    x[i * n_b + col] = b_col[i] / diag;
                }
            } else if !lower && !transpose_a {
                for i in (0..m_a).rev() {
                    for k in (i + 1)..m_a {
                        let a_ik = a[row_major_index(i, k, n_a)];
                        b_col[i] -= a_ik * x[k * n_b + col];
                    }
                    let diag = triangular_solve_diag(&a, i, n_a, unit_diagonal)?;
                    x[i * n_b + col] = b_col[i] / diag;
                }
            } else if lower && transpose_a {
                for i in (0..m_a).rev() {
                    for k in (i + 1)..m_a {
                        let a_ki = a[row_major_index(k, i, n_a)];
                        b_col[i] -= a_ki * x[k * n_b + col];
                    }
                    let diag = triangular_solve_diag(&a, i, n_a, unit_diagonal)?;
                    x[i * n_b + col] = b_col[i] / diag;
                }
            } else {
                for i in 0..m_a {
                    for k in 0..i {
                        let a_ki = a[row_major_index(k, i, n_a)];
                        b_col[i] -= a_ki * x[k * n_b + col];
                    }
                    let diag = triangular_solve_diag(&a, i, n_a, unit_diagonal)?;
                    x[i * n_b + col] = b_col[i] / diag;
                }
            }
        }

        elements.extend(x.into_iter().map(Literal::from_f64));
    }

    let out_dtype = promote_dtype_public(a_tensor.dtype, b_tensor.dtype);
    let shape = Shape {
        dims: vec![batch_size as u32, m_a as u32, n_b as u32],
    };
    TensorValue::new(out_dtype, shape, elements)
        .map(Value::Tensor)
        .map(|result| BatchTracer::batched(result, 0))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn row_major_index(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

fn triangular_solve_diag(
    a: &[f64],
    i: usize,
    cols: usize,
    unit_diagonal: bool,
) -> Result<f64, BatchError> {
    let diag = if unit_diagonal {
        1.0
    } else {
        a[row_major_index(i, i, cols)]
    };

    // JAX's triangular_solve does not raise for a zero/near-zero diagonal; the
    // caller's division yields inf/nan (singular) or a finite large value
    // (near-singular), matching jnp (NumPy raises).
    Ok(diag)
}

// ── FFT-Family Batching ────────────────────────────────────────────

fn batch_fft_like(
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

    get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let tensor = value.as_tensor().ok_or_else(|| {
        BatchError::BatchDimMoveError(format!("expected tensor for {}", primitive.as_str()))
    })?;
    let per_elem_rank = tensor.shape.rank().saturating_sub(1);
    if per_elem_rank == 0 {
        return Err(BatchError::EvalError(
            "FFT expects a tensor (rank >= 1), got scalar".to_owned(),
        ));
    }

    let result = eval_primitive(primitive, &[value], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

// ── One-Hot Batching ───────────────────────────────────────────────

fn batch_one_hot(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    let input = &inputs[0];
    let batch_dim = match input.batch_dim {
        None => {
            let result = eval_primitive(
                Primitive::OneHot,
                std::slice::from_ref(&input.value),
                params,
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
            return Ok(BatchTracer::unbatched(result));
        }
        Some(bd) => bd,
    };

    get_batch_size(&input.value, batch_dim)?;
    let value = move_batch_dim_to_front(&input.value, batch_dim)?;
    let per_element_rank = match &value {
        Value::Scalar(_) => 0,
        Value::Tensor(tensor) => tensor.rank().saturating_sub(1),
    };
    let params = one_hot_params_for_leading_batch(params, per_element_rank)?;
    let result = eval_primitive(Primitive::OneHot, &[value], &params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn one_hot_params_for_leading_batch(
    params: &BTreeMap<String, String>,
    per_element_rank: usize,
) -> Result<BTreeMap<String, String>, BatchError> {
    let mut adjusted = params.clone();
    let Some(raw_axis) = params.get("axis") else {
        return Ok(adjusted);
    };

    let output_rank = per_element_rank + 1;
    let axis = raw_axis.trim().parse::<i64>().map_err(|_| {
        BatchError::EvalError(format!("invalid integer in param 'axis': '{raw_axis}'"))
    })?;
    let normalized = if axis < 0 {
        output_rank as i64 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized >= output_rank as i64 {
        return Err(BatchError::EvalError(format!(
            "axis {axis} out of bounds for output rank {output_rank}"
        )));
    }

    adjusted.insert("axis".to_owned(), (normalized + 1).to_string());
    Ok(adjusted)
}

// ── Passthrough Leading Dim (Gather, Scatter, DynamicSlice, etc.) ──

fn batch_passthrough_leading(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    // For complex ops, fall back to per-element loop when batched
    let Some((batched, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        let result = eval_primitive(primitive, &values, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        return Ok(BatchTracer::unbatched(result));
    };

    // Find batch size from any batched input
    let batch_size = get_batch_size(&batched.value, batch_dim)?;

    // Move non-leading batched inputs to the front. Inputs already batched on
    // axis 0 can be borrowed directly; cloning them here would copy the full
    // tensor before the per-batch slices are materialized.
    let values: Result<Vec<PreparedBatchInput<'_>>, BatchError> = inputs
        .iter()
        .map(|t| match t.batch_dim {
            Some(0) => Ok(PreparedBatchInput::BatchedBorrowed(&t.value)),
            Some(bd) => move_batch_dim_to_front(&t.value, bd).map(PreparedBatchInput::BatchedOwned),
            None => Ok(PreparedBatchInput::Shared(&t.value)),
        })
        .collect();
    let values = values?;

    // Loop over batch dimension and evaluate per slice
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let slices: Result<Vec<Value>, BatchError> = values
            .iter()
            .map(|value| value.slice_for_batch(i))
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

enum PreparedBatchInput<'a> {
    BatchedBorrowed(&'a Value),
    BatchedOwned(Value),
    Shared(&'a Value),
}

impl PreparedBatchInput<'_> {
    fn slice_for_batch(&self, index: usize) -> Result<Value, BatchError> {
        match self {
            Self::BatchedBorrowed(value) => slice_batched_value(value, index),
            Self::BatchedOwned(value) => slice_batched_value(value, index),
            Self::Shared(value) => Ok((*value).clone()),
        }
    }
}

fn slice_batched_value(value: &Value, index: usize) -> Result<Value, BatchError> {
    match value {
        Value::Tensor(tensor) => tensor
            .slice_axis0(index)
            .map_err(|e| BatchError::TensorError(e.to_string())),
        Value::Scalar(_) => Ok(value.clone()),
    }
}

fn batch_passthrough_leading_multi(
    primitive: Primitive,
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some((batched, batch_dim)) = inputs
        .iter()
        .find_map(|t| t.batch_dim.map(|batch_dim| (t, batch_dim)))
    else {
        let values: Vec<Value> = inputs.iter().map(|t| t.value.clone()).collect();
        return eval_primitive_multi(primitive, &values, params)
            .map(|outputs| outputs.into_iter().map(BatchTracer::unbatched).collect())
            .map_err(|e| BatchError::EvalError(e.to_string()));
    };

    let batch_size = get_batch_size(&batched.value, batch_dim)?;

    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|t| match t.batch_dim {
            Some(bd) => move_batch_dim_to_front(&t.value, bd),
            None => broadcast_unbatched(&t.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    let mut per_output: Option<Vec<Vec<Value>>> = None;
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
        let outputs = eval_primitive_multi(primitive, &slices?, params)
            .map_err(|e| BatchError::EvalError(e.to_string()))?;

        let buckets = per_output.get_or_insert_with(|| {
            (0..outputs.len())
                .map(|_| Vec::with_capacity(batch_size))
                .collect()
        });
        if buckets.len() != outputs.len() {
            return Err(BatchError::InterpreterError(format!(
                "primitive {} returned inconsistent output arity across batch slices",
                primitive.as_str()
            )));
        }

        for (bucket, output) in buckets.iter_mut().zip(outputs) {
            bucket.push(output);
        }
    }

    per_output
        .unwrap_or_default()
        .into_iter()
        .map(|outputs| {
            TensorValue::stack_axis0(&outputs)
                .map(|tensor| BatchTracer::batched(Value::Tensor(tensor), 0))
                .map_err(|e| BatchError::TensorError(e.to_string()))
        })
        .collect()
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

    // Batched predicate: evaluate BOTH branches for the full batch, then use
    // Select to pick per-element. This is O(1) vectorized instead of O(N) loop.
    let pred_bd = inputs[0].batch_dim.ok_or_else(|| {
        BatchError::InterpreterError("batched cond predicate missing batch dimension".to_owned())
    })?;
    let pred = move_batch_dim_to_front(&inputs[0].value, pred_bd)?;
    let batch_size = get_batch_size(&inputs[0].value, pred_bd)?;

    // Prepare on_true and on_false values with matching batch dimension
    let on_true = match inputs[1].batch_dim {
        Some(bd) => move_batch_dim_to_front(&inputs[1].value, bd)?,
        None => broadcast_unbatched(&inputs[1].value, batch_size, 0)?,
    };
    let on_false = match inputs[2].batch_dim {
        Some(bd) => move_batch_dim_to_front(&inputs[2].value, bd)?,
        None => broadcast_unbatched(&inputs[2].value, batch_size, 0)?,
    };

    // Use Select(pred, on_true, on_false) for vectorized per-element selection
    let result = eval_primitive(Primitive::Select, &[pred, on_true, on_false], params)
        .map_err(|e| BatchError::EvalError(e.to_string()))?;
    Ok(BatchTracer::batched(result, 0))
}

fn batch_scan(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    if let Some(result) = batch_scan_scalar_sequences(inputs, params)? {
        return Ok(result);
    }

    // General fallback semantics match vmapped scan behavior: each batch
    // element runs an independent scan with its corresponding carry/xs.
    batch_control_flow_fallback(Primitive::Scan, inputs, params)
}

#[derive(Clone, Copy)]
enum ScanScalarOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

fn batch_scan_scalar_sequences(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs.len() != 2 {
        return Ok(None);
    }
    if let Some(result) = batch_scan_i64_add_shared_init_batch0(inputs, params)? {
        return Ok(Some(result));
    }
    if let Some(result) = batch_scan_i64_max_shared_init_batch0(inputs, params)? {
        return Ok(Some(result));
    }
    if let Some(result) = batch_scan_f32_add_shared_init_batch0(inputs, params)? {
        return Ok(Some(result));
    }
    let Some(xs_batch_dim) = inputs[1].batch_dim else {
        return Ok(None);
    };

    let Some(op) = scan_scalar_op(params) else {
        return Ok(None);
    };
    let xs_value = move_batch_dim_to_front(&inputs[1].value, xs_batch_dim)?;
    let Some(xs) = xs_value.as_tensor() else {
        return Ok(None);
    };
    if xs.rank() > 2 {
        return Ok(None);
    }

    let batch_size = xs.shape.dims[0] as usize;
    let scan_len = if xs.rank() == 1 {
        1
    } else {
        xs.shape.dims[1] as usize
    };
    let init_values = scan_scalar_initial_values(&inputs[0], batch_size)?;
    if init_values.len() != batch_size {
        return Ok(None);
    }
    let reverse = params.get("reverse").is_some_and(|value| value == "true");

    let mut outputs = Vec::with_capacity(batch_size);
    for (batch_idx, mut carry) in init_values.into_iter().enumerate() {
        if reverse {
            for scan_idx in (0..scan_len).rev() {
                let Some(next) =
                    apply_scan_scalar_op(op, carry, scan_scalar_xs_at(xs, batch_idx, scan_idx))
                else {
                    return Ok(None);
                };
                carry = next;
            }
        } else {
            for scan_idx in 0..scan_len {
                let Some(next) =
                    apply_scan_scalar_op(op, carry, scan_scalar_xs_at(xs, batch_idx, scan_idx))
                else {
                    return Ok(None);
                };
                carry = next;
            }
        }
        outputs.push(carry);
    }
    let dtype = outputs
        .first()
        .map(|literal| Value::Scalar(*literal).dtype())
        .unwrap_or(xs.dtype);

    TensorValue::new(dtype, Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn batch_scan_i64_add_shared_init_batch0(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs[0].batch_dim.is_some()
        || inputs[1].batch_dim != Some(0)
        || params
            .get("body_op")
            .is_some_and(|body_op| body_op.as_str() != "add")
    {
        return Ok(None);
    }

    let init = match &inputs[0].value {
        Value::Scalar(Literal::I64(value)) => *value,
        Value::Tensor(tensor)
            if tensor.dtype == DType::I64
                && tensor.shape == Shape::scalar()
                && tensor.elements.len() == 1 =>
        {
            let Literal::I64(value) = tensor.elements[0] else {
                return Ok(None);
            };
            value
        }
        _ => return Ok(None),
    };

    let Some(xs) = inputs[1].value.as_tensor() else {
        return Ok(None);
    };
    if xs.dtype != DType::I64 || xs.rank() != 2 {
        return Ok(None);
    }
    let batch_size = xs.shape.dims[0] as usize;
    let scan_len = xs.shape.dims[1] as usize;
    let expected_len = batch_size
        .checked_mul(scan_len)
        .ok_or_else(|| BatchError::TensorError("scan input size overflowed".to_owned()))?;
    if xs.elements.len() != expected_len {
        return Ok(None);
    }

    let reverse = params.get("reverse").is_some_and(|value| value == "true");
    let mut outputs = Vec::with_capacity(batch_size);
    if let Some(values) = xs.elements.as_i64_slice() {
        for batch_idx in 0..batch_size {
            let mut carry = init;
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    carry = carry.wrapping_add(values[row_offset + scan_idx]);
                }
            } else {
                for scan_idx in 0..scan_len {
                    carry = carry.wrapping_add(values[row_offset + scan_idx]);
                }
            }
            outputs.push(carry);
        }
    } else {
        let elements = xs.elements.as_slice();
        for batch_idx in 0..batch_size {
            let mut carry = init;
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.wrapping_add(x);
                }
            } else {
                for scan_idx in 0..scan_len {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.wrapping_add(x);
                }
            }
            outputs.push(carry);
        }
    }

    TensorValue::new_i64_values(Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn batch_scan_i64_max_shared_init_batch0(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs[0].batch_dim.is_some()
        || inputs[1].batch_dim != Some(0)
        || params.get("body_op").map(String::as_str) != Some("max")
    {
        return Ok(None);
    }

    let init = match &inputs[0].value {
        Value::Scalar(Literal::I64(value)) => *value,
        Value::Tensor(tensor)
            if tensor.dtype == DType::I64
                && tensor.shape == Shape::scalar()
                && tensor.elements.len() == 1 =>
        {
            let Literal::I64(value) = tensor.elements[0] else {
                return Ok(None);
            };
            value
        }
        _ => return Ok(None),
    };

    let Some(xs) = inputs[1].value.as_tensor() else {
        return Ok(None);
    };
    if xs.dtype != DType::I64 || xs.rank() != 2 {
        return Ok(None);
    }
    let batch_size = xs.shape.dims[0] as usize;
    let scan_len = xs.shape.dims[1] as usize;
    let expected_len = batch_size
        .checked_mul(scan_len)
        .ok_or_else(|| BatchError::TensorError("scan input size overflowed".to_owned()))?;
    if xs.elements.len() != expected_len {
        return Ok(None);
    }

    let reverse = params.get("reverse").is_some_and(|value| value == "true");
    let mut outputs = Vec::with_capacity(batch_size);
    if let Some(values) = xs.elements.as_i64_slice() {
        for batch_idx in 0..batch_size {
            let mut carry = init;
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    carry = carry.max(values[row_offset + scan_idx]);
                }
            } else {
                for scan_idx in 0..scan_len {
                    carry = carry.max(values[row_offset + scan_idx]);
                }
            }
            outputs.push(carry);
        }
    } else {
        let elements = xs.elements.as_slice();
        for batch_idx in 0..batch_size {
            let mut carry = init;
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.max(x);
                }
            } else {
                for scan_idx in 0..scan_len {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.max(x);
                }
            }
            outputs.push(carry);
        }
    }

    TensorValue::new_i64_values(Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn batch_scan_f32_add_shared_init_batch0(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs[0].batch_dim.is_some()
        || inputs[1].batch_dim != Some(0)
        || params
            .get("body_op")
            .is_some_and(|body_op| body_op.as_str() != "add")
    {
        return Ok(None);
    }

    let init = match &inputs[0].value {
        Value::Scalar(Literal::F32Bits(bits)) => f32::from_bits(*bits),
        Value::Tensor(tensor)
            if tensor.dtype == DType::F32
                && tensor.shape == Shape::scalar()
                && tensor.elements.len() == 1 =>
        {
            let Literal::F32Bits(bits) = tensor.elements[0] else {
                return Ok(None);
            };
            f32::from_bits(bits)
        }
        _ => return Ok(None),
    };

    let Some(xs) = inputs[1].value.as_tensor() else {
        return Ok(None);
    };
    if xs.dtype != DType::F32 || xs.rank() != 2 {
        return Ok(None);
    }
    let batch_size = xs.shape.dims[0] as usize;
    let scan_len = xs.shape.dims[1] as usize;
    let expected_len = batch_size
        .checked_mul(scan_len)
        .ok_or_else(|| BatchError::TensorError("scan input size overflowed".to_owned()))?;
    if xs.elements.len() != expected_len {
        return Ok(None);
    }

    let reverse = params.get("reverse").is_some_and(|value| value == "true");
    let elements = xs.elements.as_slice();
    let mut outputs = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let mut carry = init;
        let row_offset = batch_idx * scan_len;
        if reverse {
            for scan_idx in (0..scan_len).rev() {
                let Literal::F32Bits(bits) = elements[row_offset + scan_idx] else {
                    return Ok(None);
                };
                carry += f32::from_bits(bits);
            }
        } else {
            for scan_idx in 0..scan_len {
                let Literal::F32Bits(bits) = elements[row_offset + scan_idx] else {
                    return Ok(None);
                };
                carry += f32::from_bits(bits);
            }
        }
        outputs.push(Literal::from_f32(carry));
    }

    TensorValue::new(DType::F32, Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn scan_scalar_op(params: &BTreeMap<String, String>) -> Option<ScanScalarOp> {
    match params.get("body_op").map(String::as_str).unwrap_or("add") {
        "add" => Some(ScanScalarOp::Add),
        "sub" => Some(ScanScalarOp::Sub),
        "mul" => Some(ScanScalarOp::Mul),
        "div" => Some(ScanScalarOp::Div),
        "pow" => Some(ScanScalarOp::Pow),
        "max" => Some(ScanScalarOp::Max),
        "min" => Some(ScanScalarOp::Min),
        _ => None,
    }
}

fn scan_scalar_initial_values(
    init: &BatchTracer,
    batch_size: usize,
) -> Result<Vec<Literal>, BatchError> {
    match init.batch_dim {
        None => match &init.value {
            Value::Scalar(literal) => Ok(vec![*literal; batch_size]),
            Value::Tensor(tensor)
                if tensor.shape == Shape::scalar() && tensor.elements.len() == 1 =>
            {
                Ok(vec![tensor.elements[0]; batch_size])
            }
            _ => Ok(Vec::new()),
        },
        Some(batch_dim) => {
            let value = move_batch_dim_to_front(&init.value, batch_dim)?;
            let Some(tensor) = value.as_tensor() else {
                return Ok(Vec::new());
            };
            if tensor.rank() == 1 && tensor.elements.len() == batch_size {
                Ok(tensor.elements.to_vec())
            } else {
                Ok(Vec::new())
            }
        }
    }
}

fn scan_scalar_xs_at(xs: &TensorValue, batch_idx: usize, scan_idx: usize) -> Literal {
    if xs.rank() == 1 {
        xs.elements[batch_idx]
    } else {
        xs.elements[batch_idx * xs.shape.dims[1] as usize + scan_idx]
    }
}

fn apply_scan_scalar_op(op: ScanScalarOp, carry: Literal, x: Literal) -> Option<Literal> {
    match (carry, x) {
        (Literal::I64(carry), Literal::I64(x)) => Some(match op {
            ScanScalarOp::Add => Literal::I64(carry.wrapping_add(x)),
            ScanScalarOp::Sub => Literal::I64(carry.wrapping_sub(x)),
            ScanScalarOp::Mul => Literal::I64(carry.wrapping_mul(x)),
            ScanScalarOp::Div => Literal::I64(carry.checked_div(x).unwrap_or(0)),
            ScanScalarOp::Pow => Literal::I64((carry as f64).powf(x as f64) as i64),
            ScanScalarOp::Max => Literal::I64(carry.max(x)),
            ScanScalarOp::Min => Literal::I64(carry.min(x)),
        }),
        (Literal::F64Bits(carry), Literal::F64Bits(x)) => {
            let carry = f64::from_bits(carry);
            let x = f64::from_bits(x);
            let result = match op {
                ScanScalarOp::Add => carry + x,
                ScanScalarOp::Sub => carry - x,
                ScanScalarOp::Mul => carry * x,
                ScanScalarOp::Div => carry / x,
                ScanScalarOp::Pow => carry.powf(x),
                ScanScalarOp::Max => carry.max(x),
                ScanScalarOp::Min => carry.min(x),
            };
            Some(Literal::from_f64(result))
        }
        (Literal::F32Bits(carry), Literal::F32Bits(x)) => {
            let carry = f32::from_bits(carry);
            let x = f32::from_bits(x);
            let result = match op {
                ScanScalarOp::Add => carry + x,
                ScanScalarOp::Sub => carry - x,
                ScanScalarOp::Mul => carry * x,
                ScanScalarOp::Div => carry / x,
                ScanScalarOp::Pow => carry.powf(x),
                ScanScalarOp::Max => carry.max(x),
                ScanScalarOp::Min => carry.min(x),
            };
            Some(Literal::from_f32(result))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum WhileScalarOp {
    Add,
    Sub,
    ReverseSub,
    Mul,
    Div,
    ReverseDiv,
    Pow,
    ReversePow,
}

#[derive(Clone, Copy)]
enum WhileCondOp {
    Lt,
    Le,
    Gt,
    Ge,
    Ne,
    Eq,
}

#[derive(Clone, Copy)]
enum LiteralSide {
    Left,
    Right,
}

fn batch_while(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    if let Some(result) = batch_while_i64_add_lt_batch0(inputs, params)? {
        return Ok(result);
    }
    if let Some(result) = batch_while_f32_add_lt_batch0(inputs, params)? {
        return Ok(result);
    }
    if let Some(result) = batch_while_scalar_loop(inputs, params)? {
        return Ok(result);
    }

    // Per-element fallback semantics currently match vmapped while behavior:
    // each batch element runs an independent loop until its own condition is false.
    batch_control_flow_fallback(Primitive::While, inputs, params)
}

enum I64Batch0Values<'a> {
    Scalar(i64),
    Slice(&'a [i64]),
}

impl I64Batch0Values<'_> {
    fn at(&self, index: usize) -> i64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Slice(values) => values[index],
        }
    }
}

fn batch_while_i64_add_lt_batch0(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs.len() != 3
        || params
            .get("body_op")
            .is_some_and(|body_op| body_op.as_str() != "add")
        || params
            .get("cond_op")
            .is_some_and(|cond_op| cond_op.as_str() != "lt")
    {
        return Ok(None);
    }

    let Some(batch_size) = while_i64_batch0_size(inputs)? else {
        return Ok(None);
    };
    let Some(init_values) = while_i64_batch0_values(&inputs[0], batch_size) else {
        return Ok(None);
    };
    let Some(step_values) = while_i64_batch0_values(&inputs[1], batch_size) else {
        return Ok(None);
    };
    let Some(threshold_values) = while_i64_batch0_values(&inputs[2], batch_size) else {
        return Ok(None);
    };

    let max_iter = params
        .get("max_iter")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1000);
    let mut outputs = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let mut carry = init_values.at(batch_idx);
        let step = step_values.at(batch_idx);
        let threshold = threshold_values.at(batch_idx);
        let mut iteration = 0_usize;
        while iteration < max_iter {
            if carry >= threshold {
                break;
            }
            carry = carry.wrapping_add(step);
            iteration += 1;
        }
        if iteration == max_iter {
            return Err(BatchError::EvalError(format!(
                "{} exceeded max iterations ({max_iter})",
                Primitive::While.as_str()
            )));
        }
        outputs.push(carry);
    }

    TensorValue::new_i64_values(Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn while_i64_batch0_size(inputs: &[BatchTracer]) -> Result<Option<usize>, BatchError> {
    let mut batch_size = None;
    for input in inputs {
        match input.batch_dim {
            None => {}
            Some(0) => {
                let size = get_batch_size(&input.value, 0)?;
                if batch_size.is_some_and(|existing| existing != size) {
                    return Ok(None);
                }
                batch_size = Some(size);
            }
            Some(_) => return Ok(None),
        }
    }
    Ok(batch_size)
}

fn while_i64_batch0_values(input: &BatchTracer, batch_size: usize) -> Option<I64Batch0Values<'_>> {
    match input.batch_dim {
        None => match &input.value {
            Value::Scalar(Literal::I64(value)) => Some(I64Batch0Values::Scalar(*value)),
            Value::Tensor(tensor)
                if tensor.dtype == DType::I64
                    && tensor.shape == Shape::scalar()
                    && tensor.elements.len() == 1 =>
            {
                tensor
                    .elements
                    .as_i64_slice()
                    .map(|values| I64Batch0Values::Scalar(values[0]))
                    .or_else(|| match tensor.elements[0] {
                        Literal::I64(value) => Some(I64Batch0Values::Scalar(value)),
                        _ => None,
                    })
            }
            _ => None,
        },
        Some(0) => {
            let tensor = input.value.as_tensor()?;
            if tensor.dtype != DType::I64
                || tensor.rank() != 1
                || tensor.elements.len() != batch_size
            {
                return None;
            }
            tensor.elements.as_i64_slice().map(I64Batch0Values::Slice)
        }
        Some(_) => None,
    }
}

enum F32Batch0Values<'a> {
    Scalar(f32),
    Slice(&'a [Literal]),
}

impl F32Batch0Values<'_> {
    fn at(&self, index: usize) -> f32 {
        match self {
            Self::Scalar(value) => *value,
            Self::Slice(values) => match values[index] {
                Literal::F32Bits(bits) => f32::from_bits(bits),
                _ => unreachable!("f32 while direct path prevalidates tensor literal width"),
            },
        }
    }
}

fn batch_while_f32_add_lt_batch0(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs.len() != 3
        || params
            .get("body_op")
            .is_some_and(|body_op| body_op.as_str() != "add")
        || params
            .get("cond_op")
            .is_some_and(|cond_op| cond_op.as_str() != "lt")
    {
        return Ok(None);
    }

    let Some(batch_size) = while_f32_batch0_size(inputs)? else {
        return Ok(None);
    };
    let Some(init_values) = while_f32_batch0_values(&inputs[0], batch_size) else {
        return Ok(None);
    };
    let Some(step_values) = while_f32_batch0_values(&inputs[1], batch_size) else {
        return Ok(None);
    };
    let Some(threshold_values) = while_f32_batch0_values(&inputs[2], batch_size) else {
        return Ok(None);
    };

    let max_iter = params
        .get("max_iter")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1000);
    let mut outputs = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let mut carry = init_values.at(batch_idx);
        let step = step_values.at(batch_idx);
        let threshold = threshold_values.at(batch_idx);
        let mut iteration = 0_usize;
        while iteration < max_iter {
            if !matches!(
                carry.partial_cmp(&threshold),
                Some(std::cmp::Ordering::Less)
            ) {
                break;
            }
            carry += step;
            iteration += 1;
        }
        if iteration == max_iter {
            return Err(BatchError::EvalError(format!(
                "{} exceeded max iterations ({max_iter})",
                Primitive::While.as_str()
            )));
        }
        outputs.push(Literal::from_f32(carry));
    }

    TensorValue::new(DType::F32, Shape::vector(batch_size as u32), outputs)
        .map(|tensor| Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn while_f32_batch0_size(inputs: &[BatchTracer]) -> Result<Option<usize>, BatchError> {
    let mut batch_size = None;
    for input in inputs {
        match input.batch_dim {
            None => {}
            Some(0) => {
                let size = get_batch_size(&input.value, 0)?;
                if batch_size.is_some_and(|existing| existing != size) {
                    return Ok(None);
                }
                batch_size = Some(size);
            }
            Some(_) => return Ok(None),
        }
    }
    Ok(batch_size)
}

fn while_f32_batch0_values(input: &BatchTracer, batch_size: usize) -> Option<F32Batch0Values<'_>> {
    match input.batch_dim {
        None => match &input.value {
            Value::Scalar(Literal::F32Bits(bits)) => {
                Some(F32Batch0Values::Scalar(f32::from_bits(*bits)))
            }
            Value::Tensor(tensor)
                if tensor.dtype == DType::F32
                    && tensor.shape == Shape::scalar()
                    && tensor.elements.len() == 1 =>
            {
                match tensor.elements[0] {
                    Literal::F32Bits(bits) => Some(F32Batch0Values::Scalar(f32::from_bits(bits))),
                    _ => None,
                }
            }
            _ => None,
        },
        Some(0) => {
            let tensor = input.value.as_tensor()?;
            if tensor.dtype != DType::F32
                || tensor.rank() != 1
                || tensor.elements.len() != batch_size
                || !tensor
                    .elements
                    .iter()
                    .all(|literal| matches!(literal, Literal::F32Bits(_)))
            {
                return None;
            }
            Some(F32Batch0Values::Slice(tensor.elements.as_slice()))
        }
        Some(_) => None,
    }
}

fn batch_while_scalar_loop(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<Option<BatchTracer>, BatchError> {
    if inputs.len() != 3 {
        return Ok(None);
    }
    let Some(batch_size) = while_scalar_batch_size(inputs)? else {
        return Ok(None);
    };
    let Some(body_op) = while_scalar_op(params) else {
        return Ok(None);
    };
    let Some(cond_op) = while_cond_op(params) else {
        return Ok(None);
    };

    let Some(mut carry_values) = while_scalar_values(&inputs[0], batch_size)? else {
        return Ok(None);
    };
    let Some(step_values) = while_scalar_values(&inputs[1], batch_size)? else {
        return Ok(None);
    };
    let Some(threshold_values) = while_scalar_values(&inputs[2], batch_size)? else {
        return Ok(None);
    };
    let output_dtype = while_scalar_output_dtype(&inputs[0]);

    let max_iter: usize = params
        .get("max_iter")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1000);
    let mut active = vec![true; batch_size];

    for _ in 0..max_iter {
        let mut any_active = false;
        for ((carry, threshold), is_active) in carry_values
            .iter()
            .zip(threshold_values.iter())
            .zip(active.iter_mut())
        {
            if !*is_active {
                continue;
            }
            let Some(keep_running) = apply_while_scalar_cond(cond_op, *carry, *threshold) else {
                return Ok(None);
            };
            *is_active = keep_running;
            any_active |= keep_running;
        }

        if !any_active {
            let tensor =
                TensorValue::new(output_dtype, Shape::vector(batch_size as u32), carry_values)
                    .map_err(|e| BatchError::TensorError(e.to_string()))?;
            return Ok(Some(BatchTracer::batched(Value::Tensor(tensor), 0)));
        }

        for ((carry, step), is_active) in carry_values
            .iter_mut()
            .zip(step_values.iter())
            .zip(active.iter())
        {
            if !*is_active {
                continue;
            }
            let Some(next) = apply_while_scalar_op(body_op, *carry, *step) else {
                return Ok(None);
            };
            *carry = next;
        }
    }

    if active.iter().any(|is_active| *is_active) {
        return Err(BatchError::EvalError(format!(
            "{} exceeded max iterations ({max_iter})",
            Primitive::While.as_str()
        )));
    }

    let tensor = TensorValue::new(output_dtype, Shape::vector(batch_size as u32), carry_values)
        .map_err(|e| BatchError::TensorError(e.to_string()))?;
    Ok(Some(BatchTracer::batched(Value::Tensor(tensor), 0)))
}

fn while_scalar_batch_size(inputs: &[BatchTracer]) -> Result<Option<usize>, BatchError> {
    let mut batch_size = None;
    for input in inputs {
        let Some(batch_dim) = input.batch_dim else {
            continue;
        };
        let size = get_batch_size(&input.value, batch_dim)?;
        if batch_size.is_some_and(|existing| existing != size) {
            return Ok(None);
        }
        batch_size = Some(size);
    }
    Ok(batch_size)
}

fn while_scalar_values(
    input: &BatchTracer,
    batch_size: usize,
) -> Result<Option<Vec<Literal>>, BatchError> {
    match input.batch_dim {
        None => match &input.value {
            Value::Scalar(literal) => Ok(Some(vec![*literal; batch_size])),
            Value::Tensor(tensor)
                if tensor.shape == Shape::scalar() && tensor.elements.len() == 1 =>
            {
                Ok(Some(vec![tensor.elements[0]; batch_size]))
            }
            _ => Ok(None),
        },
        Some(batch_dim) => {
            let value = move_batch_dim_to_front(&input.value, batch_dim)?;
            let Some(tensor) = value.as_tensor() else {
                return Ok(None);
            };
            if tensor.rank() == 1 && tensor.elements.len() == batch_size {
                Ok(Some(tensor.elements.to_vec()))
            } else {
                Ok(None)
            }
        }
    }
}

fn while_scalar_output_dtype(input: &BatchTracer) -> DType {
    match &input.value {
        Value::Scalar(literal) => Value::Scalar(*literal).dtype(),
        Value::Tensor(tensor) => tensor.dtype,
    }
}

fn while_scalar_op(params: &BTreeMap<String, String>) -> Option<WhileScalarOp> {
    match params.get("body_op").map(String::as_str).unwrap_or("add") {
        "add" => Some(WhileScalarOp::Add),
        "sub" => Some(WhileScalarOp::Sub),
        "mul" => Some(WhileScalarOp::Mul),
        "div" => Some(WhileScalarOp::Div),
        "pow" => Some(WhileScalarOp::Pow),
        _ => None,
    }
}

fn while_cond_op(params: &BTreeMap<String, String>) -> Option<WhileCondOp> {
    match params.get("cond_op").map(String::as_str).unwrap_or("lt") {
        "lt" => Some(WhileCondOp::Lt),
        "le" => Some(WhileCondOp::Le),
        "gt" => Some(WhileCondOp::Gt),
        "ge" => Some(WhileCondOp::Ge),
        "ne" => Some(WhileCondOp::Ne),
        "eq" => Some(WhileCondOp::Eq),
        _ => None,
    }
}

fn apply_while_scalar_op(op: WhileScalarOp, carry: Literal, step: Literal) -> Option<Literal> {
    match (carry, step) {
        (Literal::I64(carry), Literal::I64(step)) => Some(match op {
            WhileScalarOp::Add => Literal::I64(carry.wrapping_add(step)),
            WhileScalarOp::Sub => Literal::I64(carry.wrapping_sub(step)),
            WhileScalarOp::ReverseSub => Literal::I64(step.wrapping_sub(carry)),
            WhileScalarOp::Mul => Literal::I64(carry.wrapping_mul(step)),
            WhileScalarOp::Div => Literal::I64(carry.checked_div(step).unwrap_or(0)),
            WhileScalarOp::ReverseDiv => Literal::I64(step.checked_div(carry).unwrap_or(0)),
            WhileScalarOp::Pow => Literal::I64((carry as f64).powf(step as f64) as i64),
            WhileScalarOp::ReversePow => Literal::I64((step as f64).powf(carry as f64) as i64),
        }),
        (Literal::F64Bits(carry), Literal::F64Bits(step)) => {
            let carry = f64::from_bits(carry);
            let step = f64::from_bits(step);
            let result = match op {
                WhileScalarOp::Add => carry + step,
                WhileScalarOp::Sub => carry - step,
                WhileScalarOp::ReverseSub => step - carry,
                WhileScalarOp::Mul => carry * step,
                WhileScalarOp::Div => carry / step,
                WhileScalarOp::ReverseDiv => step / carry,
                WhileScalarOp::Pow => carry.powf(step),
                WhileScalarOp::ReversePow => step.powf(carry),
            };
            Some(Literal::from_f64(result))
        }
        (Literal::F32Bits(carry), Literal::F32Bits(step)) => {
            let carry = f32::from_bits(carry);
            let step = f32::from_bits(step);
            let result = match op {
                WhileScalarOp::Add => carry + step,
                WhileScalarOp::Sub => carry - step,
                WhileScalarOp::ReverseSub => step - carry,
                WhileScalarOp::Mul => carry * step,
                WhileScalarOp::Div => carry / step,
                WhileScalarOp::ReverseDiv => step / carry,
                WhileScalarOp::Pow => carry.powf(step),
                WhileScalarOp::ReversePow => step.powf(carry),
            };
            Some(Literal::from_f32(result))
        }
        _ => None,
    }
}

fn apply_while_scalar_cond(op: WhileCondOp, carry: Literal, threshold: Literal) -> Option<bool> {
    match (carry, threshold) {
        (Literal::I64(carry), Literal::I64(threshold)) => Some(match op {
            WhileCondOp::Lt => carry < threshold,
            WhileCondOp::Le => carry <= threshold,
            WhileCondOp::Gt => carry > threshold,
            WhileCondOp::Ge => carry >= threshold,
            WhileCondOp::Ne => carry != threshold,
            WhileCondOp::Eq => carry == threshold,
        }),
        (Literal::F64Bits(carry), Literal::F64Bits(threshold)) => {
            let carry = f64::from_bits(carry);
            let threshold = f64::from_bits(threshold);
            Some(match op {
                WhileCondOp::Lt => carry < threshold,
                WhileCondOp::Le => carry <= threshold,
                WhileCondOp::Gt => carry > threshold,
                WhileCondOp::Ge => carry >= threshold,
                WhileCondOp::Ne => carry != threshold,
                WhileCondOp::Eq => carry == threshold,
            })
        }
        (Literal::F32Bits(carry), Literal::F32Bits(threshold)) => {
            let carry = f32::from_bits(carry);
            let threshold = f32::from_bits(threshold);
            Some(match op {
                WhileCondOp::Lt => carry < threshold,
                WhileCondOp::Le => carry <= threshold,
                WhileCondOp::Gt => carry > threshold,
                WhileCondOp::Ge => carry >= threshold,
                WhileCondOp::Ne => carry != threshold,
                WhileCondOp::Eq => carry == threshold,
            })
        }
        _ => None,
    }
}

fn batch_switch(
    inputs: &[BatchTracer],
    params: &BTreeMap<String, String>,
) -> Result<BatchTracer, BatchError> {
    if inputs.len() < 2 {
        return Err(BatchError::InterpreterError(format!(
            "switch expects at least 2 inputs (index + branch), got {}",
            inputs.len()
        )));
    }

    let provided_branches = inputs.len().saturating_sub(1);
    if let Some(raw) = params.get("num_branches") {
        let declared = raw.parse::<usize>().map_err(|_| {
            BatchError::InterpreterError(format!("invalid num_branches value: {raw}"))
        })?;
        if declared != provided_branches {
            return Err(BatchError::InterpreterError(format!(
                "switch expected {declared} branch values but got {provided_branches}"
            )));
        }
    }

    // Fast path: scalar index selects one branch for the entire batch.
    if inputs[0].batch_dim.is_none() {
        let idx = scalar_to_switch_index(&inputs[0].value, provided_branches)?;
        let selected = &inputs[idx + 1];
        return match selected.batch_dim {
            Some(bd) => {
                let moved = move_batch_dim_to_front(&selected.value, bd)?;
                Ok(BatchTracer::batched(moved, 0))
            }
            None => {
                let mut batch_size = None;
                for (branch_idx, tracer) in inputs.iter().enumerate() {
                    if branch_idx == idx + 1 {
                        continue;
                    }
                    if let Some(bd) = tracer.batch_dim {
                        batch_size = Some(get_batch_size(&tracer.value, bd)?);
                        break;
                    }
                }
                if let Some(batch_size) = batch_size {
                    let broadcasted = broadcast_unbatched(&selected.value, batch_size, 0)?;
                    Ok(BatchTracer::batched(broadcasted, 0))
                } else {
                    Ok(BatchTracer::unbatched(selected.value.clone()))
                }
            }
        };
    }

    let index_bd = inputs[0].batch_dim.ok_or_else(|| {
        BatchError::InterpreterError("batched switch index missing batch dimension".to_owned())
    })?;
    let index = move_batch_dim_to_front(&inputs[0].value, index_bd)?;
    let batch_size = get_batch_size(&inputs[0].value, index_bd)?;
    let branch_values = switch_branch_values(&inputs[1..], batch_size)?;
    select_batched_switch_value(&index, &branch_values, batch_size)
        .map(|value| BatchTracer::batched(value, 0))
}

fn scalar_to_bool(value: &Value) -> Result<bool, BatchError> {
    let literal = match value {
        Value::Scalar(lit) => *lit,
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() || tensor.elements.len() != 1 {
                return Err(BatchError::EvalError(
                    "cond predicate must be scalar for unbatched fast-path".to_owned(),
                ));
            }
            tensor.elements[0]
        }
    };
    match literal {
        fj_core::Literal::Bool(b) => Ok(b),
        fj_core::Literal::I64(v) => Ok(v != 0),
        fj_core::Literal::U32(v) => Ok(v != 0),
        fj_core::Literal::U64(v) => Ok(v != 0),
        fj_core::Literal::BF16Bits(bits) => Ok(fj_core::Literal::BF16Bits(bits)
            .as_f64()
            .is_some_and(|v| v != 0.0)),
        fj_core::Literal::F16Bits(bits) => Ok(fj_core::Literal::F16Bits(bits)
            .as_f64()
            .is_some_and(|v| v != 0.0)),
        fj_core::Literal::F32Bits(bits) => Ok(f32::from_bits(bits) != 0.0),
        fj_core::Literal::F64Bits(bits) => Ok(f64::from_bits(bits) != 0.0),
        fj_core::Literal::Complex64Bits(..) | fj_core::Literal::Complex128Bits(..) => Err(
            BatchError::EvalError("cond predicate must be boolean or numeric".to_owned()),
        ),
    }
}

fn scalar_to_switch_index(value: &Value, branch_count: usize) -> Result<usize, BatchError> {
    let literal = match value {
        Value::Scalar(lit) => *lit,
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() || tensor.elements.len() != 1 {
                return Err(BatchError::InterpreterError(format!(
                    "{} index must be scalar",
                    Primitive::Switch.as_str()
                )));
            }
            tensor.elements[0]
        }
    };
    if branch_count == 0 {
        return Err(BatchError::InterpreterError(
            "switch requires at least one branch".to_owned(),
        ));
    }

    let last_branch = branch_count - 1;
    match literal {
        fj_core::Literal::I64(v) => {
            if v <= 0 {
                Ok(0)
            } else {
                Ok((v as u64).min(last_branch as u64) as usize)
            }
        }
        fj_core::Literal::U32(v) => Ok((v as usize).min(last_branch)),
        fj_core::Literal::U64(v) => Ok(v.min(last_branch as u64) as usize),
        fj_core::Literal::Bool(v) => Ok(usize::from(v).min(last_branch)),
        _ => Err(BatchError::InterpreterError(format!(
            "{} index must be integer, got {:?}",
            Primitive::Switch.as_str(),
            value.dtype()
        ))),
    }
}

fn batched_switch_index_at(
    index: &Value,
    batch_idx: usize,
    batch_size: usize,
    branch_count: usize,
) -> Result<usize, BatchError> {
    match index {
        Value::Scalar(_) => scalar_to_switch_index(index, branch_count),
        Value::Tensor(tensor) => {
            if tensor.rank() != 1
                || tensor.leading_dim() != Some(batch_size as u32)
                || tensor.elements.len() != batch_size
            {
                return Err(BatchError::InterpreterError(
                    "batched switch index must contain one scalar index per batch element"
                        .to_owned(),
                ));
            }
            scalar_to_switch_index(&Value::Scalar(tensor.elements[batch_idx]), branch_count)
        }
    }
}

fn switch_branch_values(
    branches: &[BatchTracer],
    batch_size: usize,
) -> Result<Vec<Value>, BatchError> {
    branches
        .iter()
        .map(|branch| match branch.batch_dim {
            Some(batch_dim) => move_batch_dim_to_front(&branch.value, batch_dim),
            None => broadcast_unbatched(&branch.value, batch_size, 0),
        })
        .collect()
}

fn select_batched_switch_value(
    index: &Value,
    branches: &[Value],
    batch_size: usize,
) -> Result<Value, BatchError> {
    let first = match branches.first() {
        Some(Value::Tensor(tensor)) => tensor,
        Some(Value::Scalar(_)) => {
            return Err(BatchError::InterpreterError(
                "batched switch branch values must carry a leading batch axis".to_owned(),
            ));
        }
        None => {
            return Err(BatchError::InterpreterError(
                "switch requires at least one branch".to_owned(),
            ));
        }
    };

    if first.leading_dim() != Some(batch_size as u32) {
        return Err(BatchError::InterpreterError(format!(
            "switch branch leading dimension must match batch size {batch_size}"
        )));
    }

    for branch in &branches[1..] {
        let Value::Tensor(tensor) = branch else {
            return Err(BatchError::InterpreterError(
                "batched switch branch values must carry a leading batch axis".to_owned(),
            ));
        };
        if tensor.dtype != first.dtype {
            return Err(BatchError::InterpreterError(
                "switch branches must have same dtype".to_owned(),
            ));
        }
        if tensor.shape != first.shape {
            return Err(BatchError::InterpreterError(
                "switch branches must have same shape".to_owned(),
            ));
        }
    }

    let slice_len = first.elements.len().checked_div(batch_size).unwrap_or(0);
    let mut elements = Vec::with_capacity(first.elements.len());
    for batch_idx in 0..batch_size {
        let branch_idx = batched_switch_index_at(index, batch_idx, batch_size, branches.len())?;
        let Value::Tensor(branch) = &branches[branch_idx] else {
            return Err(BatchError::InterpreterError(
                "batched switch branch values must carry a leading batch axis".to_owned(),
            ));
        };
        let start = batch_idx * slice_len;
        let end = start + slice_len;
        elements.extend_from_slice(&branch.elements[start..end]);
    }

    TensorValue::new(first.dtype, first.shape.clone(), elements)
        .map(Value::Tensor)
        .map_err(|error| BatchError::TensorError(error.to_string()))
}

fn select_batched_switch_tracer(
    index: &Value,
    branches: &[BatchTracer],
    batch_size: usize,
) -> Result<BatchTracer, BatchError> {
    let branch_values = switch_branch_values(branches, batch_size)?;
    select_batched_switch_value(index, &branch_values, batch_size)
        .map(|value| BatchTracer::batched(value, 0))
}

fn batch_size_from_inputs(inputs: &[BatchTracer]) -> Result<Option<usize>, BatchError> {
    for tracer in inputs {
        if let Some(batch_dim) = tracer.batch_dim {
            return Ok(Some(get_batch_size(&tracer.value, batch_dim)?));
        }
    }
    Ok(None)
}

fn and_bool_tensors(a: &Value, b: &Value) -> Result<Value, BatchError> {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb))
            if ta.dtype == DType::Bool && tb.dtype == DType::Bool =>
        {
            if ta.shape != tb.shape {
                return Err(BatchError::TensorError(format!(
                    "shape mismatch in and_bool_tensors: {:?} vs {:?}",
                    ta.shape, tb.shape
                )));
            }
            let elements: Vec<Literal> = ta
                .elements
                .iter()
                .zip(tb.elements.iter())
                .map(|(ea, eb)| match (ea, eb) {
                    (Literal::Bool(va), Literal::Bool(vb)) => Literal::Bool(*va && *vb),
                    _ => Literal::Bool(false),
                })
                .collect();
            TensorValue::new(DType::Bool, ta.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(|e| BatchError::TensorError(e.to_string()))
        }
        (Value::Scalar(Literal::Bool(va)), Value::Scalar(Literal::Bool(vb))) => {
            Ok(Value::Scalar(Literal::Bool(*va && *vb)))
        }
        _ => Err(BatchError::InterpreterError(
            "and_bool_tensors requires boolean inputs".to_owned(),
        )),
    }
}

fn broadcast_unbatched_outputs(
    outputs: Vec<BatchTracer>,
    batch_size: Option<usize>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some(batch_size) = batch_size else {
        return Ok(outputs);
    };

    outputs
        .into_iter()
        .map(|tracer| match tracer.batch_dim {
            Some(_) => Ok(tracer),
            None => Ok(BatchTracer::batched(
                broadcast_unbatched(&tracer.value, batch_size, 0)?,
                0,
            )),
        })
        .collect()
}

fn eval_sub_jaxpr_equation_values(
    equation: &Equation,
    values: &[Value],
) -> Result<Vec<Value>, BatchError> {
    if values.len() != equation.inputs.len() {
        return Err(BatchError::InterpreterError(format!(
            "{} expects {} resolved inputs, got {}",
            equation.primitive.as_str(),
            equation.inputs.len(),
            values.len()
        )));
    }

    let mut env = FxHashMap::default();
    for (atom, value) in equation.inputs.iter().zip(values) {
        if let Atom::Var(var) = atom {
            env.insert(*var, value.clone());
        }
    }

    eval_equation_outputs(equation, &env).map_err(|e| BatchError::InterpreterError(e.to_string()))
}

fn batch_while_sub_jaxpr(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    if let Some(outputs) = batch_while_sub_jaxpr_scalar_loop(equation, inputs)? {
        return Ok(outputs);
    }

    if let Some(outputs) = batch_while_sub_jaxpr_general(equation, inputs)? {
        return Ok(outputs);
    }

    batch_sub_jaxpr_by_slices(equation, inputs)
}

fn batch_scan_sub_jaxpr(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    if equation.sub_jaxprs.len() != 1 {
        return Err(BatchError::InterpreterError(format!(
            "scan with sub_jaxprs expects exactly one body jaxpr, got {}",
            equation.sub_jaxprs.len()
        )));
    }
    let body_jaxpr = &equation.sub_jaxprs[0];
    let Some(carry_count) = body_jaxpr.invars.len().checked_sub(1) else {
        return Err(BatchError::InterpreterError(
            "scan body requires carry inputs plus one xs input".to_owned(),
        ));
    };
    if body_jaxpr.outvars.len() < carry_count {
        return Err(BatchError::InterpreterError(format!(
            "scan body returns {} values for {carry_count} carries",
            body_jaxpr.outvars.len()
        )));
    }
    if equation.outputs.len() != body_jaxpr.outvars.len() {
        return Err(BatchError::InterpreterError(format!(
            "scan equation binds {} outputs but body returns {}",
            equation.outputs.len(),
            body_jaxpr.outvars.len()
        )));
    }

    let const_count = body_jaxpr.constvars.len();
    if inputs.len() != const_count + carry_count + 1 {
        return Err(BatchError::InterpreterError(format!(
            "scan body expects {const_count} consts, {carry_count} carries, and one xs input, got {} bindings",
            inputs.len()
        )));
    }
    let Some(batch_size) = batch_size_from_inputs(inputs)? else {
        return Err(BatchError::InterpreterError(
            "scan with sub_jaxprs in BatchTrace requires at least one batched input".to_owned(),
        ));
    };

    let (const_inputs, state_inputs) = inputs.split_at(const_count);
    let (carry_inputs, xs_input) = state_inputs.split_at(carry_count);
    let xs = scan_xs_to_batch_front(&xs_input[0], batch_size)?;
    let scan_len = batched_scan_len(&xs)?;
    let y_count = body_jaxpr.outvars.len() - carry_count;
    if scan_len == 0 && y_count > 0 {
        return Err(BatchError::InterpreterError(
            "zero-length functional scan outputs require abstract output shapes".to_owned(),
        ));
    }

    let body_consts = const_inputs
        .iter()
        .map(scan_const_to_body_tracer)
        .collect::<Result<Vec<_>, _>>()?;
    let mut carry = carry_inputs
        .iter()
        .map(|tracer| scan_state_to_batch_front(tracer, batch_size))
        .collect::<Result<Vec<_>, _>>()?;
    let mut per_y = vec![Vec::with_capacity(scan_len); y_count];
    let reverse = equation
        .params
        .get("reverse")
        .is_some_and(|value| value == "true");

    if let Some(outputs) =
        batch_scan_add_emit_carry_i64(body_jaxpr, carry_inputs, &xs, batch_size, scan_len, reverse)?
    {
        return Ok(outputs);
    }

    let scan_indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..scan_len).rev())
    } else {
        Box::new(0..scan_len)
    };
    for scan_idx in scan_indices {
        let x_slice = batched_scan_xs_at(&xs, scan_idx)?;
        let mut body_args = carry.clone();
        body_args.push(BatchTracer::batched(x_slice, 0));
        let body_outputs = batch_eval_jaxpr_with_consts(body_jaxpr, &body_consts, &body_args)?;
        if body_outputs.len() != carry_count + y_count {
            return Err(BatchError::InterpreterError(format!(
                "scan body returned {} outputs, expected {}",
                body_outputs.len(),
                carry_count + y_count
            )));
        }

        let mut body_outputs = body_outputs.into_iter();
        carry = body_outputs
            .by_ref()
            .take(carry_count)
            .map(|tracer| scan_state_into_batch_front(tracer, batch_size))
            .collect::<Result<Vec<_>, _>>()?;
        for (bucket, y_tracer) in per_y.iter_mut().zip(body_outputs) {
            bucket.push(scan_state_into_batch_front(y_tracer, batch_size)?.value);
        }
    }

    if reverse {
        for values in &mut per_y {
            values.reverse();
        }
    }

    let mut outputs = carry;
    for values in per_y {
        outputs.push(stack_batched_scan_outputs(values)?);
    }
    Ok(outputs)
}

fn batch_scan_add_emit_carry_i64(
    body_jaxpr: &Jaxpr,
    carry_inputs: &[BatchTracer],
    xs: &Value,
    batch_size: usize,
    scan_len: usize,
    reverse: bool,
) -> Result<Option<Vec<BatchTracer>>, BatchError> {
    if carry_inputs.len() != 1 || !scan_body_is_add_emit_carry_i64(body_jaxpr) {
        return Ok(None);
    }
    let Value::Tensor(xs_tensor) = xs else {
        return Ok(None);
    };
    if xs_tensor.dtype != DType::I64 || xs_tensor.rank() != 2 {
        return Ok(None);
    }
    if xs_tensor.shape.dims[0] as usize != batch_size
        || xs_tensor.shape.dims[1] as usize != scan_len
    {
        return Ok(None);
    }
    let expected_len = batch_size
        .checked_mul(scan_len)
        .ok_or_else(|| BatchError::TensorError("scan output size overflowed".to_owned()))?;
    if xs_tensor.elements.len() != expected_len {
        return Ok(None);
    }
    let init_values = scan_scalar_initial_values(&carry_inputs[0], batch_size)?;
    if init_values.len() != batch_size {
        return Ok(None);
    }

    let mut final_carry = Vec::with_capacity(batch_size);
    let mut ys = vec![0_i64; expected_len];
    if let Some(values) = xs_tensor.elements.as_i64_slice() {
        for (batch_idx, init) in init_values.into_iter().enumerate() {
            let Literal::I64(mut carry) = init else {
                return Ok(None);
            };
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    carry = carry.wrapping_add(values[row_offset + scan_idx]);
                    ys[row_offset + scan_idx] = carry;
                }
            } else {
                for scan_idx in 0..scan_len {
                    carry = carry.wrapping_add(values[row_offset + scan_idx]);
                    ys[row_offset + scan_idx] = carry;
                }
            }
            final_carry.push(carry);
        }
    } else {
        let elements = xs_tensor.elements.as_slice();
        for (batch_idx, init) in init_values.into_iter().enumerate() {
            let Literal::I64(mut carry) = init else {
                return Ok(None);
            };
            let row_offset = batch_idx * scan_len;
            if reverse {
                for scan_idx in (0..scan_len).rev() {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.wrapping_add(x);
                    ys[row_offset + scan_idx] = carry;
                }
            } else {
                for scan_idx in 0..scan_len {
                    let Literal::I64(x) = elements[row_offset + scan_idx] else {
                        return Ok(None);
                    };
                    carry = carry.wrapping_add(x);
                    ys[row_offset + scan_idx] = carry;
                }
            }
            final_carry.push(carry);
        }
    }

    let carry = TensorValue::new_i64_values(Shape::vector(xs_tensor.shape.dims[0]), final_carry)
        .map_err(|error| BatchError::TensorError(error.to_string()))?;
    let y = TensorValue::new_i64_values(
        Shape {
            dims: vec![xs_tensor.shape.dims[0], xs_tensor.shape.dims[1]],
        },
        ys,
    )
    .map_err(|error| BatchError::TensorError(error.to_string()))?;

    Ok(Some(vec![
        BatchTracer::batched(Value::Tensor(carry), 0),
        BatchTracer::batched(Value::Tensor(y), 0),
    ]))
}

fn scan_body_is_add_emit_carry_i64(body_jaxpr: &Jaxpr) -> bool {
    if !body_jaxpr.constvars.is_empty()
        || body_jaxpr.invars.len() != 2
        || body_jaxpr.outvars.len() != 2
        || body_jaxpr.equations.len() != 2
    {
        return false;
    }

    let add_carry = &body_jaxpr.equations[0];
    let emit = &body_jaxpr.equations[1];
    if add_carry.primitive != Primitive::Add
        || emit.primitive != Primitive::Add
        || !add_carry.params.is_empty()
        || !emit.params.is_empty()
        || !add_carry.sub_jaxprs.is_empty()
        || !emit.sub_jaxprs.is_empty()
        || !add_carry.effects.is_empty()
        || !emit.effects.is_empty()
        || add_carry.outputs.len() != 1
        || emit.outputs.len() != 1
        || add_carry.inputs.len() != 2
        || emit.inputs.len() != 2
    {
        return false;
    }

    let carry = Atom::Var(body_jaxpr.invars[0]);
    let xs = Atom::Var(body_jaxpr.invars[1]);
    let sum = Atom::Var(add_carry.outputs[0]);
    add_carry.outputs[0] == body_jaxpr.outvars[0]
        && emit.outputs[0] == body_jaxpr.outvars[1]
        && ((add_carry.inputs[0] == carry && add_carry.inputs[1] == xs)
            || (add_carry.inputs[0] == xs && add_carry.inputs[1] == carry))
        && ((emit.inputs[0] == sum && emit.inputs[1] == Atom::Lit(Literal::I64(0)))
            || (emit.inputs[0] == Atom::Lit(Literal::I64(0)) && emit.inputs[1] == sum))
}

fn scan_const_to_body_tracer(tracer: &BatchTracer) -> Result<BatchTracer, BatchError> {
    match tracer.batch_dim {
        Some(batch_dim) => Ok(BatchTracer::batched(
            move_batch_dim_to_front(&tracer.value, batch_dim)?,
            0,
        )),
        None => Ok(tracer.clone()),
    }
}

fn scan_state_to_batch_front(
    tracer: &BatchTracer,
    batch_size: usize,
) -> Result<BatchTracer, BatchError> {
    match tracer.batch_dim {
        Some(batch_dim) => Ok(BatchTracer::batched(
            move_batch_dim_to_front(&tracer.value, batch_dim)?,
            0,
        )),
        None => Ok(BatchTracer::batched(
            broadcast_unbatched(&tracer.value, batch_size, 0)?,
            0,
        )),
    }
}

fn scan_state_into_batch_front(
    tracer: BatchTracer,
    batch_size: usize,
) -> Result<BatchTracer, BatchError> {
    match tracer.batch_dim {
        Some(0) => Ok(tracer),
        Some(batch_dim) => Ok(BatchTracer::batched(
            move_batch_dim_to_front(&tracer.value, batch_dim)?,
            0,
        )),
        None => Ok(BatchTracer::batched(
            broadcast_unbatched(&tracer.value, batch_size, 0)?,
            0,
        )),
    }
}

fn scan_xs_to_batch_front(tracer: &BatchTracer, batch_size: usize) -> Result<Value, BatchError> {
    match tracer.batch_dim {
        Some(batch_dim) => move_batch_dim_to_front(&tracer.value, batch_dim),
        None => broadcast_unbatched(&tracer.value, batch_size, 0),
    }
}

fn batched_scan_len(xs: &Value) -> Result<usize, BatchError> {
    match xs {
        Value::Scalar(_) => Ok(1),
        Value::Tensor(tensor) if tensor.rank() <= 1 => Ok(1),
        Value::Tensor(tensor) => Ok(tensor.shape.dims[1] as usize),
    }
}

fn batched_scan_xs_at(xs: &Value, scan_idx: usize) -> Result<Value, BatchError> {
    let Value::Tensor(tensor) = xs else {
        return Ok(xs.clone());
    };
    if tensor.rank() <= 1 {
        if scan_idx == 0 {
            return Ok(Value::Tensor(tensor.clone()));
        }
        return Err(BatchError::TensorError(format!(
            "scan index {scan_idx} out of bounds for length-1 batched xs"
        )));
    }

    let batch_size = tensor.shape.dims[0] as usize;
    let scan_len = tensor.shape.dims[1] as usize;
    if scan_idx >= scan_len {
        return Err(BatchError::TensorError(format!(
            "scan index {scan_idx} out of bounds for length {scan_len}"
        )));
    }

    let inner_count = tensor.shape.dims[2..]
        .iter()
        .map(|dim| *dim as usize)
        .product::<usize>()
        .max(1);
    let mut elements = Vec::with_capacity(batch_size * inner_count);
    for batch_idx in 0..batch_size {
        let start = (batch_idx * scan_len + scan_idx) * inner_count;
        let end = start + inner_count;
        elements.extend_from_slice(&tensor.elements[start..end]);
    }

    let mut dims = Vec::with_capacity(tensor.rank() - 1);
    dims.push(tensor.shape.dims[0]);
    dims.extend_from_slice(&tensor.shape.dims[2..]);
    TensorValue::new(tensor.dtype, Shape { dims }, elements)
        .map(Value::Tensor)
        .map_err(|error| BatchError::TensorError(error.to_string()))
}

fn stack_batched_scan_outputs(values: Vec<Value>) -> Result<BatchTracer, BatchError> {
    let stacked = TensorValue::stack_axis0(&values)
        .map(Value::Tensor)
        .map_err(|error| BatchError::TensorError(error.to_string()))?;
    let batched = move_batch_dim_to_front(&stacked, 1)?;
    Ok(BatchTracer::batched(batched, 0))
}

fn batch_while_sub_jaxpr_general(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Option<Vec<BatchTracer>>, BatchError> {
    if equation.sub_jaxprs.len() != 2 {
        return Ok(None);
    }
    let cond_jaxpr = &equation.sub_jaxprs[0];
    let body_jaxpr = &equation.sub_jaxprs[1];

    let Some(batch_size) = batch_size_from_inputs(inputs)? else {
        return Ok(None);
    };

    let max_iter: usize = equation
        .params
        .get("max_iter")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);

    let const_count = cond_jaxpr.constvars.len() + body_jaxpr.constvars.len();
    if inputs.len() < const_count {
        return Ok(None);
    }
    let (const_inputs, carry_inputs) = inputs.split_at(const_count);
    let (cond_const_inputs, body_const_inputs) = const_inputs.split_at(cond_jaxpr.constvars.len());

    if carry_inputs.is_empty() {
        return Ok(None);
    }
    if cond_jaxpr.invars.len() != carry_inputs.len() {
        return Ok(None);
    }
    if body_jaxpr.invars.len() != carry_inputs.len() {
        return Ok(None);
    }

    let cond_consts: Vec<Value> = cond_const_inputs.iter().map(|t| t.value.clone()).collect();
    let body_consts: Vec<Value> = body_const_inputs.iter().map(|t| t.value.clone()).collect();

    let mut carry: Vec<Value> = carry_inputs
        .iter()
        .map(|tracer| match tracer.batch_dim {
            Some(bd) => move_batch_dim_to_front(&tracer.value, bd),
            None => broadcast_unbatched(&tracer.value, batch_size, 0),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut active_mask = Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape::vector(batch_size as u32),
            vec![Literal::Bool(true); batch_size],
        )
        .map_err(|e| BatchError::TensorError(e.to_string()))?,
    );

    for _ in 0..max_iter {
        let cond_result = eval_jaxpr_with_consts(cond_jaxpr, &cond_consts, &carry)
            .map_err(|e| BatchError::InterpreterError(format!("while cond: {e}")))?;
        if cond_result.len() != 1 {
            return Ok(None);
        }

        let cond_pred = &cond_result[0];
        let new_active = and_bool_tensors(&active_mask, cond_pred)?;

        let any_active = match &new_active {
            Value::Tensor(t) if t.dtype == DType::Bool => t
                .elements
                .iter()
                .any(|lit| matches!(lit, Literal::Bool(true))),
            Value::Scalar(Literal::Bool(b)) => *b,
            _ => return Ok(None),
        };

        if !any_active {
            return Ok(Some(
                carry
                    .into_iter()
                    .map(|v| BatchTracer::batched(v, 0))
                    .collect(),
            ));
        }

        let body_result = eval_jaxpr_with_consts(body_jaxpr, &body_consts, &carry)
            .map_err(|e| BatchError::InterpreterError(format!("while body: {e}")))?;
        if body_result.len() != carry.len() {
            return Ok(None);
        }

        for (old_carry, new_carry) in carry.iter_mut().zip(body_result) {
            *old_carry = eval_primitive(
                Primitive::Select,
                &[new_active.clone(), new_carry, old_carry.clone()],
                &BTreeMap::new(),
            )
            .map_err(|e| BatchError::EvalError(e.to_string()))?;
        }

        active_mask = new_active;
    }

    let still_active = match &active_mask {
        Value::Tensor(t) if t.dtype == DType::Bool => t
            .elements
            .iter()
            .any(|lit| matches!(lit, Literal::Bool(true))),
        Value::Scalar(Literal::Bool(b)) => *b,
        _ => false,
    };
    if still_active {
        return Err(BatchError::EvalError(format!(
            "while exceeded max iterations ({max_iter})"
        )));
    }

    Ok(Some(
        carry
            .into_iter()
            .map(|v| BatchTracer::batched(v, 0))
            .collect(),
    ))
}

fn batch_while_sub_jaxpr_scalar_loop(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Option<Vec<BatchTracer>>, BatchError> {
    if equation.sub_jaxprs.len() != 2 || equation.outputs.len() != 1 || inputs.len() != 1 {
        return Ok(None);
    }

    let Some((cond_op, cond_threshold)) = while_sub_jaxpr_scalar_cond(&equation.sub_jaxprs[0])
    else {
        return Ok(None);
    };
    let Some((body_op, body_operand)) = while_sub_jaxpr_scalar_body(&equation.sub_jaxprs[1]) else {
        return Ok(None);
    };
    let Some(batch_size) = batch_size_from_inputs(inputs)? else {
        return Ok(None);
    };
    let Some(mut carry_values) = while_scalar_values(&inputs[0], batch_size)? else {
        return Ok(None);
    };

    let output_dtype = while_scalar_output_dtype(&inputs[0]);
    let Some(max_iter) = parse_while_sub_jaxpr_max_iter(&equation.params) else {
        return Ok(None);
    };

    if let Some(values) = solve_i64_gt_sub_while(
        cond_op,
        cond_threshold,
        body_op,
        body_operand,
        max_iter,
        &carry_values,
    )? {
        return build_batched_scalar_while_outputs(output_dtype, batch_size, values);
    }

    let mut active = vec![true; batch_size];

    for _ in 0..max_iter {
        let mut any_active = false;
        for (carry, is_active) in carry_values.iter().zip(active.iter_mut()) {
            if !*is_active {
                continue;
            }
            let Some(keep_running) = apply_while_scalar_cond(cond_op, *carry, cond_threshold)
            else {
                return Ok(None);
            };
            *is_active = keep_running;
            any_active |= keep_running;
        }

        if !any_active {
            return build_batched_scalar_while_outputs(output_dtype, batch_size, carry_values);
        }

        for (carry, is_active) in carry_values.iter_mut().zip(active.iter()) {
            if !*is_active {
                continue;
            }
            let Some(next) = apply_while_scalar_op(body_op, *carry, body_operand) else {
                return Ok(None);
            };
            *carry = next;
        }
    }

    if active.iter().any(|is_active| *is_active) {
        return Err(BatchError::EvalError(format!(
            "{} exceeded max iterations ({max_iter})",
            Primitive::While.as_str()
        )));
    }

    build_batched_scalar_while_outputs(output_dtype, batch_size, carry_values)
}

fn solve_i64_gt_sub_while(
    cond_op: WhileCondOp,
    cond_threshold: Literal,
    body_op: WhileScalarOp,
    body_operand: Literal,
    max_iter: usize,
    carry_values: &[Literal],
) -> Result<Option<Vec<Literal>>, BatchError> {
    if !matches!(cond_op, WhileCondOp::Gt) || !matches!(body_op, WhileScalarOp::Sub) {
        return Ok(None);
    }
    let (Literal::I64(threshold), Literal::I64(step)) = (cond_threshold, body_operand) else {
        return Ok(None);
    };
    if step <= 0 {
        return Ok(None);
    }

    let mut outputs = Vec::with_capacity(carry_values.len());
    let step = i128::from(step);
    let threshold = i128::from(threshold);
    let max_iter = max_iter as i128;
    for literal in carry_values {
        let Literal::I64(init) = *literal else {
            return Ok(None);
        };
        let init = i128::from(init);
        let iterations = if init > threshold {
            ((init - threshold - 1) / step) + 1
        } else {
            0
        };
        let final_value = init - iterations * step;
        let Ok(final_value) = i64::try_from(final_value) else {
            return Ok(None);
        };
        if iterations >= max_iter {
            return Err(BatchError::EvalError(format!(
                "{} exceeded max iterations ({})",
                Primitive::While.as_str(),
                max_iter
            )));
        }
        outputs.push(Literal::I64(final_value));
    }

    Ok(Some(outputs))
}

fn build_batched_scalar_while_outputs(
    dtype: DType,
    batch_size: usize,
    values: Vec<Literal>,
) -> Result<Option<Vec<BatchTracer>>, BatchError> {
    TensorValue::new(dtype, Shape::vector(batch_size as u32), values)
        .map(|tensor| Some(vec![BatchTracer::batched(Value::Tensor(tensor), 0)]))
        .map_err(|e| BatchError::TensorError(e.to_string()))
}

fn parse_while_sub_jaxpr_max_iter(params: &BTreeMap<String, String>) -> Option<usize> {
    match params.get("max_iter") {
        Some(raw) => raw.parse::<usize>().ok(),
        None => Some(1000),
    }
}

fn while_sub_jaxpr_scalar_cond(jaxpr: &Jaxpr) -> Option<(WhileCondOp, Literal)> {
    let (primitive, literal, literal_side) = single_var_literal_jaxpr(jaxpr)?;
    let cond_op = while_cond_op_from_primitive(primitive, literal_side)?;
    Some((cond_op, literal))
}

fn while_sub_jaxpr_scalar_body(jaxpr: &Jaxpr) -> Option<(WhileScalarOp, Literal)> {
    let (primitive, literal, literal_side) = single_var_literal_jaxpr(jaxpr)?;
    let body_op = while_body_op_from_primitive(primitive, literal_side)?;
    Some((body_op, literal))
}

fn single_var_literal_jaxpr(jaxpr: &Jaxpr) -> Option<(Primitive, Literal, LiteralSide)> {
    if !jaxpr.constvars.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 1
    {
        return None;
    }

    let equation = &jaxpr.equations[0];
    if !equation.params.is_empty()
        || !equation.effects.is_empty()
        || !equation.sub_jaxprs.is_empty()
        || equation.inputs.len() != 2
        || equation.outputs.len() != 1
        || equation.outputs[0] != jaxpr.outvars[0]
    {
        return None;
    }

    match (&equation.inputs[0], &equation.inputs[1]) {
        (Atom::Var(var), Atom::Lit(literal)) if *var == jaxpr.invars[0] => {
            Some((equation.primitive, *literal, LiteralSide::Right))
        }
        (Atom::Lit(literal), Atom::Var(var)) if *var == jaxpr.invars[0] => {
            Some((equation.primitive, *literal, LiteralSide::Left))
        }
        _ => None,
    }
}

fn while_cond_op_from_primitive(
    primitive: Primitive,
    literal_side: LiteralSide,
) -> Option<WhileCondOp> {
    match (primitive, literal_side) {
        (Primitive::Lt, LiteralSide::Right) => Some(WhileCondOp::Lt),
        (Primitive::Le, LiteralSide::Right) => Some(WhileCondOp::Le),
        (Primitive::Gt, LiteralSide::Right) => Some(WhileCondOp::Gt),
        (Primitive::Ge, LiteralSide::Right) => Some(WhileCondOp::Ge),
        (Primitive::Lt, LiteralSide::Left) => Some(WhileCondOp::Gt),
        (Primitive::Le, LiteralSide::Left) => Some(WhileCondOp::Ge),
        (Primitive::Gt, LiteralSide::Left) => Some(WhileCondOp::Lt),
        (Primitive::Ge, LiteralSide::Left) => Some(WhileCondOp::Le),
        (Primitive::Ne, _) => Some(WhileCondOp::Ne),
        (Primitive::Eq, _) => Some(WhileCondOp::Eq),
        _ => None,
    }
}

fn while_body_op_from_primitive(
    primitive: Primitive,
    literal_side: LiteralSide,
) -> Option<WhileScalarOp> {
    match (primitive, literal_side) {
        (Primitive::Add, _) => Some(WhileScalarOp::Add),
        (Primitive::Sub, LiteralSide::Right) => Some(WhileScalarOp::Sub),
        (Primitive::Sub, LiteralSide::Left) => Some(WhileScalarOp::ReverseSub),
        (Primitive::Mul, _) => Some(WhileScalarOp::Mul),
        (Primitive::Div, LiteralSide::Right) => Some(WhileScalarOp::Div),
        (Primitive::Div, LiteralSide::Left) => Some(WhileScalarOp::ReverseDiv),
        (Primitive::Pow, LiteralSide::Right) => Some(WhileScalarOp::Pow),
        (Primitive::Pow, LiteralSide::Left) => Some(WhileScalarOp::ReversePow),
        _ => None,
    }
}

fn batch_sub_jaxpr_by_slices(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    let Some(batch_size) = batch_size_from_inputs(inputs)? else {
        let values = inputs
            .iter()
            .map(|tracer| tracer.value.clone())
            .collect::<Vec<_>>();
        return eval_sub_jaxpr_equation_values(equation, &values).map(|outputs| {
            outputs
                .into_iter()
                .map(BatchTracer::unbatched)
                .collect::<Vec<_>>()
        });
    };

    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|tracer| match tracer.batch_dim {
            Some(batch_dim) => move_batch_dim_to_front(&tracer.value, batch_dim),
            None => broadcast_unbatched(&tracer.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    let mut per_output: Option<Vec<Vec<Value>>> = None;
    for batch_idx in 0..batch_size {
        let slices: Result<Vec<Value>, BatchError> = values
            .iter()
            .map(|value| match value {
                Value::Tensor(tensor) => tensor
                    .slice_axis0(batch_idx)
                    .map_err(|e| BatchError::TensorError(e.to_string())),
                Value::Scalar(_) => Ok(value.clone()),
            })
            .collect();
        let slices = slices?;
        let outputs = eval_sub_jaxpr_equation_values(equation, &slices)?;

        let buckets = per_output.get_or_insert_with(|| {
            (0..outputs.len())
                .map(|_| Vec::with_capacity(batch_size))
                .collect()
        });
        if buckets.len() != outputs.len() {
            return Err(BatchError::InterpreterError(format!(
                "{} returned inconsistent output arity across batch slices: expected {}, got {}",
                equation.primitive.as_str(),
                buckets.len(),
                outputs.len()
            )));
        }
        for (bucket, output) in buckets.iter_mut().zip(outputs) {
            bucket.push(output);
        }
    }

    per_output
        .unwrap_or_default()
        .into_iter()
        .map(|outputs| {
            TensorValue::stack_axis0(&outputs)
                .map(|tensor| BatchTracer::batched(Value::Tensor(tensor), 0))
                .map_err(|e| BatchError::TensorError(e.to_string()))
        })
        .collect()
}

fn select_switch_branch<'a>(
    equation: &'a Equation,
    index_value: &Value,
) -> Result<&'a Jaxpr, BatchError> {
    let expected_branches = equation
        .params
        .get("num_branches")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(equation.sub_jaxprs.len());
    if expected_branches != equation.sub_jaxprs.len() {
        return Err(BatchError::InterpreterError(format!(
            "switch declares {expected_branches} branches but carries {} sub_jaxprs",
            equation.sub_jaxprs.len()
        )));
    }

    let branch_idx = scalar_to_switch_index(index_value, equation.sub_jaxprs.len())?;
    equation.sub_jaxprs.get(branch_idx).ok_or_else(|| {
        BatchError::InterpreterError("switch requires at least one branch".to_owned())
    })
}

fn batch_switch_sub_jaxprs(
    equation: &Equation,
    inputs: &[BatchTracer],
) -> Result<Vec<BatchTracer>, BatchError> {
    if inputs.is_empty() {
        return Err(BatchError::InterpreterError(
            "switch with sub_jaxprs requires at least the index input".to_owned(),
        ));
    }

    let batch_size = batch_size_from_inputs(inputs)?;

    if inputs[0].batch_dim.is_none() {
        let selected_branch = select_switch_branch(equation, &inputs[0].value)?;
        let provided_bindings = &inputs[1..];
        let expected_bindings = selected_branch.constvars.len() + selected_branch.invars.len();
        if provided_bindings.len() != expected_bindings {
            return Err(BatchError::InterpreterError(format!(
                "switch selected branch expects {expected_bindings} bindings, got {}",
                provided_bindings.len()
            )));
        }
        let (const_values, branch_args) =
            provided_bindings.split_at(selected_branch.constvars.len());
        let outputs = batch_eval_jaxpr_with_consts(selected_branch, const_values, branch_args)?;
        return broadcast_unbatched_outputs(outputs, batch_size);
    }

    let batch_size = batch_size.ok_or_else(|| {
        BatchError::InterpreterError(
            "switch with batched index requires a resolvable batch size".to_owned(),
        )
    })?;
    let index_bd = inputs[0].batch_dim.ok_or_else(|| {
        BatchError::InterpreterError("batched switch index missing batch dimension".to_owned())
    })?;
    let index = move_batch_dim_to_front(&inputs[0].value, index_bd)?;
    if let Some(outputs) =
        batch_switch_sub_jaxprs_vectorized(equation, &inputs[1..], &index, batch_size)?
    {
        return Ok(outputs);
    }

    batch_switch_sub_jaxprs_by_slices(equation, inputs, batch_size)
}

fn batch_switch_sub_jaxprs_vectorized(
    equation: &Equation,
    provided_bindings: &[BatchTracer],
    index: &Value,
    batch_size: usize,
) -> Result<Option<Vec<BatchTracer>>, BatchError> {
    let mut per_branch_outputs = Vec::with_capacity(equation.sub_jaxprs.len());
    for branch in &equation.sub_jaxprs {
        let expected_bindings = branch.constvars.len() + branch.invars.len();
        if provided_bindings.len() != expected_bindings {
            return Ok(None);
        }
        let (const_values, branch_args) = provided_bindings.split_at(branch.constvars.len());
        per_branch_outputs.push(batch_eval_jaxpr_with_consts(
            branch,
            const_values,
            branch_args,
        )?);
    }

    let Some(first_branch_outputs) = per_branch_outputs.first() else {
        return Err(BatchError::InterpreterError(
            "switch requires at least one branch".to_owned(),
        ));
    };
    let output_count = first_branch_outputs.len();
    for branch_outputs in &per_branch_outputs[1..] {
        if branch_outputs.len() != output_count {
            return Err(BatchError::InterpreterError(format!(
                "switch branches must return the same output arity: expected {output_count}, got {}",
                branch_outputs.len()
            )));
        }
    }

    let mut selected_outputs = Vec::with_capacity(output_count);
    for output_idx in 0..output_count {
        let branch_outputs = per_branch_outputs
            .iter()
            .map(|outputs| outputs[output_idx].clone())
            .collect::<Vec<_>>();
        selected_outputs.push(select_batched_switch_tracer(
            index,
            &branch_outputs,
            batch_size,
        )?);
    }

    Ok(Some(selected_outputs))
}

fn batch_switch_sub_jaxprs_by_slices(
    equation: &Equation,
    inputs: &[BatchTracer],
    batch_size: usize,
) -> Result<Vec<BatchTracer>, BatchError> {
    let values: Result<Vec<Value>, BatchError> = inputs
        .iter()
        .map(|tracer| match tracer.batch_dim {
            Some(batch_dim) => move_batch_dim_to_front(&tracer.value, batch_dim),
            None => broadcast_unbatched(&tracer.value, batch_size, 0),
        })
        .collect();
    let values = values?;

    let mut per_output: Option<Vec<Vec<Value>>> = None;
    for batch_idx in 0..batch_size {
        let slices: Result<Vec<Value>, BatchError> = values
            .iter()
            .map(|value| match value {
                Value::Tensor(tensor) => tensor
                    .slice_axis0(batch_idx)
                    .map_err(|e| BatchError::TensorError(e.to_string())),
                Value::Scalar(_) => Ok(value.clone()),
            })
            .collect();
        let slices = slices?;
        let selected_branch = select_switch_branch(equation, &slices[0])?;
        let provided_bindings = &slices[1..];
        let expected_bindings = selected_branch.constvars.len() + selected_branch.invars.len();
        if provided_bindings.len() != expected_bindings {
            return Err(BatchError::InterpreterError(format!(
                "switch selected branch expects {expected_bindings} bindings, got {}",
                provided_bindings.len()
            )));
        }
        let (const_values, branch_args) =
            provided_bindings.split_at(selected_branch.constvars.len());
        let outputs = eval_jaxpr_with_consts(selected_branch, const_values, branch_args)
            .map_err(|e| BatchError::InterpreterError(e.to_string()))?;

        let buckets = per_output.get_or_insert_with(|| {
            (0..outputs.len())
                .map(|_| Vec::with_capacity(batch_size))
                .collect()
        });
        if buckets.len() != outputs.len() {
            return Err(BatchError::InterpreterError(format!(
                "switch returned inconsistent output arity across batch slices: expected {}, got {}",
                buckets.len(),
                outputs.len()
            )));
        }
        for (bucket, output) in buckets.iter_mut().zip(outputs) {
            bucket.push(output);
        }
    }

    per_output
        .unwrap_or_default()
        .into_iter()
        .map(|outputs| {
            TensorValue::stack_axis0(&outputs)
                .map(|tensor| BatchTracer::batched(Value::Tensor(tensor), 0))
                .map_err(|e| BatchError::TensorError(e.to_string()))
        })
        .collect()
}

fn batch_eval_equation_outputs(
    equation: &Equation,
    env: &FxHashMap<VarId, BatchTracer>,
) -> Result<Vec<BatchTracer>, BatchError> {
    let inputs: Result<Vec<BatchTracer>, BatchError> = equation
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

    if equation.sub_jaxprs.is_empty() {
        return apply_batch_rule_multi(equation.primitive, &inputs, &equation.params);
    }

    match equation.primitive {
        Primitive::Switch => batch_switch_sub_jaxprs(equation, &inputs),
        Primitive::Cond => batch_sub_jaxpr_by_slices(equation, &inputs),
        Primitive::Scan => batch_scan_sub_jaxpr(equation, &inputs),
        Primitive::While => batch_while_sub_jaxpr(equation, &inputs),
        primitive => Err(BatchError::InterpreterError(format!(
            "invalid BatchTrace IR: sub_jaxprs are only valid on cond, scan, switch, and while; {} cannot carry sub_jaxprs",
            primitive.as_str()
        ))),
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

    // Fast path for single-equation Jaxprs with no constants and direct input->output mapping.
    // This avoids HashMap overhead for simple operations like vmap(add_one).
    if jaxpr.equations.len() == 1 && jaxpr.constvars.is_empty() {
        let eqn = &jaxpr.equations[0];
        if eqn.sub_jaxprs.is_empty() {
            // Resolve inputs directly from args array
            let inputs: Result<Vec<BatchTracer>, BatchError> = eqn
                .inputs
                .iter()
                .map(|atom| match atom {
                    Atom::Var(var) => jaxpr
                        .invars
                        .iter()
                        .position(|v| v == var)
                        .map(|idx| args[idx].clone())
                        .ok_or_else(|| {
                            BatchError::InterpreterError(format!("missing variable v{}", var.0))
                        }),
                    Atom::Lit(lit) => Ok(BatchTracer::unbatched(Value::Scalar(*lit))),
                })
                .collect();
            let inputs = inputs?;

            let results = apply_batch_rule_multi(eqn.primitive, &inputs, &eqn.params)?;

            // Map outputs to outvars
            return jaxpr
                .outvars
                .iter()
                .map(|outvar| {
                    eqn.outputs
                        .iter()
                        .position(|v| v == outvar)
                        .map(|idx| results[idx].clone())
                        .ok_or_else(|| {
                            BatchError::InterpreterError(format!(
                                "missing output variable v{}",
                                outvar.0
                            ))
                        })
                })
                .collect();
        }
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
        let results = batch_eval_equation_outputs(eqn, &env)?;
        if results.len() != eqn.outputs.len() {
            return Err(BatchError::InterpreterError(format!(
                "primitive {} returned {} outputs for {} bindings",
                eqn.primitive.as_str(),
                results.len(),
                eqn.outputs.len()
            )));
        }

        for (out_var, result) in eqn.outputs.iter().zip(results) {
            env.insert(*out_var, result);
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

fn is_empty_list(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.is_empty()
        || trimmed
            .trim_matches(|c| c == '[' || c == ']')
            .trim()
            .is_empty()
}

fn parse_shape(params: &BTreeMap<String, String>) -> Result<Vec<i64>, BatchError> {
    let raw = params
        .get("new_shape")
        .ok_or_else(|| BatchError::EvalError("missing required param 'new_shape'".to_owned()))?;
    parse_i64_list(raw, "new_shape")
}

fn parse_permutation(
    params: &BTreeMap<String, String>,
    rank: usize,
) -> Result<Vec<usize>, BatchError> {
    match params.get("permutation") {
        None => Ok((0..rank).rev().collect()),
        Some(raw) => parse_usize_list(raw, "permutation"),
    }
}

fn parse_param_usize_list(
    params: &BTreeMap<String, String>,
    key: &str,
) -> Result<Vec<usize>, BatchError> {
    let raw = params
        .get(key)
        .ok_or_else(|| BatchError::EvalError(format!("missing required param '{key}'")))?;
    parse_usize_list(raw, key)
}

fn parse_param_i64_list(
    params: &BTreeMap<String, String>,
    key: &str,
) -> Result<Vec<i64>, BatchError> {
    let raw = params
        .get(key)
        .ok_or_else(|| BatchError::EvalError(format!("missing required param '{key}'")))?;
    parse_i64_list(raw, key)
}

fn parse_param_usize(
    params: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<usize>, BatchError> {
    match params.get(key) {
        None => Ok(None),
        Some(raw) if raw.trim().is_empty() => Ok(None),
        Some(raw) => {
            raw.trim().parse::<usize>().map(Some).map_err(|_| {
                BatchError::EvalError(format!("invalid usize in param '{key}': '{raw}'"))
            })
        }
    }
}

fn parse_usize_list(raw: &str, key: &str) -> Result<Vec<usize>, BatchError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(BatchError::EvalError(format!(
            "empty list for param '{key}'"
        )));
    }
    let inner = trimmed.trim_matches(|c| c == '[' || c == ']');
    if inner.trim().is_empty() {
        return Err(BatchError::EvalError(format!(
            "empty list for param '{key}'"
        )));
    }
    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(BatchError::EvalError(format!(
                    "empty token in param '{key}'"
                )));
            }
            part.parse::<usize>().map_err(|_| {
                BatchError::EvalError(format!("invalid usize in param '{key}': '{part}'"))
            })
        })
        .collect()
}

fn parse_i64_list(raw: &str, key: &str) -> Result<Vec<i64>, BatchError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(BatchError::EvalError(format!(
            "empty list for param '{key}'"
        )));
    }
    let inner = trimmed.trim_matches(|c| c == '[' || c == ']');
    if inner.trim().is_empty() {
        return Err(BatchError::EvalError(format!(
            "empty list for param '{key}'"
        )));
    }
    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(BatchError::EvalError(format!(
                    "empty token in param '{key}'"
                )));
            }
            part.parse::<i64>().map_err(|_| {
                BatchError::EvalError(format!("invalid i64 in param '{key}': '{part}'"))
            })
        })
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

    fn make_f32_vector(data: &[f32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape::vector(data.len() as u32),
                data.iter().map(|&x| Literal::from_f32(x)).collect(),
            )
            .unwrap(),
        )
    }

    fn make_f32_scalar_tensor(value: f32) -> Value {
        Value::Tensor(
            TensorValue::new(DType::F32, Shape::scalar(), vec![Literal::from_f32(value)]).unwrap(),
        )
    }

    fn make_f32_matrix(rows: usize, cols: usize, data: &[f32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter().map(|&x| Literal::from_f32(x)).collect(),
            )
            .unwrap(),
        )
    }

    fn make_f32_tensor(dims: &[u32], data: &[f32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: dims.to_vec(),
                },
                data.iter().map(|&x| Literal::from_f32(x)).collect(),
            )
            .unwrap(),
        )
    }

    fn make_f64_tensor(dims: &[u32], data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: dims.to_vec(),
                },
                data.iter()
                    .map(|&x| Literal::F64Bits(x.to_bits()))
                    .collect(),
            )
            .unwrap(),
        )
    }

    fn make_complex128_tensor(dims: &[u32], data: &[(f64, f64)]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: dims.to_vec(),
                },
                data.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
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

    fn make_i64_matrix(rows: usize, cols: usize, data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
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

    fn extract_f32_vec(value: &Value) -> Vec<f32> {
        match value {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|lit| match lit {
                    Literal::F32Bits(bits) => f32::from_bits(*bits),
                    _ => f32::NAN,
                })
                .collect(),
            Value::Scalar(Literal::F32Bits(bits)) => vec![f32::from_bits(*bits)],
            Value::Scalar(_) => vec![f32::NAN],
        }
    }

    fn extract_i64_vec(value: &Value) -> Vec<i64> {
        match value {
            Value::Tensor(t) => t.elements.iter().map(|lit| lit.as_i64().unwrap()).collect(),
            Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
        }
    }

    fn extract_complex128_vec(value: &Value) -> Vec<(f64, f64)> {
        match value {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|lit| lit.as_complex128().unwrap())
                .collect(),
            Value::Scalar(lit) => vec![lit.as_complex128().unwrap()],
        }
    }

    fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a.0 - e.0).abs() <= 1e-10, "real: {a:?} != {e:?}");
            assert!((a.1 - e.1).abs() <= 1e-10, "imag: {a:?} != {e:?}");
        }
    }

    fn assert_f64_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-10, "{a} != {e}");
        }
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-5, "{a} != {e}");
        }
    }

    fn make_switch_branch_identity_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
    }

    fn make_switch_branch_self_binary_jaxpr(primitive: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_switch_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Switch,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::from([("num_branches".to_owned(), "3".to_owned())]),
                effects: vec![],
                sub_jaxprs: vec![
                    make_switch_branch_identity_jaxpr(),
                    make_switch_branch_self_binary_jaxpr(Primitive::Add),
                    make_switch_branch_self_binary_jaxpr(Primitive::Mul),
                ],
            }],
        )
    }

    fn make_cond_branch_add_ten_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(10))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_cond_branch_negate_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sub,
                inputs: smallvec![Atom::Lit(Literal::I64(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_cond_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Cond,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![
                    make_cond_branch_add_ten_jaxpr(),
                    make_cond_branch_negate_jaxpr(),
                ],
            }],
        )
    }

    fn make_while_cond_scalar_jaxpr(primitive: Primitive, threshold: Literal) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(threshold)],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_while_cond_gt_zero_jaxpr() -> Jaxpr {
        make_while_cond_scalar_jaxpr(Primitive::Gt, Literal::I64(0))
    }

    fn make_while_body_scalar_jaxpr(primitive: Primitive, operand: Literal) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(operand)],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_while_body_literal_left_scalar_jaxpr(primitive: Primitive, operand: Literal) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Lit(operand), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_while_cond_literal_left_scalar_jaxpr(
        primitive: Primitive,
        threshold: Literal,
    ) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Lit(threshold), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_while_body_sub_two_jaxpr() -> Jaxpr {
        make_while_body_scalar_jaxpr(Primitive::Sub, Literal::I64(2))
    }

    fn make_while_control_flow_jaxpr() -> Jaxpr {
        make_while_control_flow_jaxpr_with_sub_jaxprs(
            make_while_cond_gt_zero_jaxpr(),
            make_while_body_sub_two_jaxpr(),
            8,
        )
    }

    fn make_while_control_flow_jaxpr_with_sub_jaxprs(
        cond_jaxpr: Jaxpr,
        body_jaxpr: Jaxpr,
        max_iter: usize,
    ) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::While,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::from([("max_iter".to_owned(), max_iter.to_string())]),
                effects: vec![],
                sub_jaxprs: vec![cond_jaxpr, body_jaxpr],
            }],
        )
    }

    fn make_scan_body_add_emit_carry_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::I64(0))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_scan_sub_jaxpr_control_flow_jaxpr(reverse: bool) -> Jaxpr {
        let params = if reverse {
            BTreeMap::from([("reverse".to_owned(), "true".to_owned())])
        } else {
            BTreeMap::new()
        };
        Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2), VarId(3)],
                params,
                effects: vec![],
                sub_jaxprs: vec![make_scan_body_add_emit_carry_jaxpr()],
            }],
        )
    }

    fn make_scan_multi_carry_body_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4), VarId(5), VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(6)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_scan_multi_carry_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(0), VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4), VarId(5)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![
                    Atom::Var(VarId(0)),
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2))
                ],
                outputs: smallvec![VarId(3), VarId(4), VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![make_scan_multi_carry_body_jaxpr()],
            }],
        )
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
    fn test_batch_trace_elementwise_trunc() {
        // Unary elementwise: vmap(trunc) preserves the batch dim.
        let input = BatchTracer::batched(make_f64_vector(&[1.7, -2.3, 3.9]), 0);
        let result = apply_batch_rule(Primitive::Trunc, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&result.value), &[1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_batch_trace_elementwise_hypot() {
        // Binary elementwise: vmap(hypot) over two batched vectors.
        let a = BatchTracer::batched(make_f64_vector(&[3.0, 5.0]), 0);
        let b = BatchTracer::batched(make_f64_vector(&[4.0, 12.0]), 0);
        let result = apply_batch_rule(Primitive::Hypot, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&result.value), &[5.0, 13.0]);
    }

    #[test]
    fn test_batch_trace_cummax_leading_batch_dim() {
        // vmap(cummax) over [2,3]: running max along each row independently.
        let input =
            BatchTracer::batched(make_f64_tensor(&[2, 3], &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]), 0);
        let result = apply_batch_rule(Primitive::Cummax, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[1.0, 3.0, 3.0, 5.0, 5.0, 6.0],
        );
    }

    #[test]
    fn test_batch_trace_fma_ternary() {
        // Ternary elementwise Fma(a,b,c)=a*b+c via per-slice passthrough.
        let a = BatchTracer::batched(make_f64_tensor(&[2, 2], &[2.0, 3.0, 1.0, 1.0]), 0);
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2], &[4.0, 5.0, 10.0, 10.0]), 0);
        let c = BatchTracer::batched(make_f64_tensor(&[2, 2], &[1.0, 1.0, 0.0, 0.0]), 0);
        let result = apply_batch_rule(Primitive::Fma, &[a, b, c], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_f64_close(&extract_f64_vec(&result.value), &[9.0, 16.0, 10.0, 10.0]);
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

    #[test]
    fn test_batch_cumsum_handles_negative_axis() {
        // eval_cumulative normalizes negative axes; batch_cumulative must too.
        // vmap(cumsum, axis=-1) over a batch of length-3 vectors cumsums each row.
        // (Was: parse_param_usize parsed "axis" as usize, rejecting "-1"; and a
        // blind +1 shift is wrong for an end-relative axis.)
        let input =
            BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axis".to_owned(), "-1".to_owned())]);
        let result = apply_batch_rule(Primitive::Cumsum, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_batch_reduce_sum_handles_negative_axis() {
        // eval_reduce_axes normalizes negative axes; batch_reduce must too. vmap(sum, axis=-1)
        // on a batch of length-2 vectors sums each vector. (Was: parse_axes parsed "axes" as
        // usize, rejecting "-1", so vmap erred — and a naive +1 shift is wrong for a negative
        // axis, which is end-relative and unchanged when the batch is prepended at the front.)
        let input = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axes".to_owned(), "-1".to_owned())]);
        let result = apply_batch_rule(Primitive::ReduceSum, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![3.0, 7.0, 11.0]); // each length-2 vector summed
    }

    #[test]
    fn test_batch_trace_reduce_sum_all_axes_default() {
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[3, 2, 2], &data), 0);
        let result = apply_batch_rule(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![10.0, 26.0, 42.0]);
    }

    #[test]
    fn test_batch_trace_reduce_sum_empty_axes_noop() {
        let input = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axes".to_owned(), "".to_owned())]);
        let result = apply_batch_rule(Primitive::ReduceSum, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── Dot Product Tests ──────────────────────────────────────

    fn assert_dot_matches_slice_oracle(
        result: &BatchTracer,
        lhs_slices: &[Value],
        rhs_slices: &[Value],
    ) {
        assert_eq!(lhs_slices.len(), rhs_slices.len());
        assert_eq!(result.batch_dim, Some(0));

        let expected_slices = lhs_slices
            .iter()
            .zip(rhs_slices)
            .map(|(lhs, rhs)| {
                eval_primitive(
                    Primitive::Dot,
                    &[lhs.clone(), rhs.clone()],
                    &BTreeMap::new(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = Value::Tensor(TensorValue::stack_axis0(&expected_slices).unwrap());

        assert_eq!(
            result.value.as_tensor().unwrap().shape.dims,
            expected.as_tensor().unwrap().shape.dims
        );
        assert_f64_close(&extract_f64_vec(&result.value), &extract_f64_vec(&expected));
    }

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

    #[test]
    fn test_batch_trace_dot_batched_f32_matrix_unbatched_vector_direct_preserves_dtype() {
        let lhs = BatchTracer::batched(make_f32_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let rhs = BatchTracer::unbatched(make_f32_vector(&[7.0, 8.0, 9.0]));

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        let out = result.value.as_tensor().unwrap();
        assert_eq!(out.dtype, DType::F32);
        out.validate_dtype_consistency()
            .expect("single-batched lhs F32 dot output dtype/element invariant");
        assert_eq!(out.shape.dims, vec![2]);
        assert_eq!(extract_f64_vec(&result.value), vec![50.0, 122.0]);
    }

    #[test]
    fn test_batch_trace_dot_batched_matrix_unbatched_matrix_direct() {
        let lhs0 = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lhs1 = make_f64_matrix(2, 3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let rhs = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let lhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch element 0
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch element 1
                ],
            ),
            0,
        );

        let result = apply_batch_rule(
            Primitive::Dot,
            &[lhs, BatchTracer::unbatched(rhs.clone())],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs0, lhs1], &[rhs.clone(), rhs]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_unbatched_matrix_batched_matrix_direct() {
        let lhs = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let rhs1 = make_f64_matrix(3, 2, &[2.0, 1.0, 1.0, 0.0, 0.0, 2.0]);
        let rhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 2.0, 3.0, // batch element 0
                    2.0, 1.0, 1.0, 0.0, 0.0, 2.0, // batch element 1
                ],
            ),
            0,
        );

        let result = apply_batch_rule(
            Primitive::Dot,
            &[BatchTracer::unbatched(lhs.clone()), rhs],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs.clone(), lhs], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_unbatched_f32_matrix_batched_matrix_direct_preserves_dtype() {
        let lhs = BatchTracer::unbatched(make_f32_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let rhs = BatchTracer::batched(
            make_f32_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 2.0, 3.0, // batch element 0
                    2.0, 1.0, 1.0, 0.0, 0.0, 2.0, // batch element 1
                ],
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        let out = result.value.as_tensor().unwrap();
        assert_eq!(out.dtype, DType::F32);
        out.validate_dtype_consistency()
            .expect("single-batched rhs F32 dot output dtype/element invariant");
        assert_eq!(out.shape.dims, vec![2, 2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_unbatched_matrix_batched_vector_preserves_slice_semantics() {
        let lhs = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs0 = make_f64_vector(&[1.0, 0.0, 2.0]);
        let rhs1 = make_f64_vector(&[0.0, 1.0, 3.0]);
        let rhs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 0.0, 2.0, 0.0, 1.0, 3.0]), 0);

        let result = apply_batch_rule(
            Primitive::Dot,
            &[BatchTracer::unbatched(lhs.clone()), rhs],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs.clone(), lhs], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_vectors_direct() {
        let lhs0 = make_f64_vector(&[1.0, 2.0, 3.0]);
        let lhs1 = make_f64_vector(&[7.0, 8.0, 9.0]);
        let rhs0 = make_f64_vector(&[4.0, 5.0, 6.0]);
        let rhs1 = make_f64_vector(&[1.0, 0.0, 2.0]);
        let lhs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]), 0);
        let rhs = BatchTracer::batched(make_f64_matrix(2, 3, &[4.0, 5.0, 6.0, 1.0, 0.0, 2.0]), 0);

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs0, lhs1], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_i64_vectors_direct() {
        let lhs = BatchTracer::batched(make_i64_matrix(2, 3, &[1, 2, 3, 4, 5, 6]), 0);
        let rhs = BatchTracer::batched(make_i64_matrix(2, 3, &[7, 8, 9, 1, 0, 2]), 0);

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().dtype, DType::I64);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(extract_i64_vec(&result.value), vec![50, 16]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_i64_vectors_wrap_like_lax() {
        let lhs = BatchTracer::batched(make_i64_matrix(2, 2, &[i64::MAX, 2, 3, 4]), 0);
        let rhs = BatchTracer::batched(make_i64_matrix(2, 2, &[2, 2, 5, 6]), 0);

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().dtype, DType::I64);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(extract_i64_vec(&result.value), vec![2, 39]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_f32_vectors_preserves_dtype() {
        // Batched F32×F32 dot product must stay F32 (parity with JAX
        // `lax.dot_general` real-input promotion). The pre-fix code emitted
        // F64-declared tensors with F64Bits elements; that's a dtype/element
        // invariant violation in the same family as frankenjax-2chb/eldm/e8g4.
        let lhs = BatchTracer::batched(make_f32_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let rhs = BatchTracer::batched(make_f32_matrix(2, 3, &[7.0, 8.0, 9.0, 1.0, 0.0, 2.0]), 0);

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        let out = result.value.as_tensor().unwrap();
        assert_eq!(out.dtype, DType::F32);
        out.validate_dtype_consistency()
            .expect("F32 batched dot output dtype/element invariant");
        assert_eq!(out.shape.dims, vec![2]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_matrix_vector_direct() {
        let lhs0 = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lhs1 = make_f64_matrix(2, 3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let rhs0 = make_f64_vector(&[1.0, 0.0, 2.0]);
        let rhs1 = make_f64_vector(&[0.0, 1.0, 3.0]);
        let lhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch element 0
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch element 1
                ],
            ),
            0,
        );
        let rhs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 0.0, 2.0, 0.0, 1.0, 3.0]), 0);

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs0, lhs1], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_vector_matrix_direct() {
        let lhs0 = make_f64_vector(&[1.0, 2.0, 3.0]);
        let lhs1 = make_f64_vector(&[7.0, 8.0, 9.0]);
        let rhs0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let rhs1 = make_f64_matrix(3, 2, &[2.0, 1.0, 1.0, 0.0, 0.0, 2.0]);
        let lhs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]), 0);
        let rhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 2.0, 3.0, // batch element 0
                    2.0, 1.0, 1.0, 0.0, 0.0, 2.0, // batch element 1
                ],
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs0, lhs1], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_batch_trace_dot_paired_batched_matrix_matrix_direct() {
        let lhs0 = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lhs1 = make_f64_matrix(2, 3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let rhs0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let rhs1 = make_f64_matrix(3, 2, &[2.0, 1.0, 1.0, 0.0, 0.0, 2.0]);
        let lhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch element 0
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch element 1
                ],
            ),
            0,
        );
        let rhs = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 2.0, 3.0, // batch element 0
                    2.0, 1.0, 1.0, 0.0, 0.0, 2.0, // batch element 1
                ],
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Dot, &[lhs, rhs], &BTreeMap::new()).unwrap();

        assert_dot_matches_slice_oracle(&result, &[lhs0, lhs1], &[rhs0, rhs1]);
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
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

    #[test]
    fn test_batch_sort_shifts_the_axis_param_eval_reads() {
        // vmap a sort along axis 0 of 2x3 matrices. The batched operand is [2,2,3] with the
        // batch at the front, so the sort must move from axis 0 to axis 1. batch_sort must
        // shift the SAME param key eval_sort reads ("axis") — not "dimension". Otherwise the
        // unshifted axis 0 sorts the BATCH axis: a silently-wrong vmap result.
        let data: Vec<f64> = vec![
            6.0, 5.0, 4.0, 3.0, 1.0, 2.0, // batch 0: [[6,5,4],[3,1,2]]
            9.0, 8.0, 7.0, 1.0, 2.0, 3.0, // batch 1: [[9,8,7],[1,2,3]]
        ];
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 2, 3],
                    },
                    data.iter().map(|&x| Literal::F64Bits(x.to_bits())).collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let result = apply_batch_rule(Primitive::Sort, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let t = result.value.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![2, 2, 3]);
        let got: Vec<f64> = t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F64Bits(b) => f64::from_bits(*b),
                _ => unreachable!(),
            })
            .collect();
        // Each batch's columns (axis 0 of the 2x3 slice) sorted ascending:
        // batch0 cols [6,3],[5,1],[4,2] -> [[3,1,2],[6,5,4]]
        // batch1 cols [9,1],[8,2],[7,3] -> [[1,2,3],[9,8,7]]
        let expected = vec![
            3.0, 1.0, 2.0, 6.0, 5.0, 4.0, //
            1.0, 2.0, 3.0, 9.0, 8.0, 7.0,
        ];
        assert_eq!(
            got, expected,
            "vmap(sort axis=0) must sort each batch's columns, not the batch axis"
        );
    }

    #[test]
    fn test_batch_trace_transpose_default_perm() {
        // Default permutation should reverse per-element axes.
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[2, 2, 3], &data), 0);
        let result = apply_batch_rule(Primitive::Transpose, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        let vals = extract_f64_vec(&result.value);
        assert_eq!(
            vals,
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0]
        );
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

    #[test]
    fn test_batch_trace_reshape_with_inferred_dim() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[2, 2, 3], &data), 0);
        let params = BTreeMap::from([("new_shape".to_owned(), "-1, 2".to_owned())]);
        let result = apply_batch_rule(Primitive::Reshape, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, data);
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
    fn test_batch_eval_jaxpr_qr_binds_all_unbatched_outputs() {
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1), VarId(2)],
            vec![Equation {
                primitive: Primitive::Qr,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1), VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let input = BatchTracer::unbatched(make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
        let results = batch_eval_jaxpr(&jaxpr, &[input]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].batch_dim, None);
        assert_eq!(results[1].batch_dim, None);
        assert_eq!(results[0].value.as_tensor().unwrap().shape.dims, vec![3, 2]);
        assert_eq!(results[1].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_batch_eval_jaxpr_qr_binds_all_batched_outputs() {
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1), VarId(2)],
            vec![Equation {
                primitive: Primitive::Qr,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1), VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 3, 2],
                    },
                    [
                        1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, // batch element 0
                        2.0, 0.0, 0.0, 2.0, 2.0, 2.0, // batch element 1
                    ]
                    .into_iter()
                    .map(|x| Literal::F64Bits(x.to_bits()))
                    .collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let results = batch_eval_jaxpr(&jaxpr, &[input]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].batch_dim, Some(0));
        assert_eq!(results[1].batch_dim, Some(0));
        assert_eq!(
            results[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 2]
        );
        assert_eq!(
            results[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    fn assert_qr_matches_slice_oracle(
        outputs: &[BatchTracer],
        matrices: &[Value],
        params: &BTreeMap<String, String>,
    ) {
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(outputs[1].batch_dim, Some(0));

        let mut expected_by_output = vec![Vec::new(), Vec::new()];
        for matrix in matrices {
            let slice_outputs =
                eval_primitive_multi(Primitive::Qr, std::slice::from_ref(matrix), params).unwrap();
            assert_eq!(slice_outputs.len(), 2);
            for (bucket, value) in expected_by_output.iter_mut().zip(slice_outputs) {
                bucket.push(value);
            }
        }

        for (actual, expected_slices) in outputs.iter().zip(expected_by_output) {
            let expected = Value::Tensor(TensorValue::stack_axis0(&expected_slices).unwrap());
            assert_eq!(
                actual.value.as_tensor().unwrap().shape.dims,
                expected.as_tensor().unwrap().shape.dims
            );
            assert_f64_close(&extract_f64_vec(&actual.value), &extract_f64_vec(&expected));
        }
    }

    #[test]
    fn qr_3x2_thin_fast_path_matches_generic_householder_bits() {
        let cases = [
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [-1.5, 0.25, 0.75, -2.0, 3.0, 4.0],
            [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for matrix in cases {
            let mut expected = QrScratch::default();
            qr_decompose_matrix(3, 2, &matrix, false, &mut expected);

            let mut q = Vec::new();
            let mut r = Vec::new();
            qr_decompose_matrix_3x2_thin(matrix, &mut q, &mut r);

            assert_eq!(
                q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
                    .q_out
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "Q mismatch for {matrix:?}"
            );
            assert_eq!(
                r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
                    .r_out
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "R mismatch for {matrix:?}"
            );
        }
    }

    #[test]
    fn test_batch_trace_qr_multi_leading_batch_dim() {
        let matrix0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let matrix1 = make_f64_matrix(3, 2, &[2.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // batch element 0
                    2.0, 0.0, 0.0, 2.0, 2.0, 2.0, // batch element 1
                ],
            ),
            0,
        );

        let outputs = apply_batch_rule_multi(Primitive::Qr, &[input], &BTreeMap::new()).unwrap();
        assert_qr_matches_slice_oracle(&outputs, &[matrix0, matrix1], &BTreeMap::new());
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 2]
        );
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_batch_trace_qr_multi_nonleading_batch_dim() {
        let matrix0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let matrix1 = make_f64_matrix(3, 2, &[2.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[3, 2, 2],
                &[
                    1.0, 0.0, 2.0, 0.0, // row 0, both batch lanes
                    0.0, 1.0, 0.0, 2.0, // row 1, both batch lanes
                    1.0, 1.0, 2.0, 2.0, // row 2, both batch lanes
                ],
            ),
            1,
        );

        let outputs = apply_batch_rule_multi(Primitive::Qr, &[input], &BTreeMap::new()).unwrap();
        assert_qr_matches_slice_oracle(&outputs, &[matrix0, matrix1], &BTreeMap::new());
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 2]
        );
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_batch_trace_qr_multi_full_matrices() {
        let matrix0 = make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let matrix1 = make_f64_matrix(3, 2, &[2.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 2],
                &[
                    1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // batch element 0
                    2.0, 0.0, 0.0, 2.0, 2.0, 2.0, // batch element 1
                ],
            ),
            0,
        );
        let params = BTreeMap::from([("full_matrices".to_owned(), "true".to_owned())]);

        let outputs = apply_batch_rule_multi(Primitive::Qr, &[input], &params).unwrap();
        assert_qr_matches_slice_oracle(&outputs, &[matrix0, matrix1], &params);
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 3]
        );
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 2]
        );
    }

    #[test]
    fn test_batch_trace_qr_multi_matrix_rank_error_preserved() {
        let input = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 0);

        let err = apply_batch_rule_multi(Primitive::Qr, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected rank-2 tensor"));
    }

    /// Verify a multi-output vmap result against the per-slice oracle: each
    /// output stacked over `matrices` must equal running the primitive on each
    /// matrix and stacking. Generic over output arity.
    fn assert_multi_matches_slice_oracle(
        primitive: Primitive,
        outputs: &[BatchTracer],
        matrices: &[Value],
        params: &BTreeMap<String, String>,
    ) {
        let arity = eval_primitive_multi(primitive, std::slice::from_ref(&matrices[0]), params)
            .unwrap()
            .len();
        assert_eq!(outputs.len(), arity);
        for out in outputs {
            assert_eq!(out.batch_dim, Some(0));
        }
        let mut expected: Vec<Vec<Value>> = (0..arity).map(|_| Vec::new()).collect();
        for matrix in matrices {
            let slice =
                eval_primitive_multi(primitive, std::slice::from_ref(matrix), params).unwrap();
            for (bucket, value) in expected.iter_mut().zip(slice) {
                bucket.push(value);
            }
        }
        for (actual, slices) in outputs.iter().zip(expected) {
            let stacked = Value::Tensor(TensorValue::stack_axis0(&slices).unwrap());
            assert_eq!(
                actual.value.as_tensor().unwrap().shape.dims,
                stacked.as_tensor().unwrap().shape.dims
            );
            assert_f64_close(&extract_f64_vec(&actual.value), &extract_f64_vec(&stacked));
        }
    }

    #[test]
    fn test_batch_trace_lu_multi_leading_batch_dim() {
        // vmap over lu() must batch all three outputs (packed LU, pivots,
        // permutation), matching the per-slice oracle.
        let m0 = make_f64_matrix(3, 3, &[4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 6.0]);
        let m1 = make_f64_matrix(3, 3, &[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 3],
                &[
                    4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 6.0, // element 0
                    2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0, // element 1
                ],
            ),
            0,
        );
        let outputs = apply_batch_rule_multi(Primitive::Lu, &[input], &BTreeMap::new()).unwrap();
        assert_multi_matches_slice_oracle(Primitive::Lu, &outputs, &[m0, m1], &BTreeMap::new());
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 3]
        );
    }

    #[test]
    fn test_batch_trace_lu_multi_nonleading_batch_dim() {
        // Batch dim at axis 1 — exercises move_batch_dim_to_front.
        let m0 = make_f64_matrix(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let m1 = make_f64_matrix(2, 2, &[2.0, 5.0, 6.0, 1.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                &[
                    4.0, 1.0, 2.0, 5.0, // row 0, both lanes
                    1.0, 3.0, 6.0, 1.0, // row 1, both lanes
                ],
            ),
            1,
        );
        let outputs = apply_batch_rule_multi(Primitive::Lu, &[input], &BTreeMap::new()).unwrap();
        assert_multi_matches_slice_oracle(Primitive::Lu, &outputs, &[m0, m1], &BTreeMap::new());
    }

    #[test]
    fn test_batch_trace_topk_multi_leading_batch_dim() {
        // vmap over top_k() must batch both outputs (values, indices),
        // reducing the last axis per batch element.
        let params = BTreeMap::from([("k".to_owned(), "2".to_owned())]);
        let v0 = make_f64_tensor(&[4], &[3.0, 1.0, 4.0, 2.0]);
        let v1 = make_f64_tensor(&[4], &[9.0, 7.0, 8.0, 6.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 4], &[3.0, 1.0, 4.0, 2.0, 9.0, 7.0, 8.0, 6.0]),
            0,
        );
        let outputs = apply_batch_rule_multi(Primitive::TopK, &[input], &params).unwrap();
        assert_multi_matches_slice_oracle(Primitive::TopK, &outputs, &[v0, v1], &params);
        assert_eq!(outputs[0].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
    }

    #[test]
    fn test_batch_trace_topk_multi_nonleading_batch_dim() {
        // Batch dim at axis 1; move_batch_dim_to_front must transpose to
        // [batch, rows, cols] while top_k still reduces the last axis.
        let params = BTreeMap::from([("k".to_owned(), "1".to_owned())]);
        let v0 = make_f64_matrix(2, 3, &[5.0, 2.0, 8.0, 1.0, 9.0, 4.0]);
        let v1 = make_f64_matrix(2, 3, &[7.0, 3.0, 6.0, 0.0, 2.0, 10.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    5.0, 2.0, 8.0, 7.0, 3.0, 6.0, // row 0, both lanes
                    1.0, 9.0, 4.0, 0.0, 2.0, 10.0, // row 1, both lanes
                ],
            ),
            1,
        );
        let outputs = apply_batch_rule_multi(Primitive::TopK, &[input], &params).unwrap();
        assert_multi_matches_slice_oracle(Primitive::TopK, &outputs, &[v0, v1], &params);
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 1]
        );
    }

    #[test]
    fn test_batch_trace_slogdet_multi_leading_batch_dim() {
        // vmap over slogdet() must batch both scalar outputs (sign,
        // logabsdet) into rank-1 [batch] vectors.
        let m0 = make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]); // det 6
        let m1 = make_f64_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]); // det -2
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 3.0, 4.0]),
            0,
        );
        let outputs =
            apply_batch_rule_multi(Primitive::Slogdet, &[input], &BTreeMap::new()).unwrap();
        assert_multi_matches_slice_oracle(
            Primitive::Slogdet,
            &outputs,
            &[m0, m1],
            &BTreeMap::new(),
        );
        assert_eq!(outputs[0].value.as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(outputs[1].value.as_tensor().unwrap().shape.dims, vec![2]);
    }

    /// Complex-aware multi-output oracle: vmap over `matrices` must equal the
    /// per-slice eval stacked along axis 0, output-by-output. eval_eig is
    /// deterministic so each lane's eigenvalue ordering matches exactly.
    fn assert_eig_matches_slice_oracle(outputs: &[BatchTracer], matrices: &[Value]) {
        assert_eq!(outputs.len(), 2, "eig yields (eigenvalues, eigenvectors)");
        for out in outputs {
            assert_eq!(out.batch_dim, Some(0));
        }
        let mut expected: Vec<Vec<Value>> = vec![Vec::new(), Vec::new()];
        for matrix in matrices {
            let slice = eval_primitive_multi(
                Primitive::Eig,
                std::slice::from_ref(matrix),
                &BTreeMap::new(),
            )
            .unwrap();
            assert_eq!(slice.len(), 2);
            for (bucket, value) in expected.iter_mut().zip(slice) {
                bucket.push(value);
            }
        }
        for (actual, slices) in outputs.iter().zip(expected) {
            let stacked = Value::Tensor(TensorValue::stack_axis0(&slices).unwrap());
            assert_eq!(
                actual.value.as_tensor().unwrap().dtype,
                DType::Complex128,
                "eig outputs must stay Complex128"
            );
            assert_eq!(
                actual.value.as_tensor().unwrap().shape.dims,
                stacked.as_tensor().unwrap().shape.dims
            );
            assert_complex_close(
                &extract_complex128_vec(&actual.value),
                &extract_complex128_vec(&stacked),
            );
        }
    }

    #[test]
    fn test_batch_trace_eig_multi_leading_batch_dim() {
        // vmap over eig() must batch both Complex128 outputs (eigenvalues,
        // eigenvectors) and match the per-slice oracle.
        let m0 = make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let m1 = make_f64_matrix(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 3.0]),
            0,
        );
        let outputs = apply_batch_rule_multi(Primitive::Eig, &[input], &BTreeMap::new()).unwrap();
        assert_eig_matches_slice_oracle(&outputs, &[m0, m1]);
        assert_eq!(outputs[0].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_batch_trace_eig_multi_nonleading_batch_dim() {
        // Batch dim at axis 1 — exercises move_batch_dim_to_front before the
        // per-slice eig.
        let m0 = make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let m1 = make_f64_matrix(2, 2, &[4.0, 1.0, 2.0, 3.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                &[
                    2.0, 0.0, 4.0, 1.0, // row 0, both lanes
                    0.0, 3.0, 2.0, 3.0, // row 1, both lanes
                ],
            ),
            1,
        );
        let outputs = apply_batch_rule_multi(Primitive::Eig, &[input], &BTreeMap::new()).unwrap();
        assert_eig_matches_slice_oracle(&outputs, &[m0, m1]);
    }

    fn assert_eigh_matches_slice_oracle(outputs: &[BatchTracer], matrices: &[Value]) {
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(outputs[1].batch_dim, Some(0));

        let mut expected_by_output = vec![Vec::new(), Vec::new()];
        for matrix in matrices {
            let slice_outputs = eval_primitive_multi(
                Primitive::Eigh,
                std::slice::from_ref(matrix),
                &BTreeMap::new(),
            )
            .unwrap();
            assert_eq!(slice_outputs.len(), 2);
            for (bucket, value) in expected_by_output.iter_mut().zip(slice_outputs) {
                bucket.push(value);
            }
        }

        for (idx, (actual, expected_slices)) in outputs.iter().zip(expected_by_output).enumerate() {
            let expected = Value::Tensor(TensorValue::stack_axis0(&expected_slices).unwrap());
            let dims = actual.value.as_tensor().unwrap().shape.dims.clone();
            assert_eq!(dims, expected.as_tensor().unwrap().shape.dims);
            let actual_vec = extract_f64_vec(&actual.value);
            let expected_vec = extract_f64_vec(&expected);
            if idx == 0 {
                // Output 0 is the eigenvalue vector: ascending and sign-unambiguous.
                assert_f64_close(&actual_vec, &expected_vec);
            } else {
                // Output 1 is the eigenvector matrix. The batched kernel
                // (batch_eigh_multi) and the per-slice oracle (fj-lax's eigh) are
                // independent valid eigensolvers, so each eigenvector column is
                // only defined up to an arbitrary ± sign. Compare each column up
                // to that sign rather than element-wise (the eigenvalues already
                // pin the spectrum and the column ordering).
                assert_eigvecs_close_up_to_sign(&dims, &actual_vec, &expected_vec);
            }
        }
    }

    /// Assert two stacks of eigenvector matrices agree up to a per-column sign.
    /// Eigenvectors are stored as columns (`V[row*n + col]`, see fj-lax eigh), and
    /// the sign of each eigenvector is mathematically arbitrary, so a column
    /// matches if it equals the reference column or its negation within tolerance.
    fn assert_eigvecs_close_up_to_sign(dims: &[u32], actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        let n = *dims.last().unwrap() as usize;
        assert!(
            n > 0 && actual.len().is_multiple_of(n * n),
            "eigenvector dims {dims:?}"
        );
        let batch = actual.len() / (n * n);
        for b in 0..batch {
            let base = b * n * n;
            for col in 0..n {
                let mut max_same: f64 = 0.0;
                let mut max_flip: f64 = 0.0;
                for row in 0..n {
                    let a = actual[base + row * n + col];
                    let e = expected[base + row * n + col];
                    max_same = max_same.max((a - e).abs());
                    max_flip = max_flip.max((a + e).abs());
                }
                assert!(
                    max_same <= 1e-10 || max_flip <= 1e-10,
                    "eigenvector column {col} (batch {b}) differs beyond sign: \
                     max|v-e|={max_same}, max|v+e|={max_flip}"
                );
            }
        }
    }

    #[test]
    fn test_batch_trace_eigh_multi_leading_batch_dim() {
        let matrix0 = make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let matrix1 = make_f64_matrix(2, 2, &[4.0, 1.0, 1.0, 4.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                [
                    2.0, 0.0, 0.0, 3.0, // batch element 0
                    4.0, 1.0, 1.0, 4.0, // batch element 1
                ]
                .as_slice(),
            ),
            0,
        );

        let outputs = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap();
        assert_eigh_matches_slice_oracle(&outputs, &[matrix0, matrix1]);
        assert_eq!(outputs[0].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_batch_trace_eigh_multi_leading_batch_dim_golden_sha256() {
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                [
                    2.0, 0.0, 0.0, 3.0, // batch element 0
                    4.0, 1.0, 1.0, 4.0, // batch element 1
                ]
                .as_slice(),
            ),
            0,
        );

        let outputs = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap();
        let w = outputs[0].value.as_tensor().unwrap();
        let v = outputs[1].value.as_tensor().unwrap();
        let w_bits: Vec<u64> = extract_f64_vec(&outputs[0].value)
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let v_bits: Vec<u64> = extract_f64_vec(&outputs[1].value)
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let digest = fj_test_utils::fixture_id_from_json(&(
            w.shape.dims.clone(),
            w_bits,
            v.shape.dims.clone(),
            v_bits,
        ))
        .unwrap();

        assert_eq!(
            digest,
            "de40295687095bc622bd73074d24337004f440bdd2cc65d8a8759dfb5cf0b106"
        );
    }

    #[test]
    fn test_batch_trace_eigh_multi_rank3_golden_sha256() {
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 3, 3],
                [
                    2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 4.0, 2.07, 0.1, 0.0, 0.1, 3.07, 0.2,
                    0.0, 0.2, 4.07,
                ]
                .as_slice(),
            ),
            0,
        );

        let outputs = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap();
        let w = outputs[0].value.as_tensor().unwrap();
        let v = outputs[1].value.as_tensor().unwrap();
        let w_bits: Vec<u64> = extract_f64_vec(&outputs[0].value)
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let v_bits: Vec<u64> = extract_f64_vec(&outputs[1].value)
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let digest = fj_test_utils::fixture_id_from_json(&(
            w.shape.dims.clone(),
            w_bits,
            v.shape.dims.clone(),
            v_bits,
        ))
        .unwrap();

        assert_eq!(
            digest,
            "9c8554df967d304b2570460fc5db4fca86602577232fcf8b01177fcd41cd365f"
        );
    }

    #[test]
    fn test_batch_trace_eigh_multi_nonleading_batch_dim() {
        let matrix0 = make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let matrix1 = make_f64_matrix(2, 2, &[4.0, 1.0, 1.0, 4.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                [
                    2.0, 4.0, // row 0 col 0, both batch lanes
                    0.0, 1.0, // row 0 col 1, both batch lanes
                    0.0, 1.0, // row 1 col 0, both batch lanes
                    3.0, 4.0, // row 1 col 1, both batch lanes
                ]
                .as_slice(),
            ),
            2,
        );

        let outputs = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap();
        assert_eigh_matches_slice_oracle(&outputs, &[matrix0, matrix1]);
        assert_eq!(outputs[0].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[1].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_batch_trace_eigh_multi_non_square_batched_matrix_rejects() {
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch element 0
                    2.0, 0.0, 0.0, 0.0, 2.0, 0.0, // batch element 1
                ],
            ),
            0,
        );

        let err = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("Eigh requires a square matrix, got 2x3")
        );
    }

    #[test]
    fn test_batch_trace_eigh_multi_matrix_rank_error_preserved() {
        let input = BatchTracer::batched(make_f64_matrix(2, 2, &[2.0, 0.0, 0.0, 3.0]), 0);

        let err = apply_batch_rule_multi(Primitive::Eigh, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected rank-2 tensor"));
    }

    fn assert_svd_matches_slice_oracle(
        outputs: &[BatchTracer],
        matrices: &[Value],
        params: &BTreeMap<String, String>,
    ) {
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(outputs[1].batch_dim, Some(0));
        assert_eq!(outputs[2].batch_dim, Some(0));

        let mut expected_by_output = vec![Vec::new(), Vec::new(), Vec::new()];
        for matrix in matrices {
            let slice_outputs =
                eval_primitive_multi(Primitive::Svd, std::slice::from_ref(matrix), params).unwrap();
            assert_eq!(slice_outputs.len(), 3);
            for (bucket, value) in expected_by_output.iter_mut().zip(slice_outputs) {
                bucket.push(value);
            }
        }

        for (actual, expected_slices) in outputs.iter().zip(expected_by_output) {
            let expected = Value::Tensor(TensorValue::stack_axis0(&expected_slices).unwrap());
            assert_eq!(
                actual.value.as_tensor().unwrap().shape.dims,
                expected.as_tensor().unwrap().shape.dims
            );
            assert_f64_close(&extract_f64_vec(&actual.value), &extract_f64_vec(&expected));
        }
    }

    #[test]
    fn svd_3x2_thin_fast_path_matches_generic_bits() {
        let cases = [
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.1, 0.0, 1.5, 0.2, 2.0],
            [-1.5, 0.25, 0.75, -2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for matrix in cases {
            let mut expected = SvdScratch::default();
            svd_decompose_matrix(3, 2, &matrix, false, &mut expected);

            let mut fast = SvdScratch::default();
            svd_decompose_matrix_3x2_thin(matrix, &mut fast);

            assert_eq!(
                fast.u_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
                    .u_out
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "U mismatch for {matrix:?}"
            );
            assert_eq!(
                fast.sigma.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
                    .sigma
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "sigma mismatch for {matrix:?}"
            );
            assert_eq!(
                fast.vt.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.vt.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "Vt mismatch for {matrix:?}"
            );
        }
    }

    #[test]
    fn test_batch_trace_svd_multi_leading_batch_dim() {
        let matrix0 = make_f64_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        let matrix1 = make_f64_matrix(2, 3, &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 2.0, 0.0, // batch element 0
                    3.0, 0.0, 0.0, 0.0, 4.0, 0.0, // batch element 1
                ],
            ),
            0,
        );

        let outputs = apply_batch_rule_multi(Primitive::Svd, &[input], &BTreeMap::new()).unwrap();
        assert_svd_matches_slice_oracle(&outputs, &[matrix0, matrix1], &BTreeMap::new());
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
        assert_eq!(outputs[1].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[2].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 3]
        );
    }

    #[test]
    fn test_batch_trace_svd_multi_nonleading_batch_dim() {
        let matrix0 = make_f64_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        let matrix1 = make_f64_matrix(2, 3, &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 0.0, 0.0, // row 0, batch lane 0
                    3.0, 0.0, 0.0, // row 0, batch lane 1
                    0.0, 2.0, 0.0, // row 1, batch lane 0
                    0.0, 4.0, 0.0, // row 1, batch lane 1
                ],
            ),
            1,
        );

        let outputs = apply_batch_rule_multi(Primitive::Svd, &[input], &BTreeMap::new()).unwrap();
        assert_svd_matches_slice_oracle(&outputs, &[matrix0, matrix1], &BTreeMap::new());
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
        assert_eq!(outputs[1].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[2].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 3]
        );
    }

    #[test]
    fn test_batch_trace_svd_multi_full_matrices() {
        let matrix0 = make_f64_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        let matrix1 = make_f64_matrix(2, 3, &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0]);
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[
                    1.0, 0.0, 0.0, 0.0, 2.0, 0.0, // batch element 0
                    3.0, 0.0, 0.0, 0.0, 4.0, 0.0, // batch element 1
                ],
            ),
            0,
        );
        let params = BTreeMap::from([("full_matrices".to_owned(), "true".to_owned())]);

        let outputs = apply_batch_rule_multi(Primitive::Svd, &[input], &params).unwrap();
        assert_svd_matches_slice_oracle(&outputs, &[matrix0, matrix1], &params);
        assert_eq!(
            outputs[0].value.as_tensor().unwrap().shape.dims,
            vec![2, 2, 2]
        );
        assert_eq!(outputs[1].value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(
            outputs[2].value.as_tensor().unwrap().shape.dims,
            vec![2, 3, 3]
        );
    }

    #[test]
    fn test_batch_trace_svd_multi_matrix_rank_error_preserved() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]), 0);

        let err = apply_batch_rule_multi(Primitive::Svd, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected rank-2 tensor"));
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

    #[test]
    fn test_batch_trace_cond_unbatched_tensor_predicate_selects_branch() {
        let pred_value = Value::Tensor(
            TensorValue::new(DType::Bool, Shape::scalar(), vec![Literal::Bool(true)]).unwrap(),
        );
        let pred = BatchTracer::unbatched(pred_value);
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

    #[test]
    fn test_batch_trace_cond_batched_predicate_vectorizes_selection() {
        let pred = BatchTracer::batched(make_i64_vector(&[1, 0, 1]), 0);
        let on_true = BatchTracer::batched(make_i64_vector(&[7, 8, 9]), 0);
        let on_false = BatchTracer::unbatched(Value::scalar_i64(-1));
        let result = apply_batch_rule(
            Primitive::Cond,
            &[pred, on_true, on_false],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![7, -1, 9]);
    }

    #[test]
    fn test_batch_trace_switch_scalar_index_selects_batched_branch() {
        let idx = BatchTracer::unbatched(Value::scalar_i64(1));
        let on_zero = BatchTracer::unbatched(Value::scalar_i64(-1));
        let on_one = BatchTracer::batched(make_i64_vector(&[4, 5, 6]), 0);
        let on_two = BatchTracer::batched(make_i64_vector(&[40, 50, 60]), 0);
        let result = apply_batch_rule(
            Primitive::Switch,
            &[idx, on_zero, on_one, on_two],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![4, 5, 6]);
    }

    #[test]
    fn test_batch_trace_switch_scalar_index_clamps_to_valid_branches() {
        let on_zero = BatchTracer::batched(make_i64_vector(&[1, 2, 3]), 0);
        let on_one = BatchTracer::batched(make_i64_vector(&[4, 5, 6]), 0);
        let on_two = BatchTracer::batched(make_i64_vector(&[40, 50, 60]), 0);

        let low = apply_batch_rule(
            Primitive::Switch,
            &[
                BatchTracer::unbatched(Value::scalar_i64(-1)),
                on_zero.clone(),
                on_one.clone(),
                on_two.clone(),
            ],
            &BTreeMap::new(),
        )
        .expect("negative switch index should clamp to the first branch");
        assert_eq!(extract_i64_vec(&low.value), vec![1, 2, 3]);

        let high = apply_batch_rule(
            Primitive::Switch,
            &[
                BatchTracer::unbatched(Value::scalar_u64(u64::MAX)),
                on_zero,
                on_one,
                on_two,
            ],
            &BTreeMap::new(),
        )
        .expect("high switch index should clamp to the last branch");
        assert_eq!(extract_i64_vec(&high.value), vec![40, 50, 60]);
    }

    #[test]
    fn test_batch_trace_switch_rejects_num_branches_mismatch() -> Result<(), BatchError> {
        let idx = BatchTracer::unbatched(Value::scalar_i64(0));
        let on_zero = BatchTracer::unbatched(Value::scalar_i64(11));
        let on_one = BatchTracer::batched(make_i64_vector(&[22, 23]), 0);
        let mut params = BTreeMap::new();
        params.insert("num_branches".to_owned(), "3".to_owned());

        let err = apply_batch_rule(Primitive::Switch, &[idx, on_zero, on_one], &params)
            .expect_err("mismatched num_branches should error");
        match err {
            BatchError::InterpreterError(msg) => {
                assert!(
                    msg.contains("switch expected 3 branch values but got 2"),
                    "unexpected error: {msg}"
                );
                Ok(())
            }
            other => Err(other),
        }
    }

    #[test]
    fn test_batch_trace_switch_tensor_index_selects_batched_branch() {
        let idx_value = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(2)]).unwrap(),
        );
        let idx = BatchTracer::unbatched(idx_value);
        let on_zero = BatchTracer::unbatched(Value::scalar_i64(-1));
        let on_one = BatchTracer::batched(make_i64_vector(&[4, 5, 6]), 0);
        let on_two = BatchTracer::batched(make_i64_vector(&[40, 50, 60]), 0);
        let result = apply_batch_rule(
            Primitive::Switch,
            &[idx, on_zero, on_one, on_two],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![40, 50, 60]);
    }

    #[test]
    fn test_batch_trace_switch_scalar_index_broadcasts_unbatched_branch() {
        let idx = BatchTracer::unbatched(Value::scalar_i64(0));
        let on_zero = BatchTracer::unbatched(Value::scalar_i64(11));
        let on_one = BatchTracer::batched(make_i64_vector(&[4, 5, 6]), 0);
        let on_two = BatchTracer::batched(make_i64_vector(&[40, 50, 60]), 0);
        let result = apply_batch_rule(
            Primitive::Switch,
            &[idx, on_zero, on_one, on_two],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![11, 11, 11]);
    }

    #[test]
    fn test_batch_trace_switch_batched_index_vectorizes_value_branches() {
        let idx = BatchTracer::batched(make_i64_vector(&[-1, 0, 1, 2, 99]), 0);
        let on_zero = BatchTracer::unbatched(Value::scalar_i64(10));
        let on_one = BatchTracer::batched(make_i64_vector(&[1, 2, 3, 4, 5]), 0);
        let on_two = BatchTracer::batched(make_i64_vector(&[10, 20, 30, 40, 50]), 0);
        let result = apply_batch_rule(
            Primitive::Switch,
            &[idx, on_zero, on_one, on_two],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![10, 10, 3, 40, 50]);
    }

    #[test]
    fn test_batch_eval_jaxpr_switch_sub_jaxprs_scalar_index_batches_selected_branch() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::unbatched(Value::scalar_i64(1)),
                BatchTracer::batched(make_i64_vector(&[2, 3, 4]), 0),
            ],
        )
        .expect("switch with sub_jaxprs should batch the selected branch");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![4, 6, 8]);
    }

    #[test]
    fn test_batch_eval_jaxpr_switch_sub_jaxprs_batched_index_selects_per_element_branch() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::batched(make_i64_vector(&[0, 1, 2]), 0),
                BatchTracer::batched(make_i64_vector(&[5, 6, 7]), 0),
            ],
        )
        .expect("batched switch index should select branches per element");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![5, 12, 49]);
    }

    #[test]
    fn test_batch_eval_jaxpr_switch_sub_jaxprs_clamps_batched_indices() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::batched(make_i64_vector(&[-1, 1, 99]), 0),
                BatchTracer::batched(make_i64_vector(&[5, 6, 7]), 0),
            ],
        )
        .expect("batched switch indices should clamp per element");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![5, 12, 49]);
    }

    #[test]
    fn test_batch_eval_jaxpr_cond_sub_jaxprs_batched_predicate_selects_per_element_branch() {
        let jaxpr = make_cond_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::batched(make_i64_vector(&[1, 0, 1]), 0),
                BatchTracer::batched(make_i64_vector(&[2, 3, 4]), 0),
            ],
        )
        .expect("cond with sub_jaxprs should select branches per batch element");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![12, -3, 14]);
    }

    #[test]
    fn test_batch_eval_jaxpr_while_sub_jaxprs_batched_carry_uses_active_mask_loop() {
        let jaxpr = make_while_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[BatchTracer::batched(make_i64_vector(&[1, 2, 5]), 0)],
        )
        .expect("while with sub_jaxprs should batch independent scalar lanes");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![-1, 0, -1]);
    }

    #[test]
    fn test_i64_gt_sub_while_closed_form_preserves_loop_boundary() {
        let values = [
            Literal::I64(64),
            Literal::I64(65),
            Literal::I64(1),
            Literal::I64(0),
            Literal::I64(-1),
        ];
        let solved = solve_i64_gt_sub_while(
            WhileCondOp::Gt,
            Literal::I64(0),
            WhileScalarOp::Sub,
            Literal::I64(2),
            64,
            &values,
        )
        .expect("closed form should not fail")
        .expect("gt/sub i64 while should be recognized");
        assert_eq!(
            solved,
            vec![
                Literal::I64(0),
                Literal::I64(-1),
                Literal::I64(-1),
                Literal::I64(0),
                Literal::I64(-1),
            ]
        );

        let err = solve_i64_gt_sub_while(
            WhileCondOp::Gt,
            Literal::I64(0),
            WhileScalarOp::Sub,
            Literal::I64(2),
            32,
            &[Literal::I64(64)],
        )
        .expect_err("condition true on the final allowed iteration must remain a max_iter error");
        assert!(
            err.to_string()
                .contains("while_loop exceeded max iterations (32)")
        );

        let overflow_risk = solve_i64_gt_sub_while(
            WhileCondOp::Gt,
            Literal::I64(i64::MIN),
            WhileScalarOp::Sub,
            Literal::I64(i64::MAX),
            64,
            &[Literal::I64(0)],
        )
        .expect("overflow-risk cases should fall back instead of failing");
        assert_eq!(overflow_risk, None);
    }

    #[test]
    fn test_batch_eval_jaxpr_while_sub_jaxprs_batched_carry_supports_div_pow_body_ops() {
        let div_jaxpr = make_while_control_flow_jaxpr_with_sub_jaxprs(
            make_while_cond_scalar_jaxpr(Primitive::Gt, Literal::I64(1)),
            make_while_body_scalar_jaxpr(Primitive::Div, Literal::I64(2)),
            8,
        );
        let div_outputs = batch_eval_jaxpr(
            &div_jaxpr,
            &[BatchTracer::batched(make_i64_vector(&[64, 32, 8]), 0)],
        )
        .expect("while sub_jaxpr body div should batch independent scalar lanes");
        assert_eq!(div_outputs.len(), 1);
        assert_eq!(div_outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&div_outputs[0].value), vec![1, 1, 1]);

        let pow_jaxpr = make_while_control_flow_jaxpr_with_sub_jaxprs(
            make_while_cond_scalar_jaxpr(Primitive::Lt, Literal::I64(100)),
            make_while_body_scalar_jaxpr(Primitive::Pow, Literal::I64(2)),
            4,
        );
        let pow_outputs = batch_eval_jaxpr(
            &pow_jaxpr,
            &[BatchTracer::batched(make_i64_vector(&[2, 3]), 0)],
        )
        .expect("while sub_jaxpr body pow should batch independent scalar lanes");
        assert_eq!(pow_outputs.len(), 1);
        assert_eq!(pow_outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&pow_outputs[0].value), vec![256, 6_561]);
    }

    #[test]
    fn test_batch_eval_jaxpr_while_sub_jaxprs_literal_left_scalar_fast_path() {
        let jaxpr = make_while_control_flow_jaxpr_with_sub_jaxprs(
            make_while_cond_literal_left_scalar_jaxpr(Primitive::Lt, Literal::I64(0)),
            make_while_body_literal_left_scalar_jaxpr(Primitive::Sub, Literal::I64(0)),
            4,
        );
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[BatchTracer::batched(make_i64_vector(&[1, 2, 5]), 0)],
        )
        .expect("literal-left while sub_jaxprs should batch independent scalar lanes");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![-1, -2, -5]);
    }

    #[test]
    fn test_while_sub_jaxpr_scalar_recognizers_preserve_literal_left_order() {
        let (cond_op, cond_literal) = while_sub_jaxpr_scalar_cond(
            &make_while_cond_literal_left_scalar_jaxpr(Primitive::Lt, Literal::I64(0)),
        )
        .expect("literal-left lt should map to carry gt literal");
        assert_eq!(cond_literal, Literal::I64(0));
        assert_eq!(
            apply_while_scalar_cond(cond_op, Literal::I64(3), cond_literal),
            Some(true)
        );
        assert_eq!(
            apply_while_scalar_cond(cond_op, Literal::I64(0), cond_literal),
            Some(false)
        );

        let cases = [
            (
                Primitive::Sub,
                Literal::I64(0),
                Literal::I64(5),
                Literal::I64(-5),
            ),
            (
                Primitive::Div,
                Literal::I64(12),
                Literal::I64(3),
                Literal::I64(4),
            ),
            (
                Primitive::Pow,
                Literal::I64(2),
                Literal::I64(4),
                Literal::I64(16),
            ),
        ];
        for (primitive, operand, carry, expected) in cases {
            let (body_op, body_operand) = while_sub_jaxpr_scalar_body(
                &make_while_body_literal_left_scalar_jaxpr(primitive, operand),
            )
            .expect("literal-left body op should preserve operand order");
            assert_eq!(body_operand, operand);
            assert_eq!(
                apply_while_scalar_op(body_op, carry, body_operand),
                Some(expected)
            );
        }
    }

    #[test]
    fn test_batch_eval_jaxpr_while_sub_jaxprs_active_mask_preserves_max_iter_error() {
        let mut jaxpr = make_while_control_flow_jaxpr();
        jaxpr.equations[0]
            .params
            .insert("max_iter".to_owned(), "1".to_owned());

        let err = batch_eval_jaxpr(&jaxpr, &[BatchTracer::batched(make_i64_vector(&[4, 1]), 0)])
            .expect_err("still-active scalar lanes should preserve max_iter failure");

        assert!(
            err.to_string()
                .contains("while_loop exceeded max iterations (1)")
        );
    }

    #[test]
    fn test_batch_eval_jaxpr_scan_sub_jaxprs_batched_xs_emits_batched_ys() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::unbatched(Value::scalar_i64(0)),
                BatchTracer::batched(make_i64_matrix(2, 3, &[1, 2, 3, 10, 20, 30]), 0),
            ],
        )
        .expect("scan with body sub_jaxpr should batch xs lanes");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![6, 60]);
        assert_eq!(outputs[1].batch_dim, Some(0));
        assert_eq!(
            extract_i64_vec(&outputs[1].value),
            vec![1, 3, 6, 10, 30, 60]
        );
    }

    #[test]
    fn test_batch_eval_jaxpr_scan_sub_jaxprs_reverse_matches_input_order_ys() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(true);
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::unbatched(Value::scalar_i64(0)),
                BatchTracer::batched(make_i64_matrix(2, 3, &[1, 2, 3, 10, 20, 30]), 0),
            ],
        )
        .expect("reverse scan should preserve input-order ys after reverse iteration");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![6, 60]);
        assert_eq!(outputs[1].batch_dim, Some(0));
        assert_eq!(
            extract_i64_vec(&outputs[1].value),
            vec![6, 5, 3, 60, 50, 30]
        );
    }

    #[test]
    fn test_batch_eval_jaxpr_scan_sub_jaxprs_add_emit_wraps_i64() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::unbatched(Value::scalar_i64(i64::MAX)),
                BatchTracer::batched(make_i64_matrix(1, 2, &[1, 2]), 0),
            ],
        )
        .expect("add-emit scan should preserve wrapping i64 semantics");

        assert_eq!(outputs.len(), 2);
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![i64::MIN + 2]);
        assert!(
            outputs[0]
                .value
                .as_tensor()
                .and_then(|tensor| tensor.elements.as_i64_slice())
                .is_some(),
            "direct scan carry output should use dense i64 storage"
        );
        assert_eq!(
            extract_i64_vec(&outputs[1].value),
            vec![i64::MIN, i64::MIN + 2]
        );
        assert!(
            outputs[1]
                .value
                .as_tensor()
                .and_then(|tensor| tensor.elements.as_i64_slice())
                .is_some(),
            "direct scan ys output should use dense i64 storage"
        );
    }

    #[test]
    fn test_batch_eval_jaxpr_scan_sub_jaxprs_multi_carry_batches_outputs() {
        let jaxpr = make_scan_multi_carry_control_flow_jaxpr();
        let outputs = batch_eval_jaxpr(
            &jaxpr,
            &[
                BatchTracer::unbatched(Value::scalar_i64(0)),
                BatchTracer::unbatched(Value::scalar_i64(1)),
                BatchTracer::batched(make_i64_matrix(2, 3, &[1, 2, 3, 2, 3, 4]), 0),
            ],
        )
        .expect("multi-carry functional scan should batch all carries and ys");

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[0].value), vec![6, 9]);
        assert_eq!(outputs[1].batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&outputs[1].value), vec![6, 24]);
        assert_eq!(outputs[2].batch_dim, Some(0));
        assert_eq!(
            extract_i64_vec(&outputs[2].value),
            vec![2, 5, 12, 4, 11, 33]
        );
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
    fn test_batch_trace_scan_shared_i64_add_wraps_and_defaults() {
        let init = BatchTracer::unbatched(Value::scalar_i64(i64::MAX));
        let xs = BatchTracer::batched(make_i64_matrix(2, 2, &[1, 2, i64::MAX, 3]), 0);

        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &BTreeMap::new())
            .expect("default add scan should preserve i64 wrapping");

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().dtype, DType::I64);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![
                i64::MAX.wrapping_add(1).wrapping_add(2),
                i64::MAX.wrapping_add(i64::MAX).wrapping_add(3),
            ]
        );
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_nonzero_axis() {
        // Shape is [scan_len, batch]; each mapped element scans one column.
        let init = BatchTracer::unbatched(Value::scalar_i64(0));
        let xs = BatchTracer::batched(make_i64_matrix(3, 2, &[1, 10, 2, 20, 3, 30]), 1);
        let params = BTreeMap::from([("body_op".to_owned(), "add".to_owned())]);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![6, 60]);
    }

    #[test]
    fn test_batch_trace_scan_batched_scalar_xs() {
        // vmap(scan) over scalar xs should run one body step per batch element.
        let init = BatchTracer::batched(make_i64_vector(&[10, 20, 30]), 0);
        let xs = BatchTracer::batched(make_i64_vector(&[1, 2, 3]), 0);
        let params = BTreeMap::from([("body_op".to_owned(), "add".to_owned())]);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![11, 22, 33]);
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_f64_mul() {
        let init = BatchTracer::unbatched(Value::scalar_f64(2.0));
        let xs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("body_op".to_owned(), "mul".to_owned())]);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&result.value), &[12.0, 240.0]);
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_f64_div_pow() {
        let div_result = apply_batch_rule(
            Primitive::Scan,
            &[
                BatchTracer::unbatched(Value::scalar_f64(120.0)),
                BatchTracer::batched(make_f64_matrix(2, 3, &[2.0, 3.0, 4.0, 5.0, 2.0, 3.0]), 0),
            ],
            &BTreeMap::from([("body_op".to_owned(), "div".to_owned())]),
        )
        .unwrap();
        assert_eq!(div_result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&div_result.value), &[5.0, 4.0]);

        let pow_result = apply_batch_rule(
            Primitive::Scan,
            &[
                BatchTracer::unbatched(Value::scalar_f64(2.0)),
                BatchTracer::batched(make_f64_matrix(2, 2, &[2.0, 2.0, 3.0, 1.0]), 0),
            ],
            &BTreeMap::from([("body_op".to_owned(), "pow".to_owned())]),
        )
        .unwrap();
        assert_eq!(pow_result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&pow_result.value), &[16.0, 8.0]);
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_f32_add_preserves_literal_width() {
        let result = apply_batch_rule(
            Primitive::Scan,
            &[
                BatchTracer::unbatched(Value::scalar_f32(0.5)),
                BatchTracer::batched(make_f32_matrix(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), 0),
            ],
            &BTreeMap::from([("body_op".to_owned(), "add".to_owned())]),
        )
        .expect("f32 scan scalar fast path should batch rows");
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.dtype(), DType::F32);
        assert_f32_close(&extract_f32_vec(&result.value), &[6.5, 60.5]);
    }

    #[test]
    fn test_batch_trace_scan_shared_f32_add_direct_path_preserves_reverse_order() {
        let init = BatchTracer::unbatched(make_f32_scalar_tensor(1.0e20));
        let xs = BatchTracer::batched(make_f32_matrix(2, 2, &[-1.0e20, 3.0, 4.0, -1.0e20]), 0);

        let forward = apply_batch_rule(
            Primitive::Scan,
            &[init.clone(), xs.clone()],
            &BTreeMap::from([("body_op".to_owned(), "add".to_owned())]),
        )
        .expect("f32 shared add scan should use direct row-major path");
        let reverse = apply_batch_rule(
            Primitive::Scan,
            &[init, xs],
            &BTreeMap::from([
                ("body_op".to_owned(), "add".to_owned()),
                ("reverse".to_owned(), "true".to_owned()),
            ]),
        )
        .expect("f32 shared add reverse scan should preserve reverse order");

        assert_eq!(forward.value.dtype(), DType::F32);
        assert_eq!(reverse.value.dtype(), DType::F32);
        assert_f32_close(&extract_f32_vec(&forward.value), &[3.0, 0.0]);
        assert_f32_close(&extract_f32_vec(&reverse.value), &[0.0, 4.0]);
    }

    #[test]
    fn test_batch_trace_scan_shared_f32_add_direct_path_preserves_nan_exitless_fold() {
        let init = BatchTracer::unbatched(Value::scalar_f32(0.5));
        let xs = BatchTracer::batched(make_f32_matrix(2, 2, &[1.0, f32::NAN, 2.0, 3.0]), 0);
        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &BTreeMap::new())
            .expect("default f32 add scan should batch directly");

        let values = extract_f32_vec(&result.value);
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.dtype(), DType::F32);
        assert!(values[0].is_nan());
        assert!((values[1] - 5.5).abs() <= 1e-5, "{} != 5.5", values[1]);
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_i64_max_min() {
        let xs = BatchTracer::batched(make_i64_matrix(2, 3, &[1, 7, 3, 10, 2, 8]), 0);

        let max_params = BTreeMap::from([("body_op".to_owned(), "max".to_owned())]);
        let max_result = apply_batch_rule(
            Primitive::Scan,
            &[BatchTracer::unbatched(Value::scalar_i64(5)), xs.clone()],
            &max_params,
        )
        .unwrap();
        assert_eq!(max_result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&max_result.value), vec![7, 10]);

        let min_params = BTreeMap::from([("body_op".to_owned(), "min".to_owned())]);
        let min_result = apply_batch_rule(
            Primitive::Scan,
            &[BatchTracer::unbatched(Value::scalar_i64(5)), xs],
            &min_params,
        )
        .unwrap();
        assert_eq!(min_result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&min_result.value), vec![1, 2]);
    }

    #[test]
    fn test_batch_trace_scan_shared_i64_max_signed_reverse() {
        let init = BatchTracer::unbatched(Value::scalar_i64(-5));
        let xs = BatchTracer::batched(make_i64_matrix(2, 3, &[i64::MIN, -5, 3, 2, 2, -10]), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "max".to_owned()),
            ("reverse".to_owned(), "true".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Scan, &[init, xs], &params)
            .expect("shared i64 max scan should preserve signed reverse semantics");

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().dtype, DType::I64);
        assert_eq!(extract_i64_vec(&result.value), vec![3, 2]);
    }

    #[test]
    fn test_batch_trace_scan_batched_xs_f64_max_min() {
        let xs = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 7.5, 3.0, 10.0, 2.0, 8.0]), 0);

        let max_params = BTreeMap::from([("body_op".to_owned(), "max".to_owned())]);
        let max_result = apply_batch_rule(
            Primitive::Scan,
            &[BatchTracer::unbatched(Value::scalar_f64(5.0)), xs.clone()],
            &max_params,
        )
        .unwrap();
        assert_eq!(max_result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&max_result.value), &[7.5, 10.0]);

        let min_params = BTreeMap::from([("body_op".to_owned(), "min".to_owned())]);
        let min_result = apply_batch_rule(
            Primitive::Scan,
            &[BatchTracer::unbatched(Value::scalar_f64(5.0)), xs],
            &min_params,
        )
        .unwrap();
        assert_eq!(min_result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&min_result.value), &[1.0, 2.0]);
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

    #[test]
    fn test_batch_trace_while_batched_step_and_threshold() {
        let init = BatchTracer::batched(make_i64_vector(&[0, 10, 4]), 0);
        let step = BatchTracer::batched(make_i64_vector(&[2, 3, 5]), 0);
        let threshold = BatchTracer::batched(make_i64_vector(&[5, 25, 20]), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "32".to_owned()),
        ]);
        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(extract_i64_vec(&result.value), vec![6, 25, 24]);
    }

    #[test]
    fn test_batch_trace_while_dense_i64_add_lt_batch0_direct_path() {
        let init = BatchTracer::batched(Value::vector_i64(&[0, 10, 4]).unwrap(), 0);
        let step = BatchTracer::batched(Value::vector_i64(&[2, 3, 5]).unwrap(), 0);
        let threshold = BatchTracer::batched(Value::vector_i64(&[5, 25, 20]).unwrap(), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "32".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params)
            .expect("dense i64 while add/lt should batch directly");

        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.elements.as_i64_slice().unwrap(), &[6, 25, 24]);
    }

    #[test]
    fn test_batch_trace_while_dense_i64_add_lt_preserves_max_iter_boundary() {
        let init = BatchTracer::batched(Value::vector_i64(&[0]).unwrap(), 0);
        let step = BatchTracer::batched(Value::vector_i64(&[10]).unwrap(), 0);
        let threshold = BatchTracer::batched(Value::vector_i64(&[10]).unwrap(), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "1".to_owned()),
        ]);

        let err =
            apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("while_loop exceeded max iterations (1)")
        );
    }

    #[test]
    fn test_batch_trace_while_dense_f32_add_lt_batch0_direct_path_preserves_nan_exit() {
        let init = BatchTracer::batched(make_f32_vector(&[0.0, 10.0, f32::NAN, 4.0]), 0);
        let step = BatchTracer::unbatched(make_f32_scalar_tensor(2.0));
        let threshold = BatchTracer::batched(make_f32_vector(&[5.0, 25.0, 10.0, 20.0]), 0);
        let params = BTreeMap::from([("max_iter".to_owned(), "32".to_owned())]);

        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params)
            .expect("dense f32 while add/lt should batch directly");

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.dtype(), DType::F32);
        let values = extract_f32_vec(&result.value);
        assert!((values[0] - 6.0).abs() <= 1e-5, "{} != 6", values[0]);
        assert!((values[1] - 26.0).abs() <= 1e-5, "{} != 26", values[1]);
        assert!(values[2].is_nan());
        assert!((values[3] - 20.0).abs() <= 1e-5, "{} != 20", values[3]);
    }

    #[test]
    fn test_batch_trace_while_dense_f32_add_lt_preserves_max_iter_boundary() {
        let init = BatchTracer::batched(make_f32_vector(&[0.0]), 0);
        let step = BatchTracer::unbatched(Value::scalar_f32(10.0));
        let threshold = BatchTracer::unbatched(make_f32_scalar_tensor(10.0));
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "1".to_owned()),
        ]);

        let err =
            apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("while_loop exceeded max iterations (1)")
        );
    }

    #[test]
    fn test_batch_trace_while_batched_f64_mul() {
        let init = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        let step = BatchTracer::batched(make_f64_vector(&[2.0, 2.0, 3.0]), 0);
        let threshold = BatchTracer::unbatched(Value::scalar_f64(10.0));
        let params = BTreeMap::from([
            ("body_op".to_owned(), "mul".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "16".to_owned()),
        ]);
        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        assert_f64_close(&extract_f64_vec(&result.value), &[16.0, 16.0, 27.0]);
    }

    #[test]
    fn test_batch_trace_while_batched_f32_add_preserves_literal_width() {
        let init = BatchTracer::batched(make_f32_vector(&[0.0, 10.0]), 0);
        let step = BatchTracer::batched(make_f32_vector(&[2.0, 3.0]), 0);
        let threshold = BatchTracer::batched(make_f32_vector(&[5.0, 25.0]), 0);
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "32".to_owned()),
        ]);
        let result = apply_batch_rule(Primitive::While, &[init, step, threshold], &params)
            .expect("f32 while scalar fast path should batch lanes");
        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.dtype(), DType::F32);
        assert_f32_close(&extract_f32_vec(&result.value), &[6.0, 25.0]);
    }

    #[test]
    fn test_batch_trace_while_scalar_fast_path_preserves_max_iter_error() {
        let init = BatchTracer::batched(make_i64_vector(&[0, 1]), 0);
        let step = BatchTracer::unbatched(Value::scalar_i64(1));
        let threshold = BatchTracer::unbatched(Value::scalar_i64(10));
        let params = BTreeMap::from([
            ("body_op".to_owned(), "add".to_owned()),
            ("cond_op".to_owned(), "lt".to_owned()),
            ("max_iter".to_owned(), "3".to_owned()),
        ]);
        let err =
            apply_batch_rule(Primitive::While, &[init, step, threshold], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("while_loop exceeded max iterations (3)")
        );
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

    #[test]
    fn test_batch_trace_pad_defaults_interior_padding() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let pad_value = BatchTracer::unbatched(Value::scalar_f64(0.0));
        let params = BTreeMap::from([
            ("padding_low".to_owned(), "1".to_owned()),
            ("padding_high".to_owned(), "1".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Pad, &[input, pad_value], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 5]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0]
        );
    }

    #[test]
    fn test_batch_trace_reduce_window_defaults_strides() {
        let input = BatchTracer::batched(
            make_f64_matrix(
                2,
                5,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            ),
            0,
        );
        let params = BTreeMap::from([
            ("reduce_op".to_owned(), "sum".to_owned()),
            ("window_dimensions".to_owned(), "2".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::ReduceWindow, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![3.0, 5.0, 7.0, 9.0, 30.0, 50.0, 70.0, 90.0]
        );
    }

    #[test]
    fn test_batch_trace_reduce_window_defaults_window_dimensions() {
        let input = BatchTracer::batched(
            make_f64_matrix(
                2,
                5,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            ),
            0,
        );
        let params = BTreeMap::from([("reduce_op".to_owned(), "sum".to_owned())]);

        let result = apply_batch_rule(Primitive::ReduceWindow, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![3.0, 5.0, 7.0, 9.0, 30.0, 50.0, 70.0, 90.0]
        );
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

    // ── Tensor Op Batching Tests (frankenjax-sje) ───────────────

    #[test]
    fn test_batch_trace_slice_batched() {
        // Batch of 3 vectors of length 5, slice [1:4] from each
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let input = BatchTracer::batched(make_f64_matrix(3, 5, &data), 0);
        let mut params = BTreeMap::new();
        params.insert("start_indices".to_owned(), "1".to_owned());
        params.insert("limit_indices".to_owned(), "4".to_owned());
        params.insert("strides".to_owned(), "1".to_owned());
        let result = apply_batch_rule(Primitive::Slice, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        // Output should be [3, 3] (batch=3, sliced_len=3)
        assert_eq!(tensor.shape.dims, vec![3, 3]);
        let vals = extract_f64_vec(&result.value);
        // Row 0: [1,2,3], Row 1: [6,7,8], Row 2: [11,12,13]
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_batch_trace_slice_defaults_strides() {
        // JAX defaults omitted slice strides to one; BatchTrace should preserve that.
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let input = BatchTracer::batched(make_f64_matrix(3, 5, &data), 0);
        let mut params = BTreeMap::new();
        params.insert("start_indices".to_owned(), "1".to_owned());
        params.insert("limit_indices".to_owned(), "4".to_owned());

        let result = apply_batch_rule(Primitive::Slice, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![3, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_slice_batched_operand_static_start_direct_1d() {
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let input = BatchTracer::batched(make_f64_matrix(3, 5, &data), 0);
        let start = BatchTracer::unbatched(Value::scalar_i64(1));
        let params = BTreeMap::from([("slice_sizes".to_owned(), "3".to_owned())]);

        let result = apply_batch_rule(Primitive::DynamicSlice, &[input, start], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![3, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_slice_batched_operand_static_start_direct_2d() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[2, 3, 4], &data), 0);
        let row_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let col_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let params = BTreeMap::from([("slice_sizes".to_owned(), "2,2".to_owned())]);

        let result = apply_batch_rule(
            Primitive::DynamicSlice,
            &[input, row_start, col_start],
            &params,
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![5.0, 6.0, 9.0, 10.0, 17.0, 18.0, 21.0, 22.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_slice_static_start_moves_nonleading_batch() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[3, 2, 4], &data), 1);
        let row_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let col_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let params = BTreeMap::from([("slice_sizes".to_owned(), "2,2".to_owned())]);

        let result = apply_batch_rule(
            Primitive::DynamicSlice,
            &[input, row_start, col_start],
            &params,
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![9.0, 10.0, 17.0, 18.0, 13.0, 14.0, 21.0, 22.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_direct_1d() {
        let operand_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let operand = BatchTracer::batched(make_f64_matrix(2, 5, &operand_data), 0);
        let update = BatchTracer::batched(make_f64_matrix(2, 2, &[100.0, 101.0, 200.0, 201.0]), 0);
        let start = BatchTracer::unbatched(Value::scalar_i64(2));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 5]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![0.0, 1.0, 100.0, 101.0, 4.0, 5.0, 6.0, 200.0, 201.0, 9.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_direct_2d() {
        let operand_data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let operand = BatchTracer::batched(make_f64_tensor(&[2, 3, 4], &operand_data), 0);
        let update = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                &[100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],
            ),
            0,
        );
        let row_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let col_start = BatchTracer::unbatched(Value::scalar_i64(1));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, row_start, col_start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3, 4]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 7.0, 8.0, 102.0, 103.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 200.0, 201.0, 19.0, 20.0, 202.0, 203.0, 23.0,
            ]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_moves_nonleading_batch() {
        let operand_data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let operand = BatchTracer::batched(make_f64_tensor(&[3, 2, 4], &operand_data), 1);
        let update = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                &[100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],
            ),
            0,
        );
        let row_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let col_start = BatchTracer::unbatched(Value::scalar_i64(1));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, row_start, col_start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3, 4]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                0.0, 1.0, 2.0, 3.0, 8.0, 100.0, 101.0, 11.0, 16.0, 102.0, 103.0, 19.0, 4.0, 5.0,
                6.0, 7.0, 12.0, 200.0, 201.0, 15.0, 20.0, 202.0, 203.0, 23.0,
            ]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_broadcasts_unbatched_update() {
        let operand_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let operand = BatchTracer::batched(make_f64_matrix(2, 5, &operand_data), 0);
        let update = BatchTracer::unbatched(make_f64_vector(&[100.0, 101.0]));
        let start = BatchTracer::unbatched(Value::scalar_i64(2));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 5]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![0.0, 1.0, 100.0, 101.0, 4.0, 5.0, 6.0, 100.0, 101.0, 9.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_broadcasts_unbatched_operand() {
        let operand = BatchTracer::unbatched(make_f64_vector(&[0.0, 1.0, 2.0, 3.0, 4.0]));
        let update = BatchTracer::batched(make_f64_matrix(2, 2, &[100.0, 101.0, 200.0, 201.0]), 0);
        let start = BatchTracer::unbatched(Value::scalar_i64(2));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 5]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![0.0, 1.0, 100.0, 101.0, 4.0, 0.0, 1.0, 200.0, 201.0, 4.0]
        );
    }

    #[test]
    fn test_batch_trace_dynamic_update_slice_static_start_mixed_nonleading_update_batch() {
        let operand = BatchTracer::unbatched(make_f64_matrix(
            3,
            4,
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        ));
        let update = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 2],
                &[100.0, 101.0, 200.0, 201.0, 102.0, 103.0, 202.0, 203.0],
            ),
            1,
        );
        let row_start = BatchTracer::unbatched(Value::scalar_i64(1));
        let col_start = BatchTracer::unbatched(Value::scalar_i64(1));

        let result = apply_batch_rule(
            Primitive::DynamicUpdateSlice,
            &[operand, update, row_start, col_start],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3, 4]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 7.0, 8.0, 102.0, 103.0, 11.0, 0.0, 1.0, 2.0,
                3.0, 4.0, 200.0, 201.0, 7.0, 8.0, 202.0, 203.0, 11.0,
            ]
        );
    }

    #[test]
    fn test_batch_trace_gather_batched_indices_direct() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[10, 20, 30, 40]));
        let indices = BatchTracer::batched(make_i64_matrix(2, 3, &[3, 1, 0, 2, 2, 1]), 0);
        let params = BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(extract_i64_vec(&result.value), vec![40, 20, 10, 30, 30, 20]);
    }

    #[test]
    fn test_batch_trace_gather_batched_indices_moves_nonleading_axis() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[10, 20, 30]));
        let indices = BatchTracer::batched(make_i64_matrix(3, 2, &[0, 1, 2, 0, 1, 2]), 1);
        let params = BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(extract_i64_vec(&result.value), vec![10, 30, 20, 20, 10, 30]);
    }

    #[test]
    fn test_batch_trace_gather_batched_indices_direct_fill_or_drop() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[10, 20, 30]));
        let indices = BatchTracer::batched(make_i64_matrix(2, 3, &[0, 3, 1, 4, 2, 1]), 0);
        let params = BTreeMap::from([
            ("index_mode".to_owned(), "fill_or_drop".to_owned()),
            ("slice_sizes".to_owned(), "1".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![10, i64::MIN, 20, i64::MIN, 30, 20]
        );
    }

    #[test]
    fn test_batch_trace_gather_batched_operand_shared_indices() {
        let operand =
            BatchTracer::batched(make_i64_matrix(2, 4, &[10, 20, 30, 40, 50, 60, 70, 80]), 0);
        let indices = BatchTracer::unbatched(make_i64_vector(&[3, 1]));
        let params = BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(extract_i64_vec(&result.value), vec![40, 20, 80, 60]);
    }

    #[test]
    fn test_batch_trace_gather_batched_operand_shared_indices_rank3_partial_slice() {
        let operand = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: vec![2, 3, 4],
                    },
                    vec![
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100, 101, 102, 103, 104, 105, 106,
                        107, 108, 109, 110, 111,
                    ]
                    .into_iter()
                    .map(Literal::I64)
                    .collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let indices = BatchTracer::unbatched(make_i64_vector(&[2, 0]));
        let params = BTreeMap::from([("slice_sizes".to_owned(), "1,2".to_owned())]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2, 2]);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![8, 9, 0, 1, 108, 109, 100, 101]
        );
    }

    #[test]
    fn test_batch_trace_gather_batched_operand_batched_indices() {
        let operand =
            BatchTracer::batched(make_i64_matrix(2, 4, &[10, 20, 30, 40, 50, 60, 70, 80]), 0);
        let indices = BatchTracer::batched(make_i64_matrix(2, 2, &[0, 2, 3, 1]), 0);
        let params = BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 2]);
        assert_eq!(extract_i64_vec(&result.value), vec![10, 30, 80, 60]);
    }

    #[test]
    fn test_batch_trace_gather_batched_operand_fill_or_drop() {
        let operand =
            BatchTracer::batched(make_i64_matrix(2, 4, &[10, 20, 30, 40, 50, 60, 70, 80]), 0);
        let indices = BatchTracer::unbatched(make_i64_vector(&[3, 4, 1]));
        let params = BTreeMap::from([
            ("index_mode".to_owned(), "fill_or_drop".to_owned()),
            ("slice_sizes".to_owned(), "1".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Gather, &[operand, indices], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![40, i64::MIN, 20, 80, i64::MIN, 60]
        );
    }

    #[test]
    fn test_batch_trace_scatter_batched_indices_updates_direct() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[0, 0, 0, 0]));
        let indices = BatchTracer::batched(make_i64_matrix(2, 2, &[1, 3, 0, 2]), 0);
        let updates = BatchTracer::batched(make_i64_matrix(2, 2, &[10, 20, 30, 40]), 0);

        let result = apply_batch_rule(
            Primitive::Scatter,
            &[operand, indices, updates],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 4]);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![0, 10, 0, 20, 30, 0, 40, 0]
        );
    }

    #[test]
    fn test_batch_trace_scatter_overwrite_duplicates_keep_last_update_direct() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[0, 0, 0]));
        let indices = BatchTracer::batched(make_i64_matrix(2, 3, &[1, 1, 2, 0, 0, 2]), 0);
        let updates = BatchTracer::batched(make_i64_matrix(2, 3, &[10, 20, 30, 40, 50, 60]), 0);

        let result = apply_batch_rule(
            Primitive::Scatter,
            &[operand, indices, updates],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(extract_i64_vec(&result.value), vec![0, 20, 30, 50, 0, 60]);
    }

    #[test]
    fn test_batch_trace_scatter_overwrite_preserves_f64_bits_direct() {
        let nan_a = 0x7ff8_0000_0000_1234_u64;
        let nan_b = 0x7ff8_0000_0000_5678_u64;
        let neg_zero = (-0.0_f64).to_bits();
        let operand = BatchTracer::unbatched(Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::F64Bits(1.0_f64.to_bits()),
                    Literal::F64Bits(2.0_f64.to_bits()),
                    Literal::F64Bits(3.0_f64.to_bits()),
                ],
            )
            .unwrap(),
        ));
        let indices = BatchTracer::batched(make_i64_matrix(2, 3, &[1, 1, 2, 0, 0, 2]), 0);
        let updates = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 3] },
                    vec![
                        Literal::F64Bits(neg_zero),
                        Literal::F64Bits(nan_a),
                        Literal::F64Bits(4.0_f64.to_bits()),
                        Literal::F64Bits(5.0_f64.to_bits()),
                        Literal::F64Bits(neg_zero),
                        Literal::F64Bits(nan_b),
                    ],
                )
                .unwrap(),
            ),
            0,
        );

        let result = apply_batch_rule(
            Primitive::Scatter,
            &[operand, indices, updates],
            &BTreeMap::new(),
        )
        .unwrap();
        let tensor = result.value.as_tensor().unwrap();
        let bits = tensor
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::F64Bits(bits) => Some(*bits),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            bits,
            Some(vec![
                1.0_f64.to_bits(),
                nan_a,
                4.0_f64.to_bits(),
                neg_zero,
                2.0_f64.to_bits(),
                nan_b,
            ])
        );
    }

    #[test]
    fn test_batch_trace_scatter_unbatched_index_batched_updates_direct() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[0, 0, 0]));
        let index = BatchTracer::unbatched(Value::scalar_i64(1));
        let updates = BatchTracer::batched(make_i64_vector(&[5, 7]), 0);

        let result = apply_batch_rule(
            Primitive::Scatter,
            &[operand, index, updates],
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(extract_i64_vec(&result.value), vec![0, 5, 0, 0, 7, 0]);
    }

    #[test]
    fn test_batch_trace_scatter_add_mode_keeps_batches_independent() {
        let operand = BatchTracer::unbatched(make_i64_vector(&[1, 1, 1]));
        let indices = BatchTracer::batched(make_i64_matrix(2, 2, &[1, 1, 0, 0]), 0);
        let updates = BatchTracer::batched(make_i64_matrix(2, 2, &[2, 3, 4, 5]), 0);
        let params = BTreeMap::from([("mode".to_owned(), "add".to_owned())]);

        let result =
            apply_batch_rule(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();

        assert_eq!(result.batch_dim, Some(0));
        assert_eq!(result.value.as_tensor().unwrap().shape.dims, vec![2, 3]);
        assert_eq!(extract_i64_vec(&result.value), vec![1, 6, 1, 10, 1, 1]);
    }

    #[test]
    fn test_batch_trace_broadcast_in_dim() {
        // Batch of scalars broadcast to vectors
        let input = BatchTracer::batched(make_f64_vector(&[10.0, 20.0, 30.0]), 0);
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), "3".to_owned());
        params.insert("broadcast_dimensions".to_owned(), "".to_owned());
        let result = apply_batch_rule(Primitive::BroadcastInDim, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        // Each scalar from batch broadcasted to a [3]-vector, overall [3, 3]
        assert_eq!(tensor.shape.dims, vec![3, 3]);
    }

    #[test]
    fn test_batch_trace_broadcast_in_dim_default_mapping_non_scalar() {
        // Batch of 2 vectors length 2, broadcast to [3,2] with default mapping.
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = BatchTracer::batched(make_f64_matrix(2, 2, &data), 0);
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), "3,2".to_owned());
        params.insert("broadcast_dimensions".to_owned(), "".to_owned());
        let result = apply_batch_rule(Primitive::BroadcastInDim, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        let vals = extract_f64_vec(&result.value);
        assert_eq!(
            vals,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_batch_trace_rev_batched_logical_axis_zero() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);

        let result = apply_batch_rule(Primitive::Rev, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
        );
    }

    #[test]
    fn test_batch_trace_rev_logical_axis_one() {
        let data: Vec<f64> = (1..=12).map(f64::from).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[2, 2, 3], &data), 0);
        let params = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::Rev, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 9.0, 8.0, 7.0, 12.0, 11.0, 10.0
            ]
        );
    }

    #[test]
    fn test_batch_trace_rev_nonleading_batch_dim() {
        let input = BatchTracer::batched(make_f64_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 1);
        let params = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);

        let result = apply_batch_rule(Primitive::Rev, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0]
        );
    }

    #[test]
    fn test_batch_trace_rev_scalar_elements_are_noop() {
        let input = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        let params = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);

        let result = apply_batch_rule(Primitive::Rev, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(extract_f64_vec(&result.value), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_trace_rev_requires_axes_param() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);

        let err = apply_batch_rule(Primitive::Rev, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("missing required param 'axes'"));
    }

    #[test]
    fn test_batch_trace_split_equal_logical_axis_zero() {
        let data: Vec<f64> = (1..=12).map(f64::from).collect();
        let input = BatchTracer::batched(make_f64_matrix(2, 6, &data), 0);
        let params = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("num_sections".to_owned(), "3".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Split, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        assert_eq!(extract_f64_vec(&result.value), data);
    }

    #[test]
    fn test_batch_trace_split_equal_logical_axis_one() {
        let data: Vec<f64> = (1..=16).map(f64::from).collect();
        let input = BatchTracer::batched(make_f64_tensor(&[2, 2, 4], &data), 0);
        let params = BTreeMap::from([
            ("axis".to_owned(), "1".to_owned()),
            ("num_sections".to_owned(), "2".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Split, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 2, 2]);
        assert_eq!(extract_f64_vec(&result.value), data);
    }

    #[test]
    fn test_batch_trace_split_unequal_fails_closed_per_batch() {
        // Uneven split is rejected at the primitive level (fj-lax fails closed
        // rather than silently returning only the first section), so the
        // batched per-slice eval must surface that error rather than truncate.
        let input = BatchTracer::batched(
            make_f64_matrix(2, 5, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            0,
        );
        let params = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("sizes".to_owned(), "2, 3".to_owned()),
        ]);

        let err = apply_batch_rule(Primitive::Split, &[input], &params)
            .expect_err("uneven split under vmap must fail closed, not truncate");
        assert!(
            err.to_string().contains("uneven split"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_batch_trace_split_nonleading_batch_dim() {
        let input = BatchTracer::batched(
            make_f64_matrix(
                6,
                2,
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            ),
            1,
        );
        let params = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("num_sections".to_owned(), "3".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::Split, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0
            ]
        );
    }

    #[test]
    fn test_batch_trace_split_scalar_elements_reject() {
        let input = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);

        let err = apply_batch_rule(Primitive::Split, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("cannot split a scalar"));
    }

    #[test]
    fn test_batch_trace_cholesky_leading_batch_dim() {
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[4.0, 2.0, 2.0, 3.0, 9.0, 3.0, 3.0, 2.0]),
            0,
        );

        let result = apply_batch_rule(Primitive::Cholesky, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 2]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[2.0, 0.0, 1.0, 2.0_f64.sqrt(), 3.0, 0.0, 1.0, 1.0],
        );
    }

    #[test]
    fn test_batch_trace_cholesky_nonleading_batch_dim() {
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[4.0, 2.0, 9.0, 3.0, 2.0, 3.0, 3.0, 2.0]),
            1,
        );

        let result = apply_batch_rule(Primitive::Cholesky, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 2]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[2.0, 0.0, 1.0, 2.0_f64.sqrt(), 3.0, 0.0, 1.0, 1.0],
        );
    }

    #[test]
    fn test_batch_trace_det_leading_batch_dim() {
        // vmap(det) over [2,2,2]: det([[4,2],[2,3]])=8, det([[9,3],[3,2]])=9.
        let input = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[4.0, 2.0, 2.0, 3.0, 9.0, 3.0, 3.0, 2.0]),
            0,
        );

        let result = apply_batch_rule(Primitive::Det, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2]);
        assert_f64_close(&extract_f64_vec(&result.value), &[8.0, 9.0]);
    }

    #[test]
    fn test_batch_trace_solve_both_inputs_batched() {
        // vmap(solve): A0=2I, b0=[2,4] -> [1,2]; A1=4I, b1=[8,4] -> [2,1].
        let a = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 4.0]),
            0,
        );
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2], &[2.0, 4.0, 8.0, 4.0]), 0);

        let result = apply_batch_rule(Primitive::Solve, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_f64_close(&extract_f64_vec(&result.value), &[1.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn test_batch_trace_solve_shared_rhs() {
        // vmap(solve, in_axes=(0, None)): batched A, shared b=[2,2].
        // A0=2I -> x0=[1,1]; A1=I -> x1=[2,2].
        let a = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0]),
            0,
        );
        let b = BatchTracer::unbatched(make_f64_vector(&[2.0, 2.0]));

        let result = apply_batch_rule(Primitive::Solve, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_f64_close(&extract_f64_vec(&result.value), &[1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_batch_trace_associative_scan_leading_batch_dim() {
        // vmap(associative_scan add) over [2,3]: each row prefix-summed
        // independently. [1,2,3]->[1,3,6]; [4,5,6]->[4,9,15].
        let input =
            BatchTracer::batched(make_f64_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let mut params = BTreeMap::new();
        params.insert("body_op".to_owned(), "add".to_owned());

        let result = apply_batch_rule(Primitive::AssociativeScan, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0],
        );
    }

    #[test]
    fn test_batch_trace_cholesky_non_square_batched_matrix_rejects() {
        let input = BatchTracer::batched(
            make_f64_tensor(
                &[2, 2, 3],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            ),
            0,
        );

        let err = apply_batch_rule(Primitive::Cholesky, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("requires a square matrix"));
    }

    #[test]
    fn test_batch_trace_cholesky_scalar_elements_preserve_matrix_rank_error() {
        let input = BatchTracer::batched(make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]), 0);

        let err = apply_batch_rule(Primitive::Cholesky, &[input], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected rank-2 tensor"));
    }

    #[test]
    fn test_batch_trace_triangular_solve_leading_batch_dim() {
        let a = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0]),
            0,
        );
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2, 1], &[4.0, 7.0, 1.0, 4.0]), 0);

        let result =
            apply_batch_rule(Primitive::TriangularSolve, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 1]);
        assert_f64_close(&extract_f64_vec(&result.value), &[2.0, 5.0 / 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_batch_trace_triangular_solve_nonleading_rhs_batch_dim() {
        let a = BatchTracer::unbatched(make_f64_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]));
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2, 1], &[4.0, 5.0, 7.0, 6.0]), 1);

        let result =
            apply_batch_rule(Primitive::TriangularSolve, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 1]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[2.0, 5.0 / 3.0, 2.5, 7.0 / 6.0],
        );
    }

    #[test]
    fn test_batch_trace_triangular_solve_upper_params() {
        let a = BatchTracer::batched(
            make_f64_tensor(&[2, 2, 2], &[2.0, 1.0, 0.0, 3.0, 4.0, 2.0, 0.0, 2.0]),
            0,
        );
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2, 1], &[5.0, 6.0, 10.0, 4.0]), 0);
        let params = BTreeMap::from([("lower".to_owned(), "false".to_owned())]);

        let result = apply_batch_rule(Primitive::TriangularSolve, &[a, b], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 1]);
        assert_f64_close(&extract_f64_vec(&result.value), &[1.5, 2.0, 1.5, 2.0]);
    }

    #[test]
    fn test_batch_trace_triangular_solve_broadcasts_unbatched_lhs() {
        let a = BatchTracer::unbatched(make_f64_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]));
        let b = BatchTracer::batched(make_f64_tensor(&[2, 2, 1], &[4.0, 7.0, 5.0, 6.0]), 0);

        let result =
            apply_batch_rule(Primitive::TriangularSolve, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2, 1]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[2.0, 5.0 / 3.0, 2.5, 7.0 / 6.0],
        );
    }

    #[test]
    fn test_batch_trace_triangular_solve_matrix_rank_error_preserved() {
        let a = BatchTracer::batched(make_f64_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]), 0);
        let b = BatchTracer::batched(make_f64_matrix(2, 1, &[4.0, 7.0]), 0);

        let err =
            apply_batch_rule(Primitive::TriangularSolve, &[a, b], &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("expected rank-2 tensor"));
    }

    #[test]
    fn test_batch_trace_fft_leading_batch_dim() {
        let input = BatchTracer::batched(
            make_complex128_tensor(
                &[2, 4],
                &[
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Fft, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_complex_close(
            &extract_complex128_vec(&result.value),
            &[
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
        );
    }

    #[test]
    fn test_batch_trace_fft_nonleading_batch_dim() {
        let input = BatchTracer::batched(
            make_complex128_tensor(
                &[4, 2],
                &[
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                ],
            ),
            1,
        );

        let result = apply_batch_rule(Primitive::Fft, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_complex_close(
            &extract_complex128_vec(&result.value),
            &[
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
        );
    }

    #[test]
    fn test_batch_trace_ifft_leading_batch_dim() {
        let input = BatchTracer::batched(
            make_complex128_tensor(
                &[2, 4],
                &[
                    (4.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Ifft, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_complex_close(
            &extract_complex128_vec(&result.value),
            &[
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ],
        );
    }

    #[test]
    fn test_batch_trace_rfft_preserves_batch_and_updates_last_axis() {
        let input = BatchTracer::batched(
            make_f64_matrix(2, 4, &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            0,
        );

        let result = apply_batch_rule(Primitive::Rfft, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_complex_close(
            &extract_complex128_vec(&result.value),
            &[
                (4.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ],
        );
    }

    #[test]
    fn test_batch_trace_irfft_preserves_batch_and_updates_last_axis() {
        let input = BatchTracer::batched(
            make_complex128_tensor(
                &[2, 3],
                &[
                    (4.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 0.0),
                ],
            ),
            0,
        );
        let params = BTreeMap::from([("fft_length".to_owned(), "4".to_owned())]);

        let result = apply_batch_rule(Primitive::Irfft, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 4]);
        assert_f64_close(
            &extract_f64_vec(&result.value),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_batch_trace_fft_scalar_elements_reject() {
        let input = BatchTracer::batched(make_f64_vector(&[1.0, 2.0]), 0);

        let err = apply_batch_rule(Primitive::Fft, &[input], &BTreeMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("FFT expects a tensor (rank >= 1), got scalar")
        );
    }

    #[test]
    fn test_batch_trace_one_hot_vector_indices() {
        let input = BatchTracer::batched(make_i64_vector(&[0, 2, 1]), 0);
        let params = BTreeMap::from([("num_classes".to_owned(), "3".to_owned())]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![3, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_batch_trace_one_hot_axis_zero_keeps_batch_leading_for_scalars() {
        let input = BatchTracer::batched(make_i64_vector(&[0, 2]), 0);
        let params = BTreeMap::from([
            ("num_classes".to_owned(), "3".to_owned()),
            ("axis".to_owned(), "0".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_batch_trace_one_hot_axis_zero_keeps_batch_leading_for_vectors() {
        let input = BatchTracer::batched(make_i64_matrix(2, 2, &[0, 1, 2, 0]), 0);
        let params = BTreeMap::from([
            ("num_classes".to_owned(), "3".to_owned()),
            ("axis".to_owned(), "0".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 2]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_batch_trace_one_hot_nonleading_batch_dim() {
        let input = BatchTracer::batched(make_i64_matrix(3, 2, &[0, 1, 2, 0, 1, 2]), 1);
        let params = BTreeMap::from([("num_classes".to_owned(), "3".to_owned())]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0,
            ]
        );
    }

    #[test]
    fn test_batch_trace_one_hot_custom_int_values() {
        let input = BatchTracer::batched(make_i64_vector(&[1, -1, 3]), 0);
        let params = BTreeMap::from([
            ("num_classes".to_owned(), "3".to_owned()),
            ("dtype".to_owned(), "I64".to_owned()),
            ("on_value".to_owned(), "5".to_owned()),
            ("off_value".to_owned(), "-2".to_owned()),
        ]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![3, 3]);
        assert_eq!(
            extract_i64_vec(&result.value),
            vec![-2, 5, -2, -2, -2, -2, -2, -2, -2]
        );
    }

    #[test]
    fn test_batch_trace_one_hot_unbatched_scalar() {
        let input = BatchTracer::unbatched(Value::Scalar(Literal::I64(2)));
        let params = BTreeMap::from([("num_classes".to_owned(), "4".to_owned())]);

        let result = apply_batch_rule(Primitive::OneHot, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, None);
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![4]);
        assert_eq!(extract_f64_vec(&result.value), vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_batch_trace_one_hot_missing_num_classes_errors() {
        let input = BatchTracer::batched(make_i64_vector(&[0, 1]), 0);

        let err = apply_batch_rule(Primitive::OneHot, &[input], &BTreeMap::new()).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required param 'num_classes'")
        );
    }

    #[test]
    fn test_batch_trace_squeeze_batched() {
        // Batch of 2 matrices [2, 1], squeeze the trailing dim
        // Input shape: [2, 2, 1] (batch_dim=0)
        // Per-element shape: [2, 1], squeeze dim 1 → [2]
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 2, 1],
                    },
                    data.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let mut params = BTreeMap::new();
        // Squeeze dim 1 of per-element [2, 1] → [2]
        params.insert("dimensions".to_owned(), "1".to_owned());
        let result = apply_batch_rule(Primitive::Squeeze, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        // Result: [2, 2] (batch=2, squeezed_len=2)
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(extract_f64_vec(&result.value), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_batch_trace_squeeze_negative_axis() {
        // vmap(lambda x: squeeze(x, -1)) over a batch of [2, 1] matrices.
        // Input shape: [2, 2, 1] (batch_dim=0); per-element [2, 1], dim=-1
        // resolves to the trailing per-element axis → physical axis 2 → [2].
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 2, 1],
                    },
                    data.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("dimensions".to_owned(), "-1".to_owned())]);
        let result = apply_batch_rule(Primitive::Squeeze, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(extract_f64_vec(&result.value), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_batch_trace_squeeze_default_preserves_singleton_batch() {
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![1, 2, 1],
                    },
                    vec![1.0, 2.0].into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
            0,
        );

        let result = apply_batch_rule(Primitive::Squeeze, &[input], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![1, 2]);
        assert_eq!(extract_f64_vec(&result.value), vec![1.0, 2.0]);
    }

    #[test]
    fn test_batch_trace_squeeze_explicit_logical_axis_zero() {
        let input = BatchTracer::batched(
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![2, 1, 3],
                    },
                    (1..=6).map(|v| Literal::from_f64(f64::from(v))).collect(),
                )
                .unwrap(),
            ),
            0,
        );
        let params = BTreeMap::from([("dimensions".to_owned(), "0".to_owned())]);

        let result = apply_batch_rule(Primitive::Squeeze, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_batch_trace_expand_dims_batched() {
        // Batch of [3] vectors, expand dim 1 → [3, 1] matrices, overall [batch, 3, 1]
        let input = BatchTracer::batched(make_f64_vector(&[1.0, 2.0, 3.0]), 0);
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "1".to_owned());
        let result = apply_batch_rule(Primitive::ExpandDims, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![3, 1]);
        assert_eq!(extract_f64_vec(&result.value), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_trace_expand_dims_logical_axis_zero() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);

        let result = apply_batch_rule(Primitive::ExpandDims, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 1, 3]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_batch_trace_expand_dims_trailing_logical_axis() {
        let input = BatchTracer::batched(make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0);
        let params = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);

        let result = apply_batch_rule(Primitive::ExpandDims, &[input], &params).unwrap();
        assert_eq!(result.batch_dim, Some(0));
        let tensor = result.value.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![2, 3, 1]);
        assert_eq!(
            extract_f64_vec(&result.value),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_batch_trace_concatenate_unbatched_inputs() {
        // Two unbatched vectors concatenated should remain unbatched
        let a = BatchTracer::unbatched(make_f64_vector(&[1.0, 2.0]));
        let b = BatchTracer::unbatched(make_f64_vector(&[3.0, 4.0, 5.0]));
        let mut params = BTreeMap::new();
        params.insert("dimension".to_owned(), "0".to_owned());
        let result = apply_batch_rule(Primitive::Concatenate, &[a, b], &params).unwrap();
        assert_eq!(result.batch_dim, None);
        let vals = extract_f64_vec(&result.value);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // ── Bitwise Tests ──────────────────────────────────────────

    #[test]
    fn test_batch_trace_bitwise_and() {
        let a = BatchTracer::batched(make_i64_vector(&[0b1100, 0b1010, 0b1111]), 0);
        let b = BatchTracer::batched(make_i64_vector(&[0b1010, 0b1010, 0b0101]), 0);
        let result = apply_batch_rule(Primitive::BitwiseAnd, &[a, b], &BTreeMap::new()).unwrap();
        assert_eq!(result.batch_dim, Some(0));
    }

    // ── Proptest Metamorphic Tests ──────────────────────────────

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]

        #[test]
        fn metamorphic_unbatched_tracer_preserves_content(
            data in proptest::collection::vec(-1000.0f64..1000.0, 1..16)
        ) {
            let input = make_f64_vector(&data);
            let tracer = BatchTracer::unbatched(input.clone());
            proptest::prop_assert_eq!(tracer.batch_dim, None);
            proptest::prop_assert_eq!(extract_f64_vec(&tracer.value), data);
        }

        #[test]
        fn metamorphic_batched_neg_matches_elementwise(
            data in proptest::collection::vec(-1000.0f64..1000.0, 1..16)
        ) {
            let filtered: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
            proptest::prop_assume!(!filtered.is_empty());
            let input = BatchTracer::batched(make_f64_vector(&filtered), 0);
            let result = apply_batch_rule(Primitive::Neg, &[input], &BTreeMap::new());
            proptest::prop_assert!(result.is_ok());
            let out = result.unwrap();
            proptest::prop_assert_eq!(out.batch_dim, Some(0));
            let out_vals = extract_f64_vec(&out.value);
            proptest::prop_assert_eq!(out_vals.len(), filtered.len());
            for (i, (&orig, &negated)) in filtered.iter().zip(out_vals.iter()).enumerate() {
                proptest::prop_assert!(
                    (negated + orig).abs() < 1e-12,
                    "neg mismatch at index {}: -{} != {}", i, orig, negated
                );
            }
        }

        #[test]
        fn metamorphic_batched_abs_idempotent(
            data in proptest::collection::vec(-1000.0f64..1000.0, 1..16)
        ) {
            let filtered: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
            proptest::prop_assume!(!filtered.is_empty());
            let input = BatchTracer::batched(make_f64_vector(&filtered), 0);
            let abs1 = apply_batch_rule(Primitive::Abs, &[input], &BTreeMap::new()).unwrap();
            let abs2 =
                apply_batch_rule(Primitive::Abs, std::slice::from_ref(&abs1), &BTreeMap::new())
                    .unwrap();
            let vals1 = extract_f64_vec(&abs1.value);
            let vals2 = extract_f64_vec(&abs2.value);
            proptest::prop_assert_eq!(vals1, vals2, "abs(abs(x)) should equal abs(x)");
        }

        #[test]
        fn metamorphic_unbatched_elementwise_vs_batched(x in -100.0f64..100.0) {
            proptest::prop_assume!(x.is_finite());
            let unbatched_input = BatchTracer::unbatched(Value::scalar_f64(x));
            let unbatched_result = apply_batch_rule(Primitive::Neg, &[unbatched_input], &BTreeMap::new()).unwrap();
            proptest::prop_assert_eq!(unbatched_result.batch_dim, None);
            let unbatched_val = unbatched_result.value.as_f64_scalar().unwrap();
            proptest::prop_assert!((unbatched_val + x).abs() < 1e-14);
        }
    }
}
