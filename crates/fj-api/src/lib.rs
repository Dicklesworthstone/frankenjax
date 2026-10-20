#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod errors;
pub mod transforms;

pub use errors::ApiError;
pub use fj_ad::{
    AdError, JvpResult, LinearizedTape, VjpTape, build_grad_jaxpr, build_hessian_jaxpr,
    build_hvp_jaxpr, build_jvp_jaxpr, build_value_and_grad_jaxpr, clear_custom_derivative_rules,
    grad_jaxpr_with_cotangent, grad_to_jaxpr, hessian_to_jaxpr, hvp_to_jaxpr, jvp_to_jaxpr,
    linearize_jaxpr, record_vjp_tape, register_custom_jaxpr_jvp, register_custom_jaxpr_vjp,
    register_custom_jvp, register_custom_vjp, symbolic_hessian_jaxpr,
};
pub use fj_core::{
    DType, Literal, PrecisionPolicy, Shape, TransformCompositionProof, Value,
    current_precision_policy, set_precision_policy, with_precision_policy,
};
pub use fj_dispatch::{
    DispatchGovernanceSnapshot, DispatchResponse, dispatch_cancellation_token,
    dispatch_governance_stats, observe_dispatch_drift_likelihood_ratio, reset_dispatch_governance,
    set_dispatch_memory_budget,
};
pub use fj_egraph::{
    OptimizationConfig, canonicalize_jaxpr_fast, optimize_jaxpr, optimize_jaxpr_with_config,
};
pub use fj_ffi::{
    FfiBuffer, FfiCall, FfiError, FfiFnPtr, FfiRegistry, FfiTarget, buffer_to_value,
    global_ffi_registry, register_global_ffi_target, value_to_buffer,
};
pub use fj_interpreters::{
    CompiledJaxpr, InterpreterError, compile_jaxpr_for_repeated_eval, eval_jaxpr,
    eval_jaxpr_with_consts, eval_jaxpr_with_ffi,
    partial_eval::{
        PartialEvalError, PartialEvalResult, PartialVal, partial_eval_jaxpr,
        partial_eval_jaxpr_with_consts,
    },
    staging::{StagedProgram, StagingError, execute_staged, stage_jaxpr, stage_jaxpr_with_consts},
    with_ffi_registry,
};
pub use fj_ledger::EvidenceLedger;
pub use fj_runtime::{AdmissionRejection, CancellationToken, ExecutionRegime, MemoryBudget};
pub use transforms::{
    CheckpointWrapped, ComposedTransform, CustomJvpWrapped, CustomVjpWrapped, GradWrapped,
    HessianWrapped, JacobianWrapped, JitWrapped, LinearizeResult, LinearizedFunction, PmapWrapped,
    TransposedLinearFunction, ValueAndGradWrapped, VjpPullback, VjpResult, VmapWrapped,
};
pub use transforms::{
    ShapedArray, checkpoint, compose, custom_jvp, custom_vjp, eval_shape, eval_shape_avals,
    eval_shape_fn, grad, grad_fn, hessian, hessian_fn, hvp, hvp_fn, jacobian, jacobian_fn, jit,
    jit_fn, jvp, jvp_fn, linear_transpose, linearize, linearize_fn, optimize_fn, pmap, pmap_fn,
    trace_jaxpr_for_values, value_and_grad, value_and_grad_fn, vjp, vjp_fn, vmap, vmap_fn,
};

// Re-export make_jaxpr, ClosedJaxpr, and control-flow tracing API from fj-trace
pub use fj_trace::{
    ClosedJaxpr, TracerRef, make_jaxpr, make_jaxpr_fallible, trace_cond, trace_scan, trace_switch,
    trace_while_loop, with_pmap_trace_context,
};

/// Neural-network activations, normalizations, and loss helpers (`jax.nn` parity).
///
/// All functions on [`TracerRef`] lower into canonical `fj_core::Primitive` equations so they
/// compose cleanly with [`jit`], [`grad`], and [`vmap`]. Eager `f64` reference helpers are also
/// re-exported under [`nn::eager`].
pub mod nn {
    use fj_core::{DType, Literal};
    use fj_trace::{TraceError, TracerRef};

    /// Eager `f64` reference implementations from `fj_lax::nn`.
    pub mod eager {
        pub use fj_lax::nn::*;
    }

    fn lit_f64(val: f64) -> Result<TracerRef, TraceError> {
        TracerRef::literal(Literal::from_f64(val))
    }

    fn normalize_axis(rank: usize, axis: isize) -> Result<usize, TraceError> {
        let rank_isize = rank as isize;
        let normalized = if axis < 0 { rank_isize + axis } else { axis };
        if normalized < 0 || normalized >= rank_isize {
            return Err(TraceError::ShapeInferenceFailed {
                primitive: fj_core::Primitive::ReduceSum,
                detail: format!("nn axis {axis} out of bounds for rank {rank}"),
            });
        }
        Ok(normalized as usize)
    }

    fn reduce_keepdims_sum(x: &TracerRef, axis: usize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        let sum = x.reduce_sum(Some(&[axis as i64]))?;
        if rank == 0 {
            return Ok(sum);
        }
        let mut kept_dims: Vec<u32> = x.aval().shape.dims.clone();
        kept_dims[axis] = 1;
        let broadcast_dims: Vec<usize> = (0..rank).filter(|&d| d != axis).collect();
        sum.broadcast_in_dim(&kept_dims, &broadcast_dims)
    }

    fn reduce_keepdims_max(x: &TracerRef, axis: usize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        let max_val = x.reduce_max(Some(&[axis as i64]))?;
        if rank == 0 {
            return Ok(max_val);
        }
        let mut kept_dims: Vec<u32> = x.aval().shape.dims.clone();
        kept_dims[axis] = 1;
        let broadcast_dims: Vec<usize> = (0..rank).filter(|&d| d != axis).collect();
        max_val.broadcast_in_dim(&kept_dims, &broadcast_dims)
    }

    /// Rectified Linear Unit: `max(x, 0)`.
    pub fn relu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.max(&lit_f64(0.0)?)
    }

    /// Leaky ReLU: `select(x >= 0, x, negative_slope * x)`.
    pub fn leaky_relu(x: &TracerRef, negative_slope: f64) -> Result<TracerRef, TraceError> {
        let zero = lit_f64(0.0)?;
        let cond = x.ge(&zero)?;
        let scaled = x.mul_op(&lit_f64(negative_slope)?)?;
        cond.select(x, &scaled)
    }

    /// ReLU6: `clamp(0, x, 6)`.
    pub fn relu6(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let zero = lit_f64(0.0)?;
        let six = lit_f64(6.0)?;
        x.clamp(&zero, &six)
    }

    /// Exponential Linear Unit: `select(x > 0, x, alpha * expm1(x))`.
    pub fn elu(x: &TracerRef, alpha: f64) -> Result<TracerRef, TraceError> {
        let zero = lit_f64(0.0)?;
        let cond = x.gt(&zero)?;
        let neg_branch = x.expm1()?.mul_op(&lit_f64(alpha)?)?;
        cond.select(x, &neg_branch)
    }

    /// Scaled Exponential Linear Unit (SELU).
    pub fn selu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        const SCALE: f64 = 1.050_700_987_355_480_5;
        elu(x, ALPHA)?.mul_op(&lit_f64(SCALE)?)
    }

    /// Continuously Differentiable ELU: `max(0, x) + min(0, alpha * expm1(x / alpha))`.
    pub fn celu(x: &TracerRef, alpha: f64) -> Result<TracerRef, TraceError> {
        let zero = lit_f64(0.0)?;
        let alpha_lit = lit_f64(alpha)?;
        let pos = x.max(&zero)?;
        let neg = x
            .div_op(&alpha_lit)?
            .expm1()?
            .mul_op(&alpha_lit)?
            .min(&zero)?;
        pos.add_op(&neg)
    }

    /// Gaussian Error Linear Unit (GELU).
    ///
    /// When `approximate` is `false`, computes `0.5 * x * (1 + erf(x / sqrt(2)))`.
    /// When `approximate` is `true`, computes `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    pub fn gelu(x: &TracerRef, approximate: bool) -> Result<TracerRef, TraceError> {
        let half = lit_f64(0.5)?;
        let one = lit_f64(1.0)?;
        if approximate {
            let sqrt_2_over_pi = lit_f64((2.0_f64 / std::f64::consts::PI).sqrt())?;
            let coeff = lit_f64(0.044_715)?;
            let x2 = x.square()?;
            let x3 = x2.mul_op(x)?;
            let inner = x.add_op(&x3.mul_op(&coeff)?)?.mul_op(&sqrt_2_over_pi)?;
            x.mul_op(&half)?.mul_op(&one.add_op(&inner.tanh()?)?)
        } else {
            let inv_sqrt2 = lit_f64(std::f64::consts::FRAC_1_SQRT_2)?;
            let erf_part = x.mul_op(&inv_sqrt2)?.erf()?;
            x.mul_op(&half)?.mul_op(&one.add_op(&erf_part)?)
        }
    }

    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.logistic()
    }

    /// Sigmoid Linear Unit (SiLU / Swish): `x * sigmoid(x)`.
    pub fn silu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.mul_op(&sigmoid(x)?)
    }

    /// Alias for [`silu`].
    pub fn swish(x: &TracerRef) -> Result<TracerRef, TraceError> {
        silu(x)
    }

    /// Softplus: `log(1 + exp(x))`.
    pub fn softplus(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.exp()?.log1p()
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.mul_op(&softplus(x)?.tanh()?)
    }

    /// Softsign: `x / (1 + |x|)`.
    pub fn soft_sign(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let denom = x.abs()?.add_op(&lit_f64(1.0)?)?;
        x.div_op(&denom)
    }

    /// Hard tanh: `clamp(-1, x, 1)`.
    pub fn hard_tanh(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.clamp(&lit_f64(-1.0)?, &lit_f64(1.0)?)
    }

    /// Hard sigmoid: `relu6(x + 3) / 6`.
    pub fn hard_sigmoid(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let shifted = x.add_op(&lit_f64(3.0)?)?;
        relu6(&shifted)?.div_op(&lit_f64(6.0)?)
    }

    /// Hard swish: `x * hard_sigmoid(x)`.
    pub fn hard_swish(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.mul_op(&hard_sigmoid(x)?)
    }

    /// Alias for [`hard_swish`].
    pub fn hard_silu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        hard_swish(x)
    }

    /// Log-sigmoid: `-softplus(-x)`.
    pub fn log_sigmoid(x: &TracerRef) -> Result<TracerRef, TraceError> {
        softplus(&x.neg_op()?)?.neg_op()
    }

    /// SwiGLU gated activation: `silu(gate) * up`.
    pub fn swiglu(gate: &TracerRef, up: &TracerRef) -> Result<TracerRef, TraceError> {
        silu(gate)?.mul_op(up)
    }

    /// GeGLU gated activation: `gelu(gate, approximate) * up`.
    pub fn geglu(
        gate: &TracerRef,
        up: &TracerRef,
        approximate: bool,
    ) -> Result<TracerRef, TraceError> {
        gelu(gate, approximate)?.mul_op(up)
    }

    /// Numerically stable log-sum-exp along `axis` (defaulting to the last axis via `-1`).
    pub fn logsumexp(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return Ok(x.clone());
        }
        let ax = normalize_axis(rank, axis)?;
        let max_unreduced = x.reduce_max(Some(&[ax as i64]))?;
        let max_keep = reduce_keepdims_max(x, ax)?;
        let shifted_exp = x.sub_op(&max_keep)?.exp()?;
        let sum_exp = shifted_exp.reduce_sum(Some(&[ax as i64]))?;
        max_unreduced.add_op(&sum_exp.log()?)
    }

    /// Numerically stable softmax along `axis` (use `-1` for the last axis).
    pub fn softmax(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return lit_f64(1.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let max_keep = reduce_keepdims_max(x, ax)?;
        let shifted_exp = x.sub_op(&max_keep)?.exp()?;
        let sum_keep = reduce_keepdims_sum(&shifted_exp, ax)?;
        shifted_exp.div_op(&sum_keep)
    }

    /// Numerically stable log-softmax along `axis` (use `-1` for the last axis).
    pub fn log_softmax(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return lit_f64(0.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let max_keep = reduce_keepdims_max(x, ax)?;
        let shifted = x.sub_op(&max_keep)?;
        let sum_exp_keep = reduce_keepdims_sum(&shifted.exp()?, ax)?;
        shifted.sub_op(&sum_exp_keep.log()?)
    }

    /// Standardizes `x` along `axis` to zero mean and unit variance: `(x - mean) / sqrt(var + eps)`.
    pub fn standardize(x: &TracerRef, axis: isize, eps: f64) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return lit_f64(0.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let n = x.aval().shape.dims[ax] as f64;
        let inv_n = lit_f64(1.0 / n)?;
        let mean_keep = reduce_keepdims_sum(x, ax)?.mul_op(&inv_n)?;
        let centered = x.sub_op(&mean_keep)?;
        let var_keep = reduce_keepdims_sum(&centered.square()?, ax)?.mul_op(&inv_n)?;
        let std_keep = var_keep.add_op(&lit_f64(eps)?)?.sqrt()?;
        centered.div_op(&std_keep)
    }

    /// Layer normalization along `axis` with optional affine `weight` and `bias`.
    pub fn layer_norm(
        x: &TracerRef,
        axis: isize,
        eps: f64,
        weight: Option<&TracerRef>,
        bias: Option<&TracerRef>,
    ) -> Result<TracerRef, TraceError> {
        let mut normed = standardize(x, axis, eps)?;
        if let Some(w) = weight {
            normed = normed.mul_op(w)?;
        }
        if let Some(b) = bias {
            normed = normed.add_op(b)?;
        }
        Ok(normed)
    }

    /// Root-Mean-Square Layer Normalization (RMSNorm) along `axis`.
    pub fn rms_norm(
        x: &TracerRef,
        axis: isize,
        eps: f64,
        weight: Option<&TracerRef>,
    ) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        let ax = normalize_axis(rank, axis)?;
        let n = x.aval().shape.dims[ax] as f64;
        let inv_n = lit_f64(1.0 / n)?;
        let ms_keep = reduce_keepdims_sum(&x.square()?, ax)?.mul_op(&inv_n)?;
        let inv_rms = ms_keep.add_op(&lit_f64(eps)?)?.rsqrt()?;
        let mut out = x.mul_op(&inv_rms)?;
        if let Some(w) = weight {
            out = out.mul_op(w)?;
        }
        Ok(out)
    }

    /// One-hot encoding wrapper on [`TracerRef`].
    pub fn one_hot(
        indices: &TracerRef,
        num_classes: u32,
        dtype: DType,
    ) -> Result<TracerRef, TraceError> {
        indices.one_hot(num_classes, dtype, None)
    }

    /// Half L2 loss: `0.5 * sum(x^2)`.
    pub fn l2_loss(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let sq = x.square()?;
        let sum = sq.reduce_sum(None)?;
        sum.mul_op(&lit_f64(0.5)?)
    }
}

/// Deterministic counter-based PRNG and distributions (`jax.random` parity).
///
/// Wraps `fj_lax::threefry` with ergonomic [`Value`]-producing tensor samplers shaped to [`Shape`].
pub mod random {
    use crate::ApiError;
    use fj_core::{DType, Literal, Shape, TensorValue, Value};

    pub use fj_lax::threefry::{
        CategoricalError, ChoiceError, PRNGKey, random_fold_in as fold_in, random_key as key,
        random_split as split, random_split_n as split_n, threefry2x32,
    };

    fn shape_count(shape: &Shape) -> Result<usize, ApiError> {
        shape
            .element_count()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| ApiError::EvalError {
                detail: format!("random shape {:?} element count overflow", shape.dims),
            })
    }

    fn f64_vec_to_value(elements: Vec<f64>, shape: Shape) -> Result<Value, ApiError> {
        if shape.rank() == 0 && elements.len() == 1 {
            return Ok(Value::scalar_f64(elements[0]));
        }
        let lits: Vec<Literal> = elements.into_iter().map(Literal::from_f64).collect();
        TensorValue::new(DType::F64, shape, lits)
            .map(Value::Tensor)
            .map_err(|e| ApiError::EvalError {
                detail: e.to_string(),
            })
    }

    fn i64_vec_to_value(elements: Vec<i64>, shape: Shape) -> Result<Value, ApiError> {
        if shape.rank() == 0 && elements.len() == 1 {
            return Ok(Value::scalar_i64(elements[0]));
        }
        let lits: Vec<Literal> = elements.into_iter().map(Literal::I64).collect();
        TensorValue::new(DType::I64, shape, lits)
            .map(Value::Tensor)
            .map_err(|e| ApiError::EvalError {
                detail: e.to_string(),
            })
    }

    fn bool_vec_to_value(elements: Vec<bool>, shape: Shape) -> Result<Value, ApiError> {
        if shape.rank() == 0 && elements.len() == 1 {
            return Ok(Value::scalar_bool(elements[0]));
        }
        let lits: Vec<Literal> = elements.into_iter().map(Literal::Bool).collect();
        TensorValue::new(DType::Bool, shape, lits)
            .map(Value::Tensor)
            .map_err(|e| ApiError::EvalError {
                detail: e.to_string(),
            })
    }

    /// Sample uniform `f64` values in `[minval, maxval)` with the given `shape`.
    pub fn uniform(
        key: PRNGKey,
        shape: Shape,
        minval: f64,
        maxval: f64,
    ) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_uniform(key, count, minval, maxval);
        f64_vec_to_value(vals, shape)
    }

    /// Sample standard normal `N(0, 1)` `f64` values with the given `shape`.
    pub fn normal(key: PRNGKey, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_normal(key, count);
        f64_vec_to_value(vals, shape)
    }

    /// Sample Bernoulli boolean values with success probability `p` and the given `shape`.
    pub fn bernoulli(key: PRNGKey, p: f64, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_bernoulli(key, count, p);
        bool_vec_to_value(vals, shape)
    }

    /// Sample uniform integers in `[minval, maxval)` with the given `shape`.
    pub fn randint(
        key: PRNGKey,
        shape: Shape,
        minval: i64,
        maxval: i64,
    ) -> Result<Value, ApiError> {
        if minval >= maxval {
            return Err(ApiError::EvalError {
                detail: format!("randint requires minval < maxval, got [{minval}, {maxval})"),
            });
        }
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_randint(key, count, minval, maxval);
        i64_vec_to_value(vals, shape)
    }

    /// Sample a single categorical index from unnormalized `logits`.
    pub fn categorical(key: PRNGKey, logits: &[f64]) -> Result<usize, ApiError> {
        let samples = fj_lax::threefry::random_categorical(key, logits, 1).map_err(|e| {
            ApiError::EvalError {
                detail: e.to_string(),
            }
        })?;
        Ok(samples[0])
    }

    /// Generate a uniform random permutation of `0..n` as a 1-D `I64` tensor.
    pub fn permutation(key: PRNGKey, n: usize) -> Result<Value, ApiError> {
        let perm: Vec<i64> = fj_lax::threefry::random_permutation(key, n)
            .into_iter()
            .map(|idx| idx as i64)
            .collect();
        i64_vec_to_value(perm, Shape::vector(n as u32))
    }

    /// Sample standard Gumbel(0, 1) values with the given `shape`.
    pub fn gumbel(key: PRNGKey, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_gumbel(key, count, 0.0, 1.0);
        f64_vec_to_value(vals, shape)
    }

    /// Sample standard Laplace(0, 1) values with the given `shape`.
    pub fn laplace(key: PRNGKey, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_laplace(key, count, 0.0, 1.0);
        f64_vec_to_value(vals, shape)
    }

    /// Sample standard Exponential(1) values with the given `shape`.
    pub fn exponential(key: PRNGKey, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_exponential(key, count, 1.0);
        f64_vec_to_value(vals, shape)
    }

    /// Sample standard Logistic(0, 1) values with the given `shape`.
    pub fn logistic(key: PRNGKey, shape: Shape) -> Result<Value, ApiError> {
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_logistic(key, count, 0.0, 1.0);
        f64_vec_to_value(vals, shape)
    }

    /// Sample Gamma(`alpha`, 1) values with the given `shape` (`alpha > 0`).
    pub fn gamma(key: PRNGKey, alpha: f64, shape: Shape) -> Result<Value, ApiError> {
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(ApiError::EvalError {
                detail: format!("gamma requires finite alpha > 0, got {alpha}"),
            });
        }
        let count = shape_count(&shape)?;
        let vals = fj_lax::threefry::random_gamma(key, count, alpha);
        f64_vec_to_value(vals, shape)
    }
}

/// Generic PyTree flattening, unflattening, mapping, and tracing (`jax.tree_util` parity).
///
/// Unlike `fj_lax::tree_util::TreeNode` (which is specialized to `f64`), [`tree_util::TreeNode<T>`]
/// is generic over any leaf type `T` (defaulting to [`Value`], and supporting [`TracerRef`],
/// [`ShapedArray`], etc.) and returns `Result` on all structural mismatches instead of panicking.
pub mod tree_util {
    use fj_core::Value;
    use fj_trace::{ClosedJaxpr, ShapedArray, TraceError, TracerRef, make_jaxpr_fallible};
    use std::collections::BTreeMap;

    /// A generic nested container tree (`PyTree`).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TreeNode<T = Value> {
        Leaf(T),
        Tuple(Vec<TreeNode<T>>),
        Dict(BTreeMap<String, TreeNode<T>>),
        None,
    }

    /// Structural skeleton of a [`TreeNode`] with leaves stripped.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TreeDef {
        Leaf,
        Tuple(Vec<TreeDef>),
        Dict(Vec<(String, TreeDef)>),
        None,
    }

    /// Errors returned by fallible PyTree operations.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum TreeError {
        LeafCountMismatch { expected: usize, actual: usize },
        StructureMismatch { expected: TreeDef, actual: TreeDef },
    }

    impl std::fmt::Display for TreeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::LeafCountMismatch { expected, actual } => write!(
                    f,
                    "tree_unflatten leaf count mismatch: expected {expected} leaves, got {actual}"
                ),
                Self::StructureMismatch { .. } => {
                    write!(f, "tree structure mismatch between arguments")
                }
            }
        }
    }

    impl std::error::Error for TreeError {}

    /// Flatten a [`TreeNode<T>`] into its ordered leaves and structural [`TreeDef`].
    pub fn tree_flatten<T: Clone>(node: &TreeNode<T>) -> (Vec<T>, TreeDef) {
        match node {
            TreeNode::Leaf(v) => (vec![v.clone()], TreeDef::Leaf),
            TreeNode::None => (vec![], TreeDef::None),
            TreeNode::Tuple(children) => {
                let mut leaves = Vec::new();
                let mut defs = Vec::with_capacity(children.len());
                for child in children {
                    let (child_leaves, child_def) = tree_flatten(child);
                    leaves.extend(child_leaves);
                    defs.push(child_def);
                }
                (leaves, TreeDef::Tuple(defs))
            }
            TreeNode::Dict(map) => {
                let mut leaves = Vec::new();
                let mut defs = Vec::with_capacity(map.len());
                for (k, child) in map {
                    let (child_leaves, child_def) = tree_flatten(child);
                    leaves.extend(child_leaves);
                    defs.push((k.clone(), child_def));
                }
                (leaves, TreeDef::Dict(defs))
            }
        }
    }

    /// Count the number of leaves in a [`TreeDef`].
    #[must_use]
    pub fn tree_def_leaf_count(treedef: &TreeDef) -> usize {
        match treedef {
            TreeDef::Leaf => 1,
            TreeDef::None => 0,
            TreeDef::Tuple(children) => children.iter().map(tree_def_leaf_count).sum(),
            TreeDef::Dict(entries) => entries.iter().map(|(_, d)| tree_def_leaf_count(d)).sum(),
        }
    }

    /// Reconstruct a [`TreeNode<T>`] from a [`TreeDef`] and a slice of leaves.
    pub fn tree_unflatten<T: Clone>(
        treedef: &TreeDef,
        leaves: &[T],
    ) -> Result<TreeNode<T>, TreeError> {
        let expected = tree_def_leaf_count(treedef);
        if leaves.len() != expected {
            return Err(TreeError::LeafCountMismatch {
                expected,
                actual: leaves.len(),
            });
        }
        let mut cursor = 0;
        Ok(unflatten_helper(treedef, leaves, &mut cursor))
    }

    fn unflatten_helper<T: Clone>(
        treedef: &TreeDef,
        leaves: &[T],
        cursor: &mut usize,
    ) -> TreeNode<T> {
        match treedef {
            TreeDef::Leaf => {
                let val = leaves[*cursor].clone();
                *cursor += 1;
                TreeNode::Leaf(val)
            }
            TreeDef::None => TreeNode::None,
            TreeDef::Tuple(children) => {
                let nodes = children
                    .iter()
                    .map(|c| unflatten_helper(c, leaves, cursor))
                    .collect();
                TreeNode::Tuple(nodes)
            }
            TreeDef::Dict(entries) => {
                let mut map = BTreeMap::new();
                for (k, def) in entries {
                    map.insert(k.clone(), unflatten_helper(def, leaves, cursor));
                }
                TreeNode::Dict(map)
            }
        }
    }

    /// Extract all leaves from a [`TreeNode<T>`] in deterministic traversal order.
    pub fn tree_leaves<T: Clone>(node: &TreeNode<T>) -> Vec<T> {
        tree_flatten(node).0
    }

    /// Extract the structural [`TreeDef`] from a [`TreeNode<T>`].
    pub fn tree_structure<T>(node: &TreeNode<T>) -> TreeDef {
        match node {
            TreeNode::Leaf(_) => TreeDef::Leaf,
            TreeNode::None => TreeDef::None,
            TreeNode::Tuple(children) => {
                TreeDef::Tuple(children.iter().map(tree_structure).collect())
            }
            TreeNode::Dict(map) => TreeDef::Dict(
                map.iter()
                    .map(|(k, v)| (k.clone(), tree_structure(v)))
                    .collect(),
            ),
        }
    }

    /// Count the number of leaves in a [`TreeNode<T>`].
    pub fn tree_leaf_count<T>(node: &TreeNode<T>) -> usize {
        match node {
            TreeNode::Leaf(_) => 1,
            TreeNode::None => 0,
            TreeNode::Tuple(children) => children.iter().map(tree_leaf_count).sum(),
            TreeNode::Dict(map) => map.values().map(tree_leaf_count).sum(),
        }
    }

    /// Apply a function `f` to every leaf of a [`TreeNode<T>`], preserving structure.
    pub fn tree_map<T, U, F>(node: &TreeNode<T>, mut f: F) -> TreeNode<U>
    where
        F: FnMut(&T) -> U,
    {
        tree_map_ref(node, &mut f)
    }

    fn tree_map_ref<T, U, F>(node: &TreeNode<T>, f: &mut F) -> TreeNode<U>
    where
        F: FnMut(&T) -> U,
    {
        match node {
            TreeNode::Leaf(v) => TreeNode::Leaf(f(v)),
            TreeNode::None => TreeNode::None,
            TreeNode::Tuple(children) => {
                TreeNode::Tuple(children.iter().map(|c| tree_map_ref(c, f)).collect())
            }
            TreeNode::Dict(map) => TreeNode::Dict(
                map.iter()
                    .map(|(k, v)| (k.clone(), tree_map_ref(v, f)))
                    .collect(),
            ),
        }
    }

    /// Apply a fallible function `f` to every leaf of a [`TreeNode<T>`], preserving structure.
    pub fn tree_map_fallible<T, U, E, F>(node: &TreeNode<T>, mut f: F) -> Result<TreeNode<U>, E>
    where
        F: FnMut(&T) -> Result<U, E>,
    {
        tree_map_fallible_ref(node, &mut f)
    }

    fn tree_map_fallible_ref<T, U, E, F>(node: &TreeNode<T>, f: &mut F) -> Result<TreeNode<U>, E>
    where
        F: FnMut(&T) -> Result<U, E>,
    {
        match node {
            TreeNode::Leaf(v) => Ok(TreeNode::Leaf(f(v)?)),
            TreeNode::None => Ok(TreeNode::None),
            TreeNode::Tuple(children) => {
                let mut mapped = Vec::with_capacity(children.len());
                for c in children {
                    mapped.push(tree_map_fallible_ref(c, f)?);
                }
                Ok(TreeNode::Tuple(mapped))
            }
            TreeNode::Dict(map) => {
                let mut mapped = BTreeMap::new();
                for (k, v) in map {
                    mapped.insert(k.clone(), tree_map_fallible_ref(v, f)?);
                }
                Ok(TreeNode::Dict(mapped))
            }
        }
    }

    /// Apply a binary function `f` to corresponding leaves of two identically structured trees.
    pub fn tree_map2<A: Clone, B: Clone, C: Clone, F>(
        a: &TreeNode<A>,
        b: &TreeNode<B>,
        mut f: F,
    ) -> Result<TreeNode<C>, TreeError>
    where
        F: FnMut(&A, &B) -> C,
    {
        let (leaves_a, def_a) = tree_flatten(a);
        let (leaves_b, def_b) = tree_flatten(b);
        if def_a != def_b {
            return Err(TreeError::StructureMismatch {
                expected: def_a,
                actual: def_b,
            });
        }
        let leaves_c: Vec<C> = leaves_a
            .iter()
            .zip(leaves_b.iter())
            .map(|(x, y)| f(x, y))
            .collect();
        tree_unflatten(&def_a, &leaves_c)
    }

    /// Trace a function over a [`TreeNode<TracerRef>`] and return `(ClosedJaxpr, out_treedef)`.
    pub fn make_jaxpr_tree<F>(
        f: F,
        in_tree: &TreeNode<ShapedArray>,
    ) -> Result<(ClosedJaxpr, TreeDef), TraceError>
    where
        F: FnOnce(TreeNode<TracerRef>) -> Result<TreeNode<TracerRef>, TraceError>,
    {
        let (in_avals, in_def) = tree_flatten(in_tree);
        let mut out_def_slot: Option<TreeDef> = None;
        let closed = make_jaxpr_fallible(
            |flat_inputs| {
                let tracer_tree = tree_unflatten(&in_def, flat_inputs)
                    .map_err(|_| TraceError::CompositionViolation)?;
                let out_tree = f(tracer_tree)?;
                let (out_leaves, out_def) = tree_flatten(&out_tree);
                out_def_slot = Some(out_def);
                Ok(out_leaves)
            },
            in_avals,
        )?;
        let out_def = out_def_slot.ok_or(TraceError::CompositionViolation)?;
        Ok((closed, out_def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{
        Atom, Equation, Jaxpr, Primitive, ProgramSpec, Transform, Value, VarId, build_program,
    };
    use std::sync::{Mutex, OnceLock};

    fn custom_rule_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        match GUARD.get_or_init(|| Mutex::new(())).lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn custom_square_jaxpr(input: u32, output: u32) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(input)],
            vec![],
            vec![VarId(output)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: vec![Atom::Var(VarId(input)), Atom::Var(VarId(input))].into(),
                outputs: vec![VarId(output)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    // --- Basic transform tests ---

    #[test]
    fn jit_add_scalar() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
            .expect("jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn jit_is_identity() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_i64(5), Value::scalar_i64(7)])
            .expect("jit should succeed");
        let direct_result =
            fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(7)])
                .expect("direct eval should succeed");
        assert_eq!(jit_result, direct_result);
    }

    #[test]
    fn grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("grad should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn grad_at_zero() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(0.0)])
            .expect("grad at zero should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!(derivative.abs() < 1e-3);
    }

    #[test]
    fn grad_negative_input() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(-5.0)])
            .expect("grad with negative input should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!((derivative - (-10.0)).abs() < 1e-3);
    }

    #[test]
    fn vmap_add_one() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = vmap(jaxpr)
            .call(vec![
                Value::vector_i64(&[10, 20, 30]).expect("vector should build"),
            ])
            .expect("vmap should succeed");
        let output = result[0].as_tensor().expect("vmap output should be tensor");
        let values: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64 element"))
            .collect();
        assert_eq!(values, vec![11, 21, 31]);
    }

    #[test]
    fn value_and_grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let (value, gradient) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad should succeed");
        assert_eq!(value.len(), 1);
        assert_eq!(gradient.len(), 1);

        let val = value[0].as_f64_scalar().expect("value should be scalar");
        assert!((val - 16.0).abs() < 1e-6);

        let grad_val = gradient[0]
            .as_f64_scalar()
            .expect("gradient should be scalar");
        assert!((grad_val - 8.0).abs() < 1e-3);
    }

    #[test]
    fn value_and_grad_returns_gradient_for_each_input() {
        let jaxpr = build_program(ProgramSpec::LaxMul);
        let (value, gradients) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("value_and_grad should succeed");

        assert_eq!(value.len(), 1);
        assert_eq!(gradients.len(), 2);
        assert!((value[0].as_f64_scalar().expect("scalar output") - 6.0).abs() < 1e-6);
        assert!((gradients[0].as_f64_scalar().expect("scalar gradient") - 3.0).abs() < 1e-6);
        assert!((gradients[1].as_f64_scalar().expect("scalar gradient") - 2.0).abs() < 1e-6);
    }

    #[test]
    fn linearize_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result =
            linearize(jaxpr, vec![Value::scalar_f64(3.0)]).expect("linearize should succeed");

        let primal_out = result.primal_outputs[0]
            .as_f64_scalar()
            .expect("primal should be scalar");
        assert!((primal_out - 9.0).abs() < 1e-6);

        let tangent_out = result
            .linearized
            .call(vec![Value::scalar_f64(1.0)])
            .expect("linearized call should succeed");
        let tangent_val = tangent_out[0]
            .as_f64_scalar()
            .expect("tangent should be scalar");
        assert!((tangent_val - 6.0).abs() < 1e-6);
    }

    #[test]
    fn linearize_reuses_primals() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = linearize(jaxpr, vec![Value::scalar_f64(4.0)]).expect("linearize");

        let t1 = result
            .linearized
            .call(vec![Value::scalar_f64(1.0)])
            .expect("call 1")[0]
            .as_f64_scalar()
            .unwrap();
        let t2 = result
            .linearized
            .call(vec![Value::scalar_f64(2.0)])
            .expect("call 2")[0]
            .as_f64_scalar()
            .unwrap();

        assert!((t1 - 8.0).abs() < 1e-6);
        assert!((t2 - 16.0).abs() < 1e-6);
    }

    #[test]
    fn linear_transpose_identity() {
        let jaxpr = build_program(ProgramSpec::Identity);
        let transposed =
            linear_transpose(jaxpr, vec![Value::scalar_f64(1.0)]).expect("linear_transpose");
        let result = transposed.call(Value::scalar_f64(5.0)).expect("call")[0]
            .as_f64_scalar()
            .unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn linear_transpose_neg() {
        let jaxpr = build_program(ProgramSpec::LaxNeg);
        let transposed =
            linear_transpose(jaxpr, vec![Value::scalar_f64(1.0)]).expect("linear_transpose");
        let result = transposed.call(Value::scalar_f64(3.0)).expect("call")[0]
            .as_f64_scalar()
            .unwrap();
        assert!((result - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn eval_shape_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let shapes = eval_shape(&jaxpr, &[Value::scalar_f64(0.0)]).expect("eval_shape");
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].shape.dims, Vec::<u32>::new());
        assert_eq!(shapes[0].dtype, fj_core::DType::F64);
    }

    #[test]
    fn eval_shape_add() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let shapes =
            eval_shape(&jaxpr, &[Value::scalar_i64(0), Value::scalar_i64(0)]).expect("eval_shape");
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].dtype, fj_core::DType::I64);
    }

    #[test]
    fn eval_shape_vector() {
        let jaxpr = build_program(ProgramSpec::LaxNeg);
        let shapes = eval_shape(&jaxpr, &[Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vec")])
            .expect("eval_shape");
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].shape.dims, vec![3u32]);
        assert_eq!(shapes[0].dtype, fj_core::DType::F64);
    }

    #[test]
    fn eval_shape_comparison_returns_bool_dtype_and_broadcast_shape() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Lt,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let shapes = eval_shape_avals(
            &jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 1] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![1, 3] },
                },
            ],
        )
        .expect("eval_shape comparison should succeed");
        assert_eq!(
            shapes,
            vec![ShapedArray {
                dtype: DType::Bool,
                shape: Shape { dims: vec![2, 3] },
            }]
        );
    }

    #[test]
    fn eval_shape_reduce_axis_and_permuted_transpose() {
        let mut reduce_params = std::collections::BTreeMap::new();
        reduce_params.insert("axes".to_owned(), "1".to_owned());
        let mut transpose_params = std::collections::BTreeMap::new();
        transpose_params.insert("permutation".to_owned(), "2,0,1".to_owned());

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Transpose,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(2)].into(),
                    params: transpose_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: vec![Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: reduce_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let shapes = eval_shape_avals(
            &jaxpr,
            &[ShapedArray {
                dtype: DType::F32,
                shape: Shape {
                    dims: vec![2, 3, 4],
                },
            }],
        )
        .expect("eval_shape transpose + axis reduction should succeed");
        assert_eq!(
            shapes,
            vec![
                ShapedArray {
                    dtype: DType::F32,
                    shape: Shape {
                        dims: vec![4, 2, 3],
                    },
                },
                ShapedArray {
                    dtype: DType::F32,
                    shape: Shape { dims: vec![4, 3] },
                },
            ]
        );
    }

    #[test]
    fn eval_shape_dot_slice_concatenate_multi_output_linalg_and_rfft() {
        // 1. Dot: [2, 3] x [3, 4] -> [2, 4]
        let dot_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Dot,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let dot_shapes = eval_shape_avals(
            &dot_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 3] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![3, 4] },
                },
            ],
        )
        .expect("dot eval_shape");
        assert_eq!(dot_shapes[0].shape.dims, vec![2, 4]);

        // Negative Dot test: incompatible inner dimensions [2, 3] x [5, 4]
        let dot_err = eval_shape_avals(
            &dot_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 3] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![5, 4] },
                },
            ],
        )
        .expect_err("incompatible dot inner dimensions must fail");
        assert!(dot_err.to_string().contains("contraction mismatch"));

        // 2. Slice + Concatenate
        let mut slice_params = std::collections::BTreeMap::new();
        slice_params.insert("start_indices".to_owned(), "1,0".to_owned());
        slice_params.insert("limit_indices".to_owned(), "3,3".to_owned());
        let mut concat_params = std::collections::BTreeMap::new();
        concat_params.insert("dimension".to_owned(), "0".to_owned());
        let slice_concat_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Slice,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: slice_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Concatenate,
                    inputs: vec![Atom::Var(VarId(3)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: concat_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let sc_shapes = eval_shape_avals(
            &slice_concat_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![4, 3] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![5, 3] },
                },
            ],
        )
        .expect("slice + concatenate eval_shape");
        assert_eq!(sc_shapes[0].shape.dims, vec![7, 3]);

        // 3. Multi-output QR & SVD + Rfft
        let qr_svd_rfft_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4), VarId(5), VarId(6), VarId(7), VarId(8)],
            vec![
                Equation {
                    primitive: Primitive::Qr,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3), VarId(4)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Svd,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(5), VarId(6), VarId(7)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Rfft,
                    inputs: vec![Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(8)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let multi_shapes = eval_shape_avals(
            &qr_svd_rfft_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![4, 3] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![8] },
                },
            ],
        )
        .expect("qr + svd + rfft eval_shape");
        assert_eq!(multi_shapes[0].shape.dims, vec![4, 3]); // Q
        assert_eq!(multi_shapes[1].shape.dims, vec![3, 3]); // R
        assert_eq!(multi_shapes[2].shape.dims, vec![4, 3]); // U
        assert_eq!(multi_shapes[3].shape.dims, vec![3]); // S
        assert_eq!(multi_shapes[4].shape.dims, vec![3, 3]); // Vt
        assert_eq!(
            multi_shapes[5],
            ShapedArray {
                dtype: DType::Complex128,
                shape: Shape { dims: vec![5] },
            }
        );
    }

    #[test]
    fn eval_shape_sub_jaxpr_cond_and_scan() {
        // Cond with 2 sub_jaxprs: both return [3] F64
        let branch = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: vec![Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let cond_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Cond,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![branch.clone(), branch],
                effects: vec![],
            }],
        );
        let cond_out = eval_shape_avals(
            &cond_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::Bool,
                    shape: Shape::scalar(),
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![3] },
                },
            ],
        )
        .expect("cond sub_jaxpr eval_shape");
        assert_eq!(cond_out[0].shape.dims, vec![3]);

        // Scan with 1 sub_jaxpr: carry [2] + xs [5, 2] -> carry [2], ys [5, 2]
        let scan_body = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let scan_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3), VarId(4)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![scan_body],
                effects: vec![],
            }],
        );
        let scan_out = eval_shape_avals(
            &scan_jaxpr,
            &[
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2] },
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![5, 2] },
                },
            ],
        )
        .expect("scan sub_jaxpr eval_shape");
        assert_eq!(scan_out[0].shape.dims, vec![2]);
        assert_eq!(scan_out[1].shape.dims, vec![5, 2]);
    }

    // --- Transform composition tests ---

    #[test]
    fn jit_grad_composition() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jit(jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(5.0)])
            .expect("jit(grad(f)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 10.0).abs() < 1e-3);
    }

    #[test]
    fn jit_vmap_composition() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = jit(jaxpr)
            .compose_vmap()
            .call(vec![
                Value::vector_i64(&[1, 2, 3]).expect("vector should build"),
            ])
            .expect("jit(vmap(f)) should succeed");
        let output = result[0].as_tensor().expect("should be tensor");
        let values: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn vmap_grad_composition() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = vmap(jaxpr)
            .compose_grad()
            .call(vec![
                Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
            ])
            .expect("vmap(grad(f)) should succeed");
        let output = result[0].as_tensor().expect("should be tensor");
        let values = output.to_f64_vec().expect("f64 elements");
        assert_eq!(values.len(), 3);
        assert!((values[0] - 2.0).abs() < 1e-3);
        assert!((values[1] - 4.0).abs() < 1e-3);
        assert!((values[2] - 6.0).abs() < 1e-3);
    }

    #[test]
    fn compose_helper() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = compose(jaxpr, vec![Transform::Jit, Transform::Grad])
            .call(vec![Value::scalar_f64(7.0)])
            .expect("compose(jit, grad) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 14.0).abs() < 1e-3);
    }

    // --- Error path tests ---

    #[test]
    fn grad_non_scalar_input_fails() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = grad(jaxpr)
            .call(vec![
                Value::vector_f64(&[1.0, 2.0]).expect("vector should build"),
            ])
            .expect_err("grad with vector input should fail");
        assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    }

    #[test]
    fn grad_empty_args_fails() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = grad(jaxpr)
            .call(vec![])
            .expect_err("grad with no args should fail");
        assert!(matches!(err, ApiError::EvalError { .. }));
    }

    #[test]
    fn vmap_scalar_input_fails() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let err = vmap(jaxpr)
            .call(vec![Value::scalar_i64(42)])
            .expect_err("vmap with scalar input should fail");
        assert!(matches!(err, ApiError::EvalError { .. }));
    }

    #[test]
    fn grad_vmap_fails_non_scalar() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = compose(jaxpr, vec![Transform::Grad, Transform::Vmap])
            .call(vec![
                Value::vector_f64(&[1.0, 2.0]).expect("vector should build"),
            ])
            .expect_err("grad(vmap(f)) should fail with non-scalar input to grad");
        assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    }

    #[test]
    fn vmap_dimension_mismatch_preserves_info() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = vmap(jaxpr)
            .call(vec![
                Value::vector_i64(&[1, 2, 3]).expect("vec3"),
                Value::vector_i64(&[1, 2]).expect("vec2"),
            ])
            .expect_err("vmap with mismatched leading dims should fail");
        match err {
            ApiError::VmapDimensionMismatch { expected, actual } => {
                assert_eq!(expected, 3, "expected leading dim should be 3");
                assert_eq!(actual, 2, "actual leading dim should be 2");
            }
            other => {
                std::panic::panic_any(format!("expected VmapDimensionMismatch, got: {other:?}"))
            }
        }
    }

    // --- Mode configuration tests ---

    #[test]
    fn with_mode_hardened() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .with_mode(fj_core::CompatibilityMode::Hardened)
            .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
            .expect("hardened jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(3)]);
    }

    #[test]
    fn composed_with_mode() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jit(jaxpr)
            .compose_grad()
            .with_mode(fj_core::CompatibilityMode::Hardened)
            .call(vec![Value::scalar_f64(2.0)])
            .expect("hardened jit(grad(f)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 4.0).abs() < 1e-3);
    }

    #[test]
    fn api_exposes_custom_vjp_registration() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();
        register_custom_vjp(Primitive::CountLeadingZeros, |_inputs, _g, _params| {
            Ok(vec![Value::scalar_f64(7.0)])
        });

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::CountLeadingZeros,
                inputs: vec![Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let result = grad(jaxpr)
            .call(vec![Value::scalar_i64(8)])
            .expect("grad should use custom VJP");
        let derivative = result[0].as_f64_scalar().expect("scalar derivative");
        assert!((derivative - 7.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn api_exposes_custom_jvp_registration() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();
        register_custom_jvp(
            Primitive::CountLeadingZeros,
            |_primals, _tangents, _params| Ok(Value::scalar_f64(5.0)),
        );

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::CountLeadingZeros,
                inputs: vec![Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let result = fj_ad::jvp(&jaxpr, &[Value::scalar_i64(8)], &[Value::scalar_f64(1.0)])
            .expect("jvp should use custom rule");
        let tangent = result.tangents[0].as_f64_scalar().expect("scalar tangent");
        assert!((tangent - 5.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_vjp_composes_through_jit_grad_and_value_and_grad() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let wrapped = custom_vjp(
            custom_square_jaxpr(901, 902),
            |primals| {
                let x = primals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP forward expected scalar x".to_owned())
                })?;
                Ok((vec![Value::scalar_f64(x * x)], vec![Value::scalar_f64(x)]))
            },
            |residuals, cotangent| {
                let x = residuals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP backward expected residual x".to_owned())
                })?;
                let g = cotangent.as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP backward expected scalar cotangent".to_owned())
                })?;
                Ok(vec![Value::scalar_f64(10.0 * x * g)])
            },
        );

        let primal = wrapped
            .call(vec![Value::scalar_f64(3.0)])
            .expect("custom_vjp primal call should preserve original function");
        assert!((primal[0].as_f64_scalar().expect("scalar primal") - 9.0).abs() < 1e-10);

        let composed_grad = wrapped
            .compose_jit_grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("jit(grad(custom_vjp(f))) should use the custom VJP");
        assert!((composed_grad[0].as_f64_scalar().expect("scalar gradient") - 30.0).abs() < 1e-10);

        let (values, gradients) = wrapped
            .value_and_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad(custom_vjp(f)) should use the custom VJP");
        assert!((values[0].as_f64_scalar().expect("scalar value") - 16.0).abs() < 1e-10);
        assert!((gradients[0].as_f64_scalar().expect("scalar gradient") - 40.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_vjp_wrappers_keep_equal_jaxpr_rules_isolated() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let make_wrapped = |scale: f64| {
            custom_vjp(
                custom_square_jaxpr(905, 906),
                |primals| {
                    let x = primals[0].as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed("custom VJP forward expected scalar x".to_owned())
                    })?;
                    Ok((vec![Value::scalar_f64(x * x)], vec![Value::scalar_f64(x)]))
                },
                move |residuals, cotangent| {
                    let x = residuals[0].as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed("custom VJP backward expected residual x".to_owned())
                    })?;
                    let g = cotangent.as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed(
                            "custom VJP backward expected scalar cotangent".to_owned(),
                        )
                    })?;
                    Ok(vec![Value::scalar_f64(scale * x * g)])
                },
            )
        };

        let first = make_wrapped(10.0);
        let second = make_wrapped(20.0);

        let first_grad = first
            .grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("first custom_vjp grad should use first rule");
        let second_grad = second
            .grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("second custom_vjp grad should use second rule");
        assert!(
            (first_grad[0]
                .as_f64_scalar()
                .expect("first scalar gradient")
                - 30.0)
                .abs()
                < 1e-10
        );
        assert!(
            (second_grad[0]
                .as_f64_scalar()
                .expect("second scalar gradient")
                - 60.0)
                .abs()
                < 1e-10
        );

        let (_, first_value_grad) = first
            .value_and_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first value_and_grad should use first rule");
        let first_composed_grad = first
            .compose_jit_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first jit(grad(custom_vjp)) should use first rule");
        assert!(
            (first_value_grad[0]
                .as_f64_scalar()
                .expect("first value_and_grad scalar gradient")
                - 40.0)
                .abs()
                < 1e-10
        );
        assert!(
            (first_composed_grad[0]
                .as_f64_scalar()
                .expect("first composed scalar gradient")
                - 40.0)
                .abs()
                < 1e-10
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_jvp_drives_jvp_and_jacobian() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let wrapped = custom_jvp(custom_square_jaxpr(903, 904), |primals, tangents| {
            let x = primals[0]
                .as_f64_scalar()
                .ok_or_else(|| AdError::EvalFailed("custom JVP expected scalar x".to_owned()))?;
            let dx = tangents[0].as_f64_scalar().ok_or_else(|| {
                AdError::EvalFailed("custom JVP expected scalar tangent".to_owned())
            })?;
            Ok((
                vec![Value::scalar_f64(x * x)],
                vec![Value::scalar_f64(40.0 + dx)],
            ))
        });

        let jvp = wrapped
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("custom_jvp function rule should drive direct JVP");
        assert!((jvp.primals[0].as_f64_scalar().expect("scalar primal") - 16.0).abs() < 1e-10);
        assert!((jvp.tangents[0].as_f64_scalar().expect("scalar tangent") - 42.0).abs() < 1e-10);

        let jacobian = wrapped
            .jacobian()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("jacobian(custom_jvp(f)) should use the custom JVP");
        if let Some(value) = jacobian.as_f64_scalar() {
            assert!((value - 41.0).abs() < 1e-10);
        } else {
            let tensor = jacobian.as_tensor().expect("jacobian tensor output");
            let values = tensor.to_f64_vec().expect("f64 jacobian tensor");
            assert_eq!(values.len(), 1);
            assert!((values[0] - 41.0).abs() < 1e-10);
        }

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_jvp_wrappers_keep_equal_jaxpr_rules_isolated() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let make_wrapped = |offset: f64| {
            custom_jvp(custom_square_jaxpr(907, 908), move |primals, tangents| {
                let x = primals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom JVP expected scalar x".to_owned())
                })?;
                let dx = tangents[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom JVP expected scalar tangent".to_owned())
                })?;
                Ok((
                    vec![Value::scalar_f64(x * x)],
                    vec![Value::scalar_f64(offset + dx)],
                ))
            })
        };

        let first = make_wrapped(100.0);
        let second = make_wrapped(200.0);

        let first_jvp = first
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("first custom_jvp call should use first rule");
        let second_jvp = second
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("second custom_jvp call should use second rule");
        assert!(
            (first_jvp.tangents[0]
                .as_f64_scalar()
                .expect("first scalar tangent")
                - 102.0)
                .abs()
                < 1e-10
        );
        assert!(
            (second_jvp.tangents[0]
                .as_f64_scalar()
                .expect("second scalar tangent")
                - 202.0)
                .abs()
                < 1e-10
        );

        let first_jacobian = first
            .jacobian()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first jacobian should use first rule");
        let first_jacobian_value = first_jacobian
            .as_f64_scalar()
            .or_else(|| {
                let tensor = first_jacobian.as_tensor()?;
                tensor.to_f64_vec()?.first().copied()
            })
            .expect("first jacobian should be scalar-like");
        assert!((first_jacobian_value - 101.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn jacobian_two_outputs_two_inputs() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let jac = jacobian(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("jacobian should succeed");
        let tensor = jac.as_tensor().expect("jacobian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let values = tensor.to_f64_vec().expect("f64 tensor");
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn hessian_matches_quadratic_cross_term() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(3)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let hes = hessian(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("hessian should succeed");
        let tensor = hes.as_tensor().expect("hessian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let values = tensor.to_f64_vec().expect("f64 tensor");
        assert!((values[0] - 6.0).abs() < 1e-3);
        assert!((values[1] - 4.0).abs() < 1e-3);
        assert!((values[2] - 4.0).abs() < 1e-3);
        assert!(values[3].abs() < 1e-3);
    }

    // ── End-to-end trace → transform integration tests ───────

    #[test]
    fn trace_jit_square_e2e() {
        // Trace x*x via make_jaxpr, then jit it
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(7.0)])
            .expect("jit(traced x*x) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 49.0).abs() < 1e-10, "7^2 should be 49, got {val}");
    }

    #[test]
    fn trace_grad_square_e2e() {
        // Trace x*x, then grad: d(x^2)/dx = 2x
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(5.0)])
            .expect("grad(traced x*x) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 10.0).abs() < 1e-3,
            "d(x^2)/dx at x=5 should be 10, got {derivative}"
        );
    }

    #[test]
    fn trace_jit_grad_square_e2e() {
        // Trace x*x, then jit(grad): d(x^2)/dx = 2x via composition
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("jit(grad(traced x*x)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 6.0).abs() < 1e-3,
            "d(x^2)/dx at x=3 should be 6, got {derivative}"
        );
    }

    #[test]
    fn trace_chain_e2e() {
        // Trace x*x + x + x + x = x^2 + 3x, then grad: d(x^2 + 3x)/dx = 2x + 3
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let x = &inputs[0];
                let x_sq = x.binary_op(Primitive::Mul, x).unwrap();
                let sum1 = x_sq.binary_op(Primitive::Add, x).unwrap();
                let sum2 = sum1.binary_op(Primitive::Add, x).unwrap();
                let sum3 = sum2.binary_op(Primitive::Add, x).unwrap();
                vec![sum3]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        // f(x) = x^2 + 3x, f'(x) = 2x + 3
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("grad(traced x^2 + 3x) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 11.0).abs() < 1e-3,
            "d(x^2 + 3x)/dx at x=4 should be 11, got {derivative}"
        );
    }

    #[test]
    fn trace_multi_input_e2e() {
        // Trace f(x, y) = x + y, then jit
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let aval = ShapedArray {
            dtype: DType::F64,
            shape: fj_core::Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let sum = inputs[0].binary_op(Primitive::Add, &inputs[1]).unwrap();
                vec![sum]
            },
            vec![aval.clone(), aval],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
            .expect("jit(traced x+y) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 7.0).abs() < 1e-10, "3+4 should be 7, got {val}");
    }

    #[test]
    fn trace_unary_chain_e2e() {
        // Trace neg(exp(x)), then jit
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp = inputs[0].unary_op(Primitive::Exp).unwrap();
                let neg = exp.unary_op(Primitive::Neg).unwrap();
                vec![neg]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(1.0)])
            .expect("jit(traced neg(exp(x))) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        let expected = -(1.0_f64.exp());
        assert!(
            (val - expected).abs() < 1e-10,
            "-exp(1) should be {expected}, got {val}"
        );
    }

    // ================================================================
    // E2E trace→transform tests for complex programs (frankenjax-kdi)
    // ================================================================

    #[test]
    fn trace_sin_cos_chain_grad_e2e() {
        // f(x) = sin(cos(x)), grad: f'(x) = cos(cos(x)) * (-sin(x))
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let cos_x = inputs[0].unary_op(Primitive::Cos).unwrap();
                let sin_cos = cos_x.unary_op(Primitive::Sin).unwrap();
                vec![sin_cos]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 1.0_f64;
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(sin(cos(x))) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = x.cos().cos() * (-x.sin());
        assert!(
            (d - expected).abs() < 1e-10,
            "grad(sin(cos(1.0))) should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_exp_mul_grad_e2e() {
        // f(x) = x * exp(x), grad: f'(x) = exp(x) + x * exp(x) = (1+x) * exp(x)
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp_x = inputs[0].unary_op(Primitive::Exp).unwrap();
                let x_exp = inputs[0].binary_op(Primitive::Mul, &exp_x).unwrap();
                vec![x_exp]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 2.0_f64;
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(x*exp(x)) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = (1.0 + x) * x.exp();
        assert!(
            (d - expected).abs() < 1e-6,
            "grad(x*exp(x)) at x=2 should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_polynomial_jit_grad_e2e() {
        // f(x) = x^3 + 2*x^2 + x via tracing: t1=x*x, t2=t1*x, t3=t1+t1, t4=t2+t3, y=t4+x
        // f'(x) = 3x^2 + 4x + 1
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let x = &inputs[0];
                let x2 = x.binary_op(Primitive::Mul, x).unwrap();
                let x3 = x2.binary_op(Primitive::Mul, x).unwrap();
                let two_x2 = x2.binary_op(Primitive::Add, &x2).unwrap();
                let sum = x3.binary_op(Primitive::Add, &two_x2).unwrap();
                let y = sum.binary_op(Primitive::Add, x).unwrap();
                vec![y]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 3.0_f64;
        let result = jit(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad(x^3+2x^2+x)) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = 3.0 * x * x + 4.0 * x + 1.0;
        assert!(
            (d - expected).abs() < 1e-3,
            "jit(grad(x^3+2x^2+x)) at x=3 should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_sin_vmap_grad_e2e() {
        // Trace sin(x), then vmap(grad) over a batch
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sin_x = inputs[0].unary_op(Primitive::Sin).unwrap();
                vec![sin_x]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let inputs = [0.0, 1.0, 2.0, 3.0];
        let result = vmap(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::vector_f64(&inputs).expect("vector")])
            .expect("vmap(grad(sin)) should succeed");
        let t = result[0].as_tensor().expect("tensor");
        let vals = t.to_f64_vec().expect("f64 vec");
        for (i, (&actual, &x)) in vals.iter().zip(inputs.iter()).enumerate() {
            let expected = x.cos();
            assert!(
                (actual - expected).abs() < 1e-6,
                "vmap(grad(sin))[{i}]: expected cos({x})={expected}, got {actual}"
            );
        }
    }

    #[test]
    fn trace_exp_neg_jit_e2e() {
        // f(x) = -exp(x), jit execution
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp_x = inputs[0].unary_op(Primitive::Exp).unwrap();
                let neg_exp = exp_x.unary_op(Primitive::Neg).unwrap();
                vec![neg_exp]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(0.0)])
            .expect("jit(-exp(x)) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (val - (-1.0)).abs() < 1e-10,
            "-exp(0) should be -1, got {val}"
        );
    }

    #[test]
    fn trace_tanh_grad_grad_e2e() {
        // f(x) = tanh(x), f''(x) = -2*tanh(x)*sech^2(x) = -2*tanh(x)*(1-tanh^2(x))
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let tanh_x = inputs[0].unary_op(Primitive::Tanh).unwrap();
                vec![tanh_x]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 0.5_f64;
        let result = compose(closed.jaxpr, vec![Transform::Grad, Transform::Grad])
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(grad(tanh)) should succeed");
        let d2 = result[0].as_f64_scalar().expect("should be f64");
        let t = x.tanh();
        let expected = -2.0 * t * (1.0 - t * t);
        assert!(
            (d2 - expected).abs() < 1e-4,
            "tanh''(0.5) should be {expected}, got {d2}"
        );
    }

    #[test]
    fn trace_fallible_error_propagation() {
        // make_jaxpr_fallible should propagate errors cleanly
        use fj_core::DType;
        use fj_trace::{ShapedArray, TraceError};

        let result = make_jaxpr_fallible(
            |_inputs| Err(TraceError::InvalidAbstractValue),
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        );
        assert!(result.is_err());
    }

    // ── value_and_grad e2e test (frankenjax-rjj) ──

    #[test]
    fn trace_value_and_grad_e2e() {
        // Trace x*x, then value_and_grad: returns (x^2, 2x)
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let (values, gradients) = value_and_grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad(traced x*x) should succeed");
        let val = values[0].as_f64_scalar().expect("value should be f64");
        assert!((val - 16.0).abs() < 1e-3, "f(4) = 4^2 = 16, got {val}");
        assert!(!gradients.is_empty(), "should produce gradients");
    }

    #[test]
    fn jacobian_single_output_single_input() {
        // f(x) = x*x, Jacobian = [2x]
        let jaxpr = build_program(ProgramSpec::Square);
        let jac = jacobian(jaxpr)
            .call(vec![Value::scalar_f64(5.0)])
            .expect("jacobian(x^2) should succeed");
        // Result should be a scalar (1x1 Jacobian) or a 1-element tensor
        if let Some(v) = jac.as_f64_scalar() {
            assert!(
                (v - 10.0).abs() < 1e-3,
                "J(x^2) at x=5 should be 10, got {v}"
            );
        } else if let Some(t) = jac.as_tensor() {
            let vals = t.to_f64_vec().expect("f64 tensor");
            assert!(!vals.is_empty(), "jacobian tensor should have values");
        } else {
            std::panic::panic_any("jacobian should return a scalar or tensor");
        }
    }

    #[test]
    fn hessian_quadratic() {
        // f(x) = x^2, H = [2]
        let jaxpr = build_program(ProgramSpec::Square);
        let hes = hessian(jaxpr)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("hessian(x^2) should succeed");
        if let Some(h) = hes.as_f64_scalar() {
            assert!((h - 2.0).abs() < 1e-2, "d²/dx²(x²) should be 2, got {h}");
        } else if let Some(t) = hes.as_tensor() {
            let vals = t.to_f64_vec().expect("f64 tensor");
            assert!(!vals.is_empty(), "hessian tensor should have values");
        }
    }

    #[test]
    fn nn_activations_and_reductions_trace_jit_grad_vmap() {
        // Test scalar activations with jit + grad
        let closed = make_jaxpr_fallible(
            |inputs| {
                let x = &inputs[0];
                let a = nn::relu(x)?;
                let b = nn::leaky_relu(x, 0.1)?;
                let c = nn::relu6(x)?;
                let d = nn::elu(x, 1.0)?;
                let e = nn::selu(x)?;
                let f = nn::celu(x, 1.0)?;
                let g = nn::gelu(x, false)?;
                let h = nn::gelu(x, true)?;
                let i = nn::silu(x)?;
                let j = nn::mish(x)?;
                let k = nn::softplus(x)?;
                let l = nn::soft_sign(x)?;
                let m = nn::sigmoid(x)?;
                let n = nn::hard_sigmoid(x)?;
                let o = nn::hard_swish(x)?;
                let p = nn::hard_tanh(x)?;
                let q = nn::log_sigmoid(x)?;
                let r = nn::swiglu(x, &a)?;
                let s = nn::geglu(x, &b, false)?;
                let total = a
                    .add_op(&b)?
                    .add_op(&c)?
                    .add_op(&d)?
                    .add_op(&e)?
                    .add_op(&f)?
                    .add_op(&g)?
                    .add_op(&h)?
                    .add_op(&i)?
                    .add_op(&j)?
                    .add_op(&k)?
                    .add_op(&l)?
                    .add_op(&m)?
                    .add_op(&n)?
                    .add_op(&o)?
                    .add_op(&p)?
                    .add_op(&q)?
                    .add_op(&r)?
                    .add_op(&s)?;
                Ok(vec![total])
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: Shape::scalar(),
            }],
        )
        .expect("nn scalar activations should trace");

        let out = jit(closed.jaxpr.clone())
            .call(vec![Value::scalar_f64(0.5)])
            .expect("jit on nn activations should succeed");
        assert!(out[0].as_f64_scalar().unwrap().is_finite());

        let g_out = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(0.5)])
            .expect("grad on nn activations should succeed");
        assert!(g_out[0].as_f64_scalar().unwrap().is_finite());

        // Test vector softmax, log_softmax, logsumexp, layer_norm, rms_norm, l2_loss
        let vec_closed = make_jaxpr_fallible(
            |inputs| {
                let x = &inputs[0];
                let sm = nn::softmax(x, -1)?;
                let lsm = nn::log_softmax(x, -1)?;
                let lse = nn::logsumexp(x, -1)?;
                let ln = nn::layer_norm(x, -1, 1e-5, None, None)?;
                let rn = nn::rms_norm(x, -1, 1e-5, None)?;
                let loss = nn::l2_loss(&sm.add_op(&lsm)?.add_op(&ln)?.add_op(&rn)?)?;
                Ok(vec![loss.add_op(&lse)?])
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: Shape::vector(4),
            }],
        )
        .expect("nn vector reductions should trace");

        let vec_in = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let vec_out = jit(vec_closed.jaxpr)
            .call(vec![vec_in])
            .expect("jit on nn vector ops should succeed");
        assert!(vec_out[0].as_f64_scalar().unwrap().is_finite());
    }

    #[test]
    fn random_distributions_and_prng_keys_work() {
        let k0 = random::key(42);
        let (k1, k2) = random::split(k0);
        assert_ne!(k1, k2);
        let keys = random::split_n(k0, 4);
        assert_eq!(keys.len(), 4);
        let k_folded = random::fold_in(k0, 7);
        assert_ne!(k_folded, k0);

        let shape = Shape { dims: vec![2, 3] };
        let u = random::uniform(k1, shape.clone(), -1.0, 2.0).expect("uniform");
        assert_eq!(u.as_tensor().unwrap().shape, shape);

        let n = random::normal(k2, shape.clone()).expect("normal");
        assert_eq!(n.as_tensor().unwrap().shape, shape);

        let b = random::bernoulli(keys[0], 0.5, shape.clone()).expect("bernoulli");
        assert_eq!(b.as_tensor().unwrap().dtype, DType::Bool);

        let ri = random::randint(keys[1], shape.clone(), 0, 10).expect("randint");
        assert_eq!(ri.as_tensor().unwrap().dtype, DType::I64);
        assert!(random::randint(keys[1], shape.clone(), 10, 5).is_err());

        let cat = random::categorical(keys[2], &[0.1, 2.0, 0.5]).expect("categorical");
        assert!(cat < 3);

        let perm = random::permutation(keys[3], 5).expect("permutation");
        assert_eq!(perm.as_tensor().unwrap().shape, Shape::vector(5));

        for dist_res in [
            random::gumbel(k1, shape.clone()),
            random::laplace(k1, shape.clone()),
            random::exponential(k1, shape.clone()),
            random::logistic(k1, shape.clone()),
            random::gamma(k1, 2.5, shape.clone()),
        ] {
            let val = dist_res.expect("distribution should succeed");
            assert_eq!(val.as_tensor().unwrap().shape, shape);
        }
        assert!(random::gamma(k1, -1.0, shape).is_err());
    }

    #[test]
    fn tree_util_generic_flatten_unflatten_map_and_trace() {
        use std::collections::BTreeMap;
        use tree_util::{
            TreeError, TreeNode, make_jaxpr_tree, tree_flatten, tree_leaf_count, tree_leaves,
            tree_map, tree_map2, tree_structure, tree_unflatten,
        };

        let mut dict = BTreeMap::new();
        dict.insert("a".to_owned(), TreeNode::Leaf(Value::scalar_f64(2.0)));
        dict.insert(
            "b".to_owned(),
            TreeNode::Tuple(vec![
                TreeNode::Leaf(Value::scalar_f64(3.0)),
                TreeNode::None,
                TreeNode::Leaf(Value::scalar_f64(5.0)),
            ]),
        );
        let tree = TreeNode::Dict(dict);

        assert_eq!(tree_leaf_count(&tree), 3);
        let (leaves, treedef) = tree_flatten(&tree);
        assert_eq!(leaves.len(), 3);
        assert_eq!(tree_structure(&tree), treedef);
        assert_eq!(tree_leaves(&tree), leaves);

        let roundtrip = tree_unflatten(&treedef, &leaves).expect("unflatten should succeed");
        assert_eq!(roundtrip, tree);

        // Leaf count mismatch returns Err instead of panicking
        let mismatch = tree_unflatten(&treedef, &leaves[..2]);
        assert_eq!(
            mismatch,
            Err(TreeError::LeafCountMismatch {
                expected: 3,
                actual: 2
            })
        );

        let doubled = tree_map(&tree, |v| {
            Value::scalar_f64(v.as_f64_scalar().unwrap() * 2.0)
        });
        let sum_tree = tree_map2(&tree, &doubled, |x, y| {
            Value::scalar_f64(x.as_f64_scalar().unwrap() + y.as_f64_scalar().unwrap())
        })
        .expect("tree_map2 should succeed");
        let sum_leaves: Vec<f64> = tree_leaves(&sum_tree)
            .into_iter()
            .map(|v| v.as_f64_scalar().unwrap())
            .collect();
        assert_eq!(sum_leaves, vec![6.0, 9.0, 15.0]);

        // Trace over a PyTree of ShapedArrays
        let aval_tree = tree_map(&tree, |_| ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        });
        let (closed, out_def) = make_jaxpr_tree(
            |tracer_tree| {
                tree_util::tree_map_fallible(&tracer_tree, |leaf| {
                    leaf.mul_op(&TracerRef::literal(fj_core::Literal::from_f64(10.0))?)
                })
            },
            &aval_tree,
        )
        .expect("make_jaxpr_tree should succeed");
        assert_eq!(out_def, treedef);
        let flat_out = jit(closed.jaxpr).call(leaves).expect("jit on tree jaxpr");
        let out_tree = tree_unflatten(&out_def, &flat_out).expect("unflatten outputs");
        let out_f64: Vec<f64> = tree_leaves(&out_tree)
            .into_iter()
            .map(|v| v.as_f64_scalar().unwrap())
            .collect();
        assert_eq!(out_f64, vec![20.0, 30.0, 50.0]);
    }

    #[test]
    fn control_flow_trace_cond_switch_while_scan_via_fj_api() {
        // Trace cond + while_loop through fj_api::make_jaxpr_fallible and execute via jit
        let closed = make_jaxpr_fallible(
            |inputs| {
                let pred = &inputs[0];
                let x = &inputs[1];
                let branched = trace_cond(
                    pred,
                    &[x],
                    |ops| Ok(vec![&ops[0] + 10.0_f64]),
                    |ops| Ok(vec![&ops[0] * 2.0_f64]),
                )?;
                let b_refs: Vec<&TracerRef> = branched.iter().collect();
                trace_while_loop(
                    &b_refs,
                    |state| state[0].lt(&TracerRef::literal(fj_core::Literal::from_f64(20.0))?),
                    |state| Ok(vec![&state[0] + 3.0_f64]),
                    Some(100),
                )
            },
            vec![
                ShapedArray {
                    dtype: DType::Bool,
                    shape: Shape::scalar(),
                },
                ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::scalar(),
                },
            ],
        )
        .expect("cond + while_loop trace should succeed");

        let out_true = jit(closed.jaxpr.clone())
            .call(vec![Value::scalar_bool(true), Value::scalar_f64(5.0)])
            .expect("jit(cond+while) true");
        // 5 + 10 = 15 -> +3 = 18 -> +3 = 21
        assert_eq!(out_true, vec![Value::scalar_f64(21.0)]);

        let out_false = jit(closed.jaxpr)
            .call(vec![Value::scalar_bool(false), Value::scalar_f64(5.0)])
            .expect("jit(cond+while) false");
        // 5 * 2 = 10 -> 13 -> 16 -> 19 -> 22
        assert_eq!(out_false, vec![Value::scalar_f64(22.0)]);
    }

    #[test]
    fn test_rc26_14_api_evidence_proof_and_admission_governance() {
        reset_dispatch_governance();
        let jaxpr = build_program(ProgramSpec::Square);

        // 1. JitWrapped: call_with_evidence + last_evidence_ledger + last_composition_proof
        let jit_wrapped = jit(jaxpr.clone());
        assert!(jit_wrapped.last_evidence_ledger().is_none());
        assert!(jit_wrapped.last_composition_proof().is_none());

        let jit_resp = jit_wrapped
            .call_with_evidence(vec![Value::scalar_f64(4.0)])
            .expect("jit call_with_evidence should succeed");
        assert_eq!(jit_resp.outputs, vec![Value::scalar_f64(16.0)]);
        assert_eq!(jit_resp.evidence_ledger.len(), 1);
        assert!(jit_resp.composition_proof.verify().is_ok());
        assert_eq!(
            jit_wrapped
                .last_evidence_ledger()
                .expect("jit last_evidence_ledger")
                .len(),
            1
        );
        assert!(
            jit_wrapped
                .last_composition_proof()
                .expect("jit last_composition_proof")
                .verify()
                .is_ok()
        );

        // 2. GradWrapped: call_with_evidence + last_evidence_ledger + last_composition_proof
        let grad_wrapped = grad(jaxpr.clone());
        let grad_resp = grad_wrapped
            .call_with_evidence(vec![Value::scalar_f64(5.0)])
            .expect("grad call_with_evidence should succeed");
        assert_eq!(grad_resp.outputs, vec![Value::scalar_f64(10.0)]);
        assert_eq!(
            grad_wrapped
                .last_evidence_ledger()
                .expect("grad last_evidence_ledger")
                .len(),
            1
        );
        assert!(
            grad_wrapped
                .last_composition_proof()
                .expect("grad last_composition_proof")
                .verify()
                .is_ok()
        );

        // 3. ValueAndGradWrapped: call_with_evidence + last_evidence_ledger + last_composition_proof
        let vg_wrapped = value_and_grad(jaxpr.clone());
        let ((vals, grads), vg_resp) = vg_wrapped
            .call_with_evidence(vec![Value::scalar_f64(3.0)])
            .expect("value_and_grad call_with_evidence should succeed");
        assert_eq!(vals, vec![Value::scalar_f64(9.0)]);
        assert_eq!(grads, vec![Value::scalar_f64(6.0)]);
        assert_eq!(vg_resp.evidence_ledger.len(), 1);
        assert!(
            vg_wrapped
                .last_composition_proof()
                .expect("vg last_composition_proof")
                .verify()
                .is_ok()
        );

        // 4. VmapWrapped & ComposedTransform: call_with_evidence + last_composition_proof
        let vmap_wrapped = vmap(jaxpr.clone());
        let batch_arg = Value::vector_f64(&[2.0, 3.0]).expect("vector");
        let vmap_resp = vmap_wrapped
            .call_with_evidence(vec![batch_arg.clone()])
            .expect("vmap call_with_evidence should succeed");
        assert_eq!(
            vmap_resp.outputs,
            vec![Value::vector_f64(&[4.0, 9.0]).expect("vector")]
        );
        assert!(
            vmap_wrapped
                .last_composition_proof()
                .expect("vmap last_composition_proof")
                .verify()
                .is_ok()
        );

        let composed = jit(jaxpr.clone()).compose_grad();
        let comp_resp = composed
            .call_with_evidence(vec![Value::scalar_f64(6.0)])
            .expect("composed call_with_evidence should succeed");
        assert_eq!(comp_resp.outputs, vec![Value::scalar_f64(12.0)]);
        assert!(
            composed
                .last_composition_proof()
                .expect("composed last_composition_proof")
                .verify()
                .is_ok()
        );

        // 5. Pre-execution admission rejection through fj-api fast paths (CancellationToken + MemoryBudget)
        let token = dispatch_cancellation_token();
        token.cancel("api test cancellation");
        let cancelled_err = jit_wrapped
            .call(vec![Value::scalar_f64(2.0)])
            .expect_err("cancelled jit call must fail closed before execution");
        assert!(
            cancelled_err.to_string().contains("cancelled"),
            "unexpected error: {cancelled_err}"
        );
        token.reset();

        set_dispatch_memory_budget(MemoryBudget {
            max_working_set_bytes: 1,
            max_equation_count: 100,
            max_transform_depth: 10,
        });
        let budget_err = grad_wrapped
            .call(vec![Value::scalar_f64(2.0)])
            .expect_err("over-budget grad call must fail closed before execution");
        assert!(
            budget_err.to_string().contains("exceeds budget"),
            "unexpected error: {budget_err}"
        );

        reset_dispatch_governance();
    }

    #[test]
    fn wave9_vjp_jvp_and_symbolic_ad_jaxpr_work_end_to_end() {
        let jaxpr = build_program(ProgramSpec::Square);
        let jvp_res = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[Value::scalar_f64(2.0)])
            .expect("top-level jvp should succeed");
        assert_eq!(jvp_res.primals, vec![Value::scalar_f64(9.0)]);
        assert_eq!(jvp_res.tangents, vec![Value::scalar_f64(12.0)]);

        let vjp_res = vjp(jaxpr.clone(), vec![Value::scalar_f64(3.0)])
            .expect("top-level vjp should record tape once");
        assert_eq!(vjp_res.primal_outputs, vec![Value::scalar_f64(9.0)]);
        assert_eq!(
            vjp_res
                .pullback
                .call(Value::scalar_f64(1.0))
                .expect("pullback(1.0)"),
            vec![Value::scalar_f64(6.0)]
        );
        assert_eq!(
            vjp_res
                .pullback
                .call(Value::scalar_f64(2.5))
                .expect("pullback(2.5)"),
            vec![Value::scalar_f64(15.0)]
        );

        let sym_jvp = jvp_to_jaxpr(&jaxpr).expect("jvp_to_jaxpr should succeed");
        let sym_jvp_out = jit(sym_jvp)
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(2.0)])
            .expect("jit(jvp_to_jaxpr) should succeed");
        assert_eq!(
            sym_jvp_out,
            vec![Value::scalar_f64(9.0), Value::scalar_f64(12.0)]
        );

        let sym_grad = grad_to_jaxpr(&jaxpr).expect("grad_to_jaxpr should succeed");
        let sym_grad_out = jit(sym_grad)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("jit(grad_to_jaxpr) should succeed");
        assert_eq!(sym_grad_out, vec![Value::scalar_f64(6.0)]);
    }

    #[test]
    fn wave9_jit_compile_option_builders_wire_into_dispatch() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let wrapped = jit(jaxpr.clone())
            .with_egraph_optimization(true)
            .with_wave_parallel(true);
        let opts = wrapped.compile_options();
        assert_eq!(
            opts.get("egraph_optimize").map(String::as_str),
            Some("true")
        );
        assert_eq!(opts.get("wave_parallel").map(String::as_str), Some("true"));

        let out = wrapped
            .call(vec![Value::scalar_f64(4.0), Value::scalar_f64(5.0)])
            .expect("jit with egraph + wave_parallel options should succeed");
        assert_eq!(out, vec![Value::scalar_f64(9.0)]);

        let over_limit = jit(jaxpr).with_max_equation_count(0);
        let err = over_limit
            .call(vec![Value::scalar_f64(4.0), Value::scalar_f64(5.0)])
            .expect_err("max_equation_count=0 must reject execution");
        assert!(
            err.to_string().contains("exceeds budget"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn wave9_closed_jaxpr_into_executable_and_closure_transforms_work() {
        // 1. ClosedJaxpr with captured tensor const_values lowered via into_executable_jaxpr
        let closed = make_jaxpr_fallible(
            |xs| {
                let c = TracerRef::constant(Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap())?;
                Ok(vec![xs[0].add_op(&c)?.reduce_sum(None)?])
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: Shape::vector(3),
            }],
        )
        .expect("trace with tensor constant should succeed");
        assert_eq!(closed.const_values.len(), 1);
        let exec_jaxpr = closed.into_executable_jaxpr();
        assert!(exec_jaxpr.constvars.is_empty());

        let x = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (vals, grads) = value_and_grad(exec_jaxpr)
            .call(vec![x.clone()])
            .expect("value_and_grad on lowered ClosedJaxpr should succeed");
        assert_eq!(vals, vec![Value::scalar_f64(66.0)]);
        assert_eq!(grads, vec![Value::vector_f64(&[1.0, 1.0, 1.0]).unwrap()]);

        // 2. Closure-tracing transform wrappers (jit_fn, grad_fn, value_and_grad_fn, vmap_fn, jacobian_fn, hessian_fn, eval_shape_fn)
        let jit_out = jit_fn(|xs| Ok(vec![xs[0].square()?]), vec![Value::scalar_f64(5.0)])
            .expect("jit_fn should succeed");
        assert_eq!(jit_out, vec![Value::scalar_f64(25.0)]);

        let grad_out = grad_fn(|xs| Ok(vec![xs[0].square()?]), vec![Value::scalar_f64(5.0)])
            .expect("grad_fn should succeed");
        assert_eq!(grad_out, vec![Value::scalar_f64(10.0)]);

        let (vg_v, vg_g) =
            value_and_grad_fn(|xs| Ok(vec![xs[0].square()?]), vec![Value::scalar_f64(4.0)])
                .expect("value_and_grad_fn should succeed");
        assert_eq!(vg_v, vec![Value::scalar_f64(16.0)]);
        assert_eq!(vg_g, vec![Value::scalar_f64(8.0)]);

        let vmap_out =
            vmap_fn(|xs| Ok(vec![xs[0].square()?]), vec![x]).expect("vmap_fn should succeed");
        assert_eq!(vmap_out, vec![Value::vector_f64(&[1.0, 4.0, 9.0]).unwrap()]);

        let jac_out = jacobian_fn(|xs| Ok(vec![xs[0].square()?]), vec![Value::scalar_f64(3.0)])
            .expect("jacobian_fn should succeed");
        assert_eq!(
            jac_out.as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![6.0]
        );

        let hess_out = hessian_fn(|xs| Ok(vec![xs[0].square()?]), vec![Value::scalar_f64(3.0)])
            .expect("hessian_fn should succeed");
        assert_eq!(
            hess_out.as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![2.0]
        );

        let shapes = eval_shape_fn(
            |xs| Ok(vec![xs[0].reduce_sum(Some(&[1]))?]),
            &[ShapedArray {
                dtype: DType::F64,
                shape: Shape { dims: vec![4, 8] },
            }],
        )
        .expect("eval_shape_fn should succeed");
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].shape.dims, vec![4]);
    }

    #[test]
    fn wave13_symbolic_hvp_hessian_egraph_staging_and_ffi_work() {
        // 1. Symbolic HVP, hvp_fn, and HessianWrapped methods on f(x) = x^3 + 2*x
        let cubic_jaxpr = trace_jaxpr_for_values(
            |xs| {
                let x = &xs[0];
                let x3 = x.square()?.mul_op(x)?;
                let two_x = x.mul_op(&TracerRef::literal(Literal::from_f64(2.0))?)?;
                Ok(vec![x3.add_op(&two_x)?])
            },
            &[Value::scalar_f64(3.0)],
        )
        .expect("trace cubic");

        // f''(x) = 6*x => at x=3.0, f''(3)=18; with v=2.5, HVP = 45.0
        let hvp_wrapped = hessian(cubic_jaxpr.clone());
        let hvp_out = hvp_wrapped
            .hvp(&[Value::scalar_f64(3.0)], &[Value::scalar_f64(2.5)])
            .expect("HessianWrapped::hvp");
        assert!((hvp_out[0].as_f64_scalar().unwrap() - 45.0).abs() < 1e-10);

        let hvp_fn_out = hvp_fn(
            |xs| {
                let x = &xs[0];
                Ok(vec![x.square()?.mul_op(x)?])
            },
            &[Value::scalar_f64(3.0)],
            &[Value::scalar_f64(2.5)],
        )
        .expect("hvp_fn");
        assert!((hvp_fn_out[0].as_f64_scalar().unwrap() - 45.0).abs() < 1e-10);

        let sym_hess = hvp_wrapped
            .call_symbolic(vec![Value::scalar_f64(3.0)])
            .expect("call_symbolic");
        assert_eq!(
            sym_hess.as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![18.0]
        );

        let hess_ir = hvp_wrapped.to_jaxpr().expect("HessianWrapped::to_jaxpr");
        let hess_ir_out =
            eval_jaxpr(&hess_ir, &[Value::scalar_f64(3.0)]).expect("eval hessian_to_jaxpr");
        assert!((hess_ir_out[0].as_f64_scalar().unwrap() - 18.0).abs() < 1e-10);

        let hvp_ir = hvp_wrapped
            .to_hvp_jaxpr()
            .expect("HessianWrapped::to_hvp_jaxpr");
        let hvp_ir_out = eval_jaxpr(&hvp_ir, &[Value::scalar_f64(3.0), Value::scalar_f64(2.5)])
            .expect("eval hvp_to_jaxpr");
        assert!((hvp_ir_out[0].as_f64_scalar().unwrap() - 45.0).abs() < 1e-10);

        let jvp_ir = jacobian(cubic_jaxpr.clone())
            .to_jvp_jaxpr()
            .expect("JacobianWrapped::to_jvp_jaxpr");
        let jvp_ir_out = eval_jaxpr(&jvp_ir, &[Value::scalar_f64(3.0), Value::scalar_f64(1.0)])
            .expect("eval jvp_to_jaxpr");
        // f(3) = 27 + 6 = 33; f'(3) = 3*9 + 2 = 29
        assert!((jvp_ir_out[0].as_f64_scalar().unwrap() - 33.0).abs() < 1e-10);
        assert!((jvp_ir_out[1].as_f64_scalar().unwrap() - 29.0).abs() < 1e-10);

        // 2. E-graph optimization and staging re-exports in fj-api
        let opt_ir = optimize_fn(
            |xs| {
                let zero = TracerRef::literal(Literal::from_f64(0.0))?;
                let one = TracerRef::literal(Literal::from_f64(1.0))?;
                Ok(vec![xs[0].add_op(&zero)?.mul_op(&one)?])
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: Shape::scalar(),
            }],
        )
        .expect("optimize_fn");
        let opt_out = eval_jaxpr(&opt_ir, &[Value::scalar_f64(7.5)]).expect("eval opt_ir");
        assert_eq!(opt_out, vec![Value::scalar_f64(7.5)]);

        let staged = stage_jaxpr(
            &cubic_jaxpr,
            &[false],
            &[Value::scalar_f64(2.0)], // f(2) = 8 + 4 = 12
        )
        .expect("stage_jaxpr");
        let staged_out = execute_staged(&staged, &[]).expect("execute_staged");
        assert_eq!(staged_out, vec![Value::scalar_f64(12.0)]);

        // 3. FFI CustomCall through JitWrapped::with_ffi_registry and call_with_ffi
        let mut params = std::collections::BTreeMap::new();
        params.insert("call_target_name".to_owned(), "double_plus5".to_owned());
        params.insert("result_dtypes".to_owned(), "f64".to_owned());
        params.insert("result_shapes".to_owned(), "".to_owned());
        let custom_call_jaxpr = fj_core::Jaxpr::new(
            vec![fj_core::VarId(1)],
            vec![],
            vec![fj_core::VarId(2)],
            vec![fj_core::Equation {
                primitive: fj_core::Primitive::CustomCall,
                inputs: smallvec::smallvec![fj_core::Atom::Var(fj_core::VarId(1))],
                outputs: smallvec::smallvec![fj_core::VarId(2)],
                params,
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );
        let registry = FfiRegistry::new();
        registry
            .register(
                "double_plus5",
                fj_ffi::fj_ffi_builtin_double_plus5_scalar_f64,
            )
            .expect("register ffi target");

        let jit_ffi = jit(custom_call_jaxpr.clone()).with_ffi_registry(registry.clone());
        let ffi_out = jit_ffi
            .call(vec![Value::scalar_f64(6.0)])
            .expect("jit with_ffi_registry");
        assert_eq!(ffi_out, vec![Value::scalar_f64(17.0)]);

        let call_with_ffi_out = jit(custom_call_jaxpr)
            .call_with_ffi(vec![Value::scalar_f64(10.0)], &registry)
            .expect("jit call_with_ffi");
        assert_eq!(call_with_ffi_out, vec![Value::scalar_f64(25.0)]);
    }

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]

            #[test]
            fn metamorphic_jit_transparent(x in -100.0f64..100.0) {
                prop_assume!(x.is_finite());
                let jaxpr = build_program(ProgramSpec::Square);
                let jit_result = jit(jaxpr.clone())
                    .call(vec![Value::scalar_f64(x)])
                    .expect("jit");
                let direct_result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("direct");
                prop_assert_eq!(jit_result, direct_result, "jit not transparent at x={}", x);
            }

            #[test]
            fn metamorphic_grad_square_is_2x(x in -100.0f64..100.0) {
                prop_assume!(x.is_finite());
                let jaxpr = build_program(ProgramSpec::Square);
                let grad_result = grad(jaxpr)
                    .call(vec![Value::scalar_f64(x)])
                    .expect("grad");
                let actual = grad_result[0].as_f64_scalar().unwrap();
                let expected = 2.0 * x;
                prop_assert!((actual - expected).abs() < 1e-8, "grad(x^2) != 2x: {} vs {} at x={}", actual, expected, x);
            }

            #[test]
            fn metamorphic_value_and_grad_consistent(x in -100.0f64..100.0) {
                prop_assume!(x.is_finite());
                let jaxpr = build_program(ProgramSpec::Square);
                let vg_result = value_and_grad(jaxpr.clone())
                    .call(vec![Value::scalar_f64(x)])
                    .expect("value_and_grad");
                let grad_only = grad(jaxpr)
                    .call(vec![Value::scalar_f64(x)])
                    .expect("grad");
                let vg_grad = vg_result.1[0].as_f64_scalar().unwrap();
                let grad_val = grad_only[0].as_f64_scalar().unwrap();
                prop_assert!((vg_grad - grad_val).abs() < 1e-14, "value_and_grad grad != grad: {} vs {}", vg_grad, grad_val);
            }
        }
    }
}
