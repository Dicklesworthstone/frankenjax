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
    grad_jaxpr_with_cotangent, grad_to_jaxpr, hessian_to_jaxpr, hvp_to_jaxpr, jacfwd_jaxpr,
    jacrev_jaxpr, jvp_to_jaxpr, linearize_jaxpr, record_vjp_tape, register_custom_jaxpr_jvp,
    register_custom_jaxpr_vjp, register_custom_jvp, register_custom_vjp, symbolic_hessian_jaxpr,
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
pub use fj_ledger::{
    CalibratedEstimate, CalibrationReport, ConformalPredictor, DecisionAction, DecisionRecord,
    EProcess, EvidenceLedger, EvidenceSignal, LedgerEntry, LogDomainPosterior, LossMatrix,
    expected_loss_keep, expected_loss_kill, log_sum_exp, recommend_action,
};
pub use fj_runtime::{
    AdmissionDecision, AdmissionRejection, CancellationToken, ExecutionRegime, MemoryBudget,
    RuntimeAdmissionModel,
    backend::{Backend, BackendCapabilities, BackendError, BackendRegistry},
    buffer::{Buffer, BufferView},
    device::{DeviceId, DeviceInfo, DevicePlacement, Platform},
};
pub use transforms::{
    CheckpointWrapped, ComposedTransform, CustomJvpWrapped, CustomVjpWrapped, GradWrapped,
    HessianWrapped, JacobianMode, JacobianWrapped, JitWrapped, LinearizeResult, LinearizedFunction,
    PmapWrapped, TransposedLinearFunction, ValueAndGradWrapped, VjpPullback, VjpResult,
    VmapWrapped,
};
pub use transforms::{
    ShapedArray, checkpoint, compose, custom_jvp, custom_vjp, eval_shape, eval_shape_avals,
    eval_shape_fn, grad, grad_fn, hessian, hessian_fn, hvp, hvp_fn, jacfwd, jacfwd_fn, jacobian,
    jacobian_fn, jacrev, jacrev_fn, jit, jit_fn, jvp, jvp_fn, linear_transpose, linearize,
    linearize_fn, optimize_fn, pmap, pmap_fn, trace_jaxpr_for_values, value_and_grad,
    value_and_grad_fn, vjp, vjp_fn, vmap, vmap_fn,
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

    /// Broadcast `reduced` (`x` reduced along `axis`) back onto `x`'s shape, repeating it
    /// along `axis`. `x` must have rank at least 1.
    fn broadcast_back(
        reduced: &TracerRef,
        x: &TracerRef,
        axis: usize,
    ) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        let broadcast_dims: Vec<usize> = (0..rank).filter(|&d| d != axis).collect();
        reduced.broadcast_in_dim(&x.aval().shape.dims, &broadcast_dims)
    }

    fn reduce_keepdims_sum(x: &TracerRef, axis: usize) -> Result<TracerRef, TraceError> {
        let sum = x.reduce_sum(Some(&[axis as i64]))?;
        if x.aval().shape.rank() == 0 {
            return Ok(sum);
        }
        broadcast_back(&sum, x, axis)
    }

    fn reduce_keepdims_max(x: &TracerRef, axis: usize) -> Result<TracerRef, TraceError> {
        let max = x.reduce_max(Some(&[axis as i64]))?;
        if x.aval().shape.rank() == 0 {
            return Ok(max);
        }
        broadcast_back(&max, x, axis)
    }

    /// The scalar `value` in `x`'s floating dtype. (An `f64` literal would promote an
    /// `f32`/`bf16`/`f16` computation to `f64`.)
    fn scalar_like(x: &TracerRef, value: f64) -> Result<TracerRef, TraceError> {
        let lit = match x.aval().dtype {
            DType::F32 => Literal::from_f32(value as f32),
            DType::BF16 => Literal::from_bf16_f64(value),
            DType::F16 => Literal::from_f16_f64(value),
            _ => Literal::from_f64(value),
        };
        x.bind_literal(lit)
    }

    /// Zeros with `x`'s shape and floating dtype.
    fn zeros_like(x: &TracerRef) -> Result<TracerRef, TraceError> {
        scalar_like(x, 0.0)?.broadcast_in_dim(&x.aval().shape.dims, &[])
    }

    /// Identity activation: returns `x` unmodified (`jax.nn.identity`).
    pub fn identity(x: &TracerRef) -> Result<TracerRef, TraceError> {
        Ok(x.clone())
    }

    /// Rectified Linear Unit: `max(x, 0)`.
    pub fn relu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.max(&scalar_like(x, 0.0)?)
    }

    /// Leaky ReLU: `select(x >= 0, x, negative_slope * x)`.
    pub fn leaky_relu(x: &TracerRef, negative_slope: f64) -> Result<TracerRef, TraceError> {
        let zero = scalar_like(x, 0.0)?;
        let cond = x.ge(&zero)?;
        let scaled = x.mul_op(&scalar_like(x, negative_slope)?)?;
        cond.select(x, &scaled)
    }

    /// ReLU6: `clamp(0, x, 6)`.
    pub fn relu6(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let zero = scalar_like(x, 0.0)?;
        let six = scalar_like(x, 6.0)?;
        x.clamp(&zero, &six)
    }

    /// Exponential Linear Unit: `select(x > 0, x, alpha * expm1(x))`.
    pub fn elu(x: &TracerRef, alpha: f64) -> Result<TracerRef, TraceError> {
        let zero = scalar_like(x, 0.0)?;
        let cond = x.gt(&zero)?;
        let neg_branch = x.expm1()?.mul_op(&scalar_like(x, alpha)?)?;
        cond.select(x, &neg_branch)
    }

    /// Scaled Exponential Linear Unit (SELU).
    pub fn selu(x: &TracerRef) -> Result<TracerRef, TraceError> {
        const ALPHA: f64 = 1.673_263_242_354_377_2;
        const SCALE: f64 = 1.050_700_987_355_480_5;
        elu(x, ALPHA)?.mul_op(&scalar_like(x, SCALE)?)
    }

    /// Continuously Differentiable ELU: `max(0, x) + min(0, alpha * expm1(x / alpha))`.
    pub fn celu(x: &TracerRef, alpha: f64) -> Result<TracerRef, TraceError> {
        let zero = scalar_like(x, 0.0)?;
        let alpha_lit = scalar_like(x, alpha)?;
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
        let half = scalar_like(x, 0.5)?;
        let one = scalar_like(x, 1.0)?;
        if approximate {
            let sqrt_2_over_pi = scalar_like(x, (2.0_f64 / std::f64::consts::PI).sqrt())?;
            let coeff = scalar_like(x, 0.044_715)?;
            let x2 = x.square()?;
            let x3 = x2.mul_op(x)?;
            let inner = x.add_op(&x3.mul_op(&coeff)?)?.mul_op(&sqrt_2_over_pi)?;
            x.mul_op(&half)?.mul_op(&one.add_op(&inner.tanh()?)?)
        } else {
            let inv_sqrt2 = scalar_like(x, std::f64::consts::FRAC_1_SQRT_2)?;
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

    /// Squareplus: `0.5 * (x + sqrt(x^2 + b))` (`jax.nn.squareplus`).
    pub fn squareplus(x: &TracerRef, b: f64) -> Result<TracerRef, TraceError> {
        let rad = x.square()?.add_op(&scalar_like(x, b)?)?.sqrt()?;
        x.add_op(&rad)?.mul_op(&scalar_like(x, 0.5)?)
    }

    /// Sparse plus: `0` for `x <= -1`, `(x + 1)^2 / 4` for `-1 < x < 1`, `x` for `x >= 1`
    /// (`jax.nn.sparse_plus`).
    pub fn sparse_plus(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let neg_one = scalar_like(x, -1.0)?;
        let pos_one = scalar_like(x, 1.0)?;
        let zero_vec = zeros_like(x)?;
        let mid = x
            .add_op(&pos_one)?
            .square()?
            .mul_op(&scalar_like(x, 0.25)?)?;
        let inner = x.ge(&pos_one)?.select(x, &mid)?;
        x.le(&neg_one)?.select(&zero_vec, &inner)
    }

    /// Sparse sigmoid: `0.5 * clamp(0, x + 1, 2)` (`jax.nn.sparse_sigmoid`).
    pub fn sparse_sigmoid(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let shifted = x.add_op(&scalar_like(x, 1.0)?)?;
        let clamped = shifted.clamp(&scalar_like(x, 0.0)?, &scalar_like(x, 2.0)?)?;
        clamped.mul_op(&scalar_like(x, 0.5)?)
    }

    /// Numerically stable `log(1 - exp(-x))` for `x > 0` (`jax.nn.log1mexp`), using the
    /// Mächler split at `ln(2)`.
    pub fn log1mexp(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let ln2 = scalar_like(x, std::f64::consts::LN_2)?;
        let neg_x = x.neg_op()?;
        let small_branch = neg_x.expm1()?.neg_op()?.log()?;
        let large_branch = neg_x.exp()?.neg_op()?.log1p()?;
        x.lt(&ln2)?.select(&small_branch, &large_branch)
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.mul_op(&softplus(x)?.tanh()?)
    }

    /// Softsign: `x / (1 + |x|)`.
    pub fn soft_sign(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let denom = x.abs()?.add_op(&scalar_like(x, 1.0)?)?;
        x.div_op(&denom)
    }

    /// Hard tanh: `clamp(-1, x, 1)`.
    pub fn hard_tanh(x: &TracerRef) -> Result<TracerRef, TraceError> {
        x.clamp(&scalar_like(x, -1.0)?, &scalar_like(x, 1.0)?)
    }

    /// Hard sigmoid: `relu6(x + 3) / 6`.
    pub fn hard_sigmoid(x: &TracerRef) -> Result<TracerRef, TraceError> {
        let shifted = x.add_op(&scalar_like(x, 3.0)?)?;
        relu6(&shifted)?.div_op(&scalar_like(x, 6.0)?)
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

    /// Gated Linear Unit (`jax.nn.glu`): splits `x` in half along `axis` and computes
    /// `first_half * sigmoid(second_half)`.
    pub fn glu(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        let ax = normalize_axis(rank, axis)?;
        let half = x.aval().shape.dims[ax] / 2;
        let start_a = vec![0u32; rank];
        let mut limit_a = x.aval().shape.dims.clone();
        limit_a[ax] = half;
        let mut start_b = vec![0u32; rank];
        start_b[ax] = half;
        let mut limit_b = x.aval().shape.dims.clone();
        limit_b[ax] = half * 2;
        let first_half = x.slice(&start_a, &limit_a, None)?;
        let second_half = x.slice(&start_b, &limit_b, None)?;
        first_half.mul_op(&sigmoid(&second_half)?)
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
    ///
    /// Computed as `max + log1p(Σ_j exp(x_j - max) - [j = i])`, with `i` the first argmax: the
    /// argmax's own `exp(0) = 1` is taken out of the sum exactly and added back by `log1p`. The
    /// usual `max + log(Σ_j exp(x_j - max))` adds that 1 to the other terms first and rounds
    /// their total away once it is below one ulp of 1, so `logsumexp([0, -40, -40])` came out as
    /// exactly 0 in f64 ("softmax collapse", Barron, Gnanapragasam and Tschannen, *A Fix for
    /// Softmax and Cross-Entropy*). As in `jax.nn.logsumexp` the max is under `stop_gradient`
    /// (the value does not depend on it), so the gradient is `softmax(x)`.
    pub fn logsumexp(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return Ok(x.clone());
        }
        let ax = normalize_axis(rank, axis)?;
        if x.aval().shape.dims[ax] == 0 {
            // log(Σ exp(∅)) = -inf, without an argmax over an empty axis.
            return x.exp()?.reduce_sum(Some(&[ax as i64]))?.log();
        }
        let max = x.reduce_max(Some(&[ax as i64]))?.stop_gradient()?;
        let shifted = x.sub_op(&broadcast_back(&max, x, ax)?)?;
        let first_max = x.argmax(Some(ax as i64))?.one_hot(
            x.aval().shape.dims[ax],
            x.aval().dtype,
            Some(ax as i64),
        )?;
        let tail = shifted
            .exp()?
            .sub_op(&first_max)?
            .reduce_sum(Some(&[ax as i64]))?;
        max.add_op(&tail.log1p()?)
    }

    /// Numerically stable log-mean-exp along `axis` (`logsumexp(x, axis) - ln(N)`).
    pub fn logmeanexp(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return Ok(x.clone());
        }
        let ax = normalize_axis(rank, axis)?;
        let n = x.aval().shape.dims[ax] as f64;
        let lse = logsumexp(x, axis)?;
        lse.sub_op(&scalar_like(&lse, n.ln())?)
    }

    /// Numerically stable softmax along `axis` (use `-1` for the last axis).
    ///
    /// Unlike `jax.nn.softmax`, the max is not under `stop_gradient`, and that matters: when one
    /// probability rounds to 1, the part of `p_i - p_i²` that flows through its own exponential
    /// cancels to 0, and the max's gradient, which the `ReduceMax` rule routes to the argmax, is
    /// what restores it from the other entries ("softmax collapse", see [`log_softmax`]).
    pub fn softmax(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return scalar_like(x, 1.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let max_keep = reduce_keepdims_max(x, ax)?;
        let shifted_exp = x.sub_op(&max_keep)?.exp()?;
        let sum_keep = reduce_keepdims_sum(&shifted_exp, ax)?;
        shifted_exp.div_op(&sum_keep)
    }

    /// Numerically stable log-softmax along `axis` (use `-1` for the last axis).
    ///
    /// Follows Barron, Gnanapragasam and Tschannen, *A Fix for Softmax and Cross-Entropy*: with
    /// `i` the first argmax, `log_softmax(x)_k = [k ≠ i](x_k - x_i) - log1p(Σ_{j≠i} exp(x_j -
    /// x_i))`. The usual `(x - max) - log(Σ_j exp(x_j - max))` adds the argmax's `exp(0) = 1` to
    /// the other terms, and once their total is below one ulp of 1 (a logit margin above ~16.6
    /// in f32, ~36.7 in f64) the argmax entry, and with it a hard-target cross-entropy
    /// `-Σ onehot · log_softmax`, becomes exactly 0 ("softmax collapse"). Here the argmax entry
    /// keeps `-log1p(tail)`.
    ///
    /// Entry `i` of the shift is exactly 0 and carries no gradient of its own, so `x_i` enters
    /// only through the differences `x_j - x_i`. Reverse and forward mode therefore build the
    /// argmax entry's derivative from the other entries: `1 - p_i` becomes `Σ_{j≠i} p_j`, and for
    /// a hard target on the argmax the gradient is minus the sum of the other softmax values
    /// rather than `p_i - 1`, which rounds to 0 in the same regime. `x_i` is picked out with the
    /// argmax mask rather than a `ReduceMax`, whose gradient would be shared between tied maxima
    /// while only entry `i` is cut here. Rows with a `+inf`, NaN or only `-inf` entries give
    /// NaN, as before.
    pub fn log_softmax(x: &TracerRef, axis: isize) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return scalar_like(x, 0.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let n = x.aval().shape.dims[ax];
        if n == 0 {
            // Nothing to normalise (and no argmax).
            return Ok(x.clone());
        }
        let argmax = x.argmax(Some(ax as i64))?;
        let first_max = argmax.one_hot(n, DType::Bool, Some(ax as i64))?;
        let max = first_max
            .select(x, &zeros_like(x)?)?
            .reduce_sum(Some(&[ax as i64]))?;
        let shift = x.sub_op(&broadcast_back(&max, x, ax)?)?;
        let shifted = first_max.select(&shift.stop_gradient()?, &shift)?;
        // exp(0) - 1 = 0 takes entry i out of the sum exactly; a NaN there (a non-finite row)
        // stays in and keeps the row NaN.
        let tail = shifted
            .exp()?
            .sub_op(&argmax.one_hot(n, x.aval().dtype, Some(ax as i64))?)?
            .reduce_sum(Some(&[ax as i64]))?;
        shifted.sub_op(&broadcast_back(&tail.log1p()?, x, ax)?)
    }

    /// Categorical cross-entropy loss `-Σ_k labels_k · log_softmax(logits)_k` reduced along `axis`.
    pub fn softmax_cross_entropy(
        logits: &TracerRef,
        labels: &TracerRef,
        axis: isize,
    ) -> Result<TracerRef, TraceError> {
        let ax = normalize_axis(logits.aval().shape.rank(), axis)?;
        let log_probs = log_softmax(logits, axis)?;
        labels
            .mul_op(&log_probs)?
            .reduce_sum(Some(&[ax as i64]))?
            .neg_op()
    }

    /// Soft-target (distillation) cross-entropy along `axis` between a teacher and a student
    /// given as logits: `-Σ_k p_k · log_softmax(student)_k` with `p = softmax(teacher)`, reduced
    /// over `axis`. `student` and `teacher` must have the same shape.
    ///
    /// The value is that of the plain composition. The gradient for `student` is not: through
    /// `log_softmax` it is `q - p` with `q = softmax(student)`, a difference of two separately
    /// rounded vectors, which is not 0 when `student == teacher` and loses most of its digits
    /// when they are close ("softmax cancellation", Barron, Gnanapragasam and Tschannen, *A Fix
    /// for Softmax and Cross-Entropy*). Following that paper, the student gradient is computed
    /// from the logit difference `δ = student - teacher` instead:
    ///
    /// ```text
    /// d = δ - Σ_i p_i δ_i,   s = d - log1p(Σ_i p_i expm1(d_i))   (s = log(q / p))
    /// u = p · expm1(s)   (= q - p),   g = u - q · Σ_i u_i
    /// ```
    ///
    /// and swapped in with `stop_gradient`: the loss is `-Σ p · sg(log_softmax(student)) + Σ
    /// sg(g) · (student - sg(student))`. So `g` is exactly 0 for equal logits and keeps its
    /// relative precision near them. Two details beyond the paper keep it from losing digits
    /// elsewhere: `δ` is first shifted by its value at the teacher's argmax (exact when the
    /// logits differ by a common offset, which softmax ignores), and an entry whose
    /// probabilities differ by more than a factor `e` (`|s| > 1`, where `q - p` does not
    /// cancel) or whose `s` is not finite uses `q - p` directly.
    ///
    /// The teacher's gradient is the plain one, through `p`. Only first derivatives are
    /// rewritten: derivatives of second or higher order with respect to `student` come out as 0,
    /// so use the plain composition for Hessians.
    pub fn soft_target_cross_entropy(
        student: &TracerRef,
        teacher: &TracerRef,
        axis: isize,
    ) -> Result<TracerRef, TraceError> {
        if student.aval().shape != teacher.aval().shape {
            return Err(TraceError::ShapeInferenceFailed {
                primitive: fj_core::Primitive::Sub,
                detail: format!(
                    "soft_target_cross_entropy: student shape {:?} != teacher shape {:?}",
                    student.aval().shape.dims,
                    teacher.aval().shape.dims
                ),
            });
        }
        let ax = normalize_axis(student.aval().shape.rank(), axis)?;
        let axes = [ax as i64];
        let row_sum = |v: &TracerRef| -> Result<TracerRef, TraceError> {
            broadcast_back(&v.reduce_sum(Some(&axes))?, student, ax)
        };

        let log_q = log_softmax(student, axis)?.stop_gradient()?;
        let p = softmax(teacher, axis)?;
        let loss = p.mul_op(&log_q)?.reduce_sum(Some(&axes))?.neg_op()?;
        if student.aval().shape.dims[ax] == 0 {
            return Ok(loss);
        }

        let q = softmax(student, axis)?.stop_gradient()?;
        let p_c = p.stop_gradient()?;
        let delta = student.stop_gradient()?.sub_op(&teacher.stop_gradient()?)?;
        let at_teacher_max = teacher.argmax(Some(ax as i64))?.one_hot(
            student.aval().shape.dims[ax],
            DType::Bool,
            Some(ax as i64),
        )?;
        let delta_ref = at_teacher_max
            .select(&delta, &zeros_like(&delta)?)?
            .reduce_sum(Some(&axes))?;
        let delta = delta.sub_op(&broadcast_back(&delta_ref, student, ax)?)?;
        let d = delta.sub_op(&row_sum(&p_c.mul_op(&delta)?)?)?;
        let s = d.sub_op(&row_sum(&p_c.mul_op(&d.expm1()?)?)?.log1p()?)?;
        let q_minus_p = q.sub_op(&p_c)?;
        // A NaN `s` fails the comparison and takes `q - p`.
        let u = s
            .abs()?
            .le(&scalar_like(&s, 1.0)?)?
            .select(&p_c.mul_op(&s.expm1()?)?, &q_minus_p)?;
        let g = u.sub_op(&q.mul_op(&row_sum(&u)?)?)?.stop_gradient()?;

        // student - sg(student) is 0 with gradient 1; a non-finite logit would make it NaN.
        let student_offset = student.is_finite()?.select(
            &student.sub_op(&student.stop_gradient()?)?,
            &zeros_like(student)?,
        )?;
        loss.add_op(&g.mul_op(&student_offset)?.reduce_sum(Some(&axes))?)
    }

    /// Standardizes `x` along `axis` to zero mean and unit variance: `(x - mean) / sqrt(var + eps)`.
    pub fn standardize(x: &TracerRef, axis: isize, eps: f64) -> Result<TracerRef, TraceError> {
        let rank = x.aval().shape.rank();
        if rank == 0 {
            return scalar_like(x, 0.0);
        }
        let ax = normalize_axis(rank, axis)?;
        let n = x.aval().shape.dims[ax] as f64;
        let inv_n = scalar_like(x, 1.0 / n)?;
        let mean_keep = reduce_keepdims_sum(x, ax)?.mul_op(&inv_n)?;
        let centered = x.sub_op(&mean_keep)?;
        let var_keep = reduce_keepdims_sum(&centered.square()?, ax)?.mul_op(&inv_n)?;
        let std_keep = var_keep.add_op(&scalar_like(x, eps)?)?.sqrt()?;
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
        let inv_n = scalar_like(x, 1.0 / n)?;
        let ms_keep = reduce_keepdims_sum(&x.square()?, ax)?.mul_op(&inv_n)?;
        let inv_rms = ms_keep.add_op(&scalar_like(x, eps)?)?.rsqrt()?;
        let mut out = x.mul_op(&inv_rms)?;
        if let Some(w) = weight {
            out = out.mul_op(w)?;
        }
        Ok(out)
    }

    /// Scaled dot-product attention (`softmax(Q @ K^T / sqrt(d_k) + mask, axis=-1) @ V`)
    /// for 2-D `query` (`[T_q, d_k]`), `key` (`[T_k, d_k]`), and `value` (`[T_k, d_v]`).
    pub fn dot_product_attention(
        query: &TracerRef,
        key: &TracerRef,
        value: &TracerRef,
        mask: Option<&TracerRef>,
    ) -> Result<TracerRef, TraceError> {
        let d_k = query.aval().shape.dims.last().copied().unwrap_or(1).max(1) as f64;
        let key_t = key.transpose(None)?;
        let scale = scalar_like(query, 1.0 / d_k.sqrt())?;
        let mut scores = query.dot(&key_t)?.mul_op(&scale)?;
        if let Some(m) = mask {
            scores = scores.add_op(m)?;
        }
        let weights = softmax(&scores, -1)?;
        weights.dot(value)
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
        sum.mul_op(&scalar_like(&sum, 0.5)?)
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

/// Canonical `jax.lax` primitive facade (`fj_api::lax`).
///
/// Provides both traced [`TracerRef`] free functions (which lower directly into canonical
/// `fj_core::Primitive` equations and compose with [`jit`], [`grad`], [`vmap`], [`hessian`],
/// and [`pmap`]) and re-exports of eager `fj_lax` evaluation modules (`array_creation`,
/// `einsum`, `linalg`, `eval_primitive`, `eval_primitive_multi`).
pub mod lax {
    use fj_core::DType;
    use fj_trace::{TraceError, TracerRef};
    use std::collections::BTreeMap;

    pub use fj_lax::{
        EvalError, array_creation, einsum, eval_primitive, eval_primitive_multi, linalg,
    };
    pub use fj_trace::{
        trace_cond as cond, trace_scan as scan, trace_switch as switch,
        trace_while_loop as while_loop,
    };

    macro_rules! def_lax_unary {
        ($(($fn_name:ident, $method:ident)),* $(,)?) => {
            $(
                #[doc = concat!("Traced `lax.", stringify!($fn_name), "(x)` operation on [`TracerRef`].")]
                pub fn $fn_name(x: &TracerRef) -> Result<TracerRef, TraceError> {
                    x.$method()
                }
            )*
        };
    }

    def_lax_unary!(
        (neg, neg_op),
        (abs, abs),
        (exp, exp),
        (expm1, expm1),
        (exp2, exp2),
        (log, log),
        (log1p, log1p),
        (log2, log2),
        (sqrt, sqrt),
        (rsqrt, rsqrt),
        (cbrt, cbrt),
        (floor, floor),
        (ceil, ceil),
        (round, round),
        (trunc, trunc),
        (sin, sin),
        (cos, cos),
        (tan, tan),
        (asin, asin),
        (acos, acos),
        (atan, atan),
        (deg2rad, deg2rad),
        (rad2deg, rad2deg),
        (sinh, sinh),
        (cosh, cosh),
        (tanh, tanh),
        (asinh, asinh),
        (acosh, acosh),
        (atanh, atanh),
        (sinc, sinc),
        (sign, sign),
        (square, square),
        (reciprocal, reciprocal),
        (logistic, logistic),
        (erf, erf),
        (erfc, erfc),
        (erf_inv, erf_inv),
        (lgamma, lgamma),
        (digamma, digamma),
        (bessel_i0e, bessel_i0e),
        (bessel_i1e, bessel_i1e),
        (is_finite, is_finite),
        (is_nan, is_nan),
        (is_inf, is_inf),
        (signbit, signbit),
        (conj, conj),
        (real, real),
        (imag, imag),
        (copy, copy),
        (stop_gradient, stop_gradient),
        (bitwise_not, bitwise_not),
        (population_count, population_count),
        (count_leading_zeros, count_leading_zeros),
        (count_trailing_zeros, count_trailing_zeros),
        (cholesky, cholesky),
        (det, det),
    );

    macro_rules! def_lax_binary {
        ($(($fn_name:ident, $method:ident)),* $(,)?) => {
            $(
                #[doc = concat!("Traced `lax.", stringify!($fn_name), "(lhs, rhs)` operation on [`TracerRef`].")]
                pub fn $fn_name(lhs: &TracerRef, rhs: &TracerRef) -> Result<TracerRef, TraceError> {
                    lhs.$method(rhs)
                }
            )*
        };
    }

    def_lax_binary!(
        (add, add_op),
        (sub, sub_op),
        (mul, mul_op),
        (div, div_op),
        (rem, rem_op),
        (max, max),
        (min, min),
        (pow, pow),
        (atan2, atan2),
        (hypot, hypot),
        (logaddexp, log_add_exp),
        (logaddexp2, log_add_exp2),
        (gcd, gcd),
        (lcm, lcm),
        (copysign, copy_sign),
        (ldexp, ldexp),
        (xlogy, xlogy),
        (xlog1py, xlog1py),
        (polygamma, polygamma),
        (heaviside, heaviside),
        (igamma, igamma),
        (igammac, igammac),
        (zeta, zeta),
        (nextafter, nextafter),
        (complex, complex),
        (eq, eq),
        (ne, ne),
        (lt, lt),
        (le, le),
        (gt, gt),
        (ge, ge),
        (bitwise_and, bitwise_and),
        (bitwise_or, bitwise_or),
        (bitwise_xor, bitwise_xor),
        (shift_left, shift_left),
        (shift_right_arithmetic, shift_right_arithmetic),
        (shift_right_logical, shift_right_logical),
        (dot, dot),
        (matmul, matmul),
        (solve, solve),
    );

    /// Traced `lax.integer_pow(x, y)` raising `x` to fixed integer power `y`.
    pub fn integer_pow(x: &TracerRef, y: i64) -> Result<TracerRef, TraceError> {
        x.integer_pow(y)
    }

    /// Traced `lax.convert_element_type(x, new_dtype)`.
    pub fn convert_element_type(x: &TracerRef, new_dtype: DType) -> Result<TracerRef, TraceError> {
        x.convert_element_type(new_dtype)
    }

    /// Traced `lax.bitcast_convert_type(x, new_dtype)`.
    pub fn bitcast_convert_type(x: &TracerRef, new_dtype: DType) -> Result<TracerRef, TraceError> {
        x.bitcast_convert_type(new_dtype)
    }

    /// Traced `lax.fma(a, b, c) = a * b + c`.
    pub fn fma(a: &TracerRef, b: &TracerRef, c: &TracerRef) -> Result<TracerRef, TraceError> {
        a.fma(b, c)
    }

    /// Traced `lax.betainc(a, b, x)`.
    pub fn betainc(a: &TracerRef, b: &TracerRef, x: &TracerRef) -> Result<TracerRef, TraceError> {
        a.betainc(b, x)
    }

    /// Traced `lax.select(pred, on_true, on_false)`.
    pub fn select(
        pred: &TracerRef,
        on_true: &TracerRef,
        on_false: &TracerRef,
    ) -> Result<TracerRef, TraceError> {
        pred.select(on_true, on_false)
    }

    /// Traced `lax.select_n(which, cases)`.
    pub fn select_n(which: &TracerRef, cases: &[&TracerRef]) -> Result<TracerRef, TraceError> {
        which.select_n(cases)
    }

    /// Traced `lax.clamp(min, x, max)`.
    pub fn clamp(min: &TracerRef, x: &TracerRef, max: &TracerRef) -> Result<TracerRef, TraceError> {
        x.clamp(min, max)
    }

    /// Traced `lax.reshape(x, new_shape)`.
    pub fn reshape(x: &TracerRef, new_shape: &[u32]) -> Result<TracerRef, TraceError> {
        x.reshape(new_shape)
    }

    /// Traced `lax.transpose(x, permutation)`.
    pub fn transpose(
        x: &TracerRef,
        permutation: Option<&[usize]>,
    ) -> Result<TracerRef, TraceError> {
        x.transpose(permutation)
    }

    /// Traced `lax.broadcast_in_dim(x, shape, broadcast_dimensions)`.
    pub fn broadcast_in_dim(
        x: &TracerRef,
        shape: &[u32],
        broadcast_dimensions: &[usize],
    ) -> Result<TracerRef, TraceError> {
        x.broadcast_in_dim(shape, broadcast_dimensions)
    }

    /// Traced `lax.broadcast(x, sizes)`.
    pub fn broadcast(x: &TracerRef, sizes: &[u32]) -> Result<TracerRef, TraceError> {
        x.broadcast(sizes)
    }

    /// Traced `lax.slice(x, start_indices, limit_indices, strides)`.
    pub fn slice(
        x: &TracerRef,
        start_indices: &[u32],
        limit_indices: &[u32],
        strides: Option<&[u32]>,
    ) -> Result<TracerRef, TraceError> {
        x.slice(start_indices, limit_indices, strides)
    }

    /// Traced `lax.dynamic_slice(x, start_indices, slice_sizes)`.
    pub fn dynamic_slice(
        x: &TracerRef,
        start_indices: &[&TracerRef],
        slice_sizes: &[u32],
    ) -> Result<TracerRef, TraceError> {
        x.dynamic_slice(start_indices, slice_sizes)
    }

    /// Traced `lax.dynamic_update_slice(x, update, start_indices)`.
    pub fn dynamic_update_slice(
        x: &TracerRef,
        update: &TracerRef,
        start_indices: &[&TracerRef],
    ) -> Result<TracerRef, TraceError> {
        x.dynamic_update_slice(update, start_indices)
    }

    /// Traced `lax.gather(x, start_indices, slice_sizes)`.
    pub fn gather(
        x: &TracerRef,
        start_indices: &TracerRef,
        slice_sizes: &[u32],
    ) -> Result<TracerRef, TraceError> {
        x.gather(start_indices, slice_sizes)
    }

    /// Traced `lax.scatter(x, scatter_indices, updates, mode)`.
    pub fn scatter(
        x: &TracerRef,
        scatter_indices: &TracerRef,
        updates: &TracerRef,
        mode: Option<&str>,
    ) -> Result<TracerRef, TraceError> {
        x.scatter(scatter_indices, updates, mode)
    }

    /// Traced `lax.concatenate(inputs, dimension)`.
    pub fn concatenate(inputs: &[&TracerRef], dimension: usize) -> Result<TracerRef, TraceError> {
        TracerRef::concatenate(inputs, dimension)
    }

    /// Traced `lax.pad(x, padding_value, padding_low, padding_high, padding_interior)`.
    pub fn pad(
        x: &TracerRef,
        padding_value: &TracerRef,
        padding_low: &[i64],
        padding_high: &[i64],
        padding_interior: &[u32],
    ) -> Result<TracerRef, TraceError> {
        x.pad(padding_value, padding_low, padding_high, padding_interior)
    }

    /// Traced `lax.rev(x, dimensions)`.
    pub fn rev(x: &TracerRef, dimensions: &[usize]) -> Result<TracerRef, TraceError> {
        x.rev(dimensions)
    }

    /// Traced `lax.squeeze(x, dimensions)`.
    pub fn squeeze(x: &TracerRef, dimensions: Option<&[i64]>) -> Result<TracerRef, TraceError> {
        x.squeeze(dimensions)
    }

    /// Traced `lax.expand_dims(x, axis)`.
    pub fn expand_dims(x: &TracerRef, axis: i64) -> Result<TracerRef, TraceError> {
        x.expand_dims(axis)
    }

    /// Traced `lax.tile(x, reps)`.
    pub fn tile(x: &TracerRef, reps: &[usize]) -> Result<TracerRef, TraceError> {
        x.tile(reps)
    }

    /// Traced `lax.reduce_sum(x, axes)`.
    pub fn reduce_sum(x: &TracerRef, axes: Option<&[i64]>) -> Result<TracerRef, TraceError> {
        x.reduce_sum(axes)
    }

    /// Traced `lax.reduce_max(x, axes)`.
    pub fn reduce_max(x: &TracerRef, axes: Option<&[i64]>) -> Result<TracerRef, TraceError> {
        x.reduce_max(axes)
    }

    /// Traced `lax.reduce_min(x, axes)`.
    pub fn reduce_min(x: &TracerRef, axes: Option<&[i64]>) -> Result<TracerRef, TraceError> {
        x.reduce_min(axes)
    }

    /// Traced `lax.reduce_prod(x, axes)`.
    pub fn reduce_prod(x: &TracerRef, axes: Option<&[i64]>) -> Result<TracerRef, TraceError> {
        x.reduce_prod(axes)
    }

    /// Traced `lax.cumsum(x, axis, reverse)`.
    pub fn cumsum(
        x: &TracerRef,
        axis: Option<i64>,
        reverse: bool,
    ) -> Result<TracerRef, TraceError> {
        x.cumsum(axis, reverse)
    }

    /// Traced `lax.cumprod(x, axis, reverse)`.
    pub fn cumprod(
        x: &TracerRef,
        axis: Option<i64>,
        reverse: bool,
    ) -> Result<TracerRef, TraceError> {
        x.cumprod(axis, reverse)
    }

    /// Traced `lax.cummax(x, axis, reverse)`.
    pub fn cummax(
        x: &TracerRef,
        axis: Option<i64>,
        reverse: bool,
    ) -> Result<TracerRef, TraceError> {
        x.cummax(axis, reverse)
    }

    /// Traced `lax.cummin(x, axis, reverse)`.
    pub fn cummin(
        x: &TracerRef,
        axis: Option<i64>,
        reverse: bool,
    ) -> Result<TracerRef, TraceError> {
        x.cummin(axis, reverse)
    }

    /// Traced `lax.sort(x, axis, descending)`.
    pub fn sort(
        x: &TracerRef,
        axis: Option<i64>,
        descending: bool,
    ) -> Result<TracerRef, TraceError> {
        x.sort(axis, descending)
    }

    /// Traced `lax.argsort(x, axis, descending)`.
    pub fn argsort(
        x: &TracerRef,
        axis: Option<i64>,
        descending: bool,
    ) -> Result<TracerRef, TraceError> {
        x.argsort(axis, descending)
    }

    /// Traced `lax.top_k(x, k)`.
    pub fn top_k(x: &TracerRef, k: u32) -> Result<(TracerRef, TracerRef), TraceError> {
        x.top_k(k)
    }

    /// Traced `lax.dot_general(lhs, rhs, lhs_contracting, rhs_contracting, lhs_batch, rhs_batch)`.
    pub fn dot_general(
        lhs: &TracerRef,
        rhs: &TracerRef,
        lhs_contracting_dims: &[usize],
        rhs_contracting_dims: &[usize],
        lhs_batch_dims: &[usize],
        rhs_batch_dims: &[usize],
    ) -> Result<TracerRef, TraceError> {
        lhs.dot_general(
            rhs,
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        )
    }

    /// Traced `lax.conv(lhs, rhs, strides, padding)`.
    pub fn conv(
        lhs: &TracerRef,
        rhs: &TracerRef,
        strides: &[usize],
        padding: &str,
    ) -> Result<TracerRef, TraceError> {
        lhs.conv(rhs, strides, padding)
    }

    /// Traced `lax.qr(x, full_matrices)`.
    pub fn qr(x: &TracerRef, full_matrices: bool) -> Result<(TracerRef, TracerRef), TraceError> {
        x.qr(full_matrices)
    }

    /// Traced `lax.lu(x)`.
    pub fn lu(x: &TracerRef) -> Result<(TracerRef, TracerRef, TracerRef), TraceError> {
        x.lu()
    }

    /// Traced `lax.svd(x, full_matrices)`.
    pub fn svd(
        x: &TracerRef,
        full_matrices: bool,
    ) -> Result<(TracerRef, TracerRef, TracerRef), TraceError> {
        x.svd(full_matrices)
    }

    /// Traced `lax.eigh(x)`.
    pub fn eigh(x: &TracerRef) -> Result<(TracerRef, TracerRef), TraceError> {
        x.eigh()
    }

    /// Traced `lax.triangular_solve(a, b, lower, transpose_a, unit_diagonal)`.
    pub fn triangular_solve(
        a: &TracerRef,
        b: &TracerRef,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracerRef, TraceError> {
        a.triangular_solve(b, lower, transpose_a, unit_diagonal)
    }

    /// Traced `lax.slogdet(x)`.
    pub fn slogdet(x: &TracerRef) -> Result<(TracerRef, TracerRef), TraceError> {
        x.slogdet()
    }

    /// Traced `lax.fft(x, fft_length)`.
    pub fn fft(x: &TracerRef, fft_length: Option<u32>) -> Result<TracerRef, TraceError> {
        x.fft(fft_length)
    }

    /// Traced `lax.ifft(x, fft_length)`.
    pub fn ifft(x: &TracerRef, fft_length: Option<u32>) -> Result<TracerRef, TraceError> {
        x.ifft(fft_length)
    }

    /// Traced `lax.rfft(x, fft_length)`.
    pub fn rfft(x: &TracerRef, fft_length: Option<u32>) -> Result<TracerRef, TraceError> {
        x.rfft(fft_length)
    }

    /// Traced `lax.irfft(x, fft_length)`.
    pub fn irfft(x: &TracerRef, fft_length: Option<u32>) -> Result<TracerRef, TraceError> {
        x.irfft(fft_length)
    }

    /// Traced `lax.psum(x, axis_name)`.
    pub fn psum(x: &TracerRef, axis_name: &str) -> Result<TracerRef, TraceError> {
        x.psum(axis_name)
    }

    /// Traced `lax.pmean(x, axis_name)`.
    pub fn pmean(x: &TracerRef, axis_name: &str) -> Result<TracerRef, TraceError> {
        x.pmean(axis_name)
    }

    /// Traced `lax.all_gather(x, axis_name)`.
    pub fn all_gather(x: &TracerRef, axis_name: &str) -> Result<TracerRef, TraceError> {
        x.all_gather(axis_name)
    }

    /// Traced `lax.all_to_all(x, axis_name)`.
    pub fn all_to_all(x: &TracerRef, axis_name: &str) -> Result<TracerRef, TraceError> {
        x.all_to_all(axis_name)
    }

    /// Traced `lax.axis_index(axis_name)` (in the currently active trace frame).
    pub fn axis_index(axis_name: &str) -> Result<TracerRef, TraceError> {
        TracerRef::axis_index(axis_name)
    }

    /// Traced `lax.bind_axis_index(x, axis_name)` bound to `x`'s trace frame.
    pub fn bind_axis_index(x: &TracerRef, axis_name: &str) -> Result<TracerRef, TraceError> {
        x.bind_axis_index(axis_name)
    }

    /// Traced `lax.custom_call(first_input, target, other_inputs, params)`.
    pub fn custom_call(
        first_input: &TracerRef,
        target: &str,
        other_inputs: &[&TracerRef],
        params: BTreeMap<String, String>,
    ) -> Result<TracerRef, TraceError> {
        first_input.custom_call(target, other_inputs, params)
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
    fn gradients_keep_the_input_dtype_through_grad_jit_and_vmap() {
        // Regression for frankenjax#2: reverse mode was seeded with an F64 one
        // and built its internal constants as F64, so under implicit promotion
        // an F32/BF16/F16 backward pass ran in F64 and returned F64 gradients
        // while the primal stayed F32. As in JAX, each gradient now has the
        // dtype of its input, on every grad entry point.
        fn lit(dtype: DType, x: f64) -> Literal {
            match dtype {
                DType::F32 => Literal::from_f32(x as f32),
                DType::BF16 => Literal::from_bf16_f64(x),
                DType::F16 => Literal::from_f16_f64(x),
                _ => Literal::from_f64(x),
            }
        }
        fn tensor(dtype: DType, dims: Vec<u32>, xs: &[f64]) -> Value {
            let lits = xs.iter().map(|&x| lit(dtype, x)).collect();
            Value::Tensor(fj_core::TensorValue::new(dtype, Shape { dims }, lits).expect("tensor"))
        }
        fn check(what: &str, got: &Value, dtype: DType, want: &[f64]) {
            assert_eq!(got.dtype(), dtype, "{what}: dtype");
            let vals = match got {
                Value::Tensor(t) => t.to_f64_vec().expect("float tensor"),
                Value::Scalar(lit) => vec![lit.as_f64().expect("float scalar")],
            };
            let tol = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 2e-3,
                _ => 2e-2,
            };
            assert_eq!(vals.len(), want.len(), "{what}: length");
            for (i, (&g, &w)) in vals.iter().zip(want).enumerate() {
                assert!((g - w).abs() <= tol, "{what}[{i}] = {g}, expected {w}");
            }
        }

        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let aval = ShapedArray {
                dtype,
                shape: Shape::vector(3),
            };
            // The report's loss: -sum(onehot * log_softmax(z)).
            let ce = make_jaxpr_fallible(
                |inp| {
                    let ls = nn::log_softmax(&inp[0], -1)?;
                    Ok(vec![inp[1].mul_op(&ls)?.reduce_sum(None)?.neg_op()?])
                },
                vec![aval.clone(), aval.clone()],
            )
            .expect("trace cross-entropy");
            let args = vec![
                tensor(dtype, vec![3], &[2.0, 1.0, 0.0]),
                tensor(dtype, vec![3], &[1.0, 0.0, 0.0]),
            ];
            // softmax([2, 1, 0]) - onehot(0), and -log softmax([2, 1, 0])[0].
            let want = [
                -0.3347590442251781,
                0.24472847105479764,
                0.09003057317038046,
            ];

            let (value, grads) = value_and_grad(ce.jaxpr.clone())
                .call(args.clone())
                .expect("value_and_grad");
            check(
                "value_and_grad value",
                &value[0],
                dtype,
                &[0.4076059644443803],
            );
            check("value_and_grad grad", &grads[0], dtype, &want);
            let grads = grad(ce.jaxpr.clone()).call(args.clone()).expect("grad");
            check("grad", &grads[0], dtype, &want);
            for transforms in [
                vec![Transform::Jit, Transform::Grad],
                vec![Transform::Grad, Transform::Jit],
            ] {
                let grads = compose(ce.jaxpr.clone(), transforms.clone())
                    .call(args.clone())
                    .expect("composed grad");
                check(&format!("{transforms:?}"), &grads[0], dtype, &want);
            }

            // grad(grad) lowers the inner grad through the symbolic builder.
            let cube = make_jaxpr_fallible(
                |inp| Ok(vec![inp[0].mul_op(&inp[0])?.mul_op(&inp[0])?]),
                vec![ShapedArray {
                    dtype,
                    shape: Shape::scalar(),
                }],
            )
            .expect("trace cube");
            let second = compose(cube.jaxpr, vec![Transform::Grad, Transform::Grad])
                .call(vec![Value::Scalar(lit(dtype, 2.0))])
                .expect("grad(grad(cube))");
            check("grad(grad(cube))", &second[0], dtype, &[12.0]);

            // vmap(grad) lowers through the symbolic gradient builder.
            let batch = tensor(dtype, vec![2, 3], &[2.0, 1.0, 0.0, 0.0, 2.0, 1.0]);
            let sum_sq = make_jaxpr_fallible(
                |inp| Ok(vec![inp[0].mul_op(&inp[0])?.reduce_sum(None)?]),
                vec![aval.clone()],
            )
            .expect("trace sum of squares");
            let grads = compose(sum_sq.jaxpr, vec![Transform::Vmap, Transform::Grad])
                .call(vec![batch.clone()])
                .expect("vmap(grad(sum of squares))");
            check(
                "vmap(grad(sum of squares))",
                &grads[0],
                dtype,
                &[4.0, 2.0, 0.0, 0.0, 4.0, 2.0],
            );
            let lse = make_jaxpr_fallible(|inp| Ok(vec![nn::logsumexp(&inp[0], -1)?]), vec![aval])
                .expect("trace logsumexp");
            let grads = compose(lse.jaxpr, vec![Transform::Vmap, Transform::Grad])
                .call(vec![batch])
                .expect("vmap(grad(logsumexp))");
            check(
                "vmap(grad(logsumexp))",
                &grads[0],
                dtype,
                &[
                    0.6652409557748219,
                    0.24472847105479764,
                    0.09003057317038046,
                    0.09003057317038046,
                    0.6652409557748219,
                    0.24472847105479764,
                ],
            );
        }
    }

    fn float_tensor(dtype: DType, dims: Vec<u32>, xs: &[f64]) -> Value {
        let lits = xs
            .iter()
            .map(|&x| match dtype {
                DType::F32 => Literal::from_f32(x as f32),
                _ => Literal::from_f64(x),
            })
            .collect();
        Value::Tensor(fj_core::TensorValue::new(dtype, Shape { dims }, lits).expect("tensor"))
    }

    fn float_values(what: &str, v: &Value, dtype: DType) -> Vec<f64> {
        assert_eq!(v.dtype(), dtype, "{what}: dtype");
        match v {
            Value::Tensor(t) => t.to_f64_vec().expect("float tensor"),
            Value::Scalar(lit) => vec![lit.as_f64().expect("float scalar")],
        }
    }

    fn assert_rel_close(what: &str, got: &[f64], want: &[f64], rel: f64) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= rel * w.abs(),
                "{what}[{i}] = {g:e}, expected {w:e} (rel tol {rel:e})"
            );
        }
    }

    /// The softmax-collapse cases from frankenjax#1: one logit ahead of the other two by a
    /// margin `m` past the point where `1 + 2e^-m` rounds to 1 (~16.6 in f32, ~36.7 in f64).
    /// `a = log1p(2e^-m) ≈ 2e^-m` is the exact loss of a hard target on the leading logit
    /// (mpmath), and to the precision checked here also the softmax mass of the other two.
    const COLLAPSE_CASES: [(DType, f64, f64, f64); 2] = [
        (DType::F32, 18.0, 3.045_995_8e-8, 1e-5),
        (DType::F64, 40.0, 8.496_708_510_583_178e-18, 1e-12),
    ];

    #[test]
    fn log_softmax_and_logsumexp_avoid_softmax_collapse() {
        // Regression for frankenjax#1 ("softmax collapse", Barron, Gnanapragasam and Tschannen,
        // "A Fix for Softmax and Cross-Entropy"). log_softmax(z)[argmax] and the hard-target loss
        // -Σ onehot·log_softmax(z) came out as exactly 0, and logsumexp([0, -m, -m]) as exactly
        // 0. The target gradient was right only because it flowed back through the max shift.
        for (dtype, m, a, rel) in COLLAPSE_CASES {
            let z = [m / 2.0, -m / 2.0, -m / 2.0];
            let aval = ShapedArray {
                dtype,
                shape: Shape::vector(3),
            };
            let lsm = make_jaxpr_fallible(
                |inp| Ok(vec![nn::log_softmax(&inp[0], -1)?]),
                vec![aval.clone()],
            )
            .expect("trace log_softmax");
            let out = jit(lsm.jaxpr.clone())
                .call(vec![float_tensor(dtype, vec![3], &z)])
                .expect("log_softmax");
            assert_rel_close(
                &format!("{dtype:?} log_softmax"),
                &float_values("log_softmax", &out[0], dtype),
                &[-a, -m, -m],
                rel,
            );
            // Forward mode: d log_softmax / d z_0 = onehot_0 - softmax, entry 0 = 1 - p_0.
            let tangents = jvp_tangents(
                &lsm.jaxpr,
                &[float_tensor(dtype, vec![3], &z)],
                &[float_tensor(dtype, vec![3], &[1.0, 0.0, 0.0])],
            );
            assert_rel_close(
                &format!("{dtype:?} jvp(log_softmax)"),
                &float_values("jvp", &tangents[0], dtype),
                &[a, -1.0, -1.0],
                rel,
            );

            let ce = make_jaxpr_fallible(
                |inp| {
                    let ls = nn::log_softmax(&inp[0], -1)?;
                    Ok(vec![inp[1].mul_op(&ls)?.reduce_sum(None)?.neg_op()?])
                },
                vec![aval.clone(), aval.clone()],
            )
            .expect("trace cross-entropy");
            let args = vec![
                float_tensor(dtype, vec![3], &z),
                float_tensor(dtype, vec![3], &[1.0, 0.0, 0.0]),
            ];
            // softmax - onehot: minus the other entries' mass at the target, summing to zero.
            let want = [-a, a / 2.0, a / 2.0];
            let (value, grads) = value_and_grad(ce.jaxpr.clone())
                .call(args.clone())
                .expect("value_and_grad");
            assert_rel_close(
                &format!("{dtype:?} loss"),
                &float_values("loss", &value[0], dtype),
                &[a],
                rel,
            );
            let mut all_grads = vec![("value_and_grad".to_owned(), grads[0].clone())];
            let grads = grad(ce.jaxpr.clone()).call(args.clone()).expect("grad");
            all_grads.push(("grad".to_owned(), grads[0].clone()));
            for transforms in [
                vec![Transform::Jit, Transform::Grad],
                vec![Transform::Grad, Transform::Jit],
            ] {
                let grads = compose(ce.jaxpr.clone(), transforms.clone())
                    .call(args.clone())
                    .expect("composed grad");
                all_grads.push((format!("{transforms:?}"), grads[0].clone()));
            }
            for (what, g) in &all_grads {
                let g = float_values(what, g, dtype);
                assert_rel_close(&format!("{dtype:?} {what}"), &g, &want, rel);
                assert_eq!(g.iter().sum::<f64>(), 0.0, "{dtype:?} {what}: {g:?}");
            }

            // vmap(grad) goes through the symbolic gradient builder. The label is fixed to
            // class 0, the argmax of the first row but not of the second.
            let ce0 = make_jaxpr_fallible(
                |inp| {
                    let label = TracerRef::literal(Literal::I64(0))?.one_hot(3, dtype, None)?;
                    let ls = nn::log_softmax(&inp[0], -1)?;
                    Ok(vec![label.mul_op(&ls)?.reduce_sum(None)?.neg_op()?])
                },
                vec![aval.clone()],
            )
            .expect("trace fixed-label cross-entropy");
            let batch = float_tensor(dtype, vec![2, 3], &[z[0], z[1], z[2], z[1], z[0], z[2]]);
            let grads = compose(ce0.jaxpr, vec![Transform::Vmap, Transform::Grad])
                .call(vec![batch])
                .expect("vmap(grad(cross-entropy))");
            assert_rel_close(
                &format!("{dtype:?} vmap(grad)"),
                &float_values("vmap(grad)", &grads[0], dtype),
                &[-a, a / 2.0, a / 2.0, -1.0, 1.0, a / 2.0],
                rel,
            );

            // logsumexp([0, -m, -m]) = log1p(2e^-m), and its gradient is softmax.
            let lse = make_jaxpr_fallible(|inp| Ok(vec![nn::logsumexp(&inp[0], -1)?]), vec![aval])
                .expect("trace logsumexp");
            let shifted = float_tensor(dtype, vec![3], &[0.0, -m, -m]);
            let (value, grads) = value_and_grad(lse.jaxpr)
                .call(vec![shifted])
                .expect("value_and_grad(logsumexp)");
            assert_rel_close(
                &format!("{dtype:?} logsumexp"),
                &float_values("logsumexp", &value[0], dtype),
                &[a],
                rel,
            );
            assert_rel_close(
                &format!("{dtype:?} grad(logsumexp)"),
                &float_values("grad(logsumexp)", &grads[0], dtype),
                &[1.0, a / 2.0, a / 2.0],
                rel,
            );
        }
    }

    /// The output tangents of `jvp(jaxpr)` at `primals` along `tangents`.
    fn jvp_tangents(jaxpr: &Jaxpr, primals: &[Value], tangents: &[Value]) -> Vec<Value> {
        jvp(jaxpr, primals, tangents).expect("jvp").tangents
    }

    #[test]
    fn softmax_gradient_survives_softmax_collapse() {
        // nn::softmax keeps its max outside stop_gradient: once p_0 rounds to 1, the part of
        // dp_0/dz_0 = p_0(1 - p_0) that flows through exp(z_0 - max) cancels to 0, and the max's
        // gradient is what restores it. This guards that path: the gradient of p_0 must be
        // [p_0(1 - p_0), -p_0 p_1, -p_0 p_2] ≈ [a, -a/2, -a/2], summing to zero.
        for (dtype, m, a, rel) in COLLAPSE_CASES {
            let aval = ShapedArray {
                dtype,
                shape: Shape::vector(3),
            };
            let p0 = make_jaxpr_fallible(
                |inp| {
                    let label = TracerRef::literal(Literal::I64(0))?.one_hot(3, dtype, None)?;
                    let p = nn::softmax(&inp[0], -1)?;
                    Ok(vec![label.mul_op(&p)?.reduce_sum(None)?])
                },
                vec![aval],
            )
            .expect("trace softmax");
            let z = vec![float_tensor(dtype, vec![3], &[m / 2.0, -m / 2.0, -m / 2.0])];
            for transforms in [vec![Transform::Grad], vec![Transform::Jit, Transform::Grad]] {
                let g = compose(p0.jaxpr.clone(), transforms.clone())
                    .call(z.clone())
                    .expect("grad(softmax[0])");
                let g = float_values("grad(softmax[0])", &g[0], dtype);
                let what = format!("{dtype:?} {transforms:?} grad(softmax[0])");
                assert_rel_close(&what, &g, &[a, -a / 2.0, -a / 2.0], rel);
                assert_eq!(g.iter().sum::<f64>(), 0.0, "{what}: {g:?}");
            }
        }
    }

    /// `x`'s last-axis reduction broadcast back onto `x`'s shape.
    fn broadcast_last(
        reduced: &TracerRef,
        x: &TracerRef,
    ) -> Result<TracerRef, fj_trace::TraceError> {
        let dims = x.aval().shape.dims.clone();
        let kept: Vec<usize> = (0..dims.len() - 1).collect();
        reduced.broadcast_in_dim(&dims, &kept)
    }

    /// The previous `nn::log_softmax`, `(x - max) - log(Σ exp(x - max))`, along the last axis.
    fn previous_log_softmax(x: &TracerRef) -> Result<TracerRef, fj_trace::TraceError> {
        let last = [x.aval().shape.rank() as i64 - 1];
        let shifted = x.sub_op(&broadcast_last(&x.reduce_max(Some(&last))?, x)?)?;
        let sum = shifted.exp()?.reduce_sum(Some(&last))?;
        shifted.sub_op(&broadcast_last(&sum.log()?, x)?)
    }

    /// The previous `nn::logsumexp`, `max + log(Σ exp(x - max))`, along the last axis.
    fn previous_logsumexp(x: &TracerRef) -> Result<TracerRef, fj_trace::TraceError> {
        let last = [x.aval().shape.rank() as i64 - 1];
        let max = x.reduce_max(Some(&last))?;
        let shifted = x.sub_op(&broadcast_last(&max, x)?)?;
        max.add_op(&shifted.exp()?.reduce_sum(Some(&last))?.log()?)
    }

    #[test]
    fn log_softmax_and_logsumexp_match_the_previous_formula_away_from_collapse() {
        // Outside the collapse regime the rewrite only changes rounding: values, gradients and
        // (f64) symbolic Hessians stay within a few ulp of the previous composition. The rows
        // cover a clear winner, a three-way tie for the max, a near-uniform row, a heavily
        // masked logit and large magnitudes.
        type Op = fn(&TracerRef) -> Result<TracerRef, fj_trace::TraceError>;
        let rows: [[f64; 5]; 5] = [
            [2.0, 1.0, 0.0, -0.5, 0.25],
            [1.5, 1.5, 0.0, -1.0, 1.5],
            [0.01, -0.02, 0.0, 0.015, -0.005],
            [3.0, -1e4, 0.5, 2.5, -2.0],
            [1000.0, 999.0, 990.0, 1000.5, 995.0],
        ];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let new_lsm: Op = |x| nn::log_softmax(x, -1);
        let new_lse: Op = |x| nn::logsumexp(x, -1);
        let cases = [
            ("log_softmax", new_lsm, previous_log_softmax as Op),
            ("logsumexp", new_lse, previous_logsumexp as Op),
        ];
        for (dtype, tol) in [
            (DType::F64, 64.0 * f64::EPSILON),
            (DType::F32, 64.0 * f64::from(f32::EPSILON)),
        ] {
            let close = |what: &str, got: &[f64], want: &[f64]| {
                assert_eq!(got.len(), want.len(), "{dtype:?} {what}: length");
                for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (g - w).abs() <= tol * w.abs().max(1.0),
                        "{dtype:?} {what}[{i}] = {g:e}, previous formula {w:e}"
                    );
                }
            };
            let x_aval = ShapedArray {
                dtype,
                shape: Shape { dims: vec![5, 5] },
            };
            let x = float_tensor(dtype, vec![5, 5], &flat);
            for (what, new, old) in cases {
                // Weights w for Σ w · op(x), a second input with op(x)'s shape ([5, 5] or [5]),
                // so that the gradient covers more than one cotangent direction.
                let w_dims = if what == "log_softmax" {
                    vec![5, 5]
                } else {
                    vec![5]
                };
                let w_len = w_dims.iter().product::<u32>() as usize;
                let w_vals: Vec<f64> = (0..w_len)
                    .map(|k| ((k * 7 % 11) as f64 - 5.0) / 4.0)
                    .collect();
                let w = float_tensor(dtype, w_dims.clone(), &w_vals);
                let w_aval = ShapedArray {
                    dtype,
                    shape: Shape { dims: w_dims },
                };
                let forward = |op: Op| {
                    let closed =
                        make_jaxpr_fallible(|inp| Ok(vec![op(&inp[0])?]), vec![x_aval.clone()])
                            .expect("trace");
                    let out = jit(closed.jaxpr).call(vec![x.clone()]).expect("forward");
                    float_values(what, &out[0], dtype)
                };
                close(what, &forward(new), &forward(old));

                let weighted = |op: Op| {
                    make_jaxpr_fallible(
                        |inp| Ok(vec![op(&inp[0])?.mul_op(&inp[1])?.reduce_sum(None)?]),
                        vec![x_aval.clone(), w_aval.clone()],
                    )
                    .expect("trace weighted sum")
                    .jaxpr
                };
                let gradient = |op: Op| {
                    let g = grad(weighted(op))
                        .call(vec![x.clone(), w.clone()])
                        .expect("grad");
                    float_values(what, &g[0], dtype)
                };
                close(&format!("grad {what}"), &gradient(new), &gradient(old));

                if dtype == DType::F64 {
                    // Second order through the symbolic gradient, over both inputs.
                    let hessian_of = |op: Op| {
                        let h = hessian(weighted(op))
                            .call_symbolic(vec![x.clone(), w.clone()])
                            .expect("symbolic hessian");
                        float_values(what, &h, dtype)
                    };
                    close(
                        &format!("hessian {what}"),
                        &hessian_of(new),
                        &hessian_of(old),
                    );
                }
            }
        }
    }

    /// `(loss, student gradient, teacher gradient)` of `loss_fn(student, teacher)` summed over rows.
    fn distillation_value_and_grads(
        loss_fn: fn(&TracerRef, &TracerRef) -> Result<TracerRef, fj_trace::TraceError>,
        dtype: DType,
        dims: &[u32],
        student: &[f64],
        teacher: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let aval = ShapedArray {
            dtype,
            shape: Shape {
                dims: dims.to_vec(),
            },
        };
        let closed = make_jaxpr_fallible(
            |inp| Ok(vec![loss_fn(&inp[0], &inp[1])?.reduce_sum(None)?]),
            vec![aval.clone(), aval],
        )
        .expect("trace distillation loss");
        let args = vec![
            float_tensor(dtype, dims.to_vec(), student),
            float_tensor(dtype, dims.to_vec(), teacher),
        ];
        let value = jit(closed.jaxpr.clone()).call(args.clone()).expect("loss");
        let grads = grad(closed.jaxpr)
            .with_argnums(&[0, 1])
            .call(args)
            .expect("grad");
        (
            float_values("loss", &value[0], dtype),
            float_values("student grad", &grads[0], dtype),
            float_values("teacher grad", &grads[1], dtype),
        )
    }

    fn soft_target_ce(
        student: &TracerRef,
        teacher: &TracerRef,
    ) -> Result<TracerRef, fj_trace::TraceError> {
        nn::soft_target_cross_entropy(student, teacher, -1)
    }

    /// The plain composition `-Σ softmax(teacher) · log_softmax(student)`.
    fn plain_soft_target_ce(
        student: &TracerRef,
        teacher: &TracerRef,
    ) -> Result<TracerRef, fj_trace::TraceError> {
        let last = [student.aval().shape.rank() as i64 - 1];
        nn::softmax(teacher, -1)?
            .mul_op(&nn::log_softmax(student, -1)?)?
            .reduce_sum(Some(&last))?
            .neg_op()
    }

    #[test]
    fn soft_target_cross_entropy_avoids_softmax_cancellation() {
        // Regression for frankenjax#1 ("softmax cancellation"). Through the plain composition the
        // student gradient q - p was ~1e-9 (f32) / ~1e-17 (f64) for equal logits and lost most of
        // its digits for nearly equal ones. References: q - p in 80-digit arithmetic from the
        // (rounded) logits below.
        let teacher = [2.0, 1.0, 0.0];
        let f32_ulp = 2f64.powi(-23);
        let cases: [(DType, [f64; 3], [f64; 3], f64); 7] = [
            (DType::F32, teacher, [0.0; 3], 0.0),
            (DType::F64, teacher, [0.0; 3], 0.0),
            // A common offset changes nothing (softmax ignores it).
            (DType::F32, [12.0, 11.0, 10.0], [0.0; 3], 0.0),
            (DType::F64, [12.0, 11.0, 10.0], [0.0; 3], 0.0),
            // The issue's f32 example: one ulp below 2 and above 1.
            (
                DType::F32,
                [2.0 - f32_ulp, 1.0 + f32_ulp, 0.0],
                [
                    -4.595_504_334_917_698e-8,
                    4.144_190_159_319_526_5e-8,
                    4.513_141_755_981_719e-9,
                ],
                1e-5,
            ),
            // The same in f64, and with a common offset added.
            (
                DType::F64,
                [2.0 - 2f64.powi(-51), 1.0 + 2f64.powi(-51), 0.0],
                [
                    -1.711_958_701_575_379_4e-16,
                    1.543_831_056_326_642e-16,
                    1.681_276_452_487_373_7e-17,
                ],
                1e-12,
            ),
            (
                DType::F64,
                [12.0 - 2f64.powi(-49), 11.0 + 2f64.powi(-49), 10.0],
                [
                    -6.847_834_806_301_521e-16,
                    6.175_324_225_306_572e-16,
                    6.725_105_809_949_488e-17,
                ],
                1e-12,
            ),
        ];
        for (dtype, student, want, rel) in cases {
            let (loss, g_student, g_teacher) =
                distillation_value_and_grads(soft_target_ce, dtype, &[3], &student, &teacher);
            let what = format!("{dtype:?} student {student:?}");
            if rel == 0.0 {
                assert_eq!(g_student, want, "{what}: gradient must be exactly 0");
            } else {
                assert_rel_close(&what, &g_student, &want, rel);
            }
            // Same loss and teacher gradient as the plain composition.
            let (plain_loss, _, plain_g_teacher) =
                distillation_value_and_grads(plain_soft_target_ce, dtype, &[3], &student, &teacher);
            assert_eq!(loss, plain_loss, "{what}: loss");
            assert_rel_close(
                &format!("{what} teacher grad"),
                &g_teacher,
                &plain_g_teacher,
                1e-12,
            );
        }

        // Batched, through jit, vmap and grad: rows with equal logits have exactly zero gradient.
        for dtype in [DType::F32, DType::F64] {
            let aval = ShapedArray {
                dtype,
                shape: Shape::vector(3),
            };
            let row_loss = make_jaxpr_fallible(
                |inp| Ok(vec![nn::soft_target_cross_entropy(&inp[0], &inp[1], 0)?]),
                vec![aval.clone(), aval],
            )
            .expect("trace soft-target cross-entropy");
            let students = float_tensor(dtype, vec![2, 3], &[0.5, -1.0, 3.0, 2.0, 1.0, 0.0]);
            let teachers = float_tensor(dtype, vec![2, 3], &[0.5, -1.0, 3.0, 2.0, 1.0, 0.0]);
            for transforms in [
                vec![Transform::Vmap, Transform::Grad],
                vec![Transform::Jit, Transform::Vmap, Transform::Grad],
            ] {
                let g = compose(row_loss.jaxpr.clone(), transforms.clone())
                    .call(vec![students.clone(), teachers.clone()])
                    .expect("batched gradient");
                assert_eq!(
                    float_values("batched gradient", &g[0], dtype),
                    [0.0; 6],
                    "{dtype:?} {transforms:?}"
                );
            }
        }
    }

    #[test]
    fn soft_target_cross_entropy_matches_plain_gradient_away_from_cancellation() {
        // Where q - p does not cancel, the rewritten gradient agrees with the plain composition.
        // Rows: unrelated logits, a confident teacher, a student far from the teacher, a shared
        // offset with a large difference, a tie, and (row 6) a teacher so confident that p
        // underflows in f32, where the paper's p · expm1(s) alone would overflow or lose digits.
        let teacher: [f64; 18] = [
            0.3, -1.2, 2.0, //
            8.0, 0.0, -3.0, //
            0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, //
            1.5, 1.5, -0.5, //
            0.0, -100.0, -100.0,
        ];
        let student: [f64; 18] = [
            -0.7, 0.4, 1.1, //
            1.0, 0.5, 0.0, //
            5.0, -5.0, 0.0, //
            41.0, 40.0, 42.5, //
            1.5, 0.5, 1.5, //
            0.0, 0.0, 0.0,
        ];
        for (dtype, tol) in [(DType::F64, 1e-13), (DType::F32, 1e-5)] {
            let (loss, g, g_teacher) =
                distillation_value_and_grads(soft_target_ce, dtype, &[6, 3], &student, &teacher);
            let (plain_loss, plain_g, plain_g_teacher) = distillation_value_and_grads(
                plain_soft_target_ce,
                dtype,
                &[6, 3],
                &student,
                &teacher,
            );
            assert_eq!(loss, plain_loss, "{dtype:?} loss");
            for (what, got, want) in [
                ("student", &g, &plain_g),
                ("teacher", &g_teacher, &plain_g_teacher),
            ] {
                for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= tol * b.abs().max(1.0),
                        "{dtype:?} {what} grad[{i}] = {a:e}, plain {b:e}"
                    );
                }
            }
            // Row 6: q - p = [-2/3, 1/3, 1/3].
            assert_rel_close(
                &format!("{dtype:?} confident-teacher row"),
                &g[15..],
                &[-2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                tol,
            );
        }
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

    #[test]
    fn wave15_lax_module_and_runtime_ledger_facade_work() {
        // 1. Traced lax::integer_pow, lax::lgamma, lax::add composed with hessian_fn and symbolic hessian
        let x = Value::scalar_f64(2.5);
        let h_pow_out = hessian_fn(
            |inputs| {
                let y = lax::integer_pow(&inputs[0], 3)?;
                Ok(vec![y])
            },
            vec![x.clone()],
        )
        .expect("hessian_fn lax::integer_pow");
        // d^2/dx^2 (x^3) = 6x = 15.0
        let h_pow_val = h_pow_out.as_tensor().unwrap().to_f64_vec().unwrap()[0];
        assert!((h_pow_val - 15.0).abs() < 1e-10);

        let h_lgamma_out = hessian_fn(|inputs| Ok(vec![lax::lgamma(&inputs[0])?]), vec![x])
            .expect("hessian_fn lax::lgamma");
        let expected_trigamma = std::f64::consts::PI.powi(2) / 2.0 - 4.0 - 4.0 / 9.0;
        let h_lgamma_val = h_lgamma_out.as_tensor().unwrap().to_f64_vec().unwrap()[0];
        assert!((h_lgamma_val - expected_trigamma).abs() < 1e-10);

        // 2. Traced lax::psum composed with pmap_fn across 4 devices
        let vec_in = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let pmap_out = pmap_fn(
            |inputs| {
                let sq = lax::square(&inputs[0])?;
                Ok(vec![lax::psum(&sq, "i")?])
            },
            4,
            vec![vec_in],
        )
        .expect("pmap_fn with lax::psum");
        assert_eq!(
            pmap_out[0].as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![30.0, 30.0, 30.0, 30.0]
        );

        // 3. Eager lax::array_creation and runtime/ledger facade types
        let eye2 =
            lax::array_creation::eye(2, None, 0, DType::F64).expect("lax::array_creation::eye");
        assert_eq!(eye2.as_tensor().unwrap().shape.dims, vec![2, 2]);

        let buf = Buffer::new(vec![10, 20, 30, 40], DeviceId(0));
        let view = BufferView::from_buffer(&buf).slice(1, 2).expect("slice");
        assert_eq!(view.as_bytes(), &[20, 30]);
        assert_eq!(Platform::Cpu.as_str(), "cpu");

        let mut cp = ConformalPredictor::new(0.9, 3);
        cp.observe(0.05);
        cp.observe(0.10);
        cp.observe(0.15);
        assert!(cp.is_calibrated());
        let est = cp.calibrated_posterior(0.2);
        assert!(est.used_conformal);

        let mut ep = EProcess::new(10.0);
        ep.update(2.5);
        assert!(!ep.rejected());
        assert_eq!(
            recommend_action(0.05, &LossMatrix::default()),
            DecisionAction::Keep
        );

        // 4. Symbolic grad_to_jaxpr and hvp_to_jaxpr on 2D nn::softmax, nn::layer_norm, nn::rms_norm
        let mat2x3 = fj_core::TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
                .into_iter()
                .map(Literal::from_f64)
                .collect(),
        )
        .map(Value::Tensor)
        .unwrap();
        let dmat2x3 = fj_core::TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                .into_iter()
                .map(Literal::from_f64)
                .collect(),
        )
        .map(Value::Tensor)
        .unwrap();

        let softmax_loss_jaxpr = trace_jaxpr_for_values(
            |inputs| {
                let sm = nn::softmax(&inputs[0], -1)?;
                Ok(vec![nn::l2_loss(&sm)?])
            },
            std::slice::from_ref(&mat2x3),
        )
        .expect("trace 2D softmax l2_loss");
        let sym_grad_sm = grad_to_jaxpr(&softmax_loss_jaxpr).expect("grad_to_jaxpr 2D softmax");
        let sym_hvp_sm = hvp_to_jaxpr(&softmax_loss_jaxpr).expect("hvp_to_jaxpr 2D softmax");
        let sym_grad_out = jit(sym_grad_sm)
            .call(vec![mat2x3.clone()])
            .expect("eval sym_grad_sm")[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let tape_grad_out = grad(softmax_loss_jaxpr)
            .call(vec![mat2x3.clone()])
            .expect("eval tape grad sm")[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        for (a, b) in sym_grad_out.iter().zip(tape_grad_out.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "2D softmax grad mismatch: {a} vs {b}"
            );
        }
        let hvp_sm_out = jit(sym_hvp_sm)
            .call(vec![mat2x3.clone(), dmat2x3])
            .expect("eval sym_hvp_sm");
        assert_eq!(hvp_sm_out[0].as_tensor().unwrap().shape.dims, vec![2, 3]);

        let ln_loss_jaxpr = trace_jaxpr_for_values(
            |inputs| {
                let ln = nn::layer_norm(&inputs[0], -1, 1e-5, None, None)?;
                Ok(vec![nn::l2_loss(&ln)?])
            },
            std::slice::from_ref(&mat2x3),
        )
        .expect("trace 2D layer_norm l2_loss");
        let sym_grad_ln = grad_to_jaxpr(&ln_loss_jaxpr).expect("grad_to_jaxpr 2D layer_norm");
        let sym_ln_out = jit(sym_grad_ln)
            .call(vec![mat2x3])
            .expect("eval sym_grad_ln");
        assert_eq!(sym_ln_out[0].as_tensor().unwrap().shape.dims, vec![2, 3]);
    }

    #[test]
    fn vmap_grad_softmax_and_layer_norm_compose_cleanly() {
        use fj_core::TensorValue;

        let row0 = TensorValue::new_f64_values(Shape::vector(3), vec![0.5, 1.2, -0.3])
            .map(Value::Tensor)
            .unwrap();
        let row1 = TensorValue::new_f64_values(Shape::vector(3), vec![-1.0, 0.0, 2.0])
            .map(Value::Tensor)
            .unwrap();
        let batch_input = TensorValue::new_f64_values(
            Shape { dims: vec![2, 3] },
            vec![0.5, 1.2, -0.3, -1.0, 0.0, 2.0],
        )
        .map(Value::Tensor)
        .unwrap();

        let softmax_loss_jaxpr = trace_jaxpr_for_values(
            |inputs| {
                let sm = nn::softmax(&inputs[0], -1)?;
                Ok(vec![nn::l2_loss(&sm)?])
            },
            std::slice::from_ref(&row0),
        )
        .expect("trace 1D softmax l2_loss");

        let sym_grad_sm = grad_to_jaxpr(&softmax_loss_jaxpr).expect("grad_to_jaxpr 1D softmax");
        let batched_grads = vmap(sym_grad_sm)
            .call(vec![batch_input.clone()])
            .expect("vmap(grad_to_jaxpr(softmax)) should succeed");
        assert_eq!(batched_grads[0].as_tensor().unwrap().shape.dims, vec![2, 3]);

        // Also test via composable vmap(jaxpr).compose_grad()
        let composed_grads = vmap(softmax_loss_jaxpr.clone())
            .compose_grad()
            .call(vec![batch_input.clone()])
            .expect("vmap(softmax).compose_grad() should succeed");
        assert_eq!(
            composed_grads[0].as_tensor().unwrap().shape.dims,
            vec![2, 3]
        );

        // Compare each row against unbatched grad(softmax)
        let g0 = grad(softmax_loss_jaxpr.clone())
            .call(vec![row0.clone()])
            .expect("grad row0")[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let g1 = grad(softmax_loss_jaxpr)
            .call(vec![row1])
            .expect("grad row1")[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let got = batched_grads[0].as_tensor().unwrap().to_f64_vec().unwrap();
        for (a, b) in got.iter().zip(g0.iter().chain(g1.iter())) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }

        // Also test vmap(grad_to_jaxpr(layer_norm))
        let ln_loss_jaxpr = trace_jaxpr_for_values(
            |inputs| {
                let ln = nn::layer_norm(&inputs[0], -1, 1e-5, None, None)?;
                Ok(vec![nn::l2_loss(&ln)?])
            },
            std::slice::from_ref(&row0),
        )
        .expect("trace 1D layer_norm l2_loss");
        let sym_grad_ln = grad_to_jaxpr(&ln_loss_jaxpr).expect("grad_to_jaxpr 1D layer_norm");
        let batched_ln_grads = vmap(sym_grad_ln)
            .call(vec![batch_input])
            .expect("vmap(grad_to_jaxpr(layer_norm)) should succeed");
        assert_eq!(
            batched_ln_grads[0].as_tensor().unwrap().shape.dims,
            vec![2, 3]
        );
    }

    #[test]
    fn nn_f32_dtype_preservation_and_full_parity() {
        let x_f32 = Value::Tensor(
            fj_core::TensorValue::new(
                DType::F32,
                Shape::vector(4),
                vec![
                    Literal::from_f32(-2.0),
                    Literal::from_f32(-0.5),
                    Literal::from_f32(0.5),
                    Literal::from_f32(2.0),
                ],
            )
            .unwrap(),
        );
        let f32_outs = jit_fn(
            |xs| {
                Ok(vec![
                    nn::identity(&xs[0])?,
                    nn::relu(&xs[0])?,
                    nn::leaky_relu(&xs[0], 0.01)?,
                    nn::relu6(&xs[0])?,
                    nn::elu(&xs[0], 1.0)?,
                    nn::selu(&xs[0])?,
                    nn::celu(&xs[0], 1.0)?,
                    nn::gelu(&xs[0], false)?,
                    nn::gelu(&xs[0], true)?,
                    nn::squareplus(&xs[0], 4.0)?,
                    nn::sparse_plus(&xs[0])?,
                    nn::sparse_sigmoid(&xs[0])?,
                    nn::soft_sign(&xs[0])?,
                    nn::hard_tanh(&xs[0])?,
                    nn::hard_sigmoid(&xs[0])?,
                    nn::hard_swish(&xs[0])?,
                    nn::glu(&xs[0], -1)?,
                    nn::logmeanexp(&xs[0], -1)?,
                    nn::softmax(&xs[0], -1)?,
                    nn::log_softmax(&xs[0], -1)?,
                    nn::standardize(&xs[0], -1, 1e-5)?,
                    nn::rms_norm(&xs[0], -1, 1e-5, None)?,
                    nn::l2_loss(&xs[0])?,
                ])
            },
            vec![x_f32],
        )
        .expect("jit_fn over F32 nn suite");
        for (idx, out) in f32_outs.iter().enumerate() {
            assert_eq!(
                out.dtype(),
                DType::F32,
                "nn output #{idx} should preserve F32 dtype, got {:?}",
                out.dtype()
            );
        }

        // Parity against eager fj_lax::nn on F64
        let raw = [-2.0_f64, -0.5, 0.5, 2.0];
        let x_f64 = Value::vector_f64(&raw).unwrap();
        let f64_outs = jit_fn(
            |xs| {
                Ok(vec![
                    nn::squareplus(&xs[0], 4.0)?,
                    nn::sparse_plus(&xs[0])?,
                    nn::sparse_sigmoid(&xs[0])?,
                    nn::glu(&xs[0], -1)?,
                    nn::logmeanexp(&xs[0], -1)?,
                ])
            },
            vec![x_f64],
        )
        .expect("jit_fn over F64 new nn ops");

        let assert_close = |actual: &[f64], expected: &[f64], label: &str| {
            assert_eq!(actual.len(), expected.len(), "{label} len mismatch");
            for (a, b) in actual.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12, "{label} mismatch: {a} vs {b}");
            }
        };
        assert_close(
            &f64_outs[0].as_tensor().unwrap().to_f64_vec().unwrap(),
            &nn::eager::squareplus(&raw, 4.0),
            "squareplus",
        );
        assert_close(
            &f64_outs[1].as_tensor().unwrap().to_f64_vec().unwrap(),
            &nn::eager::sparse_plus(&raw),
            "sparse_plus",
        );
        assert_close(
            &f64_outs[2].as_tensor().unwrap().to_f64_vec().unwrap(),
            &nn::eager::sparse_sigmoid(&raw),
            "sparse_sigmoid",
        );
        assert_close(
            &f64_outs[3].as_tensor().unwrap().to_f64_vec().unwrap(),
            &nn::eager::glu(&raw),
            "glu",
        );
        let lme = f64_outs[4].as_f64_scalar().unwrap();
        assert!((lme - nn::eager::logmeanexp(&raw)).abs() < 1e-12);

        // log1mexp parity on positive inputs
        let pos_raw = [0.2_f64, 0.7, 1.5, 4.0];
        let pos_val = Value::vector_f64(&pos_raw).unwrap();
        let l1me = jit_fn(|xs| Ok(vec![nn::log1mexp(&xs[0])?]), vec![pos_val]).expect("log1mexp")
            [0]
        .as_tensor()
        .unwrap()
        .to_f64_vec()
        .unwrap();
        assert_close(&l1me, &nn::eager::log1mexp(&pos_raw), "log1mexp");

        // softmax_cross_entropy + dot_product_attention + grad
        let logits = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let labels = Value::vector_f64(&[0.0, 1.0, 0.0]).unwrap();
        let ce_grads = grad_fn(
            |xs| Ok(vec![nn::softmax_cross_entropy(&xs[0], &xs[1], -1)?]),
            vec![logits, labels],
        )
        .expect("grad softmax_cross_entropy");
        let g_logits = ce_grads[0].as_tensor().unwrap().to_f64_vec().unwrap();
        let expected_sm = nn::eager::softmax(&[1.0, 2.0, 3.0]);
        assert_close(
            &g_logits,
            &[expected_sm[0], expected_sm[1] - 1.0, expected_sm[2]],
            "grad softmax_cross_entropy",
        );

        let q = Value::Tensor(
            fj_core::TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![1.0, 0.0, 0.0, 1.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let attn_out = jit_fn(
            |xs| {
                Ok(vec![nn::dot_product_attention(
                    &xs[0], &xs[0], &xs[0], None,
                )?])
            },
            vec![q],
        )
        .expect("dot_product_attention");
        assert_eq!(attn_out[0].as_tensor().unwrap().shape.dims, vec![2, 2]);
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
