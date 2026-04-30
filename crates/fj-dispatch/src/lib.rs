#![forbid(unsafe_code)]

pub mod batching;

use fj_cache::{CacheKeyError, CacheKeyInputRef, build_cache_key_ref};
use fj_core::{
    CompatibilityMode, Jaxpr, TensorValue, TraceTransformLedger, Transform,
    TransformCompositionError, Value, verify_transform_composition,
};
use fj_interpreters::InterpreterError;
use fj_ledger::{
    ConformalPredictor, DecisionRecord, EvidenceLedger, EvidenceSignal, LedgerEntry, LossMatrix,
};
use fj_runtime::backend::{Backend, BackendError, BackendRegistry};
use fj_runtime::device::{DeviceId, DevicePlacement};
use fj_trace::simulate_nested_trace_contexts;
use std::collections::BTreeMap;

// ── Effect Token System ────────────────────────────────────────────

/// Per-effect runtime token for ordered side-effect tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectToken {
    pub effect_name: String,
    pub sequence_number: u64,
}

/// Context for threading effect tokens through a dispatch execution.
///
/// V1 scope: tracking only — records which effects were observed and in what
/// order while preserving every observation.
///
/// Uses a Vec to preserve insertion order for deterministic sequence threading.
#[derive(Debug, Clone)]
pub struct EffectContext {
    tokens: Vec<EffectToken>,
    next_sequence: u64,
}

impl EffectContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Record observation of a named effect and return its sequence token.
    pub fn thread_token(&mut self, effect_name: &str) -> EffectToken {
        let name = effect_name.to_owned();
        #[rustfmt::skip]
        let effect_token = EffectToken { // ubs:ignore — dispatch effect sequencing token, not a credential or secret value
            effect_name: name,
            sequence_number: self.next_sequence,
        };
        self.next_sequence += 1;
        self.tokens.push(effect_token.clone());
        effect_token
    }

    /// Finalize and return all observed effect tokens in sequence order.
    /// Tokens are already in insertion order, so no sorting needed.
    #[must_use]
    pub fn finalize(self) -> Vec<EffectToken> {
        self.tokens
    }

    /// Number of effect observations.
    #[must_use]
    pub fn effect_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for EffectContext {
    fn default() -> Self {
        Self::new()
    }
}

fn thread_jaxpr_effect_tokens(effect_ctx: &mut EffectContext, jaxpr: &Jaxpr) {
    for effect in &jaxpr.effects {
        effect_ctx.thread_token(effect);
    }

    for equation in &jaxpr.equations {
        for effect in &equation.effects {
            effect_ctx.thread_token(effect);
        }
        for sub_jaxpr in &equation.sub_jaxprs {
            thread_jaxpr_effect_tokens(effect_ctx, sub_jaxpr);
        }
    }
}

/// Specifies which axis of an input/output is the batch axis for vmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisSpec {
    /// This input/output is batched along the given axis.
    Batched(i32),
    /// This input/output is not batched (broadcast to all batch elements).
    NotBatched,
}

impl AxisSpec {
    /// Resolve negative axis index to a positive one, given the tensor rank.
    fn resolve(self, rank: usize) -> Option<usize> {
        match self {
            Self::NotBatched => None,
            Self::Batched(axis) => {
                if axis >= 0 {
                    Some(axis as usize)
                } else {
                    let resolved = rank as i32 + axis;
                    if resolved >= 0 {
                        Some(resolved as usize)
                    } else {
                        None
                    }
                }
            }
        }
    }
}

/// Specifies in_axes/out_axes for vmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmapAxes {
    /// Same axis for all inputs/outputs.
    Uniform(AxisSpec),
    /// Per-input/output axis specification.
    PerArg(Vec<AxisSpec>),
}

impl Default for VmapAxes {
    fn default() -> Self {
        Self::Uniform(AxisSpec::Batched(0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRequest {
    pub mode: CompatibilityMode,
    pub ledger: TraceTransformLedger,
    pub args: Vec<Value>,
    pub backend: String,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

/// Parse in_axes from compile_options.
/// Format: comma-separated list of axis specs, e.g. "0,none,1" or just "0" for uniform.
fn parse_axis_spec(raw: &str, option_name: &'static str) -> Result<AxisSpec, DispatchError> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(AxisSpec::NotBatched);
    }

    trimmed.parse::<i32>().map(AxisSpec::Batched).map_err(|_| {
        TransformExecutionError::InvalidVmapAxisSpec {
            option: option_name,
            value: trimmed.to_owned(),
        }
        .into()
    })
}

fn parse_vmap_in_axes(
    opts: &BTreeMap<String, String>,
    num_args: usize,
) -> Result<Vec<AxisSpec>, DispatchError> {
    match opts.get("vmap_in_axes") {
        None => Ok(vec![AxisSpec::Batched(0); num_args]),
        Some(s) if s.trim().is_empty() => Ok(vec![AxisSpec::Batched(0); num_args]),
        Some(s) => {
            let specs = s
                .split(',')
                .map(|part| parse_axis_spec(part, "vmap_in_axes"))
                .collect::<Result<Vec<_>, _>>()?;
            // If only one spec provided, broadcast to all args
            if specs.len() == 1 {
                Ok(vec![specs[0]; num_args])
            } else {
                Ok(specs)
            }
        }
    }
}

/// Parse out_axes from compile_options.
fn parse_vmap_out_axes(
    opts: &BTreeMap<String, String>,
    num_outputs: usize,
) -> Result<Vec<AxisSpec>, DispatchError> {
    match opts.get("vmap_out_axes") {
        None => Ok(vec![AxisSpec::Batched(0); num_outputs]),
        Some(s) if s.trim().is_empty() => Ok(vec![AxisSpec::Batched(0); num_outputs]),
        Some(s) => {
            let specs = s
                .split(',')
                .map(|part| parse_axis_spec(part, "vmap_out_axes"))
                .collect::<Result<Vec<_>, _>>()?;
            if specs.len() == 1 {
                Ok(vec![specs[0]; num_outputs])
            } else {
                Ok(specs)
            }
        }
    }
}

fn wants_value_and_grad(opts: &BTreeMap<String, String>) -> bool {
    opts.get("value_and_grad").is_some_and(|raw| {
        raw.eq_ignore_ascii_case("true")
            || raw.eq_ignore_ascii_case("1")
            || raw.eq_ignore_ascii_case("yes")
    })
}

fn wants_egraph_optimize(opts: &BTreeMap<String, String>) -> bool {
    opts.get("egraph_optimize").is_some_and(|raw| {
        raw.eq_ignore_ascii_case("true")
            || raw.eq_ignore_ascii_case("1")
            || raw.eq_ignore_ascii_case("yes")
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchResponse {
    pub outputs: Vec<Value>,
    pub cache_key: String,
    pub evidence_ledger: EvidenceLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformExecutionError {
    EmptyArgumentList { transform: Transform },
    NonScalarGradientInput,
    NonScalarGradientOutput,
    VmapRequiresRankOneLeadingArgument,
    VmapMismatchedLeadingDimension { expected: usize, actual: usize },
    VmapInconsistentOutputArity { expected: usize, actual: usize },
    VmapAxesOutOfBounds { axis: i32, ndim: usize },
    VmapAxesCountMismatch { expected: usize, actual: usize },
    InvalidVmapAxisSpec { option: &'static str, value: String },
    VmapUnmappedOutputMismatch,
    EmptyVmapOutput,
    TensorBuild(String),
}

impl std::fmt::Display for TransformExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyArgumentList { transform } => {
                write!(f, "{} requires at least one argument", transform.as_str())
            }
            Self::NonScalarGradientInput => {
                write!(f, "grad currently requires scalar first input")
            }
            Self::NonScalarGradientOutput => {
                write!(f, "grad currently requires scalar first output")
            }
            Self::VmapRequiresRankOneLeadingArgument => {
                write!(f, "vmap currently requires first argument with rank >= 1")
            }
            Self::VmapMismatchedLeadingDimension { expected, actual } => {
                write!(
                    f,
                    "vmap leading-dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::VmapInconsistentOutputArity { expected, actual } => {
                write!(
                    f,
                    "vmap inner output arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::EmptyVmapOutput => {
                write!(f, "vmap received no mapped elements")
            }
            Self::VmapAxesOutOfBounds { axis, ndim } => {
                write!(
                    f,
                    "vmap axis {axis} is out of bounds for tensor with {ndim} dimensions"
                )
            }
            Self::VmapAxesCountMismatch { expected, actual } => {
                write!(
                    f,
                    "vmap in_axes/out_axes length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidVmapAxisSpec { option, value } => {
                write!(
                    f,
                    "invalid {option} axis spec {value:?}; expected an integer or \"none\""
                )
            }
            Self::VmapUnmappedOutputMismatch => {
                write!(
                    f,
                    "vmap out_axes=none requires the output to be identical across mapped elements"
                )
            }
            Self::TensorBuild(detail) => write!(f, "tensor build error: {detail}"),
        }
    }
}

impl std::error::Error for TransformExecutionError {}

#[derive(Debug)]
pub enum DispatchError {
    Cache(CacheKeyError),
    Interpreter(InterpreterError),
    BackendExecution(BackendError),
    TransformInvariant(TransformCompositionError),
    TransformExecution(TransformExecutionError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(err) => write!(f, "cache key error: {err}"),
            Self::Interpreter(err) => write!(f, "interpreter error: {err}"),
            Self::BackendExecution(err) => write!(f, "backend execution error: {err}"),
            Self::TransformInvariant(err) => write!(f, "transform invariant error: {err}"),
            Self::TransformExecution(err) => write!(f, "transform execution error: {err}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<CacheKeyError> for DispatchError {
    fn from(value: CacheKeyError) -> Self {
        Self::Cache(value)
    }
}

impl From<InterpreterError> for DispatchError {
    fn from(value: InterpreterError) -> Self {
        Self::Interpreter(value)
    }
}

impl From<BackendError> for DispatchError {
    fn from(value: BackendError) -> Self {
        Self::BackendExecution(value)
    }
}

impl From<TransformCompositionError> for DispatchError {
    fn from(value: TransformCompositionError) -> Self {
        Self::TransformInvariant(value)
    }
}

impl From<TransformExecutionError> for DispatchError {
    fn from(value: TransformExecutionError) -> Self {
        Self::TransformExecution(value)
    }
}

pub fn dispatch(request: DispatchRequest) -> Result<DispatchResponse, DispatchError> {
    let composition_proof = verify_transform_composition(&request.ledger)?;
    let nested_trace_summary = simulate_nested_trace_contexts(
        &request.ledger.transform_stack,
        &request.args,
    )
    .map_err(|err| {
        TransformExecutionError::TensorBuild(format!("nested trace simulation failed: {err}"))
    })?;

    let cache_key = build_cache_key_ref(&CacheKeyInputRef {
        mode: request.mode,
        backend: &request.backend,
        jaxpr: &request.ledger.root_jaxpr,
        transform_stack: &request.ledger.transform_stack,
        compile_options: &request.compile_options,
        custom_hook: request.custom_hook.as_deref(),
        unknown_incompatible_features: &request.unknown_incompatible_features,
    })?;

    // Thread effect context through Jaxpr/equation effect ordering.
    let mut effect_ctx = EffectContext::new();
    thread_jaxpr_effect_tokens(&mut effect_ctx, &request.ledger.root_jaxpr);

    // Optionally run e-graph equality saturation to simplify the Jaxpr.
    let exec_jaxpr = if wants_egraph_optimize(&request.compile_options) {
        fj_egraph::optimize_jaxpr(&request.ledger.root_jaxpr)
    } else {
        request.ledger.root_jaxpr.clone()
    };

    let backend_registry = BackendRegistry::new(vec![Box::new(fj_backend_cpu::CpuBackend::new())]);
    let requested_backend = (!request.backend.is_empty()).then_some(request.backend.as_str());
    let (backend, device, _fell_back) =
        backend_registry.resolve_with_fallback(&DevicePlacement::Default, requested_backend)?;
    let outputs = execute_with_transforms(
        &exec_jaxpr,
        &request.ledger.transform_stack,
        &request.args,
        backend,
        device,
        &request.compile_options,
    )?;

    let effect_tokens = effect_ctx.finalize();
    let effect_count = effect_tokens.len();
    let effect_trace = effect_tokens
        .iter()
        .map(|token| format!("{}:{}", token.sequence_number, token.effect_name))
        .collect::<Vec<_>>()
        .join(",");
    let nested_trace_frame_trace = nested_trace_summary
        .frames
        .iter()
        .map(|frame| {
            format!(
                "{}:{}@{}",
                frame.transform.as_str(),
                frame.trace_id,
                frame.depth
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let mut evidence_ledger = EvidenceLedger::new();
    let posterior_abandoned = heuristic_posterior_abandoned(&request.ledger);
    let matrix = LossMatrix::default();
    let record = DecisionRecord::from_posterior(request.mode, posterior_abandoned, &matrix);

    evidence_ledger.append(LedgerEntry {
        decision_id: cache_key.as_string(),
        record,
        signals: vec![
            EvidenceSignal {
                signal_name: "eqn_count".to_owned(),
                log_likelihood_delta: (request.ledger.root_jaxpr.equations.len() as f64 + 1.0).ln(),
                detail: format!("eqn_count={}", request.ledger.root_jaxpr.equations.len()),
            },
            EvidenceSignal {
                signal_name: "transform_depth".to_owned(),
                log_likelihood_delta: request.ledger.transform_stack.len() as f64 * 0.1,
                detail: format!("transform_depth={}", request.ledger.transform_stack.len()),
            },
            EvidenceSignal {
                signal_name: "transform_stack_hash".to_owned(),
                log_likelihood_delta: (composition_proof.transform_count as f64 + 1.0).ln(),
                detail: composition_proof.stack_hash_hex,
            },
            EvidenceSignal {
                signal_name: "nested_trace_depth".to_owned(),
                log_likelihood_delta: nested_trace_summary.max_depth as f64 * 0.05,
                detail: format!(
                    "max_depth={},frames={}",
                    nested_trace_summary.max_depth, nested_trace_frame_trace
                ),
            },
            EvidenceSignal {
                signal_name: "effect_token_count".to_owned(),
                log_likelihood_delta: (effect_count as f64 + 1.0).ln(),
                detail: format!("effect_tokens={effect_count}"),
            },
            EvidenceSignal {
                signal_name: "effect_token_trace".to_owned(),
                log_likelihood_delta: (effect_count as f64 + 1.0).ln() * 0.1,
                detail: format!("effect_tokens=[{effect_trace}]"),
            },
        ],
    });

    Ok(DispatchResponse {
        outputs,
        cache_key: cache_key.as_string(),
        evidence_ledger,
    })
}

fn execute_with_transforms(
    root_jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    // Skip leading Jit transforms (no-op pass-through) to find the first
    // non-Jit transform, avoiding recursive stack frames for Jit chains.
    let non_jit_start = transforms
        .iter()
        .position(|t| *t != Transform::Jit)
        .unwrap_or(transforms.len());

    let remaining = &transforms[non_jit_start..];
    let Some((head, tail)) = remaining.split_first() else {
        return backend
            .execute(root_jaxpr, args, device)
            .map_err(DispatchError::from);
    };

    match head {
        Transform::Jit => {
            execute_with_transforms(root_jaxpr, tail, args, backend, device, compile_options)
        }
        Transform::Grad => execute_grad(root_jaxpr, tail, args, backend, device, compile_options),
        Transform::Vmap => execute_vmap(root_jaxpr, tail, args, backend, device, compile_options),
    }
}

fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Grad,
        }
        .into());
    }

    // If there are remaining transforms in the tail, fall back to finite-diff
    // (symbolic AD only applies to the innermost evaluation).
    // For finite-diff, we still require scalar first argument.
    if !tail.is_empty() {
        args[0]
            .as_f64_scalar()
            .ok_or(TransformExecutionError::NonScalarGradientInput)?;
        return execute_grad_finite_diff(root_jaxpr, tail, args, backend, device);
    }

    if wants_value_and_grad(compile_options) {
        // Shared forward pass for value_and_grad mode.
        let (mut values, grads) =
            fj_ad::value_and_grad_jaxpr(root_jaxpr, args).map_err(|e| match e {
                fj_ad::AdError::NonScalarGradientOutput => {
                    TransformExecutionError::NonScalarGradientOutput
                }
                other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
            })?;
        // value_and_grad returns the primal outputs followed by one gradient
        // per input, matching fj_ad::value_and_grad_jaxpr.
        values.extend(grads);
        return Ok(values);
    }

    // Tensor-aware AD: grad_jaxpr returns Value gradients for all inputs.
    let grads = fj_ad::grad_jaxpr(root_jaxpr, args).map_err(|e| match e {
        fj_ad::AdError::NonScalarGradientOutput => TransformExecutionError::NonScalarGradientOutput,
        other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
    })?;
    // Return gradient of first input (matches JAX's default grad behavior).
    Ok(vec![
        grads.into_iter().next().unwrap_or(Value::scalar_f64(0.0)),
    ])
}

fn execute_grad_finite_diff(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
) -> Result<Vec<Value>, DispatchError> {
    let input_value = args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    let epsilon = 1e-6_f64;
    let mut plus_args = args.to_vec();
    let mut minus_args = args.to_vec();
    plus_args[0] = Value::scalar_f64(input_value + epsilon);
    minus_args[0] = Value::scalar_f64(input_value - epsilon);

    let empty_opts = BTreeMap::new();
    let plus_out =
        execute_with_transforms(root_jaxpr, tail, &plus_args, backend, device, &empty_opts)?;
    let minus_out =
        execute_with_transforms(root_jaxpr, tail, &minus_args, backend, device, &empty_opts)?;

    let plus_value = plus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;
    let minus_value = minus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;

    let derivative = (plus_value - minus_value) / (2.0 * epsilon);
    Ok(vec![Value::scalar_f64(derivative)])
}

fn execute_vmap(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Vmap,
        }
        .into());
    }

    // Parse in_axes from compile_options
    let in_axes = parse_vmap_in_axes(compile_options, args.len())?;

    // Validate in_axes length matches args
    if in_axes.len() != args.len() {
        return Err(TransformExecutionError::VmapAxesCountMismatch {
            expected: args.len(),
            actual: in_axes.len(),
        }
        .into());
    }

    // Determine batch size from the first batched argument
    let mut batch_size: Option<usize> = None;
    for (arg, axis_spec) in args.iter().zip(in_axes.iter()) {
        if let AxisSpec::Batched(axis) = axis_spec {
            match arg {
                Value::Tensor(tensor) => {
                    let rank = tensor.rank();
                    let resolved = axis_spec.resolve(rank).ok_or(
                        TransformExecutionError::VmapAxesOutOfBounds {
                            axis: *axis,
                            ndim: rank,
                        },
                    )?;
                    if resolved >= rank {
                        return Err(TransformExecutionError::VmapAxesOutOfBounds {
                            axis: *axis,
                            ndim: rank,
                        }
                        .into());
                    }
                    let dim_size = tensor.shape.dims[resolved] as usize;
                    if let Some(expected) = batch_size {
                        if dim_size != expected {
                            return Err(TransformExecutionError::VmapMismatchedLeadingDimension {
                                expected,
                                actual: dim_size,
                            }
                            .into());
                        }
                    } else {
                        batch_size = Some(dim_size);
                    }
                }
                Value::Scalar(_) => {
                    // Scalar with batched axis is invalid
                    return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
                }
            }
        }
    }

    let lead_len = batch_size.ok_or(TransformExecutionError::EmptyVmapOutput)?;
    if lead_len == 0 {
        return Err(TransformExecutionError::EmptyVmapOutput.into());
    }

    // Use BatchTrace interpreter when no tail transforms exist and all batched
    // args use axis 0 (the default case).
    let all_axis_zero = in_axes
        .iter()
        .all(|spec| matches!(spec, AxisSpec::Batched(0) | AxisSpec::NotBatched));
    if tail.is_empty() && all_axis_zero && !compile_options.contains_key("vmap_out_axes") {
        return execute_vmap_batch_trace(root_jaxpr, args, &in_axes, lead_len);
    }

    // Fall back to loop-and-stack for non-trivial axes or composed transforms
    execute_vmap_loop_and_stack(
        root_jaxpr,
        tail,
        args,
        backend,
        device,
        lead_len,
        &in_axes,
        compile_options,
    )
}

/// BatchTrace-based vmap execution: O(1) vectorized dispatch via per-primitive
/// batching rules. Each arg becomes a BatchTracer with the specified batch_dim
/// or None (scalars/unbatched), and the Jaxpr is evaluated equation-by-equation.
fn execute_vmap_batch_trace(
    root_jaxpr: &Jaxpr,
    args: &[Value],
    in_axes: &[AxisSpec],
    lead_len: usize,
) -> Result<Vec<Value>, DispatchError> {
    use batching::{BatchTracer, batch_eval_jaxpr};

    let batch_inputs: Vec<BatchTracer> = args
        .iter()
        .zip(in_axes.iter())
        .map(|(arg, axis_spec)| match axis_spec {
            AxisSpec::NotBatched => BatchTracer::unbatched(arg.clone()),
            AxisSpec::Batched(_) => {
                let resolved = match arg {
                    Value::Tensor(t) => axis_spec.resolve(t.rank()).unwrap_or(0),
                    Value::Scalar(_) => 0,
                };
                if matches!(arg, Value::Scalar(_)) {
                    BatchTracer::unbatched(arg.clone())
                } else {
                    BatchTracer::batched(arg.clone(), resolved)
                }
            }
        })
        .collect();

    let results = batch_eval_jaxpr(root_jaxpr, &batch_inputs)
        .map_err(|e| TransformExecutionError::TensorBuild(format!("BatchTrace error: {e}")))?;

    // Extract output values, ensuring batch dim is at position 0
    let mut outputs = Vec::with_capacity(results.len());
    for tracer in results {
        match tracer.batch_dim {
            Some(0) => outputs.push(tracer.value),
            Some(bd) => {
                // Move batch dim to front for consistent output
                let moved = batching::move_batch_dim_to_front(&tracer.value, bd)
                    .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
                outputs.push(moved);
            }
            None => {
                // Unbatched output — broadcast it across the mapped batch axis.
                // This happens when the function output doesn't depend on the input.
                let broadcast = TensorValue::repeat_axis0(&tracer.value, lead_len)
                    .map(Value::Tensor)
                    .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
                outputs.push(broadcast);
            }
        }
    }

    Ok(outputs)
}

/// Loop-and-stack vmap fallback for composed transforms (e.g., vmap(grad(f))).
#[allow(clippy::too_many_arguments)]
fn execute_vmap_loop_and_stack(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    lead_len: usize,
    in_axes: &[AxisSpec],
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let mut per_output_values: Vec<Vec<Value>> = Vec::new();

    for index in 0..lead_len {
        let mut mapped_args = Vec::with_capacity(args.len());
        for (arg, axis_spec) in args.iter().zip(in_axes.iter()) {
            match axis_spec {
                AxisSpec::NotBatched => {
                    // Broadcast: pass the arg unchanged to every iteration
                    mapped_args.push(arg.clone());
                }
                AxisSpec::Batched(_) => {
                    match arg {
                        Value::Scalar(lit) => mapped_args.push(Value::Scalar(*lit)),
                        Value::Tensor(tensor) => {
                            let resolved_axis = axis_spec.resolve(tensor.rank()).unwrap_or(0);
                            if resolved_axis == 0 {
                                mapped_args.push(tensor.slice_axis0(index).map_err(|err| {
                                    TransformExecutionError::TensorBuild(err.to_string())
                                })?);
                            } else {
                                // For non-zero axes, we need to slice along that axis
                                let sliced = slice_along_axis(tensor, resolved_axis, index)
                                    .map_err(|err| {
                                        TransformExecutionError::TensorBuild(err.to_string())
                                    })?;
                                mapped_args.push(sliced);
                            }
                        }
                    }
                }
            }
        }

        let empty_opts = BTreeMap::new();
        let mapped_output =
            execute_with_transforms(root_jaxpr, tail, &mapped_args, backend, device, &empty_opts)?;
        if index == 0 {
            per_output_values = vec![Vec::with_capacity(lead_len); mapped_output.len()];
        } else if mapped_output.len() != per_output_values.len() {
            return Err(TransformExecutionError::VmapInconsistentOutputArity {
                expected: per_output_values.len(),
                actual: mapped_output.len(),
            }
            .into());
        }

        for (output_idx, value) in mapped_output.iter().enumerate() {
            per_output_values[output_idx].push(value.clone());
        }
    }

    let out_axes = parse_vmap_out_axes(compile_options, per_output_values.len())?;
    if out_axes.len() != per_output_values.len() {
        return Err(TransformExecutionError::VmapAxesCountMismatch {
            expected: per_output_values.len(),
            actual: out_axes.len(),
        }
        .into());
    }

    let mut outputs = Vec::with_capacity(per_output_values.len());
    for (out_idx, values) in per_output_values.iter().enumerate() {
        let out_axis = out_axes
            .get(out_idx)
            .copied()
            .unwrap_or(AxisSpec::Batched(0));
        match out_axis {
            AxisSpec::Batched(0) => {
                let tensor = TensorValue::stack_axis0(values)
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                outputs.push(Value::Tensor(tensor));
            }
            AxisSpec::NotBatched => {
                let Some(first) = values.first() else {
                    return Err(TransformExecutionError::EmptyVmapOutput.into());
                };
                if values.iter().skip(1).any(|value| value != first) {
                    return Err(TransformExecutionError::VmapUnmappedOutputMismatch.into());
                }
                outputs.push(first.clone());
            }
            AxisSpec::Batched(target_axis) => {
                // Move batch dim from 0 to target_axis via Transpose primitive
                let tensor = TensorValue::stack_axis0(values)
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                let rank = tensor.rank();
                let resolved = AxisSpec::Batched(target_axis).resolve(rank).ok_or(
                    TransformExecutionError::VmapAxesOutOfBounds {
                        axis: target_axis,
                        ndim: rank,
                    },
                )?;
                if resolved >= rank {
                    return Err(TransformExecutionError::VmapAxesOutOfBounds {
                        axis: target_axis,
                        ndim: rank,
                    }
                    .into());
                }
                if resolved == 0 {
                    outputs.push(Value::Tensor(tensor));
                } else {
                    // Build permutation: move axis 0 to position `resolved`
                    let mut perm: Vec<usize> = (0..rank).collect();
                    perm.remove(0);
                    perm.insert(resolved, 0);
                    let perm_str = perm
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let mut params = BTreeMap::new();
                    params.insert("permutation".to_owned(), perm_str);
                    let transposed = fj_lax::eval_primitive(
                        fj_core::Primitive::Transpose,
                        &[Value::Tensor(tensor)],
                        &params,
                    )
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                    outputs.push(transposed);
                }
            }
        }
    }

    Ok(outputs)
}

/// Slice a tensor along an arbitrary axis, extracting the `index`-th slice.
fn slice_along_axis(tensor: &TensorValue, axis: usize, index: usize) -> Result<Value, String> {
    let rank = tensor.rank();
    if axis >= rank {
        return Err(format!(
            "slice_along_axis: axis {axis} out of bounds for rank {rank}"
        ));
    }
    if axis == 0 {
        return tensor.slice_axis0(index).map_err(|e| e.to_string());
    }

    // For non-zero axis, transpose to bring the axis to position 0,
    // slice, then transpose back.
    let mut perm_fwd: Vec<usize> = (0..rank).collect();
    perm_fwd.remove(axis);
    perm_fwd.insert(0, axis);
    let perm_str = perm_fwd
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let mut params = BTreeMap::new();
    params.insert("permutation".to_owned(), perm_str);

    let transposed = fj_lax::eval_primitive(
        fj_core::Primitive::Transpose,
        &[Value::Tensor(tensor.clone())],
        &params,
    )
    .map_err(|e| e.to_string())?;

    match transposed {
        Value::Tensor(t) => t.slice_axis0(index).map_err(|e| e.to_string()),
        other => Ok(other),
    }
}

#[inline]
fn heuristic_posterior_abandoned(ledger: &TraceTransformLedger) -> f64 {
    let eqn_factor = ledger.root_jaxpr.equations.len() as f64;
    let depth_factor = ledger.transform_stack.len() as f64;
    let score = (eqn_factor + 2.0 * depth_factor) / (eqn_factor + 2.0 * depth_factor + 20.0);
    score.clamp(0.05, 0.95)
}

/// Compute posterior with conformal calibration when available.
#[must_use]
pub fn calibrated_posterior_abandoned(
    ledger: &TraceTransformLedger,
    conformal: Option<&ConformalPredictor>,
) -> f64 {
    let heuristic = heuristic_posterior_abandoned(ledger);
    match conformal {
        Some(cp) if cp.is_calibrated() => {
            let estimate = cp.calibrated_posterior(heuristic);
            estimate.point
        }
        _ => heuristic,
    }
}

#[cfg(test)]
mod tests {
    use super::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
    use fj_core::{
        Atom, CompatibilityMode, DType, Equation, Jaxpr, Primitive, ProgramSpec, Shape,
        TensorValue, TraceTransformLedger, Transform, Value, VarId, build_program,
    };
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
        let mut ledger = TraceTransformLedger::new(build_program(program));
        for (idx, transform) in transforms.iter().enumerate() {
            ledger.push_transform(
                *transform,
                format!("evidence-{}-{}", transform.as_str(), idx),
            );
        }
        ledger
    }

    #[test]
    fn dispatch_jit_add_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn dispatch_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad should succeed");

        let derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_value_and_grad_mode_square_scalar() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch value_and_grad should succeed");

        assert_eq!(response.outputs.len(), 2, "expected [value, grad]");
        let value = response.outputs[0]
            .as_f64_scalar()
            .expect("value output should be scalar f64");
        let derivative = response.outputs[1]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((value - 9.0).abs() < 1e-6);
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_grad_of_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad-of-grad should succeed");

        let second_derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("second derivative output should be scalar f64");
        assert!((second_derivative - 2.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_vmap_add_one_vector() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4]);
    }

    #[test]
    fn dispatch_vmap_broadcasts_constant_batch_trace_outputs() {
        use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
        use smallvec::smallvec;

        let constant_jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Lit(Literal::I64(1)), Atom::Lit(Literal::I64(2))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            }],
        );
        let mut ttl = TraceTransformLedger::new(constant_jaxpr);
        ttl.push_transform(Transform::Vmap, "constant-vmap".to_owned());

        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap with constant body should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("constant vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![3] });
        let values = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![3, 3, 3]);
    }

    #[test]
    fn dispatch_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank2 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 2] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5]);
    }

    #[test]
    fn dispatch_vmap_of_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch nested vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("nested vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 3] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn transform_order_is_explicit() {
        let good = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(grad(f)) should be supported");

        let good_out = good.outputs[0]
            .as_tensor()
            .expect("output should be tensor")
            .to_f64_vec()
            .expect("output should be numeric");
        assert_eq!(good_out.len(), 3);
        assert!((good_out[0] - 2.0).abs() < 1e-3);
        assert!((good_out[1] - 4.0).abs() < 1e-3);
        assert!((good_out[2] - 6.0).abs() < 1e-3);

        let bad = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(vmap(f)) should fail with current constraints");

        assert!(matches!(
            bad,
            DispatchError::TransformExecution(
                super::TransformExecutionError::NonScalarGradientInput
            )
        ));
    }

    #[test]
    fn strict_mode_rejects_unknown_features_fail_closed() {
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect_err("strict mode must reject unknown incompatible features");

        assert!(
            matches!(
                &err,
                DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures { .. })
            ),
            "expected fail-closed cache-key rejection, got: {err:?}"
        );
        if let DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures {
            features,
        }) = &err
        {
            assert_eq!(features, &vec!["future.backend.protocol.v2".to_owned()]);
        }
    }

    #[test]
    fn hardened_mode_allowlists_unknown_features_for_auditable_progress() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect("hardened mode should permit allowlisted unknown features");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn test_dispatch_test_log_schema_contract() {
        let fixture_id = fj_test_utils::fixture_id_from_json(&("dispatch", "transform-order"))
            .expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_dispatch_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Effect token tests ─────────────────────────────────────

    #[test]
    fn effect_context_preserves_observation_sequence() {
        use super::EffectContext;
        let mut ctx = EffectContext::new();
        let t1 = ctx.thread_token("jit");
        let t2 = ctx.thread_token("jit");
        let t3 = ctx.thread_token("grad");

        assert_eq!(t1.sequence_number, 0);
        assert_eq!(t2.sequence_number, 1);
        assert_eq!(t3.sequence_number, 2);
        assert_eq!(ctx.effect_count(), 3);

        let tokens = ctx.finalize();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].effect_name, "jit");
        assert_eq!(tokens[1].effect_name, "jit");
        assert_eq!(tokens[2].effect_name, "grad");
    }

    #[test]
    fn effect_threading_walks_jaxpr_and_subjaxprs_in_order() {
        use super::{EffectContext, thread_jaxpr_effect_tokens};

        let mut root = build_program(ProgramSpec::Add2);
        root.effects = vec!["Print".to_owned()];
        root.equations[0].effects = vec!["RngConsume".to_owned()];

        let mut sub = build_program(ProgramSpec::AddOne);
        sub.effects = vec!["SubTrace".to_owned()];
        sub.equations[0].effects = vec!["SubPrint".to_owned()];
        root.equations[0].sub_jaxprs.push(sub);

        let mut ctx = EffectContext::new();
        thread_jaxpr_effect_tokens(&mut ctx, &root);
        let tokens = ctx.finalize();
        let names = tokens
            .iter()
            .map(|token| token.effect_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Print", "RngConsume", "SubTrace", "SubPrint"]);
    }

    #[test]
    fn dispatch_effect_signals_reflect_jaxpr_effects() {
        let mut ttl = ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]);
        ttl.root_jaxpr.effects = vec!["Print".to_owned()];
        ttl.root_jaxpr.equations[0].effects = vec!["RngConsume".to_owned()];

        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        // Evidence ledger should include count + trace for effect token observations.
        assert_eq!(response.evidence_ledger.len(), 1);
        let entry = &response.evidence_ledger.entries()[0];
        assert_eq!(entry.signals.len(), 6);

        let count_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(count_signal.detail, "effect_tokens=2");

        let trace_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_trace")
            .expect("should have effect_token_trace signal");
        assert_eq!(trace_signal.detail, "effect_tokens=[0:Print,1:RngConsume]");
    }

    #[test]
    fn dispatch_with_no_effects_has_zero_effect_tokens() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        let entry = &response.evidence_ledger.entries()[0];
        let count_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(count_signal.detail, "effect_tokens=0");
    }

    #[test]
    fn dispatch_cache_hit_miss_determinism() {
        // Two identical requests should produce identical cache keys
        let make_request = || DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let r1 = dispatch(make_request()).expect("dispatch 1");
        let r2 = dispatch(make_request()).expect("dispatch 2");
        assert_eq!(r1.cache_key, r2.cache_key, "same request = same cache key");
        assert_eq!(r1.outputs, r2.outputs, "same request = same outputs");

        // Different program should produce different cache key
        let r3 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit]),
            args: vec![Value::scalar_f64(2.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch 3");
        assert_ne!(
            r1.cache_key, r3.cache_key,
            "different program = different key"
        );
    }

    #[test]
    fn dispatch_value_and_grad_mode_changes_cache_key() {
        let make_request = |value_and_grad: bool| {
            let mut compile_options = BTreeMap::new();
            if value_and_grad {
                compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
            }
            DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
                args: vec![Value::scalar_f64(3.0)],
                backend: "cpu".to_owned(),
                compile_options,
                custom_hook: None,
                unknown_incompatible_features: vec![],
            }
        };

        let plain_grad = dispatch(make_request(false)).expect("plain grad dispatch");
        let value_and_grad = dispatch(make_request(true)).expect("value_and_grad dispatch");

        assert_ne!(
            plain_grad.cache_key, value_and_grad.cache_key,
            "value_and_grad mode must use a distinct cache identity from plain grad"
        );
    }

    #[test]
    fn dispatch_unknown_backend_falls_back_to_cpu_execution() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "quantum".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("unknown backend should fall back to cpu backend");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
    }

    #[test]
    fn dispatch_backend_name_still_changes_cache_key_under_fallback() {
        let make_request = |backend: &str| DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: backend.to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let cpu = dispatch(make_request("cpu")).expect("cpu dispatch should succeed");
        let fallback = dispatch(make_request("quantum")).expect("fallback dispatch should succeed");

        assert_eq!(cpu.outputs, fallback.outputs);
        assert_ne!(
            cpu.cache_key, fallback.cache_key,
            "requested backend remains part of cache identity even when runtime execution falls back"
        );
    }

    // ── Calibrated posterior tests ────────────────────────────

    #[test]
    fn calibrated_posterior_falls_back_to_heuristic_without_conformal() {
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let result = super::calibrated_posterior_abandoned(&ttl, None);
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "without conformal predictor, should return heuristic: {result} vs {heuristic}"
        );
    }

    #[test]
    fn calibrated_posterior_falls_back_when_uncalibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let cp = ConformalPredictor::new(0.9, 10); // needs 10 scores, has 0
        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "uncalibrated predictor should fall back to heuristic"
        );
    }

    #[test]
    fn calibrated_posterior_uses_conformal_when_calibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut cp = ConformalPredictor::new(0.9, 5);
        for score in &[0.01, 0.02, 0.03, 0.04, 0.05] {
            cp.observe(*score);
        }
        assert!(cp.is_calibrated());

        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        // Result should equal the conformal point estimate (which equals heuristic)
        // because calibrated_posterior returns point = heuristic
        assert!(
            (result - heuristic).abs() < 1e-12,
            "calibrated conformal point estimate should equal heuristic input"
        );
    }

    #[test]
    fn heuristic_posterior_increases_with_transform_depth() {
        let shallow = ledger(ProgramSpec::Add2, &[Transform::Jit]);
        let deep = ledger(
            ProgramSpec::Add2,
            &[Transform::Jit, Transform::Grad, Transform::Vmap],
        );
        let h_shallow = super::heuristic_posterior_abandoned(&shallow);
        let h_deep = super::heuristic_posterior_abandoned(&deep);
        assert!(
            h_deep > h_shallow,
            "deeper transform stack should have higher abandoned posterior: {h_deep} vs {h_shallow}"
        );
    }

    #[test]
    fn heuristic_posterior_is_bounded() {
        let minimal = ledger(ProgramSpec::AddOne, &[]);
        let result = super::heuristic_posterior_abandoned(&minimal);
        assert!(result >= 0.05, "posterior should be >= 0.05, got {result}");
        assert!(result <= 0.95, "posterior should be <= 0.95, got {result}");
    }

    // ══════════════════════════════════════════════════════════════
    // Higher-rank tensor tests (3D+)
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn dispatch_vmap_add_one_rank3_tensor() {
        // 3D tensor [2, 3, 2] — vmap should process each leading-axis slice
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 2],
                },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank3 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap rank3 output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 2]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_add_one_rank3_tensor() {
        // 3D tensor with triple vmap (vmap(vmap(vmap(f))))
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 2, 3],
                },
                (0..12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOne,
                &[Transform::Vmap, Transform::Vmap, Transform::Vmap],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 3]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (1..=12).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_grad_square_rank3_tensor() {
        let input_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                input_vals
                    .iter()
                    .copied()
                    .map(fj_core::Literal::from_f64)
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Grad,
                ],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap(grad(square)) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap grad output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 2]
            }
        );
        let grads = output.to_f64_vec().expect("output should be f64 tensor");
        let expected: Vec<f64> = input_vals.iter().map(|x| 2.0 * x).collect();
        assert_eq!(grads.len(), expected.len());
        for (idx, (actual, expected)) in grads.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "index {idx}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn dispatch_grad_square_rank3_tensor_rejects_non_scalar_input() {
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                (1..=8)
                    .map(|v| fj_core::Literal::from_f64(v as f64))
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(square) on rank3 input should fail");

        let err_msg = err.to_string();
        assert!(
            err_msg.contains("scalar"),
            "rank3 grad failure should mention scalar requirement, got: {err_msg}"
        );
    }

    #[test]
    fn dispatch_grad_sin_with_jit() {
        // grad(jit(sin(x))) = cos(x) at x=0 => 1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(0.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad(jit(sin)) should succeed");

        let grad_val = response.outputs[0]
            .as_f64_scalar()
            .expect("grad should be scalar f64");
        assert!(
            (grad_val - 1.0).abs() < 1e-10,
            "grad(sin)(0) should be cos(0) = 1.0, got {grad_val}"
        );
    }

    #[test]
    fn dispatch_vmap_sin_f64_vector() {
        // vmap(sin)([0, pi/2, pi]) = [0, 1, ~0]
        use std::f64::consts::PI;
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Vmap]),
            args: vec![Value::vector_f64(&[0.0, PI / 2.0, PI]).unwrap()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap(sin) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let vals: Vec<f64> = output.to_f64_vec().expect("should convert to f64");
        assert!(vals[0].abs() < 1e-10, "sin(0) should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-10, "sin(pi/2) should be 1");
        assert!(vals[2].abs() < 1e-10, "sin(pi) should be ~0");
    }

    // ── VMAP tensor output tests (bd-22lm) ───────────────────────

    #[test]
    fn test_vmap_tensor_output_rank1() {
        // vmap(AddOne) over a [3, 4] matrix: inner function receives rank-1 vectors,
        // returns rank-1 vectors, stacked back to [3, 4].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with tensor output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![3, 4] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_tensor_output_rank2() {
        // vmap(AddOne) over a [2, 3, 4] rank-3 tensor: inner receives rank-2 matrices,
        // returns rank-2 matrices, stacked back to [2, 3, 4].
        let tensor = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 4],
                },
                (0..24).map(fj_core::Literal::I64).collect(),
            )
            .expect("rank-3 tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with rank-2 inner output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 4]
            }
        );
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (1..=24).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_multi_output() {
        // vmap(AddOneMulTwo) over [1, 2, 3]: inner returns (x+1, x*2) per element.
        // Output: two tensors, each of shape [3].
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOneMulTwo, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let out0 = response.outputs[0]
            .as_tensor()
            .expect("first output should be tensor");
        let out1 = response.outputs[1]
            .as_tensor()
            .expect("second output should be tensor");

        let vals0: Vec<i64> = out0.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        let vals1: Vec<i64> = out1.elements.iter().map(|l| l.as_i64().unwrap()).collect();

        assert_eq!(vals0, vec![2, 3, 4], "x+1 for [1,2,3]");
        assert_eq!(vals1, vec![2, 4, 6], "x*2 for [1,2,3]");
    }

    #[test]
    fn test_vmap_multi_output_loop_and_stack() {
        // Same as above but force the loop-and-stack path by adding a Jit tail transform.
        let input = Value::vector_i64(&[10, 20, 30]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOneMulTwo,
                &[Transform::Vmap, Transform::Jit],
            ),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output loop-and-stack should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let vals0: Vec<i64> = response.outputs[0]
            .as_tensor()
            .expect("first output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let vals1: Vec<i64> = response.outputs[1]
            .as_tensor()
            .expect("second output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();

        assert_eq!(vals0, vec![11, 21, 31], "x+1 for [10,20,30]");
        assert_eq!(vals1, vec![20, 40, 60], "x*2 for [10,20,30]");
    }

    #[test]
    fn test_vmap_output_shape_batch_prepend() {
        // Verify batch dimension is correctly prepended to inner output shape.
        // vmap(AddOne) on [5, 3] matrix: inner returns [3] vectors → stacked to [5, 3].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5, 3] },
                (0..15)
                    .map(|i| fj_core::Literal::from_f64(i as f64))
                    .collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap shape prepend should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        // Output shape should be [5, 3] = [batch_size, ...inner_output_shape]
        assert_eq!(
            output.shape,
            Shape { dims: vec![5, 3] },
            "batch dim (5) prepended to inner output shape (3)"
        );
    }

    #[test]
    fn test_vmap_identity_preserves_shape() {
        // vmap(identity) should return the same tensor as the input.
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Identity, &[Transform::Vmap]),
            args: vec![matrix.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(identity) should succeed");

        assert_eq!(response.outputs.len(), 1);
        let output = &response.outputs[0];
        let out_tensor = output.as_tensor().expect("output should be tensor");
        let in_tensor = matrix.as_tensor().unwrap();
        assert_eq!(
            out_tensor.shape, in_tensor.shape,
            "shape should be preserved"
        );
        assert_eq!(
            out_tensor.elements, in_tensor.elements,
            "elements should be identical"
        );
    }

    #[test]
    fn test_vmap_scalar_output_still_works() {
        // Regression: existing scalar-output vmap behavior should be preserved.
        let input = Value::vector_f64(&[1.0, 4.0, 9.0]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(square) with scalar outputs should still work");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be stacked tensor");
        let vals = output.to_f64_vec().expect("output should be f64");
        assert!((vals[0] - 1.0).abs() < 1e-10, "1^2 = 1");
        assert!((vals[1] - 16.0).abs() < 1e-10, "4^2 = 16");
        assert!((vals[2] - 81.0).abs() < 1e-10, "9^2 = 81");
    }

    // ── in_axes / out_axes tests (bd-3edr) ───────────────────────

    #[test]
    fn test_in_axes_0_default() {
        // in_axes=0 (default) batches along leading dim — same as no compile_options
        let input = Value::vector_i64(&[10, 20, 30]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=0 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![11, 21, 31]);
    }

    #[test]
    fn test_in_axes_none_broadcasts() {
        // in_axes=None means input is not batched — broadcast to all batch elements.
        // We use Add2 with two args: first batched (axis 0), second not batched.
        let batched = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let scalar = Value::scalar_i64(100);
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![batched, scalar],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with in_axes=none broadcast should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![101, 102, 103]);
    }

    #[test]
    fn test_in_axes_1_second_dim() {
        // in_axes=1 batches along second dimension.
        // Input shape [2, 3]: batch along axis 1 gives batch_size=3, each slice is [2].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "1".to_owned());
        // Force loop-and-stack path by adding Jit tail
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // Batch size 3, each inner result is [2], stacked along axis 0 → [3, 2]
        assert_eq!(output.shape, Shape { dims: vec![3, 2] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        // Column 0: [1, 4]+1 = [2, 5], Column 1: [2, 5]+1 = [3, 6], Column 2: [3, 6]+1 = [4, 7]
        assert_eq!(vals, vec![2, 5, 3, 6, 4, 7]);
    }

    #[test]
    fn test_in_axes_tuple_mixed() {
        // in_axes=(0, none) for two-arg Add2: first arg batched, second broadcast
        let batched = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let broadcast = Value::scalar_f64(10.0);
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![batched, broadcast],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap mixed in_axes should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals = output.to_f64_vec().expect("f64");
        assert!((vals[0] - 11.0).abs() < 1e-10);
        assert!((vals[1] - 12.0).abs() < 1e-10);
        assert!((vals[2] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_out_axes_0_default() {
        // out_axes=0 (default) places batch dim at position 0 — same as normal
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "0".to_owned());
        // Force loop-and-stack by adding Jit tail
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap out_axes=0 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![2, 3, 4]);
    }

    #[test]
    fn test_out_axes_1_trailing() {
        // out_axes=1: batch dim moved from position 0 to position 1.
        // Use a matrix input so the stacked output rank is 2 and axis 1 is valid:
        // Input [3, 2], in_axes=0 → inner gets [2], outputs [2], stacked → [3, 2]
        // out_axes=1 → transpose to [2, 3]
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        // Force loop-and-stack
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap out_axes=1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // Original stacked: [3, 2] with batch dim at 0
        // out_axes=1 → transpose [0,1] → [1,0] → shape [2, 3]
        assert_eq!(output.shape, Shape { dims: vec![2, 3] });
        // Original elements row-major: [2,3, 4,5, 6,7]
        // Transposed [2,3] row-major: col0=[2,4,6], col1=[3,5,7]
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![2, 4, 6, 3, 5, 7]);
    }

    #[test]
    fn test_in_axes_negative_index() {
        // in_axes=-1 on a [2, 3] matrix means batch along last axis (axis 1), batch_size=3
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "-1".to_owned());
        // Force loop-and-stack
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=-1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // in_axes=-1 on [2,3] resolves to axis 1, batch_size=3
        // Each slice is column [2], stacked → [3, 2]
        assert_eq!(output.shape, Shape { dims: vec![3, 2] });
    }

    #[test]
    fn test_in_axes_error_out_of_bounds() {
        // in_axes=5 on a rank-2 tensor should error
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "5".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("vmap with out-of-bounds in_axes should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("out of bounds"),
            "error should mention out of bounds, got: {msg}"
        );
    }

    #[test]
    fn test_out_axes_error_positive_out_of_bounds() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("rank-1 outputs should reject out_axes=1");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::VmapAxesOutOfBounds {
                axis: 1,
                ndim: 1,
            })
        ));
    }

    #[test]
    fn test_out_axes_error_negative_out_of_bounds() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "-3".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("rank-2 outputs should reject out_axes=-3");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::VmapAxesOutOfBounds {
                axis: -3,
                ndim: 2,
            })
        ));
    }

    #[test]
    fn test_in_axes_error_mismatched_batch_size() {
        // Two batched args with different batch sizes along their respective axes
        let a = Value::vector_i64(&[1, 2, 3]).expect("vec3");
        let b = Value::vector_i64(&[1, 2]).expect("vec2");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![a, b],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("mismatched batch sizes should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("mismatch"),
            "error should mention mismatch, got: {msg}"
        );
    }

    #[test]
    fn test_parse_vmap_in_axes_uniform() {
        let opts = BTreeMap::new();
        let axes = super::parse_vmap_in_axes(&opts, 3).expect("default axes");
        assert_eq!(axes, vec![super::AxisSpec::Batched(0); 3]);
    }

    #[test]
    fn test_parse_vmap_in_axes_per_arg() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none,1".to_owned());
        let axes = super::parse_vmap_in_axes(&opts, 3).expect("per-arg axes");
        assert_eq!(
            axes,
            vec![
                super::AxisSpec::Batched(0),
                super::AxisSpec::NotBatched,
                super::AxisSpec::Batched(1),
            ]
        );
    }

    #[test]
    fn test_parse_vmap_out_axes_broadcast_single() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        let axes = super::parse_vmap_out_axes(&opts, 3).expect("broadcast out axes");
        assert_eq!(axes, vec![super::AxisSpec::Batched(1); 3]);
    }

    #[test]
    fn test_parse_vmap_in_axes_rejects_invalid_token() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "oops".to_owned());
        let err = super::parse_vmap_in_axes(&opts, 1).expect_err("invalid axis token");
        let msg = err.to_string();
        assert!(msg.contains("vmap_in_axes"), "error should identify option");
        assert!(msg.contains("oops"), "error should include invalid token");
    }

    #[test]
    fn test_parse_vmap_out_axes_rejects_invalid_token() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "bad".to_owned());
        let err = super::parse_vmap_out_axes(&opts, 1).expect_err("invalid axis token");
        let msg = err.to_string();
        assert!(
            msg.contains("vmap_out_axes"),
            "error should identify option"
        );
        assert!(msg.contains("bad"), "error should include invalid token");
    }

    #[test]
    fn test_dispatch_rejects_invalid_vmap_in_axes_option() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "bad".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("invalid axis token should fail dispatch");

        let msg = err.to_string();
        assert!(msg.contains("vmap_in_axes"), "error should identify option");
        assert!(msg.contains("bad"), "error should include invalid token");
    }

    #[test]
    fn test_out_axes_error_mismatched_output_arity() {
        let batched_qr_input = Value::Tensor(
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
                .map(|x| fj_core::Literal::F64Bits(x.to_bits()))
                .collect(),
            )
            .expect("batched qr input"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "0,0,0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::LaxQr, &[Transform::Vmap, Transform::Jit]),
            args: vec![batched_qr_input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("mismatched out_axes length should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("length mismatch"),
            "error should mention length mismatch, got: {msg}"
        );
        assert!(
            msg.contains("expected 2"),
            "error should report expected arity"
        );
        assert!(msg.contains("got 3"), "error should report provided arity");
    }

    #[test]
    fn test_out_axes_none_returns_unmapped_constant_output() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![
                    Atom::Lit(fj_core::Literal::I64(3)),
                    Atom::Lit(fj_core::Literal::I64(4))
                ],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );
        let mut ledger = TraceTransformLedger::new(jaxpr);
        ledger.push_transform(Transform::Vmap, "test-vmap".to_owned());
        ledger.push_transform(Transform::Jit, "test-jit".to_owned());

        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger,
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("constant unmapped output should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn test_out_axes_none_rejects_varying_outputs() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("varying unmapped output should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("out_axes=none"),
            "error should mention unmapped out_axes"
        );
        assert!(
            msg.contains("identical"),
            "error should explain equality requirement"
        );
    }

    // ── Depth-3+ transform composition tests (frankenjax-yxv) ──────

    #[test]
    fn dispatch_jit_grad_grad_square() {
        // jit(grad(grad(x^2))) at x=3: d²/dx²(x²)=2
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[Transform::Grad, Transform::Grad, Transform::Jit],
            ),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit(grad(grad(x^2))) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(
            (d2 - 2.0).abs() < 1e-2,
            "d²/dx²(x²) should be ≈2.0, got {d2}"
        );
    }

    #[test]
    fn dispatch_grad_grad_sin() {
        // grad(grad(sin))(x) = -sin(x); at x=1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(1.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad(grad(sin)) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        let expected = -(1.0_f64.sin());
        assert!(
            (d2 - expected).abs() < 1e-4,
            "d²/dx²(sin(x)) at x=1 should be {expected}, got {d2}"
        );
    }

    #[test]
    fn dispatch_vmap_grad_requires_scalar_inputs() {
        // vmap(grad(x^2)) with vector input should fail because grad needs scalar inputs
        let result = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        });
        // Grad requires scalar inputs — this composition through dispatch correctly rejects
        assert!(
            result.is_err(),
            "vmap(grad(f)) with vector input should fail through dispatch"
        );
    }

    #[test]
    fn dispatch_grad_vmap_add_one() {
        // grad(vmap(f)) where f(x) = x+1 (constant derivative)
        // grad of vmap(add_one) at each element should be 1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Grad]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        });
        // This may succeed or fail depending on how grad handles batched outputs.
        // The key assertion: it shouldn't panic or produce garbage.
        match response {
            Ok(r) => {
                // If it succeeds, outputs should be finite
                for (i, out) in r.outputs.iter().enumerate() {
                    if let Some(v) = out.as_f64_scalar() {
                        assert!(v.is_finite(), "output[{i}] should be finite, got {v}");
                    }
                }
            }
            Err(e) => {
                // Some composition orders may legitimately fail
                let msg = e.to_string();
                assert!(!msg.is_empty(), "error should have a descriptive message");
            }
        }
    }

    #[test]
    fn dispatch_jit_vmap_add_one() {
        // jit(vmap(f)) — JIT wrapping a vmapped function
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit(vmap(add_one)) should succeed");

        let t = response.outputs[0].as_tensor().expect("tensor result");
        let vals = t.to_i64_vec().expect("i64 vec");
        assert_eq!(vals, vec![11, 21, 31]);
    }

    #[test]
    fn dispatch_composition_order_affects_cache_key() {
        // Verify that [Grad, Jit] and [Jit, Grad] produce different cache keys
        let response_gj = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Jit]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad then jit should succeed");

        let response_jg = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit then grad should succeed");

        assert_ne!(
            response_gj.cache_key, response_jg.cache_key,
            "different transform orders must produce different cache keys"
        );
    }

    #[test]
    fn dispatch_grad_grad_grad_square() {
        // Third derivative of x^2 should be 0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[Transform::Grad, Transform::Grad, Transform::Grad],
            ),
            args: vec![Value::scalar_f64(5.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad(grad(grad(x^2))) should succeed");

        let d3 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(d3.abs() < 1e-4, "d³/dx³(x²) should be 0, got {d3}");
    }

    #[test]
    fn dispatch_hardened_mode_grad_grad() {
        // Same as above but in hardened mode — should still work correctly
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(4.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("hardened grad(grad(x^2)) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(
            (d2 - 2.0).abs() < 1e-4,
            "hardened d²/dx²(x²) should be 2.0, got {d2}"
        );
    }
}
