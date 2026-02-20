#![forbid(unsafe_code)]

use fj_cache::{CacheKeyError, CacheKeyInputRef, build_cache_key_ref};
use fj_core::{
    CompatibilityMode, Jaxpr, TensorValue, TraceTransformLedger, Transform,
    TransformCompositionError, Value, verify_transform_composition,
};
use fj_interpreters::{InterpreterError, eval_jaxpr};
use fj_ledger::{
    ConformalPredictor, DecisionRecord, EvidenceLedger, EvidenceSignal, LedgerEntry, LossMatrix,
};
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
/// order. No execution ordering is enforced (effects modeled via evidence
/// ledger signals rather than runtime token threading).
#[derive(Debug, Clone)]
pub struct EffectContext {
    tokens: BTreeMap<String, EffectToken>,
    next_sequence: u64,
}

impl EffectContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: BTreeMap::new(),
            next_sequence: 0,
        }
    }

    /// Record observation of a named effect. Returns the token with its
    /// sequence number in the observation order.
    pub fn thread_token(&mut self, effect_name: &str) -> EffectToken {
        let token = EffectToken {
            effect_name: effect_name.to_owned(),
            sequence_number: self.next_sequence,
        };
        self.next_sequence += 1;
        self.tokens.insert(effect_name.to_owned(), token.clone());
        token
    }

    /// Finalize and return all observed effect tokens in sequence order.
    #[must_use]
    pub fn finalize(self) -> Vec<EffectToken> {
        let mut tokens: Vec<EffectToken> = self.tokens.into_values().collect();
        tokens.sort_by_key(|t| t.sequence_number);
        tokens
    }

    /// Number of distinct effects observed.
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
    VmapNonScalarOutput,
    VmapInconsistentOutputArity { expected: usize, actual: usize },
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
            Self::VmapNonScalarOutput => {
                write!(
                    f,
                    "vmap inner function must currently return scalar outputs"
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
            Self::TensorBuild(detail) => write!(f, "tensor build error: {detail}"),
        }
    }
}

impl std::error::Error for TransformExecutionError {}

#[derive(Debug)]
pub enum DispatchError {
    Cache(CacheKeyError),
    Interpreter(InterpreterError),
    TransformInvariant(TransformCompositionError),
    TransformExecution(TransformExecutionError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(err) => write!(f, "cache key error: {err}"),
            Self::Interpreter(err) => write!(f, "interpreter error: {err}"),
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

    let cache_key = build_cache_key_ref(&CacheKeyInputRef {
        mode: request.mode,
        backend: &request.backend,
        jaxpr: &request.ledger.root_jaxpr,
        transform_stack: &request.ledger.transform_stack,
        compile_options: &request.compile_options,
        custom_hook: request.custom_hook.as_deref(),
        unknown_incompatible_features: &request.unknown_incompatible_features,
    })?;

    // Thread effect context through transform execution
    let mut effect_ctx = EffectContext::new();
    for transform in &request.ledger.transform_stack {
        effect_ctx.thread_token(transform.as_str());
    }

    let outputs = execute_with_transforms(
        &request.ledger.root_jaxpr,
        &request.ledger.transform_stack,
        &request.args,
    )?;

    let effect_tokens = effect_ctx.finalize();
    let effect_count = effect_tokens.len();

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
                signal_name: "effect_token_count".to_owned(),
                log_likelihood_delta: (effect_count as f64 + 1.0).ln(),
                detail: format!("effect_tokens={effect_count}"),
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
) -> Result<Vec<Value>, DispatchError> {
    let Some((head, tail)) = transforms.split_first() else {
        return eval_jaxpr(root_jaxpr, args).map_err(DispatchError::from);
    };

    match head {
        Transform::Jit => execute_with_transforms(root_jaxpr, tail, args),
        Transform::Grad => execute_grad(root_jaxpr, tail, args),
        Transform::Vmap => execute_vmap(root_jaxpr, tail, args),
    }
}

fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Grad,
        }
        .into());
    }

    args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    // If there are remaining transforms in the tail, fall back to finite-diff
    // (symbolic AD only applies to the innermost evaluation).
    if !tail.is_empty() {
        return execute_grad_finite_diff(root_jaxpr, tail, args);
    }

    let derivative = fj_ad::grad_first(root_jaxpr, args)
        .map_err(|e| TransformExecutionError::TensorBuild(format!("AD error: {e}")))?;
    Ok(vec![Value::scalar_f64(derivative)])
}

fn execute_grad_finite_diff(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError> {
    let input_value = args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    let epsilon = 1e-6_f64;
    let mut plus_args = args.to_vec();
    let mut minus_args = args.to_vec();
    plus_args[0] = Value::scalar_f64(input_value + epsilon);
    minus_args[0] = Value::scalar_f64(input_value - epsilon);

    let plus_out = execute_with_transforms(root_jaxpr, tail, &plus_args)?;
    let minus_out = execute_with_transforms(root_jaxpr, tail, &minus_args)?;

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
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Vmap,
        }
        .into());
    }

    let lead_tensor = args[0]
        .as_tensor()
        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?;
    if lead_tensor.rank() == 0 {
        return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
    }

    let lead_len = lead_tensor
        .leading_dim()
        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?
        as usize;
    if lead_len == 0 {
        return Err(TransformExecutionError::EmptyVmapOutput.into());
    }

    let mut per_output_values: Vec<Vec<Value>> = Vec::new();

    for index in 0..lead_len {
        let mut mapped_args = Vec::with_capacity(args.len());
        mapped_args.push(
            lead_tensor
                .slice_axis0(index)
                .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?,
        );

        for arg in &args[1..] {
            match arg {
                Value::Scalar(lit) => mapped_args.push(Value::Scalar(*lit)),
                Value::Tensor(tensor) => {
                    if tensor.rank() == 0 {
                        return Err(
                            TransformExecutionError::VmapRequiresRankOneLeadingArgument.into()
                        );
                    }
                    let arg_lead = tensor
                        .leading_dim()
                        .ok_or(TransformExecutionError::VmapRequiresRankOneLeadingArgument)?
                        as usize;
                    if arg_lead != lead_len {
                        return Err(TransformExecutionError::VmapMismatchedLeadingDimension {
                            expected: lead_len,
                            actual: arg_lead,
                        }
                        .into());
                    }
                    mapped_args.push(
                        tensor
                            .slice_axis0(index)
                            .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?,
                    );
                }
            }
        }

        let mapped_output = execute_with_transforms(root_jaxpr, tail, &mapped_args)?;
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

    let mut outputs = Vec::with_capacity(per_output_values.len());
    for values in per_output_values {
        let tensor = TensorValue::stack_axis0(&values)
            .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
        outputs.push(Value::Tensor(tensor));
    }

    Ok(outputs)
}

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
    use super::{DispatchError, DispatchRequest, dispatch};
    use fj_core::{
        CompatibilityMode, DType, ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform,
        Value, build_program,
    };
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

        match err {
            DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures {
                features,
            }) => {
                assert_eq!(features, vec!["future.backend.protocol.v2".to_owned()]);
            }
            other => {
                panic!("expected fail-closed cache-key rejection, got: {other:?}");
            }
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
    fn effect_context_tracks_transform_tokens() {
        use super::EffectContext;
        let mut ctx = EffectContext::new();
        let t1 = ctx.thread_token("jit");
        let t2 = ctx.thread_token("grad");
        let t3 = ctx.thread_token("vmap");

        assert_eq!(t1.sequence_number, 0);
        assert_eq!(t2.sequence_number, 1);
        assert_eq!(t3.sequence_number, 2);
        assert_eq!(ctx.effect_count(), 3);

        let tokens = ctx.finalize();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].effect_name, "jit");
        assert_eq!(tokens[1].effect_name, "grad");
        assert_eq!(tokens[2].effect_name, "vmap");
    }

    #[test]
    fn dispatch_includes_effect_token_signal() {
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

        // Evidence ledger should have 1 entry with 4 signals (including effect_token_count)
        assert_eq!(response.evidence_ledger.len(), 1);
        let entry = &response.evidence_ledger.entries()[0];
        assert_eq!(entry.signals.len(), 4);

        let effect_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(effect_signal.detail, "effect_tokens=2");
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
        assert_ne!(r1.cache_key, r3.cache_key, "different program = different key");
    }
}
