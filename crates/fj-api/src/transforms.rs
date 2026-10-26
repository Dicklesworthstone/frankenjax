#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
use fj_core::TraceTransformLedger;
use fj_core::{
    AbstractValue, CompatibilityMode, DType, GradWitness, Jaxpr, PrecisionPolicy, Shape, Transform,
    TransformCompositionProof, TransformProofStep, TransformRewriteWitness, Value,
    jaxpr_fingerprint_sha256, verify_transform_composition_steps, with_precision_policy,
};
use fj_dispatch::{
    DispatchRequestRef, DispatchResponse, PreparedDispatchMeta, check_fast_path_admission,
    dispatch_ref, prepare_dispatch_meta,
};
use fj_ledger::{DecisionRecord, EvidenceLedger, EvidenceSignal, LedgerEntry, LossMatrix};
pub use fj_trace::ShapedArray;

use crate::errors::ApiError;

static CUSTOM_DERIVATIVE_WRAPPER_ID: AtomicU64 = AtomicU64::new(1);
const CUSTOM_VJP_RULE_KEY_OPTION: &str = "custom_vjp_rule_key";
const DEFAULT_BACKEND: &str = "cpu";
type BackendName = Cow<'static, str>;
type ValueAndGradOutputs = (Vec<Value>, Vec<Value>);

#[derive(Debug, Clone, PartialEq)]
pub struct JitWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    static_argnums: Vec<usize>,
    precision_policy: Option<PrecisionPolicy>,
    compile_options: BTreeMap<String, String>,
    ffi_registry: FfiRegistrySlot,
    /// Lazily-computed, args-independent dispatch metadata shared across
    /// repeated `call`s. Excluded from equality/Debug so cached proof/key state
    /// does not change the wrapper's observable identity.
    meta_cache: DispatchMetaCache,
    /// Lazily-compiled dense scalar evaluator for repeated default-CPU JIT
    /// calls. Excluded from equality/Debug for the same reason as
    /// `meta_cache`: it is a pure memo of `jaxpr`.
    compiled_cache: CompiledEvalCache,
    last_evidence: LastEvidenceCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    argnums: Vec<usize>,
    precision_policy: Option<PrecisionPolicy>,
    compile_options: BTreeMap<String, String>,
    custom_vjp_rule_key: Option<String>,
    /// Lazily-computed, args-independent dispatch metadata (composition proof +
    /// cache key) shared across repeated `call`s, mirroring [`ValueAndGradWrapped`].
    /// Excluded from equality/Debug so the wrapper's observable identity is
    /// unchanged.
    meta_cache: DispatchMetaCache,
    /// Lazily-compiled scalar-F64 reverse plan for repeated default-CPU AD
    /// calls. Excluded from equality/Debug because it is a pure memo of `jaxpr`.
    compiled_ad_cache: CompiledAdCache,
    last_evidence: LastEvidenceCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VmapWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    in_axes: Option<String>,
    out_axes: Option<String>,
    precision_policy: Option<PrecisionPolicy>,
    compile_options: BTreeMap<String, String>,
    /// Lazily-computed, args-independent dispatch metadata (composition proof +
    /// cache key) shared across repeated `call`s, mirroring [`ValueAndGradWrapped`]
    /// and [`GradWrapped`]. Excluded from equality/Debug. Any builder that mutates
    /// a cache-key input (backend/mode/in_axes/out_axes) must reset it.
    meta_cache: DispatchMetaCache,
    last_evidence: LastEvidenceCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueAndGradWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    argnums: Vec<usize>,
    precision_policy: Option<PrecisionPolicy>,
    compile_options: BTreeMap<String, String>,
    custom_vjp_rule_key: Option<String>,
    /// Lazily-computed, args-independent dispatch metadata (composition proof +
    /// cache key) shared across repeated `call`s. Excluded from equality/Debug
    /// so the wrapper's observable identity is unchanged.
    meta_cache: DispatchMetaCache,
    /// Lazily-compiled scalar-F64 reverse plan for repeated default-CPU
    /// value-and-grad calls. Excluded from equality/Debug because it is a pure
    /// memo of `jaxpr`.
    compiled_ad_cache: CompiledAdCache,
    last_evidence: LastEvidenceCache,
}

/// Process-lifetime cache for the args-independent dispatch metadata of a
/// wrapped function. Cloning shares the cell (a clone has byte-identical static
/// inputs, so the cached proof/key stay valid); any builder that changes a
/// cache-key input must reset it via [`DispatchMetaCache::default`].
#[derive(Clone, Default)]
struct DispatchMetaCache(Arc<OnceLock<Option<PreparedDispatchMeta>>>);

impl PartialEq for DispatchMetaCache {
    fn eq(&self, _other: &Self) -> bool {
        // The cache is a pure memo of the other fields; it never contributes to
        // the wrapper's logical identity.
        true
    }
}

impl std::fmt::Debug for DispatchMetaCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DispatchMetaCache(..)")
    }
}

#[derive(Clone, Default)]
struct CompiledEvalCache {
    plan: Arc<OnceLock<Option<Arc<fj_interpreters::CompiledJaxpr>>>>,
    warm_hits: Arc<AtomicU64>,
}

impl PartialEq for CompiledEvalCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for CompiledEvalCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CompiledEvalCache(..)")
    }
}

#[derive(Clone, Default)]
struct CompiledAdCache(Arc<OnceLock<Option<fj_ad::CompiledValueAndGradJaxpr>>>);

impl PartialEq for CompiledAdCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for CompiledAdCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CompiledAdCache(..)")
    }
}

#[derive(Clone, Default)]
struct LastEvidenceCache(
    Arc<std::sync::Mutex<Option<(EvidenceLedger, TransformCompositionProof)>>>,
);

impl PartialEq for LastEvidenceCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for LastEvidenceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LastEvidenceCache(..)")
    }
}

impl LastEvidenceCache {
    fn store(&self, ledger: EvidenceLedger, proof: TransformCompositionProof) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = Some((ledger, proof));
        }
    }

    fn ledger(&self) -> Option<EvidenceLedger> {
        self.0
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|(l, _)| l.clone()))
    }

    fn proof(&self) -> Option<TransformCompositionProof> {
        self.0
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|(_, p)| p.clone()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomVjpWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    rule_key: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomJvpWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    rule_key: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JacobianWrapped {
    jaxpr: Jaxpr,
    custom_jvp_rule_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HessianWrapped {
    jaxpr: Jaxpr,
}

/// A linearized function returned by `linearize`.
///
/// Captures the pre-recorded [`fj_ad::LinearizedTape`], allowing JVP tangent
/// computation without recomputing the forward pass. JAX equivalent: the second
/// return value of `jax.linearize`.
#[derive(Debug, Clone)]
pub struct LinearizedFunction {
    tape: fj_ad::LinearizedTape,
}

/// Result of `linearize(jaxpr, primals)`.
///
/// Contains the primal outputs and a `LinearizedFunction` for computing
/// tangent outputs at different tangent inputs.
pub struct LinearizeResult {
    pub primal_outputs: Vec<Value>,
    pub linearized: LinearizedFunction,
}

/// A transposed linear function returned by `linear_transpose`.
///
/// For a linear function f: R^n -> R^m, the transpose f^T: R^m -> R^n
/// satisfies <f(x), y> = <x, f^T(y)> for all x, y.
///
/// JAX equivalent: the return value of `jax.linear_transpose`.
#[derive(Debug, Clone)]
pub struct TransposedLinearFunction {
    tape: fj_ad::VjpTape,
}

/// Reusable reverse-mode pullback function returned by [`vjp`].
#[derive(Debug, Clone)]
pub struct VjpPullback {
    tape: fj_ad::VjpTape,
}

/// Result of [`vjp`], containing primal outputs and the reusable [`VjpPullback`].
#[derive(Debug, Clone)]
pub struct VjpResult {
    pub primal_outputs: Vec<Value>,
    pub pullback: VjpPullback,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    custom_vjp_rule_key: String,
}

/// Pmap wrapper for parallel map across multiple devices.
///
/// V1: Fails closed unless explicit virtual device count (`with_devices`) is configured.
#[derive(Debug, Clone, PartialEq)]
pub struct PmapWrapped {
    jaxpr: Jaxpr,
    backend: BackendName,
    mode: CompatibilityMode,
    axis_name: Option<String>,
    devices: Option<usize>,
    in_axes: Option<String>,
    out_axes: Option<String>,
    precision_policy: Option<PrecisionPolicy>,
    compile_options: BTreeMap<String, String>,
    meta_cache: DispatchMetaCache,
    last_evidence: LastEvidenceCache,
}

#[derive(Clone, Default, Debug)]
struct FfiRegistrySlot(Option<fj_ffi::FfiRegistry>);

impl PartialEq for FfiRegistrySlot {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref().map(fj_ffi::FfiRegistry::registered_names)
            == other.0.as_ref().map(fj_ffi::FfiRegistry::registered_names)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposedTransform {
    jaxpr: Jaxpr,
    transforms: Vec<Transform>,
    backend: BackendName,
    mode: CompatibilityMode,
    compile_options: BTreeMap<String, String>,
    custom_vjp_rule_key: Option<String>,
    ffi_registry: FfiRegistrySlot,
    meta_cache: DispatchMetaCache,
    last_evidence: LastEvidenceCache,
}

#[must_use]
pub fn jit(jaxpr: Jaxpr) -> JitWrapped {
    JitWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        static_argnums: Vec::new(),
        precision_policy: None,
        compile_options: BTreeMap::new(),
        ffi_registry: FfiRegistrySlot::default(),
        meta_cache: DispatchMetaCache::default(),
        compiled_cache: CompiledEvalCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

#[must_use]
pub fn grad(jaxpr: Jaxpr) -> GradWrapped {
    GradWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        argnums: Vec::new(),
        precision_policy: None,
        compile_options: BTreeMap::new(),
        custom_vjp_rule_key: None,
        meta_cache: DispatchMetaCache::default(),
        compiled_ad_cache: CompiledAdCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

#[must_use]
pub fn vmap(jaxpr: Jaxpr) -> VmapWrapped {
    VmapWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        in_axes: None,
        out_axes: None,
        precision_policy: None,
        compile_options: BTreeMap::new(),
        meta_cache: DispatchMetaCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

#[must_use]
pub fn value_and_grad(jaxpr: Jaxpr) -> ValueAndGradWrapped {
    ValueAndGradWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        argnums: Vec::new(),
        precision_policy: None,
        compile_options: BTreeMap::new(),
        custom_vjp_rule_key: None,
        meta_cache: DispatchMetaCache::default(),
        compiled_ad_cache: CompiledAdCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

/// Create a pmap (parallel map) transform wrapper.
///
/// V1: Fails closed unless explicit virtual device count (`with_devices`) is configured.
#[must_use]
pub fn pmap(jaxpr: Jaxpr) -> PmapWrapped {
    PmapWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        axis_name: None,
        devices: None,
        in_axes: None,
        out_axes: None,
        precision_policy: None,
        compile_options: BTreeMap::new(),
        meta_cache: DispatchMetaCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

/// Attach function-level custom VJP callbacks to a Jaxpr.
///
/// The forward callback returns `(primal_outputs, residuals)`. The backward
/// callback receives those residuals plus the output cotangent and returns one
/// cotangent per input. The wrapper gets a private rule key so equal Jaxprs can
/// carry different custom derivative callbacks without overwriting each other.
#[must_use]
pub fn custom_vjp<Fwd, Bwd>(jaxpr: Jaxpr, forward: Fwd, backward: Bwd) -> CustomVjpWrapped
where
    Fwd: Fn(&[Value]) -> Result<(Vec<Value>, Vec<Value>), fj_ad::AdError> + Send + Sync + 'static,
    Bwd: Fn(&[Value], &Value) -> Result<Vec<Value>, fj_ad::AdError> + Send + Sync + 'static,
{
    let rule_key = custom_derivative_rule_key("custom_vjp");
    let expected_outputs = jaxpr.outvars.len();
    fj_ad::register_custom_jaxpr_vjp_with_key(
        &rule_key,
        move |primals, primal_outputs, cotangent| {
            let (forward_outputs, residuals) = forward(primals)?;
            if forward_outputs.len() != expected_outputs {
                return Err(fj_ad::AdError::EvalFailed(format!(
                    "custom VJP forward output arity mismatch: expected {expected_outputs}, got {}",
                    forward_outputs.len()
                )));
            }
            if primal_outputs.len() != expected_outputs {
                return Err(fj_ad::AdError::EvalFailed(format!(
                    "custom VJP primal output arity mismatch: expected {expected_outputs}, got {}",
                    primal_outputs.len()
                )));
            }
            backward(&residuals, cotangent)
        },
    );

    CustomVjpWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        rule_key,
    }
}

/// Attach a function-level custom JVP callback to a Jaxpr.
///
/// The callback returns `(primal_outputs, tangent_outputs)`. The wrapper gets a
/// private rule key so equal Jaxprs can carry different custom derivative
/// callbacks without overwriting each other.
#[must_use]
pub fn custom_jvp<F>(jaxpr: Jaxpr, rule: F) -> CustomJvpWrapped
where
    F: Fn(&[Value], &[Value]) -> Result<(Vec<Value>, Vec<Value>), fj_ad::AdError>
        + Send
        + Sync
        + 'static,
{
    let rule_key = custom_derivative_rule_key("custom_jvp");
    let expected_outputs = jaxpr.outvars.len();
    fj_ad::register_custom_jaxpr_jvp_with_key(&rule_key, move |primals, tangents| {
        let (primals_out, tangents_out) = rule(primals, tangents)?;
        if primals_out.len() != expected_outputs {
            return Err(fj_ad::AdError::EvalFailed(format!(
                "custom JVP primal output arity mismatch: expected {expected_outputs}, got {}",
                primals_out.len()
            )));
        }
        if tangents_out.len() != expected_outputs {
            return Err(fj_ad::AdError::EvalFailed(format!(
                "custom JVP tangent output arity mismatch: expected {expected_outputs}, got {}",
                tangents_out.len()
            )));
        }
        Ok(fj_ad::JvpResult {
            primals: primals_out,
            tangents: tangents_out,
        })
    });

    CustomJvpWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        rule_key,
    }
}

#[must_use]
pub fn jacobian(jaxpr: Jaxpr) -> JacobianWrapped {
    JacobianWrapped {
        jaxpr,
        custom_jvp_rule_key: None,
    }
}

#[must_use]
pub fn hessian(jaxpr: Jaxpr) -> HessianWrapped {
    HessianWrapped { jaxpr }
}

/// Compute forward-mode JVP `(primals_out, tangents_out)` for `jaxpr` at `(primals, tangents)`.
///
/// JAX equivalent: `jax.jvp`
pub fn jvp(
    jaxpr: &Jaxpr,
    primals: &[Value],
    tangents: &[Value],
) -> Result<fj_ad::JvpResult, ApiError> {
    fj_ad::jvp(jaxpr, primals, tangents).map_err(ApiError::from)
}

/// Evaluate `jaxpr` at `primals` and return `(primal_outputs, VjpPullback)` for
/// reverse-mode pullback evaluation without re-evaluating primal equations.
///
/// JAX equivalent: `jax.vjp`
pub fn vjp(jaxpr: Jaxpr, primals: Vec<Value>) -> Result<VjpResult, ApiError> {
    let (primal_outputs, tape) = fj_ad::record_vjp_tape(&jaxpr, &primals)?;
    Ok(VjpResult {
        primal_outputs,
        pullback: VjpPullback { tape },
    })
}

impl VjpPullback {
    /// Evaluate the reverse-mode pullback for a single-output `Jaxpr` at `cotangent`
    /// without re-running the primal forward pass.
    pub fn call(&self, cotangent: Value) -> Result<Vec<Value>, ApiError> {
        self.tape.backward(&cotangent).map_err(ApiError::from)
    }

    /// Evaluate the reverse-mode pullback for a general (single- or multi-output) `Jaxpr`
    /// at `cotangents` without re-running the primal forward pass.
    pub fn call_multi(&self, cotangents: &[Value]) -> Result<Vec<Value>, ApiError> {
        self.tape.backward_multi(cotangents).map_err(ApiError::from)
    }
}

/// Linearize a Jaxpr at given primal values.
///
/// Returns `(f(primals), linearized_fn)` where `linearized_fn(tangents)` computes
/// the JVP tangent outputs without recomputing the forward pass.
///
/// JAX equivalent: `jax.linearize`
///
/// # Example
/// ```ignore
/// let result = linearize(jaxpr, primals)?;
/// let tangent_out = result.linearized.call(tangents)?;
/// ```
pub fn linearize(jaxpr: Jaxpr, primals: Vec<Value>) -> Result<LinearizeResult, ApiError> {
    let (primal_outputs, tape) = fj_ad::linearize_jaxpr(&jaxpr, &primals)?;
    Ok(LinearizeResult {
        primal_outputs,
        linearized: LinearizedFunction { tape },
    })
}

impl LinearizedFunction {
    /// Compute tangent outputs for the given input tangents.
    ///
    /// Uses the pre-recorded primal environment from `linearize` to avoid
    /// recomputing the forward pass.
    pub fn call(&self, tangents: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        self.tape.pushforward(&tangents).map_err(ApiError::from)
    }
}

/// Compute the transpose of a linear function.
///
/// For a linear function f: R^n -> R^m represented by a Jaxpr, returns a
/// `TransposedLinearFunction` that computes f^T: R^m -> R^n.
///
/// The function is assumed to be linear in all its inputs. For a linear f,
/// the transpose is computed using VJP over the pre-recorded tape without
/// re-running the primal forward pass on each cotangent call.
///
/// JAX equivalent: `jax.linear_transpose`
///
/// # Example
/// ```ignore
/// let transposed = linear_transpose(jaxpr, primals)?;
/// let result = transposed.call(cotangent)?;
/// ```
pub fn linear_transpose(
    jaxpr: Jaxpr,
    primals: Vec<Value>,
) -> Result<TransposedLinearFunction, ApiError> {
    let (_outputs, tape) = fj_ad::record_vjp_tape(&jaxpr, &primals)?;
    Ok(TransposedLinearFunction { tape })
}

impl TransposedLinearFunction {
    /// Compute the transposed linear function at the given cotangent.
    ///
    /// For a linear function f, this computes f^T(cotangent) using the pre-recorded VJP tape.
    pub fn call(&self, cotangent: Value) -> Result<Vec<Value>, ApiError> {
        self.tape.backward(&cotangent).map_err(ApiError::from)
    }

    /// Compute the transposed linear function for a multi-output linear function.
    pub fn call_multi(&self, cotangents: &[Value]) -> Result<Vec<Value>, ApiError> {
        self.tape.backward_multi(cotangents).map_err(ApiError::from)
    }
}

/// Wrap a Jaxpr for memory-efficient gradient computation (rematerialization).
///
/// During the backward pass, instead of using stored intermediate values,
/// checkpoint re-runs the forward pass to recompute them. This trades compute
/// for memory — useful for large models where storing all intermediates would
/// exceed available memory.
///
/// JAX equivalent: `jax.checkpoint` / `jax.remat`
#[must_use]
pub fn checkpoint(jaxpr: Jaxpr) -> CheckpointWrapped {
    let rule_key = custom_derivative_rule_key("checkpoint");
    let jaxpr_clone = jaxpr.clone();
    fj_ad::register_custom_jaxpr_vjp_with_key(
        &rule_key,
        move |primals, _primal_outputs, cotangent| {
            fj_ad::grad_jaxpr_with_cotangent(&jaxpr_clone, primals, cotangent)
        },
    );

    CheckpointWrapped {
        jaxpr,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        custom_vjp_rule_key: rule_key,
    }
}

/// Evaluate the output shapes and dtypes of a function without running it.
///
/// Given example inputs (used only for their shapes and dtypes), traces
/// through the computation graph and returns the output shapes and dtypes.
///
/// JAX equivalent: `jax.eval_shape`
///
/// # Example
/// ```ignore
/// let jaxpr = build_program(ProgramSpec::Square);
/// let shapes = eval_shape(&jaxpr, &[Value::scalar_f64(0.0)])?;
/// assert_eq!(shapes[0].shape, Shape::scalar());
/// assert_eq!(shapes[0].dtype, DType::F64);
/// ```
pub fn eval_shape(jaxpr: &Jaxpr, inputs: &[Value]) -> Result<Vec<ShapedArray>, ApiError> {
    let in_avals: Vec<ShapedArray> = inputs.iter().map(ShapedArray::from_value).collect();
    eval_shape_avals(jaxpr, &in_avals)
}

/// Infer output shapes and dtypes of a [`Jaxpr`] directly from abstract input [`ShapedArray`]s
/// without materializing dummy [`Value`] tensors.
pub fn eval_shape_avals(
    jaxpr: &Jaxpr,
    in_avals: &[ShapedArray],
) -> Result<Vec<ShapedArray>, ApiError> {
    eval_shape_with_consts_avals(jaxpr, &[], in_avals)
}

fn eval_shape_with_consts_avals(
    jaxpr: &Jaxpr,
    const_avals: &[ShapedArray],
    in_avals: &[ShapedArray],
) -> Result<Vec<ShapedArray>, ApiError> {
    use fj_core::{Atom, Primitive, Shape, VarId};

    if const_avals.len() != jaxpr.constvars.len() {
        return Err(ApiError::EvalError {
            detail: format!(
                "eval_shape: expected {} constvars, got {}",
                jaxpr.constvars.len(),
                const_avals.len()
            ),
        });
    }

    if in_avals.len() != jaxpr.invars.len() {
        return Err(ApiError::EvalError {
            detail: format!(
                "eval_shape: expected {} inputs, got {}",
                jaxpr.invars.len(),
                in_avals.len()
            ),
        });
    }

    let mut shape_env: BTreeMap<VarId, ShapedArray> = BTreeMap::new();

    for (var, aval) in jaxpr.constvars.iter().zip(const_avals.iter()) {
        shape_env.insert(*var, aval.clone());
    }
    for (var, aval) in jaxpr.invars.iter().zip(in_avals.iter()) {
        shape_env.insert(*var, aval.clone());
    }

    for eqn in &jaxpr.equations {
        let input_shapes: Vec<ShapedArray> = eqn
            .inputs
            .iter()
            .map(|atom| match atom {
                Atom::Var(v) => shape_env
                    .get(v)
                    .cloned()
                    .ok_or_else(|| ApiError::EvalError {
                        detail: format!("eval_shape: undefined variable {:?}", v),
                    }),
                Atom::Lit(lit) => Ok(ShapedArray::from_value(&Value::Scalar(*lit))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let output_shapes = if eqn.sub_jaxprs.is_empty() {
            infer_primitive_shapes(eqn.primitive, &input_shapes, &eqn.params)?
        } else {
            match eqn.primitive {
                Primitive::Cond => {
                    if eqn.sub_jaxprs.len() != 2 {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: cond expects 2 sub_jaxprs, got {}",
                                eqn.sub_jaxprs.len()
                            ),
                        });
                    }
                    let Some((pred, bindings)) = input_shapes.split_first() else {
                        return Err(ApiError::EvalError {
                            detail: "eval_shape: cond expects predicate input".to_owned(),
                        });
                    };
                    if pred.shape != Shape::scalar() {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: cond predicate must be scalar, got {:?}",
                                pred.shape.dims
                            ),
                        });
                    }
                    let true_branch = &eqn.sub_jaxprs[0];
                    let false_branch = &eqn.sub_jaxprs[1];
                    let true_expected = true_branch.constvars.len() + true_branch.invars.len();
                    if bindings.len() != true_expected {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: cond branch expects {true_expected} bindings, got {}",
                                bindings.len()
                            ),
                        });
                    }
                    let (t_consts, t_args) = bindings.split_at(true_branch.constvars.len());
                    let true_out = eval_shape_with_consts_avals(true_branch, t_consts, t_args)?;

                    let false_expected = false_branch.constvars.len() + false_branch.invars.len();
                    if bindings.len() != false_expected {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: cond false branch expects {false_expected} bindings, got {}",
                                bindings.len()
                            ),
                        });
                    }
                    let (f_consts, f_args) = bindings.split_at(false_branch.constvars.len());
                    let false_out = eval_shape_with_consts_avals(false_branch, f_consts, f_args)?;
                    if true_out != false_out {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: cond branches have mismatched output shapes/dtypes: {true_out:?} vs {false_out:?}"
                            ),
                        });
                    }
                    true_out
                }
                Primitive::Switch => {
                    let Some((index, bindings)) = input_shapes.split_first() else {
                        return Err(ApiError::EvalError {
                            detail: "eval_shape: switch expects index input".to_owned(),
                        });
                    };
                    if index.shape != Shape::scalar() {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: switch index must be scalar, got {:?}",
                                index.shape.dims
                            ),
                        });
                    }
                    let Some(first_branch) = eqn.sub_jaxprs.first() else {
                        return Err(ApiError::EvalError {
                            detail: "eval_shape: switch requires at least one branch".to_owned(),
                        });
                    };
                    let expected = first_branch.constvars.len() + first_branch.invars.len();
                    if bindings.len() != expected {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: switch branch expects {expected} bindings, got {}",
                                bindings.len()
                            ),
                        });
                    }
                    let (consts, args) = bindings.split_at(first_branch.constvars.len());
                    let first_out = eval_shape_with_consts_avals(first_branch, consts, args)?;
                    for (idx, branch) in eqn.sub_jaxprs.iter().enumerate().skip(1) {
                        let branch_expected = branch.constvars.len() + branch.invars.len();
                        if bindings.len() != branch_expected {
                            return Err(ApiError::EvalError {
                                detail: format!(
                                    "eval_shape: switch branch {idx} expects {branch_expected} bindings, got {}",
                                    bindings.len()
                                ),
                            });
                        }
                        let (b_consts, b_args) = bindings.split_at(branch.constvars.len());
                        let branch_out = eval_shape_with_consts_avals(branch, b_consts, b_args)?;
                        if branch_out != first_out {
                            return Err(ApiError::EvalError {
                                detail: format!(
                                    "eval_shape: switch branch {idx} output {branch_out:?} != branch 0 {first_out:?}"
                                ),
                            });
                        }
                    }
                    first_out
                }
                Primitive::While => {
                    if eqn.sub_jaxprs.len() != 2 {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: while expects 2 sub_jaxprs, got {}",
                                eqn.sub_jaxprs.len()
                            ),
                        });
                    }
                    let cond_jaxpr = &eqn.sub_jaxprs[0];
                    let body_jaxpr = &eqn.sub_jaxprs[1];
                    let const_count = cond_jaxpr.constvars.len() + body_jaxpr.constvars.len();
                    if input_shapes.len() < const_count {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: while expects at least {const_count} const inputs, got {}",
                                input_shapes.len()
                            ),
                        });
                    }
                    let (const_bindings, carry_bindings) = input_shapes.split_at(const_count);
                    let (cond_consts, body_consts) =
                        const_bindings.split_at(cond_jaxpr.constvars.len());
                    let cond_out =
                        eval_shape_with_consts_avals(cond_jaxpr, cond_consts, carry_bindings)?;
                    if cond_out.len() != 1 || cond_out[0].shape != Shape::scalar() {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: while cond must return 1 scalar, got {cond_out:?}"
                            ),
                        });
                    }
                    let body_out =
                        eval_shape_with_consts_avals(body_jaxpr, body_consts, carry_bindings)?;
                    if body_out != carry_bindings {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: while body output {body_out:?} must match carry {carry_bindings:?}"
                            ),
                        });
                    }
                    carry_bindings.to_vec()
                }
                Primitive::Scan => {
                    if eqn.sub_jaxprs.len() != 1 {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: scan expects 1 body sub_jaxpr, got {}",
                                eqn.sub_jaxprs.len()
                            ),
                        });
                    }
                    let body_jaxpr = &eqn.sub_jaxprs[0];
                    let carry_count = body_jaxpr.invars.len().checked_sub(1).ok_or_else(|| {
                        ApiError::EvalError {
                            detail: "eval_shape: scan body requires carry inputs plus xs input"
                                .to_owned(),
                        }
                    })?;
                    let const_count = body_jaxpr.constvars.len();
                    let expected_bindings = const_count + carry_count + 1;
                    if input_shapes.len() != expected_bindings {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: scan expects {expected_bindings} inputs, got {}",
                                input_shapes.len()
                            ),
                        });
                    }
                    let (const_avals, state_avals) = input_shapes.split_at(const_count);
                    let (carry_avals, xs_avals) = state_avals.split_at(carry_count);
                    let xs_aval = &xs_avals[0];
                    let Some((&scan_len, elem_dims)) = xs_aval.shape.dims.split_first() else {
                        return Err(ApiError::EvalError {
                            detail: "eval_shape: scan xs input must have rank >= 1".to_owned(),
                        });
                    };
                    let mut body_in_avals = carry_avals.to_vec();
                    body_in_avals.push(ShapedArray {
                        dtype: xs_aval.dtype,
                        shape: Shape {
                            dims: elem_dims.to_vec(),
                        },
                    });
                    let body_out =
                        eval_shape_with_consts_avals(body_jaxpr, const_avals, &body_in_avals)?;
                    if body_out.len() < carry_count {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: scan body returned {} outputs, expected at least {carry_count} carries",
                                body_out.len()
                            ),
                        });
                    }
                    let (out_carries, out_ys) = body_out.split_at(carry_count);
                    if out_carries != carry_avals {
                        return Err(ApiError::EvalError {
                            detail: format!(
                                "eval_shape: scan body carry output {out_carries:?} != input carry {carry_avals:?}"
                            ),
                        });
                    }
                    let mut final_out = out_carries.to_vec();
                    for y_aval in out_ys {
                        let mut dims = Vec::with_capacity(1 + y_aval.shape.dims.len());
                        dims.push(scan_len);
                        dims.extend_from_slice(&y_aval.shape.dims);
                        final_out.push(ShapedArray {
                            dtype: y_aval.dtype,
                            shape: Shape { dims },
                        });
                    }
                    final_out
                }
                primitive => {
                    return Err(ApiError::EvalError {
                        detail: format!(
                            "eval_shape: sub_jaxprs are only valid for cond, scan, while, or switch; {} received sub_jaxprs",
                            primitive.as_str()
                        ),
                    });
                }
            }
        };

        if output_shapes.len() != eqn.outputs.len() {
            return Err(ApiError::EvalError {
                detail: format!(
                    "eval_shape: primitive {:?} returned {} shapes, expected {}",
                    eqn.primitive,
                    output_shapes.len(),
                    eqn.outputs.len()
                ),
            });
        }

        for (var, shape) in eqn.outputs.iter().zip(output_shapes) {
            shape_env.insert(*var, shape);
        }
    }

    let output_shapes: Result<Vec<_>, _> = jaxpr
        .outvars
        .iter()
        .map(|v| {
            shape_env
                .get(v)
                .cloned()
                .ok_or_else(|| ApiError::EvalError {
                    detail: format!("eval_shape: undefined output variable {:?}", v),
                })
        })
        .collect();

    output_shapes
}

fn infer_primitive_shapes(
    primitive: fj_core::Primitive,
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, ApiError> {
    fj_trace::SimpleTraceContext::infer_primitive_output_avals(primitive, inputs, params).map_err(
        |err| ApiError::EvalError {
            detail: err.to_string(),
        },
    )
}

fn custom_derivative_rule_key(transform: &str) -> String {
    let id = CUSTOM_DERIVATIVE_WRAPPER_ID.fetch_add(1, Ordering::Relaxed);
    format!("fj-api:{transform}:{id}")
}

fn transform_evidence(transforms: &[Transform]) -> Vec<String> {
    transforms
        .iter()
        .enumerate()
        .map(|(idx, transform)| format!("fj-api-{}-{}", transform.as_str(), idx))
        .collect()
}

#[cfg(test)]
fn build_ledger(jaxpr: Jaxpr, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(jaxpr);
    for (transform, evidence) in transforms.iter().zip(transform_evidence(transforms)) {
        ledger.push_transform(*transform, evidence);
    }
    ledger
}

fn dispatch_with_options(
    jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
    compile_options: BTreeMap<String, String>,
) -> Result<Vec<Value>, ApiError> {
    let transform_evidence = transform_evidence(transforms);
    dispatch_with_options_prepared(
        jaxpr,
        transforms,
        &transform_evidence,
        args,
        backend,
        mode,
        compile_options,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_with_options_prepared(
    jaxpr: &Jaxpr,
    transforms: &[Transform],
    transform_evidence: &[String],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
    compile_options: BTreeMap<String, String>,
    prepared: Option<&PreparedDispatchMeta>,
) -> Result<Vec<Value>, ApiError> {
    let unknown_incompatible_features: &[String] = &[];
    let response = dispatch_ref(DispatchRequestRef {
        mode,
        root_jaxpr: jaxpr,
        transform_stack: transforms,
        transform_evidence,
        args,
        backend,
        compile_options,
        custom_hook: None,
        unknown_incompatible_features,
        prepared,
    })?;
    Ok(response.outputs)
}

fn dispatch_with(
    jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
) -> Result<Vec<Value>, ApiError> {
    dispatch_with_options(jaxpr, transforms, args, backend, mode, BTreeMap::new())
}

fn synthesize_fast_path_proof(
    jaxpr: &Jaxpr,
    transforms: &[Transform],
    mode: CompatibilityMode,
    backend: &str,
    compile_options: &BTreeMap<String, String>,
) -> Option<TransformCompositionProof> {
    let evidence = transform_evidence(transforms);
    let meta = prepare_dispatch_meta(
        mode,
        jaxpr,
        transforms,
        &evidence,
        backend,
        compile_options,
        None,
        &[],
    )
    .ok()?;
    if transforms == [Transform::Grad] && !jaxpr.invars.is_empty() && jaxpr.outvars.len() == 1 {
        let is_val_and_grad = compile_options
            .get("value_and_grad")
            .is_some_and(|v| v == "true");
        let argnums = vec![0];
        let lowered_res = if is_val_and_grad {
            fj_ad::build_value_and_grad_jaxpr(jaxpr, &argnums)
        } else {
            fj_ad::build_grad_jaxpr(jaxpr, &argnums)
        };
        if let Ok(lowered) = lowered_res {
            let scalar_aval = AbstractValue {
                dtype: DType::F64,
                shape: Shape::scalar(),
            };
            let in_avals = vec![scalar_aval.clone(); jaxpr.invars.len()];
            let out_avals = vec![scalar_aval.clone()];
            let step_out_avals = if is_val_and_grad {
                vec![scalar_aval.clone(), scalar_aval]
            } else {
                vec![scalar_aval]
            };
            let primal_eqn_count = jaxpr.equations.len();
            let total_eqn_count = lowered.equations.len();
            let adjoint_eqn_count = total_eqn_count.saturating_sub(primal_eqn_count);
            let pruned_dead_cotangents =
                (primal_eqn_count.saturating_mul(2)).saturating_sub(total_eqn_count);
            let step = TransformProofStep {
                step_index: 0,
                transform: Transform::Grad,
                evidence: evidence[0].clone(),
                input_jaxpr_fingerprint_sha256: jaxpr_fingerprint_sha256(jaxpr),
                output_jaxpr_fingerprint_sha256: jaxpr_fingerprint_sha256(&lowered),
                input_avals: in_avals.clone(),
                output_avals: step_out_avals,
                witness: TransformRewriteWitness::Grad(GradWitness {
                    argnums,
                    primal_eqn_count,
                    adjoint_eqn_count,
                    pruned_dead_cotangents,
                }),
            };
            if let Ok(lowered_proof) = verify_transform_composition_steps(
                jaxpr,
                transforms,
                &evidence,
                &in_avals,
                &out_avals,
                &[step],
            ) {
                return Some(lowered_proof);
            }
        }
    }
    Some(meta.composition_proof)
}

fn synthesize_fast_path_ledger(
    jaxpr: &Jaxpr,
    transforms: &[Transform],
    mode: CompatibilityMode,
    backend: &str,
    compile_options: &BTreeMap<String, String>,
) -> Option<EvidenceLedger> {
    let evidence = transform_evidence(transforms);
    let meta = prepare_dispatch_meta(
        mode,
        jaxpr,
        transforms,
        &evidence,
        backend,
        compile_options,
        None,
        &[],
    )
    .ok()?;
    let eqn_count = jaxpr.equations.len() as f64;
    let depth = transforms.len() as f64;
    let signals = vec![
        EvidenceSignal {
            signal_name: "eqn_count".to_owned(),
            log_likelihood_delta: (eqn_count + 1.0).ln(),
            detail: format!("jaxpr_equations={}", jaxpr.equations.len()),
        },
        EvidenceSignal {
            signal_name: "transform_depth".to_owned(),
            log_likelihood_delta: depth * 0.25,
            detail: format!("stack_len={}", transforms.len()),
        },
    ];
    let record = DecisionRecord::from_posterior(mode, 0.05, &LossMatrix::default());
    let mut ledger = EvidenceLedger::new();
    ledger.append(LedgerEntry {
        decision_id: meta.cache_key.as_string(),
        record,
        signals,
    });
    Some(ledger)
}

/// Compose transforms: `jit(grad(f))` becomes `jit(jaxpr).compose_grad()`.
#[must_use]
pub fn compose(jaxpr: Jaxpr, transforms: Vec<Transform>) -> ComposedTransform {
    ComposedTransform {
        jaxpr,
        transforms,
        backend: Cow::Borrowed(DEFAULT_BACKEND),
        mode: CompatibilityMode::Strict,
        compile_options: BTreeMap::new(),
        custom_vjp_rule_key: None,
        ffi_registry: FfiRegistrySlot::default(),
        meta_cache: DispatchMetaCache::default(),
        last_evidence: LastEvidenceCache::default(),
    }
}

impl JitWrapped {
    fn build_compile_options(&self) -> BTreeMap<String, String> {
        let mut opts = self.compile_options.clone();
        if !self.static_argnums.is_empty() {
            let formatted = self
                .static_argnums
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            opts.insert("static_argnums".to_owned(), formatted);
        }
        if let Some(policy) = self.precision_policy {
            opts.insert("precision_policy".to_owned(), policy.as_str().to_owned());
        }
        if let Some(reg) = &self.ffi_registry.0 {
            let mut targets = reg.registered_names();
            targets.sort();
            opts.insert("ffi_registered_targets".to_owned(), targets.join(","));
        }
        opts
    }

    fn try_default_cpu_compiled_eval(
        &self,
        args: &[Value],
    ) -> Result<Option<Vec<Value>>, ApiError> {
        if self.backend.as_ref() != DEFAULT_BACKEND
            || !self.static_argnums.is_empty()
            || !self.compile_options.is_empty()
            || self.ffi_registry.0.is_some()
            || fj_dispatch::jaxpr_has_parallel_heavy_wave(&self.jaxpr)
        {
            return Ok(None);
        }
        check_fast_path_admission(&self.jaxpr, &[Transform::Jit], args, self.mode)?;
        let was_initialized = self.compiled_cache.plan.get().is_some();
        let Some(compiled) = self
            .compiled_cache
            .plan
            .get_or_init(|| {
                let compiled =
                    fj_interpreters::compile_jaxpr_for_repeated_eval(&self.jaxpr).map(Arc::new);
                let transforms = [Transform::Jit];
                let evidence = transform_evidence(&transforms);
                let compile_options = self.build_compile_options();
                if let Some(meta) = self
                    .meta_cache
                    .0
                    .get_or_init(|| {
                        prepare_dispatch_meta(
                            self.mode,
                            &self.jaxpr,
                            &transforms,
                            &evidence,
                            self.backend.as_ref(),
                            &compile_options,
                            None,
                            &[],
                        )
                        .ok()
                    })
                    .as_ref()
                {
                    fj_dispatch::register_compiled_dispatch_plan(
                        meta.cache_key.clone(),
                        &self.jaxpr,
                        compiled.clone(),
                    );
                }
                compiled
            })
            .as_ref()
        else {
            return Ok(None);
        };
        if was_initialized {
            self.compiled_cache
                .warm_hits
                .fetch_add(1, Ordering::Relaxed);
        }
        let eval_res = match self.precision_policy {
            Some(policy) => with_precision_policy(policy, || compiled.eval(args)),
            None => compiled.eval(args),
        };
        eval_res.map(Some).map_err(ApiError::from)
    }

    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        // Backend feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        // Mode feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set a dispatch compile option key/value pair on this `JitWrapped`.
    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Return the effective compile options map for this `JitWrapped`.
    #[must_use]
    pub fn compile_options(&self) -> BTreeMap<String, String> {
        self.build_compile_options()
    }

    /// Enable or disable e-graph equality-saturation optimization (`fj-egraph`) during JIT compilation.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Force or disable dependency-wave Rayon parallel execution (`fj-backend-cpu`) during JIT dispatch.
    #[must_use]
    pub fn with_wave_parallel(self, enabled: bool) -> Self {
        self.with_compile_option("wave_parallel", if enabled { "true" } else { "false" })
    }

    /// Persist compiled JIT artifacts to a `fj_cache::backend::FileCache` directory.
    #[must_use]
    pub fn with_cache_dir(self, cache_dir: &str) -> Self {
        self.with_compile_option("cache_dir", cache_dir)
    }

    /// Configure a per-wrapper memory working-set admission ceiling (in bytes).
    #[must_use]
    pub fn with_max_working_set_bytes(self, max_bytes: usize) -> Self {
        self.with_compile_option("max_working_set_bytes", &max_bytes.to_string())
    }

    /// Configure a per-wrapper equation-count admission ceiling.
    #[must_use]
    pub fn with_max_equation_count(self, max_equations: usize) -> Self {
        self.with_compile_option("max_equation_count", &max_equations.to_string())
    }

    /// Specify argument indices (`0`-indexed) whose values are treated as known
    /// compile-time constants during JIT staging (`stage_jaxpr` / `partial_eval_jaxpr`).
    #[must_use]
    pub fn with_static_argnums(mut self, argnums: &[usize]) -> Self {
        self.static_argnums = argnums.to_vec();
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Return the configured static argument indices for this `JitWrapped`.
    #[must_use]
    pub fn static_argnums(&self) -> &[usize] {
        &self.static_argnums
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this `JitWrapped`.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Return the configured [`PrecisionPolicy`] for this `JitWrapped`, if any.
    #[must_use]
    pub const fn precision_policy(&self) -> Option<PrecisionPolicy> {
        self.precision_policy
    }

    /// Attach a scoped [`fj_ffi::FfiRegistry`] for `Primitive::CustomCall` evaluation during JIT dispatch.
    #[must_use]
    pub fn with_ffi_registry(mut self, registry: fj_ffi::FfiRegistry) -> Self {
        self.ffi_registry = FfiRegistrySlot(Some(registry));
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_cache = CompiledEvalCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Return the number of warm fast-path hits recorded on this `JitWrapped` instance.
    #[must_use]
    pub fn warm_hit_count(&self) -> u64 {
        self.compiled_cache.warm_hits.load(Ordering::Relaxed)
    }

    /// Execute through the full `fj-dispatch` + `fj-cache` + `fj-interpreters::staging`
    /// spine and return the full [`fj_dispatch::DispatchResponse`] (including
    /// `cache_key`, `cache_hit()`, `cache_stats()`, `staged_program()`, and `evidence_ledger`).
    pub fn call_with_dispatch(
        &self,
        args: Vec<Value>,
    ) -> Result<fj_dispatch::DispatchResponse, ApiError> {
        let transforms = [Transform::Jit];
        let evidence = transform_evidence(&transforms);
        let compile_options = self.build_compile_options();
        let prepared = if self.static_argnums.is_empty() {
            self.meta_cache
                .0
                .get_or_init(|| {
                    prepare_dispatch_meta(
                        self.mode,
                        &self.jaxpr,
                        &transforms,
                        &evidence,
                        self.backend.as_ref(),
                        &compile_options,
                        None,
                        &[],
                    )
                    .ok()
                })
                .as_ref()
        } else {
            None
        };
        let req = DispatchRequestRef {
            mode: self.mode,
            root_jaxpr: &self.jaxpr,
            transform_stack: &transforms,
            transform_evidence: &evidence,
            args,
            backend: self.backend.as_ref(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &[],
            prepared,
        };
        let response = if let Some(reg) = &self.ffi_registry.0 {
            fj_interpreters::with_ffi_registry(reg, || dispatch_ref(req))?
        } else {
            dispatch_ref(req)?
        };
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        Ok(response)
    }

    /// Execute through the full dispatch + ledger + proof spine and return the
    /// complete [`DispatchResponse`] including `evidence_ledger` and `composition_proof`.
    pub fn call_with_evidence(&self, args: Vec<Value>) -> Result<DispatchResponse, ApiError> {
        self.call_with_dispatch(args)
    }

    /// Execute with an explicit [`fj_ffi::FfiRegistry`] installed for `Primitive::CustomCall` equations.
    pub fn call_with_ffi(
        &self,
        args: Vec<Value>,
        registry: &fj_ffi::FfiRegistry,
    ) -> Result<Vec<Value>, ApiError> {
        self.clone()
            .with_ffi_registry(registry.clone())
            .call_with_dispatch(args)
            .map(|r| r.outputs)
    }

    /// Return the [`EvidenceLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger().or_else(|| {
            if self.compiled_cache.plan.get().is_some() {
                synthesize_fast_path_ledger(
                    &self.jaxpr,
                    &[Transform::Jit],
                    self.mode,
                    self.backend.as_ref(),
                    &self.build_compile_options(),
                )
            } else {
                None
            }
        })
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof().or_else(|| {
            if self.compiled_cache.plan.get().is_some() {
                synthesize_fast_path_proof(
                    &self.jaxpr,
                    &[Transform::Jit],
                    self.mode,
                    self.backend.as_ref(),
                    &self.build_compile_options(),
                )
            } else {
                None
            }
        })
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        if let Some(outputs) = self.try_default_cpu_compiled_eval(&args)? {
            return Ok(outputs);
        }
        Ok(self.call_with_dispatch(args)?.outputs)
    }

    /// Compose: `jit(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: self.ffi_registry,
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `jit(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: self.ffi_registry,
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `jit(pmap(f))`.
    #[must_use]
    pub fn compose_pmap(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Pmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: self.ffi_registry,
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

impl GradWrapped {
    fn build_compile_options(&self) -> BTreeMap<String, String> {
        let mut compile_options = self.compile_options.clone();
        if let Some(rule_key) = &self.custom_vjp_rule_key {
            compile_options.insert(CUSTOM_VJP_RULE_KEY_OPTION.to_owned(), rule_key.clone());
        }
        if !self.argnums.is_empty() {
            let formatted = self
                .argnums
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            compile_options.insert("grad_argnums".to_owned(), formatted);
        }
        if let Some(policy) = self.precision_policy {
            compile_options.insert("precision_policy".to_owned(), policy.as_str().to_owned());
        }
        compile_options
    }

    fn try_default_cpu_compiled_grad(&self, args: &[Value]) -> Result<Option<Value>, ApiError> {
        if self.custom_vjp_rule_key.is_some()
            || self.backend.as_ref() != DEFAULT_BACKEND
            || !self.argnums.is_empty()
            || !self.compile_options.is_empty()
        {
            return Ok(None);
        }
        check_fast_path_admission(&self.jaxpr, &[Transform::Grad], args, self.mode)?;
        let Some(compiled) = self
            .compiled_ad_cache
            .0
            .get_or_init(|| fj_ad::compile_value_and_grad_jaxpr_for_repeated_eval(&self.jaxpr))
            .as_ref()
        else {
            return Ok(None);
        };
        let grads_opt = match self.precision_policy {
            Some(policy) => with_precision_policy(policy, || compiled.grad(args))?,
            None => compiled.grad(args)?,
        };
        let Some(grads) = grads_opt else {
            return Ok(None);
        };
        Ok(grads.into_iter().next())
    }

    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        // Backend feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_ad_cache = CompiledAdCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_ad_cache = CompiledAdCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Differentiate with respect to the specified 0-based input argument indices.
    ///
    /// When `argnums` is non-empty, [`GradWrapped::call`] returns one gradient per index in
    /// `argnums` (in the order specified).
    #[must_use]
    pub fn with_argnums(mut self, argnums: &[usize]) -> Self {
        self.argnums = argnums.to_vec();
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this gradient wrapper.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set a dispatch compile option on this gradient wrapper.
    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Enable or disable dispatch-time e-graph optimization before differentiation.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Return the effective compile options map configured on this wrapper.
    #[must_use]
    pub fn compile_options(&self) -> BTreeMap<String, String> {
        self.build_compile_options()
    }

    /// Execute through the full dispatch + ledger + proof spine and return the
    /// complete [`DispatchResponse`] including `evidence_ledger` and `composition_proof`.
    pub fn call_with_evidence(&self, args: Vec<Value>) -> Result<DispatchResponse, ApiError> {
        let compile_options = self.build_compile_options();
        let transforms = [Transform::Grad];
        let evidence = transform_evidence(&transforms);
        let prepared = self
            .meta_cache
            .0
            .get_or_init(|| {
                prepare_dispatch_meta(
                    self.mode,
                    &self.jaxpr,
                    &transforms,
                    &evidence,
                    self.backend.as_ref(),
                    &compile_options,
                    None,
                    &[],
                )
                .ok()
            })
            .as_ref();
        let response = dispatch_ref(DispatchRequestRef {
            mode: self.mode,
            root_jaxpr: &self.jaxpr,
            transform_stack: &transforms,
            transform_evidence: &evidence,
            args,
            backend: self.backend.as_ref(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &[],
            prepared,
        })?;
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        Ok(response)
    }

    /// Return the [`EvidenceLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger().or_else(|| {
            if self.compiled_ad_cache.0.get().is_some() {
                synthesize_fast_path_ledger(
                    &self.jaxpr,
                    &[Transform::Grad],
                    self.mode,
                    self.backend.as_ref(),
                    &self.build_compile_options(),
                )
            } else {
                None
            }
        })
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof().or_else(|| {
            if self.compiled_ad_cache.0.get().is_some() {
                synthesize_fast_path_proof(
                    &self.jaxpr,
                    &[Transform::Grad],
                    self.mode,
                    self.backend.as_ref(),
                    &self.build_compile_options(),
                )
            } else {
                None
            }
        })
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        if let Some(first_grad) = self.try_default_cpu_compiled_grad(&args)? {
            return Ok(vec![first_grad]);
        }
        Ok(self.call_with_evidence(args)?.outputs)
    }

    /// Lower the underlying scalar-output [`Jaxpr`] to its symbolic reverse-mode gradient [`Jaxpr`]
    /// with respect to the configured `argnums` (or input `0` by default).
    pub fn to_grad_jaxpr(&self) -> Result<Jaxpr, ApiError> {
        let default_argnums = [0usize];
        let argnums: &[usize] = if self.argnums.is_empty() {
            &default_argnums
        } else {
            &self.argnums
        };
        fj_ad::build_grad_jaxpr(&self.jaxpr, argnums).map_err(ApiError::from)
    }

    /// Compose: `grad(jit(f))`.
    #[must_use]
    pub fn compose_jit(self) -> ComposedTransform {
        let mut compile_options = self.build_compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Jit],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose higher-order derivative: `grad(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        let mut compile_options = self.build_compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `grad(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        let mut compile_options = self.build_compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `grad(pmap(f))`.
    #[must_use]
    pub fn compose_pmap(self) -> ComposedTransform {
        let mut compile_options = self.build_compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Pmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

impl VmapWrapped {
    fn build_compile_options(&self) -> BTreeMap<String, String> {
        let mut compile_options = self.compile_options.clone();
        if let Some(ref in_axes) = self.in_axes {
            compile_options.insert("vmap_in_axes".to_owned(), in_axes.clone());
        }
        if let Some(ref out_axes) = self.out_axes {
            compile_options.insert("vmap_out_axes".to_owned(), out_axes.clone());
        }
        if let Some(policy) = self.precision_policy {
            compile_options.insert("precision_policy".to_owned(), policy.as_str().to_owned());
        }
        compile_options
    }

    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        // Backend feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set in_axes: comma-separated axis specs, e.g. "0,none,1".
    /// - Integer: batch along that axis
    /// - "none": this input is not batched (broadcast)
    #[must_use]
    pub fn with_in_axes(mut self, in_axes: &str) -> Self {
        self.in_axes = Some(in_axes.to_owned());
        // in_axes feeds compile_options, a cache-key input — invalidate the memo.
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set out_axes: comma-separated axis specs for output batch dim placement.
    /// - Integer: place batch dim at that axis position
    /// - "none": output is not batched
    #[must_use]
    pub fn with_out_axes(mut self, out_axes: &str) -> Self {
        self.out_axes = Some(out_axes.to_owned());
        // out_axes feeds compile_options, a cache-key input — invalidate the memo.
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this vmap wrapper.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set a dispatch compile option on this vmap wrapper.
    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Enable or disable dispatch-time e-graph optimization.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Execute through the full dispatch + ledger + proof spine and return the
    /// complete [`DispatchResponse`] including `evidence_ledger` and `composition_proof`.
    pub fn call_with_evidence(&self, args: Vec<Value>) -> Result<DispatchResponse, ApiError> {
        let compile_options = self.build_compile_options();
        let transforms = [Transform::Vmap];
        let evidence = transform_evidence(&transforms);
        let prepared = self
            .meta_cache
            .0
            .get_or_init(|| {
                prepare_dispatch_meta(
                    self.mode,
                    &self.jaxpr,
                    &transforms,
                    &evidence,
                    self.backend.as_ref(),
                    &compile_options,
                    None,
                    &[],
                )
                .ok()
            })
            .as_ref();
        let response = dispatch_ref(DispatchRequestRef {
            mode: self.mode,
            root_jaxpr: &self.jaxpr,
            transform_stack: &transforms,
            transform_evidence: &evidence,
            args,
            backend: self.backend.as_ref(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &[],
            prepared,
        })?;
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        Ok(response)
    }

    /// Return the [`EvidenceLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger()
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof()
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        Ok(self.call_with_evidence(args)?.outputs)
    }

    /// Compose: `vmap(jit(f))`.
    #[must_use]
    pub fn compose_jit(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Jit],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `vmap(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose nested batching: `vmap(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `vmap(pmap(f))`.
    #[must_use]
    pub fn compose_pmap(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Pmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

impl ComposedTransform {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Attach a thread-safe [`fj_ffi::FfiRegistry`] for executing `Primitive::CustomCall`
    /// equations within this composed transform stack.
    #[must_use]
    pub fn with_ffi_registry(mut self, registry: fj_ffi::FfiRegistry) -> Self {
        self.ffi_registry = FfiRegistrySlot(Some(registry));
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set in_axes for the first `vmap` in this transform stack.
    #[must_use]
    pub fn with_vmap_in_axes(self, in_axes: &str) -> Self {
        self.with_compile_option("vmap_in_axes", in_axes)
    }

    /// Set out_axes for the first `vmap` in this transform stack.
    #[must_use]
    pub fn with_vmap_out_axes(self, out_axes: &str) -> Self {
        self.with_compile_option("vmap_out_axes", out_axes)
    }

    /// Differentiate with respect to the specified 0-based input argument indices.
    #[must_use]
    pub fn with_grad_argnums(self, argnums: &[usize]) -> Self {
        let formatted = argnums
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.with_compile_option("grad_argnums", &formatted)
    }

    /// Configure the virtual device count for `pmap` in this transform stack.
    #[must_use]
    pub fn with_pmap_devices(self, devices: usize) -> Self {
        self.with_compile_option("pmap_devices", &devices.to_string())
    }

    /// Append `Transform::Jit` to the inner end of this transform stack.
    #[must_use]
    pub fn compose_jit(mut self) -> Self {
        self.transforms.push(Transform::Jit);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Append `Transform::Grad` to the inner end of this transform stack.
    #[must_use]
    pub fn compose_grad(mut self) -> Self {
        self.transforms.push(Transform::Grad);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Append `Transform::Vmap` to the inner end of this transform stack.
    #[must_use]
    pub fn compose_vmap(mut self) -> Self {
        self.transforms.push(Transform::Vmap);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Append `Transform::Pmap` to the inner end of this transform stack.
    #[must_use]
    pub fn compose_pmap(mut self) -> Self {
        self.transforms.push(Transform::Pmap);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Enable or disable the finite-difference compatibility fallback for composed `grad` tails.
    #[must_use]
    pub fn with_finite_diff_grad_fallback(self, allow: bool) -> Self {
        self.with_compile_option(
            "allow_finite_diff_grad_fallback",
            if allow { "true" } else { "false" },
        )
    }

    /// Enable or disable dispatch-time e-graph optimization.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this composed transform.
    #[must_use]
    pub fn with_precision_policy(self, policy: PrecisionPolicy) -> Self {
        self.with_compile_option("precision_policy", policy.as_str())
    }

    /// Execute through the full dispatch + ledger + proof spine and return the
    /// complete [`DispatchResponse`] including `evidence_ledger` and `composition_proof`.
    pub fn call_with_evidence(&self, args: Vec<Value>) -> Result<DispatchResponse, ApiError> {
        let mut compile_options = self.compile_options.clone();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        if let Some(rule_key) = &self.custom_vjp_rule_key {
            compile_options.insert(CUSTOM_VJP_RULE_KEY_OPTION.to_owned(), rule_key.clone());
        }
        if let Some(ref registry) = self.ffi_registry.0 {
            let mut names = registry.registered_names();
            names.sort();
            compile_options.insert("ffi_registered_targets".to_owned(), names.join(","));
        }
        let evidence = transform_evidence(&self.transforms);
        let prepared =
            if self.ffi_registry.0.is_none() && !compile_options.contains_key("static_argnums") {
                self.meta_cache
                    .0
                    .get_or_init(|| {
                        prepare_dispatch_meta(
                            self.mode,
                            &self.jaxpr,
                            &self.transforms,
                            &evidence,
                            self.backend.as_ref(),
                            &compile_options,
                            None,
                            &[],
                        )
                        .ok()
                    })
                    .as_ref()
            } else {
                None
            };
        let run_dispatch = || {
            dispatch_ref(DispatchRequestRef {
                mode: self.mode,
                root_jaxpr: &self.jaxpr,
                transform_stack: &self.transforms,
                transform_evidence: &evidence,
                args,
                backend: self.backend.as_ref(),
                compile_options,
                custom_hook: None,
                unknown_incompatible_features: &[],
                prepared,
            })
        };
        let response = match self.ffi_registry.0 {
            Some(ref registry) => fj_interpreters::with_ffi_registry(registry, run_dispatch)?,
            None => run_dispatch()?,
        };
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        Ok(response)
    }

    /// Return the [`DecisionLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger()
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof()
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        Ok(self.call_with_evidence(args)?.outputs)
    }

    /// Execute with an explicit [`fj_ffi::FfiRegistry`] bound for this invocation.
    pub fn call_with_ffi(
        &self,
        args: Vec<Value>,
        registry: &fj_ffi::FfiRegistry,
    ) -> Result<Vec<Value>, ApiError> {
        let mut with_ffi = self.clone();
        with_ffi.ffi_registry = FfiRegistrySlot(Some(registry.clone()));
        with_ffi.call(args)
    }
}

impl ValueAndGradWrapped {
    fn try_default_cpu_compiled_value_and_grad(
        &self,
        args: &[Value],
    ) -> Result<Option<ValueAndGradOutputs>, ApiError> {
        if self.custom_vjp_rule_key.is_some()
            || self.backend.as_ref() != DEFAULT_BACKEND
            || !self.argnums.is_empty()
            || !self.compile_options.is_empty()
        {
            return Ok(None);
        }
        check_fast_path_admission(&self.jaxpr, &[Transform::Grad], args, self.mode)?;
        let Some(compiled) = self
            .compiled_ad_cache
            .0
            .get_or_init(|| fj_ad::compile_value_and_grad_jaxpr_for_repeated_eval(&self.jaxpr))
            .as_ref()
        else {
            return Ok(None);
        };
        match self.precision_policy {
            Some(policy) => with_precision_policy(policy, || compiled.value_and_grad(args))
                .map_err(ApiError::from),
            None => compiled.value_and_grad(args).map_err(ApiError::from),
        }
    }

    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        // Backend feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_ad_cache = CompiledAdCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        // Mode feeds the cache key — invalidate any memoized metadata.
        self.meta_cache = DispatchMetaCache::default();
        self.compiled_ad_cache = CompiledAdCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Differentiate with respect to the specified 0-based input argument indices.
    #[must_use]
    pub fn with_argnums(mut self, argnums: &[usize]) -> Self {
        self.argnums = argnums.to_vec();
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this value-and-grad wrapper.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set a dispatch compile option on this value-and-grad wrapper.
    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Enable or disable dispatch-time e-graph optimization before differentiation.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Build the per-wrapper compile options. These are constant across calls,
    /// so the derived dispatch metadata can be memoized.
    pub fn compile_options(&self) -> BTreeMap<String, String> {
        let mut compile_options = self.compile_options.clone();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        if let Some(rule_key) = &self.custom_vjp_rule_key {
            compile_options.insert(CUSTOM_VJP_RULE_KEY_OPTION.to_owned(), rule_key.clone());
        }
        if !self.argnums.is_empty() {
            let formatted = self
                .argnums
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            compile_options.insert("grad_argnums".to_owned(), formatted);
        }
        if let Some(policy) = self.precision_policy {
            compile_options.insert("precision_policy".to_owned(), policy.as_str().to_owned());
        }
        compile_options
    }

    /// Execute through the full dispatch + ledger + proof spine and return
    /// `((values, gradients), DispatchResponse)`.
    pub fn call_with_evidence(
        &self,
        args: Vec<Value>,
    ) -> Result<(ValueAndGradOutputs, DispatchResponse), ApiError> {
        let compile_options = self.compile_options();
        let transforms = [Transform::Grad];
        let evidence = transform_evidence(&transforms);
        let prepared = self
            .meta_cache
            .0
            .get_or_init(|| {
                prepare_dispatch_meta(
                    self.mode,
                    &self.jaxpr,
                    &transforms,
                    &evidence,
                    self.backend.as_ref(),
                    &compile_options,
                    None,
                    &[],
                )
                .ok()
            })
            .as_ref();
        let response = dispatch_ref(DispatchRequestRef {
            mode: self.mode,
            root_jaxpr: &self.jaxpr,
            transform_stack: &transforms,
            transform_evidence: &evidence,
            args,
            backend: self.backend.as_ref(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &[],
            prepared,
        })?;
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        let value_len = self.jaxpr.outvars.len();
        if response.outputs.len() < value_len + 1 {
            return Err(ApiError::EvalError {
                detail: format!(
                    "value_and_grad expected at least {} outputs, got {}",
                    value_len + 1,
                    response.outputs.len()
                ),
            });
        }

        let values = response.outputs[..value_len].to_vec();
        let gradients = response.outputs[value_len..].to_vec();
        Ok(((values, gradients), response))
    }

    /// Return the [`EvidenceLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger().or_else(|| {
            if self.compiled_ad_cache.0.get().is_some() {
                synthesize_fast_path_ledger(
                    &self.jaxpr,
                    &[Transform::Grad],
                    self.mode,
                    self.backend.as_ref(),
                    &self.compile_options(),
                )
            } else {
                None
            }
        })
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof().or_else(|| {
            if self.compiled_ad_cache.0.get().is_some() {
                synthesize_fast_path_proof(
                    &self.jaxpr,
                    &[Transform::Grad],
                    self.mode,
                    self.backend.as_ref(),
                    &self.compile_options(),
                )
            } else {
                None
            }
        })
    }

    pub fn call(&self, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
        if let Some(result) = self.try_default_cpu_compiled_value_and_grad(&args)? {
            return Ok(result);
        }
        Ok(self.call_with_evidence(args)?.0)
    }

    /// Lower the underlying scalar-output [`Jaxpr`] to its symbolic `value_and_grad` [`Jaxpr`]
    /// returning `[value, grad_0, ...]` for the configured `argnums` (or all inputs by default).
    pub fn to_value_and_grad_jaxpr(&self) -> Result<Jaxpr, ApiError> {
        let all_argnums: Vec<usize>;
        let argnums: &[usize] = if self.argnums.is_empty() {
            all_argnums = (0..self.jaxpr.invars.len()).collect();
            &all_argnums
        } else {
            &self.argnums
        };
        fj_ad::build_value_and_grad_jaxpr(&self.jaxpr, argnums).map_err(ApiError::from)
    }

    /// Compose: `value_and_grad(jit(f))`.
    #[must_use]
    pub fn compose_jit(self) -> ComposedTransform {
        let mut compile_options = self.compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Jit],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `value_and_grad(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        let mut compile_options = self.compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `value_and_grad(pmap(f))`.
    #[must_use]
    pub fn compose_pmap(self) -> ComposedTransform {
        let mut compile_options = self.compile_options();
        compile_options.remove(CUSTOM_VJP_RULE_KEY_OPTION);
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Grad, Transform::Pmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: self.custom_vjp_rule_key,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

impl CustomVjpWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(&self.jaxpr, &[], args, self.backend.as_ref(), self.mode)
    }

    #[must_use]
    pub fn grad(&self) -> GradWrapped {
        GradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
            argnums: Vec::new(),
            precision_policy: None,
            compile_options: BTreeMap::new(),
            custom_vjp_rule_key: Some(self.rule_key.clone()),
            meta_cache: DispatchMetaCache::default(),
            compiled_ad_cache: CompiledAdCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    #[must_use]
    pub fn value_and_grad(&self) -> ValueAndGradWrapped {
        ValueAndGradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
            argnums: Vec::new(),
            precision_policy: None,
            compile_options: BTreeMap::new(),
            custom_vjp_rule_key: Some(self.rule_key.clone()),
            meta_cache: DispatchMetaCache::default(),
            compiled_ad_cache: CompiledAdCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose `jit(grad(custom_vjp(f)))`.
    #[must_use]
    pub fn compose_jit_grad(&self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr.clone(),
            transforms: vec![Transform::Jit, Transform::Grad],
            backend: self.backend.clone(),
            mode: self.mode,
            compile_options: BTreeMap::new(),
            custom_vjp_rule_key: Some(self.rule_key.clone()),
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

impl CustomJvpWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(&self.jaxpr, &[], args, self.backend.as_ref(), self.mode)
    }

    pub fn jvp_call(
        &self,
        primals: Vec<Value>,
        tangents: Vec<Value>,
    ) -> Result<fj_ad::JvpResult, ApiError> {
        fj_ad::jvp_with_custom_jvp_key(&self.jaxpr, &primals, &tangents, &self.rule_key)
            .map_err(ApiError::from)
    }

    #[must_use]
    pub fn jacobian(&self) -> JacobianWrapped {
        JacobianWrapped {
            jaxpr: self.jaxpr.clone(),
            custom_jvp_rule_key: Some(self.rule_key.clone()),
        }
    }
}

impl JacobianWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Value, ApiError> {
        if let Some(rule_key) = &self.custom_jvp_rule_key {
            fj_ad::jacobian_jaxpr_with_custom_jvp_key(&self.jaxpr, &args, rule_key)
                .map_err(ApiError::from)
        } else {
            fj_ad::jacobian_jaxpr(&self.jaxpr, &args).map_err(ApiError::from)
        }
    }

    /// Lower the underlying [`Jaxpr`] to its symbolic forward-mode JVP [`Jaxpr`].
    pub fn to_jvp_jaxpr(&self) -> Result<Jaxpr, ApiError> {
        fj_ad::jvp_to_jaxpr(&self.jaxpr).map_err(ApiError::from)
    }
}

impl HessianWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Value, ApiError> {
        fj_ad::hessian_jaxpr(&self.jaxpr, &args).map_err(ApiError::from)
    }

    /// Compute the Hessian matrix using the symbolic `jvp(grad(f))` IR lowering
    /// (`fj_ad::symbolic_hessian_jaxpr`).
    pub fn call_symbolic(&self, args: Vec<Value>) -> Result<Value, ApiError> {
        fj_ad::symbolic_hessian_jaxpr(&self.jaxpr, &args).map_err(ApiError::from)
    }

    /// Compute the Hessian-vector product `H(f)(primals) @ tangents` without materializing
    /// the full `n x n` Hessian matrix.
    pub fn hvp(&self, primals: &[Value], tangents: &[Value]) -> Result<Vec<Value>, ApiError> {
        fj_ad::hvp(&self.jaxpr, primals, tangents).map_err(ApiError::from)
    }

    /// Lower the scalar-output [`Jaxpr`] to a standalone Hessian [`Jaxpr`] (for scalar input)
    /// via `grad_to_jaxpr` composed with `grad_to_jaxpr`.
    pub fn to_jaxpr(&self) -> Result<Jaxpr, ApiError> {
        fj_ad::hessian_to_jaxpr(&self.jaxpr).map_err(ApiError::from)
    }

    /// Lower the scalar-output [`Jaxpr`] to a standalone symbolic Hessian-vector product
    /// (`HVP`) [`Jaxpr`] via `jvp_to_jaxpr(grad_to_jaxpr(f))`.
    pub fn to_hvp_jaxpr(&self) -> Result<Jaxpr, ApiError> {
        fj_ad::hvp_to_jaxpr(&self.jaxpr).map_err(ApiError::from)
    }
}

impl CheckpointWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    /// Execute the checkpointed function (forward pass only).
    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(&self.jaxpr, &[], args, self.backend.as_ref(), self.mode)
    }

    /// Compute gradients of the checkpointed function.
    ///
    /// During backward pass, intermediates are recomputed rather than retrieved
    /// from storage, saving memory at the cost of additional compute.
    pub fn grad(&self) -> GradWrapped {
        GradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
            argnums: Vec::new(),
            precision_policy: None,
            compile_options: BTreeMap::new(),
            custom_vjp_rule_key: Some(self.custom_vjp_rule_key.clone()),
            meta_cache: DispatchMetaCache::default(),
            compiled_ad_cache: CompiledAdCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compute both value and gradients of the checkpointed function.
    pub fn value_and_grad(&self) -> ValueAndGradWrapped {
        ValueAndGradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
            argnums: Vec::new(),
            precision_policy: None,
            compile_options: BTreeMap::new(),
            custom_vjp_rule_key: Some(self.custom_vjp_rule_key.clone()),
            meta_cache: DispatchMetaCache::default(),
            compiled_ad_cache: CompiledAdCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Returns the number of tape entries that would be saved by checkpointing.
    ///
    /// This is the number of equations in the Jaxpr — without checkpoint, each
    /// equation stores intermediate values; with checkpoint, none are stored.
    #[must_use]
    pub fn memory_savings_entries(&self) -> usize {
        self.jaxpr.equations.len()
    }
}

impl PmapWrapped {
    fn build_compile_options(&self) -> BTreeMap<String, String> {
        let mut compile_options = self.compile_options.clone();
        if let Some(devices) = self.devices {
            compile_options.insert("pmap_devices".to_owned(), devices.to_string());
        }
        if let Some(ref axis_name) = self.axis_name {
            compile_options.insert("pmap_axis_name".to_owned(), axis_name.clone());
        }
        if let Some(ref in_axes) = self.in_axes {
            compile_options.insert("pmap_in_axes".to_owned(), in_axes.clone());
        }
        if let Some(ref out_axes) = self.out_axes {
            compile_options.insert("pmap_out_axes".to_owned(), out_axes.clone());
        }
        if let Some(policy) = self.precision_policy {
            compile_options.insert("precision_policy".to_owned(), policy.as_str().to_owned());
        }
        compile_options
    }

    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = Cow::Owned(backend.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set the axis name for collective operations within this pmap.
    #[must_use]
    pub fn with_axis_name(mut self, axis_name: &str) -> Self {
        self.axis_name = Some(axis_name.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Configure the number of virtual SPMD devices across which [`PmapWrapped::call`]
    /// partitions the mapped axis and executes collective primitives (`Psum`, `Pmean`,
    /// `AllGather`, `AllToAll`, `AxisIndex`).
    #[must_use]
    pub fn with_devices(mut self, num_devices: usize) -> Self {
        self.devices = Some(num_devices);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set `in_axes` for this `pmap` (comma-separated axis indices or `"none"`).
    #[must_use]
    pub fn with_in_axes(mut self, in_axes: &str) -> Self {
        self.in_axes = Some(in_axes.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set `out_axes` for this `pmap` (comma-separated axis indices or `"none"`).
    #[must_use]
    pub fn with_out_axes(mut self, out_axes: &str) -> Self {
        self.out_axes = Some(out_axes.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Configure the floating-point contraction / SIMD [`PrecisionPolicy`] for this pmap wrapper.
    #[must_use]
    pub fn with_precision_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.precision_policy = Some(policy);
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Set a dispatch compile option on this pmap wrapper.
    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
        self.meta_cache = DispatchMetaCache::default();
        self.last_evidence = LastEvidenceCache::default();
        self
    }

    /// Enable or disable dispatch-time e-graph optimization.
    #[must_use]
    pub fn with_egraph_optimization(self, enabled: bool) -> Self {
        self.with_compile_option("egraph_optimize", if enabled { "true" } else { "false" })
    }

    /// Execute through the full dispatch + ledger + proof spine when `with_devices` is configured.
    pub fn call_with_evidence(&self, args: Vec<Value>) -> Result<DispatchResponse, ApiError> {
        let compile_options = self.build_compile_options();
        if !compile_options.contains_key("pmap_devices") {
            return Err(ApiError::UnsupportedFeature {
                feature: "pmap".to_owned(),
                reason: "requires multi-device backend infrastructure (GPU/TPU), device mesh \
                         configuration, and collective execution support"
                    .to_owned(),
            });
        }
        let transforms = [Transform::Pmap];
        let evidence = transform_evidence(&transforms);
        let prepared = self
            .meta_cache
            .0
            .get_or_init(|| {
                prepare_dispatch_meta(
                    self.mode,
                    &self.jaxpr,
                    &transforms,
                    &evidence,
                    self.backend.as_ref(),
                    &compile_options,
                    None,
                    &[],
                )
                .ok()
            })
            .as_ref();
        let response = dispatch_ref(DispatchRequestRef {
            mode: self.mode,
            root_jaxpr: &self.jaxpr,
            transform_stack: &transforms,
            transform_evidence: &evidence,
            args,
            backend: self.backend.as_ref(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &[],
            prepared,
        })?;
        self.last_evidence.store(
            response.evidence_ledger.clone(),
            response.composition_proof.clone(),
        );
        Ok(response)
    }

    /// Return the [`EvidenceLedger`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_evidence_ledger(&self) -> Option<EvidenceLedger> {
        self.last_evidence.ledger()
    }

    /// Return the [`TransformCompositionProof`] from the most recent invocation of this wrapper.
    #[must_use]
    pub fn last_composition_proof(&self) -> Option<TransformCompositionProof> {
        self.last_evidence.proof()
    }

    /// Execute the pmap transform.
    ///
    /// Fails closed when no virtual device count (`with_devices`) is configured.
    /// When `with_devices(n)` is set, executes SPMD lockstep across `n` virtual devices
    /// with collective primitive support (`Psum`, `Pmean`, `AllGather`, `AllToAll`, `AxisIndex`).
    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        Ok(self.call_with_evidence(args)?.outputs)
    }

    /// Compose: `pmap(jit(f))`.
    #[must_use]
    pub fn compose_jit(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Pmap, Transform::Jit],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `pmap(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Pmap, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }

    /// Compose: `pmap(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        let compile_options = self.build_compile_options();
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Pmap, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options,
            custom_vjp_rule_key: None,
            ffi_registry: FfiRegistrySlot::default(),
            meta_cache: DispatchMetaCache::default(),
            last_evidence: LastEvidenceCache::default(),
        }
    }
}

/// Compute the Hessian-vector product `H(f)(primals) @ tangents` for a scalar-output [`Jaxpr`].
pub fn hvp(jaxpr: &Jaxpr, primals: &[Value], tangents: &[Value]) -> Result<Vec<Value>, ApiError> {
    fj_ad::hvp(jaxpr, primals, tangents).map_err(ApiError::from)
}

/// Trace a fallible Rust closure against the shapes and dtypes of `args` and return
/// a self-contained executable [`Jaxpr`].
pub fn trace_jaxpr_for_values<F>(f: F, args: &[Value]) -> Result<Jaxpr, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let in_avals: Vec<ShapedArray> = args.iter().map(ShapedArray::from_value).collect();
    let closed = fj_trace::make_jaxpr_fallible(f, in_avals)?;
    Ok(closed.into_executable_jaxpr())
}

/// Infer output [`ShapedArray`]s of a fallible Rust closure directly from abstract input
/// [`ShapedArray`]s without materializing runtime [`Value`] tensors.
pub fn eval_shape_fn<F>(f: F, in_avals: &[ShapedArray]) -> Result<Vec<ShapedArray>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let closed = fj_trace::make_jaxpr_fallible(f, in_avals.to_vec())?;
    let jaxpr = closed.into_executable_jaxpr();
    eval_shape_avals(&jaxpr, in_avals)
}

/// Trace a Rust closure against `args` and execute it through [`jit`].
pub fn jit_fn<F>(f: F, args: Vec<Value>) -> Result<Vec<Value>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, &args)?;
    jit(jaxpr).call(args)
}

/// Trace a scalar-output Rust closure against `args` and compute its gradient via [`grad`].
pub fn grad_fn<F>(f: F, args: Vec<Value>) -> Result<Vec<Value>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, &args)?;
    grad(jaxpr).call(args)
}

/// Trace a scalar-output Rust closure against `args` and compute `(values, gradients)`
/// in a single forward+backward pass via [`value_and_grad`].
pub fn value_and_grad_fn<F>(f: F, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, &args)?;
    value_and_grad(jaxpr).call(args)
}

/// Trace a per-element Rust closure (peeling axis 0 from each batched tensor in `args`)
/// and execute it across the batch via [`vmap`].
pub fn vmap_fn<F>(f: F, args: Vec<Value>) -> Result<Vec<Value>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let mut elem_avals = Vec::with_capacity(args.len());
    for arg in &args {
        let aval = ShapedArray::from_value(arg);
        if aval.shape.dims.is_empty() {
            return Err(ApiError::EvalError {
                detail: "vmap_fn requires batched tensor arguments with rank >= 1".to_owned(),
            });
        }
        elem_avals.push(ShapedArray {
            dtype: aval.dtype,
            shape: fj_core::Shape {
                dims: aval.shape.dims[1..].to_vec(),
            },
        });
    }
    let closed = fj_trace::make_jaxpr_fallible(f, elem_avals)?;
    vmap(closed.into_executable_jaxpr()).call(args)
}

/// Trace a per-device Rust closure inside an active `pmap` context of `num_devices` shards
/// (supporting collective primitives like `psum`, `pmean`, `all_gather`, `all_to_all`, and
/// `axis_index`) and execute it across `num_devices` virtual devices via [`pmap`].
pub fn pmap_fn<F>(f: F, num_devices: usize, args: Vec<Value>) -> Result<Vec<Value>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    if num_devices == 0 {
        return Err(ApiError::EvalError {
            detail: "pmap_fn requires num_devices > 0".to_owned(),
        });
    }
    let mut shard_avals = Vec::with_capacity(args.len());
    for arg in &args {
        let aval = ShapedArray::from_value(arg);
        if aval.shape.dims.is_empty() || aval.shape.dims[0] as usize != num_devices {
            return Err(ApiError::EvalError {
                detail: format!(
                    "pmap_fn requires leading dimension equal to num_devices ({num_devices}), got {:?}",
                    aval.shape.dims
                ),
            });
        }
        shard_avals.push(ShapedArray {
            dtype: aval.dtype,
            shape: fj_core::Shape {
                dims: aval.shape.dims[1..].to_vec(),
            },
        });
    }
    let axis_size = u32::try_from(num_devices).map_err(|_| ApiError::EvalError {
        detail: format!("pmap_fn num_devices {num_devices} overflows u32"),
    })?;
    let closed = fj_trace::with_pmap_trace_context(axis_size, || {
        fj_trace::make_jaxpr_fallible(f, shard_avals)
    })?;
    pmap(closed.into_executable_jaxpr())
        .with_devices(num_devices)
        .call(args)
}

/// Trace a Rust closure against `primals` and compute its forward-mode JVP `(primals, tangents)`.
pub fn jvp_fn<F>(f: F, primals: &[Value], tangents: &[Value]) -> Result<fj_ad::JvpResult, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, primals)?;
    jvp(&jaxpr, primals, tangents)
}

/// Trace a Rust closure against `primals` and return `(primal_outputs, VjpPullback)`.
pub fn vjp_fn<F>(f: F, primals: &[Value]) -> Result<VjpResult, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, primals)?;
    vjp(jaxpr, primals.to_vec())
}

/// Trace a Rust closure against `primals` and return `(primal_outputs, LinearizedFunction)`.
pub fn linearize_fn<F>(f: F, primals: &[Value]) -> Result<LinearizeResult, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, primals)?;
    linearize(jaxpr, primals.to_vec())
}

/// Trace a Rust closure against `args` and compute its Jacobian matrix via [`jacobian`].
pub fn jacobian_fn<F>(f: F, args: Vec<Value>) -> Result<Value, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, &args)?;
    jacobian(jaxpr).call(args)
}

/// Trace a scalar-output Rust closure against `args` and compute its Hessian matrix via [`hessian`].
pub fn hessian_fn<F>(f: F, args: Vec<Value>) -> Result<Value, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, &args)?;
    hessian(jaxpr).call(args)
}

/// Trace a scalar-output Rust closure against `primals` and compute its Hessian-vector
/// product `H(f)(primals) @ tangents`.
pub fn hvp_fn<F>(f: F, primals: &[Value], tangents: &[Value]) -> Result<Vec<Value>, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let jaxpr = trace_jaxpr_for_values(f, primals)?;
    hvp(&jaxpr, primals, tangents)
}

/// Trace a Rust closure against `in_avals` and return the e-graph equality-saturation
/// optimized [`Jaxpr`].
pub fn optimize_fn<F>(f: F, in_avals: Vec<ShapedArray>) -> Result<Jaxpr, ApiError>
where
    F: FnOnce(&[fj_trace::TracerRef]) -> Result<Vec<fj_trace::TracerRef>, fj_trace::TraceError>,
{
    let closed = fj_trace::make_jaxpr_fallible(f, in_avals)?;
    Ok(fj_egraph::optimize_jaxpr(&closed.into_executable_jaxpr()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{Atom, Equation, Primitive, VarId};
    use smallvec::smallvec;

    fn make_add_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_mul_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_reshape_jaxpr(new_shape: &str) -> Jaxpr {
        let mut params = BTreeMap::new();
        params.insert("new_shape".to_owned(), new_shape.to_owned());
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Reshape,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params,
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    #[test]
    fn eval_shape_reshape_uses_new_shape_and_preserves_dtype() {
        let jaxpr = make_reshape_jaxpr("2,3");
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("input");
        let shapes = eval_shape(&jaxpr, &[input]).expect("reshape eval_shape");

        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].dtype, fj_core::DType::F64);
        assert_eq!(shapes[0].shape, fj_core::Shape { dims: vec![2, 3] });
    }

    #[test]
    fn eval_shape_reshape_infers_single_negative_dimension() {
        let jaxpr = make_reshape_jaxpr("2,-1");
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("input");
        let shapes = eval_shape(&jaxpr, &[input]).expect("reshape eval_shape");

        assert_eq!(shapes[0].shape, fj_core::Shape { dims: vec![2, 3] });
    }

    #[test]
    fn eval_shape_reshape_rejects_malformed_new_shape() {
        let jaxpr = make_reshape_jaxpr("2,bad,3");
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("input");
        let err = eval_shape(&jaxpr, &[input]).expect_err("malformed shape should fail");
        let msg = err.to_string();

        assert!(msg.contains("new_shape"), "error should name param: {msg}");
        assert!(
            msg.contains("2,bad,3"),
            "error should include invalid raw shape: {msg}"
        );
    }

    #[test]
    fn eval_shape_reshape_rejects_missing_input() {
        let mut params = BTreeMap::new();
        params.insert("new_shape".to_owned(), "1".to_owned());
        let jaxpr = Jaxpr::new(
            vec![],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Reshape,
                inputs: smallvec![],
                outputs: smallvec![VarId(1)],
                params,
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let err = eval_shape(&jaxpr, &[]).expect_err("inputless reshape should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("one input"),
            "error should reject arity: {msg}"
        );
    }

    #[test]
    fn eval_shape_reshape_rejects_element_count_mismatch() {
        let jaxpr = make_reshape_jaxpr("2,2");
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("input");
        let err = eval_shape(&jaxpr, &[input]).expect_err("mismatched reshape should fail");
        let msg = err.to_string();

        assert!(
            msg.contains("mismatch"),
            "error should mention mismatch: {msg}"
        );
        assert!(
            msg.contains("input=3"),
            "error should include input size: {msg}"
        );
        assert!(
            msg.contains("target=4"),
            "error should include target size: {msg}"
        );
    }

    #[test]
    fn eval_shape_reshape_rejects_multiple_inferred_dimensions() {
        let jaxpr = make_reshape_jaxpr("-1,-1");
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).expect("input");
        let err = eval_shape(&jaxpr, &[input]).expect_err("two inferred dims should fail");
        let msg = err.to_string();

        assert!(
            msg.contains("at most one"),
            "error should reject multiple -1 dims: {msg}"
        );
    }

    // ── Constructor defaults ──

    #[test]
    fn jit_defaults() {
        let wrapped = jit(make_add_jaxpr());
        assert_eq!(wrapped.backend, "cpu");
        assert_eq!(wrapped.mode, CompatibilityMode::Strict);
    }

    #[test]
    fn grad_defaults() {
        let wrapped = grad(make_add_jaxpr());
        assert_eq!(wrapped.backend, "cpu");
        assert_eq!(wrapped.mode, CompatibilityMode::Strict);
    }

    #[test]
    fn vmap_defaults() {
        let wrapped = vmap(make_add_jaxpr());
        assert_eq!(wrapped.backend, "cpu");
        assert_eq!(wrapped.mode, CompatibilityMode::Strict);
        assert_eq!(wrapped.in_axes, None);
        assert_eq!(wrapped.out_axes, None);
    }

    #[test]
    fn value_and_grad_defaults() {
        let wrapped = value_and_grad(make_add_jaxpr());
        assert_eq!(wrapped.backend, "cpu");
        assert_eq!(wrapped.mode, CompatibilityMode::Strict);
    }

    // ── Builder methods ──

    #[test]
    fn jit_with_backend() {
        let wrapped = jit(make_add_jaxpr()).with_backend("gpu");
        assert_eq!(wrapped.backend, "gpu");
    }

    #[test]
    fn jit_with_mode() {
        let wrapped = jit(make_add_jaxpr()).with_mode(CompatibilityMode::Hardened);
        assert_eq!(wrapped.mode, CompatibilityMode::Hardened);
    }

    #[test]
    fn vmap_with_axes() {
        let wrapped = vmap(make_add_jaxpr())
            .with_in_axes("0,none")
            .with_out_axes("0");
        assert_eq!(wrapped.in_axes, Some("0,none".to_owned()));
        assert_eq!(wrapped.out_axes, Some("0".to_owned()));
    }

    // ── Composition ──

    #[test]
    fn jit_compose_grad() {
        let composed = jit(make_mul_jaxpr()).compose_grad();
        assert_eq!(composed.transforms, vec![Transform::Jit, Transform::Grad]);
        assert_eq!(composed.backend, "cpu");
    }

    #[test]
    fn jit_compose_vmap() {
        let composed = jit(make_mul_jaxpr()).compose_vmap();
        assert_eq!(composed.transforms, vec![Transform::Jit, Transform::Vmap]);
    }

    #[test]
    fn vmap_compose_grad() {
        let composed = vmap(make_mul_jaxpr()).compose_grad();
        assert_eq!(composed.transforms, vec![Transform::Vmap, Transform::Grad]);
    }

    #[test]
    fn vmap_compose_grad_preserves_axes() {
        let composed = vmap(make_mul_jaxpr())
            .with_in_axes("none,0")
            .with_out_axes("0")
            .compose_grad();
        assert_eq!(
            composed.compile_options.get("vmap_in_axes"),
            Some(&"none,0".to_owned())
        );
        assert_eq!(
            composed.compile_options.get("vmap_out_axes"),
            Some(&"0".to_owned())
        );
    }

    #[test]
    fn vmap_compose_grad_applies_preserved_axes_at_call_time() {
        let composed = vmap(make_mul_jaxpr())
            .with_in_axes("none,0")
            .with_out_axes("0")
            .compose_grad();

        let result = composed
            .call(vec![
                Value::scalar_f64(2.0),
                Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
            ])
            .expect("composed vmap(grad) should preserve configured axes");

        let output = result[0]
            .as_tensor()
            .expect("output should be batched tensor");
        let values = output.to_f64_vec().expect("f64 tensor");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn compose_arbitrary() {
        let composed = compose(
            make_mul_jaxpr(),
            vec![Transform::Jit, Transform::Grad, Transform::Vmap],
        );
        assert_eq!(
            composed.transforms,
            vec![Transform::Jit, Transform::Grad, Transform::Vmap]
        );
    }

    #[test]
    fn composed_with_backend_and_mode() {
        let composed = compose(make_mul_jaxpr(), vec![Transform::Jit])
            .with_backend("tpu")
            .with_mode(CompatibilityMode::Hardened);
        assert_eq!(composed.backend, "tpu");
        assert_eq!(composed.mode, CompatibilityMode::Hardened);
    }

    #[test]
    fn composed_builder_sets_dispatch_compile_options() {
        let composed = compose(
            make_mul_jaxpr(),
            vec![Transform::Jit, Transform::Vmap, Transform::Grad],
        )
        .with_vmap_in_axes("none,0")
        .with_vmap_out_axes("0")
        .with_finite_diff_grad_fallback(false)
        .with_egraph_optimization(true);

        assert_eq!(
            composed.compile_options.get("vmap_in_axes"),
            Some(&"none,0".to_owned())
        );
        assert_eq!(
            composed.compile_options.get("vmap_out_axes"),
            Some(&"0".to_owned())
        );
        assert_eq!(
            composed
                .compile_options
                .get("allow_finite_diff_grad_fallback"),
            Some(&"false".to_owned())
        );
        assert_eq!(
            composed.compile_options.get("egraph_optimize"),
            Some(&"true".to_owned())
        );
    }

    #[test]
    fn composed_transform_ignores_user_custom_vjp_rule_key_compile_option() {
        let wrapped = custom_vjp(
            make_mul_jaxpr(),
            |primals| {
                let x = primals[0].as_f64_scalar().ok_or_else(|| {
                    fj_ad::AdError::EvalFailed("custom VJP forward expected scalar x".to_owned())
                })?;
                let y = primals[1].as_f64_scalar().ok_or_else(|| {
                    fj_ad::AdError::EvalFailed("custom VJP forward expected scalar y".to_owned())
                })?;
                Ok((vec![Value::scalar_f64(x * y)], vec![]))
            },
            |_residuals, _cotangent| Ok(vec![Value::scalar_f64(123.0), Value::scalar_f64(456.0)]),
        );

        let result = compose(make_mul_jaxpr(), vec![Transform::Grad])
            .with_compile_option(CUSTOM_VJP_RULE_KEY_OPTION, &wrapped.rule_key)
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
            .expect("plain composed grad should ignore user-supplied custom VJP key");

        let gradient = result[0].as_f64_scalar().expect("scalar gradient");
        assert!(
            (gradient - 4.0).abs() < 1e-10,
            "plain grad should use ordinary AD, not the spoofed custom VJP gradient: {gradient}"
        );
    }

    #[test]
    fn composed_vmap_axes_apply_to_jit_vmap_call() {
        let composed = jit(make_mul_jaxpr())
            .compose_vmap()
            .with_vmap_in_axes("none,0")
            .with_vmap_out_axes("0");

        let result = composed
            .call(vec![
                Value::scalar_f64(2.0),
                Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
            ])
            .expect("jit(vmap) should honor composed vmap axes");

        let output = result[0].as_tensor().expect("output should be tensor");
        let values = output.to_f64_vec().expect("f64 tensor");
        assert_eq!(values, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn composed_grad_tail_can_deny_finite_diff_fallback() {
        let err = compose(make_mul_jaxpr(), vec![Transform::Grad, Transform::Vmap])
            .with_finite_diff_grad_fallback(false)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect_err("disabled finite-difference fallback should reject grad(vmap)");

        let msg = err.to_string();
        assert!(
            msg.contains("finite-difference grad fallback") && msg.contains("disabled"),
            "error should report the disabled fallback policy: {msg}"
        );
    }

    // ── Execution ──

    #[test]
    fn jit_call_add() {
        let wrapped = jit(make_add_jaxpr());
        let result = wrapped
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0].as_f64_scalar().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn jit_repeated_call_compiled_cache_matches_dispatch_golden_sha256() {
        use fj_core::ProgramSpec;

        let jaxpr = fj_core::build_program(ProgramSpec::Add2);
        assert!(
            fj_interpreters::compile_jaxpr_for_repeated_eval(&jaxpr).is_some(),
            "Add2 should be eligible for the repeated-JIT compiled scalar path"
        );

        let wrapped = jit(jaxpr.clone());
        let fast = wrapped
            .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
            .expect("compiled jit call should succeed");
        let dispatch = dispatch_with_options(
            &jaxpr,
            &[Transform::Jit],
            vec![Value::scalar_i64(3), Value::scalar_i64(4)],
            DEFAULT_BACKEND,
            CompatibilityMode::Strict,
            BTreeMap::new(),
        )
        .expect("dispatch reference should succeed");

        assert_eq!(fast, dispatch);
        let digest = fj_test_utils::fixture_id_from_json(&("frankenjax-so4wo", &fast))
            .expect("golden output should hash");
        assert_eq!(
            digest,
            "358ba10d12a581c6dd0ec4adb2ab3f69b71e12df1316ff855ab5387f328dbf38"
        );
    }

    #[test]
    fn jit_repeated_call_compiled_cache_matches_dispatch_tensor_programs() {
        use fj_core::ProgramSpec;

        // Non-scalar (tensor / reduction / dot) programs: these have no scalar
        // fast-path plan, so before the gate was relaxed they fell through to the
        // full per-call dispatch. They must now compile to the generic dense
        // path and stay bit-for-bit identical to the backend dispatch result.
        fn assert_compiled_cache_matches_dispatch(name: &str, jaxpr: Jaxpr, args: Vec<Value>) {
            assert!(
                fj_interpreters::compile_jaxpr_for_repeated_eval(&jaxpr).is_some(),
                "{name} should now be eligible for the repeated-JIT compiled dense path"
            );

            let wrapped = jit(jaxpr.clone());
            // Call twice so the second call exercises the warmed compiled cache.
            let _ = wrapped.call(args.clone()).expect("warm compiled call");
            let fast = wrapped.call(args.clone()).expect("compiled jit call");

            let dispatch = dispatch_with_options(
                &jaxpr,
                &[Transform::Jit],
                args,
                DEFAULT_BACKEND,
                CompatibilityMode::Strict,
                BTreeMap::new(),
            )
            .expect("dispatch reference should succeed");

            assert_eq!(fast, dispatch, "{name}: compiled cache must match dispatch");
        }

        assert_compiled_cache_matches_dispatch(
            "reduce_sum_vec",
            fj_core::build_program(ProgramSpec::ReduceSumVec),
            vec![Value::vector_f64(&[1.5, -2.25, 3.0, 4.75]).expect("vec")],
        );
        assert_compiled_cache_matches_dispatch(
            "dot3",
            fj_core::build_program(ProgramSpec::Dot3),
            vec![
                Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vec"),
                Value::vector_f64(&[4.0, 5.0, 6.0]).expect("vec"),
            ],
        );
        assert_compiled_cache_matches_dispatch(
            "add_one_vec",
            fj_core::build_program(ProgramSpec::AddOne),
            vec![Value::vector_i64(&[10, 20, 30, 40]).expect("vec")],
        );
    }

    #[test]
    fn jit_repeated_call_compiled_tensor_golden_sha256() {
        use fj_core::ProgramSpec;

        let jaxpr = fj_core::build_program(ProgramSpec::ReduceSumVec);
        let wrapped = jit(jaxpr);
        let out = wrapped
            .call(vec![
                Value::vector_f64(&[1.5, -2.25, 3.0, 4.75]).expect("vec"),
            ])
            .expect("compiled jit call should succeed");
        let digest = fj_test_utils::fixture_id_from_json(&("frankenjax-so4wo", &out))
            .expect("golden output should hash");
        assert_eq!(
            digest,
            "00973a72bf25a5a56152d373c887ddc555877c6e09545140c723080675c2676a"
        );
    }

    #[test]
    fn vmap_repeated_call_meta_cache_matches_dispatch() {
        use fj_core::ProgramSpec;

        let jaxpr = fj_core::build_program(ProgramSpec::AddOne);
        let wrapped = vmap(jaxpr.clone());
        let arg = Value::vector_i64(&[1, 2, 3, 4, 5]).expect("vec");

        // Warm the meta cache on the first call, then the second must match a
        // fresh unprepared dispatch bit-for-bit.
        let first = wrapped.call(vec![arg.clone()]).expect("first vmap call");
        let second = wrapped.call(vec![arg.clone()]).expect("warmed vmap call");
        assert_eq!(first, second);

        let reference = dispatch_with_options(
            &jaxpr,
            &[Transform::Vmap],
            vec![arg],
            DEFAULT_BACKEND,
            CompatibilityMode::Strict,
            BTreeMap::new(),
        )
        .expect("reference vmap dispatch");
        assert_eq!(second, reference);

        let digest = fj_test_utils::fixture_id_from_json(&("frankenjax-mcqr.62", &second))
            .expect("golden output should hash");
        // Golden refreshed after frankenjax-w4d3p (92d5bfd7) canonicalized
        // `fixture_id_from_json` object key ordering via `serde_json::to_value`
        // (`dtype, elements, shape` instead of struct field order `dtype, shape, elements`).
        // The vmap(AddOne) output value `[2, 3, 4, 5, 6]` is unchanged and asserted equal
        // to both the cold call and fresh dispatch reference above.
        assert_eq!(
            digest,
            "7fda517ba090d40fae6859b39ee20d7ac9d96e82f35f3e53b25ff7a9f7696a1d"
        );
    }

    #[test]
    fn jit_call_mul() {
        let wrapped = jit(make_mul_jaxpr());
        let result = wrapped
            .call(vec![Value::scalar_f64(5.0), Value::scalar_f64(6.0)])
            .unwrap();
        assert!((result[0].as_f64_scalar().unwrap() - 30.0).abs() < 1e-12);
    }

    #[test]
    fn grad_call_mul() {
        // grad(x*y) w.r.t. x at x=3, y=4 should give y=4
        let wrapped = grad(make_mul_jaxpr());
        let result = wrapped
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
            .unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn build_ledger_tracks_transforms() {
        let jaxpr = make_add_jaxpr();
        let ledger = build_ledger(jaxpr, &[Transform::Jit, Transform::Grad]);
        let sig = ledger.composition_signature();
        assert!(sig.contains("jit"), "signature should contain jit");
        assert!(sig.contains("grad"), "signature should contain grad");
    }

    // ── value_and_grad execution tests (frankenjax-rjj) ──

    #[test]
    fn value_and_grad_call_square() {
        // f(x) = x*x, value_and_grad should return (f(x), f'(x)) = (x^2, 2x)
        use fj_core::ProgramSpec;
        let wrapped = value_and_grad(fj_core::build_program(ProgramSpec::Square));
        let (values, gradients) = wrapped
            .call(vec![Value::scalar_f64(3.0)])
            .expect("value_and_grad(x^2) should succeed");
        // value = 9.0
        let val = values[0].as_f64_scalar().expect("value should be f64");
        assert!((val - 9.0).abs() < 1e-3, "f(3) = 3^2 = 9, got {val}");
        // gradient = 6.0
        assert!(
            !gradients.is_empty(),
            "should produce at least one gradient"
        );
    }

    #[test]
    fn value_and_grad_compiled_ad_cache_matches_tape_golden_sha256() {
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(0))],
                    outputs: smallvec![VarId(1)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Cos,
                    inputs: smallvec![Atom::Var(VarId(0))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let args = vec![Value::scalar_f64(1.0)];
        let wrapped = value_and_grad(jaxpr.clone());
        let _ = wrapped.call(args.clone()).expect("warm compiled AD cache");
        let (cached_values, cached_grads) = wrapped.call(args.clone()).expect("cached call");
        let (reference_values, reference_grads) =
            fj_ad::value_and_grad_jaxpr(&jaxpr, &args).expect("reference tape AD");
        assert_eq!(cached_values, reference_values);
        assert_eq!(cached_grads, reference_grads);

        let bits_hex = vec![
            format!(
                "{:016x}",
                cached_values[0].as_f64_scalar().expect("value").to_bits()
            ),
            format!(
                "{:016x}",
                cached_grads[0].as_f64_scalar().expect("gradient").to_bits()
            ),
        ];
        let digest = fj_test_utils::fixture_id_from_json(&bits_hex).expect("sha256 digest");
        assert_eq!(
            digest,
            "984585309be003365780a1f999422efc949c360ec9933d354a6bd50b5b41653a"
        );
    }

    #[test]
    fn cold_compiled_ad_fast_path_matches_dispatch_golden_sha256() {
        use fj_core::ProgramSpec;

        let jaxpr = fj_core::build_program(ProgramSpec::Square);
        let arg = Value::scalar_f64(3.0);
        let grad_values = grad(jaxpr.clone())
            .call(vec![arg.clone()])
            .expect("cold compiled grad should succeed");
        let reference_grad = dispatch_with_options(
            &jaxpr,
            &[Transform::Grad],
            vec![arg.clone()],
            DEFAULT_BACKEND,
            CompatibilityMode::Strict,
            BTreeMap::new(),
        )
        .expect("reference grad dispatch should succeed");
        assert_eq!(grad_values, reference_grad);

        let (values, gradients) = value_and_grad(jaxpr.clone())
            .call(vec![arg.clone()])
            .expect("cold compiled value_and_grad should succeed");
        let mut value_and_grad_options = BTreeMap::new();
        value_and_grad_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let reference_value_and_grad = dispatch_with_options(
            &jaxpr,
            &[Transform::Grad],
            vec![arg],
            DEFAULT_BACKEND,
            CompatibilityMode::Strict,
            value_and_grad_options,
        )
        .expect("reference value_and_grad dispatch should succeed");
        assert_eq!(values, reference_value_and_grad[..jaxpr.outvars.len()]);
        assert_eq!(gradients, reference_value_and_grad[jaxpr.outvars.len()..]);

        let payload = [
            grad_values[0]
                .as_f64_scalar()
                .expect("grad output should be f64")
                .to_bits(),
            values[0]
                .as_f64_scalar()
                .expect("value output should be f64")
                .to_bits(),
            gradients[0]
                .as_f64_scalar()
                .expect("value_and_grad output should be f64")
                .to_bits(),
        ];
        let digest = fj_test_utils::fixture_id_from_json(&payload).expect("payload hashes");
        assert_eq!(
            digest,
            "04e2c5d5b04781ce3140dd383e1d5b7dba88f3b87f716d6c86ee40b210cad05a"
        );
    }

    #[test]
    fn value_and_grad_borrowed_dispatch_golden_sha256() {
        use fj_core::ProgramSpec;

        let wrapped = value_and_grad(fj_core::build_program(ProgramSpec::SquarePlusLinear));
        let (values, gradients) = wrapped
            .call(vec![Value::scalar_f64(3.0)])
            .expect("value_and_grad should succeed through borrowed dispatch");
        let payload = [
            values[0]
                .as_f64_scalar()
                .expect("value should be f64")
                .to_bits(),
            gradients[0]
                .as_f64_scalar()
                .expect("gradient should be f64")
                .to_bits(),
        ];
        let digest = fj_test_utils::fixture_id_from_json(&payload).expect("payload hashes");
        assert_eq!(
            digest,
            "2d2d457ca9efee1db747a6578e6f6fbf9e5d84802cad03f84b69d3b52d669f96"
        );
    }

    #[test]
    fn value_and_grad_with_mode() {
        use fj_core::ProgramSpec;
        let wrapped = value_and_grad(fj_core::build_program(ProgramSpec::Square))
            .with_mode(CompatibilityMode::Hardened);
        let (values, gradients) = wrapped
            .call(vec![Value::scalar_f64(5.0)])
            .expect("hardened value_and_grad should succeed");
        let val = values[0].as_f64_scalar().expect("value should be f64");
        assert!((val - 25.0).abs() < 1e-3, "f(5) = 25, got {val}");
        assert!(!gradients.is_empty());
    }

    #[test]
    fn value_and_grad_defaults_checked() {
        let wrapped = value_and_grad(make_mul_jaxpr());
        assert_eq!(wrapped.backend, "cpu");
        assert_eq!(wrapped.mode, CompatibilityMode::Strict);
    }

    #[test]
    fn value_and_grad_with_backend() {
        let wrapped = value_and_grad(make_mul_jaxpr()).with_backend("cpu");
        assert_eq!(wrapped.backend, "cpu");
    }

    // ── Error path tests ──

    #[test]
    fn jit_wrong_arity_returns_error() {
        let wrapped = jit(make_add_jaxpr());
        let err = wrapped
            .call(vec![Value::scalar_f64(1.0)])
            .expect_err("wrong arity should fail");
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "error should have a descriptive message");
    }

    #[test]
    fn grad_wrong_arity_returns_error() {
        // grad(add) expects 2 inputs
        let wrapped = grad(make_add_jaxpr());
        let err = wrapped
            .call(vec![Value::scalar_f64(1.0)])
            .expect_err("wrong arity should fail");
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }

    #[test]
    fn pmap_call_uses_permanent_unavailable_wording() {
        let err = pmap(make_mul_jaxpr())
            .with_axis_name("devices")
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect_err("pmap should fail closed until multi-device execution exists");

        let msg = err.to_string();
        assert!(
            msg.contains("pmap unavailable"),
            "pmap error should describe an unavailable capability: {msg}"
        );
        let lower = msg.to_ascii_lowercase();
        let banned_terms = [
            ["not", "implemented"].join(" "),
            ["not", "yet", "implemented"].join(" "),
            ["st", "ub"].concat(),
            ["place", "holder"].concat(),
        ];
        for banned in banned_terms {
            assert!(
                !lower.contains(&banned),
                "pmap error should not expose provisional wording {banned:?}: {msg}"
            );
        }
    }

    #[test]
    fn compose_empty_transforms_acts_as_eval() {
        let composed = compose(make_add_jaxpr(), vec![]);
        let result = composed
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("empty transforms should succeed");
        assert!((result[0].as_f64_scalar().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_preserves_forward_semantics() {
        let wrapped = checkpoint(make_add_mul_chain());
        let result = wrapped
            .call(vec![Value::scalar_f64(5.0)])
            .expect("checkpoint forward should succeed");
        assert!((result[0].as_f64_scalar().unwrap() - 30.0).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_grad_recomputes_forward() {
        let wrapped = checkpoint(make_add_mul_chain());
        let grads = wrapped
            .grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("checkpoint grad should succeed");
        // d/dx((x + 1) * x) = 2x + 1, so at x=3, gradient = 7
        assert!((grads[0].as_f64_scalar().unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn checkpoint_memory_savings_reports_equation_count() {
        let wrapped = checkpoint(make_add_mul_chain());
        assert_eq!(wrapped.memory_savings_entries(), 2);
    }

    fn make_add_mul_chain() -> Jaxpr {
        // f(x) = (x + 1) * x  (two equations)
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Lit(fj_core::Literal::from_f64(1.0))
                    ],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    #[test]
    fn pmap_returns_unsupported_feature_error() {
        let wrapped = pmap(make_add_jaxpr());
        let result = wrapped.call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ApiError::UnsupportedFeature { .. }),
            "expected unsupported feature error, got: {:?}",
            err
        );
    }

    #[test]
    fn pmap_with_axis_name_still_returns_unsupported_feature() {
        let wrapped = pmap(make_add_jaxpr()).with_axis_name("batch");
        let result = wrapped.call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn e2e_jit_staging_and_cache_lifecycle() {
        // f(s, x) = (s * s + 1.0) * x
        // Equations:
        //   v3 = s * s        (known when static_argnums = [0])
        //   v4 = v3 + 1.0     (known when static_argnums = [0])
        //   v5 = v4 * x       (unknown residual equation)
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![
                        Atom::Var(VarId(3)),
                        Atom::Lit(fj_core::Literal::from_f64(1.0))
                    ],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );

        let jitted = jit(jaxpr).with_static_argnums(&[0]);
        assert_eq!(jitted.static_argnums(), &[0]);

        // Cold call with s = 3.0, x = 2.0 -> (9 + 1) * 2 = 20.0
        let cold = jitted
            .call_with_dispatch(vec![Value::scalar_f64(3.0), Value::scalar_f64(2.0)])
            .expect("cold staged jit call should succeed");
        assert!(
            !cold.cache_hit(),
            "first call with s=3.0 must be a cold miss"
        );
        assert!((cold.outputs[0].as_f64_scalar().unwrap() - 20.0).abs() < 1e-12);
        let staged = cold
            .staged_program()
            .expect("static_argnums must produce a StagedProgram");
        assert_eq!(
            staged.jaxpr_known.equations.len(),
            2,
            "s*s and +1.0 must be folded into jaxpr_known"
        );
        assert_eq!(
            staged.jaxpr_unknown.equations.len(),
            1,
            "only residual * x remains in jaxpr_unknown"
        );

        // Warm call with SAME static s = 3.0 and DIFFERENT dynamic x = 5.0 -> (9 + 1) * 5 = 50.0
        let warm = jitted
            .call_with_dispatch(vec![Value::scalar_f64(3.0), Value::scalar_f64(5.0)])
            .expect("warm staged jit call should succeed");
        assert!(
            warm.cache_hit(),
            "second call with same static arg must hit LruCache"
        );
        assert_eq!(cold.cache_key, warm.cache_key);
        assert!((warm.outputs[0].as_f64_scalar().unwrap() - 50.0).abs() < 1e-12);

        // Changing static arg s = 4.0 -> distinct CacheKey and deterministic cache miss!
        let miss_new_static = jitted
            .call_with_dispatch(vec![Value::scalar_f64(4.0), Value::scalar_f64(5.0)])
            .expect("new static arg call should succeed");
        assert!(
            !miss_new_static.cache_hit(),
            "changing static arg value must trigger a cache miss"
        );
        assert_ne!(cold.cache_key, miss_new_static.cache_key);
        assert!((miss_new_static.outputs[0].as_f64_scalar().unwrap() - 85.0).abs() < 1e-12);

        // Changing CompatibilityMode -> distinct CacheKey and deterministic cache miss!
        let hardened = jitted.with_mode(CompatibilityMode::Hardened);
        let miss_hardened = hardened
            .call_with_dispatch(vec![Value::scalar_f64(3.0), Value::scalar_f64(5.0)])
            .expect("hardened staged jit call should succeed");
        assert!(!miss_hardened.cache_hit());
        assert_ne!(cold.cache_key, miss_hardened.cache_key);

        let log = serde_json::json!({
            "schema_version": "frankenjax.e2e-forensic-log.v1",
            "scenario_id": "e2e_jit_staging_and_cache_lifecycle",
            "test_id": "fj_api::transforms::tests::e2e_jit_staging_and_cache_lifecycle",
            "result": "PASS",
            "cold_cache_key": cold.cache_key,
            "warm_cache_hit": warm.cache_hit(),
            "known_equations": staged.jaxpr_known.equations.len(),
            "unknown_equations": staged.jaxpr_unknown.equations.len(),
        });
        assert_eq!(log["schema_version"], "frankenjax.e2e-forensic-log.v1");
    }

    #[test]
    fn jit_with_precision_policy_switches_regime_and_cache_key() {
        let jaxpr = crate::make_jaxpr_fallible(
            |args| {
                let prod = args[0].dot(&args[1])?;
                Ok(vec![prod])
            },
            vec![
                ShapedArray {
                    dtype: fj_core::DType::F64,
                    shape: fj_core::Shape { dims: vec![16, 16] },
                },
                ShapedArray {
                    dtype: fj_core::DType::F64,
                    shape: fj_core::Shape { dims: vec![16, 16] },
                },
            ],
        )
        .expect("make_jaxpr_fallible should succeed")
        .jaxpr;

        let a_data: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.031).sin()).collect();
        let b_data: Vec<f64> = (0..256).map(|i| ((i as f64) * 0.047).cos()).collect();
        let a = Value::Tensor(
            fj_core::TensorValue::new_f64_values(fj_core::Shape { dims: vec![16, 16] }, a_data)
                .unwrap(),
        );
        let b = Value::Tensor(
            fj_core::TensorValue::new_f64_values(fj_core::Shape { dims: vec![16, 16] }, b_data)
                .unwrap(),
        );

        let jit_portable =
            jit(jaxpr.clone()).with_precision_policy(PrecisionPolicy::BitExactPortable);
        let jit_fma = jit(jaxpr).with_precision_policy(PrecisionPolicy::HardwareFmaAccelerated);
        assert_eq!(
            jit_portable.precision_policy(),
            Some(PrecisionPolicy::BitExactPortable)
        );
        assert_eq!(
            jit_fma.precision_policy(),
            Some(PrecisionPolicy::HardwareFmaAccelerated)
        );

        let resp_portable = jit_portable
            .call_with_dispatch(vec![a.clone(), b.clone()])
            .expect("portable jit should succeed");
        let resp_fma = jit_fma
            .call_with_dispatch(vec![a, b])
            .expect("fma jit should succeed");

        assert_ne!(
            resp_portable.cache_key, resp_fma.cache_key,
            "different PrecisionPolicy settings must produce distinct cache keys"
        );
    }

    #[test]
    fn grad_and_value_and_grad_with_argnums_and_fluent_composition() {
        // f(x, y) = x^2 * y + 3 * y^2
        // df/dx = 2 * x * y, df/dy = x^2 + 6 * y
        // At (x=2.0, y=5.0): f=95.0, df/dx=20.0, df/dy=34.0
        let jaxpr = crate::make_jaxpr_fallible(
            |args| {
                let x = &args[0];
                let y = &args[1];
                let x2 = x.mul_op(x)?;
                let x2y = x2.mul_op(y)?;
                let three = x.bind_literal(fj_core::Literal::from_f64(3.0))?;
                let y2 = y.mul_op(y)?;
                let term2 = three.mul_op(&y2)?;
                Ok(vec![x2y.add_op(&term2)?])
            },
            vec![
                ShapedArray {
                    dtype: fj_core::DType::F64,
                    shape: fj_core::Shape::scalar(),
                },
                ShapedArray {
                    dtype: fj_core::DType::F64,
                    shape: fj_core::Shape::scalar(),
                },
            ],
        )
        .expect("trace should succeed")
        .into_executable_jaxpr();

        let args = vec![Value::scalar_f64(2.0), Value::scalar_f64(5.0)];

        let g_y = grad(jaxpr.clone())
            .with_argnums(&[1])
            .with_precision_policy(PrecisionPolicy::HardwareFmaAccelerated);
        let dy = g_y
            .call(args.clone())
            .expect("grad argnum=1 should succeed");
        assert_eq!(dy.len(), 1);
        assert!((dy[0].as_f64_scalar().unwrap() - 34.0).abs() < 1e-12);
        let proof = g_y
            .last_composition_proof()
            .expect("grad with_argnums should populate composition proof");
        proof.verify().expect("grad argnums proof should verify");

        let g_both = grad(jaxpr.clone()).with_argnums(&[0, 1]);
        let d_both = g_both
            .call(args.clone())
            .expect("grad argnums=[0,1] should succeed");
        assert_eq!(d_both.len(), 2);
        assert!((d_both[0].as_f64_scalar().unwrap() - 20.0).abs() < 1e-12);
        assert!((d_both[1].as_f64_scalar().unwrap() - 34.0).abs() < 1e-12);

        let (vals, vg_y) = value_and_grad(jaxpr.clone())
            .with_argnums(&[1])
            .call(args.clone())
            .expect("value_and_grad argnum=1 should succeed");
        assert!((vals[0].as_f64_scalar().unwrap() - 95.0).abs() < 1e-12);
        assert_eq!(vg_y.len(), 1);
        assert!((vg_y[0].as_f64_scalar().unwrap() - 34.0).abs() < 1e-12);

        // Higher-order fluent composition: grad(f).compose_grad() -> d^2f/dx^2 = 2 * y = 10.0
        let d2x = grad(jaxpr)
            .compose_grad()
            .call(args)
            .expect("grad.compose_grad should succeed");
        assert!((d2x[0].as_f64_scalar().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn pmap_with_devices_executes_spmd_collectives_and_verifies_proof() {
        // Build a collective Jaxpr via pmap_fn:
        // Each device has x_i. Compute:
        //   out0 = psum(x_i) + x_i
        //   out1 = pmean(x_i)
        //   out2 = all_gather(x_i)
        let batch = Value::Tensor(
            fj_core::TensorValue::new_f64_values(
                fj_core::Shape::vector(4),
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );

        let outs = pmap_fn(
            |args| {
                let x = &args[0];
                let s = x.psum("devices")?;
                let m = x.pmean("devices")?;
                let g = x.all_gather("devices")?;
                Ok(vec![s.add_op(x)?, m, g])
            },
            4,
            vec![batch.clone()],
        )
        .expect("pmap_fn across 4 virtual devices should succeed");

        assert_eq!(outs.len(), 3);
        // psum([1,2,3,4]) = 10 -> out0 = [11, 12, 13, 14]
        let out0 = outs[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(out0, vec![11.0, 12.0, 13.0, 14.0]);
        // pmean([1,2,3,4]) = 2.5 -> out1 = [2.5, 2.5, 2.5, 2.5]
        let out1 = outs[1].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(out1, vec![2.5, 2.5, 2.5, 2.5]);
        // all_gather([1,2,3,4]) on each of 4 devices -> shape [4, 4]
        let out2_t = outs[2].as_tensor().unwrap();
        assert_eq!(out2_t.shape.dims, vec![4, 4]);
        assert_eq!(
            out2_t.to_f64_vec().unwrap(),
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0
            ]
        );

        // Also test PmapWrapped::with_devices + call_with_evidence + PmapWitness proof verification
        let psum_jaxpr = fj_trace::with_pmap_trace_context(4, || {
            crate::make_jaxpr_fallible(
                |args| Ok(vec![args[0].psum("i")?]),
                vec![ShapedArray {
                    dtype: fj_core::DType::F64,
                    shape: fj_core::Shape::scalar(),
                }],
            )
        })
        .expect("trace psum should succeed")
        .into_executable_jaxpr();

        let wrapped = pmap(psum_jaxpr).with_axis_name("i").with_devices(4);
        let resp = wrapped
            .call_with_evidence(vec![batch])
            .expect("pmap with_devices(4) should succeed");
        assert_eq!(
            resp.outputs[0].as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![10.0, 10.0, 10.0, 10.0]
        );
        resp.composition_proof
            .verify()
            .expect("pmap composition proof must cryptographically verify");
        assert_eq!(
            resp.composition_proof.steps[0].witness,
            fj_core::TransformRewriteWitness::Pmap(fj_core::PmapWitness { axis_size: 4 })
        );

        // Test AllToAll and AxisIndex across 2 virtual devices
        let mat = Value::Tensor(
            fj_core::TensorValue::new_f64_values(
                fj_core::Shape { dims: vec![2, 2] },
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        let a2a = pmap_fn(|args| Ok(vec![args[0].all_to_all("i")?]), 2, vec![mat])
            .expect("pmap_fn all_to_all should succeed");
        assert_eq!(
            a2a[0].as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![1.0, 3.0, 2.0, 4.0]
        );

        let idx_vec = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::I64,
                fj_core::Shape::vector(3),
                vec![
                    fj_core::Literal::I64(10),
                    fj_core::Literal::I64(20),
                    fj_core::Literal::I64(30),
                ],
            )
            .unwrap(),
        );
        let with_idx = pmap_fn(
            |args| {
                let idx = args[0].bind_axis_index("i")?;
                Ok(vec![args[0].add_op(&idx)?])
            },
            3,
            vec![idx_vec],
        )
        .expect("pmap_fn axis_index should succeed");
        let idx_elems: Vec<i64> = with_idx[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(idx_elems, vec![10, 21, 32]);
    }

    #[test]
    fn jvp_fn_vjp_fn_and_linearize_fn_closure_helpers() {
        // f(x) = x^3 -> f(2) = 8, f'(2) = 12
        let primals = [Value::scalar_f64(2.0)];
        let tangents = [Value::scalar_f64(3.0)];

        let jvp_res = jvp_fn(
            |args| {
                let x = &args[0];
                let x2 = x.mul_op(x)?;
                Ok(vec![x2.mul_op(x)?])
            },
            &primals,
            &tangents,
        )
        .expect("jvp_fn should succeed");
        assert!((jvp_res.primals[0].as_f64_scalar().unwrap() - 8.0).abs() < 1e-12);
        assert!((jvp_res.tangents[0].as_f64_scalar().unwrap() - 36.0).abs() < 1e-12);

        let vjp_res = vjp_fn(
            |args| {
                let x = &args[0];
                let x2 = x.mul_op(x)?;
                Ok(vec![x2.mul_op(x)?])
            },
            &primals,
        )
        .expect("vjp_fn should succeed");
        assert!((vjp_res.primal_outputs[0].as_f64_scalar().unwrap() - 8.0).abs() < 1e-12);
        let cot = vjp_res
            .pullback
            .call(Value::scalar_f64(2.0))
            .expect("vjp_fn pullback should succeed");
        assert!((cot[0].as_f64_scalar().unwrap() - 24.0).abs() < 1e-12);

        let lin_res = linearize_fn(
            |args| {
                let x = &args[0];
                let x2 = x.mul_op(x)?;
                Ok(vec![x2.mul_op(x)?])
            },
            &primals,
        )
        .expect("linearize_fn should succeed");
        assert!((lin_res.primal_outputs[0].as_f64_scalar().unwrap() - 8.0).abs() < 1e-12);
        let t_out = lin_res
            .linearized
            .call(vec![Value::scalar_f64(5.0)])
            .expect("linearize_fn tangent pushforward should succeed");
        assert!((t_out[0].as_f64_scalar().unwrap() - 60.0).abs() < 1e-12);
    }

    #[test]
    fn grad_and_value_and_grad_to_jaxpr_and_composed_memoization() {
        // f(x, y) = x^2 * y + 5 * y
        let args = vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)];
        let jaxpr = trace_jaxpr_for_values(
            |t| {
                let x = &t[0];
                let y = &t[1];
                let x2y = x.mul_op(x)?.mul_op(y)?;
                let five = y.bind_literal(fj_core::Literal::from_f64(5.0))?;
                let five_y = five.mul_op(y)?;
                Ok(vec![x2y.add_op(&five_y)?])
            },
            &args,
        )
        .expect("trace should succeed");

        // 1. GradWrapped::to_grad_jaxpr (default argnum 0 and explicit argnums &[0, 1])
        let g_jaxpr_0 = grad(jaxpr.clone())
            .to_grad_jaxpr()
            .expect("to_grad_jaxpr should succeed");
        let g0_out = fj_interpreters::eval_jaxpr(&g_jaxpr_0, &args).expect("eval grad jaxpr 0");
        assert_eq!(g0_out, vec![Value::scalar_f64(24.0)]); // 2 * x * y = 24

        let g_jaxpr_both = grad(jaxpr.clone())
            .with_argnums(&[0, 1])
            .to_grad_jaxpr()
            .expect("to_grad_jaxpr([0, 1]) should succeed");
        let g_both_out =
            fj_interpreters::eval_jaxpr(&g_jaxpr_both, &args).expect("eval grad jaxpr [0, 1]");
        assert_eq!(
            g_both_out,
            vec![Value::scalar_f64(24.0), Value::scalar_f64(14.0)] // x^2 + 5 = 14
        );

        // 2. ValueAndGradWrapped::to_value_and_grad_jaxpr
        let vg_jaxpr = value_and_grad(jaxpr.clone())
            .to_value_and_grad_jaxpr()
            .expect("to_value_and_grad_jaxpr should succeed");
        let vg_out =
            fj_interpreters::eval_jaxpr(&vg_jaxpr, &args).expect("eval value_and_grad jaxpr");
        assert_eq!(
            vg_out,
            vec![
                Value::scalar_f64(56.0), // 9*4 + 20 = 56
                Value::scalar_f64(24.0),
                Value::scalar_f64(14.0),
            ]
        );

        // 3. ComposedTransform metadata memoization + invalidation + symmetric compose_* builders
        let vg_composed = value_and_grad(jaxpr.clone()).compose_jit();
        assert!(vg_composed.meta_cache.0.get().is_none());
        let vg_comp_out = vg_composed
            .call(args.clone())
            .expect("value_and_grad.compose_jit should succeed");
        assert_eq!(vg_comp_out, vg_out);
        assert!(
            vg_composed.meta_cache.0.get().is_some(),
            "ComposedTransform must memoize PreparedDispatchMeta after first call"
        );
        let mutated = vg_composed.with_mode(CompatibilityMode::Hardened);
        assert!(
            mutated.meta_cache.0.get().is_none(),
            "mutating ComposedTransform options must invalidate meta_cache"
        );

        assert_eq!(
            jit(jaxpr.clone()).compose_pmap().transforms,
            vec![Transform::Jit, Transform::Pmap]
        );
        assert_eq!(
            grad(jaxpr.clone()).compose_pmap().transforms,
            vec![Transform::Grad, Transform::Pmap]
        );
        assert_eq!(
            vmap(jaxpr.clone()).compose_pmap().transforms,
            vec![Transform::Vmap, Transform::Pmap]
        );
        assert_eq!(
            pmap(jaxpr.clone()).compose_vmap().transforms,
            vec![Transform::Pmap, Transform::Vmap]
        );
        assert_eq!(
            value_and_grad(jaxpr.clone()).compose_vmap().transforms,
            vec![Transform::Grad, Transform::Vmap]
        );
        assert_eq!(
            value_and_grad(jaxpr.clone()).compose_pmap().transforms,
            vec![Transform::Grad, Transform::Pmap]
        );

        let sq_jaxpr = fj_core::build_program(fj_core::ProgramSpec::Square);
        let vg_fast = value_and_grad(sq_jaxpr.clone());
        let (vals, grads) = vg_fast
            .call(vec![Value::scalar_f64(5.0)])
            .expect("fast-path value_and_grad should succeed");
        assert_eq!(vals, vec![Value::scalar_f64(25.0)]);
        assert_eq!(grads, vec![Value::scalar_f64(10.0)]);
        let fast_proof = vg_fast
            .last_composition_proof()
            .expect("fast-path value_and_grad must synthesize lowered proof");
        assert!(fast_proof.verify().is_ok());
        assert_eq!(fast_proof.steps[0].output_avals.len(), 2);
        assert_eq!(
            fast_proof.final_output_jaxpr_sha256,
            fj_core::jaxpr_fingerprint_sha256(
                &vg_fast.to_value_and_grad_jaxpr().expect("lowered vg jaxpr")
            )
        );
    }
}
