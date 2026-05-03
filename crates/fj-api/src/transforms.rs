#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fj_core::{CompatibilityMode, Jaxpr, TraceTransformLedger, Transform, Value};
use fj_dispatch::{DispatchRequest, dispatch};

use crate::errors::ApiError;

#[derive(Debug, Clone, PartialEq)]
pub struct JitWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VmapWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
    in_axes: Option<String>,
    out_axes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueAndGradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomVjpWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CustomJvpWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JacobianWrapped {
    jaxpr: Jaxpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HessianWrapped {
    jaxpr: Jaxpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposedTransform {
    jaxpr: Jaxpr,
    transforms: Vec<Transform>,
    backend: String,
    mode: CompatibilityMode,
    compile_options: BTreeMap<String, String>,
}

#[must_use]
pub fn jit(jaxpr: Jaxpr) -> JitWrapped {
    JitWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

#[must_use]
pub fn grad(jaxpr: Jaxpr) -> GradWrapped {
    GradWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

#[must_use]
pub fn vmap(jaxpr: Jaxpr) -> VmapWrapped {
    VmapWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
        in_axes: None,
        out_axes: None,
    }
}

#[must_use]
pub fn value_and_grad(jaxpr: Jaxpr) -> ValueAndGradWrapped {
    ValueAndGradWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

/// Attach function-level custom VJP callbacks to a Jaxpr.
///
/// The forward callback returns `(primal_outputs, residuals)`. The backward
/// callback receives those residuals plus the output cotangent and returns one
/// cotangent per input. The rule is keyed by the Jaxpr canonical fingerprint, so
/// existing `grad`, `value_and_grad`, and composed `jit(grad(...))` paths use it
/// for matching Jaxprs.
#[must_use]
pub fn custom_vjp<Fwd, Bwd>(jaxpr: Jaxpr, forward: Fwd, backward: Bwd) -> CustomVjpWrapped
where
    Fwd: Fn(&[Value]) -> Result<(Vec<Value>, Vec<Value>), fj_ad::AdError> + Send + Sync + 'static,
    Bwd: Fn(&[Value], &Value) -> Result<Vec<Value>, fj_ad::AdError> + Send + Sync + 'static,
{
    let expected_outputs = jaxpr.outvars.len();
    fj_ad::register_custom_jaxpr_vjp(&jaxpr, move |primals, primal_outputs, cotangent| {
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
    });

    CustomVjpWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

/// Attach a function-level custom JVP callback to a Jaxpr.
///
/// The callback returns `(primal_outputs, tangent_outputs)`. The rule is keyed
/// by the Jaxpr canonical fingerprint, so higher-level API paths that call
/// forward-mode AD, such as `jacobian`, use it for matching Jaxprs.
#[must_use]
pub fn custom_jvp<F>(jaxpr: Jaxpr, rule: F) -> CustomJvpWrapped
where
    F: Fn(&[Value], &[Value]) -> Result<(Vec<Value>, Vec<Value>), fj_ad::AdError>
        + Send
        + Sync
        + 'static,
{
    let expected_outputs = jaxpr.outvars.len();
    fj_ad::register_custom_jaxpr_jvp(&jaxpr, move |primals, tangents| {
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
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

#[must_use]
pub fn jacobian(jaxpr: Jaxpr) -> JacobianWrapped {
    JacobianWrapped { jaxpr }
}

#[must_use]
pub fn hessian(jaxpr: Jaxpr) -> HessianWrapped {
    HessianWrapped { jaxpr }
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
    let jaxpr_clone = jaxpr.clone();
    fj_ad::register_custom_jaxpr_vjp(&jaxpr, move |primals, _primal_outputs, cotangent| {
        fj_ad::grad_jaxpr_with_cotangent(&jaxpr_clone, primals, cotangent)
    });

    CheckpointWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

fn build_ledger(jaxpr: Jaxpr, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(jaxpr);
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(*transform, format!("fj-api-{}-{}", transform.as_str(), idx));
    }
    ledger
}

fn dispatch_with_options(
    jaxpr: Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
    compile_options: BTreeMap<String, String>,
) -> Result<Vec<Value>, ApiError> {
    let response = dispatch(DispatchRequest {
        mode,
        ledger: build_ledger(jaxpr, transforms),
        args,
        backend: backend.to_owned(),
        compile_options,
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })?;
    Ok(response.outputs)
}

fn dispatch_with(
    jaxpr: Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
) -> Result<Vec<Value>, ApiError> {
    dispatch_with_options(jaxpr, transforms, args, backend, mode, BTreeMap::new())
}

/// Compose transforms: `jit(grad(f))` becomes `jit(jaxpr).compose_grad()`.
#[must_use]
pub fn compose(jaxpr: Jaxpr, transforms: Vec<Transform>) -> ComposedTransform {
    ComposedTransform {
        jaxpr,
        transforms,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
        compile_options: BTreeMap::new(),
    }
}

impl JitWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Jit],
            args,
            &self.backend,
            self.mode,
        )
    }

    /// Compose: `jit(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options: BTreeMap::new(),
        }
    }

    /// Compose: `jit(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
            compile_options: BTreeMap::new(),
        }
    }
}

impl GradWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Grad],
            args,
            &self.backend,
            self.mode,
        )
    }
}

impl VmapWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set in_axes: comma-separated axis specs, e.g. "0,none,1".
    /// - Integer: batch along that axis
    /// - "none": this input is not batched (broadcast)
    #[must_use]
    pub fn with_in_axes(mut self, in_axes: &str) -> Self {
        self.in_axes = Some(in_axes.to_owned());
        self
    }

    /// Set out_axes: comma-separated axis specs for output batch dim placement.
    /// - Integer: place batch dim at that axis position
    /// - "none": output is not batched
    #[must_use]
    pub fn with_out_axes(mut self, out_axes: &str) -> Self {
        self.out_axes = Some(out_axes.to_owned());
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        let mut compile_options = BTreeMap::new();
        if let Some(ref in_axes) = self.in_axes {
            compile_options.insert("vmap_in_axes".to_owned(), in_axes.clone());
        }
        if let Some(ref out_axes) = self.out_axes {
            compile_options.insert("vmap_out_axes".to_owned(), out_axes.clone());
        }

        let response = dispatch(DispatchRequest {
            mode: self.mode,
            ledger: build_ledger(self.jaxpr.clone(), &[Transform::Vmap]),
            args,
            backend: self.backend.clone(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })?;
        Ok(response.outputs)
    }

    /// Compose: `vmap(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        let mut compile_options = BTreeMap::new();
        if let Some(in_axes) = self.in_axes {
            compile_options.insert("vmap_in_axes".to_owned(), in_axes);
        }
        if let Some(out_axes) = self.out_axes {
            compile_options.insert("vmap_out_axes".to_owned(), out_axes);
        }
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
            compile_options,
        }
    }
}

impl ComposedTransform {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn with_compile_option(mut self, key: &str, value: &str) -> Self {
        self.compile_options
            .insert(key.to_owned(), value.to_owned());
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

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with_options(
            self.jaxpr.clone(),
            &self.transforms,
            args,
            &self.backend,
            self.mode,
            self.compile_options.clone(),
        )
    }
}

impl ValueAndGradWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let outputs = dispatch_with_options(
            self.jaxpr.clone(),
            &[Transform::Grad],
            args,
            &self.backend,
            self.mode,
            compile_options,
        )?;
        let value_len = self.jaxpr.outvars.len();
        if outputs.len() < value_len + 1 {
            return Err(ApiError::EvalError {
                detail: format!(
                    "value_and_grad expected at least {} outputs, got {}",
                    value_len + 1,
                    outputs.len()
                ),
            });
        }

        let values = outputs[..value_len].to_vec();
        let gradients = outputs[value_len..].to_vec();
        Ok((values, gradients))
    }
}

impl CustomVjpWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(self.jaxpr.clone(), &[], args, &self.backend, self.mode)
    }

    #[must_use]
    pub fn grad(&self) -> GradWrapped {
        GradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
        }
    }

    #[must_use]
    pub fn value_and_grad(&self) -> ValueAndGradWrapped {
        ValueAndGradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
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
        }
    }
}

impl CustomJvpWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(self.jaxpr.clone(), &[], args, &self.backend, self.mode)
    }

    pub fn jvp_call(
        &self,
        primals: Vec<Value>,
        tangents: Vec<Value>,
    ) -> Result<fj_ad::JvpResult, ApiError> {
        fj_ad::jvp(&self.jaxpr, &primals, &tangents).map_err(ApiError::from)
    }

    #[must_use]
    pub fn jacobian(&self) -> JacobianWrapped {
        JacobianWrapped {
            jaxpr: self.jaxpr.clone(),
        }
    }
}

impl JacobianWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Value, ApiError> {
        fj_ad::jacobian_jaxpr(&self.jaxpr, &args).map_err(ApiError::from)
    }
}

impl HessianWrapped {
    pub fn call(&self, args: Vec<Value>) -> Result<Value, ApiError> {
        fj_ad::hessian_jaxpr(&self.jaxpr, &args).map_err(ApiError::from)
    }
}

impl CheckpointWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    /// Execute the checkpointed function (forward pass only).
    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(self.jaxpr.clone(), &[], args, &self.backend, self.mode)
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
        }
    }

    /// Compute both value and gradients of the checkpointed function.
    pub fn value_and_grad(&self) -> ValueAndGradWrapped {
        ValueAndGradWrapped {
            jaxpr: self.jaxpr.clone(),
            backend: self.backend.clone(),
            mode: self.mode,
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
    fn compose_empty_transforms_acts_as_eval() {
        let composed = compose(make_add_jaxpr(), vec![]);
        let result = composed
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("empty transforms should succeed");
        assert!((result[0].as_f64_scalar().unwrap() - 5.0).abs() < 1e-12);
    }
}
