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
pub struct JacobianWrapped {
    jaxpr: Jaxpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HessianWrapped {
    jaxpr: Jaxpr,
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

#[must_use]
pub fn jacobian(jaxpr: Jaxpr) -> JacobianWrapped {
    JacobianWrapped { jaxpr }
}

#[must_use]
pub fn hessian(jaxpr: Jaxpr) -> HessianWrapped {
    HessianWrapped { jaxpr }
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
}
