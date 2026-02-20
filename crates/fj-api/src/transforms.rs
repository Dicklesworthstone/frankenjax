#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fj_core::{CompatibilityMode, Jaxpr, TraceTransformLedger, Transform, Value};
use fj_dispatch::{DispatchRequest, dispatch};

use crate::errors::ApiError;

#[derive(Debug, Clone)]
pub struct JitWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone)]
pub struct GradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone)]
pub struct VmapWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone)]
pub struct ValueAndGradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone)]
pub struct ComposedTransform {
    jaxpr: Jaxpr,
    transforms: Vec<Transform>,
    backend: String,
    mode: CompatibilityMode,
}

pub fn jit(jaxpr: Jaxpr) -> JitWrapped {
    JitWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

pub fn grad(jaxpr: Jaxpr) -> GradWrapped {
    GradWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

pub fn vmap(jaxpr: Jaxpr) -> VmapWrapped {
    VmapWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

pub fn value_and_grad(jaxpr: Jaxpr) -> ValueAndGradWrapped {
    ValueAndGradWrapped {
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

fn dispatch_with(
    jaxpr: Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
) -> Result<Vec<Value>, ApiError> {
    let response = dispatch(DispatchRequest {
        mode,
        ledger: build_ledger(jaxpr, transforms),
        args,
        backend: backend.to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })?;
    Ok(response.outputs)
}

/// Compose transforms: `jit(grad(f))` becomes `jit(jaxpr).compose_grad()`.
pub fn compose(jaxpr: Jaxpr, transforms: Vec<Transform>) -> ComposedTransform {
    ComposedTransform {
        jaxpr,
        transforms,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

impl JitWrapped {
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

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
    pub fn compose_grad(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
        }
    }

    /// Compose: `jit(vmap(f))`.
    pub fn compose_vmap(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
        }
    }
}

impl GradWrapped {
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

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
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Vmap],
            args,
            &self.backend,
            self.mode,
        )
    }

    /// Compose: `vmap(grad(f))`.
    pub fn compose_grad(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
        }
    }
}

impl ComposedTransform {
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &self.transforms,
            args,
            &self.backend,
            self.mode,
        )
    }
}

impl ValueAndGradWrapped {
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
        let value = dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Jit],
            args.clone(),
            &self.backend,
            self.mode,
        )?;
        let gradient = dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Grad],
            args,
            &self.backend,
            self.mode,
        )?;
        Ok((value, gradient))
    }
}
