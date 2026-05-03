//! Callback support for calling Rust closures during eval_jaxpr.
//!
//! Pure callbacks have no side effects and can be reordered.
//! IO callbacks are side-effecting and require effect token ordering.
//!
//! # Purity Verification
//!
//! The `Pure` vs `Io` distinction is declared by the caller, not verified.
//! This is a fundamental limitation: Rust closures can capture mutable state,
//! access globals, or perform I/O without the type system detecting it.
//!
//! To mitigate this, `PurityConfig` provides optional runtime checks:
//! - `verify_idempotency`: calls Pure callbacks twice and compares results
//! - `warn_on_pure_failure`: flags errors from Pure callbacks as suspicious
//!
//! These checks catch common impurity patterns but cannot guarantee purity.

use fj_core::Value;

use crate::error::FfiError;

/// Configuration for purity verification at runtime.
///
/// These checks cannot guarantee purity but can detect common violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PurityConfig {
    /// When enabled, Pure callbacks are invoked twice with the same inputs.
    /// If results differ, a `PurityViolation` error is returned.
    /// This catches non-deterministic callbacks (RNG, timestamps, counters).
    ///
    /// Performance cost: 2x invocations for Pure callbacks.
    pub verify_idempotency: bool,

    /// When enabled, errors from Pure callbacks are logged as warnings,
    /// since truly pure functions should not have observable failures.
    pub warn_on_pure_failure: bool,
}

impl PurityConfig {
    /// Strict mode: all verification enabled.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            verify_idempotency: true,
            warn_on_pure_failure: true,
        }
    }

    /// Permissive mode: no verification (default).
    #[must_use]
    pub fn permissive() -> Self {
        Self::default()
    }
}

/// Type alias for the boxed callback function to keep struct definitions concise.
type CallbackFn = Box<dyn Fn(&[Value]) -> Result<Vec<Value>, FfiError> + Send + Sync>;

/// Callback flavor: pure (reorderable) or IO (sequenced).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackFlavor {
    /// No side effects. Can be reordered or eliminated.
    Pure,
    /// Side-effecting. Must execute in program order via effect tokens.
    Io,
}

/// A registered callback: a Rust closure invoked during interpretation.
pub struct FfiCallback {
    name: String,
    flavor: CallbackFlavor,
    func: CallbackFn,
}

impl FfiCallback {
    /// Create a pure callback (no side effects).
    pub fn pure_callback<F>(name: &str, func: F) -> Self
    where
        F: Fn(&[Value]) -> Result<Vec<Value>, FfiError> + Send + Sync + 'static,
    {
        FfiCallback {
            name: name.to_string(),
            flavor: CallbackFlavor::Pure,
            func: Box::new(func),
        }
    }

    /// Create an IO callback (side-effecting, ordered).
    pub fn io_callback<F>(name: &str, func: F) -> Self
    where
        F: Fn(&[Value]) -> Result<Vec<Value>, FfiError> + Send + Sync + 'static,
    {
        FfiCallback {
            name: name.to_string(),
            flavor: CallbackFlavor::Io,
            func: Box::new(func),
        }
    }

    /// Name of this callback.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Flavor of this callback.
    pub fn flavor(&self) -> CallbackFlavor {
        self.flavor
    }

    /// Invoke the callback with the given arguments.
    pub fn call(&self, args: &[Value]) -> Result<Vec<Value>, FfiError> {
        (self.func)(args)
    }

    /// Invoke the callback with purity verification.
    ///
    /// For Pure callbacks with `verify_idempotency` enabled, this calls the
    /// callback twice and returns `PurityViolation` if results differ.
    pub fn call_with_config(
        &self,
        args: &[Value],
        config: &PurityConfig,
    ) -> Result<Vec<Value>, FfiError> {
        let result = (self.func)(args);

        match (&self.flavor, &result) {
            (CallbackFlavor::Pure, Err(_)) if config.warn_on_pure_failure => {
                eprintln!(
                    "[fj-ffi] warning: Pure callback '{}' returned an error, \
                     which may indicate impurity (I/O, network, file access)",
                    self.name
                );
            }
            _ => {}
        }

        if self.flavor == CallbackFlavor::Pure && config.verify_idempotency {
            let first = result?;
            let second = (self.func)(args)?;

            if first != second {
                return Err(FfiError::PurityViolation {
                    callback_name: self.name.clone(),
                    detail: "callback returned different results on identical inputs".to_string(),
                });
            }

            return Ok(first);
        }

        result
    }
}

impl std::fmt::Debug for FfiCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiCallback")
            .field("name", &self.name)
            .field("flavor", &self.flavor)
            .finish()
    }
}

/// Registry of callbacks (separate from the FFI fn pointer registry).
pub struct CallbackRegistry {
    callbacks: Vec<FfiCallback>,
}

impl CallbackRegistry {
    /// Create an empty callback registry.
    pub fn new() -> Self {
        CallbackRegistry {
            callbacks: Vec::new(),
        }
    }

    /// Register a callback.
    pub fn register(&mut self, callback: FfiCallback) -> Result<(), FfiError> {
        if self.callbacks.iter().any(|c| c.name == callback.name) {
            return Err(FfiError::DuplicateTarget {
                name: callback.name.clone(),
            });
        }
        self.callbacks.push(callback);
        Ok(())
    }

    /// Look up a callback by name.
    pub fn get(&self, name: &str) -> Result<&FfiCallback, FfiError> {
        self.callbacks
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| FfiError::TargetNotFound {
                name: name.to_string(),
                available: self.callbacks.iter().map(|c| c.name.clone()).collect(),
            })
    }

    /// Number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }
}

impl Default for CallbackRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::Literal;

    #[test]
    fn pure_callback_invocation() {
        let cb = FfiCallback::pure_callback("identity", |args| Ok(args.to_vec()));
        assert_eq!(cb.name(), "identity");
        assert_eq!(cb.flavor(), CallbackFlavor::Pure);

        let args = vec![Value::Scalar(Literal::F64Bits(42.0f64.to_bits()))];
        let result = cb.call(&args).unwrap();
        assert_eq!(result, args);
    }

    #[test]
    fn io_callback_invocation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        let cb = FfiCallback::io_callback("log_counter", move |_args| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![])
        });

        assert_eq!(cb.flavor(), CallbackFlavor::Io);
        cb.call(&[]).unwrap();
        cb.call(&[]).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn callback_registry_operations() {
        let mut reg = CallbackRegistry::new();
        assert!(reg.is_empty());

        reg.register(FfiCallback::pure_callback("cb1", |args| Ok(args.to_vec())))
            .unwrap();
        reg.register(FfiCallback::io_callback("cb2", |_| Ok(vec![])))
            .unwrap();
        assert_eq!(reg.len(), 2);

        let cb = reg.get("cb1").unwrap();
        assert_eq!(cb.flavor(), CallbackFlavor::Pure);

        let cb = reg.get("cb2").unwrap();
        assert_eq!(cb.flavor(), CallbackFlavor::Io);
    }

    #[test]
    fn callback_registry_duplicate_rejected() {
        let mut reg = CallbackRegistry::new();
        reg.register(FfiCallback::pure_callback("dup", |args| Ok(args.to_vec())))
            .unwrap();
        let err = reg
            .register(FfiCallback::pure_callback("dup", |args| Ok(args.to_vec())))
            .unwrap_err();
        assert!(matches!(err, FfiError::DuplicateTarget { name } if name == "dup"));
    }

    #[test]
    fn callback_registry_not_found() {
        let reg = CallbackRegistry::new();
        let err = reg.get("missing").unwrap_err();
        assert!(matches!(err, FfiError::TargetNotFound { .. }));
    }

    #[test]
    fn callback_error_propagation() {
        let cb = FfiCallback::pure_callback("fail_cb", |_| {
            Err(FfiError::CallFailed {
                target: "fail_cb".to_string(),
                code: 1,
                message: "intentional failure".to_string(),
            })
        });
        let err = cb.call(&[]).unwrap_err();
        assert!(matches!(err, FfiError::CallFailed { code: 1, .. }));
    }

    #[test]
    fn purity_config_default_is_permissive() {
        let config = PurityConfig::default();
        assert!(!config.verify_idempotency);
        assert!(!config.warn_on_pure_failure);
    }

    #[test]
    fn purity_config_strict_enables_all() {
        let config = PurityConfig::strict();
        assert!(config.verify_idempotency);
        assert!(config.warn_on_pure_failure);
    }

    #[test]
    fn pure_callback_passes_idempotency_check() {
        let cb = FfiCallback::pure_callback("double", |args| {
            Ok(args
                .iter()
                .map(|v| {
                    if let Value::Scalar(Literal::I64(n)) = v {
                        Value::Scalar(Literal::I64(n * 2))
                    } else {
                        v.clone()
                    }
                })
                .collect())
        });

        let args = vec![Value::Scalar(Literal::I64(21))];
        let config = PurityConfig::strict();
        let result = cb.call_with_config(&args, &config).unwrap();
        assert_eq!(result, vec![Value::Scalar(Literal::I64(42))]);
    }

    #[test]
    fn impure_callback_fails_idempotency_check() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let cb = FfiCallback::pure_callback("impure_counter", move |_| {
            let val = counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Value::Scalar(Literal::I64(val))])
        });

        let config = PurityConfig::strict();
        let err = cb.call_with_config(&[], &config).unwrap_err();
        assert!(matches!(err, FfiError::PurityViolation { .. }));
    }

    #[test]
    fn io_callback_skips_idempotency_check() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let cb = FfiCallback::io_callback("counter", move |_| {
            let val = counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Value::Scalar(Literal::I64(val))])
        });

        let config = PurityConfig::strict();
        let result = cb.call_with_config(&[], &config).unwrap();
        assert_eq!(result, vec![Value::Scalar(Literal::I64(0))]);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn permissive_config_skips_all_checks() {
        use std::sync::atomic::{AtomicI64, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let cb = FfiCallback::pure_callback("impure_but_permissive", move |_| {
            let val = counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Value::Scalar(Literal::I64(val))])
        });

        let config = PurityConfig::permissive();
        let result = cb.call_with_config(&[], &config).unwrap();
        assert_eq!(result, vec![Value::Scalar(Literal::I64(0))]);
    }
}
