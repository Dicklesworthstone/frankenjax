//! Callback support for calling Rust closures during eval_jaxpr.
//!
//! Pure callbacks have no side effects and can be reordered.
//! IO callbacks are side-effecting and require effect token ordering.

use fj_core::Value;

use crate::error::FfiError;

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
}
