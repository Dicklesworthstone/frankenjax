//! Thread-safe FFI target registry.
//!
//! All external function pointers must be registered before use.
//! Registration validates non-null pointers and unique names.
//! The registry is immutable after first use (no deregistration in V1).

use std::collections::HashMap;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::error::FfiError;

/// The C function signature expected by the FFI boundary.
///
/// Parameters:
/// - `inputs`: array of const pointers to input buffer data
/// - `input_count`: number of input buffers
/// - `outputs`: array of mutable pointers to output buffer data
/// - `output_count`: number of output buffers
///
/// Returns: 0 on success, non-zero error code on failure.
pub type FfiFnPtr = unsafe extern "C" fn(
    inputs: *const *const u8,
    input_count: usize,
    outputs: *const *mut u8,
    output_count: usize,
) -> i32;

/// A registered FFI target with metadata.
#[derive(Clone)]
pub struct FfiTarget {
    pub name: String,
    pub fn_ptr: FfiFnPtr,
}

impl std::fmt::Debug for FfiTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiTarget")
            .field("name", &self.name)
            .field("fn_ptr", &"<extern \"C\" fn>")
            .finish()
    }
}

/// Thread-safe registry of FFI targets.
///
/// Registration is global and persistent for the process lifetime.
/// All operations are thread-safe via `RwLock`.
pub struct FfiRegistry {
    targets: RwLock<HashMap<String, FfiTarget>>,
}

impl FfiRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        FfiRegistry {
            targets: RwLock::new(HashMap::new()),
        }
    }

    fn targets_read(&self) -> RwLockReadGuard<'_, HashMap<String, FfiTarget>> {
        match self.targets.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn targets_write(&self) -> RwLockWriteGuard<'_, HashMap<String, FfiTarget>> {
        match self.targets.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Register an FFI target by name.
    ///
    /// # Errors
    /// - `FfiError::NullPointer` if `fn_ptr` is null
    /// - `FfiError::DuplicateTarget` if `name` is already registered
    pub fn register(&self, name: &str, fn_ptr: FfiFnPtr) -> Result<(), FfiError> {
        // Null check: fn_ptr as a function pointer cannot be null in safe Rust,
        // but we check the address to be defensive against transmuted pointers.
        if (fn_ptr as usize) == 0 {
            return Err(FfiError::NullPointer {
                target_name: name.to_string(),
            });
        }

        let mut targets = self.targets_write();
        if targets.contains_key(name) {
            return Err(FfiError::DuplicateTarget {
                name: name.to_string(),
            });
        }

        targets.insert(
            name.to_string(),
            FfiTarget {
                name: name.to_string(),
                fn_ptr,
            },
        );
        Ok(())
    }

    /// Look up a registered target by name.
    pub fn get(&self, name: &str) -> Result<FfiTarget, FfiError> {
        let targets = self.targets_read();
        targets
            .get(name)
            .cloned()
            .ok_or_else(|| FfiError::TargetNotFound {
                name: name.to_string(),
                available: targets.keys().cloned().collect(),
            })
    }

    /// List all registered target names.
    pub fn registered_names(&self) -> Vec<String> {
        let targets = self.targets_read();
        targets.keys().cloned().collect()
    }

    /// Number of registered targets.
    pub fn len(&self) -> usize {
        let targets = self.targets_read();
        targets.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for FfiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FfiRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let targets = self.targets_read();
        f.debug_struct("FfiRegistry")
            .field("target_count", &targets.len())
            .field("targets", &targets.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Trivial FFI function for testing: always returns 0 (success).
    unsafe extern "C" fn mock_success(
        _inputs: *const *const u8,
        _input_count: usize,
        _outputs: *const *mut u8,
        _output_count: usize,
    ) -> i32 {
        0
    }

    /// Mock FFI function that returns error code 42.
    unsafe extern "C" fn mock_error(
        _inputs: *const *const u8,
        _input_count: usize,
        _outputs: *const *mut u8,
        _output_count: usize,
    ) -> i32 {
        42
    }

    #[test]
    fn registry_new_is_empty() {
        let reg = FfiRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn registry_register_and_get() -> Result<(), Box<dyn std::error::Error>> {
        let reg = FfiRegistry::new();
        reg.register("add_vectors", mock_success)?;
        let target = reg.get("add_vectors")?;
        assert_eq!(target.name, "add_vectors");
        assert_eq!(reg.len(), 1);
        Ok(())
    }

    #[test]
    fn registry_duplicate_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let reg = FfiRegistry::new();
        reg.register("my_fn", mock_success)?;
        let err = match reg.register("my_fn", mock_error) {
            Ok(()) => return Err("duplicate registration unexpectedly succeeded".into()),
            Err(err) => err,
        };
        assert!(matches!(err, FfiError::DuplicateTarget { name } if name == "my_fn"));
        Ok(())
    }

    #[test]
    fn registry_target_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let reg = FfiRegistry::new();
        reg.register("exists", mock_success)?;
        let err = match reg.get("nonexistent") {
            Ok(target) => return Err(format!("unexpectedly found target: {target:?}").into()),
            Err(err) => err,
        };
        match err {
            FfiError::TargetNotFound { name, available } => {
                assert_eq!(name, "nonexistent");
                assert!(available.contains(&"exists".to_string()));
                Ok(())
            }
            other => Err(format!("expected TargetNotFound, got: {other}").into()),
        }
    }

    #[test]
    fn registry_recovers_from_poisoned_lock() -> Result<(), Box<dyn std::error::Error>> {
        let reg = Arc::new(FfiRegistry::new());
        let reg_for_thread = Arc::clone(&reg);

        let join = std::thread::spawn(move || {
            let _guard = match reg_for_thread.targets.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::panic::panic_any("poison registry lock");
        })
        .join();

        assert!(join.is_err());
        reg.register("after_poison", mock_success)?;

        let target = reg.get("after_poison")?;
        assert_eq!(target.name, "after_poison");
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.registered_names(), vec!["after_poison".to_owned()]);
        assert!(format!("{reg:?}").contains("after_poison"));
        Ok(())
    }

    #[test]
    fn registry_registered_names() -> Result<(), Box<dyn std::error::Error>> {
        let reg = FfiRegistry::new();
        reg.register("alpha", mock_success)?;
        reg.register("beta", mock_error)?;
        let mut names = reg.registered_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
        Ok(())
    }

    #[test]
    fn registry_multiple_registrations() -> Result<(), Box<dyn std::error::Error>> {
        let reg = FfiRegistry::new();
        for i in 0..10 {
            reg.register(&format!("fn_{i}"), mock_success)?;
        }
        assert_eq!(reg.len(), 10);
        Ok(())
    }
}
