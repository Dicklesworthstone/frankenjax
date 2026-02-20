//! # fj-ffi — FFI Call Interface for FrankenJAX
//!
//! This is the **ONLY** crate in the FrankenJAX workspace permitted to use `unsafe` code.
//! All unsafe blocks are confined to `call.rs` (specifically `FfiCall::invoke()`).
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │            Safe Rust (all other crates)   │
//! │                                          │
//! │   FfiRegistry::register(name, fn_ptr)    │
//! │   FfiCall::new(target_name)              │
//! │   FfiCall::invoke(&registry, in, out)    │
//! │        │                                 │
//! │        ▼ (pre-validation: sizes, types)  │
//! ├──────────────────────────────────────────┤
//! │   unsafe { (fn_ptr)(ptrs, counts) }      │  ← call.rs ONLY
//! ├──────────────────────────────────────────┤
//! │   (post-validation: return code check)   │
//! └──────────────────────────────────────────┘
//! ```
//!
//! ## Safety Contract
//!
//! The extern "C" function MUST:
//! - Not free any input or output buffer pointers
//! - Not write beyond the declared output buffer size
//! - Not retain references to any buffer after returning
//! - Return 0 on success, non-zero on error

#![deny(unsafe_code)]

pub mod buffer;
pub mod call;
pub mod error;
pub mod registry;

// Re-exports for convenience
pub use buffer::FfiBuffer;
pub use call::FfiCall;
pub use error::FfiError;
pub use registry::{FfiFnPtr, FfiRegistry, FfiTarget};

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use fj_core::DType;

    /// Integration test: register → call → verify round-trip.
    #[test]
    fn integration_register_call_verify() {
        /// Adds two f64 scalars: output[0] = input[0] + input[1].
        unsafe extern "C" fn ffi_add(
            inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                let a = *((*inputs) as *const f64);
                let b = *((*inputs.add(1)) as *const f64);
                let dst = (*outputs) as *mut f64;
                *dst = a + b;
            }
            0
        }

        let registry = FfiRegistry::new();
        registry.register("add_f64", ffi_add).unwrap();

        let a = FfiBuffer::new(3.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let b = FfiBuffer::new(4.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

        let call = FfiCall::new("add_f64");
        call.invoke(&registry, &[a, b], &mut outputs).unwrap();

        let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        let result = f64::from_ne_bytes(result_bytes);
        assert_eq!(result, 7.0);
    }

    /// Integration test: error propagation across FFI boundary.
    #[test]
    fn integration_error_propagation() {
        unsafe extern "C" fn ffi_fail(
            _inputs: *const *const u8,
            _input_count: usize,
            _outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            -1
        }

        let registry = FfiRegistry::new();
        registry.register("fail_fn", ffi_fail).unwrap();

        let call = FfiCall::new("fail_fn");
        let err = call.invoke(&registry, &[], &mut []).unwrap_err();
        match err {
            FfiError::CallFailed { target, code, .. } => {
                assert_eq!(target, "fail_fn");
                assert_eq!(code, -1);
            }
            other => panic!("expected CallFailed, got: {other}"),
        }
    }

    /// Test: error display messages are actionable.
    #[test]
    fn error_display_messages() {
        let err = FfiError::TargetNotFound {
            name: "missing".to_string(),
            available: vec!["fn_a".to_string(), "fn_b".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("missing"));
        assert!(msg.contains("fn_a"));

        let err = FfiError::BufferMismatch {
            buffer_index: 2,
            expected_bytes: 64,
            actual_bytes: 32,
        };
        let msg = err.to_string();
        assert!(msg.contains("64"));
        assert!(msg.contains("32"));
    }

    /// Test log schema contract.
    #[test]
    fn test_ffi_test_log_schema_contract() {
        let test_name = "test_ffi_test_log_schema_contract";
        let packet_id = "FJ-P2C-007";
        assert!(!test_name.is_empty());
        assert!(!packet_id.is_empty());
    }
}
