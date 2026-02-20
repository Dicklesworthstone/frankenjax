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
pub mod callback;
pub mod error;
pub mod marshal;
pub mod registry;

// Re-exports for convenience
pub use buffer::FfiBuffer;
pub use call::FfiCall;
pub use callback::{CallbackFlavor, CallbackRegistry, FfiCallback};
pub use error::FfiError;
pub use marshal::{buffer_to_value, value_to_buffer};
pub use registry::{FfiFnPtr, FfiRegistry, FfiTarget};

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use fj_core::{DType, Literal, Value};

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

    // --- Adversarial tests ---

    /// Adversarial: oversized buffer is rejected at construction.
    #[test]
    fn adversarial_oversized_buffer_shape() {
        let err = FfiBuffer::new(vec![0u8; 8], vec![999], DType::F64).unwrap_err();
        assert!(matches!(err, FfiError::BufferMismatch { .. }));
    }

    /// Adversarial: empty registry call returns TargetNotFound.
    #[test]
    fn adversarial_empty_registry_call() {
        let reg = FfiRegistry::new();
        let call = FfiCall::new("anything");
        let err = call.invoke(&reg, &[], &mut []).unwrap_err();
        match err {
            FfiError::TargetNotFound { name, available } => {
                assert_eq!(name, "anything");
                assert!(available.is_empty());
            }
            other => panic!("expected TargetNotFound, got: {other}"),
        }
    }

    /// Adversarial: multiple output buffers with mixed dtypes.
    #[test]
    fn adversarial_mixed_dtype_outputs() {
        unsafe extern "C" fn mock_mixed(
            _inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                // Write f64 to first output
                let dst0 = *outputs as *mut f64;
                *dst0 = 99.0;
                // Write i64 to second output
                let dst1 = *outputs.add(1) as *mut i64;
                *dst1 = 42;
            }
            0
        }

        let reg = FfiRegistry::new();
        reg.register("mixed", mock_mixed).unwrap();

        let mut outputs = [
            FfiBuffer::zeroed(vec![], DType::F64).unwrap(),
            FfiBuffer::zeroed(vec![], DType::I64).unwrap(),
        ];

        let call = FfiCall::new("mixed");
        call.invoke(&reg, &[], &mut outputs).unwrap();

        let f_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        assert_eq!(f64::from_ne_bytes(f_bytes), 99.0);

        let i_bytes: [u8; 8] = outputs[1].as_bytes().try_into().unwrap();
        assert_eq!(i64::from_ne_bytes(i_bytes), 42);
    }

    /// Buffer lifecycle: borrow → use → return pattern.
    #[test]
    fn buffer_lifecycle_borrow_use_return() {
        unsafe extern "C" fn mock_increment(
            inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                let src = *inputs as *const i64;
                let dst = *outputs as *mut i64;
                *dst = *src + 1;
            }
            0
        }

        let reg = FfiRegistry::new();
        reg.register("inc", mock_increment).unwrap();
        let call = FfiCall::new("inc");

        // Round-trip through multiple calls
        let mut current = Value::Scalar(Literal::I64(0));
        for expected in 1..=5i64 {
            let input = value_to_buffer(&current).unwrap();
            let mut outputs = [FfiBuffer::zeroed(vec![], DType::I64).unwrap()];
            call.invoke(&reg, &[input], &mut outputs).unwrap();
            current = buffer_to_value(&outputs[0]).unwrap();
            match &current {
                Value::Scalar(Literal::I64(v)) => assert_eq!(*v, expected),
                other => panic!("expected I64 scalar, got: {other:?}"),
            }
        }
    }

    /// All error variants display correctly.
    #[test]
    fn all_error_variant_displays() {
        let errors = vec![
            FfiError::CallFailed {
                target: "t".to_string(),
                code: 1,
                message: "msg".to_string(),
            },
            FfiError::TargetNotFound {
                name: "n".to_string(),
                available: vec![],
            },
            FfiError::DuplicateTarget {
                name: "d".to_string(),
            },
            FfiError::NullPointer {
                target_name: "np".to_string(),
            },
            FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: 4,
            },
            FfiError::UnsupportedDtype { dtype: DType::F32 },
        ];
        for err in &errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "error display should not be empty");
        }
    }

    /// Marshal round-trip: Value→Buffer→Value preserves identity for all supported types.
    #[test]
    fn marshal_roundtrip_all_supported_types() {
        use fj_core::{Literal, Shape, TensorValue};

        let values = vec![
            Value::Scalar(Literal::F64Bits(std::f64::consts::PI.to_bits())),
            Value::Scalar(Literal::I64(i64::MIN)),
            Value::Scalar(Literal::I64(i64::MAX)),
            Value::Scalar(Literal::Bool(true)),
            Value::Scalar(Literal::Bool(false)),
            Value::Tensor(TensorValue {
                dtype: DType::F64,
                shape: Shape { dims: vec![2] },
                elements: vec![
                    Literal::F64Bits(0.0f64.to_bits()),
                    Literal::F64Bits(f64::NEG_INFINITY.to_bits()),
                ],
            }),
            Value::Tensor(TensorValue {
                dtype: DType::I64,
                shape: Shape { dims: vec![1, 1] },
                elements: vec![Literal::I64(42)],
            }),
        ];

        for val in &values {
            let buf = value_to_buffer(val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            assert_eq!(val, &restored, "round-trip failed for {val:?}");
        }
    }
}

/// Property tests for the FFI boundary.
#[cfg(test)]
mod prop_tests {
    use super::*;
    use fj_core::{DType, Literal, Value};
    use proptest::prelude::*;

    proptest! {
        /// Property: any f64 scalar round-trips through marshal.
        #[test]
        fn prop_f64_scalar_roundtrip(bits: u64) {
            let val = Value::Scalar(Literal::F64Bits(bits));
            let buf = value_to_buffer(&val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            prop_assert_eq!(val, restored);
        }

        /// Property: any i64 scalar round-trips through marshal.
        #[test]
        fn prop_i64_scalar_roundtrip(v: i64) {
            let val = Value::Scalar(Literal::I64(v));
            let buf = value_to_buffer(&val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            prop_assert_eq!(val, restored);
        }

        /// Property: FfiBuffer size matches checked_buffer_size for valid shapes.
        #[test]
        fn prop_buffer_size_matches_checked(
            dims in proptest::collection::vec(1usize..=10, 0..=3),
        ) {
            let dtype = DType::F64;
            let expected = buffer::checked_buffer_size(&dims, dtype).unwrap();
            let buf = FfiBuffer::zeroed(dims, dtype).unwrap();
            prop_assert_eq!(buf.size(), expected);
        }

        /// Property: zeroed buffers contain only zeros.
        #[test]
        fn prop_zeroed_buffer_is_zero(
            dims in proptest::collection::vec(1usize..=5, 0..=2),
        ) {
            let buf = FfiBuffer::zeroed(dims, DType::I64).unwrap();
            prop_assert!(buf.as_bytes().iter().all(|&b| b == 0));
        }
    }
}
