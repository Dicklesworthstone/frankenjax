//! FJ-P2C-007 E2E Scenario Scripts
//!
//! End-to-end tests exercising the full FFI call interface through
//! realistic usage patterns.

#![allow(unsafe_code)]

use fj_core::{DType, Literal, Shape, TensorValue, Value};
use fj_ffi::{
    CallbackRegistry, FfiBuffer, FfiCall, FfiCallback, FfiError, FfiRegistry, buffer_to_value,
    value_to_buffer,
};

// ======================== Mock FFI functions ========================

unsafe extern "C" fn ffi_add_scalars(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let a = *((*inputs) as *const f64);
        let b = *((*inputs.add(1)) as *const f64);
        *(*outputs as *mut f64) = a + b;
    }
    0
}

unsafe extern "C" fn ffi_square_vec(
    inputs: *const *const u8,
    input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    if input_count == 0 {
        return 1;
    }
    unsafe {
        // Read first 4 bytes as u32 for element count (passed in params as second input)
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        // Square 4 elements
        for i in 0..4 {
            let val = *src.add(i);
            *dst.add(i) = val * val;
        }
    }
    0
}

unsafe extern "C" fn ffi_fail_with_code(
    _inputs: *const *const u8,
    _input_count: usize,
    _outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    -42
}

unsafe extern "C" fn ffi_identity(
    inputs: *const *const u8,
    input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    if input_count == 0 {
        return 0;
    }
    unsafe {
        // Copy 8 bytes (one f64 scalar)
        std::ptr::copy_nonoverlapping(*inputs, *outputs, 8);
    }
    0
}

// ======================== E2E Scenarios ========================

/// E2E 1: Register external function → call → verify output.
#[test]
fn e2e_ffi_basic_call() {
    let registry = FfiRegistry::new();
    registry.register("add", ffi_add_scalars).unwrap();

    let a = value_to_buffer(&Value::Scalar(Literal::F64Bits(3.0f64.to_bits()))).unwrap();
    let b = value_to_buffer(&Value::Scalar(Literal::F64Bits(4.0f64.to_bits()))).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

    let call = FfiCall::new("add");
    call.invoke(&registry, &[a, b], &mut outputs).unwrap();

    let result = buffer_to_value(&outputs[0]).unwrap();
    assert_eq!(result, Value::Scalar(Literal::F64Bits(7.0f64.to_bits())));
}

/// E2E 2: Buffer lifecycle — borrow → FFI → return → verify integrity.
#[test]
fn e2e_ffi_buffer_lifecycle() {
    let registry = FfiRegistry::new();
    registry.register("identity", ffi_identity).unwrap();

    let original = Value::Scalar(Literal::F64Bits(std::f64::consts::E.to_bits()));
    let input_buf = value_to_buffer(&original).unwrap();

    // Verify buffer state before FFI
    assert_eq!(input_buf.size(), 8);
    assert_eq!(input_buf.dtype(), DType::F64);

    let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    let call = FfiCall::new("identity");
    call.invoke(&registry, &[input_buf], &mut outputs).unwrap();

    // Verify output matches input (identity function)
    let result = buffer_to_value(&outputs[0]).unwrap();
    assert_eq!(result, original);
}

/// E2E 3: FFI error propagation — external function fails → structured error.
#[test]
fn e2e_ffi_error_propagation() {
    let registry = FfiRegistry::new();
    registry.register("fail", ffi_fail_with_code).unwrap();

    let call = FfiCall::new("fail");
    let err = call.invoke(&registry, &[], &mut []).unwrap_err();

    match err {
        FfiError::CallFailed {
            ref target,
            code,
            ref message,
        } => {
            assert_eq!(target, "fail");
            assert_eq!(code, -42);
            assert!(!message.is_empty());
            // Verify error is Display-able for logging
            let display = format!("{err}");
            assert!(display.contains("fail"));
            assert!(display.contains("-42"));
        }
        other => panic!("expected CallFailed, got: {other}"),
    }
}

/// E2E 4: Custom op dispatch — register, build vector operation, run.
#[test]
fn e2e_ffi_custom_op_dispatch() {
    let registry = FfiRegistry::new();
    registry.register("square_vec", ffi_square_vec).unwrap();

    let input_val = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape { dims: vec![4] },
        elements: vec![
            Literal::F64Bits(2.0f64.to_bits()),
            Literal::F64Bits(3.0f64.to_bits()),
            Literal::F64Bits(4.0f64.to_bits()),
            Literal::F64Bits(5.0f64.to_bits()),
        ],
    });

    let input_buf = value_to_buffer(&input_val).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![4], DType::F64).unwrap()];

    let call = FfiCall::new("square_vec");
    call.invoke(&registry, &[input_buf], &mut outputs).unwrap();

    let result = buffer_to_value(&outputs[0]).unwrap();
    let expected = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape { dims: vec![4] },
        elements: vec![
            Literal::F64Bits(4.0f64.to_bits()),
            Literal::F64Bits(9.0f64.to_bits()),
            Literal::F64Bits(16.0f64.to_bits()),
            Literal::F64Bits(25.0f64.to_bits()),
        ],
    });
    assert_eq!(result, expected);
}

/// E2E 5: Adversarial inputs — clean errors for all edge cases.
#[test]
fn e2e_ffi_adversarial_clean_errors() {
    let registry = FfiRegistry::new();

    // 5a: Call to unregistered target
    let call = FfiCall::new("nonexistent");
    let err = call.invoke(&registry, &[], &mut []).unwrap_err();
    assert!(matches!(err, FfiError::TargetNotFound { .. }));

    // 5b: Duplicate registration
    registry.register("dup", ffi_identity).unwrap();
    let err = registry.register("dup", ffi_identity).unwrap_err();
    assert!(matches!(err, FfiError::DuplicateTarget { .. }));

    // 5c: Buffer size mismatch
    let err = FfiBuffer::new(vec![0u8; 5], vec![2], DType::F64).unwrap_err();
    assert!(matches!(err, FfiError::BufferMismatch { .. }));

    // 5d: Empty-shape buffer is valid (scalar)
    let scalar_buf = FfiBuffer::new(vec![0u8; 8], vec![], DType::F64);
    assert!(scalar_buf.is_ok());

    // 5e: Zero-element tensor is valid
    let zero_buf = FfiBuffer::new(vec![], vec![0], DType::F64);
    assert!(zero_buf.is_ok());
}

/// E2E 6: Callback pipeline — pure callback + IO callback.
#[test]
fn e2e_ffi_callback_pipeline() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut cb_registry = CallbackRegistry::new();

    // Pure callback: adds 1 to scalar
    cb_registry
        .register(FfiCallback::pure_callback("add_one", |args| {
            match &args[0] {
                Value::Scalar(Literal::I64(v)) => Ok(vec![Value::Scalar(Literal::I64(v + 1))]),
                _ => Err(FfiError::CallFailed {
                    target: "add_one".to_string(),
                    code: 1,
                    message: "expected i64 scalar".to_string(),
                }),
            }
        }))
        .unwrap();

    // IO callback: counts invocations
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();
    cb_registry
        .register(FfiCallback::io_callback("counter", move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![])
        }))
        .unwrap();

    // Run pure callback
    let cb = cb_registry.get("add_one").unwrap();
    let result = cb.call(&[Value::Scalar(Literal::I64(41))]).unwrap();
    assert_eq!(result, vec![Value::Scalar(Literal::I64(42))]);

    // Run IO callback 3 times
    let io_cb = cb_registry.get("counter").unwrap();
    io_cb.call(&[]).unwrap();
    io_cb.call(&[]).unwrap();
    io_cb.call(&[]).unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}
