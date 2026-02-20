//! FJ-P2C-007 Differential Oracle + Metamorphic + Adversarial Validation
//!
//! Tests that the FFI call interface maintains correctness, safety, and
//! behavioral parity across the unsafe boundary.

#![allow(unsafe_code)]

use fj_core::{DType, Literal, Shape, TensorValue, Value};
use fj_ffi::{
    CallbackRegistry, FfiBuffer, FfiCall, FfiCallback, FfiError, FfiRegistry, buffer_to_value,
    value_to_buffer,
};

// ======================== Mock FFI functions ========================

/// Doubles an f64 scalar: out = in * 2
unsafe extern "C" fn ffi_double(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        *dst = *src * 2.0;
    }
    0
}

/// Returns the sum of a 3-element f64 vector.
unsafe extern "C" fn ffi_sum3(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        *dst = *src.add(0) + *src.add(1) + *src.add(2);
    }
    0
}

/// No-op: returns 0 (success) without reading or writing anything.
unsafe extern "C" fn ffi_noop(
    _inputs: *const *const u8,
    _input_count: usize,
    _outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    0
}

/// Always returns error code 7.
unsafe extern "C" fn ffi_error(
    _inputs: *const *const u8,
    _input_count: usize,
    _outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    7
}

/// Elementwise negate for a 3-element f64 vector.
unsafe extern "C" fn ffi_negate_vec(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        for i in 0..3 {
            *dst.add(i) = -(*src.add(i));
        }
    }
    0
}

fn make_registry() -> FfiRegistry {
    let reg = FfiRegistry::new();
    reg.register("double", ffi_double).unwrap();
    reg.register("sum3", ffi_sum3).unwrap();
    reg.register("noop", ffi_noop).unwrap();
    reg.register("error", ffi_error).unwrap();
    reg.register("negate_vec", ffi_negate_vec).unwrap();
    reg
}

// ======================== Oracle Tests ========================

/// Oracle: FFI double matches Rust computation.
#[test]
fn oracle_ffi_double_matches_rust() {
    let reg = make_registry();
    let call = FfiCall::new("double");

    for val in [
        0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        f64::MIN_POSITIVE,
        f64::MAX / 2.0,
    ] {
        let input = FfiBuffer::new(val.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
        call.invoke(&reg, &[input], &mut outputs).unwrap();

        let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        let result = f64::from_ne_bytes(result_bytes);
        let expected = val * 2.0;
        assert_eq!(
            result, expected,
            "double({val}) should be {expected}, got {result}"
        );
    }
}

/// Oracle: FFI sum3 matches Rust computation.
#[test]
fn oracle_ffi_sum3_matches_rust() {
    let reg = make_registry();
    let call = FfiCall::new("sum3");

    let values = [10.0f64, 20.0, 30.0];
    let mut input_data = Vec::new();
    for &v in &values {
        input_data.extend_from_slice(&v.to_ne_bytes());
    }
    let input = FfiBuffer::new(input_data, vec![3], DType::F64).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call.invoke(&reg, &[input], &mut outputs).unwrap();

    let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
    let result = f64::from_ne_bytes(result_bytes);
    let expected: f64 = values.iter().sum();
    assert_eq!(result, expected);
}

/// Oracle: Value→Buffer→FFI→Buffer→Value round-trip preserves semantics.
#[test]
fn oracle_value_roundtrip_through_ffi() {
    let reg = make_registry();
    let call = FfiCall::new("double");

    let input_val = Value::Scalar(Literal::F64Bits(5.0f64.to_bits()));
    let input_buf = value_to_buffer(&input_val).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call.invoke(&reg, &[input_buf], &mut outputs).unwrap();

    let output_val = buffer_to_value(&outputs[0]).unwrap();
    let expected = Value::Scalar(Literal::F64Bits(10.0f64.to_bits()));
    assert_eq!(output_val, expected);
}

/// Oracle: FFI negate vector matches Rust computation.
#[test]
fn oracle_ffi_negate_vec_matches_rust() {
    let reg = make_registry();
    let call = FfiCall::new("negate_vec");

    let input_val = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape { dims: vec![3] },
        elements: vec![
            Literal::F64Bits(1.0f64.to_bits()),
            Literal::F64Bits(2.0f64.to_bits()),
            Literal::F64Bits(3.0f64.to_bits()),
        ],
    });

    let input_buf = value_to_buffer(&input_val).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![3], DType::F64).unwrap()];
    call.invoke(&reg, &[input_buf], &mut outputs).unwrap();

    let output_val = buffer_to_value(&outputs[0]).unwrap();
    let expected = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape { dims: vec![3] },
        elements: vec![
            Literal::F64Bits((-1.0f64).to_bits()),
            Literal::F64Bits((-2.0f64).to_bits()),
            Literal::F64Bits((-3.0f64).to_bits()),
        ],
    });
    assert_eq!(output_val, expected);
}

/// Oracle: error propagation is structured and lossless.
#[test]
fn oracle_error_propagation_is_structured() {
    let reg = make_registry();
    let call = FfiCall::new("error");
    let err = call.invoke(&reg, &[], &mut []).unwrap_err();
    match err {
        FfiError::CallFailed {
            target,
            code,
            message,
        } => {
            assert_eq!(target, "error");
            assert_eq!(code, 7);
            assert!(!message.is_empty());
        }
        other => panic!("expected CallFailed, got: {other}"),
    }
}

// ======================== Metamorphic Tests ========================

/// Metamorphic: marshal(unmarshal(tensor)) == tensor (identity).
#[test]
fn metamorphic_marshal_roundtrip_is_identity() {
    let tensors = vec![
        Value::Scalar(Literal::F64Bits(42.0f64.to_bits())),
        Value::Scalar(Literal::I64(-99)),
        Value::Scalar(Literal::Bool(true)),
        Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape { dims: vec![2, 2] },
            elements: vec![
                Literal::F64Bits(1.0f64.to_bits()),
                Literal::F64Bits(2.0f64.to_bits()),
                Literal::F64Bits(3.0f64.to_bits()),
                Literal::F64Bits(4.0f64.to_bits()),
            ],
        }),
    ];

    for val in &tensors {
        let buf = value_to_buffer(val).unwrap();
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, &restored, "roundtrip failed for {val:?}");
    }
}

/// Metamorphic: FFI error does not corrupt caller state.
#[test]
fn metamorphic_error_does_not_corrupt_state() {
    let reg = make_registry();

    // First: successful call
    let call_ok = FfiCall::new("double");
    let input = FfiBuffer::new(5.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
    let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call_ok.invoke(&reg, &[input], &mut outputs).unwrap();
    let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
    let result_before = f64::from_ne_bytes(result_bytes);

    // Then: failing call (should not corrupt registry or caller)
    let call_err = FfiCall::new("error");
    let _ = call_err.invoke(&reg, &[], &mut []);

    // Verify: successful call still works identically
    let input2 = FfiBuffer::new(5.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
    let mut outputs2 = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call_ok.invoke(&reg, &[input2], &mut outputs2).unwrap();
    let result_bytes2: [u8; 8] = outputs2[0].as_bytes().try_into().unwrap();
    let result_after = f64::from_ne_bytes(result_bytes2);

    assert_eq!(result_before, result_after);
}

/// Metamorphic: FFI with empty payload succeeds (no-op).
#[test]
fn metamorphic_empty_payload_noop() {
    let reg = make_registry();
    let call = FfiCall::new("noop");
    call.invoke(&reg, &[], &mut []).unwrap();
}

/// Metamorphic: double(double(x)) == x * 4.
#[test]
fn metamorphic_double_composition() {
    let reg = make_registry();
    let call = FfiCall::new("double");

    let x = 7.0f64;
    let input1 = FfiBuffer::new(x.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
    let mut mid = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call.invoke(&reg, &[input1], &mut mid).unwrap();

    let mut final_out = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
    call.invoke(&reg, &[mid[0].clone()], &mut final_out)
        .unwrap();

    let result_bytes: [u8; 8] = final_out[0].as_bytes().try_into().unwrap();
    let result = f64::from_ne_bytes(result_bytes);
    assert_eq!(result, x * 4.0);
}

// ======================== Adversarial Tests ========================

/// Adversarial: call to unregistered target.
#[test]
fn adversarial_unregistered_target() {
    let reg = FfiRegistry::new();
    let call = FfiCall::new("nonexistent");
    let err = call.invoke(&reg, &[], &mut []).unwrap_err();
    assert!(matches!(err, FfiError::TargetNotFound { .. }));
}

/// Adversarial: duplicate registration.
#[test]
fn adversarial_duplicate_registration() {
    let reg = FfiRegistry::new();
    reg.register("dup", ffi_noop).unwrap();
    let err = reg.register("dup", ffi_noop).unwrap_err();
    assert!(matches!(err, FfiError::DuplicateTarget { .. }));
}

/// Adversarial: buffer size mismatch at construction.
#[test]
fn adversarial_buffer_size_mismatch() {
    let err = FfiBuffer::new(vec![0u8; 4], vec![3], DType::F64).unwrap_err();
    assert!(matches!(err, FfiError::BufferMismatch { .. }));
}

/// Adversarial: concurrent FFI calls from multiple threads.
#[test]
fn adversarial_concurrent_ffi_calls() {
    use std::sync::Arc;
    use std::thread;

    let reg = Arc::new(make_registry());
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let reg = Arc::clone(&reg);
            thread::spawn(move || {
                let call = FfiCall::new("double");
                let val = i as f64;
                let input = FfiBuffer::new(val.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
                let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
                call.invoke(&reg, &[input], &mut outputs).unwrap();

                let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
                let result = f64::from_ne_bytes(result_bytes);
                assert_eq!(result, val * 2.0);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Adversarial: callback that returns error.
#[test]
fn adversarial_callback_error() {
    let mut reg = CallbackRegistry::new();
    reg.register(FfiCallback::pure_callback("fail_cb", |_| {
        Err(FfiError::CallFailed {
            target: "fail_cb".to_string(),
            code: 1,
            message: "test failure".to_string(),
        })
    }))
    .unwrap();

    let cb = reg.get("fail_cb").unwrap();
    let err = cb.call(&[]).unwrap_err();
    assert!(matches!(err, FfiError::CallFailed { code: 1, .. }));
}

/// Adversarial: callback with wrong arity (passes more args than expected).
#[test]
fn adversarial_callback_unexpected_arity() {
    let mut reg = CallbackRegistry::new();
    reg.register(FfiCallback::pure_callback("expects_one", |args| {
        if args.len() != 1 {
            return Err(FfiError::CallFailed {
                target: "expects_one".to_string(),
                code: 2,
                message: format!("expected 1 arg, got {}", args.len()),
            });
        }
        Ok(args.to_vec())
    }))
    .unwrap();

    let cb = reg.get("expects_one").unwrap();
    let err = cb
        .call(&[
            Value::Scalar(Literal::I64(1)),
            Value::Scalar(Literal::I64(2)),
        ])
        .unwrap_err();
    assert!(matches!(err, FfiError::CallFailed { code: 2, .. }));
}
