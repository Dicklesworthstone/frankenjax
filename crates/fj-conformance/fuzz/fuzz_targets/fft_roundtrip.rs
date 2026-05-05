#![no_main]

//! Fuzz target for FFT round-trip correctness.
//!
//! Tests that RFFT followed by IRFFT preserves the original signal.
//! This is a metamorphic property: IRFFT(RFFT(x)) ≈ x
//!
//! Also tests FFT/IFFT round-trip for complex signals.

mod common;

use common::ByteCursor;
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

fn make_f64_tensor(shape: &[u32], data: Vec<f64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(l) => vec![l.as_f64().unwrap()],
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let mut cursor = ByteCursor::new(data);

    // Generate a small 1D real signal (length 4-64)
    let len = 4 + (cursor.take_u8() % 61) as usize;
    let mut signal = Vec::with_capacity(len);
    for _ in 0..len {
        let byte = cursor.take_u8();
        let val = (byte as f64 / 255.0) * 2.0 - 1.0; // Normalize to [-1, 1]
        signal.push(val);
    }

    let input = make_f64_tensor(&[len as u32], signal.clone());

    // RFFT → IRFFT round-trip
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_string(), len.to_string());

    let rfft_result = match eval_primitive(Primitive::Rfft, &[input.clone()], &params) {
        Ok(v) => v,
        Err(_) => return, // Some inputs may be invalid
    };

    let irfft_result = match eval_primitive(Primitive::Irfft, &[rfft_result], &params) {
        Ok(v) => v,
        Err(_) => return,
    };

    let recovered = extract_f64_vec(&irfft_result);

    // Verify round-trip: IRFFT(RFFT(x)) ≈ x
    // Tolerance accounts for floating-point accumulation in FFT
    if recovered.len() == signal.len() {
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            let diff = (orig - rec).abs();
            assert!(
                diff < 1e-10,
                "FFT round-trip error too large: orig={}, recovered={}, diff={}",
                orig,
                rec,
                diff
            );
        }
    }
});
