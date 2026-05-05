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

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Option<Value> {
    TensorValue::new(
        DType::Complex128,
        Shape {
            dims: shape.to_vec(),
        },
        data.into_iter()
            .map(|(re, im)| Literal::from_complex128(re, im))
            .collect(),
    )
    .ok()
    .map(Value::Tensor)
}

fn extract_complex128_vec(v: &Value) -> Option<Vec<(f64, f64)>> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_complex128()).collect(),
        Value::Scalar(l) => l.as_complex128().map(|value| vec![value]),
    }
}

fn fuzz_unit(cursor: &mut ByteCursor<'_>) -> f64 {
    (cursor.take_u8() as f64 / 255.0) * 2.0 - 1.0
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
        signal.push(fuzz_unit(&mut cursor));
    }

    let input = make_f64_tensor(&[len as u32], signal.clone());

    // RFFT → IRFFT round-trip
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_string(), len.to_string());

    let rfft_result = match eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &params) {
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

    let complex_len = 4 + (cursor.take_u8() % 61) as usize;
    let mut complex_signal = Vec::with_capacity(complex_len);
    for _ in 0..complex_len {
        complex_signal.push((fuzz_unit(&mut cursor), fuzz_unit(&mut cursor)));
    }

    let Some(complex_input) = make_complex128_tensor(&[complex_len as u32], complex_signal.clone())
    else {
        return;
    };

    let fft_result = match eval_primitive(
        Primitive::Fft,
        std::slice::from_ref(&complex_input),
        &BTreeMap::new(),
    ) {
        Ok(v) => v,
        Err(_) => return,
    };

    let ifft_result = match eval_primitive(
        Primitive::Ifft,
        std::slice::from_ref(&fft_result),
        &BTreeMap::new(),
    ) {
        Ok(v) => v,
        Err(_) => return,
    };

    let Some(complex_recovered) = extract_complex128_vec(&ifft_result) else {
        panic!("complex FFT/IFFT round-trip returned non-complex output");
    };
    assert_eq!(
        complex_recovered.len(),
        complex_signal.len(),
        "complex FFT/IFFT round-trip changed element count"
    );
    for ((orig_re, orig_im), (rec_re, rec_im)) in
        complex_signal.iter().zip(complex_recovered.iter())
    {
        let re_diff = (orig_re - rec_re).abs();
        let im_diff = (orig_im - rec_im).abs();
        assert!(
            re_diff < 1e-10 && im_diff < 1e-10,
            "complex FFT/IFFT round-trip error too large: orig=({}, {}), recovered=({}, {}), diff=({}, {})",
            orig_re,
            orig_im,
            rec_re,
            rec_im,
            re_diff,
            im_diff
        );
    }
});
