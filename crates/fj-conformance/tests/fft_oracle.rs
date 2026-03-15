//! FFT primitive oracle tests.
//!
//! Tests Fft, Ifft, Rfft, Irfft against hand-verified analytical values.
//! Uses small input sizes where DFT results can be computed analytically.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_complex_tensor(pairs: &[(f64, f64)]) -> Value {
    let elements: Vec<Literal> = pairs
        .iter()
        .map(|&(re, im)| Literal::from_complex128(re, im))
        .collect();
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![pairs.len() as u32],
            },
            elements,
        )
        .unwrap(),
    )
}

fn make_real_tensor(data: &[f64]) -> Value {
    let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![data.len() as u32],
            },
            elements,
        )
        .unwrap(),
    )
}

fn extract_complex_vec(val: &Value) -> Vec<(f64, f64)> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex element, got {l:?}"),
        })
        .collect()
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (ar - er).abs() < tol && (ai - ei).abs() < tol,
            "{context}[{i}]: got ({ar}, {ai}), expected ({er}, {ei}) (tol={tol})"
        );
    }
}

fn assert_f64_close(actual: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{context}[{i}]: got {a}, expected {e} (tol={tol})"
        );
    }
}

// ======================== FFT ========================

#[test]
fn oracle_fft_dc_signal() {
    // FFT of [1, 1, 1, 1] = [(4,0), (0,0), (0,0), (0,0)]
    let x = make_complex_tensor(&[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]);
    let result =
        eval_primitive(Primitive::Fft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        1e-10,
        "FFT([1,1,1,1])",
    );
}

#[test]
fn oracle_fft_impulse() {
    // FFT of [1, 0, 0, 0] = [(1,0), (1,0), (1,0), (1,0)]
    let x = make_complex_tensor(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let result =
        eval_primitive(Primitive::Fft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        1e-10,
        "FFT([1,0,0,0])",
    );
}

#[test]
fn oracle_fft_alternating() {
    // FFT of [1, -1, 1, -1] = [(0,0), (0,0), (4,0), (0,0)]
    let x = make_complex_tensor(&[(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]);
    let result =
        eval_primitive(Primitive::Fft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(0.0, 0.0), (0.0, 0.0), (4.0, 0.0), (0.0, 0.0)],
        1e-10,
        "FFT([1,-1,1,-1])",
    );
}

// ======================== IFFT ========================

#[test]
fn oracle_ifft_dc_spectrum() {
    // IFFT of [(4,0), (0,0), (0,0), (0,0)] = [(1,0), (1,0), (1,0), (1,0)]
    let x = make_complex_tensor(&[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let result =
        eval_primitive(Primitive::Ifft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        1e-10,
        "IFFT([(4,0),(0,0),(0,0),(0,0)])",
    );
}

#[test]
fn oracle_fft_ifft_roundtrip() {
    // IFFT(FFT(x)) = x
    let x = make_complex_tensor(&[(1.0, 2.0), (3.0, -1.0), (0.5, 0.0), (-2.0, 1.5)]);
    let fft_result =
        eval_primitive(Primitive::Fft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let roundtrip = eval_primitive(
        Primitive::Ifft,
        std::slice::from_ref(&fft_result),
        &BTreeMap::new(),
    )
    .unwrap();
    let original = extract_complex_vec(&x);
    let recovered = extract_complex_vec(&roundtrip);
    assert_complex_close(&recovered, &original, 1e-10, "IFFT(FFT(x)) = x");
}

// ======================== RFFT ========================

#[test]
fn oracle_rfft_dc_signal() {
    // RFFT of [1, 1, 1, 1] with fft_length=4
    // Half-spectrum: [(4,0), (0,0), (0,0)] (length = 4/2+1 = 3)
    let x = make_real_tensor(&[1.0, 1.0, 1.0, 1.0]);
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());
    let result = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &params).unwrap();
    let y = extract_complex_vec(&result);
    assert_eq!(y.len(), 3, "RFFT output length should be n/2+1 = 3");
    assert_complex_close(
        &y,
        &[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        1e-10,
        "RFFT([1,1,1,1])",
    );
}

#[test]
fn oracle_rfft_impulse() {
    // RFFT of [1, 0, 0, 0] with fft_length=4
    // Half-spectrum: [(1,0), (1,0), (1,0)]
    let x = make_real_tensor(&[1.0, 0.0, 0.0, 0.0]);
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());
    let result = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &params).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        1e-10,
        "RFFT([1,0,0,0])",
    );
}

// ======================== IRFFT ========================

#[test]
fn oracle_irfft_dc() {
    // IRFFT of [(4,0), (0,0), (0,0)] with fft_length=4 → [1, 1, 1, 1]
    let g = make_complex_tensor(&[(4.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());
    let result = eval_primitive(Primitive::Irfft, std::slice::from_ref(&g), &params).unwrap();
    let y = extract_f64_vec(&result);
    assert_f64_close(&y, &[1.0, 1.0, 1.0, 1.0], 1e-10, "IRFFT(DC spectrum)");
}

#[test]
fn oracle_rfft_irfft_roundtrip() {
    // IRFFT(RFFT(x)) = x for real input
    let x = make_real_tensor(&[1.0, 2.0, 3.0, 4.0]);
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());
    let rfft_result = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &params).unwrap();
    let roundtrip = eval_primitive(
        Primitive::Irfft,
        std::slice::from_ref(&rfft_result),
        &params,
    )
    .unwrap();
    let original = extract_f64_vec(&x);
    let recovered = extract_f64_vec(&roundtrip);
    assert_f64_close(&recovered, &original, 1e-10, "IRFFT(RFFT(x)) = x");
}

#[test]
fn oracle_rfft_known_values() {
    // RFFT of [1, 2, 3, 4] with fft_length=4
    // Full DFT: X[0]=(10,0), X[1]=(-2,2), X[2]=(-2,0), X[3]=(-2,-2)
    // RFFT returns first n/2+1=3: [(10,0), (-2,2), (-2,0)]
    let x = make_real_tensor(&[1.0, 2.0, 3.0, 4.0]);
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());
    let result = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &params).unwrap();
    let y = extract_complex_vec(&result);
    assert_complex_close(
        &y,
        &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)],
        1e-10,
        "RFFT([1,2,3,4])",
    );
}
