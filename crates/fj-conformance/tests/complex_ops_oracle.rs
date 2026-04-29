//! Oracle tests for complex number primitives.
//!
//! Tests Real, Imag, and Conj operations on complex numbers:
//! - Real: extracts real part
//! - Imag: extracts imaginary part
//! - Conj: computes complex conjugate (negates imaginary part)

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_complex_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                _ => unreachable!("expected complex128"),
            })
            .collect(),
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            vec![(f64::from_bits(*re), f64::from_bits(*im))]
        }
        _ => unreachable!("expected complex"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Real Tests ========================

#[test]
fn oracle_real_scalar() {
    // real(3 + 4i) = 3
    let input = Value::Scalar(Literal::from_complex128(3.0, 4.0));
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_real_scalar_negative() {
    // real(-2 + 5i) = -2
    let input = Value::Scalar(Literal::from_complex128(-2.0, 5.0));
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-2.0)).abs() < 1e-10);
}

#[test]
fn oracle_real_scalar_zero_imag() {
    // real(7 + 0i) = 7
    let input = Value::Scalar(Literal::from_complex128(7.0, 0.0));
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 7.0).abs() < 1e-10);
}

#[test]
fn oracle_real_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_real_2d() {
    let input = make_complex128_tensor(
        &[2, 2],
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
    );
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

// ======================== Imag Tests ========================

#[test]
fn oracle_imag_scalar() {
    // imag(3 + 4i) = 4
    let input = Value::Scalar(Literal::from_complex128(3.0, 4.0));
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_imag_scalar_negative() {
    // imag(2 - 5i) = -5
    let input = Value::Scalar(Literal::from_complex128(2.0, -5.0));
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-5.0)).abs() < 1e-10);
}

#[test]
fn oracle_imag_scalar_zero_real() {
    // imag(0 + 7i) = 7
    let input = Value::Scalar(Literal::from_complex128(0.0, 7.0));
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 7.0).abs() < 1e-10);
}

#[test]
fn oracle_imag_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0).abs() < 1e-10);
    assert!((vals[1] - 4.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_imag_2d() {
    let input = make_complex128_tensor(
        &[2, 2],
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
    );
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.0)).abs() < 1e-10);
    assert!((vals[3] - (-4.0)).abs() < 1e-10);
}

// ======================== Conj Tests ========================

#[test]
fn oracle_conj_scalar() {
    // conj(3 + 4i) = 3 - 4i
    let input = Value::Scalar(Literal::from_complex128(3.0, 4.0));
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 3.0).abs() < 1e-10);
    assert!((vals[0].1 - (-4.0)).abs() < 1e-10);
}

#[test]
fn oracle_conj_scalar_negative_imag() {
    // conj(2 - 5i) = 2 + 5i
    let input = Value::Scalar(Literal::from_complex128(2.0, -5.0));
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 2.0).abs() < 1e-10);
    assert!((vals[0].1 - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_conj_scalar_real_only() {
    // conj(7 + 0i) = 7 + 0i (real numbers are their own conjugate)
    let input = Value::Scalar(Literal::from_complex128(7.0, 0.0));
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 7.0).abs() < 1e-10);
    assert!(vals[0].1.abs() < 1e-10);
}

#[test]
fn oracle_conj_scalar_imag_only() {
    // conj(0 + 5i) = 0 - 5i
    let input = Value::Scalar(Literal::from_complex128(0.0, 5.0));
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.abs() < 1e-10);
    assert!((vals[0].1 - (-5.0)).abs() < 1e-10);
}

#[test]
fn oracle_conj_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 2.0), (3.0, -4.0), (-5.0, 6.0)]);
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - (-2.0)).abs() < 1e-10);
    assert!((vals[1].0 - 3.0).abs() < 1e-10);
    assert!((vals[1].1 - 4.0).abs() < 1e-10);
    assert!((vals[2].0 - (-5.0)).abs() < 1e-10);
    assert!((vals[2].1 - (-6.0)).abs() < 1e-10);
}

#[test]
fn oracle_conj_2d() {
    let input = make_complex128_tensor(
        &[2, 2],
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
    );
    let result = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_complex_vec(&result);
    for (i, (re, im)) in vals.iter().enumerate() {
        let expected_re = (i + 1) as f64;
        let expected_im = -((i + 1) as f64);
        assert!((*re - expected_re).abs() < 1e-10);
        assert!((*im - expected_im).abs() < 1e-10);
    }
}

#[test]
fn oracle_conj_double_is_identity() {
    // conj(conj(z)) = z
    let input = Value::Scalar(Literal::from_complex128(3.0, 4.0));
    let result1 = eval_primitive(Primitive::Conj, &[input], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::Conj, &[result1], &no_params()).unwrap();
    let vals = extract_complex_vec(&result2);
    assert!((vals[0].0 - 3.0).abs() < 1e-10);
    assert!((vals[0].1 - 4.0).abs() < 1e-10);
}
