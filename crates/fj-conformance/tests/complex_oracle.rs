//! Oracle tests for Complex primitive.
//!
//! Tests against expected behavior for creating complex numbers:
//! - Takes two inputs: real part and imaginary part
//! - Creates complex128 output

#![allow(clippy::approx_constant)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
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

fn extract_complex_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                Literal::Complex64Bits(re, im) => {
                    (f32::from_bits(*re) as f64, f32::from_bits(*im) as f64)
                }
                _ => unreachable!("expected complex"),
            })
            .collect(),
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            vec![(f64::from_bits(*re), f64::from_bits(*im))]
        }
        Value::Scalar(Literal::Complex64Bits(re, im)) => {
            vec![(f32::from_bits(*re) as f64, f32::from_bits(*im) as f64)]
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

// ======================== Scalar Tests ========================

#[test]
fn oracle_complex_scalar_basic() {
    // complex(3, 4) = 3 + 4i
    let re = Value::Scalar(Literal::from_f64(3.0));
    let im = Value::Scalar(Literal::from_f64(4.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 3.0).abs() < 1e-10);
    assert!((vals[0].1 - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_negative() {
    // complex(-2, -5) = -2 - 5i
    let re = Value::Scalar(Literal::from_f64(-2.0));
    let im = Value::Scalar(Literal::from_f64(-5.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - (-2.0)).abs() < 1e-10);
    assert!((vals[0].1 - (-5.0)).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zero_imag() {
    // complex(5, 0) = 5 + 0i (real number)
    let re = Value::Scalar(Literal::from_f64(5.0));
    let im = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 5.0).abs() < 1e-10);
    assert!(vals[0].1.abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zero_real() {
    // complex(0, 7) = 0 + 7i (pure imaginary)
    let re = Value::Scalar(Literal::from_f64(0.0));
    let im = Value::Scalar(Literal::from_f64(7.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.abs() < 1e-10);
    assert!((vals[0].1 - 7.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zeros() {
    // complex(0, 0) = 0 + 0i
    let re = Value::Scalar(Literal::from_f64(0.0));
    let im = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.abs() < 1e-10);
    assert!(vals[0].1.abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_complex_1d() {
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 4.0).abs() < 1e-10);
    assert!((vals[2].0 - 3.0).abs() < 1e-10);
    assert!((vals[2].1 - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_1d_mixed_signs() {
    let re = make_f64_tensor(&[4], vec![-1.0, 1.0, -1.0, 1.0]);
    let im = make_f64_tensor(&[4], vec![1.0, -1.0, -1.0, 1.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - (-1.0)).abs() < 1e-10);
    assert!((vals[0].1 - 1.0).abs() < 1e-10);
    assert!((vals[1].0 - 1.0).abs() < 1e-10);
    assert!((vals[1].1 - (-1.0)).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_complex_2d() {
    let re = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let im = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 5.0).abs() < 1e-10);
    assert!((vals[3].0 - 4.0).abs() < 1e-10);
    assert!((vals[3].1 - 8.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_complex_single_element() {
    let re = make_f64_tensor(&[1], vec![3.17]);
    let im = make_f64_tensor(&[1], vec![2.71]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 3.17).abs() < 1e-10);
    assert!((vals[0].1 - 2.71).abs() < 1e-10);
}

#[test]
fn oracle_complex_large() {
    let re = make_f64_tensor(&[10], (1..=10).map(|x| x as f64).collect());
    let im = make_f64_tensor(&[10], (11..=20).map(|x| x as f64).collect());
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![10]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 11.0).abs() < 1e-10);
    assert!((vals[9].0 - 10.0).abs() < 1e-10);
    assert!((vals[9].1 - 20.0).abs() < 1e-10);
}
