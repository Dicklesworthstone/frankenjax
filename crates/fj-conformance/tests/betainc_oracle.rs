//! Oracle tests for Betainc primitive.
//!
//! betainc(a, b, x) = I_x(a,b) (regularized incomplete beta function)
//!
//! Key properties:
//! - betainc(a, b, 0) = 0
//! - betainc(a, b, 1) = 1
//! - betainc(1, 1, x) = x (uniform CDF)
//! - betainc(a, b, x) + betainc(b, a, 1-x) = 1
//!
//! Tests:
//! - Boundary values
//! - Uniform case
//! - Symmetry property
//! - Tensor shapes
//! - Broadcast-compatible operands

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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Boundary Cases ========================

#[test]
fn oracle_betainc_x_zero() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "betainc(a, b, 0) = 0");
}

#[test]
fn oracle_betainc_x_one() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "betainc(a, b, 1) = 1");
}

// ======================== Uniform Case (a=1, b=1) ========================

#[test]
fn oracle_betainc_uniform() {
    // betainc(1, 1, x) = x (uniform distribution CDF)
    for x_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let a = make_f64_tensor(&[], vec![1.0]);
        let b = make_f64_tensor(&[], vec![1.0]);
        let x = make_f64_tensor(&[], vec![x_val]);
        let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
        let actual = extract_f64_scalar(&result);
        assert!(
            (actual - x_val).abs() < 1e-14,
            "betainc(1, 1, {}) = {}, got {}",
            x_val,
            x_val,
            actual
        );
    }
}

// ======================== Symmetry Property ========================

#[test]
fn oracle_betainc_symmetry() {
    // betainc(a, b, x) + betainc(b, a, 1-x) = 1
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![0.4]);
    let one_minus_x = make_f64_tensor(&[], vec![0.6]);

    let r1 = eval_primitive(Primitive::Betainc, &[a.clone(), b.clone(), x], &no_params()).unwrap();
    let r2 = eval_primitive(Primitive::Betainc, &[b, a, one_minus_x], &no_params()).unwrap();

    let sum = extract_f64_scalar(&r1) + extract_f64_scalar(&r2);
    assert!(
        (sum - 1.0).abs() < 1e-14,
        "betainc(a,b,x) + betainc(b,a,1-x) = 1, got {}",
        sum
    );
}

// ======================== Known Values ========================

#[test]
fn oracle_betainc_half() {
    // betainc(a, a, 0.5) = 0.5 for any a (symmetric case)
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 0.5).abs() < 1e-14, "betainc(a, a, 0.5) = 0.5");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_betainc_vector() {
    let a = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 0.5).abs() < 1e-14);
    assert!((vals[2] - 1.0).abs() < 1e-14);
}

// ======================== Broadcasting ========================

#[test]
fn oracle_betainc_scalar_params_vector_x_broadcast() {
    let a = Value::scalar_f64(1.0);
    let b = Value::scalar_f64(1.0);
    let x_values = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let x = make_f64_tensor(&[5], x_values.to_vec());
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![5]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "betainc scalar-parameter broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}
