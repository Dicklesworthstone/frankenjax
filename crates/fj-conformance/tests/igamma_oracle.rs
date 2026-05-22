//! Oracle tests for Igamma and Igammac primitives.
//!
//! igamma(a, x) = P(a, x) = γ(a,x)/Γ(a) (regularized lower incomplete gamma)
//! igammac(a, x) = Q(a, x) = 1 - P(a, x) (regularized upper incomplete gamma)
//!
//! Key properties:
//! - igamma(a, 0) = 0, igammac(a, 0) = 1
//! - igamma(a, inf) = 1, igammac(a, inf) = 0
//! - igamma(a, x) + igammac(a, x) = 1
//!
//! Tests:
//! - Boundary values
//! - Complementary property
//! - Known values
//! - Special values
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

// ======================== Igamma Boundary Cases ========================

#[test]
fn oracle_igamma_x_zero() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "igamma(a, 0) = 0");
}

#[test]
fn oracle_igamma_x_inf() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "igamma(a, inf) = 1");
}

// ======================== Igammac Boundary Cases ========================

#[test]
fn oracle_igammac_x_zero() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "igammac(a, 0) = 1");
}

#[test]
fn oracle_igammac_x_inf() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "igammac(a, inf) = 0");
}

// ======================== Complementary Property ========================

#[test]
fn oracle_igamma_igammac_sum_to_one() {
    for (a, x) in [(1.0, 0.5), (2.0, 1.0), (3.0, 2.0), (0.5, 1.0)] {
        let a_val = make_f64_tensor(&[], vec![a]);
        let x_val = make_f64_tensor(&[], vec![x]);
        let ig = eval_primitive(
            Primitive::Igamma,
            &[a_val.clone(), x_val.clone()],
            &no_params(),
        )
        .unwrap();
        let igc = eval_primitive(Primitive::Igammac, &[a_val, x_val], &no_params()).unwrap();
        let sum = extract_f64_scalar(&ig) + extract_f64_scalar(&igc);
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "igamma({}, {}) + igammac({}, {}) = {} (should be 1)",
            a,
            x,
            a,
            x,
            sum
        );
    }
}

// ======================== Known Values ========================

#[test]
fn oracle_igamma_a1_x1() {
    // For a=1: igamma(1, x) = 1 - exp(-x)
    let a = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((actual - expected).abs() < 1e-14, "igamma(1, 1) = 1 - e^-1");
}

#[test]
fn oracle_igammac_a1_x1() {
    // For a=1: igammac(1, x) = exp(-x)
    let a = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = (-1.0_f64).exp();
    assert!((actual - expected).abs() < 1e-14, "igammac(1, 1) = e^-1");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_igamma_vector() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|&v| v == 0.0), "igamma(a, 0) = 0 for all a");
}

#[test]
fn oracle_igammac_vector() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(
        vals.iter().all(|&v| v == 1.0),
        "igammac(a, 0) = 1 for all a"
    );
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_igamma_scalar_a_tensor_x_broadcast() {
    // scalar a with tensor x
    let a = scalar_f64(1.0);
    let x = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // igamma(1, x) = 1 - exp(-x)
    assert!((vals[0] - 0.0).abs() < 1e-14, "igamma(1, 0) = 0");
    assert!((vals[1] - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
    assert!((vals[2] - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
}

#[test]
fn oracle_igamma_tensor_a_scalar_x_broadcast() {
    // tensor a with scalar x
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let x = scalar_f64(0.0);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    // igamma(a, 0) = 0 for all a
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.0);
}

#[test]
fn oracle_igamma_singleton_a_vector_x_broadcast() {
    // [1] a with [3] x -> [3]
    let a = make_f64_tensor(&[1], vec![1.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // igamma(1, x) = 1 - exp(-x)
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
    assert!((vals[2] - (1.0 - (-2.0_f64).exp())).abs() < 1e-14);
}

#[test]
fn oracle_igamma_column_a_matrix_x_broadcast() {
    // [2, 1] a with [2, 3] x -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![1.0, 1.0]);
    let x = make_f64_tensor(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: igamma(1, 0)=0, igamma(1, 1)=1-e^-1, igamma(1, 2)=1-e^-2
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
    assert!((vals[2] - (1.0 - (-2.0_f64).exp())).abs() < 1e-14);
    // Row 1: same values
    assert_eq!(vals[3], 0.0);
    assert!((vals[4] - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
    assert!((vals[5] - (1.0 - (-2.0_f64).exp())).abs() < 1e-14);
}

#[test]
fn oracle_igamma_different_ranks_broadcast() {
    // [3] a with [2, 3] x -> [2, 3]
    let a = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let x = make_f64_tensor(&[2, 3], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x=0, so igamma=0
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    // Row 1: x=1, so igamma(1,1) = 1-e^-1
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((vals[3] - expected).abs() < 1e-14);
    assert!((vals[4] - expected).abs() < 1e-14);
    assert!((vals[5] - expected).abs() < 1e-14);
}

#[test]
fn oracle_igamma_all_scalars_broadcast() {
    // scalar igamma scalar -> scalar
    let a = scalar_f64(1.0);
    let x = scalar_f64(1.0);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((val - expected).abs() < 1e-14);
}

#[test]
fn oracle_igamma_incompatible_shapes_error() {
    // [2] igamma [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_igammac_tensor_a_scalar_x_broadcast() {
    // tensor a with scalar x (complement)
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let x = scalar_f64(0.0);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    // igammac(a, 0) = 1 for all a
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
}

#[test]
fn oracle_igammac_all_scalars_broadcast() {
    // scalar igammac scalar -> scalar
    let a = scalar_f64(1.0);
    let x = scalar_f64(0.0);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_igammac_incompatible_shapes_error() {
    // [2] igammac [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_igamma_scalar_a_vector_x_broadcast() {
    let a = make_f64_tensor(&[], vec![1.0]);
    let x_values = [0.0_f64, 1.0, 2.0];
    let x = make_f64_tensor(&[3], x_values.to_vec());
    let result = eval_primitive(Primitive::Igamma, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &x)) in actual.iter().zip(x_values.iter()).enumerate() {
        let expected = 1.0 - (-x).exp();
        assert!(
            (actual - expected).abs() < 1e-14,
            "igamma scalar a broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_igammac_vector_a_scalar_x_broadcast() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Igammac, &[a, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let actual = extract_f64_vec(&result);
    for (i, &actual) in actual.iter().enumerate() {
        assert_eq!(
            actual, 1.0,
            "igammac scalar x broadcast element {i}: expected 1, got {actual}"
        );
    }
}
