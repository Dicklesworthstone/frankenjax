//! Oracle tests for Hypot primitive.
//!
//! hypot(x, y) = sqrt(x^2 + y^2), computed without overflow for large inputs
//!
//! Tests:
//! - Zero cases: hypot(0, 0) = 0, hypot(x, 0) = |x|
//! - Pythagorean triples: hypot(3, 4) = 5
//! - Symmetry: hypot(x, y) = hypot(y, x)
//! - Infinity handling
//! - NaN propagation
//! - Broadcast-compatible operands
//! - Tensor shapes

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

// ======================== Zero Cases ========================

#[test]
fn oracle_hypot_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "hypot(0, 0) = 0");
}

#[test]
fn oracle_hypot_x_zero() {
    let x = make_f64_tensor(&[], vec![3.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0, "hypot(3, 0) = 3");
}

#[test]
fn oracle_hypot_zero_y() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "hypot(0, 4) = 4");
}

// ======================== Pythagorean Triples ========================

#[test]
fn oracle_hypot_3_4() {
    let x = make_f64_tensor(&[], vec![3.0]);
    let y = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "hypot(3, 4) = 5");
}

#[test]
fn oracle_hypot_5_12() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![12.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 13.0, "hypot(5, 12) = 13");
}

#[test]
fn oracle_hypot_8_15() {
    let x = make_f64_tensor(&[], vec![8.0]);
    let y = make_f64_tensor(&[], vec![15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 17.0, "hypot(8, 15) = 17");
}

// ======================== Symmetry ========================

#[test]
fn oracle_hypot_symmetry() {
    let x = make_f64_tensor(&[], vec![3.0]);
    let y = make_f64_tensor(&[], vec![4.0]);
    let result1 = eval_primitive(Primitive::Hypot, &[x.clone(), y.clone()], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::Hypot, &[y, x], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result1),
        extract_f64_scalar(&result2),
        "hypot(x, y) = hypot(y, x)"
    );
}

// ======================== Negative Inputs ========================

#[test]
fn oracle_hypot_negative_x() {
    let x = make_f64_tensor(&[], vec![-3.0]);
    let y = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "hypot(-3, 4) = 5");
}

#[test]
fn oracle_hypot_negative_y() {
    let x = make_f64_tensor(&[], vec![3.0]);
    let y = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "hypot(3, -4) = 5");
}

#[test]
fn oracle_hypot_both_negative() {
    let x = make_f64_tensor(&[], vec![-3.0]);
    let y = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "hypot(-3, -4) = 5");
}

// ======================== Special Values ========================

#[test]
fn oracle_hypot_inf_finite() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "hypot(inf, 1) = inf"
    );
}

#[test]
fn oracle_hypot_finite_inf() {
    let x = make_f64_tensor(&[], vec![1.0]);
    let y = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "hypot(1, inf) = inf"
    );
}

#[test]
fn oracle_hypot_inf_nan() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "hypot(inf, NaN) = inf (IEEE 754)"
    );
}

#[test]
fn oracle_hypot_nan_propagation() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "hypot(NaN, 1) = NaN");
}

// ======================== Overflow Prevention ========================

#[test]
fn oracle_hypot_large_values() {
    let large = 1e200;
    let x = make_f64_tensor(&[], vec![large]);
    let y = make_f64_tensor(&[], vec![large]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    let expected = large * 2.0_f64.sqrt();
    let actual = extract_f64_scalar(&result);
    assert!(
        (actual - expected).abs() / expected < 1e-10,
        "hypot(1e200, 1e200) should not overflow"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_hypot_vector() {
    let x = make_f64_tensor(&[3], vec![3.0, 5.0, 8.0]);
    let y = make_f64_tensor(&[3], vec![4.0, 12.0, 15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 13.0, 17.0]);
}

#[test]
fn oracle_hypot_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![3.0, 5.0, 8.0, 7.0]);
    let y = make_f64_tensor(&[2, 2], vec![4.0, 12.0, 15.0, 24.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 13.0, 17.0, 25.0]);
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_hypot_scalar_x_tensor_y_broadcast() {
    // scalar x with tensor y
    let x = scalar_f64(3.0);
    let y = make_f64_tensor(&[4], vec![4.0, 0.0, 4.0, 12.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 3.0, 5.0, (3.0_f64.powi(2) + 12.0_f64.powi(2)).sqrt()]);
}

#[test]
fn oracle_hypot_tensor_x_scalar_y_broadcast() {
    // tensor x with scalar y
    let x = make_f64_tensor(&[4], vec![3.0, 0.0, 3.0, 5.0]);
    let y = scalar_f64(4.0);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 4.0, 5.0, (25.0_f64 + 16.0).sqrt()]);
}

#[test]
fn oracle_hypot_singleton_x_vector_y_broadcast() {
    // [1] x with [3] y -> [3]
    let x = make_f64_tensor(&[1], vec![3.0]);
    let y = make_f64_tensor(&[3], vec![4.0, 0.0, 12.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 3.0, (9.0_f64 + 144.0).sqrt()]);
}

#[test]
fn oracle_hypot_vector_x_singleton_y_broadcast() {
    // [3] x with [1] y -> [3]
    let x = make_f64_tensor(&[3], vec![3.0, 5.0, 8.0]);
    let y = make_f64_tensor(&[1], vec![4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.0).abs() < 1e-14, "hypot(3, 4) = 5");
    assert!((vals[1] - (25.0_f64 + 16.0).sqrt()).abs() < 1e-14, "hypot(5, 4)");
    assert!((vals[2] - (64.0_f64 + 16.0).sqrt()).abs() < 1e-14, "hypot(8, 4)");
}

#[test]
fn oracle_hypot_column_x_matrix_y_broadcast() {
    // [2, 1] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![3.0, 5.0]);
    let y = make_f64_tensor(&[2, 3], vec![4.0, 0.0, 4.0, 12.0, 0.0, 12.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: hypot(3, 4)=5, hypot(3, 0)=3, hypot(3, 4)=5
    assert_eq!(vals[0], 5.0);
    assert_eq!(vals[1], 3.0);
    assert_eq!(vals[2], 5.0);
    // Row 1: hypot(5, 12)=13, hypot(5, 0)=5, hypot(5, 12)=13
    assert_eq!(vals[3], 13.0);
    assert_eq!(vals[4], 5.0);
    assert_eq!(vals[5], 13.0);
}

#[test]
fn oracle_hypot_row_y_matrix_x_broadcast() {
    // [2, 3] x with [1, 3] y -> [2, 3]
    let x = make_f64_tensor(&[2, 3], vec![3.0, 5.0, 8.0, 3.0, 5.0, 8.0]);
    let y = make_f64_tensor(&[1, 3], vec![4.0, 12.0, 15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 5.0);
    assert_eq!(vals[1], 13.0);
    assert_eq!(vals[2], 17.0);
    assert_eq!(vals[3], 5.0);
    assert_eq!(vals[4], 13.0);
    assert_eq!(vals[5], 17.0);
}

#[test]
fn oracle_hypot_different_ranks_broadcast() {
    // [3] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[3], vec![3.0, 5.0, 8.0]);
    let y = make_f64_tensor(&[2, 3], vec![4.0, 12.0, 15.0, 4.0, 12.0, 15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![5.0, 13.0, 17.0, 5.0, 13.0, 17.0]);
}

#[test]
fn oracle_hypot_all_scalars_broadcast() {
    // scalar hypot scalar -> scalar
    let x = scalar_f64(3.0);
    let y = scalar_f64(4.0);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_hypot_incompatible_shapes_error() {
    // [2] hypot [3] should error
    let x = make_f64_tensor(&[2], vec![3.0, 5.0]);
    let y = make_f64_tensor(&[3], vec![4.0, 12.0, 15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_hypot_vector_scalar_y_broadcast() {
    let x = make_f64_tensor(&[3], vec![3.0, 0.0, -3.0]);
    let y = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 4.0, 5.0]);
}

#[test]
fn oracle_hypot_matrix_row_y_broadcast() {
    let x = make_f64_tensor(&[2, 3], vec![3.0, 5.0, 8.0, 3.0, 5.0, 8.0]);
    let y = make_f64_tensor(&[3], vec![4.0, 12.0, 15.0]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![5.0, 13.0, 17.0, 5.0, 13.0, 17.0]
    );
}

// ======================== Unit Circle ========================

#[test]
fn oracle_hypot_unit_circle() {
    let x = make_f64_tensor(&[], vec![0.6]);
    let y = make_f64_tensor(&[], vec![0.8]);
    let result = eval_primitive(Primitive::Hypot, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 1.0).abs() < 1e-15, "hypot(0.6, 0.8) = 1.0");
}
