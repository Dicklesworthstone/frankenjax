//! Oracle tests for Select primitive.
//!
//! Tests against expected behavior matching JAX/lax.select:
//! - condition: boolean (or truthy) tensor/scalar
//! - on_true: value to use where condition is true
//! - on_false: value to use where condition is false
//! - All inputs must have same shape (for tensor case)

#![allow(clippy::approx_constant)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_bool_tensor(shape: &[u32], data: Vec<bool>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
}

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => unreachable!("expected bool"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => unreachable!("expected bool"),
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
fn oracle_select_scalar_true() {
    // JAX: lax.select(True, 10, 20) => 10
    let cond = Value::Scalar(Literal::Bool(true));
    let on_true = Value::scalar_i64(10);
    let on_false = Value::scalar_i64(20);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10]);
}

#[test]
fn oracle_select_scalar_false() {
    // JAX: lax.select(False, 10, 20) => 20
    let cond = Value::Scalar(Literal::Bool(false));
    let on_true = Value::scalar_i64(10);
    let on_false = Value::scalar_i64(20);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![20]);
}

#[test]
fn oracle_select_scalar_f64() {
    let cond = Value::Scalar(Literal::Bool(true));
    let on_true = Value::Scalar(Literal::from_f64(3.14));
    let on_false = Value::Scalar(Literal::from_f64(2.71));
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.14).abs() < 1e-10);
}

#[test]
fn oracle_select_scalar_int_cond_true() {
    // Non-zero integer as condition (truthy)
    let cond = Value::scalar_i64(1);
    let on_true = Value::scalar_i64(100);
    let on_false = Value::scalar_i64(200);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![100]);
}

#[test]
fn oracle_select_scalar_int_cond_false() {
    // Zero integer as condition (falsy)
    let cond = Value::scalar_i64(0);
    let on_true = Value::scalar_i64(100);
    let on_false = Value::scalar_i64(200);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![200]);
}

#[test]
fn oracle_select_scalar_negative_int_cond() {
    // Negative integer is truthy
    let cond = Value::scalar_i64(-5);
    let on_true = Value::scalar_i64(1);
    let on_false = Value::scalar_i64(0);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1]);
}

// ======================== 1D Tensor Tests ========================

#[test]
fn oracle_select_1d_all_true() {
    // JAX: lax.select([True, True, True], [1, 2, 3], [4, 5, 6]) => [1, 2, 3]
    let cond = make_bool_tensor(&[3], vec![true, true, true]);
    let on_true = make_i64_tensor(&[3], vec![1, 2, 3]);
    let on_false = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_select_1d_all_false() {
    // JAX: lax.select([False, False, False], [1, 2, 3], [4, 5, 6]) => [4, 5, 6]
    let cond = make_bool_tensor(&[3], vec![false, false, false]);
    let on_true = make_i64_tensor(&[3], vec![1, 2, 3]);
    let on_false = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4, 5, 6]);
}

#[test]
fn oracle_select_1d_mixed() {
    // JAX: lax.select([True, False, True], [1, 2, 3], [4, 5, 6]) => [1, 5, 3]
    let cond = make_bool_tensor(&[3], vec![true, false, true]);
    let on_true = make_i64_tensor(&[3], vec![1, 2, 3]);
    let on_false = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 5, 3]);
}

#[test]
fn oracle_select_1d_alternating() {
    // JAX: lax.select([False, True, False, True], [1, 2, 3, 4], [5, 6, 7, 8]) => [5, 2, 7, 4]
    let cond = make_bool_tensor(&[4], vec![false, true, false, true]);
    let on_true = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let on_false = make_i64_tensor(&[4], vec![5, 6, 7, 8]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 2, 7, 4]);
}

#[test]
fn oracle_select_1d_f64() {
    let cond = make_bool_tensor(&[3], vec![true, false, true]);
    let on_true = make_f64_tensor(&[3], vec![1.1, 2.2, 3.3]);
    let on_false = make_f64_tensor(&[3], vec![4.4, 5.5, 6.6]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[1] - 5.5).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

#[test]
fn oracle_select_1d_int_cond() {
    // Integer condition: 0 = false, non-zero = true
    let cond = make_i64_tensor(&[4], vec![0, 1, 0, -1]);
    let on_true = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let on_false = make_i64_tensor(&[4], vec![100, 200, 300, 400]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![100, 20, 300, 40]);
}

// ======================== 2D Tensor Tests ========================

#[test]
fn oracle_select_2d_row_mask() {
    // Select entire rows based on condition
    // Shape [2, 3]
    let cond = make_bool_tensor(&[2, 3], vec![true, true, true, false, false, false]);
    let on_true = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let on_false = make_i64_tensor(&[2, 3], vec![10, 20, 30, 40, 50, 60]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 40, 50, 60]);
}

#[test]
fn oracle_select_2d_checkerboard() {
    // Checkerboard pattern
    // Shape [2, 2]
    let cond = make_bool_tensor(&[2, 2], vec![true, false, false, true]);
    let on_true = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let on_false = make_i64_tensor(&[2, 2], vec![5, 6, 7, 8]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 6, 7, 4]);
}

#[test]
fn oracle_select_2d_column_mask() {
    // Select based on column pattern
    // Shape [3, 2]
    let cond = make_bool_tensor(&[3, 2], vec![true, false, true, false, true, false]);
    let on_true = make_i64_tensor(&[3, 2], vec![1, 2, 3, 4, 5, 6]);
    let on_false = make_i64_tensor(&[3, 2], vec![10, 20, 30, 40, 50, 60]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 20, 3, 40, 5, 60]);
}

#[test]
fn oracle_select_2d_f64() {
    let cond = make_bool_tensor(&[2, 2], vec![true, false, true, false]);
    let on_true = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let on_false = make_f64_tensor(&[2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 20.0).abs() < 1e-10);
    assert!((vals[2] - 3.0).abs() < 1e-10);
    assert!((vals[3] - 40.0).abs() < 1e-10);
}

// ======================== 3D Tensor Tests ========================

#[test]
fn oracle_select_3d() {
    // Shape [2, 2, 2]
    let cond = make_bool_tensor(
        &[2, 2, 2],
        vec![true, false, true, false, false, true, false, true],
    );
    let on_true = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let on_false = make_i64_tensor(&[2, 2, 2], vec![10, 20, 30, 40, 50, 60, 70, 80]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 20, 3, 40, 50, 6, 70, 8]);
}

#[test]
fn oracle_select_3d_all_true() {
    let cond = make_bool_tensor(&[2, 2, 2], vec![true; 8]);
    let on_true = make_i64_tensor(&[2, 2, 2], (1..=8).collect());
    let on_false = make_i64_tensor(&[2, 2, 2], (10..=17).collect());
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), (1..=8).collect::<Vec<_>>());
}

// ======================== Edge Cases ========================

#[test]
fn oracle_select_single_element() {
    let cond = make_bool_tensor(&[1], vec![true]);
    let on_true = make_i64_tensor(&[1], vec![42]);
    let on_false = make_i64_tensor(&[1], vec![99]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_select_with_negatives() {
    let cond = make_bool_tensor(&[3], vec![true, false, true]);
    let on_true = make_i64_tensor(&[3], vec![-5, -10, -15]);
    let on_false = make_i64_tensor(&[3], vec![5, 10, 15]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, 10, -15]);
}

#[test]
fn oracle_select_same_values() {
    // When on_true == on_false, result should be the same regardless of condition
    let cond = make_bool_tensor(&[3], vec![true, false, true]);
    let on_true = make_i64_tensor(&[3], vec![7, 7, 7]);
    let on_false = make_i64_tensor(&[3], vec![7, 7, 7]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![7, 7, 7]);
}

#[test]
fn oracle_select_bool_values() {
    // Select between boolean values
    let cond = make_bool_tensor(&[3], vec![true, false, true]);
    let on_true = make_bool_tensor(&[3], vec![true, true, true]);
    let on_false = make_bool_tensor(&[3], vec![false, false, false]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, false, true]);
}

#[test]
fn oracle_select_large() {
    let n = 100;
    let cond_data: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let on_true_data: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let on_false_data: Vec<i64> = (0..n).map(|i| -(i as i64)).collect();
    let cond = make_bool_tensor(&[n as u32], cond_data);
    let on_true = make_i64_tensor(&[n as u32], on_true_data);
    let on_false = make_i64_tensor(&[n as u32], on_false_data);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), n);
    assert_eq!(vals[0], 0); // true -> 0
    assert_eq!(vals[1], -1); // false -> -1
    assert_eq!(vals[2], 2); // true -> 2
    assert_eq!(vals[3], -3); // false -> -3
}

// ======================== Mask Pattern Tests ========================

#[test]
fn oracle_select_threshold_mask() {
    // Simulate threshold masking: select values > threshold
    // We manually create the condition from a threshold test
    let cond = make_bool_tensor(&[5], vec![false, false, true, true, true]);
    let on_true = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let on_false = make_i64_tensor(&[5], vec![0, 0, 0, 0, 0]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 3, 4, 5]);
}

#[test]
fn oracle_select_clamp_simulation() {
    // Simulate clamping: select original if within bounds, else bound value
    let cond = make_bool_tensor(&[5], vec![false, true, true, true, false]);
    let on_true = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]); // original values
    let on_false = make_i64_tensor(&[5], vec![1, 1, 2, 3, 3]); // clamped to [1, 3]
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 2, 3, 3]);
}

#[test]
fn oracle_select_relu_simulation() {
    // Simulate ReLU: select x if x > 0, else 0
    let cond = make_bool_tensor(&[5], vec![false, false, false, true, true]);
    let on_true = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let on_false = make_f64_tensor(&[5], vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

// ======================== Unit Dimension Tests ========================

#[test]
fn oracle_select_2d_unit_row() {
    // Shape [1, 4]
    let cond = make_bool_tensor(&[1, 4], vec![true, false, true, false]);
    let on_true = make_i64_tensor(&[1, 4], vec![1, 2, 3, 4]);
    let on_false = make_i64_tensor(&[1, 4], vec![10, 20, 30, 40]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 20, 3, 40]);
}

#[test]
fn oracle_select_2d_unit_col() {
    // Shape [4, 1]
    let cond = make_bool_tensor(&[4, 1], vec![true, false, true, false]);
    let on_true = make_i64_tensor(&[4, 1], vec![1, 2, 3, 4]);
    let on_false = make_i64_tensor(&[4, 1], vec![10, 20, 30, 40]);
    let result =
        eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 20, 3, 40]);
}
