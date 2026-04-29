//! Oracle tests for ReduceProd primitive.
//!
//! Tests product reduction semantics:
//! - Full reduction: product of all elements
//! - Axis reduction: product along specified axes
//! - Identity: empty product is 1
//! - Zero absorption: any zero makes product zero

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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
        _ => panic!("expected tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
        _ => panic!("expected scalar"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
        _ => panic!("expected scalar"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "axes".to_string(),
        axes.iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== Basic 1D Full Reduction ========================

#[test]
fn oracle_reduce_prod_1d_basic() {
    // 1 * 2 * 3 * 4 * 5 = 120
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 120);
}

#[test]
fn oracle_reduce_prod_1d_with_ones() {
    // 1 * 1 * 1 * 1 = 1
    let input = make_i64_tensor(&[4], vec![1, 1, 1, 1]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_reduce_prod_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 42);
}

#[test]
fn oracle_reduce_prod_scalar() {
    let input = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 7);
}

// ======================== Zero Absorption ========================

#[test]
fn oracle_reduce_prod_with_zero() {
    // Any zero makes the product zero
    let input = make_i64_tensor(&[5], vec![1, 2, 0, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_all_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_zero_at_start() {
    let input = make_i64_tensor(&[3], vec![0, 5, 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_zero_at_end() {
    let input = make_i64_tensor(&[3], vec![5, 10, 0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Negative Numbers ========================

#[test]
fn oracle_reduce_prod_negative_even() {
    // (-1) * (-2) = 2 (even number of negatives -> positive)
    let input = make_i64_tensor(&[2], vec![-1, -2]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2);
}

#[test]
fn oracle_reduce_prod_negative_odd() {
    // (-1) * (-2) * (-3) = -6 (odd number of negatives -> negative)
    let input = make_i64_tensor(&[3], vec![-1, -2, -3]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -6);
}

#[test]
fn oracle_reduce_prod_mixed_signs() {
    // (-2) * 3 * (-4) = 24
    let input = make_i64_tensor(&[3], vec![-2, 3, -4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

// ======================== 2D Full Reduction ========================

#[test]
fn oracle_reduce_prod_2d_full() {
    // [[1, 2], [3, 4]] -> 1*2*3*4 = 24
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

#[test]
fn oracle_reduce_prod_2d_with_zero() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 0, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Axis Reduction ========================

#[test]
fn oracle_reduce_prod_2d_axis0() {
    // [[1, 2, 3], [4, 5, 6]] -> axis 0 -> [1*4, 2*5, 3*6] = [4, 10, 18]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![4, 10, 18]);
}

#[test]
fn oracle_reduce_prod_2d_axis1() {
    // [[1, 2, 3], [4, 5, 6]] -> axis 1 -> [1*2*3, 4*5*6] = [6, 120]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![6, 120]);
}

#[test]
fn oracle_reduce_prod_3d_axis0() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 0 -> [[1*5,2*6],[3*7,4*8]] = [[5,12],[21,32]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![5, 12, 21, 32]);
}

#[test]
fn oracle_reduce_prod_3d_axis1() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 1 -> [[1*3,2*4],[5*7,6*8]] = [[3,8],[35,48]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 8, 35, 48]);
}

#[test]
fn oracle_reduce_prod_3d_axis2() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 2 -> [[1*2,3*4],[5*6,7*8]] = [[2,12],[30,56]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 12, 30, 56]);
}

#[test]
fn oracle_reduce_prod_multiple_axes() {
    // [[1, 2], [3, 4]] -> axes [0, 1] -> full reduction -> 24
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0, 1])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

// ======================== Float Tests ========================

#[test]
fn oracle_reduce_prod_f64_basic() {
    // 1.5 * 2.0 * 4.0 = 12.0
    let input = make_f64_tensor(&[3], vec![1.5, 2.0, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_with_fractions() {
    // 0.5 * 0.5 * 4.0 = 1.0
    let input = make_f64_tensor(&[3], vec![0.5, 0.5, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_negative() {
    // -1.0 * 2.0 * -3.0 = 6.0
    let input = make_f64_tensor(&[3], vec![-1.0, 2.0, -3.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_with_zero() {
    let input = make_f64_tensor(&[3], vec![5.0, 0.0, 10.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_reduce_prod_f64_axis() {
    // [[1.0, 2.0], [3.0, 4.0]] -> axis 0 -> [3.0, 8.0]
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 8.0).abs() < 1e-10);
}

// ======================== Special Float Values ========================

#[test]
fn oracle_reduce_prod_f64_infinity() {
    // Large values can overflow to infinity
    let input = make_f64_tensor(&[2], vec![1e200, 1e200]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn oracle_reduce_prod_f64_underflow() {
    // Small values can underflow to zero
    let input = make_f64_tensor(&[2], vec![1e-200, 1e-200]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_reduce_prod_f64_nan_propagates() {
    let input = make_f64_tensor(&[3], vec![1.0, f64::NAN, 3.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan());
}

#[test]
fn oracle_reduce_prod_f64_inf_times_zero() {
    // inf * 0 = NaN
    let input = make_f64_tensor(&[2], vec![f64::INFINITY, 0.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan());
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reduce_prod_large_values() {
    // Test with moderately large values that don't overflow
    let input = make_i64_tensor(&[4], vec![10, 10, 10, 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 10000);
}

#[test]
fn oracle_reduce_prod_powers_of_two() {
    // 2^10 = 1024
    let input = make_i64_tensor(&[10], vec![2; 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1024);
}

#[test]
fn oracle_reduce_prod_factorials() {
    // 5! = 120
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 120);

    // 6! = 720
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 720);
}

// ======================== Associativity Check ========================

#[test]
fn oracle_reduce_prod_associativity() {
    // Product is associative: (a*b)*c = a*(b*c)
    // Verify the order doesn't matter for integer product
    let input1 = make_i64_tensor(&[4], vec![2, 3, 5, 7]);
    let input2 = make_i64_tensor(&[4], vec![7, 5, 3, 2]);
    let result1 = eval_primitive(Primitive::ReduceProd, &[input1], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::ReduceProd, &[input2], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result1), extract_i64_scalar(&result2));
    assert_eq!(extract_i64_scalar(&result1), 210); // 2*3*5*7
}
