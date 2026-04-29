//! Oracle tests for ReduceAnd and ReduceXor primitives.
//!
//! Tests against expected behavior for bitwise/logical reductions:
//! - ReduceAnd: AND reduction along specified axes
//! - ReduceXor: XOR reduction along specified axes

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

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => panic!("expected bool"),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn reduce_params(axes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "axes".to_string(),
        axes.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== ReduceAnd - Bool Tests ========================

#[test]
fn oracle_reduce_and_bool_all_true() {
    // AND of all true = true
    let input = make_bool_tensor(&[4], vec![true, true, true, true]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_reduce_and_bool_one_false() {
    // AND with one false = false
    let input = make_bool_tensor(&[4], vec![true, true, false, true]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_reduce_and_bool_all_false() {
    // AND of all false = false
    let input = make_bool_tensor(&[4], vec![false, false, false, false]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_reduce_and_bool_2d_axis0() {
    // [2, 3] reduce along axis 0
    let input = make_bool_tensor(&[2, 3], vec![true, false, true, true, true, false]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, false]);
}

#[test]
fn oracle_reduce_and_bool_2d_axis1() {
    // [2, 3] reduce along axis 1
    let input = make_bool_tensor(&[2, 3], vec![true, true, true, true, false, true]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false]);
}

// ======================== ReduceAnd - Integer Tests ========================

#[test]
fn oracle_reduce_and_i64_all_ones() {
    // -1 in two's complement is all 1 bits
    let input = make_i64_tensor(&[3], vec![-1, -1, -1]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1]);
}

#[test]
fn oracle_reduce_and_i64_with_zero() {
    // AND with 0 = 0
    let input = make_i64_tensor(&[3], vec![0xFF, 0, 0xFF]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_reduce_and_i64_masks() {
    // 0xFF & 0x0F & 0x03 = 0x03
    let input = make_i64_tensor(&[3], vec![0xFF, 0x0F, 0x03]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0x03]);
}

#[test]
fn oracle_reduce_and_i64_2d() {
    let input = make_i64_tensor(&[2, 2], vec![0xFF, 0x0F, 0x0F, 0x03]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![0x0F, 0x03]);
}

// ======================== ReduceXor - Bool Tests ========================

#[test]
fn oracle_reduce_xor_bool_even() {
    // XOR of even number of trues = false
    let input = make_bool_tensor(&[4], vec![true, true, false, false]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_reduce_xor_bool_odd() {
    // XOR of odd number of trues = true
    let input = make_bool_tensor(&[4], vec![true, true, true, false]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_reduce_xor_bool_all_false() {
    // XOR of all false = false
    let input = make_bool_tensor(&[4], vec![false, false, false, false]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_reduce_xor_bool_single_true() {
    // XOR of single true = true
    let input = make_bool_tensor(&[4], vec![false, false, true, false]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_reduce_xor_bool_2d_axis0() {
    let input = make_bool_tensor(&[2, 3], vec![true, false, true, true, true, false]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    // Column 0: true ^ true = false
    // Column 1: false ^ true = true
    // Column 2: true ^ false = true
    assert_eq!(extract_bool_vec(&result), vec![false, true, true]);
}

#[test]
fn oracle_reduce_xor_bool_2d_axis1() {
    let input = make_bool_tensor(&[2, 3], vec![true, false, true, false, true, true]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    // Row 0: true ^ false ^ true = false
    // Row 1: false ^ true ^ true = false
    assert_eq!(extract_bool_vec(&result), vec![false, false]);
}

// ======================== ReduceXor - Integer Tests ========================

#[test]
fn oracle_reduce_xor_i64_same() {
    // x ^ x = 0
    let input = make_i64_tensor(&[2], vec![0x1234, 0x1234]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_reduce_xor_i64_with_zero() {
    // x ^ 0 = x
    let input = make_i64_tensor(&[2], vec![0x5678, 0]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0x5678]);
}

#[test]
fn oracle_reduce_xor_i64_pattern() {
    // 0xFF ^ 0x0F = 0xF0
    let input = make_i64_tensor(&[2], vec![0xFF, 0x0F]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0xF0]);
}

#[test]
fn oracle_reduce_xor_i64_three() {
    // 0xFF ^ 0x0F ^ 0xF0 = 0x00
    let input = make_i64_tensor(&[3], vec![0xFF, 0x0F, 0xF0]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0x00]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reduce_and_single_element() {
    let input = make_bool_tensor(&[1], vec![true]);
    let result = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_reduce_xor_single_element() {
    let input = make_bool_tensor(&[1], vec![true]);
    let result = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}
