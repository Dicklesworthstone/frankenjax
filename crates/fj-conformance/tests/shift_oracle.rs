//! Oracle tests for Shift primitives.
//!
//! Tests against expected behavior for:
//! - ShiftLeft: logical left shift (multiply by 2^n)
//! - ShiftRightArithmetic: arithmetic right shift (preserves sign)
//! - ShiftRightLogical: logical right shift (fills with zeros)

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

fn make_u32_tensor(shape: &[u32], data: Vec<u32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::U32).collect(),
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

fn extract_u32_vec(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::U32(v) => *v,
                _ => l.as_u64().unwrap() as u32,
            })
            .collect(),
        Value::Scalar(Literal::U32(v)) => vec![*v],
        Value::Scalar(lit) => vec![lit.as_u64().unwrap() as u32],
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

// ======================== ShiftLeft Tests ========================

#[test]
fn oracle_shift_left_scalar_1() {
    // 1 << 1 = 2
    let a = Value::scalar_i64(1);
    let b = Value::scalar_i64(1);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![2]);
}

#[test]
fn oracle_shift_left_scalar_by_zero() {
    // x << 0 = x
    let a = Value::scalar_i64(42);
    let b = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_shift_left_multiply_by_powers_of_two() {
    // 5 << n = 5 * 2^n
    let a = make_i64_tensor(&[4], vec![5, 5, 5, 5]);
    let b = make_i64_tensor(&[4], vec![0, 1, 2, 3]);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 10, 20, 40]);
}

#[test]
fn oracle_shift_left_1d() {
    let a = make_i64_tensor(&[4], vec![1, 2, 4, 8]);
    let b = make_i64_tensor(&[4], vec![4, 3, 2, 1]);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![16, 16, 16, 16]);
}

#[test]
fn oracle_shift_left_2d() {
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[2, 2], vec![1, 1, 1, 1]);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 4, 6, 8]);
}

#[test]
fn oracle_shift_left_u32() {
    let a = make_u32_tensor(&[3], vec![1, 0x0000_0001, 0x0001_0000]);
    let b = make_u32_tensor(&[3], vec![16, 31, 15]);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(
        extract_u32_vec(&result),
        vec![0x0001_0000, 0x8000_0000, 0x8000_0000]
    );
}

#[test]
fn oracle_shift_left_zero() {
    // 0 << n = 0
    let a = make_i64_tensor(&[3], vec![0, 0, 0]);
    let b = make_i64_tensor(&[3], vec![1, 10, 63]);
    let result = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0]);
}

// ======================== ShiftRightArithmetic Tests ========================

#[test]
fn oracle_shift_right_arithmetic_scalar() {
    // 16 >> 2 = 4
    let a = Value::scalar_i64(16);
    let b = Value::scalar_i64(2);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4]);
}

#[test]
fn oracle_shift_right_arithmetic_by_zero() {
    // x >> 0 = x
    let a = Value::scalar_i64(42);
    let b = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_shift_right_arithmetic_negative() {
    // -16 >> 2 = -4 (preserves sign)
    let a = Value::scalar_i64(-16);
    let b = Value::scalar_i64(2);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-4]);
}

#[test]
fn oracle_shift_right_arithmetic_negative_fills_ones() {
    // -1 >> any = -1 (all bits are 1, sign extension keeps them 1)
    let a = make_i64_tensor(&[3], vec![-1, -1, -1]);
    let b = make_i64_tensor(&[3], vec![1, 32, 63]);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1]);
}

#[test]
fn oracle_shift_right_arithmetic_1d() {
    let a = make_i64_tensor(&[4], vec![16, 32, 64, 128]);
    let b = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![8, 8, 8, 8]);
}

#[test]
fn oracle_shift_right_arithmetic_2d() {
    let a = make_i64_tensor(&[2, 2], vec![8, -8, 16, -16]);
    let b = make_i64_tensor(&[2, 2], vec![1, 1, 2, 2]);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4, -4, 4, -4]);
}

#[test]
fn oracle_shift_right_arithmetic_divide_by_powers_of_two() {
    // x >> n = x / 2^n (for positive numbers)
    let a = make_i64_tensor(&[4], vec![100, 100, 100, 100]);
    let b = make_i64_tensor(&[4], vec![0, 1, 2, 3]);
    let result = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![100, 50, 25, 12]);
}

// ======================== ShiftRightLogical Tests ========================

#[test]
fn oracle_shift_right_logical_scalar() {
    // 16 >>> 2 = 4
    let a = Value::scalar_i64(16);
    let b = Value::scalar_i64(2);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4]);
}

#[test]
fn oracle_shift_right_logical_by_zero() {
    // x >>> 0 = x
    let a = Value::scalar_i64(42);
    let b = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_shift_right_logical_fills_zeros() {
    // -1 >>> 1 fills with 0, giving a large positive number
    // In i64: -1 = 0xFFFF_FFFF_FFFF_FFFF, >>> 1 = 0x7FFF_FFFF_FFFF_FFFF
    let a = Value::scalar_i64(-1);
    let b = Value::scalar_i64(1);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    let val = extract_i64_vec(&result)[0];
    assert!(val > 0); // Logical shift makes it positive
    assert_eq!(val, i64::MAX);
}

#[test]
fn oracle_shift_right_logical_1d() {
    let a = make_i64_tensor(&[4], vec![256, 128, 64, 32]);
    let b = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![128, 32, 8, 2]);
}

#[test]
fn oracle_shift_right_logical_u32() {
    let a = make_u32_tensor(&[3], vec![0xFFFF_FFFF, 0x8000_0000, 16]);
    let b = make_u32_tensor(&[3], vec![4, 31, 2]);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_u32_vec(&result), vec![0x0FFF_FFFF, 1, 4]);
}

#[test]
fn oracle_shift_right_logical_2d() {
    let a = make_i64_tensor(&[2, 2], vec![100, 200, 300, 400]);
    let b = make_i64_tensor(&[2, 2], vec![2, 2, 2, 2]);
    let result = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![25, 50, 75, 100]);
}

// ======================== Comparison Tests ========================

#[test]
fn oracle_shift_arithmetic_vs_logical_positive() {
    // For positive numbers, arithmetic and logical shift are the same
    let a = make_i64_tensor(&[3], vec![100, 200, 300]);
    let b = make_i64_tensor(&[3], vec![2, 2, 2]);
    let arith = eval_primitive(
        Primitive::ShiftRightArithmetic,
        &[a.clone(), b.clone()],
        &no_params(),
    )
    .unwrap();
    let logic = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&arith), extract_i64_vec(&logic));
}

#[test]
fn oracle_shift_arithmetic_vs_logical_negative() {
    // For negative numbers, they differ
    let a = Value::scalar_i64(-100);
    let b = Value::scalar_i64(2);
    let arith = eval_primitive(
        Primitive::ShiftRightArithmetic,
        &[a.clone(), b.clone()],
        &no_params(),
    )
    .unwrap();
    let logic = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
    let arith_val = extract_i64_vec(&arith)[0];
    let logic_val = extract_i64_vec(&logic)[0];
    assert!(arith_val < 0); // Arithmetic preserves sign
    assert!(logic_val > 0); // Logical makes it positive
}

#[test]
fn oracle_shift_left_right_inverse() {
    // (x << n) >> n = x for small shifts
    let a = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[4], vec![5, 5, 5, 5]);
    let shifted_left = eval_primitive(Primitive::ShiftLeft, &[a, b.clone()], &no_params()).unwrap();
    let shifted_back = eval_primitive(
        Primitive::ShiftRightArithmetic,
        &[shifted_left, b],
        &no_params(),
    )
    .unwrap();
    assert_eq!(extract_i64_vec(&shifted_back), vec![1, 2, 3, 4]);
}
