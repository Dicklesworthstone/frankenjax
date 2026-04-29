//! Oracle tests for Rem (remainder/modulo) primitive.
//!
//! Tests remainder semantics:
//! - Integer: truncated remainder (sign follows dividend)
//! - Float: IEEE 754 remainder (sign follows dividend)
//! - Division by zero: integers return 0, floats return NaN

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
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
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

// ======================== Basic Integer Remainder ========================

#[test]
fn oracle_rem_i64_basic() {
    // 7 % 3 = 1, 8 % 3 = 2, 9 % 3 = 0
    let a = make_i64_tensor(&[3], vec![7, 8, 9]);
    let b = make_i64_tensor(&[3], vec![3, 3, 3]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 0]);
}

#[test]
fn oracle_rem_i64_same_values() {
    // x % x = 0 for any non-zero x
    let a = make_i64_tensor(&[4], vec![5, 10, 100, 1]);
    let b = make_i64_tensor(&[4], vec![5, 10, 100, 1]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

#[test]
fn oracle_rem_i64_smaller_dividend() {
    // When dividend < divisor, remainder = dividend
    let a = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[4], vec![10, 10, 10, 10]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

#[test]
fn oracle_rem_i64_by_one() {
    // x % 1 = 0 for any x
    let a = make_i64_tensor(&[4], vec![0, 7, 100, -5]);
    let b = make_i64_tensor(&[4], vec![1, 1, 1, 1]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

// ======================== Negative Integer Remainder ========================

#[test]
fn oracle_rem_i64_negative_dividend() {
    // Truncated remainder: sign follows dividend
    // -7 % 3 = -1 (because -7 = -2*3 + (-1))
    let a = make_i64_tensor(&[4], vec![-7, -8, -9, -10]);
    let b = make_i64_tensor(&[4], vec![3, 3, 3, 3]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -2, 0, -1]);
}

#[test]
fn oracle_rem_i64_negative_divisor() {
    // 7 % -3 = 1 (sign follows dividend, which is positive)
    let a = make_i64_tensor(&[4], vec![7, 8, 9, 10]);
    let b = make_i64_tensor(&[4], vec![-3, -3, -3, -3]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 0, 1]);
}

#[test]
fn oracle_rem_i64_both_negative() {
    // -7 % -3 = -1 (sign follows dividend)
    let a = make_i64_tensor(&[4], vec![-7, -8, -9, -10]);
    let b = make_i64_tensor(&[4], vec![-3, -3, -3, -3]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -2, 0, -1]);
}

// ======================== Division by Zero (Integer) ========================

#[test]
fn oracle_rem_i64_divide_by_zero() {
    // Integer divide by zero returns 0 (checked_rem behavior)
    let a = make_i64_tensor(&[3], vec![5, 10, -7]);
    let b = make_i64_tensor(&[3], vec![0, 0, 0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0]);
}

#[test]
fn oracle_rem_i64_mixed_zero_divisor() {
    // Some divisors zero, some not
    let a = make_i64_tensor(&[4], vec![10, 10, 10, 10]);
    let b = make_i64_tensor(&[4], vec![3, 0, 7, 0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 3, 0]);
}

// ======================== Float Remainder ========================

#[test]
fn oracle_rem_f64_basic() {
    // 7.5 % 2.0 = 1.5
    let a = make_f64_tensor(&[3], vec![7.5, 8.0, 10.5]);
    let b = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 2.5).abs() < 1e-10);
}

#[test]
fn oracle_rem_f64_fractional() {
    // Remainder with fractional divisor
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![0.3, 0.7, 1.1]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // 1.0 % 0.3 = 0.1 (approximately)
    assert!((vals[0] - 0.1).abs() < 1e-10);
    // 2.0 % 0.7 = 0.6 (2.0 = 2*0.7 + 0.6)
    assert!((vals[1] - 0.6).abs() < 1e-10);
    // 3.0 % 1.1 = 0.8 (3.0 = 2*1.1 + 0.8)
    assert!((vals[2] - 0.8).abs() < 1e-10);
}

#[test]
fn oracle_rem_f64_negative_dividend() {
    // IEEE 754: sign follows dividend
    // -7.5 % 2.0 = -1.5
    let a = make_f64_tensor(&[3], vec![-7.5, -8.0, -10.5]);
    let b = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.5)).abs() < 1e-10);
    assert!((vals[1] - (-2.0)).abs() < 1e-10);
    assert!((vals[2] - (-2.5)).abs() < 1e-10);
}

#[test]
fn oracle_rem_f64_negative_divisor() {
    // 7.5 % -2.0 = 1.5 (sign follows dividend)
    let a = make_f64_tensor(&[3], vec![7.5, 8.0, 10.5]);
    let b = make_f64_tensor(&[3], vec![-2.0, -3.0, -4.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 2.5).abs() < 1e-10);
}

// ======================== Float Division by Zero ========================

#[test]
fn oracle_rem_f64_divide_by_zero() {
    // Float divide by zero returns NaN
    let a = make_f64_tensor(&[3], vec![5.0, -10.0, 0.0]);
    let b = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
    assert!(vals[1].is_nan());
    assert!(vals[2].is_nan());
}

// ======================== Float Special Values ========================

#[test]
fn oracle_rem_f64_infinity() {
    // inf % x = NaN, x % inf = x
    let a = make_f64_tensor(&[2], vec![f64::INFINITY, 5.0]);
    let b = make_f64_tensor(&[2], vec![2.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
    assert!((vals[1] - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_rem_f64_nan_propagates() {
    // NaN % x = NaN, x % NaN = NaN
    let a = make_f64_tensor(&[2], vec![f64::NAN, 5.0]);
    let b = make_f64_tensor(&[2], vec![2.0, f64::NAN]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
    assert!(vals[1].is_nan());
}

#[test]
fn oracle_rem_f64_negative_zero() {
    // -0.0 % x = -0.0 (sign preserved)
    let a = make_f64_tensor(&[1], vec![-0.0]);
    let b = make_f64_tensor(&[1], vec![1.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] == 0.0);
    assert!(vals[0].is_sign_negative());
}

// ======================== 2D Tensors ========================

#[test]
fn oracle_rem_2d_i64() {
    // [2, 3] tensor remainder
    let a = make_i64_tensor(&[2, 3], vec![10, 11, 12, 13, 14, 15]);
    let b = make_i64_tensor(&[2, 3], vec![3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    // 10%3=1, 11%4=3, 12%5=2, 13%6=1, 14%7=0, 15%8=7
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 2, 1, 0, 7]);
}

#[test]
fn oracle_rem_2d_f64() {
    let a = make_f64_tensor(&[2, 2], vec![7.5, 8.5, 9.5, 10.5]);
    let b = make_f64_tensor(&[2, 2], vec![2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    // 7.5%2=1.5, 8.5%3=2.5, 9.5%4=1.5, 10.5%5=0.5
    assert!((vals[0] - 1.5).abs() < 1e-10);
    assert!((vals[1] - 2.5).abs() < 1e-10);
    assert!((vals[2] - 1.5).abs() < 1e-10);
    assert!((vals[3] - 0.5).abs() < 1e-10);
}

// ======================== 3D Tensors ========================

#[test]
fn oracle_rem_3d_i64() {
    let a = make_i64_tensor(&[2, 2, 2], vec![10, 20, 30, 40, 50, 60, 70, 80]);
    let b = make_i64_tensor(&[2, 2, 2], vec![3, 7, 11, 13, 17, 19, 23, 29]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    // 10%3=1, 20%7=6, 30%11=8, 40%13=1, 50%17=16, 60%19=3, 70%23=1, 80%29=22
    assert_eq!(extract_i64_vec(&result), vec![1, 6, 8, 1, 16, 3, 1, 22]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_rem_zero_dividend() {
    // 0 % x = 0 for any non-zero x
    let a = make_i64_tensor(&[3], vec![0, 0, 0]);
    let b = make_i64_tensor(&[3], vec![5, 100, -7]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0]);
}

#[test]
fn oracle_rem_scalar() {
    // Single element tensors
    let a = make_i64_tensor(&[], vec![17]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![]);
    assert_eq!(extract_i64_vec(&result), vec![2]);
}

#[test]
fn oracle_rem_large_values() {
    // Test with large values that don't overflow
    let a = make_i64_tensor(&[2], vec![1_000_000_007, i64::MAX - 1]);
    let b = make_i64_tensor(&[2], vec![1000, i64::MAX]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 1_000_000_007 % 1000);
    assert_eq!(vals[1], i64::MAX - 1);
}

#[test]
fn oracle_rem_i64_min_edge() {
    // i64::MIN % -1 would overflow, should return 0 (checked_rem behavior)
    let a = make_i64_tensor(&[1], vec![i64::MIN]);
    let b = make_i64_tensor(&[1], vec![-1]);
    let result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

// ======================== Relationship: a = (a/b)*b + (a%b) ========================

#[test]
fn oracle_rem_division_identity_i64() {
    // Verify: a = (a/b)*b + (a%b) for truncated division
    let dividends = vec![17, -17, 17, -17, 100, -100];
    let divisors = vec![5, 5, -5, -5, 7, 7];
    let a = make_i64_tensor(&[6], dividends.clone());
    let b = make_i64_tensor(&[6], divisors.clone());
    let rem_result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let remainders = extract_i64_vec(&rem_result);

    for i in 0..6 {
        let quotient = dividends[i] / divisors[i];
        let expected_rem = dividends[i] - quotient * divisors[i];
        assert_eq!(
            remainders[i], expected_rem,
            "a={}, b={}, got rem={}, expected={}",
            dividends[i], divisors[i], remainders[i], expected_rem
        );
    }
}

#[test]
fn oracle_rem_division_identity_f64() {
    // Verify: a % b has same sign as a, and |a % b| < |b|
    let dividends = vec![7.5, -7.5, 7.5, -7.5];
    let divisors = vec![2.0, 2.0, -2.0, -2.0];
    let a = make_f64_tensor(&[4], dividends.clone());
    let b = make_f64_tensor(&[4], divisors.clone());
    let rem_result = eval_primitive(Primitive::Rem, &[a, b], &no_params()).unwrap();
    let remainders = extract_f64_vec(&rem_result);

    for i in 0..4 {
        // Sign of remainder matches sign of dividend
        if dividends[i] >= 0.0 {
            assert!(
                remainders[i] >= 0.0,
                "positive dividend should give non-negative remainder"
            );
        } else {
            assert!(
                remainders[i] <= 0.0,
                "negative dividend should give non-positive remainder"
            );
        }
        // |remainder| < |divisor|
        assert!(
            remainders[i].abs() < divisors[i].abs(),
            "|remainder| should be < |divisor|"
        );
    }
}
