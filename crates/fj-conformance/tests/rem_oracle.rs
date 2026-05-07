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

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff={}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

// ======================== METAMORPHIC: Rem(Neg(x), y) = Neg(Rem(x, y)) ========================

#[test]
fn metamorphic_rem_neg_dividend_equals_neg_rem() {
    // Rem(Neg(x), y) = Neg(Rem(x, y)) for IEEE remainder semantics
    for (x, y) in [(7.0, 3.0), (10.0, 4.0), (15.5, 2.5), (1.0, 0.3)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        // Rem(Neg(x), y)
        let neg_x = eval_primitive(Primitive::Neg, std::slice::from_ref(&x_val), &no_params()).unwrap();
        let rem_neg_x = eval_primitive(Primitive::Rem, &[neg_x, y_val.clone()], &no_params()).unwrap();

        // Neg(Rem(x, y))
        let rem_x = eval_primitive(Primitive::Rem, &[x_val, y_val], &no_params()).unwrap();
        let neg_rem_x = eval_primitive(Primitive::Neg, &[rem_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&rem_neg_x),
            extract_f64_scalar(&neg_rem_x),
            1e-14,
            &format!("Rem(Neg({}), {}) = Neg(Rem({}, {}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Rem(x, y) + Rem(Neg(x), y) = 0 ========================

#[test]
fn metamorphic_rem_complementary_sum_zero() {
    // Rem(x, y) + Rem(-x, y) = 0 (remainders are additive inverses)
    for (x, y) in [(7.0, 3.0), (10.0, 4.0), (15.5, 2.5)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        let rem_pos = eval_primitive(Primitive::Rem, &[x_val.clone(), y_val.clone()], &no_params()).unwrap();
        let neg_x = eval_primitive(Primitive::Neg, &[x_val], &no_params()).unwrap();
        let rem_neg = eval_primitive(Primitive::Rem, &[neg_x, y_val], &no_params()).unwrap();
        let sum = eval_primitive(Primitive::Add, &[rem_pos, rem_neg], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sum),
            0.0,
            1e-14,
            &format!("Rem({}, {}) + Rem(-{}, {}) = 0", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Rem(x, Neg(y)) = Rem(x, y) ========================

#[test]
fn metamorphic_rem_neg_divisor_invariant() {
    // Rem(x, Neg(y)) = Rem(x, y) - sign follows dividend, not divisor
    for (x, y) in [(7.0, 3.0), (10.0, 4.0), (15.5, 2.5), (1.0, 0.3)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        let rem_pos_divisor = eval_primitive(Primitive::Rem, &[x_val.clone(), y_val.clone()], &no_params()).unwrap();
        let neg_y = eval_primitive(Primitive::Neg, &[y_val], &no_params()).unwrap();
        let rem_neg_divisor = eval_primitive(Primitive::Rem, &[x_val, neg_y], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&rem_pos_divisor),
            extract_f64_scalar(&rem_neg_divisor),
            1e-14,
            &format!("Rem({}, {}) = Rem({}, Neg({}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: a - Mul(Div(a, b), b) = Rem(a, b) ========================

#[test]
fn metamorphic_rem_division_identity_primitives() {
    // a - Mul(Floor(Div(a, b)), b) ≈ Rem(a, b) for positive values
    // Using Floor to get truncated quotient
    for (a, b) in [(7.0, 3.0), (10.0, 4.0), (17.0, 5.0)] {
        let a_val = make_f64_tensor(&[], vec![a]);
        let b_val = make_f64_tensor(&[], vec![b]);

        // Direct Rem
        let rem_direct = eval_primitive(Primitive::Rem, &[a_val.clone(), b_val.clone()], &no_params()).unwrap();

        // a - Floor(a/b) * b
        let quotient = eval_primitive(Primitive::Div, &[a_val.clone(), b_val.clone()], &no_params()).unwrap();
        let floored = eval_primitive(Primitive::Floor, &[quotient], &no_params()).unwrap();
        let product = eval_primitive(Primitive::Mul, &[floored, b_val], &no_params()).unwrap();
        let rem_computed = eval_primitive(Primitive::Sub, &[a_val, product], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&rem_direct),
            extract_f64_scalar(&rem_computed),
            1e-10,
            &format!("{} - Floor({}/{}) * {} = Rem({}, {})", a, a, b, b, a, b),
        );
    }
}
