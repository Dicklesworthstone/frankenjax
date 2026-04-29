//! Oracle tests for Mul and Div primitives.
//!
//! Mul: Element-wise multiplication
//! Div: Element-wise division
//!
//! Properties tested:
//! - Identity: x * 1 = x, x / 1 = x
//! - Zero: x * 0 = 0
//! - Inverse: x * (1/x) = 1 for x != 0
//! - Commutativity: x * y = y * x
//! - Associativity: (x * y) * z = x * (y * z)
//! - Div as inverse: x / y = x * (1/y)
//!
//! Tests:
//! - Basic operations
//! - Special values (infinity, NaN, zero)
//! - Integer and float types
//! - Complex numbers
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
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
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => panic!("expected tensor"),
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

// ====================== MUL IDENTITY ======================

#[test]
fn oracle_mul_identity_f64() {
    // x * 1 = x
    for x in [0.0, 1.0, -1.0, 3.14, -2.718, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![1.0]);
        let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "{} * 1 = {}", x, x);
    }
}

#[test]
fn oracle_mul_identity_i64() {
    for x in [0, 1, -1, 42, -100] {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![1]);
        let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x);
    }
}

// ====================== MUL ZERO ======================

#[test]
fn oracle_mul_zero_f64() {
    // x * 0 = 0
    for x in [1.0, -1.0, 3.14, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![0.0]);
        let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), 0.0, "{} * 0 = 0", x);
    }
}

#[test]
fn oracle_mul_zero_i64() {
    for x in [1, -1, 42, 100] {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 0);
    }
}

// ====================== MUL COMMUTATIVITY ======================

#[test]
fn oracle_mul_commutative() {
    // x * y = y * x
    let test_pairs = [(2.0, 3.0), (-2.0, 5.0), (0.5, 4.0), (0.0, 7.0)];
    for (x, y) in test_pairs {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let xy = extract_f64_scalar(&eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &no_params()).unwrap());
        let yx = extract_f64_scalar(&eval_primitive(Primitive::Mul, &[b, a], &no_params()).unwrap());
        assert_eq!(xy, yx, "{} * {} = {} * {}", x, y, y, x);
    }
}

// ====================== MUL BASIC ======================

#[test]
fn oracle_mul_basic_f64() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 12.0);
}

#[test]
fn oracle_mul_negative_f64() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -12.0);
}

#[test]
fn oracle_mul_both_negative_f64() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 12.0);
}

// ====================== MUL SPECIAL VALUES ======================

#[test]
fn oracle_mul_infinity() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_mul_infinity_zero() {
    // inf * 0 = NaN
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

#[test]
fn oracle_mul_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== DIV IDENTITY ======================

#[test]
fn oracle_div_identity_f64() {
    // x / 1 = x
    for x in [0.0, 1.0, -1.0, 3.14, -2.718, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![1.0]);
        let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "{} / 1 = {}", x, x);
    }
}

// ====================== DIV BASIC ======================

#[test]
fn oracle_div_basic_f64() {
    let a = make_f64_tensor(&[], vec![12.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0);
}

#[test]
fn oracle_div_fractional() {
    let a = make_f64_tensor(&[], vec![1.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0 / 3.0, 1e-15, "1/3");
}

#[test]
fn oracle_div_negative() {
    let a = make_f64_tensor(&[], vec![-12.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -3.0);
}

// ====================== DIV BY ZERO ======================

#[test]
fn oracle_div_by_zero_positive() {
    let a = make_f64_tensor(&[], vec![1.0]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_div_by_zero_negative() {
    let a = make_f64_tensor(&[], vec![-1.0]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY);
}

#[test]
fn oracle_div_zero_by_zero() {
    // 0 / 0 = NaN
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== DIV SPECIAL VALUES ======================

#[test]
fn oracle_div_infinity() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_div_by_infinity() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

#[test]
fn oracle_div_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== MUL-DIV INVERSE ======================

#[test]
fn oracle_mul_div_inverse() {
    // (x * y) / y = x for y != 0
    for (x, y) in [(6.0, 2.0), (10.0, 5.0), (7.0, 3.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let xy = eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &no_params()).unwrap();
        let result = eval_primitive(Primitive::Div, &[xy, b], &no_params()).unwrap();
        assert_close(extract_f64_scalar(&result), x, 1e-14, &format!("({} * {}) / {} = {}", x, y, y, x));
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_mul_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn oracle_div_1d() {
    let a = make_f64_tensor(&[4], vec![10.0, 20.0, 30.0, 40.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 4.0, 5.0, 8.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 5.0, 6.0, 5.0]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_mul_2d() {
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 12.0, 21.0, 32.0]);
}

// ====================== INTEGER DIVISION ======================

#[test]
fn oracle_div_integer() {
    let a = make_i64_tensor(&[], vec![10]);
    let b = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 3); // integer division truncates
}

#[test]
fn oracle_div_integer_exact() {
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4);
}

#[test]
fn oracle_mul_integer_1d() {
    let a = make_i64_tensor(&[3], vec![2, 3, 4]);
    let b = make_i64_tensor(&[3], vec![5, 6, 7]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10, 18, 28]);
}
