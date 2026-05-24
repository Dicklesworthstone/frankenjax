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

#![allow(clippy::approx_constant)]

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
        _ => unreachable!("expected tensor"),
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
        _ => unreachable!("expected tensor"),
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
    for x in [0.0, 1.0, -1.0, 3.17, -2.718, 100.0] {
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
    for x in [1.0, -1.0, 3.17, 100.0] {
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
        let xy = extract_f64_scalar(
            &eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &no_params()).unwrap(),
        );
        let yx =
            extract_f64_scalar(&eval_primitive(Primitive::Mul, &[b, a], &no_params()).unwrap());
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
    for x in [0.0, 1.0, -1.0, 3.17, -2.718, 100.0] {
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
        assert_close(
            extract_f64_scalar(&result),
            x,
            1e-14,
            &format!("({} * {}) / {} = {}", x, y, y, x),
        );
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

// ======================== METAMORPHIC: Mul(x, Reciprocal(x)) = 1 ========================

#[test]
fn metamorphic_mul_reciprocal_identity() {
    // Mul(x, Reciprocal(x)) = 1 for x != 0
    for x in [0.5, 1.0, 2.0, 3.0, 0.1, 10.0, -2.0, -0.5] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let recip = eval_primitive(
            Primitive::Reciprocal,
            std::slice::from_ref(&x_val),
            &no_params(),
        )
        .unwrap();
        let result = eval_primitive(Primitive::Mul, &[x_val, recip], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            1.0,
            1e-14,
            &format!("Mul({}, Reciprocal({})) = 1", x, x),
        );
    }
}

// ======================== METAMORPHIC: Div(x, y) = Mul(x, Reciprocal(y)) ========================

#[test]
fn metamorphic_div_equals_mul_reciprocal() {
    // Div(x, y) = Mul(x, Reciprocal(y)) for y != 0
    for (x, y) in [(6.0, 2.0), (10.0, 5.0), (7.0, 3.0), (1.0, 0.1), (-6.0, 3.0)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        let div_direct = eval_primitive(
            Primitive::Div,
            &[x_val.clone(), y_val.clone()],
            &no_params(),
        )
        .unwrap();
        let recip_y = eval_primitive(Primitive::Reciprocal, &[y_val], &no_params()).unwrap();
        let mul_recip = eval_primitive(Primitive::Mul, &[x_val, recip_y], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&div_direct),
            extract_f64_scalar(&mul_recip),
            1e-14,
            &format!("Div({}, {}) = Mul({}, Reciprocal({}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Mul(Neg(x), y) = Neg(Mul(x, y)) ========================

#[test]
fn metamorphic_mul_neg_distributive() {
    // Mul(Neg(x), y) = Neg(Mul(x, y))
    for (x, y) in [(2.0, 3.0), (5.0, 7.0), (0.5, 2.0), (1.0, 1.0)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        // Mul(Neg(x), y)
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&x_val), &no_params()).unwrap();
        let mul_neg =
            eval_primitive(Primitive::Mul, &[neg_x, y_val.clone()], &no_params()).unwrap();

        // Neg(Mul(x, y))
        let mul_xy = eval_primitive(Primitive::Mul, &[x_val, y_val], &no_params()).unwrap();
        let neg_mul = eval_primitive(Primitive::Neg, &[mul_xy], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&mul_neg),
            extract_f64_scalar(&neg_mul),
            1e-14,
            &format!("Mul(Neg({}), {}) = Neg(Mul({}, {}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Div(Neg(x), y) = Neg(Div(x, y)) ========================

#[test]
fn metamorphic_div_neg_distributive() {
    // Div(Neg(x), y) = Neg(Div(x, y))
    for (x, y) in [(6.0, 2.0), (10.0, 5.0), (7.0, 3.0), (1.0, 0.5)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        // Div(Neg(x), y)
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&x_val), &no_params()).unwrap();
        let div_neg =
            eval_primitive(Primitive::Div, &[neg_x, y_val.clone()], &no_params()).unwrap();

        // Neg(Div(x, y))
        let div_xy = eval_primitive(Primitive::Div, &[x_val, y_val], &no_params()).unwrap();
        let neg_div = eval_primitive(Primitive::Neg, &[div_xy], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&div_neg),
            extract_f64_scalar(&neg_div),
            1e-14,
            &format!("Div(Neg({}), {}) = Neg(Div({}, {}))", x, y, x, y),
        );
    }
}

// ======================== BROADCAST TESTS ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

fn scalar_i64(v: i64) -> Value {
    Value::Scalar(Literal::I64(v))
}

#[test]
fn oracle_mul_scalar_tensor_broadcast() {
    let a = scalar_f64(2.0);
    let b = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn oracle_mul_tensor_scalar_broadcast() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = scalar_f64(3.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn oracle_div_scalar_tensor_broadcast() {
    let a = scalar_f64(12.0);
    let b = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![12.0, 6.0, 4.0, 3.0]);
}

#[test]
fn oracle_div_tensor_scalar_broadcast() {
    let a = make_f64_tensor(&[4], vec![12.0, 24.0, 36.0, 48.0]);
    let b = scalar_f64(12.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn oracle_mul_row_vector_broadcast() {
    // [1, 3] * [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 4.0, 6.0, 3.0, 6.0, 9.0]);
}

#[test]
fn oracle_mul_column_vector_broadcast() {
    // [2, 1] * [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
    );
}

#[test]
fn oracle_div_row_vector_broadcast() {
    // [2, 3] / [1, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 3], vec![6.0, 12.0, 18.0, 12.0, 24.0, 36.0]);
    let b = make_f64_tensor(&[1, 3], vec![2.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 4.0, 3.0, 6.0, 8.0, 6.0]);
}

#[test]
fn oracle_mul_different_ranks_broadcast() {
    // [3] * [2, 3] -> [2, 3] (1D broadcast against 2D)
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    );
}

#[test]
fn oracle_div_different_ranks_broadcast() {
    // [2, 3] / [3] -> [2, 3]
    let a = make_f64_tensor(&[2, 3], vec![12.0, 12.0, 12.0, 24.0, 24.0, 24.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![6.0, 4.0, 3.0, 12.0, 8.0, 6.0]
    );
}

#[test]
fn oracle_mul_i64_scalar_tensor_broadcast() {
    let a = scalar_i64(3);
    let b = make_i64_tensor(&[3], vec![2, 4, 6]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 12, 18]);
}

#[test]
fn oracle_div_i64_tensor_scalar_broadcast() {
    let a = make_i64_tensor(&[3], vec![6, 12, 18]);
    let b = scalar_i64(3);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![2, 4, 6]);
}

#[test]
fn oracle_mul_incompatible_shapes_error() {
    // [2] * [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_div_incompatible_shapes_error() {
    // [2] / [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_mul_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_vec(dtype, &values);
        let b = make_vec(dtype, &[2.0_f64, 3.0, 4.0]);
        let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "mul {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_div_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let values = [1.0_f64, 4.0, 9.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_vec(dtype, &values);
        let b = make_vec(dtype, &[2.0_f64, 2.0, 3.0]);
        let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "div {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================

fn make_complex64_scalar(re: f32, im: f32) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![] },
            vec![Literal::from_complex64(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex128_scalar(re: f64, im: f64) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![] },
            vec![Literal::from_complex128(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex64_tensor(shape: &[u32], pairs: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

fn assert_complex128_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

#[test]
fn oracle_mul_complex64_basic() {
    // (2+3i) * (4+5i) = (2*4 - 3*5) + (2*5 + 3*4)i = (8-15) + (10+12)i = -7 + 22i
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(4.0, 5.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (-7.0, 22.0), 1e-5, "(2+3i)*(4+5i)");
}

#[test]
fn oracle_mul_complex64_identity() {
    // z * 1 = z
    let z = make_complex64_scalar(3.0, 4.0);
    let one = make_complex64_scalar(1.0, 0.0);
    let result = eval_primitive(Primitive::Mul, &[z, one], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "(3+4i)*1");
}

#[test]
fn oracle_mul_complex64_zero() {
    // z * 0 = 0
    let z = make_complex64_scalar(3.0, 4.0);
    let zero = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Mul, &[z, zero], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 0.0), 1e-6, "(3+4i)*0");
}

#[test]
fn oracle_mul_complex64_i_squared() {
    // i * i = -1
    let i = make_complex64_scalar(0.0, 1.0);
    let result = eval_primitive(Primitive::Mul, &[i.clone(), i], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (-1.0, 0.0), 1e-6, "i*i = -1");
}

#[test]
fn oracle_mul_complex64_conjugate_product() {
    // z * conj(z) = |z|^2 (real)
    // (3+4i) * (3-4i) = 9 + 16 = 25
    let z = make_complex64_scalar(3.0, 4.0);
    let conj_z = make_complex64_scalar(3.0, -4.0);
    let result = eval_primitive(Primitive::Mul, &[z, conj_z], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (25.0, 0.0), 1e-5, "(3+4i)*(3-4i) = 25");
}

#[test]
fn oracle_mul_complex64_vector() {
    let a = make_complex64_tensor(&[3], vec![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
    let b = make_complex64_tensor(&[3], vec![(2.0, 0.0), (0.0, 1.0), (1.0, -1.0)]);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    // 1 * 2 = 2
    assert_complex64_close(vals[0], (2.0, 0.0), 1e-5, "1*2");
    // i * i = -1
    assert_complex64_close(vals[1], (-1.0, 0.0), 1e-5, "i*i");
    // (1+i)*(1-i) = 1 + 1 = 2
    assert_complex64_close(vals[2], (2.0, 0.0), 1e-5, "(1+i)*(1-i)");
}

#[test]
fn oracle_div_complex64_basic() {
    // (3+4i) / (1+2i) = (3+4i)(1-2i) / |1+2i|^2 = (3+8 + (4-6)i) / 5 = (11 - 2i)/5 = 2.2 - 0.4i
    let a = make_complex64_scalar(3.0, 4.0);
    let b = make_complex64_scalar(1.0, 2.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (2.2, -0.4), 1e-5, "(3+4i)/(1+2i)");
}

#[test]
fn oracle_div_complex64_identity() {
    // z / 1 = z
    let z = make_complex64_scalar(3.0, 4.0);
    let one = make_complex64_scalar(1.0, 0.0);
    let result = eval_primitive(Primitive::Div, &[z, one], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "(3+4i)/1");
}

#[test]
fn oracle_div_complex64_self() {
    // z / z = 1 for z != 0
    let z = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Div, &[z.clone(), z], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (1.0, 0.0), 1e-5, "(3+4i)/(3+4i) = 1");
}

#[test]
fn oracle_div_complex64_by_i() {
    // z / i = z * (-i) = (a+bi) * (-i) = b - ai
    // (3+4i) / i = 4 - 3i
    let z = make_complex64_scalar(3.0, 4.0);
    let i = make_complex64_scalar(0.0, 1.0);
    let result = eval_primitive(Primitive::Div, &[z, i], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (4.0, -3.0), 1e-5, "(3+4i)/i");
}

#[test]
fn oracle_mul_div_complex64_inverse() {
    // (z * w) / w = z
    let z = make_complex64_scalar(2.0, 3.0);
    let w = make_complex64_scalar(4.0, 5.0);
    let zw = eval_primitive(Primitive::Mul, &[z.clone(), w.clone()], &no_params()).unwrap();
    let result = eval_primitive(Primitive::Div, &[zw, w], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (2.0, 3.0), 1e-4, "(z*w)/w = z");
}

#[test]
fn oracle_mul_complex128_basic() {
    // (2+3i) * (4+5i) = -7 + 22i
    let a = make_complex128_scalar(2.0, 3.0);
    let b = make_complex128_scalar(4.0, 5.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (-7.0, 22.0), 1e-12, "(2+3i)*(4+5i) Complex128");
}

#[test]
fn oracle_div_complex128_basic() {
    // (3+4i) / (1+2i) = 2.2 - 0.4i
    let a = make_complex128_scalar(3.0, 4.0);
    let b = make_complex128_scalar(1.0, 2.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (2.2, -0.4), 1e-12, "(3+4i)/(1+2i) Complex128");
}

#[test]
fn oracle_mul_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_mul_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_div_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_div_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
