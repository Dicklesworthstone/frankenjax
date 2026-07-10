//! Oracle tests for Add and Sub primitives.
//!
//! Add: Element-wise addition
//! Sub: Element-wise subtraction
//!
//! Properties tested:
//! - Identity: x + 0 = x, x - 0 = x
//! - Commutativity: x + y = y + x
//! - Associativity: (x + y) + z = x + (y + z)
//! - Inverse: x - x = 0
//! - Negation relationship: x - y = x + (-y)
//!
//! Tests:
//! - Basic operations
//! - Special values (infinity, NaN, zero)
//! - Integer and float types
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

// ====================== ADD IDENTITY ======================

#[test]
fn oracle_add_identity_f64() {
    // x + 0 = x
    for x in [0.0, 1.0, -1.0, 3.17, -2.718, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![0.0]);
        let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "{} + 0 = {}", x, x);
    }
}

#[test]
fn oracle_add_identity_i64() {
    for x in [0, 1, -1, 42, -100] {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x);
    }
}

// ====================== ADD COMMUTATIVITY ======================

#[test]
fn oracle_add_commutative_f64() {
    // x + y = y + x
    let test_pairs = [(2.0, 3.0), (-2.0, 5.0), (0.5, 4.0), (0.0, 7.0)];
    for (x, y) in test_pairs {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let xy = extract_f64_scalar(
            &eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap(),
        );
        let yx =
            extract_f64_scalar(&eval_primitive(Primitive::Add, &[b, a], &no_params()).unwrap());
        assert_eq!(xy, yx, "{} + {} = {} + {}", x, y, y, x);
    }
}

#[test]
fn oracle_add_commutative_i64() {
    let test_pairs = [(2, 3), (-2, 5), (0, 7), (100, -50)];
    for (x, y) in test_pairs {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![y]);
        let xy = extract_i64_scalar(
            &eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap(),
        );
        let yx =
            extract_i64_scalar(&eval_primitive(Primitive::Add, &[b, a], &no_params()).unwrap());
        assert_eq!(xy, yx);
    }
}

// ====================== ADD ASSOCIATIVITY ======================

#[test]
fn oracle_add_associative_f64() {
    // (x + y) + z = x + (y + z)
    let test_triples = [(1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (-1.0, 2.0, -3.0)];
    for (x, y, z) in test_triples {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let c = make_f64_tensor(&[], vec![z]);

        // (x + y) + z
        let xy = eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap();
        let lhs = extract_f64_scalar(
            &eval_primitive(Primitive::Add, &[xy, c.clone()], &no_params()).unwrap(),
        );

        // x + (y + z)
        let yz = eval_primitive(Primitive::Add, &[b, c], &no_params()).unwrap();
        let rhs =
            extract_f64_scalar(&eval_primitive(Primitive::Add, &[a, yz], &no_params()).unwrap());

        assert_close(
            lhs,
            rhs,
            1e-14,
            &format!("({} + {}) + {} = {} + ({} + {})", x, y, z, x, y, z),
        );
    }
}

// ====================== ADD BASIC ======================

#[test]
fn oracle_add_basic_f64() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 7.0);
}

#[test]
fn oracle_add_negative_f64() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_add_both_negative_f64() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -7.0);
}

// ====================== ADD SPECIAL VALUES ======================

#[test]
fn oracle_add_infinity() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_add_neg_infinity() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY);
}

#[test]
fn oracle_add_infinity_cancel() {
    // inf + (-inf) = NaN
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

#[test]
fn oracle_add_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== SUB IDENTITY ======================

#[test]
fn oracle_sub_identity_f64() {
    // x - 0 = x
    for x in [0.0, 1.0, -1.0, 3.17, -2.718, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![0.0]);
        let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "{} - 0 = {}", x, x);
    }
}

#[test]
fn oracle_sub_identity_i64() {
    for x in [0, 1, -1, 42, -100] {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x);
    }
}

// ====================== SUB SELF INVERSE ======================

#[test]
fn oracle_sub_self_zero_f64() {
    // x - x = 0
    for x in [0.0, 1.0, -1.0, 3.17, -2.718, 100.0] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), 0.0, "{} - {} = 0", x, x);
    }
}

#[test]
fn oracle_sub_self_zero_i64() {
    for x in [0, 1, -1, 42, -100] {
        let a = make_i64_tensor(&[], vec![x]);
        let b = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 0);
    }
}

// ====================== SUB BASIC ======================

#[test]
fn oracle_sub_basic_f64() {
    let a = make_f64_tensor(&[], vec![10.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 6.0);
}

#[test]
fn oracle_sub_negative_result() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![7.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -4.0);
}

// ====================== SUB SPECIAL VALUES ======================

#[test]
fn oracle_sub_infinity() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_sub_from_neg_infinity() {
    let a = make_f64_tensor(&[], vec![1.0]);
    let b = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY);
}

#[test]
fn oracle_sub_infinity_same() {
    // inf - inf = NaN
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

#[test]
fn oracle_sub_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== ADD-SUB RELATIONSHIP ======================

#[test]
fn oracle_add_sub_inverse() {
    // (x + y) - y = x
    for (x, y) in [(6.0, 2.0), (10.0, 5.0), (7.0, 3.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let xy = eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap();
        let result = eval_primitive(Primitive::Sub, &[xy, b], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x,
            1e-14,
            &format!("({} + {}) - {} = {}", x, y, y, x),
        );
    }
}

#[test]
fn oracle_sub_as_add_neg() {
    // x - y = x + (-y)
    for (x, y) in [(6.0, 2.0), (3.0, 7.0), (-1.0, -5.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let neg_b = make_f64_tensor(&[], vec![-y]);

        let sub_result = extract_f64_scalar(
            &eval_primitive(Primitive::Sub, &[a.clone(), b], &no_params()).unwrap(),
        );
        let add_neg_result =
            extract_f64_scalar(&eval_primitive(Primitive::Add, &[a, neg_b], &no_params()).unwrap());

        assert_eq!(
            sub_result, add_neg_result,
            "{} - {} = {} + (-{})",
            x, y, x, y
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_add_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![10.0, 20.0, 30.0, 40.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn oracle_sub_1d() {
    let a = make_f64_tensor(&[4], vec![10.0, 20.0, 30.0, 40.0]);
    let b = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![9.0, 18.0, 27.0, 36.0]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_add_2d() {
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn oracle_sub_2d() {
    let a = make_f64_tensor(&[2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let b = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![9.0, 18.0, 27.0, 36.0]);
}

// ====================== INTEGER OPERATIONS ======================

#[test]
fn oracle_add_integer_1d() {
    let a = make_i64_tensor(&[3], vec![1, 2, 3]);
    let b = make_i64_tensor(&[3], vec![10, 20, 30]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![11, 22, 33]);
}

#[test]
fn oracle_sub_integer_1d() {
    let a = make_i64_tensor(&[3], vec![100, 200, 300]);
    let b = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![99, 198, 297]);
}

// ======================== METAMORPHIC: Sub(x, y) = Add(x, Neg(y)) ========================

#[test]
fn metamorphic_sub_equals_add_neg() {
    // Sub(x, y) = Add(x, Neg(y)) using Neg primitive
    for (x, y) in [(6.0, 2.0), (10.0, 5.0), (3.0, 7.0), (-1.0, 4.0), (0.5, 0.3)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        // Sub(x, y)
        let sub_direct = eval_primitive(
            Primitive::Sub,
            &[x_val.clone(), y_val.clone()],
            &no_params(),
        )
        .unwrap();

        // Add(x, Neg(y))
        let neg_y = eval_primitive(Primitive::Neg, &[y_val], &no_params()).unwrap();
        let add_neg = eval_primitive(Primitive::Add, &[x_val, neg_y], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sub_direct),
            extract_f64_scalar(&add_neg),
            1e-14,
            &format!("Sub({}, {}) = Add({}, Neg({}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Add(x, Neg(x)) = 0 ========================

#[test]
fn metamorphic_add_neg_equals_zero() {
    // Add(x, Neg(x)) = 0 (additive inverse)
    for x in [0.5, 1.0, 2.0, 3.17, -2.0, -0.5, 100.0] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&x_val), &no_params()).unwrap();
        let result = eval_primitive(Primitive::Add, &[x_val, neg_x], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            0.0,
            1e-14,
            &format!("Add({}, Neg({})) = 0", x, x),
        );
    }
}

// ======================== METAMORPHIC: Add(Neg(x), Neg(y)) = Neg(Add(x, y)) ========================

#[test]
fn metamorphic_neg_distributes_over_add() {
    // Add(Neg(x), Neg(y)) = Neg(Add(x, y))
    for (x, y) in [(2.0, 3.0), (5.0, 7.0), (0.5, 2.0), (1.0, 1.0), (-1.0, 3.0)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let y_val = make_f64_tensor(&[], vec![y]);

        // Add(Neg(x), Neg(y))
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&x_val), &no_params()).unwrap();
        let neg_y =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&y_val), &no_params()).unwrap();
        let add_negs = eval_primitive(Primitive::Add, &[neg_x, neg_y], &no_params()).unwrap();

        // Neg(Add(x, y))
        let add_xy = eval_primitive(Primitive::Add, &[x_val, y_val], &no_params()).unwrap();
        let neg_add = eval_primitive(Primitive::Neg, &[add_xy], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&add_negs),
            extract_f64_scalar(&neg_add),
            1e-14,
            &format!("Add(Neg({}), Neg({})) = Neg(Add({}, {}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: Sub(0, x) = Neg(x) ========================

#[test]
fn metamorphic_sub_zero_equals_neg() {
    // Sub(0, x) = Neg(x)
    for x in [1.0, 2.0, 3.17, -2.0, 0.5, 100.0] {
        let zero = make_f64_tensor(&[], vec![0.0]);
        let x_val = make_f64_tensor(&[], vec![x]);

        let sub_zero =
            eval_primitive(Primitive::Sub, &[zero, x_val.clone()], &no_params()).unwrap();
        let neg_x = eval_primitive(Primitive::Neg, &[x_val], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sub_zero),
            extract_f64_scalar(&neg_x),
            1e-14,
            &format!("Sub(0, {}) = Neg({})", x, x),
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
fn oracle_add_scalar_tensor_broadcast() {
    let a = scalar_f64(10.0);
    let b = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn oracle_add_tensor_scalar_broadcast() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn oracle_sub_scalar_tensor_broadcast() {
    let a = scalar_f64(10.0);
    let b = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn oracle_sub_tensor_scalar_broadcast() {
    let a = make_f64_tensor(&[4], vec![10.0, 20.0, 30.0, 40.0]);
    let b = scalar_f64(5.0);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 15.0, 25.0, 35.0]);
}

#[test]
fn oracle_add_row_vector_broadcast() {
    // [1, 3] + [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
    );
}

#[test]
fn oracle_add_column_vector_broadcast() {
    // [2, 1] + [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![100.0, 200.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );
}

#[test]
fn oracle_sub_row_vector_broadcast() {
    // [2, 3] - [1, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let b = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
    );
}

#[test]
fn oracle_add_different_ranks_broadcast() {
    // [3] + [2, 3] -> [2, 3] (1D broadcast against 2D)
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
    );
}

#[test]
fn oracle_sub_different_ranks_broadcast() {
    // [2, 3] - [3] -> [2, 3]
    let a = make_f64_tensor(&[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
    );
}

#[test]
fn oracle_add_i64_scalar_tensor_broadcast() {
    let a = scalar_i64(100);
    let b = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![101, 102, 103]);
}

#[test]
fn oracle_sub_i64_tensor_scalar_broadcast() {
    let a = make_i64_tensor(&[3], vec![100, 200, 300]);
    let b = scalar_i64(50);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![50, 150, 250]);
}

#[test]
fn oracle_add_incompatible_shapes_error() {
    // [2] + [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_sub_incompatible_shapes_error() {
    // [2] - [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_add_preserves_all_float_dtypes() {
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
        let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "add {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_sub_preserves_all_float_dtypes() {
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

    let values = [5.0_f64, 3.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_vec(dtype, &values);
        let b = make_vec(dtype, &[2.0_f64, 1.0, 0.5]);
        let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "sub {dtype:?}: dtype mismatch");
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
fn oracle_add_complex64_basic() {
    // (2+3i) + (4+5i) = 6+8i
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(4.0, 5.0);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (6.0, 8.0), 1e-6, "(2+3i)+(4+5i)");
}

#[test]
fn oracle_add_complex64_identity() {
    // z + 0 = z
    let z = make_complex64_scalar(3.0, 4.0);
    let zero = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Add, &[z, zero], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "(3+4i)+0");
}

#[test]
fn oracle_add_complex64_commutative() {
    // z + w = w + z
    let z = make_complex64_scalar(2.0, 3.0);
    let w = make_complex64_scalar(4.0, 5.0);
    let zw = extract_complex64_scalar(
        &eval_primitive(Primitive::Add, &[z.clone(), w.clone()], &no_params()).unwrap(),
    );
    let wz =
        extract_complex64_scalar(&eval_primitive(Primitive::Add, &[w, z], &no_params()).unwrap());
    assert_complex64_close(zw, wz, 1e-6, "z+w = w+z");
}

#[test]
fn oracle_add_complex64_vector() {
    let a = make_complex64_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let b = make_complex64_tensor(&[3], vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    assert_complex64_close(vals[0], (11.0, 22.0), 1e-5, "add[0]");
    assert_complex64_close(vals[1], (33.0, 44.0), 1e-5, "add[1]");
    assert_complex64_close(vals[2], (55.0, 66.0), 1e-5, "add[2]");
}

#[test]
fn oracle_sub_complex64_basic() {
    // (5+7i) - (2+3i) = 3+4i
    let a = make_complex64_scalar(5.0, 7.0);
    let b = make_complex64_scalar(2.0, 3.0);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "(5+7i)-(2+3i)");
}

#[test]
fn oracle_sub_complex64_self_zero() {
    // z - z = 0
    let z = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sub, &[z.clone(), z], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 0.0), 1e-6, "(3+4i)-(3+4i) = 0");
}

#[test]
fn oracle_sub_complex64_identity() {
    // z - 0 = z
    let z = make_complex64_scalar(3.0, 4.0);
    let zero = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Sub, &[z, zero], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "(3+4i)-0");
}

#[test]
fn oracle_sub_complex64_vector() {
    let a = make_complex64_tensor(&[3], vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]);
    let b = make_complex64_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    assert_complex64_close(vals[0], (9.0, 18.0), 1e-5, "sub[0]");
    assert_complex64_close(vals[1], (27.0, 36.0), 1e-5, "sub[1]");
    assert_complex64_close(vals[2], (45.0, 54.0), 1e-5, "sub[2]");
}

#[test]
fn oracle_add_sub_complex64_inverse() {
    // (z + w) - w = z
    let z = make_complex64_scalar(2.0, 3.0);
    let w = make_complex64_scalar(4.0, 5.0);
    let zw = eval_primitive(Primitive::Add, &[z.clone(), w.clone()], &no_params()).unwrap();
    let result = eval_primitive(Primitive::Sub, &[zw, w], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (2.0, 3.0), 1e-5, "(z+w)-w = z");
}

#[test]
fn oracle_add_complex128_basic() {
    // (2+3i) + (4+5i) = 6+8i
    let a = make_complex128_scalar(2.0, 3.0);
    let b = make_complex128_scalar(4.0, 5.0);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (6.0, 8.0), 1e-12, "(2+3i)+(4+5i) Complex128");
}

#[test]
fn oracle_sub_complex128_basic() {
    // (5+7i) - (2+3i) = 3+4i
    let a = make_complex128_scalar(5.0, 7.0);
    let b = make_complex128_scalar(2.0, 3.0);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (3.0, 4.0), 1e-12, "(5+7i)-(2+3i) Complex128");
}

#[test]
fn oracle_add_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_add_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_sub_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_sub_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sub, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_add_u32_overflow_wraps_mod_2_pow_32() {
    // XLA/JAX unsigned arithmetic wraps mod 2^32. The u32 matmul path is tested
    // elsewhere; this guards the ELEMENTWISE add path (wrapping_add on u64 then
    // `as u32` truncation): u32::MAX + 1 = 0, and 0xFFFF_FFF0 + 0x20 = 0x10.
    let mk = |data: Vec<u32>| {
        Value::Tensor(
            TensorValue::new(
                DType::U32,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.into_iter().map(Literal::U32).collect(),
            )
            .unwrap(),
        )
    };
    let a = mk(vec![u32::MAX, 0xFFFF_FFF0]);
    let b = mk(vec![1, 0x20]);
    let result = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::U32, "u32 add preserves u32 dtype");
    let got: Vec<u32> = match &result {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_u64().unwrap() as u32)
            .collect(),
        _ => unreachable!("expected tensor"),
    };
    assert_eq!(
        got,
        vec![0, 0x10],
        "u32 add must wrap mod 2^32 (XLA unsigned)"
    );
}
