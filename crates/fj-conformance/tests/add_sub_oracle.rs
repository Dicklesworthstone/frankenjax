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
    for x in [0.0, 1.0, -1.0, 3.14, -2.718, 100.0] {
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
    for x in [0.0, 1.0, -1.0, 3.14, -2.718, 100.0] {
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
    for x in [0.0, 1.0, -1.0, 3.14, -2.718, 100.0] {
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
        let sub_direct = eval_primitive(Primitive::Sub, &[x_val.clone(), y_val.clone()], &no_params()).unwrap();

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
    for x in [0.5, 1.0, 2.0, 3.14, -2.0, -0.5, 100.0] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let neg_x = eval_primitive(Primitive::Neg, &[x_val.clone()], &no_params()).unwrap();
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
        let neg_x = eval_primitive(Primitive::Neg, &[x_val.clone()], &no_params()).unwrap();
        let neg_y = eval_primitive(Primitive::Neg, &[y_val.clone()], &no_params()).unwrap();
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
    for x in [1.0, 2.0, 3.14, -2.0, 0.5, 100.0] {
        let zero = make_f64_tensor(&[], vec![0.0]);
        let x_val = make_f64_tensor(&[], vec![x]);

        let sub_zero = eval_primitive(Primitive::Sub, &[zero, x_val.clone()], &no_params()).unwrap();
        let neg_x = eval_primitive(Primitive::Neg, &[x_val], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sub_zero),
            extract_f64_scalar(&neg_x),
            1e-14,
            &format!("Sub(0, {}) = Neg({})", x, x),
        );
    }
}
