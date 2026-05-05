//! Oracle tests for Comparison primitives: Eq, Ne, Lt, Le, Gt, Ge.
//!
//! These primitives perform elementwise comparisons, returning Bool tensors.
//!
//! Eq(a, b) = a == b
//! Ne(a, b) = a != b
//! Lt(a, b) = a < b
//! Le(a, b) = a <= b
//! Gt(a, b) = a > b
//! Ge(a, b) = a >= b
//!
//! Tests:
//! - Basic comparisons with scalars
//! - Negative numbers
//! - Zero comparisons
//! - Infinity handling
//! - NaN semantics (NaN comparisons are always false except Ne)
//! - Tensor shapes
//! - Relationship between operations

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

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => unreachable!("expected bool literal"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => unreachable!("expected tensor or bool scalar"),
    }
}

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            match &t.elements[0] {
                Literal::Bool(b) => *b,
                _ => unreachable!("expected bool literal"),
            }
        }
        Value::Scalar(Literal::Bool(b)) => *b,
        _ => unreachable!("expected bool"),
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

// ====================== EQ TESTS ======================

#[test]
fn oracle_eq_equal_values() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 == 5.0");
}

#[test]
fn oracle_eq_different_values() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "3.0 != 5.0");
}

#[test]
fn oracle_eq_zeros() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "0.0 == -0.0");
}

#[test]
fn oracle_eq_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "NaN != NaN");
}

#[test]
fn oracle_eq_infinity() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "inf == inf");
}

#[test]
fn oracle_eq_neg_infinity() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "-inf == -inf");
}

#[test]
fn oracle_eq_i64() {
    let a = make_i64_tensor(&[], vec![42]);
    let b = make_i64_tensor(&[], vec![42]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "42 == 42");
}

#[test]
fn oracle_eq_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, false]);
}

// ====================== NE TESTS ======================

#[test]
fn oracle_ne_different_values() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Ne, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "3.0 != 5.0");
}

#[test]
fn oracle_ne_equal_values() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Ne, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 == 5.0");
}

#[test]
fn oracle_ne_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Ne, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "NaN != NaN is true");
}

#[test]
fn oracle_ne_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Ne, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, false, true]);
}

// ====================== LT TESTS ======================

#[test]
fn oracle_lt_less_than() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "3.0 < 5.0");
}

#[test]
fn oracle_lt_greater_than() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 < 3.0 is false");
}

#[test]
fn oracle_lt_equal() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 < 5.0 is false");
}

#[test]
fn oracle_lt_negative() {
    let a = make_f64_tensor(&[], vec![-5.0]);
    let b = make_f64_tensor(&[], vec![-3.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "-5.0 < -3.0");
}

#[test]
fn oracle_lt_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "NaN < 5.0 is false");
}

#[test]
fn oracle_lt_infinity() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 < inf");
}

#[test]
fn oracle_lt_neg_infinity() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "-inf < 5.0");
}

#[test]
fn oracle_lt_i64() {
    let a = make_i64_tensor(&[], vec![3]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "3 < 5");
}

#[test]
fn oracle_lt_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, false, true]);
}

// ====================== LE TESTS ======================

#[test]
fn oracle_le_less_than() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "3.0 <= 5.0");
}

#[test]
fn oracle_le_equal() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 <= 5.0");
}

#[test]
fn oracle_le_greater_than() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 <= 3.0 is false");
}

#[test]
fn oracle_le_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "NaN <= 5.0 is false");
}

#[test]
fn oracle_le_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, true]);
}

// ====================== GT TESTS ======================

#[test]
fn oracle_gt_greater_than() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 > 3.0");
}

#[test]
fn oracle_gt_less_than() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "3.0 > 5.0 is false");
}

#[test]
fn oracle_gt_equal() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 > 5.0 is false");
}

#[test]
fn oracle_gt_nan() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "5.0 > NaN is false");
}

#[test]
fn oracle_gt_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, false, false]);
}

// ====================== GE TESTS ======================

#[test]
fn oracle_ge_greater_than() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 >= 3.0");
}

#[test]
fn oracle_ge_equal() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "5.0 >= 5.0");
}

#[test]
fn oracle_ge_less_than() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "3.0 >= 5.0 is false");
}

#[test]
fn oracle_ge_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "NaN >= 5.0 is false");
}

#[test]
fn oracle_ge_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, true, false]);
}

// ====================== RELATIONSHIP TESTS ======================

#[test]
fn oracle_eq_ne_complement() {
    // Eq and Ne should be complements (except for NaN)
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (-1.0, 1.0), (0.0, -0.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let eq = eval_primitive(Primitive::Eq, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ne = eval_primitive(Primitive::Ne, &[a, b], &no_params()).unwrap();
        assert_ne!(
            extract_bool_scalar(&eq),
            extract_bool_scalar(&ne),
            "Eq and Ne are complements"
        );
    }
}

#[test]
fn oracle_lt_ge_complement() {
    // Lt and Ge should be complements (except for NaN)
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (5.0, 3.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let lt = eval_primitive(Primitive::Lt, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ge = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();
        assert_ne!(
            extract_bool_scalar(&lt),
            extract_bool_scalar(&ge),
            "Lt and Ge are complements"
        );
    }
}

#[test]
fn oracle_gt_le_complement() {
    // Gt and Le should be complements (except for NaN)
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (5.0, 3.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let gt = eval_primitive(Primitive::Gt, &[a.clone(), b.clone()], &no_params()).unwrap();
        let le = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();
        assert_ne!(
            extract_bool_scalar(&gt),
            extract_bool_scalar(&le),
            "Gt and Le are complements"
        );
    }
}

#[test]
fn oracle_lt_gt_swap() {
    // a < b iff b > a
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (5.0, 3.0), (-1.0, 1.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let lt_ab = eval_primitive(Primitive::Lt, &[a.clone(), b.clone()], &no_params()).unwrap();
        let gt_ba = eval_primitive(Primitive::Gt, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_bool_scalar(&lt_ab),
            extract_bool_scalar(&gt_ba),
            "a < b iff b > a"
        );
    }
}

#[test]
fn oracle_le_ge_swap() {
    // a <= b iff b >= a
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (5.0, 3.0), (-1.0, 1.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let le_ab = eval_primitive(Primitive::Le, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ge_ba = eval_primitive(Primitive::Ge, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_bool_scalar(&le_ab),
            extract_bool_scalar(&ge_ba),
            "a <= b iff b >= a"
        );
    }
}

#[test]
fn oracle_eq_symmetry() {
    // a == b iff b == a
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (0.0, -0.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let eq_ab = eval_primitive(Primitive::Eq, &[a.clone(), b.clone()], &no_params()).unwrap();
        let eq_ba = eval_primitive(Primitive::Eq, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_bool_scalar(&eq_ab),
            extract_bool_scalar(&eq_ba),
            "a == b iff b == a"
        );
    }
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_eq_2d() {
    let a = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    let result = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![true, false, true, false, true, false]
    );
}

#[test]
fn oracle_lt_2d() {
    let a = make_f64_tensor(&[2, 2], vec![1.0, 5.0, 3.0, 3.0]);
    let b = make_f64_tensor(&[2, 2], vec![2.0, 3.0, 4.0, 3.0]);
    let result = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, false]);
}

// ====================== TRICHOTOMY ======================

#[test]
fn oracle_trichotomy() {
    // For non-NaN values: exactly one of a < b, a == b, a > b is true
    for (a_val, b_val) in [(1.0, 2.0), (3.0, 3.0), (5.0, 3.0), (-1.0, 0.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);
        let lt = eval_primitive(Primitive::Lt, &[a.clone(), b.clone()], &no_params()).unwrap();
        let eq = eval_primitive(Primitive::Eq, &[a.clone(), b.clone()], &no_params()).unwrap();
        let gt = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();

        let lt_val = extract_bool_scalar(&lt);
        let eq_val = extract_bool_scalar(&eq);
        let gt_val = extract_bool_scalar(&gt);

        let count = [lt_val, eq_val, gt_val].iter().filter(|&&x| x).count();
        assert_eq!(
            count, 1,
            "exactly one of <, ==, > is true for {} vs {}",
            a_val, b_val
        );
    }
}

// ======================== METAMORPHIC: Lt(Neg(a), Neg(b)) = Gt(a, b) ========================

#[test]
fn metamorphic_lt_neg_equals_gt() {
    // Lt(Neg(a), Neg(b)) = Gt(a, b) - negating both flips the inequality
    for (a_val, b_val) in [(1.0, 2.0), (5.0, 3.0), (3.0, 3.0), (-1.0, 2.0), (0.5, 0.3)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        // Lt(Neg(a), Neg(b))
        let neg_a = eval_primitive(Primitive::Neg, &[a.clone()], &no_params()).unwrap();
        let neg_b = eval_primitive(Primitive::Neg, &[b.clone()], &no_params()).unwrap();
        let lt_neg = eval_primitive(Primitive::Lt, &[neg_a, neg_b], &no_params()).unwrap();

        // Gt(a, b)
        let gt = eval_primitive(Primitive::Gt, &[a, b], &no_params()).unwrap();

        assert_eq!(
            extract_bool_scalar(&lt_neg),
            extract_bool_scalar(&gt),
            "Lt(Neg({}), Neg({})) = Gt({}, {})",
            a_val, b_val, a_val, b_val
        );
    }
}

// ======================== METAMORPHIC: Gt(Neg(a), Neg(b)) = Lt(a, b) ========================

#[test]
fn metamorphic_gt_neg_equals_lt() {
    // Gt(Neg(a), Neg(b)) = Lt(a, b) - negating both flips the inequality
    for (a_val, b_val) in [(1.0, 2.0), (5.0, 3.0), (3.0, 3.0), (-1.0, 2.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        // Gt(Neg(a), Neg(b))
        let neg_a = eval_primitive(Primitive::Neg, &[a.clone()], &no_params()).unwrap();
        let neg_b = eval_primitive(Primitive::Neg, &[b.clone()], &no_params()).unwrap();
        let gt_neg = eval_primitive(Primitive::Gt, &[neg_a, neg_b], &no_params()).unwrap();

        // Lt(a, b)
        let lt = eval_primitive(Primitive::Lt, &[a, b], &no_params()).unwrap();

        assert_eq!(
            extract_bool_scalar(&gt_neg),
            extract_bool_scalar(&lt),
            "Gt(Neg({}), Neg({})) = Lt({}, {})",
            a_val, b_val, a_val, b_val
        );
    }
}

// ======================== METAMORPHIC: Le(Neg(a), Neg(b)) = Ge(a, b) ========================

#[test]
fn metamorphic_le_neg_equals_ge() {
    // Le(Neg(a), Neg(b)) = Ge(a, b) - negating both flips the inequality
    for (a_val, b_val) in [(1.0, 2.0), (5.0, 3.0), (3.0, 3.0), (-1.0, 2.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        // Le(Neg(a), Neg(b))
        let neg_a = eval_primitive(Primitive::Neg, &[a.clone()], &no_params()).unwrap();
        let neg_b = eval_primitive(Primitive::Neg, &[b.clone()], &no_params()).unwrap();
        let le_neg = eval_primitive(Primitive::Le, &[neg_a, neg_b], &no_params()).unwrap();

        // Ge(a, b)
        let ge = eval_primitive(Primitive::Ge, &[a, b], &no_params()).unwrap();

        assert_eq!(
            extract_bool_scalar(&le_neg),
            extract_bool_scalar(&ge),
            "Le(Neg({}), Neg({})) = Ge({}, {})",
            a_val, b_val, a_val, b_val
        );
    }
}

// ======================== METAMORPHIC: Ge(Neg(a), Neg(b)) = Le(a, b) ========================

#[test]
fn metamorphic_ge_neg_equals_le() {
    // Ge(Neg(a), Neg(b)) = Le(a, b) - negating both flips the inequality
    for (a_val, b_val) in [(1.0, 2.0), (5.0, 3.0), (3.0, 3.0), (-1.0, 2.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        // Ge(Neg(a), Neg(b))
        let neg_a = eval_primitive(Primitive::Neg, &[a.clone()], &no_params()).unwrap();
        let neg_b = eval_primitive(Primitive::Neg, &[b.clone()], &no_params()).unwrap();
        let ge_neg = eval_primitive(Primitive::Ge, &[neg_a, neg_b], &no_params()).unwrap();

        // Le(a, b)
        let le = eval_primitive(Primitive::Le, &[a, b], &no_params()).unwrap();

        assert_eq!(
            extract_bool_scalar(&ge_neg),
            extract_bool_scalar(&le),
            "Ge(Neg({}), Neg({})) = Le({}, {})",
            a_val, b_val, a_val, b_val
        );
    }
}
