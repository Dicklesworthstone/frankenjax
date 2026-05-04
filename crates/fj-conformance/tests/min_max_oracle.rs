//! Oracle tests for Min and Max primitives.
//!
//! min(a, b) = smaller of a and b
//! max(a, b) = larger of a and b
//!
//! NaN semantics: JAX maximum/minimum propagate NaN when either operand is NaN.
//!
//! Tests:
//! - Basic comparison
//! - Equal values
//! - Negative values
//! - Zero handling
//! - Infinity
//! - NaN handling
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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
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

// ====================== MIN TESTS ======================

// ======================== Min: Basic Comparison ========================

#[test]
fn oracle_min_basic() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0, "min(3, 5) = 3");
}

#[test]
fn oracle_min_reversed() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0, "min(5, 3) = 3");
}

#[test]
fn oracle_min_equal() {
    let a = make_f64_tensor(&[], vec![4.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "min(4, 4) = 4");
}

// ======================== Min: Negative Values ========================

#[test]
fn oracle_min_negative() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "min(-3, -5) = -5");
}

#[test]
fn oracle_min_mixed_sign() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -3.0, "min(-3, 5) = -3");
}

// ======================== Min: Zero ========================

#[test]
fn oracle_min_with_zero() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "min(0, 5) = 0");
}

#[test]
fn oracle_min_with_neg_zero() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "min(0, -5) = -5");
}

// ======================== Min: Infinity ========================

#[test]
fn oracle_min_pos_inf() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "min(inf, 5) = 5");
}

#[test]
fn oracle_min_neg_inf() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "min(-inf, 5) = -inf");
}

// ======================== Min: NaN ========================

#[test]
fn oracle_min_nan_first() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "min(NaN, 5) = NaN");
}

#[test]
fn oracle_min_nan_second() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "min(5, NaN) = NaN");
}

#[test]
fn oracle_min_nan_both() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "min(NaN, NaN) = NaN");
}

// ======================== Min: Integer ========================

#[test]
fn oracle_min_i64() {
    let a = make_i64_tensor(&[], vec![3]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 3, "min(3, 5) = 3");
}

#[test]
fn oracle_min_i64_negative() {
    let a = make_i64_tensor(&[], vec![-10]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -10, "min(-10, 5) = -10");
}

// ======================== Min: 1D Tensor ========================

#[test]
fn oracle_min_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 7.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 6.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, 3.0, 6.0]);
}

#[test]
fn oracle_min_1d_nan_propagates() {
    let a = make_f64_tensor(&[4], vec![1.0, f64::NAN, 3.0, 7.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, f64::NAN, 6.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let values = extract_f64_vec(&result);
    assert_eq!(values[0], 1.0);
    assert!(values[1].is_nan());
    assert!(values[2].is_nan());
    assert_eq!(values[3], 6.0);
}

// ====================== MAX TESTS ======================

// ======================== Max: Basic Comparison ========================

#[test]
fn oracle_max_basic() {
    let a = make_f64_tensor(&[], vec![3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "max(3, 5) = 5");
}

#[test]
fn oracle_max_reversed() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "max(5, 3) = 5");
}

#[test]
fn oracle_max_equal() {
    let a = make_f64_tensor(&[], vec![4.0]);
    let b = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "max(4, 4) = 4");
}

// ======================== Max: Negative Values ========================

#[test]
fn oracle_max_negative() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -3.0, "max(-3, -5) = -3");
}

#[test]
fn oracle_max_mixed_sign() {
    let a = make_f64_tensor(&[], vec![-3.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "max(-3, 5) = 5");
}

// ======================== Max: Zero ========================

#[test]
fn oracle_max_with_zero() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "max(0, 5) = 5");
}

#[test]
fn oracle_max_with_neg_zero() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "max(0, -5) = 0");
}

// ======================== Max: Infinity ========================

#[test]
fn oracle_max_pos_inf() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "max(inf, 5) = inf");
}

#[test]
fn oracle_max_neg_inf() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "max(-inf, 5) = 5");
}

// ======================== Max: NaN ========================

#[test]
fn oracle_max_nan_first() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "max(NaN, 5) = NaN");
}

#[test]
fn oracle_max_nan_second() {
    let a = make_f64_tensor(&[], vec![5.0]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "max(5, NaN) = NaN");
}

#[test]
fn oracle_max_nan_both() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "max(NaN, NaN) = NaN");
}

// ======================== Max: Integer ========================

#[test]
fn oracle_max_i64() {
    let a = make_i64_tensor(&[], vec![3]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 5, "max(3, 5) = 5");
}

#[test]
fn oracle_max_i64_negative() {
    let a = make_i64_tensor(&[], vec![-10]);
    let b = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 5, "max(-10, 5) = 5");
}

// ======================== Max: 1D Tensor ========================

#[test]
fn oracle_max_1d() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, 3.0, 7.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 6.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 5.0, 4.0, 7.0]);
}

#[test]
fn oracle_max_1d_nan_propagates() {
    let a = make_f64_tensor(&[4], vec![1.0, f64::NAN, 3.0, 7.0]);
    let b = make_f64_tensor(&[4], vec![2.0, 3.0, f64::NAN, 6.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let values = extract_f64_vec(&result);
    assert_eq!(values[0], 2.0);
    assert!(values[1].is_nan());
    assert!(values[2].is_nan());
    assert_eq!(values[3], 7.0);
}

// ====================== MIN/MAX RELATIONSHIP ======================

#[test]
fn oracle_min_max_relationship() {
    // max(a, b) + min(a, b) = a + b
    for (a_val, b_val) in [(3.0, 5.0), (-2.0, 4.0), (0.0, 1.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        let min_result =
            eval_primitive(Primitive::Min, &[a.clone(), b.clone()], &no_params()).unwrap();
        let max_result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

        let min_val = extract_f64_scalar(&min_result);
        let max_val = extract_f64_scalar(&max_result);

        assert_eq!(
            min_val + max_val,
            a_val + b_val,
            "min({}, {}) + max({}, {}) = {} + {}",
            a_val,
            b_val,
            a_val,
            b_val,
            a_val,
            b_val
        );
    }
}

#[test]
fn oracle_min_max_ordering() {
    // min(a, b) <= max(a, b) always
    for (a_val, b_val) in [(3.0, 5.0), (-2.0, 4.0), (0.0, 0.0), (-5.0, -3.0)] {
        let a = make_f64_tensor(&[], vec![a_val]);
        let b = make_f64_tensor(&[], vec![b_val]);

        let min_result =
            eval_primitive(Primitive::Min, &[a.clone(), b.clone()], &no_params()).unwrap();
        let max_result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

        let min_val = extract_f64_scalar(&min_result);
        let max_val = extract_f64_scalar(&max_result);

        assert!(
            min_val <= max_val,
            "min({}, {}) <= max({}, {})",
            a_val,
            b_val,
            a_val,
            b_val
        );
    }
}
