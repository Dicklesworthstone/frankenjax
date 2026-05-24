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
//! - Broadcast-compatible operands

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

// ======================== Min: Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_min_scalar_lhs_tensor_rhs_broadcast() {
    // scalar min tensor
    let a = scalar_f64(3.0);
    let b = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, -3.0, 3.0]);
}

#[test]
fn oracle_min_tensor_lhs_scalar_rhs_broadcast() {
    // tensor min scalar
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let b = scalar_f64(3.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, -3.0, 3.0]);
}

#[test]
fn oracle_min_singleton_lhs_vector_rhs_broadcast() {
    // [1] min [3] -> [3]
    let a = make_f64_tensor(&[1], vec![3.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 5.0, -2.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, -2.0]);
}

#[test]
fn oracle_min_vector_lhs_singleton_rhs_broadcast() {
    // [3] min [1] -> [3]
    let a = make_f64_tensor(&[3], vec![1.0, 5.0, -2.0]);
    let b = make_f64_tensor(&[1], vec![3.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, -2.0]);
}

#[test]
fn oracle_min_column_lhs_matrix_rhs_broadcast() {
    // [2, 1] min [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![2.0, 5.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 3.0, 4.0, 6.0, 4.0, 3.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: min(2, 1)=1, min(2, 3)=2, min(2, 4)=2
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 2.0);
    assert_eq!(vals[2], 2.0);
    // Row 1: min(5, 6)=5, min(5, 4)=4, min(5, 3)=3
    assert_eq!(vals[3], 5.0);
    assert_eq!(vals[4], 4.0);
    assert_eq!(vals[5], 3.0);
}

#[test]
fn oracle_min_different_ranks_broadcast() {
    // [3] min [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[3], vec![2.0, 4.0, 6.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 7.0, 3.0, 3.0, 5.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: min(2,1)=1, min(4,5)=4, min(6,7)=6
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 4.0);
    assert_eq!(vals[2], 6.0);
    // Row 1: min(2,3)=2, min(4,3)=3, min(6,5)=5
    assert_eq!(vals[3], 2.0);
    assert_eq!(vals[4], 3.0);
    assert_eq!(vals[5], 5.0);
}

#[test]
fn oracle_min_all_scalars_broadcast() {
    // scalar min scalar -> scalar
    let a = scalar_f64(3.0);
    let b = scalar_f64(5.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0);
}

#[test]
fn oracle_min_incompatible_shapes_error() {
    // [2] min [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_min_vector_scalar_rhs_broadcast() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 3.0, -3.0, 3.0]);
}

#[test]
fn oracle_min_matrix_row_rhs_broadcast() {
    let a = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 2.0, -3.0, 4.0, 7.0]);
    let b = make_f64_tensor(&[3], vec![-2.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![-2.0, 3.0, 2.0, -3.0, 3.0, 6.0]
    );
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

// ======================== Max: Broadcasting ========================

#[test]
fn oracle_max_scalar_lhs_tensor_rhs_broadcast() {
    // scalar max tensor
    let a = scalar_f64(3.0);
    let b = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 3.0, 7.0]);
}

#[test]
fn oracle_max_tensor_lhs_scalar_rhs_broadcast() {
    // tensor max scalar
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let b = scalar_f64(3.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 3.0, 7.0]);
}

#[test]
fn oracle_max_singleton_lhs_vector_rhs_broadcast() {
    // [1] max [3] -> [3]
    let a = make_f64_tensor(&[1], vec![3.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 5.0, -2.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 3.0]);
}

#[test]
fn oracle_max_vector_lhs_singleton_rhs_broadcast() {
    // [3] max [1] -> [3]
    let a = make_f64_tensor(&[3], vec![1.0, 5.0, -2.0]);
    let b = make_f64_tensor(&[1], vec![3.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 3.0]);
}

#[test]
fn oracle_max_column_lhs_matrix_rhs_broadcast() {
    // [2, 1] max [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![2.0, 5.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 3.0, 4.0, 6.0, 4.0, 3.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: max(2, 1)=2, max(2, 3)=3, max(2, 4)=4
    assert_eq!(vals[0], 2.0);
    assert_eq!(vals[1], 3.0);
    assert_eq!(vals[2], 4.0);
    // Row 1: max(5, 6)=6, max(5, 4)=5, max(5, 3)=5
    assert_eq!(vals[3], 6.0);
    assert_eq!(vals[4], 5.0);
    assert_eq!(vals[5], 5.0);
}

#[test]
fn oracle_max_different_ranks_broadcast() {
    // [3] max [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[3], vec![2.0, 4.0, 6.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 7.0, 3.0, 3.0, 5.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: max(2,1)=2, max(4,5)=5, max(6,7)=7
    assert_eq!(vals[0], 2.0);
    assert_eq!(vals[1], 5.0);
    assert_eq!(vals[2], 7.0);
    // Row 1: max(2,3)=3, max(4,3)=4, max(6,5)=6
    assert_eq!(vals[3], 3.0);
    assert_eq!(vals[4], 4.0);
    assert_eq!(vals[5], 6.0);
}

#[test]
fn oracle_max_all_scalars_broadcast() {
    // scalar max scalar -> scalar
    let a = scalar_f64(3.0);
    let b = scalar_f64(5.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_max_incompatible_shapes_error() {
    // [2] max [3] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_max_vector_scalar_rhs_broadcast() {
    let a = make_f64_tensor(&[4], vec![1.0, 5.0, -3.0, 7.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 3.0, 7.0]);
}

#[test]
fn oracle_max_matrix_row_rhs_broadcast() {
    let a = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 2.0, -3.0, 4.0, 7.0]);
    let b = make_f64_tensor(&[3], vec![-2.0, 3.0, 6.0]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 5.0, 6.0, -2.0, 4.0, 7.0]
    );
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

// ======================== METAMORPHIC: commutativity ========================

#[test]
fn metamorphic_min_commutative() {
    // min(x, y) = min(y, x)
    for (x, y) in [(3.0, 5.0), (-2.0, 4.0), (0.0, -1.0), (7.0, 7.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);

        let min_ab = eval_primitive(Primitive::Min, &[a.clone(), b.clone()], &no_params()).unwrap();
        let min_ba = eval_primitive(Primitive::Min, &[b, a], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&min_ab),
            extract_f64_scalar(&min_ba),
            "min({}, {}) = min({}, {})",
            x,
            y,
            y,
            x
        );
    }
}

#[test]
fn metamorphic_max_commutative() {
    // max(x, y) = max(y, x)
    for (x, y) in [(3.0, 5.0), (-2.0, 4.0), (0.0, -1.0), (7.0, 7.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);

        let max_ab = eval_primitive(Primitive::Max, &[a.clone(), b.clone()], &no_params()).unwrap();
        let max_ba = eval_primitive(Primitive::Max, &[b, a], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&max_ab),
            extract_f64_scalar(&max_ba),
            "max({}, {}) = max({}, {})",
            x,
            y,
            y,
            x
        );
    }
}

// ======================== METAMORPHIC: idempotence ========================

#[test]
fn metamorphic_min_idempotent() {
    // min(x, x) = x
    for x in [-5.0, 0.0, 3.0, f64::INFINITY] {
        let a = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Min, &[a.clone(), a], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "min({}, {}) = {}", x, x, x);
    }
}

#[test]
fn metamorphic_max_idempotent() {
    // max(x, x) = x
    for x in [-5.0, 0.0, 3.0, f64::INFINITY] {
        let a = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Max, &[a.clone(), a], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), x, "max({}, {}) = {}", x, x, x);
    }
}

// ======================== METAMORPHIC: associativity ========================

#[test]
fn metamorphic_min_associative() {
    // min(min(x, y), z) = min(x, min(y, z))
    for (x, y, z) in [(1.0, 2.0, 3.0), (5.0, 3.0, 4.0), (-1.0, 0.0, 1.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let c = make_f64_tensor(&[], vec![z]);

        let min_ab = eval_primitive(Primitive::Min, &[a.clone(), b.clone()], &no_params()).unwrap();
        let left = eval_primitive(Primitive::Min, &[min_ab, c.clone()], &no_params()).unwrap();

        let min_bc = eval_primitive(Primitive::Min, &[b, c], &no_params()).unwrap();
        let right = eval_primitive(Primitive::Min, &[a, min_bc], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&left),
            extract_f64_scalar(&right),
            "min(min({}, {}), {}) = min({}, min({}, {}))",
            x,
            y,
            z,
            x,
            y,
            z
        );
    }
}

#[test]
fn metamorphic_max_associative() {
    // max(max(x, y), z) = max(x, max(y, z))
    for (x, y, z) in [(1.0, 2.0, 3.0), (5.0, 3.0, 4.0), (-1.0, 0.0, 1.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);
        let c = make_f64_tensor(&[], vec![z]);

        let max_ab = eval_primitive(Primitive::Max, &[a.clone(), b.clone()], &no_params()).unwrap();
        let left = eval_primitive(Primitive::Max, &[max_ab, c.clone()], &no_params()).unwrap();

        let max_bc = eval_primitive(Primitive::Max, &[b, c], &no_params()).unwrap();
        let right = eval_primitive(Primitive::Max, &[a, max_bc], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&left),
            extract_f64_scalar(&right),
            "max(max({}, {}), {}) = max({}, max({}, {}))",
            x,
            y,
            z,
            x,
            y,
            z
        );
    }
}

// ======================== METAMORPHIC: absorption ========================

#[test]
fn metamorphic_max_absorbs_min() {
    // max(x, min(x, y)) = x (absorption law)
    for (x, y) in [(5.0, 3.0), (3.0, 5.0), (0.0, -1.0), (-2.0, 4.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);

        let min_xy = eval_primitive(Primitive::Min, &[a.clone(), b], &no_params()).unwrap();
        let result = eval_primitive(Primitive::Max, &[a, min_xy], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&result),
            x,
            "max({}, min({}, {})) = {}",
            x,
            x,
            y,
            x
        );
    }
}

#[test]
fn metamorphic_min_absorbs_max() {
    // min(x, max(x, y)) = x (absorption law)
    for (x, y) in [(5.0, 3.0), (3.0, 5.0), (0.0, -1.0), (-2.0, 4.0)] {
        let a = make_f64_tensor(&[], vec![x]);
        let b = make_f64_tensor(&[], vec![y]);

        let max_xy = eval_primitive(Primitive::Max, &[a.clone(), b], &no_params()).unwrap();
        let result = eval_primitive(Primitive::Min, &[a, max_xy], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&result),
            x,
            "min({}, max({}, {})) = {}",
            x,
            x,
            y,
            x
        );
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_min_preserves_all_float_dtypes() {
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
        let b = make_vec(dtype, &[0.0_f64, 0.5, 0.5]);
        let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "min {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_max_preserves_all_float_dtypes() {
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
        let b = make_vec(dtype, &[0.0_f64, 0.5, 0.5]);
        let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "max {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================
// Complex min/max uses lexicographic ordering: compare real parts first, then imaginary

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

fn assert_complex64_eq(actual: (f32, f32), expected: (f32, f32), msg: &str) {
    assert_eq!(
        actual, expected,
        "{}: expected ({}, {}), got ({}, {})",
        msg, expected.0, expected.1, actual.0, actual.1
    );
}

fn assert_complex128_eq(actual: (f64, f64), expected: (f64, f64), msg: &str) {
    assert_eq!(
        actual, expected,
        "{}: expected ({}, {}), got ({}, {})",
        msg, expected.0, expected.1, actual.0, actual.1
    );
}

#[test]
fn oracle_min_complex64_different_real() {
    // 1+2i vs 3+0i: min is 1+2i (smaller real part)
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (1.0, 2.0), "min(1+2i, 3) = 1+2i");
}

#[test]
fn oracle_max_complex64_different_real() {
    // 1+2i vs 3+0i: max is 3+0i (larger real part)
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (3.0, 0.0), "max(1+2i, 3) = 3");
}

#[test]
fn oracle_min_complex64_same_real() {
    // 2+1i vs 2+3i: same real, compare imaginary -> min is 2+1i
    let a = make_complex64_scalar(2.0, 1.0);
    let b = make_complex64_scalar(2.0, 3.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (2.0, 1.0), "min(2+1i, 2+3i) = 2+1i");
}

#[test]
fn oracle_max_complex64_same_real() {
    // 2+1i vs 2+3i: same real, compare imaginary -> max is 2+3i
    let a = make_complex64_scalar(2.0, 1.0);
    let b = make_complex64_scalar(2.0, 3.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (2.0, 3.0), "max(2+1i, 2+3i) = 2+3i");
}

#[test]
fn oracle_min_complex64_identical() {
    // min(z, z) = z
    let a = make_complex64_scalar(3.0, 4.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (3.0, 4.0), "min(z, z) = z");
}

#[test]
fn oracle_max_complex64_identical() {
    // max(z, z) = z
    let a = make_complex64_scalar(3.0, 4.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_eq((re, im), (3.0, 4.0), "max(z, z) = z");
}

#[test]
fn oracle_min_complex64_vector() {
    let a = make_complex64_tensor(&[3], vec![(1.0, 5.0), (3.0, 1.0), (2.0, 2.0)]);
    let b = make_complex64_tensor(&[3], vec![(2.0, 0.0), (3.0, 0.0), (2.0, 3.0)]);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    // 1+5i vs 2+0i: 1+5i has smaller real
    assert_complex64_eq(vals[0], (1.0, 5.0), "min[0]");
    // 3+1i vs 3+0i: same real, 3+0i has smaller imag
    assert_complex64_eq(vals[1], (3.0, 0.0), "min[1]");
    // 2+2i vs 2+3i: same real, 2+2i has smaller imag
    assert_complex64_eq(vals[2], (2.0, 2.0), "min[2]");
}

#[test]
fn oracle_max_complex64_vector() {
    let a = make_complex64_tensor(&[3], vec![(1.0, 5.0), (3.0, 1.0), (2.0, 2.0)]);
    let b = make_complex64_tensor(&[3], vec![(2.0, 0.0), (3.0, 0.0), (2.0, 3.0)]);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    // 1+5i vs 2+0i: 2+0i has larger real
    assert_complex64_eq(vals[0], (2.0, 0.0), "max[0]");
    // 3+1i vs 3+0i: same real, 3+1i has larger imag
    assert_complex64_eq(vals[1], (3.0, 1.0), "max[1]");
    // 2+2i vs 2+3i: same real, 2+3i has larger imag
    assert_complex64_eq(vals[2], (2.0, 3.0), "max[2]");
}

#[test]
fn oracle_min_complex128_different_real() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_eq((re, im), (1.0, 2.0), "min Complex128");
}

#[test]
fn oracle_max_complex128_different_real() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_eq((re, im), (3.0, 0.0), "max Complex128");
}

#[test]
fn oracle_min_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_max_complex64_preserves_dtype() {
    let a = make_complex64_scalar(1.0, 2.0);
    let b = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_min_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Min, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_max_complex128_preserves_dtype() {
    let a = make_complex128_scalar(1.0, 2.0);
    let b = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Max, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
