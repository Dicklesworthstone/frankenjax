//! Oracle tests for Ceil primitive.
//!
//! ceil(x) = smallest integer >= x
//!
//! Tests:
//! - Integers: ceil(n) = n
//! - Positive fractional: ceil(1.1) = 2
//! - Negative fractional: ceil(-1.1) = -1
//! - Zero: ceil(0) = 0, ceil(-0) = -0
//! - Infinity: ceil(±inf) = ±inf
//! - NaN propagation
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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Integers ========================

#[test]
fn oracle_ceil_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(0) = 0");
}

#[test]
fn oracle_ceil_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0, "ceil(-0.0) = 0");
}

#[test]
fn oracle_ceil_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(1) = 1");
}

#[test]
fn oracle_ceil_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1) = -1");
}

#[test]
fn oracle_ceil_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "ceil(1000) = 1000");
}

// ======================== Positive Fractional ========================

#[test]
fn oracle_ceil_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.1) = 2");
}

#[test]
fn oracle_ceil_one_point_five() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.5) = 2");
}

#[test]
fn oracle_ceil_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.9) = 2");
}

#[test]
fn oracle_ceil_point_one() {
    let input = make_f64_tensor(&[], vec![0.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.1) = 1");
}

#[test]
fn oracle_ceil_point_five() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.5) = 1");
}

#[test]
fn oracle_ceil_point_nine() {
    let input = make_f64_tensor(&[], vec![0.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.9) = 1");
}

// ======================== Negative Fractional ========================

#[test]
fn oracle_ceil_neg_point_one() {
    // ceil(-0.1) = 0 (round toward positive infinity)
    let input = make_f64_tensor(&[], vec![-0.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.1) = 0");
}

#[test]
fn oracle_ceil_neg_point_five() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.5) = 0");
}

#[test]
fn oracle_ceil_neg_point_nine() {
    let input = make_f64_tensor(&[], vec![-0.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.9) = 0");
}

#[test]
fn oracle_ceil_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.1) = -1");
}

#[test]
fn oracle_ceil_neg_one_point_five() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.5) = -1");
}

#[test]
fn oracle_ceil_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.9) = -1");
}

// ======================== Infinity ========================

#[test]
fn oracle_ceil_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "ceil(+inf) = +inf");
}

#[test]
fn oracle_ceil_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "ceil(-inf) = -inf");
}

// ======================== NaN ========================

#[test]
fn oracle_ceil_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "ceil(NaN) = NaN");
}

// ======================== Very Small Values ========================

#[test]
fn oracle_ceil_very_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(1e-100) = 1");
}

#[test]
fn oracle_ceil_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-1e-100) = 0");
}

// ======================== ceil vs floor relationship ========================

#[test]
fn oracle_ceil_floor_relationship() {
    // ceil(x) = -floor(-x) for all x
    for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let ceil_result = eval_primitive(Primitive::Ceil, &[input_pos], &no_params()).unwrap();
        let floor_result = eval_primitive(Primitive::Floor, &[input_neg], &no_params()).unwrap();

        let ceil_val = extract_f64_scalar(&ceil_result);
        let floor_neg_val = -extract_f64_scalar(&floor_result);

        assert_eq!(ceil_val, floor_neg_val, "ceil({}) = -floor(-{})", x, x);
    }
}

// ======================== Bounds: x <= ceil(x) < x + 1 ========================

#[test]
fn oracle_ceil_bounds() {
    for x in [-2.7, -1.3, -0.1, 0.1, 1.3, 2.7] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);

        assert!(val >= x, "ceil({}) >= {}", x, x);
        assert!(val < x + 1.0, "ceil({}) < {} + 1", x, x);
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_ceil_1d() {
    let input = make_f64_tensor(&[5], vec![-1.5, -0.5, 0.0, 0.5, 1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], -1.0, "ceil(-1.5)");
    assert_eq!(vals[1], 0.0, "ceil(-0.5)");
    assert_eq!(vals[2], 0.0, "ceil(0)");
    assert_eq!(vals[3], 1.0, "ceil(0.5)");
    assert_eq!(vals[4], 2.0, "ceil(1.5)");
}

#[test]
fn oracle_ceil_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "ceil(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "ceil(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "ceil(-inf)");
    assert!(vals[3].is_nan(), "ceil(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_ceil_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 1.5, 1.9, -1.1, -1.5, -1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 2.0, "ceil(1.1)");
    assert_eq!(vals[1], 2.0, "ceil(1.5)");
    assert_eq!(vals[2], 2.0, "ceil(1.9)");
    assert_eq!(vals[3], -1.0, "ceil(-1.1)");
    assert_eq!(vals[4], -1.0, "ceil(-1.5)");
    assert_eq!(vals[5], -1.0, "ceil(-1.9)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_ceil_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.1, 0.9, 1.5, 2.5, -0.1, -0.9, -1.5, -2.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 1.0, "ceil(0.1)");
    assert_eq!(vals[1], 1.0, "ceil(0.9)");
    assert_eq!(vals[2], 2.0, "ceil(1.5)");
    assert_eq!(vals[3], 3.0, "ceil(2.5)");
    assert_eq!(vals[4], 0.0, "ceil(-0.1)");
    assert_eq!(vals[5], 0.0, "ceil(-0.9)");
    assert_eq!(vals[6], -1.0, "ceil(-1.5)");
    assert_eq!(vals[7], -2.0, "ceil(-2.5)");
}

// ======================== Idempotency: ceil(ceil(x)) = ceil(x) ========================

#[test]
fn oracle_ceil_idempotent() {
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Ceil, &[input1], &no_params()).unwrap();
        let ceiled = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![ceiled]);
        let result2 = eval_primitive(Primitive::Ceil, &[input2], &no_params()).unwrap();
        let double_ceiled = extract_f64_scalar(&result2);

        assert_eq!(ceiled, double_ceiled, "ceil(ceil({})) = ceil({})", x, x);
    }
}
