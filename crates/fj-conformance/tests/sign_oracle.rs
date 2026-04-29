//! Oracle tests for Sign primitive.
//!
//! sign(x) returns:
//! - -1 if x < 0
//! - 0 if x == 0
//! - +1 if x > 0
//! - NaN if x is NaN
//!
//! For integers, returns -1, 0, or 1.
//! For floats, returns -1.0, 0.0, or 1.0 (preserving -0.0 sign).

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
        _ => panic!("expected tensor"),
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
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
        _ => panic!("expected scalar"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
        _ => panic!("expected scalar"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Integer Positive ========================

#[test]
fn oracle_sign_i64_positive_one() {
    let input = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_sign_i64_positive_large() {
    let input = make_i64_tensor(&[], vec![1_000_000]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_sign_i64_max() {
    let input = make_i64_tensor(&[], vec![i64::MAX]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

// ======================== Integer Negative ========================

#[test]
fn oracle_sign_i64_negative_one() {
    let input = make_i64_tensor(&[], vec![-1]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

#[test]
fn oracle_sign_i64_negative_large() {
    let input = make_i64_tensor(&[], vec![-1_000_000]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

#[test]
fn oracle_sign_i64_min() {
    let input = make_i64_tensor(&[], vec![i64::MIN]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

// ======================== Integer Zero ========================

#[test]
fn oracle_sign_i64_zero() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Float Positive ========================

#[test]
fn oracle_sign_f64_positive_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_fraction() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_small() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_large() {
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

// ======================== Float Negative ========================

#[test]
fn oracle_sign_f64_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_fraction() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_small() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_large() {
    let input = make_f64_tensor(&[], vec![-1e100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

// ======================== Float Zero ========================

#[test]
fn oracle_sign_f64_positive_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_sign_f64_negative_zero() {
    // sign(-0.0) should preserve the sign: -0.0
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_negative(), "sign(-0.0) should be -0.0");
}

// ======================== Float NaN ========================

#[test]
fn oracle_sign_f64_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sign(NaN) should be NaN");
}

// ======================== 1D Tensor Integer ========================

#[test]
fn oracle_sign_i64_1d() {
    let input = make_i64_tensor(&[5], vec![-5, -1, 0, 1, 100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, 0, 1, 1]);
}

#[test]
fn oracle_sign_i64_all_positive() {
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1]);
}

#[test]
fn oracle_sign_i64_all_negative() {
    let input = make_i64_tensor(&[4], vec![-1, -2, -3, -4]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, -1]);
}

#[test]
fn oracle_sign_i64_all_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

// ======================== 1D Tensor Float ========================

#[test]
fn oracle_sign_f64_1d() {
    let input = make_f64_tensor(&[5], vec![-3.5, -0.1, 0.0, 0.1, 7.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
}

#[test]
fn oracle_sign_f64_1d_mixed_special() {
    let input = make_f64_tensor(
        &[6],
        vec![-f64::INFINITY, -1.0, 0.0, 1.0, f64::INFINITY, f64::NAN],
    );
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
    assert!(vals[5].is_nan());
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_sign_i64_2d() {
    let input = make_i64_tensor(&[2, 3], vec![-3, -2, -1, 0, 1, 2]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, 0, 1, 1]);
}

#[test]
fn oracle_sign_f64_2d() {
    let input = make_f64_tensor(&[2, 2], vec![-1.5, 0.0, 0.0, 2.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_sign_i64_3d() {
    let input = make_i64_tensor(&[2, 2, 2], vec![-4, -3, -2, -1, 0, 1, 2, 3]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, -1, 0, 1, 1, 1]);
}

// ======================== Identity: sign(x) * |x| = x ========================

#[test]
fn oracle_sign_identity_positive() {
    // For positive x: sign(x) = 1, so sign(x) * |x| = x
    for x in [1.0, 2.5, 100.0, 1e50] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let sign_val = extract_f64_scalar(&result);
        assert_eq!(sign_val * x.abs(), x, "sign({}) * |{}| = {}", x, x, x);
    }
}

#[test]
fn oracle_sign_identity_negative() {
    // For negative x: sign(x) = -1, so sign(x) * |x| = x
    for x in [-1.0, -2.5, -100.0, -1e50] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let sign_val = extract_f64_scalar(&result);
        assert_eq!(sign_val * x.abs(), x, "sign({}) * |{}| = {}", x, x, x);
    }
}

// ======================== Idempotence: sign(sign(x)) = sign(x) ========================

#[test]
fn oracle_sign_idempotent() {
    // sign(sign(x)) = sign(x) for non-zero x
    for x in [-5.0, -1.0, 1.0, 5.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Sign, &[input1], &no_params()).unwrap();
        let sign1 = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![sign1]);
        let result2 = eval_primitive(Primitive::Sign, &[input2], &no_params()).unwrap();
        let sign2 = extract_f64_scalar(&result2);

        assert_eq!(sign1, sign2, "sign(sign({})) should equal sign({})", x, x);
    }
}
