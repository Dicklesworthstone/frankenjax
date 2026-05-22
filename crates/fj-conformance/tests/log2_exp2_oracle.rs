//! Oracle tests for Log2 and Exp2 primitives.
//!
//! log2(x) = log base 2 of x
//! exp2(x) = 2^x
//!
//! Tests:
//! - Identity: log2(2^n) = n
//! - Inverse: exp2(log2(x)) = x
//! - Special values: log2(1) = 0, exp2(0) = 1
//! - Edge cases: infinity, NaN, negative inputs
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

// ======================== Log2: Basic Values ========================

#[test]
fn oracle_log2_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "log2(1) = 0");
}

#[test]
fn oracle_log2_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "log2(2) = 1");
}

#[test]
fn oracle_log2_four() {
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "log2(4) = 2");
}

#[test]
fn oracle_log2_power_of_two() {
    for n in 0..10 {
        let val = (1_u64 << n) as f64;
        let input = make_f64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), n as f64, "log2(2^{}) = {}", n, n);
    }
}

#[test]
fn oracle_log2_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "log2(0.5) = -1");
}

// ======================== Log2: Special Values ========================

#[test]
fn oracle_log2_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "log2(0) = -inf"
    );
}

#[test]
fn oracle_log2_negative() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log2(-1) = NaN");
}

#[test]
fn oracle_log2_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "log2(inf) = inf"
    );
}

#[test]
fn oracle_log2_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log2(NaN) = NaN");
}

// ======================== Exp2: Basic Values ========================

#[test]
fn oracle_exp2_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "exp2(0) = 1");
}

#[test]
fn oracle_exp2_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "exp2(1) = 2");
}

#[test]
fn oracle_exp2_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "exp2(2) = 4");
}

#[test]
fn oracle_exp2_ten() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1024.0, "exp2(10) = 1024");
}

#[test]
fn oracle_exp2_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5, "exp2(-1) = 0.5");
}

// ======================== Exp2: Special Values ========================

#[test]
fn oracle_exp2_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "exp2(inf) = inf"
    );
}

#[test]
fn oracle_exp2_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "exp2(-inf) = 0");
}

#[test]
fn oracle_exp2_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "exp2(NaN) = NaN");
}

// ======================== Inverse Relationship ========================

#[test]
fn oracle_exp2_log2_inverse() {
    for x in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let log2_result = eval_primitive(Primitive::Log2, &[input.clone()], &no_params()).unwrap();
        let roundtrip =
            eval_primitive(Primitive::Exp2, &[log2_result], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - x).abs() / x < 1e-10,
            "exp2(log2({})) = {} (expected {})",
            x,
            actual,
            x
        );
    }
}

#[test]
fn oracle_log2_exp2_inverse() {
    for x in [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let exp2_result = eval_primitive(Primitive::Exp2, &[input.clone()], &no_params()).unwrap();
        let roundtrip =
            eval_primitive(Primitive::Log2, &[exp2_result], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - x).abs() < 1e-10,
            "log2(exp2({})) = {} (expected {})",
            x,
            actual,
            x
        );
    }
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_log2_vector() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 4.0, 8.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn oracle_exp2_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn oracle_log2_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 4.0, 16.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 4.0]);
}

#[test]
fn oracle_exp2_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 4.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 4.0, 16.0]);
}
