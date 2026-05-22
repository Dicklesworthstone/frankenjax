//! Oracle tests for Gcd and Lcm primitives.
//!
//! gcd(a, b) = greatest common divisor
//! lcm(a, b) = least common multiple
//!
//! Tests:
//! - Basic: gcd(12, 8) = 4, lcm(4, 6) = 12
//! - Identity: gcd(n, n) = n, lcm(n, n) = n
//! - One: gcd(n, 1) = 1, lcm(n, 1) = n
//! - Zero: gcd(n, 0) = n, lcm(n, 0) = 0
//! - Negative values
//! - Tensor shapes

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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
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

// ======================== GCD Basic Cases ========================

#[test]
fn oracle_gcd_basic() {
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "gcd(12, 8) = 4");
}

#[test]
fn oracle_gcd_coprime() {
    let a = make_i64_tensor(&[], vec![17]);
    let b = make_i64_tensor(&[], vec![13]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "gcd(17, 13) = 1 (coprime)");
}

#[test]
fn oracle_gcd_same() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![15]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15, "gcd(15, 15) = 15");
}

#[test]
fn oracle_gcd_one() {
    let a = make_i64_tensor(&[], vec![42]);
    let b = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "gcd(42, 1) = 1");
}

#[test]
fn oracle_gcd_zero() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15, "gcd(15, 0) = 15");
}

#[test]
fn oracle_gcd_both_zero() {
    let a = make_i64_tensor(&[], vec![0]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "gcd(0, 0) = 0");
}

// ======================== GCD Negative Values ========================

#[test]
fn oracle_gcd_negative() {
    let a = make_i64_tensor(&[], vec![-12]);
    let b = make_i64_tensor(&[], vec![8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result).abs(), 4, "gcd(-12, 8) = ±4");
}

#[test]
fn oracle_gcd_both_negative() {
    let a = make_i64_tensor(&[], vec![-12]);
    let b = make_i64_tensor(&[], vec![-8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result).abs(), 4, "gcd(-12, -8) = ±4");
}

// ======================== LCM Basic Cases ========================

#[test]
fn oracle_lcm_basic() {
    let a = make_i64_tensor(&[], vec![4]);
    let b = make_i64_tensor(&[], vec![6]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 12, "lcm(4, 6) = 12");
}

#[test]
fn oracle_lcm_same() {
    let a = make_i64_tensor(&[], vec![7]);
    let b = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 7, "lcm(7, 7) = 7");
}

#[test]
fn oracle_lcm_one() {
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 12, "lcm(12, 1) = 12");
}

#[test]
fn oracle_lcm_zero() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "lcm(15, 0) = 0");
}

#[test]
fn oracle_lcm_coprime() {
    let a = make_i64_tensor(&[], vec![5]);
    let b = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 35, "lcm(5, 7) = 35 (coprime)");
}

// ======================== GCD/LCM Relationship ========================

#[test]
fn oracle_gcd_lcm_product() {
    // For positive integers: gcd(a,b) * lcm(a,b) = a * b
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![18]);

    let gcd_result = eval_primitive(Primitive::Gcd, &[a.clone(), b.clone()], &no_params()).unwrap();
    let lcm_result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    let gcd = extract_i64_scalar(&gcd_result);
    let lcm = extract_i64_scalar(&lcm_result);

    assert_eq!(gcd * lcm, 12 * 18, "gcd(a,b) * lcm(a,b) = a * b");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_gcd_vector() {
    let a = make_i64_tensor(&[4], vec![12, 15, 8, 100]);
    let b = make_i64_tensor(&[4], vec![8, 10, 12, 35]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![4, 5, 4, 5]);
}

#[test]
fn oracle_lcm_vector() {
    let a = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let b = make_i64_tensor(&[4], vec![3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![6, 12, 20, 30]);
}

#[test]
fn oracle_gcd_matrix() {
    let a = make_i64_tensor(&[2, 2], vec![12, 18, 24, 36]);
    let b = make_i64_tensor(&[2, 2], vec![8, 12, 16, 24]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![4, 6, 8, 12]);
}
