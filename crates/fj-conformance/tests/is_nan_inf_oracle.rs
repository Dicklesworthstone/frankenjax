//! Oracle tests for IsNan and IsInf predicates.
//!
//! isnan(x) = true if x is NaN
//! isinf(x) = true if x is +inf or -inf
//!
//! Tests:
//! - Normal values: isnan/isinf return false
//! - Zero: neither NaN nor Inf
//! - Infinity: isinf true, isnan false
//! - NaN: isnan true, isinf false
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

fn extract_bool_literal(literal: &Literal) -> bool {
    match literal {
        Literal::Bool(value) => *value,
        other => {
            assert!(matches!(other, Literal::Bool(_)), "expected Bool literal");
            false
        }
    }
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(extract_bool_literal).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            extract_bool_literal(&t.elements[0])
        }
        Value::Scalar(literal) => extract_bool_literal(literal),
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

// ======================== IsNan: Normal Values ========================

#[test]
fn oracle_isnan_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(0) = false");
}

#[test]
fn oracle_isnan_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-0) = false");
}

#[test]
fn oracle_isnan_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(1) = false");
}

#[test]
fn oracle_isnan_negative() {
    let input = make_f64_tensor(&[], vec![-42.5]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-42.5) = false");
}

// ======================== IsNan: Special Values ========================

#[test]
fn oracle_isnan_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(+inf) = false");
}

#[test]
fn oracle_isnan_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-inf) = false");
}

#[test]
fn oracle_isnan_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isnan(NaN) = true");
}

// ======================== IsInf: Normal Values ========================

#[test]
fn oracle_isinf_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(0) = false");
}

#[test]
fn oracle_isinf_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(-0) = false");
}

#[test]
fn oracle_isinf_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(1) = false");
}

#[test]
fn oracle_isinf_large() {
    let input = make_f64_tensor(&[], vec![1e308]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(1e308) = false");
}

// ======================== IsInf: Special Values ========================

#[test]
fn oracle_isinf_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isinf(+inf) = true");
}

#[test]
fn oracle_isinf_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isinf(-inf) = true");
}

#[test]
fn oracle_isinf_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(NaN) = false");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_isnan_vector() {
    let input = make_f64_tensor(
        &[5],
        vec![0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, false, true, false, false]
    );
}

#[test]
fn oracle_isinf_vector() {
    let input = make_f64_tensor(
        &[5],
        vec![0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, false, false, true, true]
    );
}

#[test]
fn oracle_isnan_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, f64::NAN, f64::INFINITY, -1.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, false, false]);
}

#[test]
fn oracle_isinf_matrix() {
    let input = make_f64_tensor(
        &[2, 2],
        vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, false, true, true]);
}

// ======================== Output DType ========================

#[test]
fn oracle_isnan_output_dtype() {
    let input = make_f64_tensor(&[2], vec![0.0, f64::NAN]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "isnan output dtype should be Bool");
    }
}

#[test]
fn oracle_isinf_output_dtype() {
    let input = make_f64_tensor(&[2], vec![0.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "isinf output dtype should be Bool");
    }
}
