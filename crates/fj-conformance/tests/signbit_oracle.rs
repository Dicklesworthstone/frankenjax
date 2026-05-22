//! Oracle tests for Signbit primitive.
//!
//! signbit(x) = true if the sign bit is set (negative numbers and -0.0)
//!
//! Tests:
//! - Positive values: signbit = false
//! - Negative values: signbit = true
//! - Zero: signbit(+0.0) = false, signbit(-0.0) = true
//! - Infinity: signbit(+inf) = false, signbit(-inf) = true
//! - NaN: signbit preserves sign of NaN
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

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected Bool literal"),
            })
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            match &t.elements[0] {
                Literal::Bool(b) => *b,
                _ => panic!("expected Bool literal"),
            }
        }
        Value::Scalar(Literal::Bool(b)) => *b,
        _ => panic!("expected Bool scalar"),
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

// ======================== Positive Values ========================

#[test]
fn oracle_signbit_positive_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(+0.0) = false");
}

#[test]
fn oracle_signbit_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1.0) = false");
}

#[test]
fn oracle_signbit_large_positive() {
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1e100) = false");
}

#[test]
fn oracle_signbit_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1e-100) = false");
}

// ======================== Negative Values ========================

#[test]
fn oracle_signbit_negative_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-0.0) = true");
}

#[test]
fn oracle_signbit_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1.0) = true");
}

#[test]
fn oracle_signbit_large_negative() {
    let input = make_f64_tensor(&[], vec![-1e100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1e100) = true");
}

#[test]
fn oracle_signbit_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1e-100) = true");
}

// ======================== Infinity ========================

#[test]
fn oracle_signbit_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(+inf) = false");
}

#[test]
fn oracle_signbit_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-inf) = true");
}

// ======================== NaN ========================

#[test]
fn oracle_signbit_positive_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(
        !extract_bool_scalar(&result),
        "signbit(+NaN) = false (positive NaN)"
    );
}

#[test]
fn oracle_signbit_negative_nan() {
    let neg_nan = -f64::NAN;
    let input = make_f64_tensor(&[], vec![neg_nan]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(
        extract_bool_scalar(&result),
        "signbit(-NaN) = true (negative NaN)"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_signbit_vector() {
    let input = make_f64_tensor(&[5], vec![1.0, -1.0, 0.0, -0.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, true, false, true, false]
    );
}

#[test]
fn oracle_signbit_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, -1.0, -0.0, 0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, true, false]);
}

// ======================== Output DType ========================

#[test]
fn oracle_signbit_output_dtype() {
    let input = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "signbit output dtype should be Bool");
    }
}
