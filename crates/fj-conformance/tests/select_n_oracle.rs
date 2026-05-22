//! Oracle tests for the SelectN primitive.
//!
//! Upstream `jax.lax.select_n(which, *cases)` selects case values by integer
//! index. Boolean `which` is allowed when len(cases) <= 2, where false selects
//! case 0 and true selects case 1.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn vector_f64(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn vector_i64(values: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn vector_bool(values: &[bool]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
}

fn select_n(inputs: Vec<Value>) -> Result<Value, fj_lax::EvalError> {
    eval_primitive(Primitive::SelectN, &inputs, &no_params())
}

fn extract_scalar(value: &Value) -> f64 {
    value.as_f64_scalar().expect("expected f64 scalar")
}

fn extract_vector(value: &Value) -> Vec<f64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_f64().expect("expected f64 literal"))
        .collect()
}

#[test]
fn select_n_scalar_index_picks_each_case() {
    let cases = || {
        vec![
            Value::scalar_f64(10.0),
            Value::scalar_f64(20.0),
            Value::scalar_f64(30.0),
        ]
    };

    for (index, expected) in [(0, 10.0), (1, 20.0), (2, 30.0)] {
        let mut inputs = vec![Value::scalar_i64(index)];
        inputs.extend(cases());
        let result = select_n(inputs).expect("scalar select_n should succeed");
        assert_eq!(extract_scalar(&result), expected);
    }
}

#[test]
fn select_n_scalar_index_selects_whole_tensor_case() {
    let result = select_n(vec![
        Value::scalar_i64(1),
        vector_f64(&[1.0, 2.0, 3.0]),
        vector_f64(&[4.0, 5.0, 6.0]),
    ])
    .expect("scalar-index tensor select_n should succeed");

    let tensor = result.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(3));
    assert_eq!(extract_vector(&result), vec![4.0, 5.0, 6.0]);
}

#[test]
fn select_n_tensor_index_selects_elementwise() {
    let result = select_n(vec![
        vector_i64(&[0, 1, 0, 1]),
        vector_f64(&[1.0, 2.0, 3.0, 4.0]),
        vector_f64(&[10.0, 20.0, 30.0, 40.0]),
    ])
    .expect("tensor-index select_n should succeed");

    let tensor = result.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(4));
    assert_eq!(extract_vector(&result), vec![1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn select_n_tensor_index_supports_three_cases() {
    let result = select_n(vec![
        vector_i64(&[0, 1, 2, 1]),
        vector_f64(&[1.0, 2.0, 3.0, 4.0]),
        vector_f64(&[10.0, 20.0, 30.0, 40.0]),
        vector_f64(&[100.0, 200.0, 300.0, 400.0]),
    ])
    .expect("three-case select_n should succeed");

    assert_eq!(extract_vector(&result), vec![1.0, 20.0, 300.0, 40.0]);
}

#[test]
fn select_n_rejects_missing_cases() {
    let err = select_n(vec![Value::scalar_i64(0)]).expect_err("case list should be non-empty");

    assert!(
        err.to_string().contains("arity")
            || err.to_string().contains("expected")
            || err.to_string().contains("actual"),
        "unexpected missing-cases error: {err}"
    );
}

#[test]
fn select_n_rejects_out_of_bounds_scalar_index() {
    let err = select_n(vec![
        Value::scalar_i64(2),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
    ])
    .expect_err("out-of-bounds select_n index should fail");

    assert!(
        err.to_string().contains("out of bounds"),
        "unexpected out-of-bounds error: {err}"
    );
}

#[test]
fn select_n_rejects_tensor_index_shape_mismatch() {
    let err = select_n(vec![
        vector_i64(&[0, 1]),
        vector_f64(&[1.0, 2.0, 3.0]),
        vector_f64(&[10.0, 20.0, 30.0]),
    ])
    .expect_err("index shape mismatch should fail");

    assert!(
        err.to_string().contains("index shape"),
        "unexpected index-shape error: {err}"
    );
}

#[test]
fn select_n_rejects_operand_shape_mismatch() {
    let err = select_n(vec![
        vector_i64(&[0, 1, 0]),
        vector_f64(&[1.0, 2.0, 3.0]),
        vector_f64(&[10.0, 20.0]),
    ])
    .expect_err("operand shape mismatch should fail");

    assert!(
        err.to_string().contains("matching shapes"),
        "unexpected operand-shape error: {err}"
    );
}

// ======================== Boolean which tests ========================

#[test]
fn select_n_boolean_scalar_false_picks_case_0() {
    let result = select_n(vec![
        Value::scalar_bool(false),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
    ])
    .expect("boolean false should select case 0");

    assert_eq!(extract_scalar(&result), 10.0);
}

#[test]
fn select_n_boolean_scalar_true_picks_case_1() {
    let result = select_n(vec![
        Value::scalar_bool(true),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
    ])
    .expect("boolean true should select case 1");

    assert_eq!(extract_scalar(&result), 20.0);
}

#[test]
fn select_n_boolean_scalar_picks_whole_tensor_case() {
    for (which, expected) in [(false, vec![1.0, 2.0, 3.0]), (true, vec![10.0, 20.0, 30.0])] {
        let result = select_n(vec![
            Value::scalar_bool(which),
            vector_f64(&[1.0, 2.0, 3.0]),
            vector_f64(&[10.0, 20.0, 30.0]),
        ])
        .expect("scalar boolean select_n should pick whole tensor cases");

        let tensor = result.as_tensor().expect("expected tensor output");
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape, Shape::vector(3));
        assert_eq!(extract_vector(&result), expected);
    }
}

#[test]
fn select_n_boolean_tensor_index_selects_elementwise() {
    let result = select_n(vec![
        vector_bool(&[false, true, false, true]),
        vector_f64(&[1.0, 2.0, 3.0, 4.0]),
        vector_f64(&[10.0, 20.0, 30.0, 40.0]),
    ])
    .expect("boolean tensor index should select elementwise");

    assert_eq!(extract_vector(&result), vec![1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn select_n_boolean_with_single_case_false() {
    let result = select_n(vec![Value::scalar_bool(false), Value::scalar_f64(42.0)])
        .expect("boolean false with single case should succeed");

    assert_eq!(extract_scalar(&result), 42.0);
}

#[test]
fn select_n_boolean_with_three_cases_rejected() {
    let err = select_n(vec![
        Value::scalar_bool(true),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
        Value::scalar_f64(30.0),
    ])
    .expect_err("boolean with 3 cases should fail");

    assert!(
        err.to_string().contains("at most 2 operands") || err.to_string().contains("boolean"),
        "unexpected boolean-3-cases error: {err}"
    );
}

// ======================== Dtype parity tests ========================

#[test]
fn select_n_rejects_mismatched_case_dtypes_scalar() {
    let err = select_n(vec![
        Value::scalar_i64(0),
        Value::scalar_f64(10.0),
        Value::scalar_i64(20),
    ])
    .expect_err("mismatched case dtypes should fail");

    assert!(
        err.to_string().contains("dtypes must match")
            || err.to_string().contains("F64")
            || err.to_string().contains("I64"),
        "unexpected dtype-mismatch error: {err}"
    );
}

#[test]
fn select_n_rejects_mismatched_case_dtypes_tensor() {
    let err = select_n(vec![
        vector_i64(&[0, 1]),
        vector_f64(&[1.0, 2.0]),
        vector_i64(&[10, 20]),
    ])
    .expect_err("mismatched tensor case dtypes should fail");

    assert!(
        err.to_string().contains("dtypes must match")
            || err.to_string().contains("F64")
            || err.to_string().contains("I64"),
        "unexpected tensor dtype-mismatch error: {err}"
    );
}

#[test]
fn select_n_accepts_matching_dtypes() {
    let result = select_n(vec![
        Value::scalar_i64(1),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
        Value::scalar_f64(30.0),
    ])
    .expect("matching dtypes should succeed");

    assert_eq!(extract_scalar(&result), 20.0);
}
