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

// ======================== Additional Coverage ========================

fn matrix_f64(rows: u32, cols: u32, values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![rows, cols] },
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn matrix_i64(rows: u32, cols: u32, values: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![rows, cols] },
            values.iter().copied().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

#[test]
fn select_n_2d_elementwise() {
    // 2D tensor index selects elementwise from 2D cases
    let result = select_n(vec![
        matrix_i64(2, 2, &[0, 1, 1, 0]),
        matrix_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]),
        matrix_f64(2, 2, &[10.0, 20.0, 30.0, 40.0]),
    ])
    .expect("2D select_n should succeed");

    let tensor = result.as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![2, 2]);
    let vals: Vec<f64> = tensor.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert_eq!(vals, vec![1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn select_n_empty_tensors() {
    let result = select_n(vec![
        Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap(),
        ),
        Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap(),
        ),
        Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap(),
        ),
    ])
    .expect("empty tensor select_n should succeed");

    let tensor = result.as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![0]);
    assert!(tensor.elements.is_empty());
}

#[test]
fn select_n_preserves_output_dtype() {
    let result = select_n(vec![
        Value::scalar_i64(0),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
    ])
    .expect("select_n should succeed");

    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn select_n_negative_index_rejected() {
    let err = select_n(vec![
        Value::scalar_i64(-1),
        Value::scalar_f64(10.0),
        Value::scalar_f64(20.0),
    ])
    .expect_err("negative index should fail");

    assert!(
        err.to_string().contains("out of bounds") || err.to_string().contains("negative"),
        "unexpected negative-index error: {err}"
    );
}

#[test]
fn select_n_tensor_negative_index_rejected() {
    let err = select_n(vec![
        vector_i64(&[0, -1, 1]),
        vector_f64(&[1.0, 2.0, 3.0]),
        vector_f64(&[10.0, 20.0, 30.0]),
    ])
    .expect_err("tensor with negative index should fail");

    assert!(
        err.to_string().contains("out of bounds") || err.to_string().contains("negative"),
        "unexpected tensor-negative-index error: {err}"
    );
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_select_n_preserves_dtype() {
    let result = select_n(vec![
        Value::scalar_i64(0),
        Value::scalar_f64(1.0),
        Value::scalar_f64(2.0),
    ])
    .unwrap();
    assert_eq!(result.dtype(), DType::F64, "select_n should preserve F64 dtype");
}

// ======================== Complex Type Tests ========================

fn scalar_complex64(re: f32, im: f32) -> Value {
    Value::Scalar(Literal::from_complex64(re, im))
}

fn scalar_complex128(re: f64, im: f64) -> Value {
    Value::Scalar(Literal::from_complex128(re, im))
}

fn vector_complex64(data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape::vector(data.len() as u32),
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64(v: &Value) -> (f32, f32) {
    match v {
        Value::Scalar(l) => l.as_complex64().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_complex64().unwrap(),
        _ => panic!("expected scalar complex64"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect()
}

#[test]
fn oracle_select_n_complex64_scalar_case0() {
    // select_n(0, [1+i, 2+2i]) = 1+i
    let result = select_n(vec![
        Value::scalar_i64(0),
        scalar_complex64(1.0, 1.0),
        scalar_complex64(2.0, 2.0),
    ])
    .expect("select_n complex64 should succeed");
    let (re, im) = extract_complex64(&result);
    assert!((re - 1.0).abs() < 1e-5);
    assert!((im - 1.0).abs() < 1e-5);
}

#[test]
fn oracle_select_n_complex64_scalar_case1() {
    // select_n(1, [1+i, 2+2i]) = 2+2i
    let result = select_n(vec![
        Value::scalar_i64(1),
        scalar_complex64(1.0, 1.0),
        scalar_complex64(2.0, 2.0),
    ])
    .expect("select_n complex64 should succeed");
    let (re, im) = extract_complex64(&result);
    assert!((re - 2.0).abs() < 1e-5);
    assert!((im - 2.0).abs() < 1e-5);
}

#[test]
fn oracle_select_n_complex64_tensor() {
    // select_n([0, 1, 0], [[1+i, 2+i, 3+i], [10+i, 20+i, 30+i]])
    // = [1+i, 20+i, 3+i]
    let result = select_n(vec![
        vector_i64(&[0, 1, 0]),
        vector_complex64(vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]),
        vector_complex64(vec![(10.0, 1.0), (20.0, 1.0), (30.0, 1.0)]),
    ])
    .expect("select_n complex64 tensor should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-5);
    assert!((vals[1].0 - 20.0).abs() < 1e-5);
    assert!((vals[2].0 - 3.0).abs() < 1e-5);
}

#[test]
fn oracle_select_n_complex128_preserves_dtype() {
    let result = select_n(vec![
        Value::scalar_i64(0),
        scalar_complex128(1.0, 1.0),
        scalar_complex128(2.0, 2.0),
    ])
    .expect("select_n complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn property_select_n_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (case0, case1) = match dtype {
            DType::Complex64 => (
                scalar_complex64(1.0, 0.0),
                scalar_complex64(2.0, 0.0),
            ),
            DType::Complex128 => (
                scalar_complex128(1.0, 0.0),
                scalar_complex128(2.0, 0.0),
            ),
            _ => unreachable!(),
        };
        let result = select_n(vec![Value::scalar_i64(0), case0, case1])
            .expect("select_n should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "select_n {dtype:?}: dtype mismatch");
    }
}
