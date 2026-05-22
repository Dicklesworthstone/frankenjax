//! Oracle tests for Argmin and Argmax primitives.
//!
//! These pin JAX/NumPy-compatible index-of-extremum behavior:
//! - explicit positive and negative axes
//! - first index wins for ties
//! - scalar inputs return index 0

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), axis.to_string());
    params
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

fn extract_i64_scalar(value: &Value) -> i64 {
    if let Value::Scalar(Literal::I64(v)) = value {
        *v
    } else {
        assert!(
            matches!(value, Value::Scalar(Literal::I64(_))),
            "expected scalar i64, got {value:?}"
        );
        0
    }
}

fn extract_i64_vec(value: &Value) -> Vec<i64> {
    if let Value::Tensor(tensor) = value {
        tensor
            .elements
            .iter()
            .map(|literal| {
                if let Some(index) = literal.as_i64() {
                    index
                } else {
                    assert!(
                        literal.as_i64().is_some(),
                        "expected i64 literal, got {literal:?}"
                    );
                    0
                }
            })
            .collect()
    } else {
        assert!(
            matches!(value, Value::Tensor(_)),
            "expected tensor i64, got {value:?}"
        );
        Vec::new()
    }
}

fn extract_shape(value: &Value) -> Vec<u32> {
    match value {
        Value::Tensor(tensor) => tensor.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

#[test]
fn oracle_argmin_1d_returns_first_minimum_index() {
    // JAX: jnp.argmin([3, 1, 4, 1, 5]) == 1
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_argmax_1d_returns_first_maximum_index() {
    // JAX: jnp.argmax([3, 5, 4, 5, 1]) == 1
    let input = make_f64_tensor(&[5], vec![3.0, 5.0, 4.0, 5.0, 1.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_argmin_2d_axis0_reduces_rows() {
    // JAX: jnp.argmin([[1, 4, 2], [3, 0, 5]], axis=0) == [0, 1, 0]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 0]);
}

#[test]
fn oracle_argmax_2d_axis0_reduces_rows() {
    // JAX: jnp.argmax([[1, 4, 2], [3, 0, 5]], axis=0) == [1, 0, 1]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 1]);
}

#[test]
fn oracle_argmax_2d_axis1_reduces_columns() {
    // JAX: jnp.argmax([[1, 4, 2], [3, 0, 5]], axis=1) == [1, 2]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2]);
}

#[test]
fn oracle_argmax_negative_axis_matches_last_axis() {
    // JAX: axis=-1 addresses the last axis.
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2]);
}

#[test]
fn oracle_argmin_negative_axis_matches_first_axis() {
    // JAX: axis=-2 addresses axis 0 for a rank-2 value.
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(-2)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 0]);
}

#[test]
fn oracle_argmin_scalar_returns_zero() {
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(Primitive::Argmin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmax_scalar_returns_zero() {
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(Primitive::Argmax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmin_axis_out_of_bounds_errors() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let err = eval_primitive(Primitive::Argmin, &[input], &axis_params(2)).unwrap_err();
    assert!(
        err.to_string().contains("axis 2 out of bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn oracle_argmax_negative_axis_out_of_bounds_errors() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let err = eval_primitive(Primitive::Argmax, &[input], &axis_params(-3)).unwrap_err();
    assert!(
        err.to_string().contains("axis -3 out of bounds"),
        "unexpected error: {err}"
    );
}

// ======================== Additional Coverage ========================

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

#[test]
fn oracle_argmin_negative_values() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2); // -4 at index 2
}

#[test]
fn oracle_argmax_negative_values() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // -1 at index 1
}

#[test]
fn oracle_argmin_all_equal_returns_first() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0); // first index
}

#[test]
fn oracle_argmax_all_equal_returns_first() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0); // first index
}

#[test]
fn oracle_argmin_integer_dtype() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // 10 at index 1
}

#[test]
fn oracle_argmax_integer_dtype() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2); // 40 at index 2
}

#[test]
fn oracle_argmin_3d_axis1() {
    // [2, 3, 2] tensor, reduce along axis 1
    let input = make_f64_tensor(
        &[2, 3, 2],
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
    );
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn oracle_argmax_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmin_inf_values() {
    let input = make_f64_tensor(&[4], vec![1.0, f64::NEG_INFINITY, 2.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // -inf at index 1
}

#[test]
fn oracle_argmax_inf_values() {
    let input = make_f64_tensor(&[4], vec![1.0, f64::NEG_INFINITY, 2.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 3); // +inf at index 3
}

#[test]
fn oracle_argmin_empty_error() {
    let input = make_f64_tensor(&[0], vec![]);
    let err = eval_primitive(Primitive::Argmin, &[input], &axis_params(0));
    assert!(err.is_err(), "argmin on empty tensor should error");
}

#[test]
fn oracle_argmax_large_tensor() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 99);
}

#[test]
fn oracle_argmin_output_dtype_is_i64() {
    let input = make_f64_tensor(&[3], vec![3.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert!(
        matches!(result.dtype(), DType::I32 | DType::I64),
        "argmin should return integer type"
    );
}
