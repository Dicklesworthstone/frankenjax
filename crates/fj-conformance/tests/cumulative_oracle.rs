//! Oracle tests for Cumsum and Cumprod primitives.
//!
//! Tests against expected behavior matching JAX/NumPy:
//! - jnp.cumsum: cumulative sum along axis
//! - jnp.cumprod: cumulative product along axis

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

fn reverse_axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = axis_params(axis);
    p.insert("reverse".to_owned(), "true".to_owned());
    p
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
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect()
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Scalar(l) => l.as_i64().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_i64().unwrap(),
        _ => panic!("expected scalar or 0-d tensor"),
    }
}

// ======================== Cumsum Oracle Tests ========================

#[test]
fn oracle_cumsum_1d_i64() {
    // JAX: jnp.cumsum(jnp.array([1, 2, 3, 4, 5])) => [1, 3, 6, 10, 15]
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 6, 10, 15]);
}

#[test]
fn oracle_cumsum_1d_f64() {
    // JAX: jnp.cumsum(jnp.array([1.0, 2.0, 3.0])) => [1.0, 3.0, 6.0]
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_cumsum_2d_last_axis() {
    // JAX: jnp.cumsum(jnp.array([[1,2,3],[4,5,6]]), axis=-1)
    // => [[1, 3, 6], [4, 9, 15]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn oracle_cumsum_2d_first_axis() {
    // JAX: jnp.cumsum(jnp.array([[1,2,3],[4,5,6]]), axis=0)
    // => [[1, 2, 3], [5, 7, 9]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 5, 7, 9]);
}

#[test]
fn oracle_cumsum_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_cumsum_with_negatives() {
    // JAX: jnp.cumsum(jnp.array([1, -2, 3, -4])) => [1, -1, 2, -2]
    let input = make_i64_tensor(&[4], vec![1, -2, 3, -4]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, -1, 2, -2]);
}

#[test]
fn oracle_cumsum_reverse_1d_i64() {
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10, 9, 7, 4]);
}

#[test]
fn oracle_cumsum_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

// ======================== Cumprod Oracle Tests ========================

#[test]
fn oracle_cumprod_1d_i64() {
    // JAX: jnp.cumprod(jnp.array([1, 2, 3, 4])) => [1, 2, 6, 24]
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 6, 24]);
}

#[test]
fn oracle_cumprod_1d_f64() {
    // JAX: jnp.cumprod(jnp.array([1.0, 2.0, 3.0])) => [1.0, 2.0, 6.0]
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_cumprod_2d_last_axis() {
    // JAX: jnp.cumprod(jnp.array([[1,2,3],[4,5,6]]), axis=-1)
    // => [[1, 2, 6], [4, 20, 120]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 6, 4, 20, 120]);
}

#[test]
fn oracle_cumprod_2d_first_axis() {
    // JAX: jnp.cumprod(jnp.array([[1,2,3],[4,5,6]]), axis=0)
    // => [[1, 2, 3], [4, 10, 18]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 10, 18]);
}

#[test]
fn oracle_cumprod_single_element() {
    let input = make_i64_tensor(&[1], vec![7]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![7]);
}

#[test]
fn oracle_cumprod_with_zero() {
    // JAX: jnp.cumprod(jnp.array([2, 3, 0, 4])) => [2, 6, 0, 0]
    let input = make_i64_tensor(&[4], vec![2, 3, 0, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![2, 6, 0, 0]);
}

#[test]
fn oracle_cumprod_with_negatives() {
    // JAX: jnp.cumprod(jnp.array([1, -2, 3, -4])) => [1, -2, -6, 24]
    let input = make_i64_tensor(&[4], vec![1, -2, 3, -4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, -2, -6, 24]);
}

#[test]
fn oracle_cumprod_all_ones() {
    let input = make_i64_tensor(&[5], vec![1, 1, 1, 1, 1]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1, 1]);
}

#[test]
fn oracle_cumprod_reverse_2d_last_axis() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &reverse_axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![6, 6, 3, 120, 30, 6]);
}

#[test]
fn oracle_cumulative_rejects_invalid_reverse_param() {
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let mut params = axis_params(0);
    params.insert("reverse".to_owned(), "maybe".to_owned());
    let err = eval_primitive(Primitive::Cumsum, &[input], &params)
        .expect_err("invalid reverse parameter should fail");
    assert!(
        err.to_string().contains("reverse"),
        "unexpected error: {err}"
    );
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_cumsum_last_equals_sum() {
    // last(cumsum(x)) = reduce_sum(x)
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let cumsum_result = eval_primitive(Primitive::Cumsum, &[input.clone()], &no_params()).unwrap();
    let cumsum_vals = extract_i64_vec(&cumsum_result);

    let sum_result = eval_primitive(Primitive::ReduceSum, &[input], &axis_params(0)).unwrap();
    let sum_val = extract_i64_scalar(&sum_result);

    assert_eq!(
        cumsum_vals.last().copied(),
        Some(sum_val),
        "last(cumsum(x)) should equal reduce_sum(x)"
    );
}

#[test]
fn metamorphic_cumsum_first_element_identity() {
    // cumsum(x)[0] = x[0]
    let input = make_i64_tensor(&[4], vec![7, 2, 9, 3]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 7, "cumsum(x)[0] should equal x[0]");
}

#[test]
fn metamorphic_cumprod_last_equals_product() {
    // last(cumprod(x)) = reduce_prod(x)
    let input = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let cumprod_result = eval_primitive(Primitive::Cumprod, &[input.clone()], &no_params()).unwrap();
    let cumprod_vals = extract_i64_vec(&cumprod_result);

    let prod_result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(0)).unwrap();
    let prod_val = extract_i64_scalar(&prod_result);

    assert_eq!(
        cumprod_vals.last().copied(),
        Some(prod_val),
        "last(cumprod(x)) should equal reduce_prod(x)"
    );
}

#[test]
fn metamorphic_cumprod_first_element_identity() {
    // cumprod(x)[0] = x[0]
    let input = make_i64_tensor(&[4], vec![5, 3, 2, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 5, "cumprod(x)[0] should equal x[0]");
}
