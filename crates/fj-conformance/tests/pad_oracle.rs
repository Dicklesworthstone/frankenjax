//! Oracle tests for Pad primitive.
//!
//! Tests against expected behavior matching JAX/lax.pad:
//! - padding_low: elements to add before each dimension
//! - padding_high: elements to add after each dimension
//! - padding_interior: elements to add between existing elements

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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

fn pad_params(low: &[i64], high: &[i64], interior: &[i64]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "padding_low".to_string(),
        low.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert(
        "padding_high".to_string(),
        high.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert(
        "padding_interior".to_string(),
        interior
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn slice_params_with_strides(
    starts: &[usize],
    limits: &[usize],
    strides: &[usize],
) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "start_indices".to_string(),
        starts
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert(
        "limit_indices".to_string(),
        limits
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert(
        "strides".to_string(),
        strides
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== 1D Padding Tests ========================

#[test]
fn oracle_pad_1d_low() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(2, 0, 0)]) => [0, 0, 1, 2, 3]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[2], &[0], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 1, 2, 3]);
}

#[test]
fn oracle_pad_1d_high() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(0, 2, 0)]) => [1, 2, 3, 0, 0]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0], &[2], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 0, 0]);
}

#[test]
fn oracle_pad_1d_both() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(1, 2, 0)]) => [0, 1, 2, 3, 0, 0]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[1], &[2], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 0, 0]);
}

#[test]
fn oracle_pad_1d_interior() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(0, 0, 1)]) => [1, 0, 2, 0, 3]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0], &[0], &[1]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 2, 0, 3]);
}

#[test]
fn oracle_pad_1d_interior_2() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(0, 0, 2)]) => [1, 0, 0, 2, 0, 0, 3]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0], &[0], &[2]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![7]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 0, 2, 0, 0, 3]);
}

#[test]
fn oracle_pad_1d_all() {
    // JAX: lax.pad(jnp.array([1, 2, 3]), 0, [(1, 1, 1)]) => [0, 1, 0, 2, 0, 3, 0]
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[1], &[1], &[1]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![7]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 0, 2, 0, 3, 0]);
}

// ======================== 2D Padding Tests ========================

#[test]
fn oracle_pad_2d_low_axis0() {
    // JAX: lax.pad(jnp.array([[1,2],[3,4]]), 0, [(1,0,0),(0,0,0)])
    // => [[0, 0], [1, 2], [3, 4]]
    let operand = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[1, 0], &[0, 0], &[0, 0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 1, 2, 3, 4]);
}

#[test]
fn oracle_pad_2d_high_axis1() {
    // JAX: lax.pad(jnp.array([[1,2],[3,4]]), 0, [(0,0,0),(0,1,0)])
    // => [[1, 2, 0], [3, 4, 0]]
    let operand = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0, 0], &[0, 1], &[0, 0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 0, 3, 4, 0]);
}

#[test]
fn oracle_pad_2d_interior_axis1() {
    // JAX: lax.pad(jnp.array([[1,2,3],[4,5,6]]), 0, [(0,0,0),(0,0,1)])
    // => [[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]
    let operand = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0, 0], &[0, 0], &[0, 1]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 2, 0, 3, 4, 0, 5, 0, 6]);
}

#[test]
fn oracle_pad_2d_all_axes() {
    // JAX: lax.pad(jnp.array([[1,2],[3,4]]), 9, [(1,1,0),(1,1,0)])
    // => [[9, 9, 9, 9],
    //     [9, 1, 2, 9],
    //     [9, 3, 4, 9],
    //     [9, 9, 9, 9]]
    let operand = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let pad_value = Value::scalar_i64(9);
    let params = pad_params(&[1, 1], &[1, 1], &[0, 0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 4]);
    assert_eq!(
        extract_i64_vec(&result),
        vec![9, 9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9, 9, 9, 9]
    );
}

// ======================== Negative Padding (Cropping) Tests ========================

#[test]
fn oracle_pad_negative_low() {
    // JAX: lax.pad(jnp.array([1, 2, 3, 4, 5]), 0, [(-2, 0, 0)]) => [3, 4, 5]
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[-2], &[0], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![3, 4, 5]);
}

#[test]
fn oracle_pad_negative_high() {
    // JAX: lax.pad(jnp.array([1, 2, 3, 4, 5]), 0, [(0, -2, 0)]) => [1, 2, 3]
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0], &[-2], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_pad_negative_both() {
    // JAX: lax.pad(jnp.array([1, 2, 3, 4, 5]), 0, [(-1, -1, 0)]) => [2, 3, 4]
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[-1], &[-1], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![2, 3, 4]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_pad_single_element() {
    let operand = make_i64_tensor(&[1], vec![42]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[2], &[3], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 42, 0, 0, 0]);
}

#[test]
fn oracle_pad_no_padding() {
    let operand = make_i64_tensor(&[3], vec![1, 2, 3]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[0], &[0], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_pad_f64() {
    let operand = make_f64_tensor(&[3], vec![1.5, 2.5, 3.5]);
    let pad_value = Value::Scalar(Literal::from_f64(0.0));
    let params = pad_params(&[1], &[1], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![5]);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[1] - 1.5).abs() < 1e-10);
    assert!((vals[2] - 2.5).abs() < 1e-10);
    assert!((vals[3] - 3.5).abs() < 1e-10);
    assert!((vals[4] - 0.0).abs() < 1e-10);
}

#[test]
fn oracle_pad_empty_input() {
    // Empty input with padding produces padded output
    let operand = make_i64_tensor(&[0], vec![]);
    let pad_value = Value::scalar_i64(9);
    let params = pad_params(&[2], &[3], &[0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![9, 9, 9, 9, 9]);
}

#[test]
fn oracle_pad_3d() {
    // 3D tensor [2, 2, 2] with low padding on axis 0
    let operand = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let pad_value = Value::scalar_i64(0);
    let params = pad_params(&[1, 0, 0], &[0, 0, 0], &[0, 0, 0]);
    let result = eval_primitive(Primitive::Pad, &[operand, pad_value], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2, 2]);
    assert_eq!(
        extract_i64_vec(&result),
        vec![0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[test]
fn metamorphic_edge_pad_then_slice_recovers_operand() {
    let operand = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let expected_shape = extract_shape(&operand);
    let expected_values = extract_i64_vec(&operand);
    let padded = eval_primitive(
        Primitive::Pad,
        &[operand, Value::scalar_i64(-7)],
        &pad_params(&[1, 2], &[2, 1], &[0, 0]),
    )
    .unwrap();
    let recovered = eval_primitive(
        Primitive::Slice,
        &[padded],
        &slice_params_with_strides(&[1, 2], &[3, 5], &[1, 1]),
    )
    .unwrap();

    assert_eq!(extract_shape(&recovered), expected_shape);
    assert_eq!(extract_i64_vec(&recovered), expected_values);
}

#[test]
fn metamorphic_interior_pad_strided_slice_recovers_operand() {
    let operand = make_i64_tensor(&[4], vec![3, 5, 8, 13]);
    let expected_shape = extract_shape(&operand);
    let expected_values = extract_i64_vec(&operand);
    let padded = eval_primitive(
        Primitive::Pad,
        &[operand, Value::scalar_i64(-1)],
        &pad_params(&[1], &[1], &[2]),
    )
    .unwrap();
    let recovered = eval_primitive(
        Primitive::Slice,
        &[padded],
        &slice_params_with_strides(&[1], &[11], &[3]),
    )
    .unwrap();

    assert_eq!(extract_shape(&recovered), expected_shape);
    assert_eq!(extract_i64_vec(&recovered), expected_values);
}
