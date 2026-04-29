//! Oracle tests for Slice primitive.
//!
//! Tests against expected behavior matching JAX/lax.slice:
//! - start_indices: starting position for each dimension
//! - limit_indices: ending position (exclusive) for each dimension
//! - strides: optional step size for each dimension (default 1)

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

fn slice_params(starts: &[usize], limits: &[usize]) -> BTreeMap<String, String> {
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
    p
}

fn slice_params_with_strides(
    starts: &[usize],
    limits: &[usize],
    strides: &[usize],
) -> BTreeMap<String, String> {
    let mut p = slice_params(starts, limits);
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

// ======================== 1D Tests ========================

#[test]
fn oracle_slice_1d_basic() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (1,), (4,)) => [1, 2, 3]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[1], &[4])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_slice_1d_from_start() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (0,), (3,)) => [0, 1, 2]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[0], &[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2]);
}

#[test]
fn oracle_slice_1d_to_end() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (2,), (5,)) => [2, 3, 4]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[2], &[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![2, 3, 4]);
}

#[test]
fn oracle_slice_1d_single_element() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (2,), (3,)) => [2]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[2], &[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![2]);
}

#[test]
fn oracle_slice_1d_full_copy() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (0,), (5,)) => [0, 1, 2, 3, 4]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[0], &[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_slice_1d_empty() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (2,), (2,)) => []
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[2], &[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

// ======================== 1D Stride Tests ========================

#[test]
fn oracle_slice_1d_stride_2() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4, 5]), (0,), (6,), (2,)) => [0, 2, 4]
    let input = make_i64_tensor(&[6], vec![0, 1, 2, 3, 4, 5]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0], &[6], &[2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 2, 4]);
}

#[test]
fn oracle_slice_1d_stride_3() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), (0,), (9,), (3,)) => [0, 3, 6]
    let input = make_i64_tensor(&[9], vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0], &[9], &[3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 3, 6]);
}

#[test]
fn oracle_slice_1d_stride_with_offset() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4, 5]), (1,), (6,), (2,)) => [1, 3, 5]
    let input = make_i64_tensor(&[6], vec![0, 1, 2, 3, 4, 5]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[1], &[6], &[2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 5]);
}

#[test]
fn oracle_slice_1d_large_stride() {
    // JAX: lax.slice(jnp.array([0, 1, 2, 3, 4]), (0,), (5,), (4,)) => [0, 4]
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0], &[5], &[4]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![0, 4]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_slice_2d_rows() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9]]), (1,0), (3,3))
    // => [[4, 5, 6], [7, 8, 9]]
    let input = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[1, 0], &[3, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![4, 5, 6, 7, 8, 9]);
}

#[test]
fn oracle_slice_2d_cols() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9]]), (0,1), (3,3))
    // => [[2, 3], [5, 6], [8, 9]]
    let input = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 1], &[3, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 3, 5, 6, 8, 9]);
}

#[test]
fn oracle_slice_2d_submatrix() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9]]), (0,0), (2,2))
    // => [[1, 2], [4, 5]]
    let input = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 0], &[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 4, 5]);
}

#[test]
fn oracle_slice_2d_center() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9]]), (1,1), (2,2))
    // => [[5]]
    let input = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[1, 1], &[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![5]);
}

#[test]
fn oracle_slice_2d_bottom_right() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9]]), (1,1), (3,3))
    // => [[5, 6], [8, 9]]
    let input = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[1, 1], &[3, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![5, 6, 8, 9]);
}

#[test]
fn oracle_slice_2d_single_row() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6]]), (0,0), (1,3))
    // => [[1, 2, 3]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 0], &[1, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_slice_2d_single_col() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6]]), (0,1), (2,2))
    // => [[2], [5]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 1], &[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1]);
    assert_eq!(extract_i64_vec(&result), vec![2, 5]);
}

// ======================== 2D Stride Tests ========================

#[test]
fn oracle_slice_2d_stride_rows() {
    // JAX: lax.slice(jnp.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), (0,0), (4,3), (2,1))
    // => [[1, 2, 3], [7, 8, 9]]
    let input = make_i64_tensor(&[4, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0, 0], &[4, 3], &[2, 1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 7, 8, 9]);
}

#[test]
fn oracle_slice_2d_stride_cols() {
    // JAX: lax.slice(jnp.array([[1,2,3,4],[5,6,7,8]]), (0,0), (2,4), (1,2))
    // => [[1, 3], [5, 7]]
    let input = make_i64_tensor(&[2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0, 0], &[2, 4], &[1, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 5, 7]);
}

#[test]
fn oracle_slice_2d_stride_both() {
    // JAX: lax.slice(jnp.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), (0,0), (4,4), (2,2))
    // => [[1, 3], [9, 11]]
    let input = make_i64_tensor(
        &[4, 4],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0, 0], &[4, 4], &[2, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 9, 11]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_slice_3d_first_axis() {
    // Shape [2, 2, 3], slice axis 0: [1:2, :, :]
    let input = make_i64_tensor(&[2, 2, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params(&[1, 0, 0], &[2, 2, 3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![7, 8, 9, 10, 11, 12]);
}

#[test]
fn oracle_slice_3d_last_axis() {
    // Shape [2, 2, 3], slice axis 2: [:, :, 1:3]
    let input = make_i64_tensor(&[2, 2, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params(&[0, 0, 1], &[2, 2, 3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 3, 5, 6, 8, 9, 11, 12]);
}

#[test]
fn oracle_slice_3d_middle_axis() {
    // Shape [2, 3, 2], slice axis 1: [:, 1:3, :]
    let input = make_i64_tensor(&[2, 3, 2], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params(&[0, 1, 0], &[2, 3, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 4, 5, 6, 9, 10, 11, 12]);
}

#[test]
fn oracle_slice_3d_all_axes() {
    // Shape [3, 3, 3], slice [1:2, 1:3, 0:2]
    let input = make_i64_tensor(&[3, 3, 3], (1..=27).collect::<Vec<i64>>());
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params(&[1, 1, 0], &[2, 3, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2]);
    // Indices: [1,1,0]=13, [1,1,1]=14, [1,2,0]=16, [1,2,1]=17
    assert_eq!(extract_i64_vec(&result), vec![13, 14, 16, 17]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_slice_f64() {
    let input = make_f64_tensor(&[5], vec![1.1, 2.2, 3.3, 4.4, 5.5]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[1], &[4])).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![3]);
    assert!((vals[0] - 2.2).abs() < 1e-10);
    assert!((vals[1] - 3.3).abs() < 1e-10);
    assert!((vals[2] - 4.4).abs() < 1e-10);
}

#[test]
fn oracle_slice_f64_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 1], &[2, 3])).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert!((vals[0] - 2.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 5.0).abs() < 1e-10);
    assert!((vals[3] - 6.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_slice_with_negatives() {
    let input = make_i64_tensor(&[5], vec![-3, -1, 0, 2, 5]);
    let result = eval_primitive(Primitive::Slice, &[input], &slice_params(&[1], &[4])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, 0, 2]);
}

#[test]
fn oracle_slice_non_contiguous_result() {
    // Slice that can't use contiguous fast path
    let input = make_i64_tensor(&[3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let result =
        eval_primitive(Primitive::Slice, &[input], &slice_params(&[0, 1], &[3, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 3, 6, 7, 10, 11]);
}

#[test]
fn oracle_slice_stride_partial_range() {
    // Stride doesn't evenly divide the range
    let input = make_i64_tensor(&[7], vec![0, 1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[0], &[7], &[3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 3, 6]);
}

#[test]
fn oracle_slice_stride_single_result() {
    // Stride larger than range produces single element
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 3, 4]);
    let result = eval_primitive(
        Primitive::Slice,
        &[input],
        &slice_params_with_strides(&[1], &[4], &[10]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![1]);
}
