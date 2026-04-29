//! Oracle tests for Transpose primitive.
//!
//! Tests against expected behavior matching JAX/lax.transpose:
//! - permutation: axis ordering (if absent, reverses all axes)
//! - Preserves element count, reorders data layout

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

fn transpose_params(permutation: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "permutation".to_string(),
        permutation
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== 2D Tests ========================

#[test]
fn oracle_transpose_2d_default() {
    // JAX: lax.transpose(jnp.array([[1, 2, 3], [4, 5, 6]])) => [[1, 4], [2, 5], [3, 6]]
    // Default permutation reverses axes: (1, 0)
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn oracle_transpose_2d_explicit() {
    // JAX: lax.transpose(jnp.array([[1, 2, 3], [4, 5, 6]]), (1, 0))
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[1, 0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn oracle_transpose_2d_identity() {
    // JAX: lax.transpose(jnp.array([[1, 2, 3], [4, 5, 6]]), (0, 1)) => same
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_transpose_2d_square() {
    // JAX: lax.transpose(jnp.array([[1, 2], [3, 4]]))
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 2, 4]);
}

#[test]
fn oracle_transpose_2d_wide() {
    // Wide matrix [1, 4] -> [4, 1]
    let input = make_i64_tensor(&[1, 4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

#[test]
fn oracle_transpose_2d_tall() {
    // Tall matrix [4, 1] -> [1, 4]
    let input = make_i64_tensor(&[4, 1], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_transpose_3d_default() {
    // JAX: lax.transpose(x.shape=(2,3,4)) => shape (4,3,2)
    // Default reverses to (2, 1, 0)
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 3, 2]);
}

#[test]
fn oracle_transpose_3d_swap_first_two() {
    // Permutation (1, 0, 2): swap first two axes
    // Shape [2, 3, 4] -> [3, 2, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[1, 0, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2, 4]);
}

#[test]
fn oracle_transpose_3d_swap_last_two() {
    // Permutation (0, 2, 1): swap last two axes
    // Shape [2, 3, 4] -> [2, 4, 3]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[0, 2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4, 3]);
}

#[test]
fn oracle_transpose_3d_rotate_left() {
    // Permutation (1, 2, 0): rotate axes left
    // Shape [2, 3, 4] -> [3, 4, 2]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[1, 2, 0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4, 2]);
}

#[test]
fn oracle_transpose_3d_rotate_right() {
    // Permutation (2, 0, 1): rotate axes right
    // Shape [2, 3, 4] -> [4, 2, 3]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[2, 0, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2, 3]);
}

#[test]
fn oracle_transpose_3d_identity() {
    // Permutation (0, 1, 2): identity
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[0, 1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    assert_eq!(extract_i64_vec(&result), (1..=24).collect::<Vec<_>>());
}

#[test]
fn oracle_transpose_3d_small() {
    // Small 3D tensor [2, 2, 2] with (2, 1, 0) permutation
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[2, 1, 0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    // Original: [[[1,2],[3,4]], [[5,6],[7,8]]]
    // Transposed (2,1,0): [[[1,5],[3,7]], [[2,6],[4,8]]]
    assert_eq!(extract_i64_vec(&result), vec![1, 5, 3, 7, 2, 6, 4, 8]);
}

// ======================== 4D Tests ========================

#[test]
fn oracle_transpose_4d_reverse() {
    // Shape [2, 3, 4, 5] -> [5, 4, 3, 2] with default permutation
    let input = make_i64_tensor(&[2, 3, 4, 5], (1..=120).collect());
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 4, 3, 2]);
}

#[test]
fn oracle_transpose_4d_swap_middle() {
    // Permutation (0, 2, 1, 3): swap middle two axes
    // Shape [2, 3, 4, 5] -> [2, 4, 3, 5]
    let input = make_i64_tensor(&[2, 3, 4, 5], (1..=120).collect());
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[0, 2, 1, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4, 3, 5]);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_transpose_1d_default() {
    // 1D transpose is identity
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_transpose_1d_explicit() {
    // Explicit identity permutation
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_transpose_scalar() {
    // Scalar transpose is identity
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 42),
        _ => panic!("expected scalar"),
    }
}

// ======================== Float Tests ========================

#[test]
fn oracle_transpose_f64_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[1] - 4.4).abs() < 1e-10);
    assert!((vals[2] - 2.2).abs() < 1e-10);
    assert!((vals[3] - 5.5).abs() < 1e-10);
}

#[test]
fn oracle_transpose_f64_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

// ======================== Double Transpose Tests ========================

#[test]
fn oracle_transpose_double_is_identity() {
    // Transpose twice with same permutation gives original
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result1 = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::Transpose, &[result1], &no_params()).unwrap();
    assert_eq!(extract_shape(&result2), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result2), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_transpose_inverse_permutation() {
    // Applying inverse permutation gives original
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result1 =
        eval_primitive(Primitive::Transpose, &[input], &transpose_params(&[1, 2, 0])).unwrap();
    // Inverse of (1, 2, 0) is (2, 0, 1)
    let result2 =
        eval_primitive(Primitive::Transpose, &[result1], &transpose_params(&[2, 0, 1])).unwrap();
    assert_eq!(extract_shape(&result2), vec![2, 3, 4]);
    assert_eq!(extract_i64_vec(&result2), (1..=24).collect::<Vec<_>>());
}

// ======================== Edge Cases ========================

#[test]
fn oracle_transpose_single_element() {
    let input = make_i64_tensor(&[1, 1], vec![42]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_transpose_unit_dims() {
    // Tensor with unit dimensions
    let input = make_i64_tensor(&[1, 3, 1], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
}

#[test]
fn oracle_transpose_with_negatives() {
    let input = make_i64_tensor(&[2, 2], vec![-3, -1, 2, 5]);
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-3, 2, -1, 5]);
}

#[test]
fn oracle_transpose_large_2d() {
    // Larger 2D transpose
    let input = make_i64_tensor(&[4, 5], (1..=20).collect());
    let result = eval_primitive(Primitive::Transpose, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 4]);
    let vals = extract_i64_vec(&result);
    // First row of transposed should be [1, 6, 11, 16] (first column of original)
    assert_eq!(vals[0], 1);
    assert_eq!(vals[1], 6);
    assert_eq!(vals[2], 11);
    assert_eq!(vals[3], 16);
}
