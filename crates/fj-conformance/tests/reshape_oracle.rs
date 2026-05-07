//! Oracle tests for Reshape primitive.
//!
//! Tests against expected behavior matching JAX/lax.reshape:
//! - new_shape: target shape specification
//! - Supports -1 for dimension inference (exactly one allowed)
//! - Element count must be preserved

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

fn reshape_params(new_shape: &[i64]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "new_shape".to_string(),
        new_shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== 1D Reshape Tests ========================

#[test]
fn oracle_reshape_1d_to_2d() {
    // JAX: lax.reshape(jnp.array([1, 2, 3, 4, 5, 6]), (2, 3))
    // => [[1, 2, 3], [4, 5, 6]]
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_reshape_1d_to_3d() {
    // JAX: lax.reshape(jnp.array([1..12]), (2, 2, 3))
    let input = make_i64_tensor(&[12], (1..=12).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, 2, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 3]);
    assert_eq!(extract_i64_vec(&result), (1..=12).collect::<Vec<_>>());
}

#[test]
fn oracle_reshape_2d_to_1d() {
    // JAX: lax.reshape(jnp.array([[1, 2, 3], [4, 5, 6]]), (6,))
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[6])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_reshape_2d_to_2d() {
    // JAX: lax.reshape(jnp.array([[1, 2, 3], [4, 5, 6]]), (3, 2))
    // Row-major: [1,2,3,4,5,6] -> [[1,2],[3,4],[5,6]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[3, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_reshape_2d_to_3d() {
    // JAX: lax.reshape(jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]), (2, 2, 2))
    let input = make_i64_tensor(&[2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, 2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn oracle_reshape_3d_to_1d() {
    // Flatten 3D to 1D
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[8])).unwrap();
    assert_eq!(extract_shape(&result), vec![8]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn oracle_reshape_3d_to_2d() {
    // JAX: lax.reshape(x.shape=(2,2,3), (4, 3))
    let input = make_i64_tensor(&[2, 2, 3], (1..=12).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[4, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 3]);
    assert_eq!(extract_i64_vec(&result), (1..=12).collect::<Vec<_>>());
}

// ======================== Inference (-1) Tests ========================

#[test]
fn oracle_reshape_infer_first_dim() {
    // JAX: lax.reshape(jnp.array([1, 2, 3, 4, 5, 6]), (-1, 3))
    // => shape (2, 3) inferred from 6/3=2
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[-1, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_reshape_infer_last_dim() {
    // JAX: lax.reshape(jnp.array([1, 2, 3, 4, 5, 6]), (2, -1))
    // => shape (2, 3) inferred from 6/2=3
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, -1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_reshape_infer_middle_dim() {
    // JAX: lax.reshape(jnp.array(range(24)), (2, -1, 3))
    // => shape (2, 4, 3) inferred from 24/(2*3)=4
    let input = make_i64_tensor(&[24], (0..24).collect());
    let result =
        eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, -1, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4, 3]);
    assert_eq!(extract_i64_vec(&result), (0..24).collect::<Vec<_>>());
}

#[test]
fn oracle_reshape_infer_to_1d() {
    // JAX: lax.reshape(jnp.array([[1, 2], [3, 4]]), (-1,))
    // => flatten to (4,)
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[-1])).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

#[test]
fn oracle_reshape_infer_from_3d() {
    // 3D to 2D with inference
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[6, -1])).unwrap();
    assert_eq!(extract_shape(&result), vec![6, 4]);
    assert_eq!(extract_i64_vec(&result), (1..=24).collect::<Vec<_>>());
}

// ======================== Same Shape Tests ========================

#[test]
fn oracle_reshape_identity_1d() {
    // No-op reshape
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_reshape_identity_2d() {
    // No-op reshape
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

// ======================== Adding/Removing Dimensions ========================

#[test]
fn oracle_reshape_add_unit_dims() {
    // JAX: lax.reshape(jnp.array([1, 2, 3]), (1, 3, 1))
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1, 3, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_reshape_remove_unit_dims() {
    // JAX: lax.reshape(jnp.array([[[1], [2], [3]]]), (3,))
    let input = make_i64_tensor(&[1, 3, 1], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_reshape_to_higher_rank() {
    // 2D -> 4D with unit dimensions
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1, 2, 3, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_reshape_f64_2d_to_3d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[3, 2])).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[5] - 6.6).abs() < 1e-10);
}

#[test]
fn oracle_reshape_f64_with_inference() {
    let input = make_f64_tensor(&[12], (1..=12).map(|x| x as f64).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[3, -1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reshape_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1, 1, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_reshape_single_element_infer() {
    let input = make_i64_tensor(&[1, 1], vec![42]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[-1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_reshape_with_negatives() {
    let input = make_i64_tensor(&[4], vec![-3, -1, 2, 5]);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![-3, -1, 2, 5]);
}

#[test]
fn oracle_reshape_large_tensor() {
    let input = make_i64_tensor(&[100], (0..100).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[10, 10])).unwrap();
    assert_eq!(extract_shape(&result), vec![10, 10]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[99], 99);
}

#[test]
fn oracle_reshape_to_single_row() {
    // Reshape 2D matrix to single row
    let input = make_i64_tensor(&[3, 4], (1..=12).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1, 12])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 12]);
    assert_eq!(extract_i64_vec(&result), (1..=12).collect::<Vec<_>>());
}

#[test]
fn oracle_reshape_to_single_col() {
    // Reshape 2D matrix to single column
    let input = make_i64_tensor(&[3, 4], (1..=12).collect());
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[12, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![12, 1]);
    assert_eq!(extract_i64_vec(&result), (1..=12).collect::<Vec<_>>());
}

// ======================== Scalar to Tensor ========================

#[test]
fn oracle_reshape_scalar_to_1d() {
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_reshape_scalar_to_3d() {
    let input = Value::scalar_i64(99);
    let result = eval_primitive(Primitive::Reshape, &[input], &reshape_params(&[1, 1, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![99]);
}

// ======================== Metamorphic Properties ========================

#[test]
fn metamorphic_reshape_roundtrip_2d_to_1d_back() {
    // Metamorphic: reshape(reshape(x, s1), original_shape) = x
    let data: Vec<i64> = (1..=12).collect();
    let input = make_i64_tensor(&[3, 4], data.clone());

    let flattened =
        eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &reshape_params(&[12])).unwrap();
    let restored =
        eval_primitive(Primitive::Reshape, &[flattened], &reshape_params(&[3, 4])).unwrap();

    assert_eq!(extract_shape(&restored), vec![3, 4]);
    assert_eq!(extract_i64_vec(&restored), data);
}

#[test]
fn metamorphic_reshape_roundtrip_3d_to_2d_back() {
    // Reshape 3D -> 2D -> 3D preserves data
    let data: Vec<i64> = (1..=24).collect();
    let input = make_i64_tensor(&[2, 3, 4], data.clone());

    let reshaped =
        eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &reshape_params(&[6, 4])).unwrap();
    let restored =
        eval_primitive(Primitive::Reshape, &[reshaped], &reshape_params(&[2, 3, 4])).unwrap();

    assert_eq!(extract_shape(&restored), vec![2, 3, 4]);
    assert_eq!(extract_i64_vec(&restored), data);
}

#[test]
fn metamorphic_reshape_any_shape_same_elements() {
    // Metamorphic: any reshape to any compatible shape preserves element order
    let data: Vec<i64> = (1..=24).collect();
    let shapes = vec![
        vec![24],
        vec![1, 24],
        vec![24, 1],
        vec![2, 12],
        vec![3, 8],
        vec![4, 6],
        vec![6, 4],
        vec![2, 3, 4],
        vec![2, 4, 3],
        vec![3, 2, 4],
        vec![4, 2, 3],
    ];

    let input = make_i64_tensor(&[24], data.clone());

    for shape in shapes {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let result =
            eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &reshape_params(&shape_i64))
                .unwrap();
        assert_eq!(
            extract_i64_vec(&result),
            data,
            "reshape to {:?} should preserve elements",
            shape
        );
    }
}

#[test]
fn metamorphic_reshape_flatten_and_restore_f64() {
    // Roundtrip with f64 data
    let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let input = make_f64_tensor(&[4, 5], data.clone());

    let flattened =
        eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &reshape_params(&[20])).unwrap();
    let restored =
        eval_primitive(Primitive::Reshape, &[flattened], &reshape_params(&[4, 5])).unwrap();

    let restored_vals = extract_f64_vec(&restored);
    for (a, b) in data.iter().zip(restored_vals.iter()) {
        assert!((a - b).abs() < 1e-15, "f64 reshape roundtrip should be exact");
    }
}

#[test]
fn metamorphic_reshape_same_shape_is_identity() {
    // Reshaping to the same shape should return identical tensor
    let data: Vec<i64> = (1..=12).collect();
    let input = make_i64_tensor(&[3, 4], data.clone());

    let same =
        eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &reshape_params(&[3, 4])).unwrap();

    assert_eq!(extract_shape(&same), vec![3, 4]);
    assert_eq!(extract_i64_vec(&same), data);
}
