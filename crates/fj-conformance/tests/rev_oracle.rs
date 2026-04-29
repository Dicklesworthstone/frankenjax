//! Oracle tests for Rev primitive.
//!
//! Tests against expected behavior matching JAX/lax.rev:
//! - Reverses elements along specified axes
//! - axes param specifies which dimensions to reverse

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

fn axes_params(axes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "axes".to_string(),
        axes.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== 1D Tests ========================

#[test]
fn oracle_rev_1d() {
    // JAX: lax.rev(jnp.array([1, 2, 3, 4, 5]), (0,)) => [5, 4, 3, 2, 1]
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![5, 4, 3, 2, 1]);
}

#[test]
fn oracle_rev_1d_single() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_rev_1d_two_elements() {
    let input = make_i64_tensor(&[2], vec![1, 2]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![2, 1]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_rev_2d_axis0() {
    // JAX: lax.rev(jnp.array([[1,2,3],[4,5,6]]), (0,))
    // => [[4, 5, 6], [1, 2, 3]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![4, 5, 6, 1, 2, 3]);
}

#[test]
fn oracle_rev_2d_axis1() {
    // JAX: lax.rev(jnp.array([[1,2,3],[4,5,6]]), (1,))
    // => [[3, 2, 1], [6, 5, 4]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![3, 2, 1, 6, 5, 4]);
}

#[test]
fn oracle_rev_2d_both_axes() {
    // JAX: lax.rev(jnp.array([[1,2,3],[4,5,6]]), (0,1))
    // => [[6, 5, 4], [3, 2, 1]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 5, 4, 3, 2, 1]);
}

#[test]
fn oracle_rev_2d_square() {
    // JAX: lax.rev(jnp.array([[1,2],[3,4]]), (0,))
    // => [[3, 4], [1, 2]]
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 4, 1, 2]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_rev_3d_axis0() {
    // Shape [2, 2, 2], reverse along axis 0
    // [[[ 1, 2], [ 3, 4]], [[ 5, 6], [ 7, 8]]] ->
    // [[[ 5, 6], [ 7, 8]], [[ 1, 2], [ 3, 4]]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![5, 6, 7, 8, 1, 2, 3, 4]);
}

#[test]
fn oracle_rev_3d_axis2() {
    // Shape [2, 2, 2], reverse along axis 2 (innermost)
    // [[[ 1, 2], [ 3, 4]], [[ 5, 6], [ 7, 8]]] ->
    // [[[ 2, 1], [ 4, 3]], [[ 6, 5], [ 8, 7]]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 1, 4, 3, 6, 5, 8, 7]);
}

#[test]
fn oracle_rev_3d_all_axes() {
    // Reverse along all axes is equivalent to full reversal of data
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0, 1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![8, 7, 6, 5, 4, 3, 2, 1]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_rev_single_axis_twice() {
    // Reversing same axis twice gives back original
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(
        Primitive::Rev,
        std::slice::from_ref(&input),
        &axes_params(&[0]),
    )
    .unwrap();
    let result2 = eval_primitive(Primitive::Rev, &[result], &axes_params(&[0])).unwrap();
    assert_eq!(extract_i64_vec(&result2), vec![1, 2, 3]);
}

#[test]
fn oracle_rev_f64() {
    let input = make_f64_tensor(&[4], vec![1.5, 2.5, 3.5, 4.5]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.5).abs() < 1e-10);
    assert!((vals[1] - 3.5).abs() < 1e-10);
    assert!((vals[2] - 2.5).abs() < 1e-10);
    assert!((vals[3] - 1.5).abs() < 1e-10);
}

#[test]
fn oracle_rev_with_negatives() {
    let input = make_i64_tensor(&[4], vec![-3, -1, 2, 5]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 2, -1, -3]);
}

#[test]
fn oracle_rev_empty_tensor() {
    // Empty tensor reversal
    let input = make_i64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Rev, &[input], &axes_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}
