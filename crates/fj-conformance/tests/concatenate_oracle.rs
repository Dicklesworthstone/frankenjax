//! Oracle tests for Concatenate primitive.
//!
//! Tests against expected behavior matching JAX/lax.concatenate:
//! - dimension: axis to concatenate along (default 0)
//! - All inputs must have same rank and matching dims on non-concat axes

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

fn concat_params(dimension: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("dimension".to_string(), dimension.to_string());
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== 1D Tests ========================

#[test]
fn oracle_concat_1d_two_arrays() {
    // JAX: lax.concatenate([[1,2,3], [4,5,6]], dimension=0) => [1,2,3,4,5,6]
    let a = make_i64_tensor(&[3], vec![1, 2, 3]);
    let b = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_1d_three_arrays() {
    let a = make_i64_tensor(&[2], vec![1, 2]);
    let b = make_i64_tensor(&[3], vec![3, 4, 5]);
    let c = make_i64_tensor(&[1], vec![6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_1d_different_sizes() {
    let a = make_i64_tensor(&[1], vec![1]);
    let b = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_concat_1d_single_array() {
    let a = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Concatenate, &[a], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

// ======================== 2D Tests - Axis 0 ========================

#[test]
fn oracle_concat_2d_axis0() {
    // JAX: lax.concatenate([[[1,2],[3,4]], [[5,6]]], dimension=0)
    // => [[1,2],[3,4],[5,6]]
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[1, 2], vec![5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_2d_axis0_same_size() {
    let a = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 3], vec![7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 3]);
    assert_eq!(
        extract_i64_vec(&result),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    );
}

#[test]
fn oracle_concat_2d_axis0_default() {
    // Default dimension is 0
    let a = make_i64_tensor(&[1, 3], vec![1, 2, 3]);
    let b = make_i64_tensor(&[2, 3], vec![4, 5, 6, 7, 8, 9]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

// ======================== 2D Tests - Axis 1 ========================

#[test]
fn oracle_concat_2d_axis1() {
    // JAX: lax.concatenate([[[1,2],[3,4]], [[5],[6]]], dimension=1)
    // => [[1,2,5],[3,4,6]]
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[2, 1], vec![5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 5, 3, 4, 6]);
}

#[test]
fn oracle_concat_2d_axis1_same_size() {
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[2, 2], vec![5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 5, 6, 3, 4, 7, 8]);
}

#[test]
fn oracle_concat_2d_axis1_three_arrays() {
    let a = make_i64_tensor(&[2, 1], vec![1, 4]);
    let b = make_i64_tensor(&[2, 2], vec![2, 3, 5, 6]);
    let c = make_i64_tensor(&[2, 1], vec![7, 8]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 7, 4, 5, 6, 8]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_concat_3d_axis0() {
    // Shape [1,2,3] + [2,2,3] -> [3,2,3]
    let a = make_i64_tensor(&[1, 2, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 2, 3], (7..=18).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2, 3]);
    assert_eq!(extract_i64_vec(&result).len(), 18);
}

#[test]
fn oracle_concat_3d_axis1() {
    // Shape [2,1,3] + [2,2,3] -> [2,3,3]
    let a = make_i64_tensor(&[2, 1, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 2, 3], (7..=18).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 3]);
}

#[test]
fn oracle_concat_3d_axis2() {
    // Shape [2,2,2] + [2,2,3] -> [2,2,5]
    let a = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let b = make_i64_tensor(&[2, 2, 3], (9..=20).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 5]);
    assert_eq!(extract_i64_vec(&result).len(), 20);
}

// ======================== Float Tests ========================

#[test]
fn oracle_concat_f64_1d() {
    let a = make_f64_tensor(&[2], vec![1.1, 2.2]);
    let b = make_f64_tensor(&[2], vec![3.3, 4.4]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![4]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[3] - 4.4).abs() < 1e-10);
}

#[test]
fn oracle_concat_f64_2d() {
    let a = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[1, 3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_concat_with_negatives() {
    let a = make_i64_tensor(&[2], vec![-3, -1]);
    let b = make_i64_tensor(&[2], vec![0, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-3, -1, 0, 5]);
}

#[test]
fn oracle_concat_empty_and_nonempty() {
    let a = make_i64_tensor(&[0], vec![]);
    let b = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_concat_large() {
    let a = make_i64_tensor(&[10], (1..=10).collect());
    let b = make_i64_tensor(&[10], (11..=20).collect());
    let c = make_i64_tensor(&[10], (21..=30).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![30]);
    assert_eq!(extract_i64_vec(&result), (1..=30).collect::<Vec<_>>());
}

#[test]
fn oracle_concat_2d_single_row() {
    // Single row matrices
    let a = make_i64_tensor(&[1, 2], vec![1, 2]);
    let b = make_i64_tensor(&[1, 3], vec![3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_concat_2d_single_col() {
    // Single column matrices
    let a = make_i64_tensor(&[2, 1], vec![1, 2]);
    let b = make_i64_tensor(&[3, 1], vec![3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

// JAX `lax.concatenate` rejects mixed-dtype operands (no implicit
// promotion). Before the guard, fj-lax silently built an output tensor
// whose declared dtype was `operands[0].dtype` but whose elements were
// a mix of literal kinds, which violated the dtype/element invariant.
#[test]
fn oracle_concat_rejects_dtype_mismatch() {
    let i64_input = make_i64_tensor(&[2], vec![1, 2]);
    let f64_input = make_f64_tensor(&[2], vec![3.0, 4.0]);
    let err = eval_primitive(
        Primitive::Concatenate,
        &[i64_input, f64_input],
        &concat_params(0),
    )
    .expect_err("mixed dtype concatenate should error");
    let msg = format!("{err}");
    assert!(
        msg.contains("dtype") && msg.contains("does not match"),
        "expected dtype mismatch error, got: {msg}"
    );
}

// Property sweep: for every primary dtype, asserting same-dtype concat
// returns a tensor whose declared dtype matches AND
// `validate_dtype_consistency` passes. Pins 98c2df7 (the uniform-dtype
// guard) doesn't accidentally widen any single-dtype variant.
#[test]
fn property_concat_same_dtype_preserves_dtype() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::I64 | DType::I32 => Literal::I64(v as i64),
            DType::U32 => Literal::U32(v as u32),
            DType::U64 => Literal::U64(v as u64),
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            DType::Bool => Literal::Bool(v != 0.0),
            DType::Complex64 => Literal::from_complex64(v as f32, 0.0),
            DType::Complex128 => Literal::from_complex128(v, 0.0),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![values.len() as u32],
                },
                values.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let a_data = [1.0_f64, 2.0];
    let b_data = [3.0_f64, 4.0];
    for dtype in [
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Bool,
        DType::Complex64,
        DType::Complex128,
    ] {
        let a = make_vec(dtype, &a_data);
        let b = make_vec(dtype, &b_data);
        let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0))
            .unwrap_or_else(|e| panic!("concat {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("concat {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "concat {dtype:?}: declared dtype");
        assert_eq!(t.shape.dims, vec![4]);
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("concat {dtype:?}: validate_dtype_consistency failed: {e}"));
    }
}
