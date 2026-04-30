//! Oracle tests for ReduceWindow primitive.
//!
//! Tests against expected behavior for windowed reduction (pooling):
//! - reduce_op: "sum", "max", "min"
//! - window_dimensions: comma-separated sizes per dim
//! - window_strides: comma-separated strides (default: 1)
//! - padding: "VALID", "SAME", or "SAME_LOWER"

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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

fn make_f32_tensor(shape: &[u32], data: Vec<f32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f32).collect(),
        )
        .unwrap(),
    )
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

fn make_bool_tensor(shape: &[u32], data: Vec<bool>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn window_params(
    reduce_op: &str,
    window_dims: &str,
    strides: &str,
    padding: &str,
) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("reduce_op".to_string(), reduce_op.to_string());
    p.insert("window_dimensions".to_string(), window_dims.to_string());
    p.insert("window_strides".to_string(), strides.to_string());
    p.insert("padding".to_string(), padding.to_string());
    p
}

fn sum_window(window_dims: &str, strides: &str, padding: &str) -> BTreeMap<String, String> {
    window_params("sum", window_dims, strides, padding)
}

fn max_window(window_dims: &str, strides: &str, padding: &str) -> BTreeMap<String, String> {
    window_params("max", window_dims, strides, padding)
}

fn min_window(window_dims: &str, strides: &str, padding: &str) -> BTreeMap<String, String> {
    window_params("min", window_dims, strides, padding)
}

// ======================== 1D Sum Pooling Tests ========================

#[test]
fn oracle_reduce_window_1d_sum_basic() {
    // input=[1,2,3,4,5], window=2, stride=1, valid
    // output = [1+2, 2+3, 3+4, 4+5] = [3, 5, 7, 9]
    let input = make_f64_tensor(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 5.0).abs() < 1e-10);
    assert!((vals[2] - 7.0).abs() < 1e-10);
    assert!((vals[3] - 9.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_1d_sum_stride2() {
    // input=[1,2,3,4,5,6], window=2, stride=2, valid
    // output = [1+2, 3+4, 5+6] = [3, 7, 11]
    let input = make_f64_tensor(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "2", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 7.0).abs() < 1e-10);
    assert!((vals[2] - 11.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_1d_sum_window3() {
    // input=[1,2,3,4], window=3, stride=1, valid
    // output = [1+2+3, 2+3+4] = [6, 9]
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("3", "1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 6.0).abs() < 1e-10);
    assert!((vals[1] - 9.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_1d_f32_preserves_literal_dtype() {
    let input = make_f32_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "VALID"),
    )
    .unwrap();

    if let Value::Tensor(t) = &result {
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(extract_shape(&result), vec![3]);
        assert!(
            t.elements
                .iter()
                .all(|literal| matches!(literal, Literal::F32Bits(_))),
            "reduce_window F32 output should store F32 literals: {:?}",
            t.elements
        );
    } else {
        assert!(matches!(result, Value::Tensor(_)), "expected tensor");
    }
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 7.0]);
}

#[test]
fn oracle_reduce_window_1d_i64_preserves_literal_dtype() {
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "VALID"),
    )
    .unwrap();

    if let Value::Tensor(t) = &result {
        assert_eq!(t.dtype, DType::I64);
        assert_eq!(extract_shape(&result), vec![3]);
        assert_eq!(
            t.elements,
            vec![Literal::I64(3), Literal::I64(5), Literal::I64(7)]
        );
    } else {
        assert!(matches!(result, Value::Tensor(_)), "expected tensor");
    }
}

#[test]
fn oracle_reduce_window_1d_bool_sum_preserves_literal_dtype() {
    let input = make_bool_tensor(&[4], vec![false, true, false, false]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "VALID"),
    )
    .unwrap();

    if let Value::Tensor(t) = &result {
        assert_eq!(t.dtype, DType::Bool);
        assert_eq!(extract_shape(&result), vec![3]);
        assert_eq!(
            t.elements,
            vec![
                Literal::Bool(true),
                Literal::Bool(true),
                Literal::Bool(false),
            ]
        );
    } else {
        assert!(matches!(result, Value::Tensor(_)), "expected tensor");
    }
}

// ======================== 1D Max Pooling Tests ========================

#[test]
fn oracle_reduce_window_1d_max_basic() {
    // input=[1,3,2,5,4], window=2, stride=1, valid
    // output = [max(1,3), max(3,2), max(2,5), max(5,4)] = [3, 3, 5, 5]
    let input = make_f64_tensor(&[5], vec![1.0, 3.0, 2.0, 5.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &max_window("2", "1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 5.0).abs() < 1e-10);
    assert!((vals[3] - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_1d_max_stride2() {
    // input=[1,3,2,5,4,6], window=2, stride=2, valid
    // output = [max(1,3), max(2,5), max(4,6)] = [3, 5, 6]
    let input = make_f64_tensor(&[6], vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &max_window("2", "2", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 5.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

// ======================== 1D Min Pooling Tests ========================

#[test]
fn oracle_reduce_window_1d_min_basic() {
    // input=[3,1,4,1,5], window=2, stride=1, valid
    // output = [min(3,1), min(1,4), min(4,1), min(1,5)] = [1, 1, 1, 1]
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &min_window("2", "1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|v| (v - 1.0).abs() < 1e-10));
}

#[test]
fn oracle_reduce_window_1d_min_stride2() {
    // input=[5,2,8,3,1,7], window=2, stride=2, valid
    // output = [min(5,2), min(8,3), min(1,7)] = [2, 3, 1]
    let input = make_f64_tensor(&[6], vec![5.0, 2.0, 8.0, 3.0, 1.0, 7.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &min_window("2", "2", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 1.0).abs() < 1e-10);
}

// ======================== 2D Sum Pooling Tests ========================

#[test]
fn oracle_reduce_window_2d_sum_basic() {
    // 3x3 input, 2x2 window, stride 1
    // Input:
    // 1 2 3
    // 4 5 6
    // 7 8 9
    // Output 2x2:
    // [1+2+4+5, 2+3+5+6] = [12, 16]
    // [4+5+7+8, 5+6+8+9] = [24, 28]
    let input = make_f64_tensor(&[3, 3], (1..=9).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2,2", "1,1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 12.0).abs() < 1e-10);
    assert!((vals[1] - 16.0).abs() < 1e-10);
    assert!((vals[2] - 24.0).abs() < 1e-10);
    assert!((vals[3] - 28.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_2d_sum_stride2() {
    // 4x4 input, 2x2 window, stride 2 -> 2x2 output
    let input = make_f64_tensor(&[4, 4], (1..=16).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2,2", "2,2", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    // Top-left 2x2: 1+2+5+6 = 14
    // Top-right 2x2: 3+4+7+8 = 22
    // Bottom-left 2x2: 9+10+13+14 = 46
    // Bottom-right 2x2: 11+12+15+16 = 54
    assert!((vals[0] - 14.0).abs() < 1e-10);
    assert!((vals[1] - 22.0).abs() < 1e-10);
    assert!((vals[2] - 46.0).abs() < 1e-10);
    assert!((vals[3] - 54.0).abs() < 1e-10);
}

// ======================== 2D Max Pooling Tests ========================

#[test]
fn oracle_reduce_window_2d_max_basic() {
    // 3x3 input, 2x2 window, stride 1
    let input = make_f64_tensor(&[3, 3], (1..=9).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &max_window("2,2", "1,1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    // max(1,2,4,5)=5, max(2,3,5,6)=6, max(4,5,7,8)=8, max(5,6,8,9)=9
    assert!((vals[0] - 5.0).abs() < 1e-10);
    assert!((vals[1] - 6.0).abs() < 1e-10);
    assert!((vals[2] - 8.0).abs() < 1e-10);
    assert!((vals[3] - 9.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_2d_max_stride2() {
    // 4x4 input, 2x2 window, stride 2 -> 2x2 output
    let input = make_f64_tensor(&[4, 4], (1..=16).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &max_window("2,2", "2,2", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    // max of each 2x2 quadrant: 6, 8, 14, 16
    assert!((vals[0] - 6.0).abs() < 1e-10);
    assert!((vals[1] - 8.0).abs() < 1e-10);
    assert!((vals[2] - 14.0).abs() < 1e-10);
    assert!((vals[3] - 16.0).abs() < 1e-10);
}

// ======================== 2D Min Pooling Tests ========================

#[test]
fn oracle_reduce_window_2d_min_basic() {
    // 3x3 input, 2x2 window, stride 1
    let input = make_f64_tensor(&[3, 3], (1..=9).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &min_window("2,2", "1,1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    // min(1,2,4,5)=1, min(2,3,5,6)=2, min(4,5,7,8)=4, min(5,6,8,9)=5
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 4.0).abs() < 1e-10);
    assert!((vals[3] - 5.0).abs() < 1e-10);
}

// ======================== Same Padding Tests ========================

#[test]
fn oracle_reduce_window_1d_same_padding() {
    // With same padding, output length equals input length
    let input = make_f64_tensor(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("3", "1", "same"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
}

#[test]
fn oracle_reduce_window_2d_same_padding() {
    // With same padding, output shape equals input shape
    let input = make_f64_tensor(&[4, 4], (1..=16).map(|i| i as f64).collect());
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("3,3", "1,1", "same"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4, 4]);
}

#[test]
fn oracle_reduce_window_uppercase_same_padding() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "SAME"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![3.0, 5.0, 7.0, 4.0]);
}

#[test]
fn oracle_reduce_window_same_lower_padding() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "SAME_LOWER"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn oracle_reduce_window_unknown_padding_rejected() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let err = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "MIRROR"),
    )
    .expect_err("unknown padding should fail closed");
    assert!(
        err.to_string()
            .contains("unsupported reduce_window padding mode"),
        "unexpected error: {err}"
    );
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reduce_window_window_equals_input() {
    // Window size = input size -> single output element
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("4", "1", "valid"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 10.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_negative_values() {
    let input = make_f64_tensor(&[4], vec![-2.0, -1.0, 1.0, 2.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("2", "1", "valid"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-3.0)).abs() < 1e-10); // -2 + -1
    assert!((vals[1] - 0.0).abs() < 1e-10); // -1 + 1
    assert!((vals[2] - 3.0).abs() < 1e-10); // 1 + 2
}

#[test]
fn oracle_reduce_window_max_negative() {
    let input = make_f64_tensor(&[4], vec![-5.0, -2.0, -3.0, -1.0]);
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &max_window("2", "1", "valid"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-2.0)).abs() < 1e-10);
    assert!((vals[1] - (-2.0)).abs() < 1e-10);
    assert!((vals[2] - (-1.0)).abs() < 1e-10);
}

#[test]
fn oracle_reduce_window_scalar_passthrough() {
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(
        Primitive::ReduceWindow,
        &[input],
        &sum_window("1", "1", "valid"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 42.0).abs() < 1e-10);
}
