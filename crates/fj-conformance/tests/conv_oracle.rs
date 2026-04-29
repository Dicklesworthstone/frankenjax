//! Oracle tests for Conv primitive.
//!
//! Tests against expected behavior for 1D and 2D convolution:
//! - lhs: input tensor [N, H, W, C_in] or [N, L, C_in]
//! - rhs: kernel [KH, KW, C_in, C_out] or [K, C_in, C_out]
//! - params: padding ("valid"/"same"), stride

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

fn conv_params(padding: &str, strides: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("padding".to_string(), padding.to_string());
    p.insert("strides".to_string(), strides.to_string());
    p
}

// ======================== 1D Convolution Tests ========================

#[test]
fn oracle_conv_1d_valid_basic() {
    // lhs=[1, 5, 1] (batch=1, length=5, channels=1)
    // rhs=[3, 1, 1] (kernel=3, c_in=1, c_out=1)
    // kernel = [1, 1, 1] -> moving sum of 3
    // input = [1, 2, 3, 4, 5]
    // output = [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    let lhs = make_f64_tensor(&[1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rhs = make_f64_tensor(&[3, 1, 1], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 6.0).abs() < 1e-10);
    assert!((vals[1] - 9.0).abs() < 1e-10);
    assert!((vals[2] - 12.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_1d_valid_stride2() {
    // lhs=[1, 6, 1], rhs=[2, 1, 1], stride=2
    // input = [1, 2, 3, 4, 5, 6]
    // kernel = [1, 1] -> sum of 2
    // positions: 0, 2, 4 -> [1+2, 3+4, 5+6] = [3, 7, 11]
    let lhs = make_f64_tensor(&[1, 6, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "2")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 7.0).abs() < 1e-10);
    assert!((vals[2] - 11.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_1d_same_padding() {
    // lhs=[1, 4, 1], rhs=[3, 1, 1], same padding
    // Output should have same length as input: 4
    let lhs = make_f64_tensor(&[1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[3, 1, 1], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4, 1]);
}

#[test]
fn oracle_conv_1d_weighted_kernel() {
    // lhs=[1, 4, 1], rhs=[2, 1, 1]
    // kernel = [1, 2] -> weighted sum
    // input = [1, 2, 3, 4]
    // output = [1*1+2*2, 1*2+2*3, 1*3+2*4] = [5, 8, 11]
    let lhs = make_f64_tensor(&[1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.0).abs() < 1e-10);
    assert!((vals[1] - 8.0).abs() < 1e-10);
    assert!((vals[2] - 11.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_1d_multi_channel_out() {
    // lhs=[1, 3, 1], rhs=[2, 1, 2] (2 output channels)
    // input = [1, 2, 3]
    // kernel ch0 = [1, 0], kernel ch1 = [0, 1]
    // output ch0 at pos 0 = 1*1+2*0 = 1, pos 1 = 2*1+3*0 = 2
    // output ch1 at pos 0 = 1*0+2*1 = 2, pos 1 = 2*0+3*1 = 3
    let lhs = make_f64_tensor(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = make_f64_tensor(&[2, 1, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2]);
    let vals = extract_f64_vec(&result);
    // [pos0_ch0, pos0_ch1, pos1_ch0, pos1_ch1] = [1, 2, 2, 3]
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 2.0).abs() < 1e-10);
    assert!((vals[3] - 3.0).abs() < 1e-10);
}

// ======================== 2D Convolution Tests ========================

#[test]
fn oracle_conv_2d_valid_basic() {
    // lhs=[1, 3, 3, 1], rhs=[2, 2, 1, 1]
    // 3x3 input, 2x2 kernel of ones
    // Input:
    // 1 2 3
    // 4 5 6
    // 7 8 9
    // Output 2x2: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
    let lhs = make_f64_tensor(&[1, 3, 3, 1], (1..=9).map(|i| i as f64).collect());
    let rhs = make_f64_tensor(&[2, 2, 1, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 12.0).abs() < 1e-10);
    assert!((vals[1] - 16.0).abs() < 1e-10);
    assert!((vals[2] - 24.0).abs() < 1e-10);
    assert!((vals[3] - 28.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_2d_same_padding() {
    // lhs=[1, 3, 3, 1], rhs=[3, 3, 1, 1], same padding
    // Output should have same spatial dims: 3x3
    let lhs = make_f64_tensor(&[1, 3, 3, 1], (1..=9).map(|i| i as f64).collect());
    let rhs = make_f64_tensor(&[3, 3, 1, 1], vec![1.0; 9]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 3, 1]);
    let vals = extract_f64_vec(&result);
    // Center element: sum of all 9 values = 45
    assert!((vals[4] - 45.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_2d_stride2() {
    // lhs=[1, 4, 4, 1], rhs=[2, 2, 1, 1], stride=2
    // 4x4 input, 2x2 kernel, stride 2 -> 2x2 output
    let lhs = make_f64_tensor(&[1, 4, 4, 1], (1..=16).map(|i| i as f64).collect());
    let rhs = make_f64_tensor(&[2, 2, 1, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "2")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 1]);
}

#[test]
fn oracle_conv_2d_identity_kernel() {
    // 1x1 kernel with value 1 should pass through values
    let lhs = make_f64_tensor(&[1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 1]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn oracle_conv_2d_scaling_kernel() {
    // 1x1 kernel with value 2 should scale values by 2
    let lhs = make_f64_tensor(&[1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![2.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn oracle_conv_2d_multi_channel() {
    // lhs=[1, 2, 2, 2] (2 input channels)
    // rhs=[1, 1, 2, 1] (pointwise, 2->1 channels)
    // kernel sums both channels
    let lhs = make_f64_tensor(
        &[1, 2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let rhs = make_f64_tensor(&[1, 1, 2, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 1]);
    let vals = extract_f64_vec(&result);
    // Each output = sum of both channels at that position
    // [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 7.0).abs() < 1e-10);
    assert!((vals[2] - 11.0).abs() < 1e-10);
    assert!((vals[3] - 15.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_2d_multi_out_channel() {
    // lhs=[1, 2, 2, 1], rhs=[1, 1, 1, 2] (2 output channels)
    // kernel ch0 = 1, ch1 = 2
    let lhs = make_f64_tensor(&[1, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[1, 1, 1, 2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 2]);
    let vals = extract_f64_vec(&result);
    // ch0 = input*1, ch1 = input*2
    assert!((vals[0] - 1.0).abs() < 1e-10); // pos(0,0) ch0
    assert!((vals[1] - 2.0).abs() < 1e-10); // pos(0,0) ch1
    assert!((vals[2] - 2.0).abs() < 1e-10); // pos(0,1) ch0
    assert!((vals[3] - 4.0).abs() < 1e-10); // pos(0,1) ch1
}

// ======================== Batch Tests ========================

#[test]
fn oracle_conv_2d_batch() {
    // lhs=[2, 2, 2, 1] (batch=2), rhs=[1, 1, 1, 1]
    // Identity kernel on batch of 2
    let lhs = make_f64_tensor(&[2, 2, 2, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 1]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_conv_kernel_equals_input() {
    // When kernel size equals input size with valid padding -> 1x1 output
    let lhs = make_f64_tensor(&[1, 3, 3, 1], (1..=9).map(|i| i as f64).collect());
    let rhs = make_f64_tensor(&[3, 3, 1, 1], vec![1.0; 9]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1, 1]);
    let vals = extract_f64_vec(&result);
    // Sum of 1..9 = 45
    assert!((vals[0] - 45.0).abs() < 1e-10);
}

#[test]
fn oracle_conv_negative_values() {
    let lhs = make_f64_tensor(&[1, 3, 1], vec![-1.0, 0.0, 1.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.0)).abs() < 1e-10); // -1 + 0
    assert!((vals[1] - 1.0).abs() < 1e-10); // 0 + 1
}

#[test]
fn oracle_conv_zeros() {
    let lhs = make_f64_tensor(&[1, 3, 1], vec![0.0, 0.0, 0.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|v| v.abs() < 1e-10));
}

// ======================== Empty Spatial Dimension Tests ========================

#[test]
fn oracle_conv_1d_empty_width_same_padding() {
    // 1D conv with width=0, SAME padding should produce empty output, not panic
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![1, 0, 1],
            },
            vec![],
        )
        .unwrap(),
    );
    let rhs = make_f64_tensor(&[1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0, 1]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_conv_1d_empty_width_valid_padding() {
    // 1D conv with width=0, valid padding should produce empty output
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![1, 0, 1],
            },
            vec![],
        )
        .unwrap(),
    );
    let rhs = make_f64_tensor(&[1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0, 1]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_conv_2d_empty_height_same_padding() {
    // 2D conv with height=0, SAME padding should produce empty output, not panic
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![1, 0, 3, 1],
            },
            vec![],
        )
        .unwrap(),
    );
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0, 3, 1]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_conv_2d_empty_width_same_padding() {
    // 2D conv with width=0, SAME padding should produce empty output, not panic
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![1, 3, 0, 1],
            },
            vec![],
        )
        .unwrap(),
    );
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 0, 1]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_conv_2d_empty_both_same_padding() {
    // 2D conv with height=0 and width=0, SAME padding should produce empty output
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![1, 0, 0, 1],
            },
            vec![],
        )
        .unwrap(),
    );
    let rhs = make_f64_tensor(&[1, 1, 1, 1], vec![1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0, 0, 1]);
    assert!(extract_f64_vec(&result).is_empty());
}
