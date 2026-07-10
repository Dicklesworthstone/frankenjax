//! Oracle tests for Conv primitive.
//!
//! Tests against expected behavior for 1D and 2D convolution:
//! - lhs: input tensor [N, H, W, C_in] or [N, L, C_in]
//! - rhs: kernel [KH, KW, C_in, C_out] or [K, C_in, C_out]
//! - params: padding ("VALID", "SAME", or "SAME_LOWER"), stride

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
fn oracle_conv_1d_explicit_padding() {
    // Explicit padding "EXPLICIT:low,high" — eval-layer support landed in us8kk
    // (commit f55faa67). lhs=[1,3,1]=[1,2,3], rhs=[2,1,1]=[1,1] (moving sum of 2),
    // pad 1 on each side, stride 1: padded=[0,1,2,3,0] -> moving-sum-2 = [1,3,5,3];
    // output length = (3+1+1) - 2 + 1 = 4.
    // NOTE: this exercises eval_conv directly; the fj-trace / partial_eval shape-
    // inference layers still reject "EXPLICIT:" (tracked separately) so a traced/jit
    // conv with explicit padding is not yet supported end-to-end.
    let lhs = make_f64_tensor(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(
        Primitive::Conv,
        &[lhs, rhs],
        &conv_params("EXPLICIT:1,1", "1"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4, 1]);
    let vals = extract_f64_vec(&result);
    let expected = [1.0, 3.0, 5.0, 3.0];
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (vals[i] - e).abs() < 1e-10,
            "explicit-pad conv[{i}] = {e}, got {}",
            vals[i]
        );
    }
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
fn oracle_conv_1d_lhs_dilation_transposed() {
    // lhs_dilation (input/operand dilation = transposed / fractionally-strided conv)
    // inserts (db-1) zeros between input elements: [1,2,3] with lhs_dilation=2 ->
    // [1,0,2,0,3]. A valid conv with kernel [1,1] (correlation, moving sum of 2)
    // then gives out[p] = dilated[p] + dilated[p+1] = [1,2,2,3]. Supported in
    // eval_conv but untested at the conformance/parity layer.
    //   dilated length = (3-1)*2 + 1 = 5, output length = 5-2+1 = 4
    let lhs = make_f64_tensor(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let mut params = conv_params("valid", "1");
    params.insert("lhs_dilation".to_string(), "2".to_string());
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10, "1+0 = 1");
    assert!((vals[1] - 2.0).abs() < 1e-10, "0+2 = 2");
    assert!((vals[2] - 2.0).abs() < 1e-10, "2+0 = 2");
    assert!((vals[3] - 3.0).abs() < 1e-10, "0+3 = 3");
}

#[test]
fn oracle_conv_1d_rhs_dilation_atrous() {
    // Atrous (dilated) convolution: rhs_dilation=2 spaces the 2 kernel taps 2 apart,
    // so the effective kernel is [1,0,1] (span 3). Supported in eval_conv but
    // untested at the conformance/parity layer.
    // input=[1,2,3,4,5], kernel=[1,1], valid: out[p] = in[p] + in[p+2]
    //   span = (2-1)*2 + 1 = 3, output length = (5-3)/1 + 1 = 3
    //   p=0: 1+3=4; p=1: 2+4=6; p=2: 3+5=8
    let lhs = make_f64_tensor(&[1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let mut params = conv_params("valid", "1");
    params.insert("rhs_dilation".to_string(), "2".to_string());
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10, "in[0]+in[2] = 4");
    assert!((vals[1] - 6.0).abs() < 1e-10, "in[1]+in[3] = 6");
    assert!((vals[2] - 8.0).abs() < 1e-10, "in[2]+in[4] = 8");
}

#[test]
fn oracle_conv_1d_f32_preserves_literal_dtype() {
    let lhs = make_f32_tensor(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = make_f32_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("VALID", "1")).unwrap();

    if let Value::Tensor(t) = &result {
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(extract_shape(&result), vec![1, 2, 1]);
        t.validate_dtype_consistency()
            .expect("conv F32 output dtype/element invariant");
    } else {
        assert!(matches!(result, Value::Tensor(_)), "expected tensor");
    }
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0]);
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
fn oracle_conv_1d_uppercase_same_padding() {
    let lhs = make_f64_tensor(&[1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("SAME", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4, 1]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![3.0, 5.0, 7.0, 4.0]);
}

#[test]
fn oracle_conv_1d_same_lower_padding() {
    let lhs = make_f64_tensor(&[1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(
        Primitive::Conv,
        &[lhs, rhs],
        &conv_params("SAME_LOWER", "1"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4, 1]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn oracle_conv_unknown_padding_rejected() {
    let lhs = make_f64_tensor(&[1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
    let err = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("MIRROR", "1"))
        .expect_err("unknown padding should fail closed");
    assert!(
        err.to_string().contains("unsupported conv padding mode"),
        "unexpected error: {err}"
    );
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

#[test]
fn oracle_conv_1d_multi_channel_in() {
    // Multi-INPUT-channel conv (c_in=2): the output sums over BOTH input channels
    // (channel reduction) — the prior multi-channel test only exercised c_out. With
    // an all-ones kernel of size 2 over channels {ch0=[1,2,3], ch1=[10,20,30]} the
    // output is just the window-sum across both channels, which is commutative and
    // so independent of the exact channel-interleaving layout:
    //   p=0: (1+10)+(2+20) = 33;  p=1: (2+20)+(3+30) = 55
    // lhs [batch=1, length=3, c_in=2] interleaved; rhs [kernel=2, c_in=2, c_out=1].
    let lhs = make_f64_tensor(&[1, 3, 2], vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    let rhs = make_f64_tensor(&[2, 2, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 33.0).abs() < 1e-10, "(1+10)+(2+20) = 33");
    assert!((vals[1] - 55.0).abs() < 1e-10, "(2+20)+(3+30) = 55");
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
fn oracle_conv_2d_rhs_dilation_atrous() {
    // 2D atrous (dilated) convolution — the realistic dilated-CNN case, completing
    // the dilation family. 3x3 input, 2x2 all-ones kernel, rhs_dilation 2x2 ->
    // effective 3x3 kernel, valid -> a single 1x1 output. With the all-ones kernel
    // the result is the (commutative, layout-robust) sum of the dilated corners
    // {(0,0),(0,2),(2,0),(2,2)} = {1,3,7,9} = 20.
    let lhs = make_f64_tensor(&[1, 3, 3, 1], (1..=9).map(|i| i as f64).collect());
    let rhs = make_f64_tensor(&[2, 2, 1, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let mut params = conv_params("valid", "1");
    params.insert("rhs_dilation".to_string(), "2,2".to_string());
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1, 1]);
    let vals = extract_f64_vec(&result);
    assert!(
        (vals[0] - 20.0).abs() < 1e-10,
        "sum of dilated corners {{1,3,7,9}} = 20"
    );
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
    let lhs = make_f64_tensor(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
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
fn oracle_conv_1d_valid_kernel_larger_than_input_returns_empty() {
    let lhs = make_f64_tensor(&[1, 1, 1], vec![2.0]);
    let rhs = make_f64_tensor(&[2, 1, 1], vec![1.0, 1.0]);
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
fn oracle_conv_2d_valid_kernel_larger_than_height_returns_empty() {
    let lhs = make_f64_tensor(&[1, 1, 3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = make_f64_tensor(&[2, 1, 1, 1], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
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

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_conv_zero_kernel() {
    // Conv(x, 0) = 0
    let lhs = make_f64_tensor(&[1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rhs = make_f64_tensor(&[3, 1, 1], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
    let vals = extract_f64_vec(&result);
    for v in vals {
        assert!(
            v.abs() < 1e-10,
            "Conv with zero kernel should be zero, got {v}"
        );
    }
}

#[test]
fn metamorphic_conv_scaling() {
    // Conv(c*x, k) = c * Conv(x, k)
    let c = 3.0;
    let lhs = make_f64_tensor(&[1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let lhs_scaled = make_f64_tensor(&[1, 5, 1], vec![3.0, 6.0, 9.0, 12.0, 15.0]);
    let rhs = make_f64_tensor(&[3, 1, 1], vec![1.0, 2.0, 1.0]);

    let result = eval_primitive(
        Primitive::Conv,
        &[lhs, rhs.clone()],
        &conv_params("valid", "1"),
    )
    .unwrap();
    let result_scaled = eval_primitive(
        Primitive::Conv,
        &[lhs_scaled, rhs],
        &conv_params("valid", "1"),
    )
    .unwrap();

    let vals = extract_f64_vec(&result);
    let vals_scaled = extract_f64_vec(&result_scaled);

    for (v, vs) in vals.iter().zip(vals_scaled.iter()) {
        assert!(
            (vs - c * v).abs() < 1e-10,
            "Conv(c*x, k) = c*Conv(x, k): got {vs}, expected {}",
            c * v
        );
    }
}

#[test]
fn metamorphic_conv_1x1_identity_kernel() {
    // Conv with 1x1 kernel of value 1 preserves input values
    let lhs = make_f64_tensor(
        &[1, 3, 3, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ],
    );
    let rhs = make_f64_tensor(&[1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = eval_primitive(
        Primitive::Conv,
        &[lhs.clone(), rhs],
        &conv_params("same", "1"),
    )
    .unwrap();
    let input_vals = extract_f64_vec(&lhs);
    let output_vals = extract_f64_vec(&result);
    assert_eq!(input_vals.len(), output_vals.len());
    for (inp, out) in input_vals.iter().zip(output_vals.iter()) {
        assert!(
            (inp - out).abs() < 1e-10,
            "1x1 identity kernel should preserve values: got {out}, expected {inp}"
        );
    }
}

#[test]
fn metamorphic_conv_additivity() {
    // Conv(x, k1 + k2) = Conv(x, k1) + Conv(x, k2)
    let lhs = make_f64_tensor(&[1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let k1 = make_f64_tensor(&[3, 1, 1], vec![1.0, 0.0, 0.0]);
    let k2 = make_f64_tensor(&[3, 1, 1], vec![0.0, 1.0, 1.0]);
    let k_sum = make_f64_tensor(&[3, 1, 1], vec![1.0, 1.0, 1.0]);

    let r1 = eval_primitive(
        Primitive::Conv,
        &[lhs.clone(), k1],
        &conv_params("valid", "1"),
    )
    .unwrap();
    let r2 = eval_primitive(
        Primitive::Conv,
        &[lhs.clone(), k2],
        &conv_params("valid", "1"),
    )
    .unwrap();
    let r_sum = eval_primitive(Primitive::Conv, &[lhs, k_sum], &conv_params("valid", "1")).unwrap();

    let v1 = extract_f64_vec(&r1);
    let v2 = extract_f64_vec(&r2);
    let v_sum = extract_f64_vec(&r_sum);

    for i in 0..v1.len() {
        assert!(
            (v1[i] + v2[i] - v_sum[i]).abs() < 1e-10,
            "Conv(x, k1+k2) = Conv(x, k1) + Conv(x, k2): got {}, expected {}",
            v_sum[i],
            v1[i] + v2[i]
        );
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_conv_preserves_float_dtypes() {
    fn make_lhs(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![1, 3, 1],
                },
                lits,
            )
            .unwrap(),
        )
    }
    fn make_rhs(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![2, 1, 1],
                },
                lits,
            )
            .unwrap(),
        )
    }
    let lhs_values = [1.0_f64, 2.0, 3.0];
    let rhs_values = [1.0_f64, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let lhs = make_lhs(dtype, &lhs_values);
        let rhs = make_rhs(dtype, &rhs_values);
        let result =
            eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "conv {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        Value::Scalar(lit) => vec![lit.as_complex64().unwrap()],
    }
}

#[test]
fn oracle_conv_1d_complex64_simple() {
    // 1D conv with complex-valued input and kernel
    // input: [1+0i, 2+0i, 3+0i] shape [1, 3, 1] (batch, length, channels)
    // kernel: [1+0i, 1+0i] shape [2, 1, 1] (kernel_size, in_channels, out_channels)
    // result: [3+0i, 5+0i] (valid padding)
    let lhs = make_complex64_tensor(&[1, 3, 1], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let rhs = make_complex64_tensor(&[2, 1, 1], vec![(1.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1"))
        .expect("conv complex64 should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 3.0).abs() < 1e-5);
    assert!((vals[1].0 - 5.0).abs() < 1e-5);
}

#[test]
fn oracle_conv_1d_complex64_with_imaginary() {
    // Complex conv: [1+i, 2+2i] * [1+0i, i]
    // Position 0: (1+i)*1 + (2+2i)*i = 1+i + 2i - 2 = -1 + 3i
    let lhs = make_complex64_tensor(&[1, 2, 1], vec![(1.0, 1.0), (2.0, 2.0)]);
    let rhs = make_complex64_tensor(&[2, 1, 1], vec![(1.0, 0.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1"))
        .expect("conv complex64 with imaginary should succeed");
    let vals = extract_complex64_vec(&result);
    assert!(
        (vals[0].0 - (-1.0)).abs() < 1e-5,
        "expected -1, got {}",
        vals[0].0
    );
    assert!(
        (vals[0].1 - 3.0).abs() < 1e-5,
        "expected 3, got {}",
        vals[0].1
    );
}

#[test]
fn oracle_conv_1d_complex128_simple() {
    let lhs = make_complex128_tensor(&[1, 3, 1], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let rhs = make_complex128_tensor(&[2, 1, 1], vec![(1.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1"))
        .expect("conv complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn property_conv_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (lhs, rhs) = match dtype {
            DType::Complex64 => (
                make_complex64_tensor(&[1, 3, 1], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]),
                make_complex64_tensor(&[2, 1, 1], vec![(1.0, 0.0), (1.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_tensor(&[1, 3, 1], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]),
                make_complex128_tensor(&[2, 1, 1], vec![(1.0, 0.0), (1.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1"))
            .expect("conv should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "conv {dtype:?}: dtype mismatch");
    }
}
