//! Oracle tests for BitcastConvertType primitive.
//!
//! Tests against expected behavior for bitwise reinterpretation:
//! - Reinterprets bit pattern as a different dtype of same width
//! - i64 <-> f64, i32 <-> f32, etc.
//! - Narrows/widens element widths by appending/removing a trailing dimension

#![allow(dead_code, clippy::approx_constant)]

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

fn make_u32_tensor(shape: &[u32], data: Vec<u32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::U32).collect(),
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
            data.into_iter()
                .map(|x| Literal::F32Bits(x.to_bits()))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_u32_vec(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::U32(v) => *v,
                _ => l.as_u64().unwrap() as u32,
            })
            .collect(),
        Value::Scalar(Literal::U32(v)) => vec![*v],
        _ => unreachable!("expected u32"),
    }
}

fn extract_f32_vec(v: &Value) -> Vec<f32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => f32::from_bits(*bits),
                _ => l.as_f64().unwrap() as f32,
            })
            .collect(),
        Value::Scalar(Literal::F32Bits(bits)) => vec![f32::from_bits(*bits)],
        _ => unreachable!("expected f32"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn bitcast_params(new_dtype: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_string(), new_dtype.to_string());
    p
}

fn f64_to_u32_chunks(value: f64) -> [u32; 2] {
    let bytes = value.to_bits().to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

// ======================== i64 <-> f64 Tests ========================

#[test]
fn oracle_bitcast_f64_to_i64_zero() {
    // f64 0.0 has bits 0x0000_0000_0000_0000
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 0);
}

#[test]
fn oracle_bitcast_f64_to_i64_preserves_special_value_bits() {
    // bitcast is a pure bit reinterpretation, so NaN / +-inf / -0.0 must map to their
    // exact IEEE bit patterns reinterpreted as i64 (matching XLA bitcast_convert) —
    // no NaN canonicalization, no value change. Reference is Rust's f64::to_bits,
    // which is the same bit-level reinterpretation (non-circular vs the eval path).
    let vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0_f64];
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![4] },
            vals.iter().map(|&x| Literal::from_f64(x)).collect(),
        )
        .unwrap(),
    );
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let got = extract_i64_vec(&result);
    let expected: Vec<i64> = vals.iter().map(|&x| x.to_bits() as i64).collect();
    assert_eq!(
        got, expected,
        "bitcast f64->i64 must preserve exact special-value bits (no canonicalization)"
    );
}

#[test]
fn oracle_bitcast_f64_to_i64_one() {
    // f64 1.0 has specific bit pattern
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 1.0_f64.to_bits() as i64);
}

#[test]
fn oracle_bitcast_i64_to_f64_zero() {
    // i64 0 -> f64 0.0
    let input = Value::scalar_i64(0);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("f64"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-15);
}

#[test]
fn oracle_bitcast_i64_to_f64_one_bits() {
    // Bitcast i64 with f64 1.0's bit pattern -> 1.0
    let one_bits = 1.0_f64.to_bits() as i64;
    let input = Value::scalar_i64(one_bits);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("f64"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
}

#[test]
fn oracle_bitcast_f64_i64_roundtrip() {
    // f64 -> i64 -> f64 should preserve value
    let original = 3.17;
    let input = Value::Scalar(Literal::from_f64(original));
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let vals = extract_f64_vec(&back);
    assert!((vals[0] - original).abs() < 1e-15);
}

#[test]
fn oracle_bitcast_f64_to_i64_1d() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 0.0_f64.to_bits() as i64);
    assert_eq!(vals[1], 1.0_f64.to_bits() as i64);
    assert_eq!(vals[2], (-1.0_f64).to_bits() as i64);
}

#[test]
fn oracle_bitcast_f64_to_u32_appends_trailing_dim() {
    let values = [1.25_f64, -0.0_f64];
    let input = make_f64_tensor(&[2], values.to_vec());
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let expected: Vec<u32> = values
        .iter()
        .flat_map(|&value| f64_to_u32_chunks(value))
        .collect();
    assert_eq!(
        extract_u32_vec(&result),
        expected,
        "f64->u32 narrowing must split each little-endian element into two trailing chunks"
    );
}

#[test]
fn oracle_bitcast_u32_to_f64_removes_trailing_dim() {
    let values = [1.25_f64, -0.0_f64];
    let chunks: Vec<u32> = values
        .iter()
        .flat_map(|&value| f64_to_u32_chunks(value))
        .collect();
    let input = make_u32_tensor(&[2, 2], chunks);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("f64"),
    )
    .unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let got_bits: Vec<u64> = extract_f64_vec(&result)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let expected_bits: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(
        got_bits, expected_bits,
        "u32->f64 widening must consume the trailing dimension without changing bits"
    );
}

#[test]
fn oracle_bitcast_u32_to_f64_rejects_wrong_trailing_dim() {
    let input = make_u32_tensor(&[3], vec![1, 2, 3]);
    let err = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("f64"),
    )
    .expect_err("u32->f64 widening requires trailing dimension size 2");
    let detail = format!("{err:?}");
    assert!(
        detail.contains("trailing dimension size 2"),
        "unexpected error: {detail}"
    );
}

// ======================== u32 <-> f32 Tests ========================

#[test]
fn oracle_bitcast_f32_to_u32_zero() {
    let input = make_f32_tensor(&[1], vec![0.0_f32]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();
    let vals = extract_u32_vec(&result);
    assert_eq!(vals[0], 0);
}

#[test]
fn oracle_bitcast_f32_to_u32_preserves_special_value_bits() {
    // bitcast is a pure bit reinterpretation: f32 NaN / +-inf / -0.0 map to their
    // exact IEEE bit patterns as u32 (XLA bitcast_convert, no canonicalization).
    // Reference is Rust f32::to_bits (the same bit-level reinterpretation).
    let vals = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0_f32];
    let input = make_f32_tensor(&[4], vals.to_vec());
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();
    let got = extract_u32_vec(&result);
    let expected: Vec<u32> = vals.iter().map(|&x| x.to_bits()).collect();
    assert_eq!(
        got, expected,
        "bitcast f32->u32 must preserve exact special-value bits (no canonicalization)"
    );
}

#[test]
fn oracle_bitcast_f32_to_u32_one() {
    let input = make_f32_tensor(&[1], vec![1.0_f32]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();
    let vals = extract_u32_vec(&result);
    assert_eq!(vals[0], 1.0_f32.to_bits());
}

#[test]
fn oracle_bitcast_u32_to_f32_roundtrip() {
    let original = vec![1.5_f32, 2.5_f32, 3.5_f32];
    let input = make_f32_tensor(&[3], original.clone());
    let as_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_u32],
        &bitcast_params("f32"),
    )
    .unwrap();
    let vals = extract_f32_vec(&back);
    for (v, o) in vals.iter().zip(original.iter()) {
        assert!((v - o).abs() < 1e-6);
    }
}

#[test]
fn oracle_bitcast_f32_u32_roundtrip_preserves_special_value_bits() {
    // Round-trip f32 -> u32 -> f32 must preserve EXACT bits for NaN / +-inf / -0.0
    // with no canonicalization on either leg. Compared by bit pattern, since NaN
    // != NaN and -0.0 == 0.0 under `==` would hide a payload/sign change.
    let original = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        1.5_f32,
    ];
    let input = make_f32_tensor(&[5], original.to_vec());
    let as_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("u32"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_u32],
        &bitcast_params("f32"),
    )
    .unwrap();
    let got_bits: Vec<u32> = extract_f32_vec(&back).iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = original.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        got_bits, want_bits,
        "f32->u32->f32 round-trip must preserve exact special-value bits"
    );
}

#[test]
fn oracle_bitcast_f64_i64_roundtrip_preserves_special_value_bits() {
    // Round-trip bitcast f64 -> i64 -> f64 must preserve EXACT bits for NaN/+-inf/-0.0
    // with no canonicalization on either leg. Sibling of the f32<->u32 round-trip,
    // distinct width/dtype; compared by bit pattern.
    let original = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0_f64,
        1.5_f64,
    ];
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![5] },
            original.iter().map(|&x| Literal::from_f64(x)).collect(),
        )
        .unwrap(),
    );
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let got: Vec<u64> = extract_f64_vec(&back).iter().map(|v| v.to_bits()).collect();
    let want: Vec<u64> = original.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        got, want,
        "f64->i64->f64 round-trip must preserve exact special-value bits"
    );
}

// ======================== 2D Tests ========================

#[test]
fn oracle_bitcast_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_bitcast_negative_f64() {
    let input = Value::Scalar(Literal::from_f64(-0.0));
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let vals = extract_i64_vec(&result);
    // -0.0 has a different bit pattern than 0.0 (sign bit set)
    assert_ne!(vals[0], 0);
}

#[test]
fn oracle_bitcast_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
}

#[test]
fn oracle_bitcast_nan_preserved() {
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let val = extract_f64_vec(&back)[0];
    assert!(val.is_nan(), "NaN bit pattern should round-trip");
}

#[test]
fn oracle_bitcast_inf_preserved() {
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let val = extract_f64_vec(&back)[0];
    assert!(val.is_infinite() && val > 0.0, "Inf should round-trip");
}

#[test]
fn oracle_bitcast_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_bitcast_3d_shape_preserved() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(result.dtype(), DType::I64);
}

#[test]
fn oracle_bitcast_neg_inf_preserved() {
    let input = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let val = extract_f64_vec(&back)[0];
    assert!(val.is_infinite() && val < 0.0, "-Inf should round-trip");
}

#[test]
fn oracle_bitcast_subnormal_preserved() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = Value::Scalar(Literal::from_f64(subnormal));
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("f64"),
    )
    .unwrap();
    let val = extract_f64_vec(&back)[0];
    assert_eq!(val, subnormal, "subnormal should round-trip");
}

#[test]
fn oracle_bitcast_large_tensor() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let input = make_f64_tensor(&[10, 10], data);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![10, 10]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 0.0_f64.to_bits() as i64);
}

#[test]
fn oracle_bitcast_preserves_dtype() {
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(result.dtype(), DType::I64);
}

#[test]
fn oracle_bitcast_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 4] }, vec![]).unwrap());
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![0, 4]);
    assert_eq!(result.dtype(), DType::I64);
}

// ====================== COMPLEX DTYPE TESTS ======================

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
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]
fn oracle_bitcast_complex64_to_i64() {
    let input = make_complex64_tensor(&[1], vec![(1.0, 2.0)]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .expect("bitcast complex64 to i64 should work");
    assert_eq!(result.dtype(), DType::I64);
}

#[test]
fn oracle_bitcast_i64_to_complex64() {
    let input = make_i64_tensor(&[1], vec![0x4000_0000_3f80_0000_i64]);
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("complex64"),
    )
    .expect("bitcast i64 to complex64 should work");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_bitcast_complex64_roundtrip() {
    let re = 3.14159_f32;
    let im = 2.71828_f32;
    let input = make_complex64_tensor(&[1], vec![(re, im)]);
    let as_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        &[input],
        &bitcast_params("i64"),
    )
    .expect("bitcast to i64");
    let back = eval_primitive(
        Primitive::BitcastConvertType,
        &[as_i64],
        &bitcast_params("complex64"),
    )
    .expect("bitcast back to complex64");
    let vals = extract_complex64_vec(&back);
    assert!(
        (vals[0].0 - re).abs() < 1e-6,
        "real part should round-trip: {} vs {}",
        vals[0].0,
        re
    );
    assert!(
        (vals[0].1 - im).abs() < 1e-6,
        "imag part should round-trip: {} vs {}",
        vals[0].1,
        im
    );
}

#[test]
fn oracle_bitcast_complex128_preserves_shape() {
    let input = make_complex128_tensor(
        &[2, 3],
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ],
    );
    let result = eval_primitive(
        Primitive::BitcastConvertType,
        &[input.clone()],
        &bitcast_params("complex128"),
    )
    .expect("identity bitcast should work");
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(result.dtype(), DType::Complex128);
}
