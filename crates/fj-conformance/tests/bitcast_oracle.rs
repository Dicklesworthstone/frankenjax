//! Oracle tests for BitcastConvertType primitive.
//!
//! Tests against expected behavior for bitwise reinterpretation:
//! - Reinterprets bit pattern as a different dtype of same width
//! - i64 <-> f64, i32 <-> f32, etc.
//! - Narrows/widens element widths by appending/removing a trailing dimension

#![allow(dead_code, clippy::approx_constant)]

use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
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

fn make_literal_backed_tensor(dtype: DType, shape: &[u32], elements: Vec<Literal>) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            dtype,
            Shape {
                dims: shape.to_vec(),
            },
            LiteralBuffer::new(elements),
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

fn extract_i32_bits_vec(v: &Value) -> Vec<u32> {
    fn literal_bits(literal: &Literal) -> u32 {
        let value = i32::try_from(literal.as_i64().unwrap()).unwrap();
        u32::from_le_bytes(value.to_le_bytes())
    }

    match v {
        Value::Tensor(t) => t.elements.iter().map(literal_bits).collect(),
        Value::Scalar(lit) => vec![literal_bits(lit)],
    }
}

fn extract_u64_vec(v: &Value) -> Vec<u64> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::U64(v) => *v,
                _ => l.as_u64().unwrap(),
            })
            .collect(),
        Value::Scalar(Literal::U64(v)) => vec![*v],
        _ => unreachable!("expected u64"),
    }
}

fn extract_half_bits_vec(v: &Value) -> Vec<u16> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::BF16Bits(bits) | Literal::F16Bits(bits) => *bits,
                other => panic!("expected half bits, got {other:?}"),
            })
            .collect(),
        Value::Scalar(Literal::BF16Bits(bits)) | Value::Scalar(Literal::F16Bits(bits)) => {
            vec![*bits]
        }
        _ => unreachable!("expected half bits"),
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

fn f32_to_u16_chunks(value: f32) -> [u16; 2] {
    let bytes = value.to_bits().to_le_bytes();
    [
        u16::from_le_bytes([bytes[0], bytes[1]]),
        u16::from_le_bytes([bytes[2], bytes[3]]),
    ]
}

fn half_literal(dtype: DType, bits: u16) -> Literal {
    match dtype {
        DType::BF16 => Literal::BF16Bits(bits),
        DType::F16 => Literal::F16Bits(bits),
        other => panic!("expected half dtype, got {other:?}"),
    }
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
fn oracle_dense_same_width_bitcast_matches_literal_backing_exact_bits() {
    let f32_values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        1.5_f32,
        -2.75_f32,
        f32::from_bits(0x7fc0_1234),
    ];
    let f32_shape = [f32_values.len() as u32];
    let dense_f32 = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: f32_shape.to_vec(),
            },
            f32_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f32 = make_literal_backed_tensor(
        DType::F32,
        &f32_shape,
        f32_values
            .iter()
            .map(|value| Literal::from_f32(*value))
            .collect(),
    );

    let dense_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_f32),
        &bitcast_params("u32"),
    )
    .unwrap();
    let literal_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_f32),
        &bitcast_params("u32"),
    )
    .unwrap();
    assert_eq!(extract_shape(&dense_u32), extract_shape(&literal_u32));
    assert_eq!(dense_u32.dtype(), literal_u32.dtype());
    assert_eq!(extract_u32_vec(&dense_u32), extract_u32_vec(&literal_u32));
    assert!(
        dense_u32
            .as_tensor()
            .unwrap()
            .elements
            .as_u32_slice()
            .is_some(),
        "dense f32->u32 output should stay packed"
    );

    let dense_f32_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_u32),
        &bitcast_params("f32"),
    )
    .unwrap();
    let literal_f32_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_u32),
        &bitcast_params("f32"),
    )
    .unwrap();
    let dense_f32_bits: Vec<u32> = extract_f32_vec(&dense_f32_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let literal_f32_bits: Vec<u32> = extract_f32_vec(&literal_f32_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(dense_f32_bits, literal_f32_bits);
    assert!(
        dense_f32_roundtrip
            .as_tensor()
            .unwrap()
            .elements
            .as_f32_slice()
            .is_some(),
        "dense u32->f32 output should stay packed"
    );

    let f64_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0_f64,
        1.5_f64,
        -2.75_f64,
        f64::from_bits(0x7ff8_0000_0000_1234),
    ];
    let f64_shape = [f64_values.len() as u32];
    let dense_f64 = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: f64_shape.to_vec(),
            },
            f64_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f64 = make_literal_backed_tensor(
        DType::F64,
        &f64_shape,
        f64_values
            .iter()
            .map(|value| Literal::from_f64(*value))
            .collect(),
    );

    let dense_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_f64),
        &bitcast_params("i64"),
    )
    .unwrap();
    let literal_i64 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_f64),
        &bitcast_params("i64"),
    )
    .unwrap();
    assert_eq!(extract_shape(&dense_i64), extract_shape(&literal_i64));
    assert_eq!(dense_i64.dtype(), literal_i64.dtype());
    assert_eq!(extract_i64_vec(&dense_i64), extract_i64_vec(&literal_i64));
    assert!(
        dense_i64
            .as_tensor()
            .unwrap()
            .elements
            .as_i64_slice()
            .is_some(),
        "dense f64->i64 output should stay packed"
    );

    let dense_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_i64),
        &bitcast_params("f64"),
    )
    .unwrap();
    let literal_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_i64),
        &bitcast_params("f64"),
    )
    .unwrap();
    let dense_f64_bits: Vec<u64> = extract_f64_vec(&dense_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let literal_f64_bits: Vec<u64> = extract_f64_vec(&literal_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(dense_f64_bits, literal_f64_bits);
    assert!(
        dense_f64_roundtrip
            .as_tensor()
            .unwrap()
            .elements
            .as_f64_slice()
            .is_some(),
        "dense i64->f64 output should stay packed"
    );
}

#[test]
fn oracle_dense_signed_unsigned_same_width_bitcast_matches_literal_backing_exact_bits() {
    let f32_values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        1.5_f32,
        -2.75_f32,
        f32::from_bits(0x7fc0_1234),
    ];
    let f32_shape = [f32_values.len() as u32];
    let dense_f32 = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: f32_shape.to_vec(),
            },
            f32_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f32 = make_literal_backed_tensor(
        DType::F32,
        &f32_shape,
        f32_values
            .iter()
            .map(|value| Literal::from_f32(*value))
            .collect(),
    );

    let dense_i32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_f32),
        &bitcast_params("i32"),
    )
    .unwrap();
    let literal_i32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_f32),
        &bitcast_params("i32"),
    )
    .unwrap();
    let expected_i32_bits: Vec<u32> = f32_values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(extract_shape(&dense_i32), extract_shape(&literal_i32));
    assert_eq!(dense_i32.dtype(), literal_i32.dtype());
    assert_eq!(
        extract_i32_bits_vec(&dense_i32),
        extract_i32_bits_vec(&literal_i32)
    );
    assert_eq!(extract_i32_bits_vec(&dense_i32), expected_i32_bits);
    assert!(
        dense_i32
            .as_tensor()
            .unwrap()
            .elements
            .as_i64_slice()
            .is_some(),
        "dense f32->i32 output should stay packed"
    );

    let dense_f32_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_i32),
        &bitcast_params("f32"),
    )
    .unwrap();
    let literal_f32_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_i32),
        &bitcast_params("f32"),
    )
    .unwrap();
    let dense_f32_bits: Vec<u32> = extract_f32_vec(&dense_f32_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let literal_f32_bits: Vec<u32> = extract_f32_vec(&literal_f32_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(dense_f32_bits, literal_f32_bits);
    assert_eq!(dense_f32_bits, expected_i32_bits);
    assert!(
        dense_f32_roundtrip
            .as_tensor()
            .unwrap()
            .elements
            .as_f32_slice()
            .is_some(),
        "dense i32->f32 output should stay packed"
    );

    let f64_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0_f64,
        1.5_f64,
        -2.75_f64,
        f64::from_bits(0x7ff8_0000_0000_1234),
    ];
    let f64_shape = [f64_values.len() as u32];
    let dense_f64 = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: f64_shape.to_vec(),
            },
            f64_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f64 = make_literal_backed_tensor(
        DType::F64,
        &f64_shape,
        f64_values
            .iter()
            .map(|value| Literal::from_f64(*value))
            .collect(),
    );

    let dense_u64 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_f64),
        &bitcast_params("u64"),
    )
    .unwrap();
    let literal_u64 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_f64),
        &bitcast_params("u64"),
    )
    .unwrap();
    let expected_u64_bits: Vec<u64> = f64_values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(extract_shape(&dense_u64), extract_shape(&literal_u64));
    assert_eq!(dense_u64.dtype(), literal_u64.dtype());
    assert_eq!(extract_u64_vec(&dense_u64), extract_u64_vec(&literal_u64));
    assert_eq!(extract_u64_vec(&dense_u64), expected_u64_bits);
    assert!(
        dense_u64
            .as_tensor()
            .unwrap()
            .elements
            .as_u64_slice()
            .is_some(),
        "dense f64->u64 output should stay packed"
    );

    let dense_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_u64),
        &bitcast_params("f64"),
    )
    .unwrap();
    let literal_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_u64),
        &bitcast_params("f64"),
    )
    .unwrap();
    let dense_f64_bits: Vec<u64> = extract_f64_vec(&dense_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let literal_f64_bits: Vec<u64> = extract_f64_vec(&literal_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(dense_f64_bits, literal_f64_bits);
    assert_eq!(dense_f64_bits, expected_u64_bits);
    assert!(
        dense_f64_roundtrip
            .as_tensor()
            .unwrap()
            .elements
            .as_f64_slice()
            .is_some(),
        "dense u64->f64 output should stay packed"
    );
}

#[test]
fn oracle_dense_width_changing_bitcast_matches_literal_backing_exact_bits() {
    let f64_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0_f64,
        1.5_f64,
        -2.75_f64,
        f64::from_bits(0x7ff8_0000_0000_1234),
    ];
    let f64_shape = [f64_values.len() as u32];
    let dense_f64 = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: f64_shape.to_vec(),
            },
            f64_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f64 = make_literal_backed_tensor(
        DType::F64,
        &f64_shape,
        f64_values
            .iter()
            .map(|value| Literal::from_f64(*value))
            .collect(),
    );

    let dense_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_f64),
        &bitcast_params("u32"),
    )
    .unwrap();
    let literal_u32 = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_f64),
        &bitcast_params("u32"),
    )
    .unwrap();
    assert_eq!(extract_shape(&dense_u32), vec![f64_values.len() as u32, 2]);
    assert_eq!(extract_shape(&dense_u32), extract_shape(&literal_u32));
    assert_eq!(dense_u32.dtype(), literal_u32.dtype());
    assert_eq!(extract_u32_vec(&dense_u32), extract_u32_vec(&literal_u32));
    assert!(
        dense_u32
            .as_tensor()
            .unwrap()
            .elements
            .as_u32_slice()
            .is_some(),
        "dense f64->u32 narrowing output should stay packed"
    );

    let literal_u32_input = make_literal_backed_tensor(
        DType::U32,
        &[f64_values.len() as u32, 2],
        f64_values
            .iter()
            .flat_map(|value| f64_to_u32_chunks(*value))
            .map(Literal::U32)
            .collect(),
    );
    let dense_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&dense_u32),
        &bitcast_params("f64"),
    )
    .unwrap();
    let literal_f64_roundtrip = eval_primitive(
        Primitive::BitcastConvertType,
        std::slice::from_ref(&literal_u32_input),
        &bitcast_params("f64"),
    )
    .unwrap();
    let dense_f64_bits: Vec<u64> = extract_f64_vec(&dense_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let literal_f64_bits: Vec<u64> = extract_f64_vec(&literal_f64_roundtrip)
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let original_bits: Vec<u64> = f64_values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(dense_f64_bits, literal_f64_bits);
    assert_eq!(dense_f64_bits, original_bits);
    assert!(
        dense_f64_roundtrip
            .as_tensor()
            .unwrap()
            .elements
            .as_f64_slice()
            .is_some(),
        "dense u32->f64 widening output should stay packed"
    );
}

#[test]
fn oracle_dense_half_width_bitcast_matches_literal_backing_exact_bits() {
    let f32_values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        1.5_f32,
        -2.75_f32,
        f32::from_bits(0x7fc0_1234),
    ];
    let f32_shape = [f32_values.len() as u32];
    let dense_f32 = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: f32_shape.to_vec(),
            },
            f32_values.to_vec(),
        )
        .unwrap(),
    );
    let literal_f32 = make_literal_backed_tensor(
        DType::F32,
        &f32_shape,
        f32_values
            .iter()
            .map(|value| Literal::from_f32(*value))
            .collect(),
    );

    for (target_dtype, target_name) in [(DType::BF16, "bf16"), (DType::F16, "f16")] {
        let expected_chunks: Vec<u16> = f32_values
            .iter()
            .flat_map(|value| f32_to_u16_chunks(*value))
            .collect();

        let dense_half = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&dense_f32),
            &bitcast_params(target_name),
        )
        .unwrap();
        let literal_half = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&literal_f32),
            &bitcast_params(target_name),
        )
        .unwrap();
        assert_eq!(extract_shape(&dense_half), vec![f32_values.len() as u32, 2]);
        assert_eq!(extract_shape(&dense_half), extract_shape(&literal_half));
        assert_eq!(dense_half.dtype(), target_dtype);
        assert_eq!(literal_half.dtype(), target_dtype);
        assert_eq!(extract_half_bits_vec(&dense_half), expected_chunks);
        assert_eq!(
            extract_half_bits_vec(&dense_half),
            extract_half_bits_vec(&literal_half)
        );
        assert!(
            dense_half
                .as_tensor()
                .unwrap()
                .elements
                .as_half_float_slice()
                .is_some(),
            "dense f32->{target_name} narrowing output should stay packed"
        );

        let literal_half_input = make_literal_backed_tensor(
            target_dtype,
            &[f32_values.len() as u32, 2],
            expected_chunks
                .iter()
                .copied()
                .map(|bits| half_literal(target_dtype, bits))
                .collect(),
        );
        let dense_f32_roundtrip = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&dense_half),
            &bitcast_params("f32"),
        )
        .unwrap();
        let literal_f32_roundtrip = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&literal_half_input),
            &bitcast_params("f32"),
        )
        .unwrap();
        let dense_bits: Vec<u32> = extract_f32_vec(&dense_f32_roundtrip)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let literal_bits: Vec<u32> = extract_f32_vec(&literal_f32_roundtrip)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let original_bits: Vec<u32> = f32_values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(dense_bits, literal_bits);
        assert_eq!(dense_bits, original_bits);
        assert!(
            dense_f32_roundtrip
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_some(),
            "dense {target_name}->f32 widening output should stay packed"
        );
    }
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
