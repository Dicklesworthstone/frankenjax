//! Oracle tests for Bitwise primitives.
//!
//! Tests against expected behavior for:
//! - BitwiseAnd: element-wise AND
//! - BitwiseOr: element-wise OR
//! - BitwiseXor: element-wise XOR
//! - BitwiseNot: element-wise NOT
//! - PopulationCount: count set bits
//! - CountLeadingZeros: count leading zeros

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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
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
        Value::Scalar(lit) => vec![lit.as_u64().unwrap() as u32],
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== BitwiseAnd Tests ========================

#[test]
fn oracle_bitwise_and_scalar() {
    // 0b1111 & 0b1010 = 0b1010 = 10
    let a = Value::scalar_i64(0b1111);
    let b = Value::scalar_i64(0b1010);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b1010]);
}

#[test]
fn oracle_bitwise_and_zeros() {
    let a = Value::scalar_i64(0xFF);
    let b = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_bitwise_and_1d() {
    let a = make_i64_tensor(&[4], vec![0b1100, 0b1010, 0b1111, 0b0000]);
    let b = make_i64_tensor(&[4], vec![0b1010, 0b1010, 0b1010, 0b1010]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![0b1000, 0b1010, 0b1010, 0b0000]);
}

#[test]
fn oracle_bitwise_and_2d() {
    let a = make_i64_tensor(&[2, 2], vec![0xFF, 0x0F, 0xF0, 0x00]);
    let b = make_i64_tensor(&[2, 2], vec![0x0F, 0x0F, 0x0F, 0x0F]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![0x0F, 0x0F, 0x00, 0x00]);
}

#[test]
fn oracle_bitwise_and_u32() {
    let a = make_u32_tensor(&[3], vec![0xFFFF_FFFF, 0x0000_FFFF, 0xFFFF_0000]);
    let b = make_u32_tensor(&[3], vec![0x0000_FFFF, 0xFFFF_FFFF, 0xFFFF_0000]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_u32_vec(&result), vec![0x0000_FFFF, 0x0000_FFFF, 0xFFFF_0000]);
}

// ======================== BitwiseOr Tests ========================

#[test]
fn oracle_bitwise_or_scalar() {
    // 0b1100 | 0b0011 = 0b1111 = 15
    let a = Value::scalar_i64(0b1100);
    let b = Value::scalar_i64(0b0011);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b1111]);
}

#[test]
fn oracle_bitwise_or_with_zero() {
    let a = Value::scalar_i64(0b1010);
    let b = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b1010]);
}

#[test]
fn oracle_bitwise_or_1d() {
    let a = make_i64_tensor(&[4], vec![0b1000, 0b0100, 0b0010, 0b0001]);
    let b = make_i64_tensor(&[4], vec![0b0001, 0b0010, 0b0100, 0b1000]);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b1001, 0b0110, 0b0110, 0b1001]);
}

#[test]
fn oracle_bitwise_or_2d() {
    let a = make_i64_tensor(&[2, 2], vec![0xF0, 0x0F, 0x00, 0xFF]);
    let b = make_i64_tensor(&[2, 2], vec![0x0F, 0xF0, 0x00, 0x00]);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0xFF, 0xFF, 0x00, 0xFF]);
}

// ======================== BitwiseXor Tests ========================

#[test]
fn oracle_bitwise_xor_scalar() {
    // 0b1100 ^ 0b1010 = 0b0110 = 6
    let a = Value::scalar_i64(0b1100);
    let b = Value::scalar_i64(0b1010);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b0110]);
}

#[test]
fn oracle_bitwise_xor_same() {
    // x ^ x = 0
    let a = Value::scalar_i64(0x12345678);
    let b = Value::scalar_i64(0x12345678);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_bitwise_xor_1d() {
    let a = make_i64_tensor(&[4], vec![0xFF, 0x00, 0xAA, 0x55]);
    let b = make_i64_tensor(&[4], vec![0xFF, 0xFF, 0x55, 0xAA]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0x00, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn oracle_bitwise_xor_toggle() {
    // XOR with all 1s toggles all bits
    let a = make_i64_tensor(&[3], vec![0b0000, 0b1111, 0b1010]);
    let b = make_i64_tensor(&[3], vec![0b1111, 0b1111, 0b1111]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0b1111, 0b0000, 0b0101]);
}

// ======================== BitwiseNot Tests ========================

#[test]
fn oracle_bitwise_not_scalar() {
    // !0 = -1 (all bits set in two's complement)
    let a = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1]);
}

#[test]
fn oracle_bitwise_not_minus_one() {
    // !(-1) = 0
    let a = Value::scalar_i64(-1);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_bitwise_not_1d() {
    let a = make_i64_tensor(&[3], vec![0, -1, 1]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, 0, -2]);
}

#[test]
fn oracle_bitwise_not_u32() {
    let a = make_u32_tensor(&[3], vec![0, 0xFFFF_FFFF, 0x0000_FFFF]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_u32_vec(&result), vec![0xFFFF_FFFF, 0, 0xFFFF_0000]);
}

#[test]
fn oracle_bitwise_not_double() {
    // !!x = x
    let a = Value::scalar_i64(42);
    let result1 = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::BitwiseNot, &[result1], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result2), vec![42]);
}

// ======================== PopulationCount Tests ========================

#[test]
fn oracle_popcount_zero() {
    let a = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_popcount_one() {
    let a = Value::scalar_i64(1);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1]);
}

#[test]
fn oracle_popcount_powers_of_two() {
    let a = make_i64_tensor(&[4], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1]);
}

#[test]
fn oracle_popcount_all_ones_byte() {
    let a = Value::scalar_i64(0xFF);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![8]);
}

#[test]
fn oracle_popcount_1d() {
    let a = make_i64_tensor(&[5], vec![0b0000, 0b0001, 0b0011, 0b0111, 0b1111]);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_popcount_u32() {
    let a = make_u32_tensor(&[3], vec![0, 0xFFFF_FFFF, 0x5555_5555]);
    let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
    assert_eq!(extract_u32_vec(&result), vec![0, 32, 16]);
}

// ======================== CountLeadingZeros Tests ========================

#[test]
fn oracle_clz_zero() {
    // CLZ of 0 returns bit width (64 for i64)
    let a = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![64]);
}

#[test]
fn oracle_clz_one() {
    // 1 has 63 leading zeros in i64
    let a = Value::scalar_i64(1);
    let result = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![63]);
}

#[test]
fn oracle_clz_powers_of_two() {
    let a = make_i64_tensor(&[4], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![63, 62, 61, 60]);
}

#[test]
fn oracle_clz_high_bit() {
    // Number with MSB set (negative in signed, high bit in unsigned view)
    let a = Value::scalar_i64(1_i64 << 62);
    let result = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1]);
}

#[test]
fn oracle_clz_u32() {
    let a = make_u32_tensor(&[4], vec![0, 1, 0x8000_0000, 0xFFFF_FFFF]);
    let result = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
    assert_eq!(extract_u32_vec(&result), vec![32, 31, 0, 0]);
}

// ======================== Combined Tests ========================

#[test]
fn oracle_bitwise_demorgan() {
    // De Morgan's law: !(a & b) = !a | !b
    let a = make_i64_tensor(&[3], vec![0b1100, 0b1010, 0b1111]);
    let b = make_i64_tensor(&[3], vec![0b1010, 0b1010, 0b0000]);

    // !(a & b)
    let and_result = eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b.clone()], &no_params()).unwrap();
    let not_and = eval_primitive(Primitive::BitwiseNot, &[and_result], &no_params()).unwrap();

    // !a | !b
    let not_a = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    let not_b = eval_primitive(Primitive::BitwiseNot, &[b], &no_params()).unwrap();
    let or_nots = eval_primitive(Primitive::BitwiseOr, &[not_a, not_b], &no_params()).unwrap();

    assert_eq!(extract_i64_vec(&not_and), extract_i64_vec(&or_nots));
}

#[test]
fn oracle_bitwise_identity_xor() {
    // a ^ a = 0
    let a = make_i64_tensor(&[4], vec![0x12, 0x34, 0x56, 0x78]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a.clone(), a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}
