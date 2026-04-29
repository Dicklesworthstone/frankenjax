//! Oracle tests for CountLeadingZeros primitive.
//!
//! clz(x) = number of leading zero bits before the first 1 bit
//!
//! Tests:
//! - Zero: clz(0) = bit_width (all zeros)
//! - One: clz(1) = bit_width - 1
//! - Powers of two
//! - All ones: clz(-1) = 0 for signed, clz(MAX) = 0 for unsigned
//! - Specific bit patterns
//! - Tensor shapes

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

fn make_u64_tensor(shape: &[u32], data: Vec<u64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::U64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::U64).collect(),
        )
        .unwrap(),
    )
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_u64_scalar(v: &Value) -> u64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_u64().unwrap()
        }
        Value::Scalar(l) => l.as_u64().unwrap(),
    }
}

fn extract_u64_vec(v: &Value) -> Vec<u64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_u64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
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

// ====================== SCALAR ZERO ======================

#[test]
fn oracle_clz_zero_i64() {
    let input = make_i64_tensor(&[], vec![0]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 64, "clz(0i64) = 64");
}

#[test]
fn oracle_clz_zero_u64() {
    let input = make_u64_tensor(&[], vec![0]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 64, "clz(0u64) = 64");
}

// ====================== SCALAR ONE ======================

#[test]
fn oracle_clz_one_i64() {
    let input = make_i64_tensor(&[], vec![1]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 63, "clz(1i64) = 63");
}

#[test]
fn oracle_clz_one_u64() {
    let input = make_u64_tensor(&[], vec![1]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 63, "clz(1u64) = 63");
}

// ====================== POWERS OF TWO ======================

#[test]
fn oracle_clz_powers_of_two_i64() {
    for exp in 0..63 {
        let val = 1i64 << exp;
        let input = make_i64_tensor(&[], vec![val]);
        let result =
            eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            63 - exp,
            "clz(2^{}) = {}",
            exp,
            63 - exp
        );
    }
}

#[test]
fn oracle_clz_powers_of_two_u64() {
    for exp in 0..64 {
        let val = 1u64 << exp;
        let input = make_u64_tensor(&[], vec![val]);
        let result =
            eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
        assert_eq!(
            extract_u64_scalar(&result),
            63 - exp,
            "clz(2^{}) = {}",
            exp,
            63 - exp
        );
    }
}

// ====================== ALL ONES ======================

#[test]
fn oracle_clz_all_ones_i64() {
    let input = make_i64_tensor(&[], vec![-1i64]); // All 64 bits set
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "clz(-1i64) = 0");
}

#[test]
fn oracle_clz_max_u64() {
    let input = make_u64_tensor(&[], vec![u64::MAX]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 0, "clz(u64::MAX) = 0");
}

// ====================== I64 BOUNDARIES ======================

#[test]
fn oracle_clz_max_i64() {
    let input = make_i64_tensor(&[], vec![i64::MAX]); // 0x7FFFFFFFFFFFFFFF
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "clz(i64::MAX) = 1");
}

#[test]
fn oracle_clz_min_i64() {
    let input = make_i64_tensor(&[], vec![i64::MIN]); // 0x8000000000000000
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "clz(i64::MIN) = 0");
}

// ====================== SPECIFIC BIT PATTERNS ======================

#[test]
fn oracle_clz_high_nibble() {
    // 0x0F = 00001111 in lowest byte, but for i64 it's 60 leading zeros
    let input = make_i64_tensor(&[], vec![0x0F]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 60, "clz(0x0F) = 60");
}

#[test]
fn oracle_clz_alternating_bits() {
    // 0x5555555555555555 = 0101... pattern, MSB is 0
    let input = make_u64_tensor(&[], vec![0x5555555555555555]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 1, "clz(0x5555...) = 1");

    // 0xAAAAAAAAAAAAAAAA = 1010... pattern, MSB is 1
    let input = make_u64_tensor(&[], vec![0xAAAAAAAAAAAAAAAA]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 0, "clz(0xAAAA...) = 0");
}

#[test]
fn oracle_clz_sparse_high_bit() {
    // Single bit at various positions
    let input = make_u64_tensor(&[], vec![0x0080000000000000]); // bit 55 set
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 8, "clz(1<<55) = 8");
}

// ====================== NEGATIVE NUMBERS ======================

#[test]
fn oracle_clz_small_negative() {
    // -1 = all bits set
    let input = make_i64_tensor(&[], vec![-1]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "clz(-1i64) = 0");
}

#[test]
fn oracle_clz_negative_two() {
    // -2 = ...11111110, MSB is 1
    let input = make_i64_tensor(&[], vec![-2]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "clz(-2i64) = 0");
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_clz_1d_i64() {
    let input = make_i64_tensor(&[5], vec![0, 1, 2, 4, 8]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![64, 63, 62, 61, 60]);
}

#[test]
fn oracle_clz_1d_u64() {
    let input = make_u64_tensor(&[4], vec![1, 256, 65536, 1 << 32]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_u64_vec(&result), vec![63, 55, 47, 31]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_clz_2d_i64() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 4, 8, 16, 32]);
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![63, 62, 61, 60, 59, 58]);
}

// ====================== MATHEMATICAL PROPERTIES ======================

#[test]
fn oracle_clz_floor_log2_relationship() {
    // For x > 0: floor(log2(x)) = 63 - clz(x) for u64
    for exp in 0..63 {
        let val = 1u64 << exp;
        let input = make_u64_tensor(&[], vec![val]);
        let result =
            eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
        let clz = extract_u64_scalar(&result);
        assert_eq!(
            63 - clz,
            exp,
            "floor(log2(2^{})) = 63 - clz(2^{})",
            exp,
            exp
        );
    }
}

#[test]
fn oracle_clz_bit_width_relationship() {
    // clz(x) + floor(log2(x)) + 1 = bit_width for x > 0
    for val in [1u64, 2, 3, 7, 15, 16, 255, 256, 1000, 1 << 30] {
        let input = make_u64_tensor(&[], vec![val]);
        let result =
            eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
        let clz = extract_u64_scalar(&result);
        let floor_log2 = 63 - val.leading_zeros() as u64;
        assert_eq!(
            clz + floor_log2 + 1,
            64,
            "clz({}) + floor_log2({}) + 1 = 64",
            val,
            val
        );
    }
}

#[test]
fn oracle_clz_consecutive_values() {
    // clz(n) <= clz(n+1) for all n >= 0 (except when crossing power of 2)
    let values: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 63, 64];
    let input = make_u64_tensor(&[values.len() as u32], values.clone());
    let result =
        eval_primitive(Primitive::CountLeadingZeros, &[input], &no_params()).unwrap();
    let clz_vals = extract_u64_vec(&result);

    for (val, clz) in values.iter().zip(clz_vals.iter()) {
        let expected = val.leading_zeros() as u64;
        assert_eq!(
            *clz, expected,
            "clz({}) = {} (expected {})",
            val, clz, expected
        );
    }
}
