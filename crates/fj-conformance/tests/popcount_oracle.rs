//! Oracle tests for PopulationCount primitive.
//!
//! popcount(x) = number of set bits (1s) in the binary representation
//!
//! Tests:
//! - Zero: popcount(0) = 0
//! - All ones: popcount(0xFF) = 8, popcount(0xFFFFFFFF) = 32
//! - Powers of two: popcount(2^n) = 1
//! - Specific bit patterns
//! - Signed integers (two's complement)
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
fn oracle_popcount_zero_i64() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "popcount(0) = 0");
}

#[test]
fn oracle_popcount_zero_u64() {
    let input = make_u64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 0, "popcount(0) = 0");
}

// ====================== POWERS OF TWO ======================

#[test]
fn oracle_popcount_power_of_two_i64() {
    for exp in 0..63 {
        let val = 1i64 << exp;
        let input = make_i64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 1, "popcount(2^{}) = 1", exp);
    }
}

#[test]
fn oracle_popcount_power_of_two_u64() {
    for exp in 0..64 {
        let val = 1u64 << exp;
        let input = make_u64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
        assert_eq!(extract_u64_scalar(&result), 1, "popcount(2^{}) = 1", exp);
    }
}

// ====================== ALL ONES ======================

#[test]
fn oracle_popcount_all_ones_byte() {
    let input = make_i64_tensor(&[], vec![0xFF]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 8, "popcount(0xFF) = 8");
}

#[test]
fn oracle_popcount_all_ones_u64() {
    let input = make_u64_tensor(&[], vec![0xFFFFFFFFFFFFFFFF]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_u64_scalar(&result),
        64,
        "popcount(0xFFFFFFFFFFFFFFFF) = 64"
    );
}

#[test]
fn oracle_popcount_all_ones_i64() {
    let input = make_i64_tensor(&[], vec![-1i64]); // All 64 bits set
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        64,
        "popcount(-1) = 64 bits set"
    );
}

// ====================== SPECIFIC BIT PATTERNS ======================

#[test]
fn oracle_popcount_alternating_bits() {
    // 0xAA = 10101010 in binary = 4 bits set
    let input = make_i64_tensor(&[], vec![0xAA]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "popcount(0xAA) = 4");

    // 0x55 = 01010101 in binary = 4 bits set
    let input = make_i64_tensor(&[], vec![0x55]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "popcount(0x55) = 4");
}

#[test]
fn oracle_popcount_sparse_bits() {
    // 0x8001 = 1000000000000001 = 2 bits set
    let input = make_i64_tensor(&[], vec![0x8001]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2, "popcount(0x8001) = 2");
}

#[test]
fn oracle_popcount_dense_bits() {
    // 0x0F = 00001111 = 4 bits set
    let input = make_i64_tensor(&[], vec![0x0F]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "popcount(0x0F) = 4");

    // 0xF0 = 11110000 = 4 bits set
    let input = make_i64_tensor(&[], vec![0xF0]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "popcount(0xF0) = 4");
}

// ====================== NEGATIVE NUMBERS (TWO'S COMPLEMENT) ======================

#[test]
fn oracle_popcount_small_negative() {
    // -1 = all bits set (64 bits for i64)
    let input = make_i64_tensor(&[], vec![-1]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 64, "popcount(-1i64) = 64");
}

#[test]
fn oracle_popcount_negative_two() {
    // -2 in two's complement = ...11111110 = 63 bits set for i64
    let input = make_i64_tensor(&[], vec![-2]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 63, "popcount(-2i64) = 63");
}

#[test]
fn oracle_popcount_negative_power_of_two() {
    // -128 = 0xFFFFFFFFFFFFFF80 for i64 = 58 bits set
    let val = -128i64;
    let expected = val.count_ones() as i64;
    let input = make_i64_tensor(&[], vec![val]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        expected,
        "popcount(-128i64) = {}",
        expected
    );
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_popcount_1d_i64() {
    let input = make_i64_tensor(&[5], vec![0, 1, 3, 7, 15]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_popcount_1d_u64() {
    let input = make_u64_tensor(&[4], vec![0, 0xFF, 0xFFFF, 0xFFFFFFFF]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_u64_vec(&result), vec![0, 8, 16, 32]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_popcount_2d_i64() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 4, 8, 16, 32]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1, 1, 1]);
}

#[test]
fn oracle_popcount_2d_mixed_values() {
    let input = make_i64_tensor(&[2, 2], vec![0b111, 0b1111, 0b11111, 0b111111]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 4, 5, 6]);
}

// ====================== MATHEMATICAL PROPERTIES ======================

#[test]
fn oracle_popcount_sum_of_parts() {
    // popcount(a | b) + popcount(a & b) = popcount(a) + popcount(b)
    // When a and b have disjoint bits: popcount(a | b) = popcount(a) + popcount(b)
    let a = 0b11110000i64;
    let b = 0b00001111i64;

    let a_input = make_i64_tensor(&[], vec![a]);
    let b_input = make_i64_tensor(&[], vec![b]);
    let ab_or = make_i64_tensor(&[], vec![a | b]);

    let pa = eval_primitive(Primitive::PopulationCount, &[a_input], &no_params()).unwrap();
    let pb = eval_primitive(Primitive::PopulationCount, &[b_input], &no_params()).unwrap();
    let pab = eval_primitive(Primitive::PopulationCount, &[ab_or], &no_params()).unwrap();

    assert_eq!(
        extract_i64_scalar(&pab),
        extract_i64_scalar(&pa) + extract_i64_scalar(&pb),
        "popcount(a | b) = popcount(a) + popcount(b) for disjoint bits"
    );
}

#[test]
fn oracle_popcount_complementary() {
    // popcount(x) + popcount(~x) = bit_width
    let x = 0x123456789ABCDEFi64;
    let not_x = !x;

    let x_input = make_i64_tensor(&[], vec![x]);
    let not_x_input = make_i64_tensor(&[], vec![not_x]);

    let px = eval_primitive(Primitive::PopulationCount, &[x_input], &no_params()).unwrap();
    let pnot_x = eval_primitive(Primitive::PopulationCount, &[not_x_input], &no_params()).unwrap();

    assert_eq!(
        extract_i64_scalar(&px) + extract_i64_scalar(&pnot_x),
        64,
        "popcount(x) + popcount(~x) = 64 for i64"
    );
}

#[test]
fn oracle_popcount_hamming_weight_identity() {
    // popcount is also known as Hamming weight
    // For small values we can verify directly
    let test_cases = [
        (0i64, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 1),
        (5, 2),
        (6, 2),
        (7, 3),
        (255, 8),
        (256, 1),
    ];

    for (val, expected) in test_cases {
        let input = make_i64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            expected,
            "popcount({}) = {}",
            val,
            expected
        );
    }
}

// ====================== LARGE VALUES ======================

#[test]
fn oracle_popcount_large_i64() {
    let val = i64::MAX; // 0x7FFFFFFFFFFFFFFF = 63 bits set
    let input = make_i64_tensor(&[], vec![val]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 63, "popcount(i64::MAX) = 63");
}

#[test]
fn oracle_popcount_min_i64() {
    let val = i64::MIN; // 0x8000000000000000 = 1 bit set
    let input = make_i64_tensor(&[], vec![val]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "popcount(i64::MIN) = 1");
}

#[test]
fn oracle_popcount_max_u64() {
    let val = u64::MAX;
    let input = make_u64_tensor(&[], vec![val]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(extract_u64_scalar(&result), 64, "popcount(u64::MAX) = 64");
}

// ====================== I32 DTYPE TESTS ======================
// I32 values are stored as Literal::I64 but should be treated as 32-bit

fn make_i32_tensor(shape: &[u32], data: Vec<i32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(|v| Literal::I64(i64::from(v))).collect(),
        )
        .unwrap(),
    )
}

#[test]
fn oracle_popcount_i32_negative_one() {
    // -1 as i32 has 32 bits set, NOT 64 (unlike i64)
    let input = make_i32_tensor(&[], vec![-1i32]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        32,
        "popcount(-1_i32) = 32 (not 64)"
    );
}

#[test]
fn oracle_popcount_i32_max() {
    // i32::MAX = 0x7FFFFFFF = 31 bits set
    let input = make_i32_tensor(&[], vec![i32::MAX]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        31,
        "popcount(i32::MAX) = 31"
    );
}

#[test]
fn oracle_popcount_i32_min() {
    // i32::MIN = 0x80000000 = 1 bit set
    let input = make_i32_tensor(&[], vec![i32::MIN]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        1,
        "popcount(i32::MIN) = 1"
    );
}

#[test]
fn oracle_popcount_i32_tensor() {
    // Test multiple I32 values in a tensor
    let input = make_i32_tensor(&[4], vec![-1i32, 0, 0xFF, i32::MAX]);
    let result = eval_primitive(Primitive::PopulationCount, &[input], &no_params()).unwrap();
    let values = extract_i64_vec(&result);
    assert_eq!(values, vec![32, 0, 8, 31], "I32 tensor popcount");
}

#[test]
fn oracle_popcount_i32_vs_i64_distinguishes() {
    // The same bit pattern interpreted as I32 vs I64 gives different results
    // -1_i32 stored as i64 is 0xFFFFFFFF (sign-extended), but we only count 32 bits
    let i32_input = make_i32_tensor(&[], vec![-1i32]);
    let i64_input = make_i64_tensor(&[], vec![-1i64]);

    let i32_result = eval_primitive(Primitive::PopulationCount, &[i32_input], &no_params()).unwrap();
    let i64_result = eval_primitive(Primitive::PopulationCount, &[i64_input], &no_params()).unwrap();

    assert_eq!(extract_i64_scalar(&i32_result), 32, "I32: 32 bits");
    assert_eq!(extract_i64_scalar(&i64_result), 64, "I64: 64 bits");
}
