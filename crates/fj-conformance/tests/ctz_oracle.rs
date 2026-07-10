//! Oracle tests for CountTrailingZeros primitive.
//!
//! ctz(x) = number of trailing zero bits after the last 1 bit
//!
//! Tests:
//! - Zero: ctz(0) = bit_width (all zeros)
//! - One: ctz(1) = 0
//! - Powers of two: ctz(2^n) = n
//! - Odd numbers: ctz(odd) = 0
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
fn oracle_ctz_zero_i64() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 64, "ctz(0i64) = 64");
}

#[test]
fn oracle_ctz_zero_u64() {
    let input = make_u64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 64, "ctz(0u64) = 64");
}

#[test]
fn oracle_ctz_zero_u32() {
    let input = make_u32_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 32, "ctz(0u32) = 32");
}

#[test]
fn oracle_ctz_zero_i32() {
    // i32 is stored as Literal::I64, but ctz must use the LOGICAL 32-bit width:
    // ctz(0i32) = 32, NOT 64. Guards the `dtype == I32` width branch in the bitwise
    // unary eval — a refactor conflating i32 with i64 storage would return 64.
    let input = Value::Tensor(
        TensorValue::new(DType::I32, Shape { dims: vec![] }, vec![Literal::I64(0)]).unwrap(),
    );
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        32,
        "ctz(0i32) = 32, not the i64 width 64"
    );
}

// ====================== SCALAR ONE ======================

#[test]
fn oracle_ctz_one_i64() {
    let input = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(1i64) = 0");
}

#[test]
fn oracle_ctz_one_u64() {
    let input = make_u64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(1u64) = 0");
}

// ====================== POWERS OF TWO ======================

#[test]
fn oracle_ctz_powers_of_two_i64() {
    for exp in 0..63 {
        let val = 1i64 << exp;
        let input = make_i64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), exp, "ctz(2^{}) = {}", exp, exp);
    }
}

#[test]
fn oracle_ctz_powers_of_two_u64() {
    for exp in 0..64 {
        let val = 1u64 << exp;
        let input = make_u64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            exp as i64,
            "ctz(2^{}) = {}",
            exp,
            exp
        );
    }
}

// ====================== ODD NUMBERS ======================

#[test]
fn oracle_ctz_odd_numbers_i64() {
    for odd in [1i64, 3, 5, 7, 9, 11, 127, 255, i64::MAX] {
        let input = make_i64_tensor(&[], vec![odd]);
        let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 0, "ctz({}) = 0 (odd)", odd);
    }
}

// ====================== SPECIFIC BIT PATTERNS ======================

#[test]
fn oracle_ctz_alternating_bits_i64() {
    // 0b...10101010 has ctz = 1
    let input = make_i64_tensor(&[], vec![0xAAAA_AAAA_AAAA_AAAAu64 as i64]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "ctz(0xAAAA...) = 1");

    // 0b...01010101 has ctz = 0
    let input = make_i64_tensor(&[], vec![0x5555_5555_5555_5555u64 as i64]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(0x5555...) = 0");
}

#[test]
fn oracle_ctz_high_bit_set_i64() {
    // Negative number (high bit set) with low bit also set
    let input = make_i64_tensor(&[], vec![-1i64]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(-1) = 0");

    // i64::MIN = 0x8000...0000 has ctz = 63
    let input = make_i64_tensor(&[], vec![i64::MIN]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 63, "ctz(i64::MIN) = 63");
}

#[test]
fn oracle_ctz_max_values() {
    // i64::MAX = 0x7FFF...FFFF has ctz = 0
    let input = make_i64_tensor(&[], vec![i64::MAX]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(i64::MAX) = 0");

    // u64::MAX = 0xFFFF...FFFF has ctz = 0
    let input = make_u64_tensor(&[], vec![u64::MAX]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "ctz(u64::MAX) = 0");
}

// ====================== TENSOR SHAPES ======================

#[test]
fn oracle_ctz_vector_i64() {
    let input = make_i64_tensor(&[4], vec![0, 1, 8, 16]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![64, 0, 3, 4]);
}

#[test]
fn oracle_ctz_matrix_u64() {
    let input = make_u64_tensor(&[2, 3], vec![1, 2, 4, 8, 16, 32]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn oracle_ctz_3d_tensor_i64() {
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 4, 8, 16, 32, 64, 128]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

// ====================== I32 DTYPE HANDLING ======================

#[test]
fn oracle_ctz_i32_via_i64_representation() {
    // When dtype is I32, the value is stored as i64 but should be treated as 32-bit
    // ctz of 0 as i32 should be 32, not 64
    let mut params = BTreeMap::new();
    params.insert("dtype".to_string(), "I32".to_string());

    let input = Value::Tensor(
        TensorValue::new(DType::I32, Shape { dims: vec![] }, vec![Literal::I64(0)]).unwrap(),
    );
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &params).unwrap();
    assert_eq!(extract_i64_scalar(&result), 32, "ctz(0i32) = 32");
}

#[test]
fn oracle_ctz_i32_power_of_two() {
    let input = Value::Tensor(
        TensorValue::new(DType::I32, Shape { dims: vec![] }, vec![Literal::I64(16)]).unwrap(),
    );
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "ctz(16i32) = 4");
}

#[test]
fn oracle_ctz_preserves_dtype_i64() {
    let input = make_i64_tensor(&[2], vec![8, 4]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_ctz_empty_tensor() {
    let input = make_i64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

#[test]
fn oracle_ctz_mixed_values() {
    // Mix of zeros, ones, powers of two, and odd numbers
    let input = make_i64_tensor(&[5], vec![0, 1, 6, 12, 7]);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    // 0 -> 64, 1 -> 0, 6 (0b110) -> 1, 12 (0b1100) -> 2, 7 (0b111) -> 0
    assert_eq!(extract_i64_vec(&result), vec![64, 0, 1, 2, 0]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_ctz_4d_shape() {
    let input = make_i64_tensor(
        &[2, 2, 2, 2],
        vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        ],
    );
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
}

#[test]
fn oracle_ctz_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_ctz_large_tensor() {
    let data: Vec<i64> = (0..100).map(|x| 1i64 << (x % 64)).collect();
    let input = make_i64_tensor(&[100], data);
    let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_i64_vec(&result);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, (i % 64) as i64, "ctz at index {i}");
    }
}

// ======================== PROPERTY: output dtype ========================

#[test]
fn property_ctz_always_outputs_i64() {
    for (dtype, lits) in [
        (
            DType::I32,
            vec![Literal::I64(1), Literal::I64(2), Literal::I64(4)],
        ),
        (
            DType::I64,
            vec![Literal::I64(1), Literal::I64(2), Literal::I64(4)],
        ),
        (
            DType::U32,
            vec![Literal::U32(1), Literal::U32(2), Literal::U32(4)],
        ),
        (
            DType::U64,
            vec![Literal::U64(1), Literal::U64(2), Literal::U64(4)],
        ),
    ] {
        let input = Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap());
        let result = eval_primitive(Primitive::CountTrailingZeros, &[input], &no_params()).unwrap();
        assert_eq!(
            result.dtype(),
            DType::I64,
            "ctz {dtype:?} input: output should always be I64"
        );
    }
}
