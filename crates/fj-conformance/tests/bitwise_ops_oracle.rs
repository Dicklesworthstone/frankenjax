//! Oracle tests for Bitwise primitives: BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot.
//!
//! These primitives perform elementwise bitwise operations on integers.
//!
//! BitwiseAnd(a, b) = a & b (bitwise AND)
//! BitwiseOr(a, b) = a | b (bitwise OR)
//! BitwiseXor(a, b) = a ^ b (bitwise XOR)
//! BitwiseNot(a) = ~a (bitwise complement)
//!
//! Tests:
//! - Basic bit patterns
//! - Identity and annihilator elements
//! - Mathematical properties
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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ====================== BITWISE NOT TESTS ======================

#[test]
fn oracle_bitwise_not_zero() {
    let a = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1, "~0 = -1 (all ones)");
}

#[test]
fn oracle_bitwise_not_neg_one() {
    let a = make_i64_tensor(&[], vec![-1]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "~(-1) = 0");
}

#[test]
fn oracle_bitwise_not_one() {
    let a = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -2, "~1 = -2");
}

#[test]
fn oracle_bitwise_not_double_negation() {
    for val in [0, 1, -1, 42, -42, 0x55555555i64, i64::MAX, i64::MIN] {
        let a = make_i64_tensor(&[], vec![val]);
        let not_result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        let double_not =
            eval_primitive(Primitive::BitwiseNot, &[not_result], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&double_not), val, "~~x = x");
    }
}

#[test]
fn oracle_bitwise_not_1d() {
    let a = make_i64_tensor(&[4], vec![0, -1, 1, 0xFF]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![-1, 0, -2, !0xFFi64]);
}

#[test]
fn oracle_bitwise_not_2d() {
    let a = make_i64_tensor(&[2, 2], vec![0, 1, 2, 3]);
    let result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -2, -3, -4]);
}

// ====================== BITWISE AND TESTS ======================

#[test]
fn oracle_bitwise_and_basic() {
    // 0b1100 & 0b1010 = 0b1000 = 8
    let a = make_i64_tensor(&[], vec![0b1100]);
    let b = make_i64_tensor(&[], vec![0b1010]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        0b1000,
        "0b1100 & 0b1010 = 0b1000"
    );
}

#[test]
fn oracle_bitwise_and_zero_annihilator() {
    // x & 0 = 0 for all x
    for x in [1, -1, 42, 0xFF, i64::MAX] {
        let a = make_i64_tensor(&[], vec![x]);
        let zero = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::BitwiseAnd, &[a, zero], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 0, "x & 0 = 0");
    }
}

#[test]
fn oracle_bitwise_and_neg_one_identity() {
    // x & -1 = x for all x (since -1 has all bits set)
    for x in [0, 1, -1, 42, 0xFF, i64::MAX, i64::MIN] {
        let a = make_i64_tensor(&[], vec![x]);
        let neg_one = make_i64_tensor(&[], vec![-1]);
        let result = eval_primitive(Primitive::BitwiseAnd, &[a, neg_one], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x, "x & -1 = x");
    }
}

#[test]
fn oracle_bitwise_and_idempotent() {
    // x & x = x
    for x in [0, 1, -1, 42, 0xFF] {
        let a = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::BitwiseAnd, &[a.clone(), a], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x, "x & x = x");
    }
}

#[test]
fn oracle_bitwise_and_commutative() {
    let pairs = [(0b1100, 0b1010), (0xFF, 0xF0), (123, 456)];
    for (a_val, b_val) in pairs {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let ab =
            eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ba = eval_primitive(Primitive::BitwiseAnd, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&ab),
            extract_i64_scalar(&ba),
            "AND is commutative"
        );
    }
}

#[test]
fn oracle_bitwise_and_1d() {
    let a = make_i64_tensor(&[4], vec![0xFF, 0xF0, 0x0F, 0x00]);
    let b = make_i64_tensor(&[4], vec![0xAA, 0xAA, 0xAA, 0xAA]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![0xAA, 0xA0, 0x0A, 0x00]);
}

#[test]
fn oracle_bitwise_and_2d() {
    let a = make_i64_tensor(&[2, 2], vec![0xFF, 0xF0, 0x0F, 0x55]);
    let b = make_i64_tensor(&[2, 2], vec![0x0F, 0x0F, 0x0F, 0xAA]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![0x0F, 0x00, 0x0F, 0x00]);
}

// ====================== BITWISE OR TESTS ======================

#[test]
fn oracle_bitwise_or_basic() {
    // 0b1100 | 0b1010 = 0b1110 = 14
    let a = make_i64_tensor(&[], vec![0b1100]);
    let b = make_i64_tensor(&[], vec![0b1010]);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        0b1110,
        "0b1100 | 0b1010 = 0b1110"
    );
}

#[test]
fn oracle_bitwise_or_zero_identity() {
    // x | 0 = x for all x
    for x in [0, 1, -1, 42, 0xFF] {
        let a = make_i64_tensor(&[], vec![x]);
        let zero = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::BitwiseOr, &[a, zero], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x, "x | 0 = x");
    }
}

#[test]
fn oracle_bitwise_or_neg_one_annihilator() {
    // x | -1 = -1 for all x (since -1 has all bits set)
    for x in [0, 1, -1, 42, 0xFF, i64::MAX, i64::MIN] {
        let a = make_i64_tensor(&[], vec![x]);
        let neg_one = make_i64_tensor(&[], vec![-1]);
        let result = eval_primitive(Primitive::BitwiseOr, &[a, neg_one], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), -1, "x | -1 = -1");
    }
}

#[test]
fn oracle_bitwise_or_idempotent() {
    // x | x = x
    for x in [0, 1, -1, 42, 0xFF] {
        let a = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::BitwiseOr, &[a.clone(), a], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x, "x | x = x");
    }
}

#[test]
fn oracle_bitwise_or_commutative() {
    let pairs = [(0b1100, 0b1010), (0xFF, 0xF0), (123, 456)];
    for (a_val, b_val) in pairs {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let ab =
            eval_primitive(Primitive::BitwiseOr, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ba = eval_primitive(Primitive::BitwiseOr, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&ab),
            extract_i64_scalar(&ba),
            "OR is commutative"
        );
    }
}

#[test]
fn oracle_bitwise_or_1d() {
    let a = make_i64_tensor(&[4], vec![0xF0, 0x0F, 0x00, 0x55]);
    let b = make_i64_tensor(&[4], vec![0x0F, 0xF0, 0xFF, 0xAA]);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![0xFF, 0xFF, 0xFF, 0xFF]);
}

// ====================== BITWISE XOR TESTS ======================

#[test]
fn oracle_bitwise_xor_basic() {
    // 0b1100 ^ 0b1010 = 0b0110 = 6
    let a = make_i64_tensor(&[], vec![0b1100]);
    let b = make_i64_tensor(&[], vec![0b1010]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(
        extract_i64_scalar(&result),
        0b0110,
        "0b1100 ^ 0b1010 = 0b0110"
    );
}

#[test]
fn oracle_bitwise_xor_zero_identity() {
    // x ^ 0 = x for all x
    for x in [0, 1, -1, 42, 0xFF] {
        let a = make_i64_tensor(&[], vec![x]);
        let zero = make_i64_tensor(&[], vec![0]);
        let result = eval_primitive(Primitive::BitwiseXor, &[a, zero], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), x, "x ^ 0 = x");
    }
}

#[test]
fn oracle_bitwise_xor_self_cancel() {
    // x ^ x = 0 for all x
    for x in [0, 1, -1, 42, 0xFF, i64::MAX, i64::MIN] {
        let a = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::BitwiseXor, &[a.clone(), a], &no_params()).unwrap();
        assert_eq!(extract_i64_scalar(&result), 0, "x ^ x = 0");
    }
}

#[test]
fn oracle_bitwise_xor_neg_one_complement() {
    // x ^ -1 = ~x (XOR with all ones is complement)
    for x in [0, 1, 42, 0xFF] {
        let a = make_i64_tensor(&[], vec![x]);
        let neg_one = make_i64_tensor(&[], vec![-1]);
        let xor_result =
            eval_primitive(Primitive::BitwiseXor, &[a.clone(), neg_one], &no_params()).unwrap();
        let not_result = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&xor_result),
            extract_i64_scalar(&not_result),
            "x ^ -1 = ~x"
        );
    }
}

#[test]
fn oracle_bitwise_xor_commutative() {
    let pairs = [(0b1100, 0b1010), (0xFF, 0xF0), (123, 456)];
    for (a_val, b_val) in pairs {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let ab =
            eval_primitive(Primitive::BitwiseXor, &[a.clone(), b.clone()], &no_params()).unwrap();
        let ba = eval_primitive(Primitive::BitwiseXor, &[b, a], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&ab),
            extract_i64_scalar(&ba),
            "XOR is commutative"
        );
    }
}

#[test]
fn oracle_bitwise_xor_1d() {
    let a = make_i64_tensor(&[4], vec![0xFF, 0x0F, 0xF0, 0x55]);
    let b = make_i64_tensor(&[4], vec![0x0F, 0xF0, 0x0F, 0xAA]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![0xF0, 0xFF, 0xFF, 0xFF]);
}

// ====================== DE MORGAN'S LAWS ======================

#[test]
fn oracle_de_morgan_and() {
    // ~(a & b) = (~a) | (~b)
    for (a_val, b_val) in [(0xFF, 0xF0), (0x55, 0xAA), (123, 456)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);

        // ~(a & b)
        let and_result =
            eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b.clone()], &no_params()).unwrap();
        let not_and = eval_primitive(Primitive::BitwiseNot, &[and_result], &no_params()).unwrap();

        // (~a) | (~b)
        let not_a = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        let not_b = eval_primitive(Primitive::BitwiseNot, &[b], &no_params()).unwrap();
        let or_nots = eval_primitive(Primitive::BitwiseOr, &[not_a, not_b], &no_params()).unwrap();

        assert_eq!(
            extract_i64_scalar(&not_and),
            extract_i64_scalar(&or_nots),
            "De Morgan: ~(a & b) = (~a) | (~b)"
        );
    }
}

#[test]
fn oracle_de_morgan_or() {
    // ~(a | b) = (~a) & (~b)
    for (a_val, b_val) in [(0xFF, 0xF0), (0x55, 0xAA), (123, 456)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);

        // ~(a | b)
        let or_result =
            eval_primitive(Primitive::BitwiseOr, &[a.clone(), b.clone()], &no_params()).unwrap();
        let not_or = eval_primitive(Primitive::BitwiseNot, &[or_result], &no_params()).unwrap();

        // (~a) & (~b)
        let not_a = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        let not_b = eval_primitive(Primitive::BitwiseNot, &[b], &no_params()).unwrap();
        let and_nots =
            eval_primitive(Primitive::BitwiseAnd, &[not_a, not_b], &no_params()).unwrap();

        assert_eq!(
            extract_i64_scalar(&not_or),
            extract_i64_scalar(&and_nots),
            "De Morgan: ~(a | b) = (~a) & (~b)"
        );
    }
}

// ====================== DISTRIBUTIVE LAWS ======================

#[test]
fn oracle_and_distributes_over_or() {
    // a & (b | c) = (a & b) | (a & c)
    for (a_val, b_val, c_val) in [(0xFF, 0xF0, 0x0F), (0x55, 0xAA, 0x33), (123, 456, 789)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let c = make_i64_tensor(&[], vec![c_val]);

        // a & (b | c)
        let b_or_c =
            eval_primitive(Primitive::BitwiseOr, &[b.clone(), c.clone()], &no_params()).unwrap();
        let lhs =
            eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b_or_c], &no_params()).unwrap();

        // (a & b) | (a & c)
        let a_and_b = eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b], &no_params()).unwrap();
        let a_and_c = eval_primitive(Primitive::BitwiseAnd, &[a, c], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::BitwiseOr, &[a_and_b, a_and_c], &no_params()).unwrap();

        assert_eq!(
            extract_i64_scalar(&lhs),
            extract_i64_scalar(&rhs),
            "AND distributes over OR"
        );
    }
}

#[test]
fn oracle_or_distributes_over_and() {
    // a | (b & c) = (a | b) & (a | c)
    for (a_val, b_val, c_val) in [(0xFF, 0xF0, 0x0F), (0x55, 0xAA, 0x33), (123, 456, 789)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let c = make_i64_tensor(&[], vec![c_val]);

        // a | (b & c)
        let b_and_c =
            eval_primitive(Primitive::BitwiseAnd, &[b.clone(), c.clone()], &no_params()).unwrap();
        let lhs =
            eval_primitive(Primitive::BitwiseOr, &[a.clone(), b_and_c], &no_params()).unwrap();

        // (a | b) & (a | c)
        let a_or_b = eval_primitive(Primitive::BitwiseOr, &[a.clone(), b], &no_params()).unwrap();
        let a_or_c = eval_primitive(Primitive::BitwiseOr, &[a, c], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::BitwiseAnd, &[a_or_b, a_or_c], &no_params()).unwrap();

        assert_eq!(
            extract_i64_scalar(&lhs),
            extract_i64_scalar(&rhs),
            "OR distributes over AND"
        );
    }
}

// ====================== ABSORPTION LAWS ======================

#[test]
fn oracle_absorption_and_or() {
    // a & (a | b) = a
    for (a_val, b_val) in [(0xFF, 0xF0), (0x55, 0xAA), (0, 1), (42, 0)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);

        let a_or_b = eval_primitive(Primitive::BitwiseOr, &[a.clone(), b], &no_params()).unwrap();
        let result =
            eval_primitive(Primitive::BitwiseAnd, &[a.clone(), a_or_b], &no_params()).unwrap();

        assert_eq!(extract_i64_scalar(&result), a_val, "a & (a | b) = a");
    }
}

#[test]
fn oracle_absorption_or_and() {
    // a | (a & b) = a
    for (a_val, b_val) in [(0xFF, 0xF0), (0x55, 0xAA), (0, 1), (42, 0)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);

        let a_and_b = eval_primitive(Primitive::BitwiseAnd, &[a.clone(), b], &no_params()).unwrap();
        let result =
            eval_primitive(Primitive::BitwiseOr, &[a.clone(), a_and_b], &no_params()).unwrap();

        assert_eq!(extract_i64_scalar(&result), a_val, "a | (a & b) = a");
    }
}

// ====================== XOR ASSOCIATIVITY ======================

#[test]
fn oracle_xor_associative() {
    // (a ^ b) ^ c = a ^ (b ^ c)
    for (a_val, b_val, c_val) in [(0xFF, 0xF0, 0x0F), (0x55, 0xAA, 0x33), (123, 456, 789)] {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let c = make_i64_tensor(&[], vec![c_val]);

        // (a ^ b) ^ c
        let a_xor_b =
            eval_primitive(Primitive::BitwiseXor, &[a.clone(), b.clone()], &no_params()).unwrap();
        let lhs =
            eval_primitive(Primitive::BitwiseXor, &[a_xor_b, c.clone()], &no_params()).unwrap();

        // a ^ (b ^ c)
        let b_xor_c = eval_primitive(Primitive::BitwiseXor, &[b, c], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::BitwiseXor, &[a, b_xor_c], &no_params()).unwrap();

        assert_eq!(
            extract_i64_scalar(&lhs),
            extract_i64_scalar(&rhs),
            "XOR is associative"
        );
    }
}

// ====================== SPECIFIC BIT PATTERNS ======================

#[test]
fn oracle_and_mask() {
    // Mask off lower 4 bits
    let a = make_i64_tensor(&[], vec![0xABCD]);
    let mask = make_i64_tensor(&[], vec![0x0F]);
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, mask], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0x0D, "lower 4 bits of 0xABCD");
}

#[test]
fn oracle_or_set_bits() {
    // Set bits 4-7
    let a = make_i64_tensor(&[], vec![0x05]);
    let bits = make_i64_tensor(&[], vec![0xF0]);
    let result = eval_primitive(Primitive::BitwiseOr, &[a, bits], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0xF5, "set bits 4-7");
}

#[test]
fn oracle_xor_toggle_bits() {
    // Toggle bits 0 and 2
    let a = make_i64_tensor(&[], vec![0b1010]);
    let toggle = make_i64_tensor(&[], vec![0b0101]);
    let result = eval_primitive(Primitive::BitwiseXor, &[a, toggle], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0b1111, "toggle bits");
}

// ====================== 3D TENSOR ======================

#[test]
fn oracle_bitwise_and_3d() {
    let a = make_i64_tensor(
        &[2, 2, 2],
        vec![0xFF, 0xF0, 0x0F, 0x00, 0xAA, 0x55, 0x33, 0xCC],
    );
    let b = make_i64_tensor(
        &[2, 2, 2],
        vec![0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F],
    );
    let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(
        extract_i64_vec(&result),
        vec![0x0F, 0x00, 0x0F, 0x00, 0x0A, 0x05, 0x03, 0x0C]
    );
}

// ======================== METAMORPHIC: BitwiseNot(BitwiseNot(x)) = x ========================

#[test]
fn metamorphic_bitwise_not_involution() {
    // BitwiseNot(BitwiseNot(x)) = x (double negation is identity)
    for x in [0i64, 1, -1, 0xFF, 0xABCD, i64::MAX, i64::MIN] {
        let x_val = make_i64_tensor(&[], vec![x]);
        let not_x = eval_primitive(Primitive::BitwiseNot, &[x_val.clone()], &no_params()).unwrap();
        let not_not_x = eval_primitive(Primitive::BitwiseNot, &[not_x], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&not_not_x),
            x,
            "BitwiseNot(BitwiseNot({})) = {}",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: BitwiseXor(x, x) = 0 ========================

#[test]
fn metamorphic_bitwise_xor_self_zero() {
    // BitwiseXor(x, x) = 0 (XOR with self is always zero)
    for x in [0i64, 1, -1, 0xFF, 0xABCD, i64::MAX, i64::MIN] {
        let x_val = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::BitwiseXor, &[x_val.clone(), x_val], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            0,
            "BitwiseXor({}, {}) = 0",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: BitwiseAnd(x, BitwiseNot(x)) = 0 ========================

#[test]
fn metamorphic_bitwise_and_complement_zero() {
    // BitwiseAnd(x, BitwiseNot(x)) = 0 (AND with complement is zero)
    for x in [0i64, 1, -1, 0xFF, 0xABCD, i64::MAX, i64::MIN] {
        let x_val = make_i64_tensor(&[], vec![x]);
        let not_x = eval_primitive(Primitive::BitwiseNot, &[x_val.clone()], &no_params()).unwrap();
        let result = eval_primitive(Primitive::BitwiseAnd, &[x_val, not_x], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            0,
            "BitwiseAnd({}, BitwiseNot({})) = 0",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: BitwiseOr(x, BitwiseNot(x)) = -1 ========================

#[test]
fn metamorphic_bitwise_or_complement_all_ones() {
    // BitwiseOr(x, BitwiseNot(x)) = -1 (all ones in two's complement)
    for x in [0i64, 1, -1, 0xFF, 0xABCD, i64::MAX, i64::MIN] {
        let x_val = make_i64_tensor(&[], vec![x]);
        let not_x = eval_primitive(Primitive::BitwiseNot, &[x_val.clone()], &no_params()).unwrap();
        let result = eval_primitive(Primitive::BitwiseOr, &[x_val, not_x], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            -1,
            "BitwiseOr({}, BitwiseNot({})) = -1",
            x,
            x
        );
    }
}
