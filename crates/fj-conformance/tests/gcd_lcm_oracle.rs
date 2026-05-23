//! Oracle tests for Gcd and Lcm primitives.
//!
//! gcd(a, b) = greatest common divisor
//! lcm(a, b) = least common multiple
//!
//! Tests:
//! - Basic: gcd(12, 8) = 4, lcm(4, 6) = 12
//! - Identity: gcd(n, n) = n, lcm(n, n) = n
//! - One: gcd(n, 1) = 1, lcm(n, 1) = n
//! - Zero: gcd(n, 0) = n, lcm(n, 0) = 0
//! - Negative values
//! - Tensor shapes
//! - Broadcast-compatible operands

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

// ======================== GCD Basic Cases ========================

#[test]
fn oracle_gcd_basic() {
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4, "gcd(12, 8) = 4");
}

#[test]
fn oracle_gcd_coprime() {
    let a = make_i64_tensor(&[], vec![17]);
    let b = make_i64_tensor(&[], vec![13]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "gcd(17, 13) = 1 (coprime)");
}

#[test]
fn oracle_gcd_same() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![15]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15, "gcd(15, 15) = 15");
}

#[test]
fn oracle_gcd_one() {
    let a = make_i64_tensor(&[], vec![42]);
    let b = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1, "gcd(42, 1) = 1");
}

#[test]
fn oracle_gcd_zero() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15, "gcd(15, 0) = 15");
}

#[test]
fn oracle_gcd_both_zero() {
    let a = make_i64_tensor(&[], vec![0]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "gcd(0, 0) = 0");
}

// ======================== GCD Negative Values ========================

#[test]
fn oracle_gcd_negative() {
    let a = make_i64_tensor(&[], vec![-12]);
    let b = make_i64_tensor(&[], vec![8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result).abs(), 4, "gcd(-12, 8) = ±4");
}

#[test]
fn oracle_gcd_both_negative() {
    let a = make_i64_tensor(&[], vec![-12]);
    let b = make_i64_tensor(&[], vec![-8]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result).abs(), 4, "gcd(-12, -8) = ±4");
}

// ======================== LCM Basic Cases ========================

#[test]
fn oracle_lcm_basic() {
    let a = make_i64_tensor(&[], vec![4]);
    let b = make_i64_tensor(&[], vec![6]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 12, "lcm(4, 6) = 12");
}

#[test]
fn oracle_lcm_same() {
    let a = make_i64_tensor(&[], vec![7]);
    let b = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 7, "lcm(7, 7) = 7");
}

#[test]
fn oracle_lcm_one() {
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 12, "lcm(12, 1) = 12");
}

#[test]
fn oracle_lcm_zero() {
    let a = make_i64_tensor(&[], vec![15]);
    let b = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0, "lcm(15, 0) = 0");
}

#[test]
fn oracle_lcm_coprime() {
    let a = make_i64_tensor(&[], vec![5]);
    let b = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 35, "lcm(5, 7) = 35 (coprime)");
}

// ======================== GCD/LCM Relationship ========================

#[test]
fn oracle_gcd_lcm_product() {
    // For positive integers: gcd(a,b) * lcm(a,b) = a * b
    let a = make_i64_tensor(&[], vec![12]);
    let b = make_i64_tensor(&[], vec![18]);

    let gcd_result = eval_primitive(Primitive::Gcd, &[a.clone(), b.clone()], &no_params()).unwrap();
    let lcm_result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    let gcd = extract_i64_scalar(&gcd_result);
    let lcm = extract_i64_scalar(&lcm_result);

    assert_eq!(gcd * lcm, 12 * 18, "gcd(a,b) * lcm(a,b) = a * b");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_gcd_vector() {
    let a = make_i64_tensor(&[4], vec![12, 15, 8, 100]);
    let b = make_i64_tensor(&[4], vec![8, 10, 12, 35]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![4, 5, 4, 5]);
}

#[test]
fn oracle_lcm_vector() {
    let a = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let b = make_i64_tensor(&[4], vec![3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![6, 12, 20, 30]);
}

#[test]
fn oracle_gcd_matrix() {
    let a = make_i64_tensor(&[2, 2], vec![12, 18, 24, 36]);
    let b = make_i64_tensor(&[2, 2], vec![8, 12, 16, 24]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![4, 6, 8, 12]);
}

// ======================== Broadcasting ========================

fn scalar_i64(v: i64) -> Value {
    Value::Scalar(Literal::I64(v))
}

#[test]
fn oracle_gcd_singleton_vector_broadcast() {
    let a = make_i64_tensor(&[1], vec![12]);
    let b = make_i64_tensor(&[3], vec![6, 9, 12]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 3, 12]);
}

#[test]
fn oracle_lcm_singleton_vector_broadcast() {
    let a = make_i64_tensor(&[1], vec![12]);
    let b = make_i64_tensor(&[3], vec![6, 9, 12]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![12, 36, 12]);
}

#[test]
fn oracle_gcd_scalar_tensor_broadcast() {
    let a = scalar_i64(12);
    let b = make_i64_tensor(&[4], vec![6, 8, 9, 12]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![6, 4, 3, 12]);
}

#[test]
fn oracle_gcd_tensor_scalar_broadcast() {
    let a = make_i64_tensor(&[4], vec![6, 8, 9, 12]);
    let b = scalar_i64(12);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![6, 4, 3, 12]);
}

#[test]
fn oracle_lcm_scalar_tensor_broadcast() {
    let a = scalar_i64(6);
    let b = make_i64_tensor(&[3], vec![4, 9, 12]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![12, 18, 12]);
}

#[test]
fn oracle_lcm_tensor_scalar_broadcast() {
    let a = make_i64_tensor(&[3], vec![4, 9, 12]);
    let b = scalar_i64(6);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![12, 18, 12]);
}

#[test]
fn oracle_gcd_row_vector_broadcast() {
    // [1, 3] gcd [2, 3] -> [2, 3]
    let a = make_i64_tensor(&[1, 3], vec![12, 18, 24]);
    let b = make_i64_tensor(&[2, 3], vec![6, 9, 12, 8, 12, 16]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 9, 12, 4, 6, 8]);
}

#[test]
fn oracle_gcd_column_vector_broadcast() {
    // [2, 1] gcd [2, 3] -> [2, 3]
    let a = make_i64_tensor(&[2, 1], vec![12, 18]);
    let b = make_i64_tensor(&[2, 3], vec![6, 8, 9, 9, 12, 15]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 4, 3, 9, 6, 3]);
}

#[test]
fn oracle_lcm_row_vector_broadcast() {
    // [1, 3] lcm [2, 3] -> [2, 3]
    let a = make_i64_tensor(&[1, 3], vec![2, 3, 4]);
    let b = make_i64_tensor(&[2, 3], vec![3, 4, 5, 5, 6, 7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 12, 20, 10, 6, 28]);
}

#[test]
fn oracle_gcd_different_ranks_broadcast() {
    // [3] gcd [2, 3] -> [2, 3] (1D broadcast against 2D)
    let a = make_i64_tensor(&[3], vec![12, 18, 24]);
    let b = make_i64_tensor(&[2, 3], vec![6, 9, 12, 8, 12, 16]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 9, 12, 4, 6, 8]);
}

#[test]
fn oracle_lcm_different_ranks_broadcast() {
    // [3] lcm [2, 3] -> [2, 3]
    let a = make_i64_tensor(&[3], vec![2, 3, 4]);
    let b = make_i64_tensor(&[2, 3], vec![3, 4, 5, 5, 6, 7]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![6, 12, 20, 10, 6, 28]);
}

#[test]
fn oracle_gcd_incompatible_shapes_error() {
    // [2] gcd [3] should error
    let a = make_i64_tensor(&[2], vec![12, 18]);
    let b = make_i64_tensor(&[3], vec![6, 9, 12]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_lcm_incompatible_shapes_error() {
    // [2] lcm [3] should error
    let a = make_i64_tensor(&[2], vec![2, 3]);
    let b = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_gcd_preserves_int_dtype() {
    let a = make_i64_tensor(&[3], vec![12, 18, 24]);
    let b = make_i64_tensor(&[3], vec![6, 9, 12]);
    let result = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::I64, "gcd should preserve I64 dtype");
}

#[test]
fn property_lcm_preserves_int_dtype() {
    let a = make_i64_tensor(&[3], vec![2, 3, 4]);
    let b = make_i64_tensor(&[3], vec![3, 4, 5]);
    let result = eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::I64, "lcm should preserve I64 dtype");
}

// ======================== METAMORPHIC: mathematical identities ========================

#[test]
fn metamorphic_gcd_commutativity() {
    // gcd(a, b) = gcd(b, a)
    let a = make_i64_tensor(&[5], vec![12, 15, 24, 35, 100]);
    let b = make_i64_tensor(&[5], vec![8, 10, 18, 14, 25]);
    let gcd_ab = eval_primitive(Primitive::Gcd, &[a.clone(), b.clone()], &no_params()).unwrap();
    let gcd_ba = eval_primitive(Primitive::Gcd, &[b, a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&gcd_ab), extract_i64_vec(&gcd_ba), "gcd(a, b) should equal gcd(b, a)");
}

#[test]
fn metamorphic_lcm_commutativity() {
    // lcm(a, b) = lcm(b, a)
    let a = make_i64_tensor(&[5], vec![12, 15, 24, 35, 100]);
    let b = make_i64_tensor(&[5], vec![8, 10, 18, 14, 25]);
    let lcm_ab = eval_primitive(Primitive::Lcm, &[a.clone(), b.clone()], &no_params()).unwrap();
    let lcm_ba = eval_primitive(Primitive::Lcm, &[b, a], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&lcm_ab), extract_i64_vec(&lcm_ba), "lcm(a, b) should equal lcm(b, a)");
}

#[test]
fn metamorphic_gcd_associativity() {
    // gcd(a, gcd(b, c)) = gcd(gcd(a, b), c)
    let a = make_i64_tensor(&[4], vec![24, 36, 48, 60]);
    let b = make_i64_tensor(&[4], vec![18, 24, 36, 45]);
    let c = make_i64_tensor(&[4], vec![12, 18, 24, 30]);

    let gcd_bc = eval_primitive(Primitive::Gcd, &[b.clone(), c.clone()], &no_params()).unwrap();
    let gcd_a_bc = eval_primitive(Primitive::Gcd, &[a.clone(), gcd_bc], &no_params()).unwrap();

    let gcd_ab = eval_primitive(Primitive::Gcd, &[a, b], &no_params()).unwrap();
    let gcd_ab_c = eval_primitive(Primitive::Gcd, &[gcd_ab, c], &no_params()).unwrap();

    assert_eq!(
        extract_i64_vec(&gcd_a_bc),
        extract_i64_vec(&gcd_ab_c),
        "gcd(a, gcd(b, c)) should equal gcd(gcd(a, b), c)"
    );
}

#[test]
fn metamorphic_gcd_lcm_product_relation() {
    // gcd(a, b) * lcm(a, b) = |a * b| for multiple pairs
    let pairs: Vec<(i64, i64)> = vec![(12, 8), (15, 10), (24, 36), (7, 11), (100, 25)];
    for (a_val, b_val) in pairs {
        let a = make_i64_tensor(&[], vec![a_val]);
        let b = make_i64_tensor(&[], vec![b_val]);
        let gcd = extract_i64_scalar(&eval_primitive(Primitive::Gcd, &[a.clone(), b.clone()], &no_params()).unwrap());
        let lcm = extract_i64_scalar(&eval_primitive(Primitive::Lcm, &[a, b], &no_params()).unwrap());
        let expected = (a_val * b_val).abs();
        assert_eq!(
            gcd * lcm, expected,
            "gcd({a_val}, {b_val}) * lcm({a_val}, {b_val}) should equal {expected}"
        );
    }
}

#[test]
fn metamorphic_gcd_divisibility() {
    // gcd(a, b) divides both a and b
    let a = make_i64_tensor(&[4], vec![12, 15, 24, 35]);
    let b = make_i64_tensor(&[4], vec![8, 10, 18, 14]);
    let gcd = eval_primitive(Primitive::Gcd, &[a.clone(), b.clone()], &no_params()).unwrap();
    let a_vals = extract_i64_vec(&a);
    let b_vals = extract_i64_vec(&b);
    let gcd_vals = extract_i64_vec(&gcd);
    for (i, (&g, (&a_v, &b_v))) in gcd_vals.iter().zip(a_vals.iter().zip(b_vals.iter())).enumerate() {
        assert_eq!(a_v % g, 0, "gcd should divide a at index {i}");
        assert_eq!(b_v % g, 0, "gcd should divide b at index {i}");
    }
}

#[test]
fn metamorphic_lcm_divisibility() {
    // lcm(a, b) is divisible by both a and b
    let a = make_i64_tensor(&[4], vec![4, 6, 8, 10]);
    let b = make_i64_tensor(&[4], vec![6, 9, 12, 15]);
    let lcm = eval_primitive(Primitive::Lcm, &[a.clone(), b.clone()], &no_params()).unwrap();
    let a_vals = extract_i64_vec(&a);
    let b_vals = extract_i64_vec(&b);
    let lcm_vals = extract_i64_vec(&lcm);
    for (i, (&l, (&a_v, &b_v))) in lcm_vals.iter().zip(a_vals.iter().zip(b_vals.iter())).enumerate() {
        assert_eq!(l % a_v, 0, "lcm should be divisible by a at index {i}");
        assert_eq!(l % b_v, 0, "lcm should be divisible by b at index {i}");
    }
}
