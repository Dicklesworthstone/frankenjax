//! Oracle tests for Ldexp primitive.
//!
//! ldexp(x, n) = x * 2^n
//!
//! Tests:
//! - Basic: ldexp(1, 3) = 8, ldexp(2, 4) = 32
//! - Zero exponent: ldexp(x, 0) = x
//! - Zero value: ldexp(0, n) = 0
//! - Negative exponents: ldexp(8, -3) = 1
//! - Special values: infinity, NaN
//! - Extreme values: signed zero, overflow, and underflow sign preservation
//! - Tensor shapes
//! - Broadcast-compatible operands

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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn assert_same_f64_bits(actual: f64, expected: f64, context: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: expected bits {:016x}, got {:016x}",
        expected.to_bits(),
        actual.to_bits()
    );
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

// ======================== Basic Cases ========================

#[test]
fn oracle_ldexp_basic_1() {
    let x = make_f64_tensor(&[], vec![1.0]);
    let n = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 8.0, "ldexp(1, 3) = 8");
}

#[test]
fn oracle_ldexp_basic_2() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let n = make_i64_tensor(&[], vec![4]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 32.0, "ldexp(2, 4) = 32");
}

#[test]
fn oracle_ldexp_basic_fraction() {
    let x = make_f64_tensor(&[], vec![0.5]);
    let n = make_i64_tensor(&[], vec![2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ldexp(0.5, 2) = 2");
}

// ======================== Zero Cases ========================

#[test]
fn oracle_ldexp_zero_exponent() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let n = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "ldexp(5, 0) = 5");
}

#[test]
fn oracle_ldexp_zero_value() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let n = make_i64_tensor(&[], vec![10]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ldexp(0, 10) = 0");
}

#[test]
fn oracle_ldexp_preserves_signed_zero() {
    let x = make_f64_tensor(&[2], vec![-0.0, 0.0]);
    let n = make_i64_tensor(&[2], vec![10, -10]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    let actual = extract_f64_vec(&result);

    assert_same_f64_bits(actual[0], -0.0, "ldexp(-0, 10)");
    assert_same_f64_bits(actual[1], 0.0, "ldexp(+0, -10)");
}

// ======================== Negative Exponents ========================

#[test]
fn oracle_ldexp_negative_exp() {
    let x = make_f64_tensor(&[], vec![8.0]);
    let n = make_i64_tensor(&[], vec![-3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ldexp(8, -3) = 1");
}

#[test]
fn oracle_ldexp_negative_exp_2() {
    let x = make_f64_tensor(&[], vec![1.0]);
    let n = make_i64_tensor(&[], vec![-2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.25, "ldexp(1, -2) = 0.25");
}

// ======================== Negative Values ========================

#[test]
fn oracle_ldexp_negative_value() {
    let x = make_f64_tensor(&[], vec![-2.0]);
    let n = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -16.0, "ldexp(-2, 3) = -16");
}

// ======================== Special Values ========================

#[test]
fn oracle_ldexp_inf() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let n = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "ldexp(inf, 5) = inf"
    );
}

#[test]
fn oracle_ldexp_neg_inf() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let n = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "ldexp(-inf, 5) = -inf"
    );
}

#[test]
fn oracle_ldexp_nan() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let n = make_i64_tensor(&[], vec![5]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "ldexp(NaN, 5) = NaN");
}

// ======================== Extreme Values ========================

#[test]
fn oracle_ldexp_overflow_preserves_sign() {
    let x = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let n = make_i64_tensor(&[2], vec![1024, 1024]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    let actual = extract_f64_vec(&result);

    assert_eq!(actual[0], f64::INFINITY, "ldexp(1, 1024) = +inf");
    assert_eq!(actual[1], f64::NEG_INFINITY, "ldexp(-1, 1024) = -inf");
}

#[test]
fn oracle_ldexp_underflow_preserves_sign() {
    let x = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let n = make_i64_tensor(&[2], vec![-1075, -1075]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    let actual = extract_f64_vec(&result);

    assert_same_f64_bits(actual[0], 0.0, "ldexp(1, -1075)");
    assert_same_f64_bits(actual[1], -0.0, "ldexp(-1, -1075)");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_ldexp_vector() {
    let x = make_f64_tensor(&[4], vec![1.0, 2.0, 0.5, 4.0]);
    let n = make_i64_tensor(&[4], vec![2, 3, 4, -2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 16.0, 8.0, 1.0]);
}

#[test]
fn oracle_ldexp_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let n = make_i64_tensor(&[2, 2], vec![1, 2, 3, 0]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 8.0, 24.0, 4.0]);
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

fn scalar_i64(v: i64) -> Value {
    Value::Scalar(Literal::I64(v))
}

#[test]
fn oracle_ldexp_scalar_x_tensor_n_broadcast() {
    // scalar mantissa with tensor exponent
    let x = scalar_f64(1.0);
    let n = make_i64_tensor(&[4], vec![0, 1, 2, 3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn oracle_ldexp_tensor_x_scalar_n_broadcast() {
    // tensor mantissa with scalar exponent
    let x = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let n = scalar_i64(2);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 8.0, 12.0, 16.0]);
}

#[test]
fn oracle_ldexp_singleton_x_vector_n_broadcast() {
    // [1] x with [3] n -> [3]
    let x = make_f64_tensor(&[1], vec![1.0]);
    let n = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 4.0, 8.0]);
}

#[test]
fn oracle_ldexp_vector_x_singleton_n_broadcast() {
    // [3] x with [1] n -> [3]
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let n = make_i64_tensor(&[1], vec![2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 8.0, 12.0]);
}

#[test]
fn oracle_ldexp_column_x_matrix_n_broadcast() {
    // [2, 1] x with [2, 3] n -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![1.0, 2.0]);
    let n = make_i64_tensor(&[2, 3], vec![1, 2, 3, 0, 1, 2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: 1.0 * 2^1, 1.0 * 2^2, 1.0 * 2^3 = 2, 4, 8
    assert_eq!(vals[0], 2.0);
    assert_eq!(vals[1], 4.0);
    assert_eq!(vals[2], 8.0);
    // Row 1: 2.0 * 2^0, 2.0 * 2^1, 2.0 * 2^2 = 2, 4, 8
    assert_eq!(vals[3], 2.0);
    assert_eq!(vals[4], 4.0);
    assert_eq!(vals[5], 8.0);
}

#[test]
fn oracle_ldexp_different_ranks_broadcast() {
    // [3] x with [2, 3] n -> [2, 3]
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 4.0]);
    let n = make_i64_tensor(&[2, 3], vec![1, 1, 1, 2, 2, 2]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x * 2^1 = 2, 4, 8
    assert_eq!(vals[0], 2.0);
    assert_eq!(vals[1], 4.0);
    assert_eq!(vals[2], 8.0);
    // Row 1: x * 2^2 = 4, 8, 16
    assert_eq!(vals[3], 4.0);
    assert_eq!(vals[4], 8.0);
    assert_eq!(vals[5], 16.0);
}

#[test]
fn oracle_ldexp_all_scalars_broadcast() {
    // scalar ldexp scalar -> scalar
    let x = scalar_f64(3.0);
    let n = scalar_i64(2);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 12.0);
}

#[test]
fn oracle_ldexp_incompatible_shapes_error() {
    // [2] ldexp [3] should error
    let x = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let n = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_ldexp_vector_scalar_exp_broadcast() {
    let x_values = [1.0, 2.0, 0.5, 4.0];
    let x = make_f64_tensor(&[4], x_values.to_vec());
    let n = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &x_value)) in actual.iter().zip(x_values.iter()).enumerate() {
        let expected = x_value * 2.0_f64.powi(3);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast scalar exponent element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_ldexp_matrix_row_exp_broadcast() {
    let x_values = [1.0, 2.0, 3.0, 4.0];
    let n_values = [1_i32, -1_i32];
    let x = make_f64_tensor(&[2, 2], x_values.to_vec());
    let n = make_i64_tensor(&[2], n_values.iter().copied().map(i64::from).collect());
    let result = eval_primitive(Primitive::Ldexp, &[x, n], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let actual = extract_f64_vec(&result);
    for (i, ((&actual, &x_value), &n_value)) in actual
        .iter()
        .zip(x_values.iter())
        .zip(n_values.iter().cycle())
        .enumerate()
    {
        let expected = x_value * 2.0_f64.powi(n_value);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast row exponent element {i}: expected {expected}, got {actual}"
        );
    }
}
