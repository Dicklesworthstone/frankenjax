//! Oracle tests for Reciprocal primitive.
//!
//! reciprocal(x) = 1/x
//!
//! Tests:
//! - Basic: reciprocal(2) = 0.5, reciprocal(4) = 0.25
//! - Zero: reciprocal(0) = +infinity, reciprocal(-0) = -infinity
//! - Negative: reciprocal(-x) = -reciprocal(x)
//! - Infinity: reciprocal(inf) = 0
//! - NaN propagation
//! - Identity: reciprocal(reciprocal(x)) = x

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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff={}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ======================== Basic Positive Values ========================

#[test]
fn oracle_reciprocal_one() {
    // reciprocal(1) = 1
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "reciprocal(1)");
}

#[test]
fn oracle_reciprocal_two() {
    // reciprocal(2) = 0.5
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "reciprocal(2)");
}

#[test]
fn oracle_reciprocal_four() {
    // reciprocal(4) = 0.25
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "reciprocal(4)");
}

#[test]
fn oracle_reciprocal_ten() {
    // reciprocal(10) = 0.1
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.1, 1e-14, "reciprocal(10)");
}

#[test]
fn oracle_reciprocal_hundred() {
    // reciprocal(100) = 0.01
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.01, 1e-14, "reciprocal(100)");
}

// ======================== Fractions (reciprocal > 1) ========================

#[test]
fn oracle_reciprocal_half() {
    // reciprocal(0.5) = 2
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "reciprocal(0.5)");
}

#[test]
fn oracle_reciprocal_quarter() {
    // reciprocal(0.25) = 4
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 4.0, 1e-14, "reciprocal(0.25)");
}

#[test]
fn oracle_reciprocal_tenth() {
    // reciprocal(0.1) = 10
    let input = make_f64_tensor(&[], vec![0.1]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 10.0, 1e-12, "reciprocal(0.1)");
}

// ======================== Negative Values ========================

#[test]
fn oracle_reciprocal_negative_one() {
    // reciprocal(-1) = -1
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "reciprocal(-1)");
}

#[test]
fn oracle_reciprocal_negative_two() {
    // reciprocal(-2) = -0.5
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -0.5, 1e-14, "reciprocal(-2)");
}

#[test]
fn oracle_reciprocal_negative_half() {
    // reciprocal(-0.5) = -2
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -2.0, 1e-14, "reciprocal(-0.5)");
}

// ======================== Zero (Infinity) ========================

#[test]
fn oracle_reciprocal_positive_zero() {
    // reciprocal(+0) = +infinity
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "reciprocal(+0) = +inf");
}

#[test]
fn oracle_reciprocal_negative_zero() {
    // reciprocal(-0) = -infinity
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "reciprocal(-0) = -inf");
}

// ======================== Infinity ========================

#[test]
fn oracle_reciprocal_positive_infinity() {
    // reciprocal(+inf) = +0
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0, "reciprocal(+inf) = 0");
    assert!(!val.is_sign_negative(), "reciprocal(+inf) should be +0");
}

#[test]
fn oracle_reciprocal_negative_infinity() {
    // reciprocal(-inf) = -0
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0, "reciprocal(-inf) = 0");
    assert!(val.is_sign_negative(), "reciprocal(-inf) should be -0");
}

// ======================== NaN ========================

#[test]
fn oracle_reciprocal_nan() {
    // reciprocal(NaN) = NaN
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "reciprocal(NaN) = NaN"
    );
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_reciprocal_very_small() {
    // reciprocal(1e-100) = 1e100
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e100,
        1e86,
        "reciprocal(1e-100)",
    );
}

#[test]
fn oracle_reciprocal_very_large() {
    // reciprocal(1e100) = 1e-100
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-100,
        1e-114,
        "reciprocal(1e100)",
    );
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_reciprocal_1d() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 4.0, 10.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[1], 0.5, 1e-14, "reciprocal(2)");
    assert_close(vals[2], 0.25, 1e-14, "reciprocal(4)");
    assert_close(vals[3], 0.1, 1e-14, "reciprocal(10)");
}

#[test]
fn oracle_reciprocal_1d_mixed() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.5, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], -0.5, 1e-14, "reciprocal(-2)");
    assert_close(vals[1], -1.0, 1e-14, "reciprocal(-1)");
    assert_close(vals[2], 2.0, 1e-14, "reciprocal(0.5)");
    assert_close(vals[3], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[4], 0.5, 1e-14, "reciprocal(2)");
}

#[test]
fn oracle_reciprocal_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert!(vals[3].is_nan());
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_reciprocal_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 4.0, 5.0, 10.0, 20.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[1], 0.5, 1e-14, "reciprocal(2)");
    assert_close(vals[2], 0.25, 1e-14, "reciprocal(4)");
    assert_close(vals[3], 0.2, 1e-14, "reciprocal(5)");
    assert_close(vals[4], 0.1, 1e-14, "reciprocal(10)");
    assert_close(vals[5], 0.05, 1e-14, "reciprocal(20)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_reciprocal_3d() {
    let input = make_f64_tensor(
        &[2, 2, 2],
        vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    );
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[7], 1.0 / 128.0, 1e-14, "reciprocal(128)");
}

// ======================== Identity: reciprocal(reciprocal(x)) = x ========================

#[test]
fn oracle_reciprocal_double_reciprocal() {
    // reciprocal(reciprocal(x)) = x for non-zero, non-inf values
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, 100.0, -1.0, -0.5] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Reciprocal, &[input1], &no_params()).unwrap();
        let recip1 = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![recip1]);
        let result2 = eval_primitive(Primitive::Reciprocal, &[input2], &no_params()).unwrap();
        let recip2 = extract_f64_scalar(&result2);

        assert_close(
            recip2,
            x,
            1e-12,
            &format!("reciprocal(reciprocal({})) = {}", x, x),
        );
    }
}

// ======================== Identity: x * reciprocal(x) = 1 ========================

#[test]
fn oracle_reciprocal_product_identity() {
    // x * reciprocal(x) = 1 for non-zero values
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, -1.0, -5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let recip = extract_f64_scalar(&result);
        let product = x * recip;
        assert_close(
            product,
            1.0,
            1e-14,
            &format!("{} * reciprocal({}) = 1", x, x),
        );
    }
}

// ======================== Symmetry: reciprocal(-x) = -reciprocal(x) ========================

#[test]
fn oracle_reciprocal_negative_symmetry() {
    for x in [1.0, 2.0, 0.5, 10.0, 100.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Reciprocal, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Reciprocal, &[input_neg], &no_params()).unwrap();

        let recip_pos = extract_f64_scalar(&result_pos);
        let recip_neg = extract_f64_scalar(&result_neg);

        assert_close(
            recip_neg,
            -recip_pos,
            1e-14,
            &format!("reciprocal(-{}) = -reciprocal({})", x, x),
        );
    }
}
