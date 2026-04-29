//! Oracle tests for Rsqrt (reciprocal square root) primitive.
//!
//! rsqrt(x) = 1/sqrt(x)
//!
//! Tests:
//! - Perfect squares: rsqrt(4) = 0.5, rsqrt(9) = 1/3
//! - Non-perfect squares
//! - Zero: rsqrt(0) = +infinity
//! - Negative: rsqrt(-x) = NaN
//! - Infinity: rsqrt(inf) = 0
//! - NaN propagation
//! - Tensor shapes

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
        _ => panic!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
        _ => panic!("expected scalar"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
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

// ======================== Perfect Squares ========================

#[test]
fn oracle_rsqrt_one() {
    // rsqrt(1) = 1/sqrt(1) = 1
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "rsqrt(1)");
}

#[test]
fn oracle_rsqrt_four() {
    // rsqrt(4) = 1/sqrt(4) = 0.5
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "rsqrt(4)");
}

#[test]
fn oracle_rsqrt_nine() {
    // rsqrt(9) = 1/sqrt(9) = 1/3
    let input = make_f64_tensor(&[], vec![9.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0 / 3.0, 1e-14, "rsqrt(9)");
}

#[test]
fn oracle_rsqrt_sixteen() {
    // rsqrt(16) = 1/sqrt(16) = 0.25
    let input = make_f64_tensor(&[], vec![16.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "rsqrt(16)");
}

#[test]
fn oracle_rsqrt_hundred() {
    // rsqrt(100) = 1/sqrt(100) = 0.1
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.1, 1e-14, "rsqrt(100)");
}

// ======================== Non-Perfect Squares ========================

#[test]
fn oracle_rsqrt_two() {
    // rsqrt(2) = 1/sqrt(2) ≈ 0.7071067811865476
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 2.0_f64.sqrt(),
        1e-14,
        "rsqrt(2)",
    );
}

#[test]
fn oracle_rsqrt_three() {
    // rsqrt(3) = 1/sqrt(3) ≈ 0.5773502691896257
    let input = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 3.0_f64.sqrt(),
        1e-14,
        "rsqrt(3)",
    );
}

#[test]
fn oracle_rsqrt_five() {
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 5.0_f64.sqrt(),
        1e-14,
        "rsqrt(5)",
    );
}

// ======================== Fractional Values ========================

#[test]
fn oracle_rsqrt_quarter() {
    // rsqrt(0.25) = 1/sqrt(0.25) = 1/0.5 = 2
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "rsqrt(0.25)");
}

#[test]
fn oracle_rsqrt_half() {
    // rsqrt(0.5) = 1/sqrt(0.5) = sqrt(2)
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.sqrt(),
        1e-14,
        "rsqrt(0.5)",
    );
}

#[test]
fn oracle_rsqrt_hundredth() {
    // rsqrt(0.01) = 1/sqrt(0.01) = 10
    let input = make_f64_tensor(&[], vec![0.01]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 10.0, 1e-12, "rsqrt(0.01)");
}

// ======================== Zero and Infinity ========================

#[test]
fn oracle_rsqrt_zero() {
    // rsqrt(0) = 1/0 = +infinity
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "rsqrt(0) should be +inf");
}

#[test]
fn oracle_rsqrt_positive_infinity() {
    // rsqrt(+inf) = 0
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "rsqrt(inf) should be 0");
}

// ======================== Negative Values (NaN) ========================

#[test]
fn oracle_rsqrt_negative_one() {
    // rsqrt(-1) = NaN (complex result)
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-1) should be NaN"
    );
}

#[test]
fn oracle_rsqrt_negative_four() {
    let input = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-4) should be NaN"
    );
}

#[test]
fn oracle_rsqrt_negative_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-inf) should be NaN"
    );
}

// ======================== NaN Propagation ========================

#[test]
fn oracle_rsqrt_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(NaN) should be NaN"
    );
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_rsqrt_very_small() {
    // rsqrt(1e-100) = 1e50
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1e50, 1e35, "rsqrt(1e-100)");
}

#[test]
fn oracle_rsqrt_very_large() {
    // rsqrt(1e100) = 1e-50
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1e-50, 1e-64, "rsqrt(1e100)");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_rsqrt_1d() {
    let input = make_f64_tensor(&[4], vec![1.0, 4.0, 9.0, 16.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "rsqrt(1)");
    assert_close(vals[1], 0.5, 1e-14, "rsqrt(4)");
    assert_close(vals[2], 1.0 / 3.0, 1e-14, "rsqrt(9)");
    assert_close(vals[3], 0.25, 1e-14, "rsqrt(16)");
}

#[test]
fn oracle_rsqrt_1d_mixed() {
    // Mix of positive values and special cases
    let input = make_f64_tensor(&[5], vec![4.0, 0.0, 1.0, -1.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.5, 1e-14, "rsqrt(4)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "rsqrt(0) = +inf");
    assert_close(vals[2], 1.0, 1e-14, "rsqrt(1)");
    assert!(vals[3].is_nan(), "rsqrt(-1) = NaN");
    assert_eq!(vals[4], 0.0, "rsqrt(inf) = 0");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_rsqrt_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "rsqrt(1)");
    assert_close(vals[1], 0.5, 1e-14, "rsqrt(4)");
    assert_close(vals[2], 1.0 / 3.0, 1e-14, "rsqrt(9)");
    assert_close(vals[3], 0.25, 1e-14, "rsqrt(16)");
    assert_close(vals[4], 0.2, 1e-14, "rsqrt(25)");
    assert_close(vals[5], 1.0 / 6.0, 1e-14, "rsqrt(36)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_rsqrt_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[6], 1.0 / 7.0, 1e-14, "rsqrt(49)");
    assert_close(vals[7], 0.125, 1e-14, "rsqrt(64)");
}

// ======================== Identity: rsqrt(x) * sqrt(x) = 1 ========================

#[test]
fn oracle_rsqrt_identity() {
    // Verify: rsqrt(x) = 1/sqrt(x) by checking rsqrt(x) * sqrt(x) ≈ 1
    for x in [2.0, 3.0, 5.0, 7.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        let rsqrt_val = extract_f64_scalar(&result);
        let product = rsqrt_val * x.sqrt();
        assert_close(product, 1.0, 1e-14, &format!("rsqrt({}) * sqrt({}) = 1", x, x));
    }
}

// ======================== Identity: rsqrt(x)^2 * x = 1 ========================

#[test]
fn oracle_rsqrt_squared_identity() {
    // rsqrt(x)^2 = 1/x
    for x in [2.0, 4.0, 9.0, 16.0, 25.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        let rsqrt_val = extract_f64_scalar(&result);
        let squared = rsqrt_val * rsqrt_val;
        assert_close(squared, 1.0 / x, 1e-14, &format!("rsqrt({})^2 = 1/{}", x, x));
    }
}
