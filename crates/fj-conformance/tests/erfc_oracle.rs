//! Oracle tests for Erfc (complementary error function) primitive.
//!
//! erfc(x) = 1 - erf(x) = (2/√π) ∫_{x}^{∞} e^{-t²} dt
//!
//! Properties:
//! - erfc(0) = 1
//! - erfc(x) → 0 as x → +∞
//! - erfc(x) → 2 as x → -∞
//! - erfc(-x) = 2 - erfc(x)
//! - Range: (0, 2) for finite x
//!
//! Tests:
//! - Zero: erfc(0) = 1
//! - Positive/negative values
//! - Large values (asymptotic behavior)
//! - Infinity: erfc(+inf) = 0, erfc(-inf) = 2
//! - NaN propagation
//! - Relationship with erf
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

// Compute reference erfc using erf: erfc(x) = 1 - erf(x)
fn ref_erfc(x: f64) -> f64 {
    1.0 - ref_erf(x)
}

// Approximation of erf using Horner's method (Abramowitz and Stegun approximation)
fn ref_erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// ======================== Zero ========================

#[test]
fn oracle_erfc_zero() {
    // erfc(0) = 1
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "erfc(0)");
}

#[test]
fn oracle_erfc_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "erfc(-0.0)");
}

// ======================== Positive Values ========================

#[test]
fn oracle_erfc_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), ref_erfc(1.0), 1e-14, "erfc(1)");
}

#[test]
fn oracle_erfc_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), ref_erfc(2.0), 1e-14, "erfc(2)");
}

#[test]
fn oracle_erfc_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), ref_erfc(0.5), 1e-14, "erfc(0.5)");
}

// ======================== Negative Values ========================

#[test]
fn oracle_erfc_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        ref_erfc(-1.0),
        1e-14,
        "erfc(-1)",
    );
}

#[test]
fn oracle_erfc_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        ref_erfc(-2.0),
        1e-14,
        "erfc(-2)",
    );
}

#[test]
fn oracle_erfc_neg_half() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        ref_erfc(-0.5),
        1e-14,
        "erfc(-0.5)",
    );
}

// ======================== Large Values (Asymptotic Behavior) ========================

#[test]
fn oracle_erfc_large_positive() {
    // erfc(5) ≈ 0 (very small)
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val < 1e-10, "erfc(5) should be very small");
    assert!(val >= 0.0, "erfc(5) >= 0");
}

#[test]
fn oracle_erfc_large_negative() {
    // erfc(-5) ≈ 2 (very close to 2)
    let input = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 2.0, 1e-10, "erfc(-5) ≈ 2");
}

#[test]
fn oracle_erfc_very_large_positive() {
    let input = make_f64_tensor(&[], vec![30.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "erfc(30) = 0");
}

#[test]
fn oracle_erfc_very_large_negative() {
    let input = make_f64_tensor(&[], vec![-30.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "erfc(-30) = 2");
}

// ======================== Infinity ========================

#[test]
fn oracle_erfc_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "erfc(+inf) = 0");
}

#[test]
fn oracle_erfc_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "erfc(-inf) = 2");
}

// ======================== NaN ========================

#[test]
fn oracle_erfc_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "erfc(NaN) = NaN");
}

// ======================== Bounds: output in (0, 2) for finite x ========================

#[test]
fn oracle_erfc_bounds_positive() {
    for x in [0.1, 0.5, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > 0.0, "erfc({}) > 0", x);
        assert!(val < 1.0, "erfc({}) < 1 for positive x", x);
    }
}

#[test]
fn oracle_erfc_bounds_negative() {
    for x in [-0.1, -0.5, -1.0, -2.0, -3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > 1.0, "erfc({}) > 1 for negative x", x);
        assert!(val < 2.0, "erfc({}) < 2", x);
    }
}

// ======================== Symmetry: erfc(-x) = 2 - erfc(x) ========================

#[test]
fn oracle_erfc_symmetry() {
    for x in [0.5, 1.0, 2.0, 3.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Erfc, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Erfc, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            2.0 - val_pos,
            1e-14,
            &format!("erfc(-{}) = 2 - erfc({})", x, x),
        );
    }
}

// ======================== Relationship: erfc(x) = 1 - erf(x) ========================

#[test]
fn oracle_erfc_erf_relationship() {
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let erfc_result = eval_primitive(Primitive::Erfc, &[input.clone()], &no_params()).unwrap();
        let erf_result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();

        let erfc_val = extract_f64_scalar(&erfc_result);
        let erf_val = extract_f64_scalar(&erf_result);

        assert_close(
            erfc_val,
            1.0 - erf_val,
            1e-14,
            &format!("erfc({}) = 1 - erf({})", x, x),
        );
    }
}

// ======================== Monotonicity: erfc is decreasing ========================

#[test]
fn oracle_erfc_monotonic_decreasing() {
    let inputs = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] < vals[i - 1],
            "erfc should be monotonically decreasing"
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_erfc_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], ref_erfc(-2.0), 1e-14, "erfc(-2)");
    assert_close(vals[1], ref_erfc(-1.0), 1e-14, "erfc(-1)");
    assert_close(vals[2], 1.0, 1e-14, "erfc(0)");
    assert_close(vals[3], ref_erfc(1.0), 1e-14, "erfc(1)");
    assert_close(vals[4], ref_erfc(2.0), 1e-14, "erfc(2)");
}

#[test]
fn oracle_erfc_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 1.0, 1e-14, "erfc(0)");
    assert_eq!(vals[1], 0.0, "erfc(+inf)");
    assert_eq!(vals[2], 2.0, "erfc(-inf)");
    assert!(vals[3].is_nan(), "erfc(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_erfc_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], ref_erfc(-2.0), 1e-14, "erfc(-2)");
    assert_close(vals[2], 1.0, 1e-14, "erfc(0)");
    assert_close(vals[5], ref_erfc(3.0), 1e-14, "erfc(3)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_erfc_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Erfc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], ref_erfc(-3.0), 1e-14, "erfc(-3)");
    assert_close(vals[3], 1.0, 1e-14, "erfc(0)");
    assert_close(vals[7], ref_erfc(4.0), 1e-14, "erfc(4)");
}
