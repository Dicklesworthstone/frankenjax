//! Oracle tests for Atan2 primitive.
//!
//! atan2(y, x) = angle in radians between positive x-axis and point (x, y)
//!
//! Range: (-π, π]
//!
//! Tests:
//! - Quadrant I: atan2(1, 1) = π/4
//! - Quadrant II: atan2(1, -1) = 3π/4
//! - Quadrant III: atan2(-1, -1) = -3π/4
//! - Quadrant IV: atan2(-1, 1) = -π/4
//! - Axis cases: atan2(0, 1) = 0, atan2(1, 0) = π/2
//! - Zero handling
//! - Infinity cases
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

// ======================== Quadrant I (x > 0, y > 0) ========================

#[test]
fn oracle_atan2_q1_equal() {
    // atan2(1, 1) = π/4
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan2(1, 1)",
    );
}

#[test]
fn oracle_atan2_q1_y_greater() {
    // atan2(2, 1) = atan(2) ≈ 1.107
    let y = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.atan(),
        1e-14,
        "atan2(2, 1)",
    );
}

// ======================== Quadrant II (x < 0, y > 0) ========================

#[test]
fn oracle_atan2_q2_equal() {
    // atan2(1, -1) = 3π/4
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0 * std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan2(1, -1)",
    );
}

// ======================== Quadrant III (x < 0, y < 0) ========================

#[test]
fn oracle_atan2_q3_equal() {
    // atan2(-1, -1) = -3π/4
    let y = make_f64_tensor(&[], vec![-1.0]);
    let x = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -3.0 * std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan2(-1, -1)",
    );
}

// ======================== Quadrant IV (x > 0, y < 0) ========================

#[test]
fn oracle_atan2_q4_equal() {
    // atan2(-1, 1) = -π/4
    let y = make_f64_tensor(&[], vec![-1.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan2(-1, 1)",
    );
}

// ======================== Axis Cases ========================

#[test]
fn oracle_atan2_positive_x_axis() {
    // atan2(0, 1) = 0
    let y = make_f64_tensor(&[], vec![0.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "atan2(0, 1)");
}

#[test]
fn oracle_atan2_negative_x_axis() {
    // atan2(0, -1) = π
    let y = make_f64_tensor(&[], vec![0.0]);
    let x = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::PI,
        1e-14,
        "atan2(0, -1)",
    );
}

#[test]
fn oracle_atan2_positive_y_axis() {
    // atan2(1, 0) = π/2
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan2(1, 0)",
    );
}

#[test]
fn oracle_atan2_negative_y_axis() {
    // atan2(-1, 0) = -π/2
    let y = make_f64_tensor(&[], vec![-1.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan2(-1, 0)",
    );
}

// ======================== Zero Cases ========================

#[test]
fn oracle_atan2_origin_positive_x() {
    // atan2(0, 0) - implementation-defined, often 0
    let y = make_f64_tensor(&[], vec![0.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(!val.is_nan(), "atan2(0, 0) should be defined");
}

// ======================== Infinity Cases ========================

#[test]
fn oracle_atan2_y_inf_x_finite() {
    // atan2(inf, 1) = π/2
    let y = make_f64_tensor(&[], vec![f64::INFINITY]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan2(inf, 1)",
    );
}

#[test]
fn oracle_atan2_y_neg_inf_x_finite() {
    // atan2(-inf, 1) = -π/2
    let y = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan2(-inf, 1)",
    );
}

#[test]
fn oracle_atan2_y_finite_x_inf() {
    // atan2(1, inf) = 0
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "atan2(1, inf)");
}

#[test]
fn oracle_atan2_y_finite_x_neg_inf() {
    // atan2(1, -inf) = π
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::PI,
        1e-14,
        "atan2(1, -inf)",
    );
}

// ======================== NaN Cases ========================

#[test]
fn oracle_atan2_y_nan() {
    let y = make_f64_tensor(&[], vec![f64::NAN]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atan2(NaN, 1) = NaN");
}

#[test]
fn oracle_atan2_x_nan() {
    let y = make_f64_tensor(&[], vec![1.0]);
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atan2(1, NaN) = NaN");
}

// ======================== Range: output in (-π, π] ========================

#[test]
fn oracle_atan2_range() {
    let test_cases = [
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
    ];
    for (y, x) in test_cases {
        let y_t = make_f64_tensor(&[], vec![y]);
        let x_t = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan2, &[y_t, x_t], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            val > -std::f64::consts::PI && val <= std::f64::consts::PI,
            "atan2({}, {}) should be in (-π, π]",
            y,
            x
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_atan2_1d() {
    let y = make_f64_tensor(&[4], vec![1.0, -1.0, 1.0, -1.0]);
    let x = make_f64_tensor(&[4], vec![1.0, 1.0, -1.0, -1.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], std::f64::consts::FRAC_PI_4, 1e-14, "Q1");
    assert_close(vals[1], -std::f64::consts::FRAC_PI_4, 1e-14, "Q4");
    assert_close(vals[2], 3.0 * std::f64::consts::FRAC_PI_4, 1e-14, "Q2");
    assert_close(vals[3], -3.0 * std::f64::consts::FRAC_PI_4, 1e-14, "Q3");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_atan2_2d() {
    let y = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 0.0, -1.0]);
    let x = make_f64_tensor(&[2, 2], vec![1.0, 0.0, -1.0, 0.0]);
    let result = eval_primitive(Primitive::Atan2, &[y, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "positive x-axis");
    assert_close(
        vals[1],
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "positive y-axis",
    );
    assert_close(vals[2], std::f64::consts::PI, 1e-14, "negative x-axis");
    assert_close(
        vals[3],
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "negative y-axis",
    );
}

// ======================== Identity: tan(atan2(y, x)) = y/x for x > 0 ========================

#[test]
fn oracle_atan2_tan_identity() {
    for (y, x) in [(1.0, 2.0), (3.0, 4.0), (1.0, 1.0), (2.0, 1.0)] {
        let y_t = make_f64_tensor(&[], vec![y]);
        let x_t = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan2, &[y_t, x_t], &no_params()).unwrap();
        let angle = extract_f64_scalar(&result);

        let tan_input = make_f64_tensor(&[], vec![angle]);
        let tan_result = eval_primitive(Primitive::Tan, &[tan_input], &no_params()).unwrap();
        let tan_val = extract_f64_scalar(&tan_result);

        assert_close(
            tan_val,
            y / x,
            1e-14,
            &format!("tan(atan2({}, {})) = {}/{}", y, x, y, x),
        );
    }
}

// ======================== METAMORPHIC: scaling invariance ========================

#[test]
fn metamorphic_atan2_scaling_invariant() {
    // atan2(k*y, k*x) = atan2(y, x) for k > 0
    for (y, x) in [(1.0, 2.0), (3.0, 4.0), (-1.0, 2.0), (1.0, -2.0)] {
        let y_t = make_f64_tensor(&[], vec![y]);
        let x_t = make_f64_tensor(&[], vec![x]);
        let base = eval_primitive(Primitive::Atan2, &[y_t, x_t], &no_params()).unwrap();

        for k in [2.0, 10.0, 0.5, 100.0] {
            let ky_t = make_f64_tensor(&[], vec![k * y]);
            let kx_t = make_f64_tensor(&[], vec![k * x]);
            let scaled =
                eval_primitive(Primitive::Atan2, &[ky_t, kx_t], &no_params()).unwrap();

            assert_close(
                extract_f64_scalar(&scaled),
                extract_f64_scalar(&base),
                1e-12,
                &format!("atan2({}*{}, {}*{}) = atan2({}, {})", k, y, k, x, y, x),
            );
        }
    }
}

// ======================== METAMORPHIC: sin/cos relationship ========================

#[test]
fn metamorphic_atan2_sin_cos_div() {
    // sin(atan2(y,x)) / cos(atan2(y,x)) = y/x for x > 0
    // This verifies atan2 via the Sin and Cos primitives
    for (y, x) in [(1.0, 2.0), (3.0, 4.0), (1.0, 1.0), (-2.0, 3.0)] {
        let y_t = make_f64_tensor(&[], vec![y]);
        let x_t = make_f64_tensor(&[], vec![x]);
        let angle = eval_primitive(Primitive::Atan2, &[y_t, x_t], &no_params()).unwrap();

        let sin_angle = eval_primitive(Primitive::Sin, &[angle.clone()], &no_params()).unwrap();
        let cos_angle = eval_primitive(Primitive::Cos, &[angle], &no_params()).unwrap();

        let sin_div_cos =
            extract_f64_scalar(&sin_angle) / extract_f64_scalar(&cos_angle);

        assert_close(
            sin_div_cos,
            y / x,
            1e-12,
            &format!("sin(atan2({},{}))/cos(atan2({},{})) = {}/{}", y, x, y, x, y, x),
        );
    }
}

// ======================== METAMORPHIC: y-negation odd symmetry ========================

#[test]
fn metamorphic_atan2_y_negation() {
    // atan2(-y, x) = -atan2(y, x) for x > 0
    for (y, x) in [(1.0, 2.0), (3.0, 4.0), (2.0, 1.0), (0.5, 3.0)] {
        let y_pos = make_f64_tensor(&[], vec![y]);
        let y_neg = make_f64_tensor(&[], vec![-y]);
        let x_t = make_f64_tensor(&[], vec![x]);

        let result_pos =
            eval_primitive(Primitive::Atan2, &[y_pos, x_t.clone()], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Atan2, &[y_neg, x_t], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&result_neg),
            -extract_f64_scalar(&result_pos),
            1e-12,
            &format!("atan2(-{}, {}) = -atan2({}, {})", y, x, y, x),
        );
    }
}

// ======================== METAMORPHIC: tensor scaling invariance ========================

#[test]
fn metamorphic_atan2_tensor_scaling() {
    let y = make_f64_tensor(&[4], vec![1.0, -1.0, 2.0, -2.0]);
    let x = make_f64_tensor(&[4], vec![2.0, 3.0, 1.0, 4.0]);
    let base = eval_primitive(Primitive::Atan2, &[y.clone(), x.clone()], &no_params()).unwrap();

    // Scale by 5
    let k = 5.0;
    let ky = make_f64_tensor(&[4], vec![k * 1.0, k * -1.0, k * 2.0, k * -2.0]);
    let kx = make_f64_tensor(&[4], vec![k * 2.0, k * 3.0, k * 1.0, k * 4.0]);
    let scaled = eval_primitive(Primitive::Atan2, &[ky, kx], &no_params()).unwrap();

    let base_vals = extract_f64_vec(&base);
    let scaled_vals = extract_f64_vec(&scaled);

    for (b, s) in base_vals.iter().zip(scaled_vals.iter()) {
        assert_close(*s, *b, 1e-12, "atan2(k*y, k*x) = atan2(y, x) element-wise");
    }
}
