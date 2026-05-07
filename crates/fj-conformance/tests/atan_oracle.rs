//! Oracle tests for Atan primitive.
//!
//! atan(x) = arctangent of x, returns angle in radians
//!
//! Domain: (-inf, inf)
//! Range: (-π/2, π/2)
//!
//! Tests:
//! - atan(0) = 0
//! - atan(1) = π/4
//! - atan(-1) = -π/4
//! - atan(inf) = π/2
//! - atan(-inf) = -π/2
//! - atan(NaN) = NaN
//! - Odd function: atan(-x) = -atan(x)
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
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

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_atan_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "atan(0) = 0");
}

#[test]
fn oracle_atan_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan(1) = π/4",
    );
}

#[test]
fn oracle_atan_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan(-1) = -π/4",
    );
}

#[test]
fn oracle_atan_sqrt3() {
    // atan(sqrt(3)) = π/3
    let input = make_f64_tensor(&[], vec![3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_3,
        1e-14,
        "atan(sqrt(3)) = π/3",
    );
}

#[test]
fn oracle_atan_one_over_sqrt3() {
    // atan(1/sqrt(3)) = π/6
    let input = make_f64_tensor(&[], vec![1.0 / 3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_6,
        1e-14,
        "atan(1/sqrt(3)) = π/6",
    );
}

// ====================== INFINITY ======================

#[test]
fn oracle_atan_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan(inf) = π/2",
    );
}

#[test]
fn oracle_atan_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan(-inf) = -π/2",
    );
}

// ====================== NaN ======================

#[test]
fn oracle_atan_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atan(NaN) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_atan_range() {
    // atan(x) should always be in (-π/2, π/2)
    let test_values = [-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0];
    for x in test_values {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            val > -std::f64::consts::FRAC_PI_2 && val < std::f64::consts::FRAC_PI_2,
            "atan({}) = {} should be in (-π/2, π/2)",
            x,
            val
        );
    }
}

// ====================== ODD FUNCTION ======================

#[test]
fn oracle_atan_odd_function() {
    // atan(-x) = -atan(x)
    for x in [0.5, 1.0, 2.0, 10.0, 100.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Atan, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Atan, &[neg_input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&neg_result),
            -extract_f64_scalar(&pos_result),
            1e-14,
            &format!("atan(-{}) = -atan({})", x, x),
        );
    }
}

// ====================== TAN INVERSE ======================

#[test]
fn oracle_atan_tan_inverse() {
    // tan(atan(x)) = x for all x
    for x in [0.0, 0.5, 1.0, 2.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let atan_result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let tan_result = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&tan_result),
            x,
            1e-13,
            &format!("tan(atan({})) = {}", x, x),
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_atan_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, 0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -std::f64::consts::FRAC_PI_4, 1e-14, "atan(-1)");
    assert_eq!(vals[1], 0.0, "atan(0)");
    assert_close(vals[2], std::f64::consts::FRAC_PI_4, 1e-14, "atan(1)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_2, 1e-14, "atan(inf)");
    assert_close(vals[4], -std::f64::consts::FRAC_PI_2, 1e-14, "atan(-inf)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_atan_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0);
    assert_close(vals[1], std::f64::consts::FRAC_PI_4, 1e-14, "atan(1)");
    assert_close(vals[2], -std::f64::consts::FRAC_PI_4, 1e-14, "atan(-1)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_3, 1e-14, "atan(sqrt(3))");
}

// ====================== ASYMPTOTIC BEHAVIOR ======================

#[test]
fn oracle_atan_large_values() {
    // For large |x|, atan(x) approaches ±π/2
    for x in [100.0, 1000.0, 1e10] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val - std::f64::consts::FRAC_PI_2).abs() < 0.01,
            "atan({}) should be close to π/2",
            x
        );
    }
}

#[test]
fn oracle_atan_small_values() {
    // For small |x|, atan(x) ≈ x
    for x in [0.001, 0.0001, 1e-10] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert_close(val, x, x * 0.01, &format!("atan({}) ≈ {}", x, x));
    }
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_atan_monotonic() {
    // atan is strictly increasing
    let values: Vec<f64> = vec![-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "atan should be strictly increasing: atan[{}] = {} > atan[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ======================== METAMORPHIC: tan(atan(x)) = x ========================

#[test]
fn metamorphic_tan_atan_identity() {
    // tan(atan(x)) = x for all real x
    // Use relative tolerance for large values where absolute error grows
    for x in [-100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let atan_result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let tan_atan = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();

        let tol = if x.abs() > 10.0 { 1e-10 } else { 1e-12 };
        assert_close(
            extract_f64_scalar(&tan_atan),
            x,
            tol,
            &format!("tan(atan({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: atan(tan(x)) = x for x in (-π/2, π/2) ========================

#[test]
fn metamorphic_atan_tan_identity() {
    // atan(tan(x)) = x for x in (-π/2, π/2)
    // Use values well within the domain to avoid precision issues near boundaries
    for x in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let tan_result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
        let atan_tan = eval_primitive(Primitive::Atan, &[tan_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&atan_tan),
            x,
            1e-12,
            &format!("atan(tan({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_atan_tensor_roundtrip() {
    let input = make_f64_tensor(&[5], vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let atan_result = eval_primitive(Primitive::Atan, std::slice::from_ref(&input), &no_params()).unwrap();
    let tan_atan = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&tan_atan);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(*rt, *orig, 1e-12, &format!("tan(atan({})) = {}", orig, orig));
    }
}
