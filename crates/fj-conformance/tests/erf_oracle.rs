//! Oracle tests for Erf primitive.
//!
//! erf(x) = Gauss error function = (2/sqrt(π)) * ∫[0,x] exp(-t²) dt
//!
//! Domain: (-inf, inf)
//! Range: (-1, 1)
//!
//! Properties:
//! - erf(0) = 0
//! - erf(inf) = 1
//! - erf(-inf) = -1
//! - Odd function: erf(-x) = -erf(x)
//!
//! Tests:
//! - Special values
//! - Odd function property
//! - Range verification
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
fn oracle_erf_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "erf(0) = 0");
}

#[test]
fn oracle_erf_one() {
    // erf(1) ≈ 0.8427007929497149
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.8427007929497149,
        1e-6,
        "erf(1)",
    );
}

#[test]
fn oracle_erf_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -0.8427007929497149,
        1e-6,
        "erf(-1)",
    );
}

#[test]
fn oracle_erf_two() {
    // erf(2) ≈ 0.9953222650189527
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.9953222650189527,
        1e-6,
        "erf(2)",
    );
}

#[test]
fn oracle_erf_half() {
    // erf(0.5) ≈ 0.5204998778130465
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5204998778130465,
        1e-6,
        "erf(0.5)",
    );
}

// ====================== INFINITY ======================

#[test]
fn oracle_erf_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "erf(inf) = 1");
}

#[test]
fn oracle_erf_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "erf(-inf) = -1");
}

// ====================== NaN ======================

#[test]
fn oracle_erf_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "erf(NaN) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_erf_range() {
    // erf(x) should always be in (-1, 1)
    let test_values = [-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0];
    for x in test_values {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (-1.0..=1.0).contains(&val),
            "erf({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

// ====================== ODD FUNCTION ======================

#[test]
fn oracle_erf_odd_function() {
    // erf(-x) = -erf(x)
    for x in [0.1, 0.5, 1.0, 2.0, 3.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Erf, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Erf, &[neg_input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&neg_result),
            -extract_f64_scalar(&pos_result),
            1e-6,
            &format!("erf(-{}) = -erf({})", x, x),
        );
    }
}

// ====================== ASYMPTOTIC BEHAVIOR ======================

#[test]
fn oracle_erf_large_positive() {
    // For large x, erf(x) → 1
    for x in [5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val - 1.0).abs() < 1e-4,
            "erf({}) should be very close to 1, got {}",
            x,
            val
        );
    }
}

#[test]
fn oracle_erf_large_negative() {
    // For large negative x, erf(x) → -1
    for x in [-5.0, -10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val + 1.0).abs() < 1e-4,
            "erf({}) should be very close to -1, got {}",
            x,
            val
        );
    }
}

#[test]
fn oracle_erf_small_values() {
    // For small |x|, erf(x) ≈ (2/sqrt(π)) * x
    let coeff = 2.0 / std::f64::consts::PI.sqrt();
    for x in [0.001, 0.0001] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = coeff * x;
        assert_close(
            val,
            expected,
            expected.abs() * 0.01,
            &format!("erf({}) ≈ 2x/sqrt(π)", x),
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_erf_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, 0.0, 0.5, 1.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -0.8427007929497149, 1e-6, "erf(-1)");
    assert_eq!(vals[1], 0.0, "erf(0)");
    assert_close(vals[2], 0.5204998778130465, 1e-6, "erf(0.5)");
    assert_close(vals[3], 0.8427007929497149, 1e-6, "erf(1)");
    assert_eq!(vals[4], 1.0, "erf(inf)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_erf_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0);
    assert_close(vals[1], 0.8427007929497149, 1e-6, "erf(1)");
    assert_close(vals[2], -0.8427007929497149, 1e-6, "erf(-1)");
    assert_close(vals[3], 0.9953222650189527, 1e-6, "erf(2)");
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_erf_monotonic() {
    // erf is strictly increasing
    let values: Vec<f64> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "erf should be strictly increasing: erf[{}] = {} > erf[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ====================== ERFC RELATIONSHIP ======================

#[test]
fn oracle_erf_erfc_relationship() {
    // erf(x) + erfc(x) = 1
    for x in [0.0, 0.5, 1.0, 2.0, -0.5, -1.0] {
        let erf_input = make_f64_tensor(&[], vec![x]);
        let erfc_input = make_f64_tensor(&[], vec![x]);
        let erf_result = eval_primitive(Primitive::Erf, &[erf_input], &no_params()).unwrap();
        let erfc_result = eval_primitive(Primitive::Erfc, &[erfc_input], &no_params()).unwrap();
        let erf_val = extract_f64_scalar(&erf_result);
        let erfc_val = extract_f64_scalar(&erfc_result);
        assert_close(
            erf_val + erfc_val,
            1.0,
            1e-6,
            &format!("erf({}) + erfc({}) = 1", x, x),
        );
    }
}

// ======================== METAMORPHIC: erfinv(erf(x)) = x ========================

#[test]
fn metamorphic_erfinv_erf_identity() {
    // erfinv(erf(x)) = x for "moderate" x (erf approaches ±1 asymptotically)
    // Use values where erf(x) is not too close to ±1 for numerical stability
    for x in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5] {
        let input = make_f64_tensor(&[], vec![x]);
        let erf_result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let erfinv_erf = eval_primitive(Primitive::ErfInv, &[erf_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&erfinv_erf),
            x,
            1e-10,
            &format!("erfinv(erf({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: erf(erfinv(y)) = y ========================

#[test]
fn metamorphic_erf_erfinv_identity() {
    // erf(erfinv(y)) = y for y in (-1, 1)
    for y in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![y]);
        let erfinv_result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
        let erf_erfinv = eval_primitive(Primitive::Erf, &[erfinv_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&erf_erfinv),
            y,
            1e-10,
            &format!("erf(erfinv({})) = {}", y, y),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_erf_erfinv_tensor_roundtrip() {
    // For a tensor with values in (-1, 1): erf(erfinv(y)) = y
    let input = make_f64_tensor(&[5], vec![-0.8, -0.4, 0.0, 0.4, 0.8]);
    let erfinv_result = eval_primitive(Primitive::ErfInv, &[input.clone()], &no_params()).unwrap();
    let roundtrip = eval_primitive(Primitive::Erf, &[erfinv_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let recovered = extract_f64_vec(&roundtrip);

    for (orig, rec) in original.iter().zip(recovered.iter()) {
        assert_close(*rec, *orig, 1e-10, &format!("erf(erfinv({})) = {}", orig, orig));
    }
}
