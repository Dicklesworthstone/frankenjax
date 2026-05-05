//! Oracle tests for Round primitive.
//!
//! round(x) = nearest integer to x
//!
//! Uses round-half-away-from-zero for values exactly between two integers
//! (e.g., 0.5 → 1, -0.5 → -1).
//!
//! Tests:
//! - Integers: round(n) = n
//! - Positive fractional parts
//! - Negative fractional parts
//! - Half values (0.5) - banker's rounding
//! - Infinity: round(±inf) = ±inf
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

fn rounding_params(method: &str) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("rounding_method".to_owned(), method.to_owned());
    params
}

// ======================== Integers ========================

#[test]
fn oracle_round_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "round(0) = 0");
}

#[test]
fn oracle_round_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "round(-0.0) = 0");
}

#[test]
fn oracle_round_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "round(1) = 1");
}

#[test]
fn oracle_round_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "round(-1) = -1");
}

#[test]
fn oracle_round_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "round(1000) = 1000");
}

// ======================== Positive Fractional Parts (< 0.5) ========================

#[test]
fn oracle_round_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "round(1.1) = 1");
}

#[test]
fn oracle_round_one_point_four() {
    let input = make_f64_tensor(&[], vec![1.4]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "round(1.4) = 1");
}

#[test]
fn oracle_round_one_point_four_nine() {
    let input = make_f64_tensor(&[], vec![1.49]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "round(1.49) = 1");
}

// ======================== Positive Fractional Parts (> 0.5) ========================

#[test]
fn oracle_round_one_point_six() {
    let input = make_f64_tensor(&[], vec![1.6]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "round(1.6) = 2");
}

#[test]
fn oracle_round_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "round(1.9) = 2");
}

#[test]
fn oracle_round_one_point_five_one() {
    let input = make_f64_tensor(&[], vec![1.51]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "round(1.51) = 2");
}

// ======================== Negative Fractional Parts (< 0.5) ========================

#[test]
fn oracle_round_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "round(-1.1) = -1");
}

#[test]
fn oracle_round_neg_one_point_four() {
    let input = make_f64_tensor(&[], vec![-1.4]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "round(-1.4) = -1");
}

// ======================== Negative Fractional Parts (> 0.5) ========================

#[test]
fn oracle_round_neg_one_point_six() {
    let input = make_f64_tensor(&[], vec![-1.6]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "round(-1.6) = -2");
}

#[test]
fn oracle_round_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "round(-1.9) = -2");
}

// ======================== Half Values - Round Half Away From Zero ========================
// Round half away from zero: 0.5 → 1, 1.5 → 2, 2.5 → 3, -0.5 → -1, etc.

#[test]
fn oracle_round_point_five() {
    // 0.5 rounds to 1 (away from zero)
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "round(0.5) = 1");
}

#[test]
fn oracle_round_one_point_five() {
    // 1.5 rounds to 2
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "round(1.5) = 2");
}

#[test]
fn oracle_round_two_point_five() {
    // 2.5 rounds to 3 (away from zero)
    let input = make_f64_tensor(&[], vec![2.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0, "round(2.5) = 3");
}

#[test]
fn oracle_round_three_point_five() {
    // 3.5 rounds to 4
    let input = make_f64_tensor(&[], vec![3.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "round(3.5) = 4");
}

#[test]
fn oracle_round_neg_point_five() {
    // -0.5 rounds to -1 (away from zero)
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "round(-0.5) = -1");
}

#[test]
fn oracle_round_neg_one_point_five() {
    // -1.5 rounds to -2
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "round(-1.5) = -2");
}

#[test]
fn oracle_round_neg_two_point_five() {
    // -2.5 rounds to -3 (away from zero)
    let input = make_f64_tensor(&[], vec![-2.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -3.0, "round(-2.5) = -3");
}

#[test]
fn oracle_round_to_nearest_even_half_values() {
    let input = make_f64_tensor(&[7], vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]);
    let result = eval_primitive(
        Primitive::Round,
        &[input],
        &rounding_params("TO_NEAREST_EVEN"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![-2.0, -1.0, -0.0, 0.0, 0.0, 1.0, 2.0]);
    assert!(
        vals[2].is_sign_negative(),
        "round(-0.5) should preserve -0.0"
    );
}

#[test]
fn oracle_round_rejects_unknown_rounding_method() {
    let input = make_f64_tensor(&[], vec![2.5]);
    let err = eval_primitive(Primitive::Round, &[input], &rounding_params("HALF_UP"))
        .expect_err("unknown rounding_method should fail");
    assert!(
        err.to_string()
            .contains("unsupported rounding_method 'HALF_UP'"),
        "unexpected error: {err}"
    );
}

// ======================== Infinity ========================

#[test]
fn oracle_round_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "round(+inf) = +inf");
}

#[test]
fn oracle_round_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "round(-inf) = -inf");
}

// ======================== NaN ========================

#[test]
fn oracle_round_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "round(NaN) = NaN");
}

// ======================== Very Small Values ========================

#[test]
fn oracle_round_very_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "round(1e-100) = 0");
}

#[test]
fn oracle_round_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "round(-1e-100) = 0");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_round_1d() {
    let input = make_f64_tensor(&[5], vec![-1.6, -0.5, 0.0, 0.5, 1.6]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], -2.0, "round(-1.6)");
    assert_eq!(vals[1], -1.0, "round(-0.5)");
    assert_eq!(vals[2], 0.0, "round(0)");
    assert_eq!(vals[3], 1.0, "round(0.5)");
    assert_eq!(vals[4], 2.0, "round(1.6)");
}

#[test]
fn oracle_round_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "round(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "round(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "round(-inf)");
    assert!(vals[3].is_nan(), "round(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_round_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 1.5, 1.9, -1.1, -1.5, -1.9]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 1.0, "round(1.1)");
    assert_eq!(vals[1], 2.0, "round(1.5)");
    assert_eq!(vals[2], 2.0, "round(1.9)");
    assert_eq!(vals[3], -1.0, "round(-1.1)");
    assert_eq!(vals[4], -2.0, "round(-1.5)");
    assert_eq!(vals[5], -2.0, "round(-1.9)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_round_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.1, 0.9, 1.5, 2.5, -0.1, -0.9, -1.5, -2.5]);
    let result = eval_primitive(Primitive::Round, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "round(0.1)");
    assert_eq!(vals[1], 1.0, "round(0.9)");
    assert_eq!(vals[2], 2.0, "round(1.5)");
    assert_eq!(vals[3], 3.0, "round(2.5)");
}

// ======================== Idempotency: round(round(x)) = round(x) ========================

#[test]
fn oracle_round_idempotent() {
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Round, &[input1], &no_params()).unwrap();
        let rounded = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![rounded]);
        let result2 = eval_primitive(Primitive::Round, &[input2], &no_params()).unwrap();
        let double_rounded = extract_f64_scalar(&result2);

        assert_eq!(
            rounded, double_rounded,
            "round(round({})) = round({})",
            x, x
        );
    }
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

// ======================== METAMORPHIC: Round(Neg(x)) = Neg(Round(x)) ========================

#[test]
fn metamorphic_round_neg_commutes() {
    // Round(Neg(x)) = Neg(Round(x)) - round commutes with negation
    for x in [0.5, 1.4, 1.5, 1.6, 2.5, 3.7, 0.0, 100.3] {
        let x_val = make_f64_tensor(&[], vec![x]);

        // Round(Neg(x))
        let neg_x = eval_primitive(Primitive::Neg, &[x_val.clone()], &no_params()).unwrap();
        let round_neg = eval_primitive(Primitive::Round, &[neg_x], &no_params()).unwrap();

        // Neg(Round(x))
        let round_x = eval_primitive(Primitive::Round, &[x_val], &no_params()).unwrap();
        let neg_round = eval_primitive(Primitive::Neg, &[round_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&round_neg),
            extract_f64_scalar(&neg_round),
            1e-14,
            &format!("Round(Neg({})) = Neg(Round({}))", x, x),
        );
    }
}

// ======================== METAMORPHIC: Round(x + n) = Round(x) + n for integer n ========================

#[test]
fn metamorphic_round_integer_translation() {
    // Round(x + n) = Round(x) + n for integer n (when x is not exactly at a half-boundary)
    // Note: values ending in .5 require special handling due to round-away-from-zero semantics
    for (x, n) in [(0.3, 5.0), (1.7, 10.0), (0.0, 7.0), (-0.4, 2.0), (2.3, -3.0)] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let n_val = make_f64_tensor(&[], vec![n]);

        // Round(x + n)
        let x_plus_n = eval_primitive(Primitive::Add, &[x_val.clone(), n_val.clone()], &no_params()).unwrap();
        let round_sum = eval_primitive(Primitive::Round, &[x_plus_n], &no_params()).unwrap();

        // Round(x) + n
        let round_x = eval_primitive(Primitive::Round, &[x_val], &no_params()).unwrap();
        let round_plus_n = eval_primitive(Primitive::Add, &[round_x, n_val], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&round_sum),
            extract_f64_scalar(&round_plus_n),
            1e-14,
            &format!("Round({} + {}) = Round({}) + {}", x, n, x, n),
        );
    }
}

// ======================== METAMORPHIC: |Round(x) - x| <= 0.5 ========================

#[test]
fn metamorphic_round_bounded_distance() {
    // |Round(x) - x| <= 0.5 (rounding moves to nearest, ties away from zero)
    for x in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5, -1.5, 2.5, -2.5, 3.14] {
        let x_val = make_f64_tensor(&[], vec![x]);
        let round_x = eval_primitive(Primitive::Round, &[x_val], &no_params()).unwrap();
        let distance = (extract_f64_scalar(&round_x) - x).abs();
        assert!(
            distance <= 0.5 + 1e-14,
            "|Round({}) - {}| = {} should be <= 0.5",
            x, x, distance
        );
    }
}

// ======================== METAMORPHIC: Floor(x) <= Round(x) <= Ceil(x) ========================

#[test]
fn metamorphic_round_between_floor_ceil() {
    // Floor(x) <= Round(x) <= Ceil(x) for all x
    for x in [0.3, 0.5, 0.7, 1.4, 1.5, 1.6, -0.3, -0.5, -0.7, 2.5] {
        let x_val = make_f64_tensor(&[], vec![x]);

        let floor_x = extract_f64_scalar(&eval_primitive(Primitive::Floor, &[x_val.clone()], &no_params()).unwrap());
        let round_x = extract_f64_scalar(&eval_primitive(Primitive::Round, &[x_val.clone()], &no_params()).unwrap());
        let ceil_x = extract_f64_scalar(&eval_primitive(Primitive::Ceil, &[x_val], &no_params()).unwrap());

        assert!(
            floor_x <= round_x && round_x <= ceil_x,
            "Floor({}) <= Round({}) <= Ceil({}): {} <= {} <= {}",
            x, x, x, floor_x, round_x, ceil_x
        );
    }
}
