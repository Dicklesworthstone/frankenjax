//! Oracle tests for the Polygamma primitive.
//!
//! Upstream JAX exposes `lax.polygamma(m, x)` as an elementwise primitive.
//! These tests pin the core scalar identities and tensor broadcasting forms
//! that FrankenJAX's primitive evaluator must preserve.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
const APERY: f64 = 1.202_056_903_159_594_2;
const POLYGAMMA_APPROX_TOL: f64 = 5e-4;

fn scalar(value: f64) -> Value {
    Value::scalar_f64(value)
}

fn vector(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn extract_scalar(value: &Value) -> f64 {
    value.as_f64_scalar().expect("expected f64 scalar")
}

fn extract_vector(value: &Value) -> Vec<f64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_f64().expect("expected f64 literal"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{context}: expected {expected}, got {actual}, diff {diff}, tol {tol}"
    );
}

fn eval_polygamma(order: Value, x: Value) -> Value {
    eval_primitive(Primitive::Polygamma, &[order, x], &no_params())
        .expect("polygamma eval should succeed")
}

#[test]
fn polygamma_zero_order_matches_digamma() {
    let x = scalar(3.0);
    let polygamma = eval_polygamma(scalar(0.0), x.clone());
    let digamma = eval_primitive(Primitive::Digamma, &[x], &no_params())
        .expect("digamma eval should succeed");

    assert_close(
        extract_scalar(&polygamma),
        extract_scalar(&digamma),
        1e-12,
        "polygamma(0, x) should equal digamma(x)",
    );
}

#[test]
fn polygamma_first_order_known_constants() {
    let at_one = eval_polygamma(scalar(1.0), scalar(1.0));
    let at_half = eval_polygamma(scalar(1.0), scalar(0.5));

    assert_close(
        extract_scalar(&at_one),
        std::f64::consts::PI.powi(2) / 6.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma(1, 1)",
    );
    assert_close(
        extract_scalar(&at_half),
        std::f64::consts::PI.powi(2) / 2.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma(1, 0.5)",
    );
}

#[test]
fn polygamma_second_order_known_constant() {
    let result = eval_polygamma(scalar(2.0), scalar(1.0));

    assert_close(
        extract_scalar(&result),
        -2.0 * APERY,
        POLYGAMMA_APPROX_TOL,
        "polygamma(2, 1)",
    );
}

#[test]
fn polygamma_tensor_argument_preserves_shape_and_values() {
    let result = eval_polygamma(scalar(1.0), vector(&[1.0, 2.0]));
    let tensor = result.as_tensor().expect("expected tensor result");

    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(2));

    let values = extract_vector(&result);
    assert_close(
        values[0],
        std::f64::consts::PI.powi(2) / 6.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma(1, [1, 2])[0]",
    );
    assert_close(
        values[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma(1, [1, 2])[1]",
    );
}

#[test]
fn polygamma_tensor_order_is_elementwise() {
    let result = eval_polygamma(vector(&[0.0, 1.0]), scalar(2.0));
    let tensor = result.as_tensor().expect("expected tensor result");

    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(2));

    let values = extract_vector(&result);
    assert_close(
        values[0],
        1.0 - EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "polygamma([0, 1], 2)[0]",
    );
    assert_close(
        values[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma([0, 1], 2)[1]",
    );
}

#[test]
fn polygamma_first_order_recurrence() {
    let x = 2.5;
    let at_x = extract_scalar(&eval_polygamma(scalar(1.0), scalar(x)));
    let at_next = extract_scalar(&eval_polygamma(scalar(1.0), scalar(x + 1.0)));

    assert_close(at_next, at_x - 1.0 / (x * x), 1e-8, "trigamma recurrence");
}

#[test]
fn polygamma_negative_order_returns_nan() {
    let result = eval_polygamma(scalar(-1.0), scalar(2.0));

    assert!(extract_scalar(&result).is_nan());
}

#[test]
fn polygamma_rejects_invalid_arity() {
    let err = eval_primitive(Primitive::Polygamma, &[scalar(1.0)], &no_params())
        .expect_err("polygamma should require two inputs");

    assert!(
        err.to_string().contains("arity")
            || err.to_string().contains("expected 2")
            || err.to_string().contains("actual: 1"),
        "unexpected arity error: {err}"
    );
}
