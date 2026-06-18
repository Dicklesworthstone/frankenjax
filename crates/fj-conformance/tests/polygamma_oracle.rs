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
const POLYGAMMA_APPROX_TOL: f64 = 1e-10;

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
fn polygamma_same_shape_tensors_are_elementwise() {
    let result = eval_polygamma(vector(&[0.0, 1.0]), vector(&[1.0, 2.0]));
    let tensor = result.as_tensor().expect("expected tensor result");

    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(2));

    let values = extract_vector(&result);
    assert_close(
        values[0],
        -EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "polygamma([0, 1], [1, 2])[0]",
    );
    assert_close(
        values[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "polygamma([0, 1], [1, 2])[1]",
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

#[test]
fn polygamma_rejects_integer_operands() {
    for (case, order, x) in [
        ("integer scalar order", Value::scalar_i64(1), scalar_f64(2.0)),
        ("integer scalar argument", scalar_f64(1.0), Value::scalar_i64(2)),
        (
            "integer tensor argument",
            scalar_f64(1.0),
            tensor_i64(&[2], &[2, 3]),
        ),
    ] {
        let err = eval_primitive(Primitive::Polygamma, &[order, x], &no_params())
            .expect_err("JAX polygamma is float-only and must reject integer operands");
        assert!(
            err.to_string().contains("floating operands"),
            "{case} returned unexpected polygamma error: {err}"
        );
    }
}

// --------------------------------------------------------------------------
// Broadcast tests: verify NumPy-style broadcasting semantics
// --------------------------------------------------------------------------

fn scalar_f64(v: f64) -> Value {
    Value::scalar_f64(v)
}

fn tensor_f64(shape: &[u32], data: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn tensor_i64(shape: &[u32], data: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter().copied().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

fn get_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Scalar(_) => vec![],
        Value::Tensor(t) => t.shape.dims.clone(),
    }
}

fn get_elements(v: &Value) -> Vec<f64> {
    match v {
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
    }
}

#[test]
fn polygamma_broadcast_scalar_scalar() {
    let result = eval_polygamma(scalar_f64(1.0), scalar_f64(2.0));
    assert_eq!(get_shape(&result), Vec::<u32>::new());
    let val = get_elements(&result)[0];
    assert_close(
        val,
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "scalar-scalar broadcast",
    );
}

#[test]
fn polygamma_broadcast_scalar_order_tensor_x() {
    let result = eval_polygamma(scalar_f64(1.0), tensor_f64(&[2], &[1.0, 2.0]));
    assert_eq!(get_shape(&result), vec![2]);
    let vals = get_elements(&result);
    assert_close(
        vals[0],
        std::f64::consts::PI.powi(2) / 6.0,
        POLYGAMMA_APPROX_TOL,
        "scalar-tensor[0]",
    );
    assert_close(
        vals[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "scalar-tensor[1]",
    );
}

#[test]
fn polygamma_broadcast_tensor_order_scalar_x() {
    let result = eval_polygamma(tensor_f64(&[2], &[0.0, 1.0]), scalar_f64(2.0));
    assert_eq!(get_shape(&result), vec![2]);
    let vals = get_elements(&result);
    assert_close(
        vals[0],
        1.0 - EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "tensor-scalar[0]",
    );
    assert_close(
        vals[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "tensor-scalar[1]",
    );
}

#[test]
fn polygamma_broadcast_singleton_to_vector() {
    let result = eval_polygamma(tensor_f64(&[1], &[1.0]), tensor_f64(&[3], &[1.0, 2.0, 3.0]));
    assert_eq!(get_shape(&result), vec![3]);
    let vals = get_elements(&result);
    let expected_trigamma_1 = std::f64::consts::PI.powi(2) / 6.0;
    let expected_trigamma_2 = expected_trigamma_1 - 1.0;
    let expected_trigamma_3 = expected_trigamma_2 - 0.25;
    assert_close(
        vals[0],
        expected_trigamma_1,
        POLYGAMMA_APPROX_TOL,
        "singleton→vec[0]",
    );
    assert_close(
        vals[1],
        expected_trigamma_2,
        POLYGAMMA_APPROX_TOL,
        "singleton→vec[1]",
    );
    assert_close(
        vals[2],
        expected_trigamma_3,
        POLYGAMMA_APPROX_TOL,
        "singleton→vec[2]",
    );
}

#[test]
fn polygamma_broadcast_vector_to_singleton() {
    let result = eval_polygamma(tensor_f64(&[3], &[0.0, 1.0, 2.0]), tensor_f64(&[1], &[2.0]));
    assert_eq!(get_shape(&result), vec![3]);
    let vals = get_elements(&result);
    assert_close(
        vals[0],
        1.0 - EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "vec→singleton[0] digamma(2)",
    );
    assert_close(
        vals[1],
        std::f64::consts::PI.powi(2) / 6.0 - 1.0,
        POLYGAMMA_APPROX_TOL,
        "vec→singleton[1] trigamma(2)",
    );
}

#[test]
fn polygamma_broadcast_column_to_row() {
    // [2,1] x [1,3] → [2,3]
    let order = tensor_f64(&[2, 1], &[0.0, 1.0]);
    let x = tensor_f64(&[1, 3], &[1.0, 2.0, 3.0]);
    let result = eval_polygamma(order, x);
    assert_eq!(get_shape(&result), vec![2, 3]);
    let vals = get_elements(&result);
    // Row 0: digamma, Row 1: trigamma
    assert_close(
        vals[0],
        -EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "col×row[0,0]",
    );
    assert_close(
        vals[1],
        1.0 - EULER_MASCHERONI,
        POLYGAMMA_APPROX_TOL,
        "col×row[0,1]",
    );
    let trigamma_1 = std::f64::consts::PI.powi(2) / 6.0;
    assert_close(vals[3], trigamma_1, POLYGAMMA_APPROX_TOL, "col×row[1,0]");
}

#[test]
fn polygamma_broadcast_different_ranks() {
    // [3] x [2,3] → [2,3]
    let order = tensor_f64(&[3], &[1.0, 1.0, 1.0]);
    let x = tensor_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_polygamma(order, x);
    assert_eq!(get_shape(&result), vec![2, 3]);
    let vals = get_elements(&result);
    let trigamma_1 = std::f64::consts::PI.powi(2) / 6.0;
    assert_close(vals[0], trigamma_1, POLYGAMMA_APPROX_TOL, "ranks[0]");
    assert_eq!(vals.len(), 6);
}

#[test]
fn polygamma_broadcast_incompatible_shapes_error() {
    let result = eval_primitive(
        Primitive::Polygamma,
        &[
            tensor_f64(&[2], &[1.0, 1.0]),
            tensor_f64(&[3], &[1.0, 2.0, 3.0]),
        ],
        &no_params(),
    );
    assert!(
        result.is_err(),
        "incompatible shapes [2] vs [3] should error"
    );
}

// ======================== Additional Coverage ========================

#[test]
fn polygamma_empty_tensor() {
    let result = eval_polygamma(scalar_f64(1.0), tensor_f64(&[0], &[]));
    assert_eq!(get_shape(&result), vec![0]);
    assert!(get_elements(&result).is_empty());
}

#[test]
fn polygamma_preserves_dtype() {
    let result = eval_polygamma(scalar_f64(1.0), scalar_f64(2.0));
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn polygamma_third_order() {
    let result = eval_polygamma(scalar_f64(3.0), scalar_f64(1.0));
    // polygamma(3, 1) = pi^4/15
    let expected = std::f64::consts::PI.powi(4) / 15.0;
    assert_close(
        extract_scalar(&result),
        expected,
        POLYGAMMA_APPROX_TOL,
        "polygamma(3, 1)",
    );
}

#[test]
fn polygamma_noninteger_order() {
    // Non-integer orders should still work (extension via gamma function)
    let result = eval_polygamma(scalar_f64(0.5), scalar_f64(1.0));
    // Just verify it returns a finite value
    let val = extract_scalar(&result);
    assert!(val.is_finite(), "polygamma(0.5, 1) should be finite");
}

#[test]
fn polygamma_2d_empty() {
    let order = scalar_f64(1.0);
    let x =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_polygamma(order, x);
    assert_eq!(get_shape(&result), vec![0, 3]);
}

#[test]
fn polygamma_3d_shape() {
    let order = scalar_f64(1.0);
    let x = tensor_f64(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_polygamma(order, x);
    assert_eq!(get_shape(&result), vec![2, 2, 2]);
    let vals = get_elements(&result);
    assert_eq!(vals.len(), 8);
    assert!(vals.iter().all(|&v| v.is_finite()));
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_polygamma_preserves_f64() {
    // Polygamma only supports F64 due to precision requirements
    let order = scalar_f64(1.0);
    let x = tensor_f64(&[3], &[1.0, 2.0, 3.0]);
    let result = eval_polygamma(order, x);
    assert_eq!(result.dtype(), DType::F64, "polygamma should output F64");
}
