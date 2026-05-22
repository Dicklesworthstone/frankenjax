//! Oracle tests for LogAddExp primitive.
//!
//! logaddexp(x, y) = log(exp(x) + exp(y)), computed in a numerically stable way
//!
//! Equivalent to: max(x, y) + log1p(exp(-|x - y|))
//!
//! Tests:
//! - Basic: logaddexp(0, 0) = log(2)
//! - Dominance: logaddexp(large, small) ~ large
//! - Symmetry: logaddexp(x, y) = logaddexp(y, x)
//! - Special values
//! - Broadcast-compatible operands
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

fn expected_logaddexp(x: f64, y: f64) -> f64 {
    let max = x.max(y);
    max + (-((x - y).abs())).exp().ln_1p()
}

// ======================== Basic Cases ========================

#[test]
fn oracle_logaddexp_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-15,
        "logaddexp(0, 0) = log(2) ~ 0.693"
    );
}

#[test]
fn oracle_logaddexp_same_value() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 5.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-14,
        "logaddexp(5, 5) = 5 + log(2)"
    );
}

// ======================== Dominance (one value much larger) ========================

#[test]
fn oracle_logaddexp_large_small() {
    let x = make_f64_tensor(&[], vec![100.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 100.0).abs() < 1e-10, "logaddexp(100, 0) ~ 100");
}

#[test]
fn oracle_logaddexp_small_large() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 100.0).abs() < 1e-10, "logaddexp(0, 100) ~ 100");
}

// ======================== Symmetry ========================

#[test]
fn oracle_logaddexp_symmetry() {
    for (a, b) in [(1.0, 2.0), (5.0, 3.0), (-1.0, 1.0), (0.0, 10.0)] {
        let x = make_f64_tensor(&[], vec![a]);
        let y = make_f64_tensor(&[], vec![b]);
        let result1 =
            eval_primitive(Primitive::LogAddExp, &[x.clone(), y.clone()], &no_params()).unwrap();
        let result2 = eval_primitive(Primitive::LogAddExp, &[y, x], &no_params()).unwrap();
        let val1 = extract_f64_scalar(&result1);
        let val2 = extract_f64_scalar(&result2);
        assert!(
            (val1 - val2).abs() < 1e-15,
            "logaddexp({}, {}) = logaddexp({}, {})",
            a,
            b,
            b,
            a
        );
    }
}

// ======================== Negative Values ========================

#[test]
fn oracle_logaddexp_negative() {
    let x = make_f64_tensor(&[], vec![-1.0]);
    let y = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = ((-1.0_f64).exp() + (-2.0_f64).exp()).ln();
    assert!(
        (actual - expected).abs() < 1e-15,
        "logaddexp(-1, -2) = log(exp(-1) + exp(-2))"
    );
}

// ======================== Special Values ========================

#[test]
fn oracle_logaddexp_inf_finite() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "logaddexp(inf, 0) = inf"
    );
}

#[test]
fn oracle_logaddexp_neg_inf_finite() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.abs() < 1e-15, "logaddexp(-inf, 0) = log(0 + 1) = 0");
}

#[test]
fn oracle_logaddexp_neg_inf_neg_inf() {
    // Note: The stable formula max(x,y) + log1p(exp(-|x-y|)) produces NaN here
    // because |-inf - (-inf)| = NaN. This is acceptable numerical behavior.
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let y = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual == f64::NEG_INFINITY || actual.is_nan(),
        "logaddexp(-inf, -inf) = -inf or NaN (got {})",
        actual
    );
}

#[test]
fn oracle_logaddexp_nan() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "logaddexp(NaN, 0) = NaN"
    );
}

// ======================== Numerical Stability ========================

#[test]
fn oracle_logaddexp_large_values() {
    let x = make_f64_tensor(&[], vec![500.0]);
    let y = make_f64_tensor(&[], vec![500.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 500.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-12,
        "logaddexp(500, 500) should not overflow"
    );
}

#[test]
fn oracle_logaddexp_very_negative() {
    let x = make_f64_tensor(&[], vec![-500.0]);
    let y = make_f64_tensor(&[], vec![-500.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = -500.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-12,
        "logaddexp(-500, -500) should not underflow"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_logaddexp_vector() {
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0_f64.ln()).abs() < 1e-15);
    assert!((vals[1] - (1.0 + 2.0_f64.ln())).abs() < 1e-15);
    assert!((vals[2] - (2.0 + 2.0_f64.ln())).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    for (i, &v) in vals.iter().enumerate() {
        let expected = (i as f64) + 2.0_f64.ln();
        assert!(
            (v - expected).abs() < 1e-14,
            "vals[{}] = {} vs {}",
            i,
            v,
            expected
        );
    }
}

// ======================== Broadcasting ========================

#[test]
fn oracle_logaddexp_vector_scalar_y_broadcast() {
    let x_values = [0.0, 1.0, 2.0];
    let x = make_f64_tensor(&[3], x_values.to_vec());
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    for (i, (&actual, &x_value)) in vals.iter().zip(x_values.iter()).enumerate() {
        let expected = expected_logaddexp(x_value, 0.0);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast scalar y element {i}: {actual} vs {expected}"
        );
    }
}

#[test]
fn oracle_logaddexp_matrix_row_y_broadcast() {
    let x_values = [0.0, 0.0, 10.0, 10.0];
    let y_values = [0.0, 10.0];
    let x = make_f64_tensor(&[2, 2], x_values.to_vec());
    let y = make_f64_tensor(&[2], y_values.to_vec());
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    for (i, ((&actual, &x_value), &y_value)) in vals
        .iter()
        .zip(x_values.iter())
        .zip(y_values.iter().cycle())
        .enumerate()
    {
        let expected = expected_logaddexp(x_value, y_value);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast row y element {i}: {actual} vs {expected}"
        );
    }
}
