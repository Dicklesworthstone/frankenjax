//! Oracle tests for XLogY and XLog1PY primitives.
//!
//! xlogy(x, y) = x * log(y), with xlogy(0, y) = 0 for any y including 0
//! xlog1py(x, y) = x * log1p(y), with xlog1py(0, y) = 0 for any y including -1
//!
//! Used in cross-entropy and KL-divergence calculations where 0*log(0) = 0.
//!
//! Tests:
//! - Basic: xlogy(2, e) = 2
//! - Zero: xlogy(0, y) = 0 for any y
//! - Negative x: xlogy(-2, e) = -2
//! - Special values: infinity, NaN
//! - Tensor shapes
//! - Broadcast-compatible operands

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

// ======================== XLogY Basic Cases ========================

#[test]
fn oracle_xlogy_basic() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let y = make_f64_tensor(&[], vec![std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 2.0).abs() < 1e-14, "xlogy(2, e) = 2*1 = 2");
}

#[test]
fn oracle_xlogy_basic_2() {
    let x = make_f64_tensor(&[], vec![3.0]);
    let y = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 3.0 * 2.0_f64.ln();
    assert!((actual - expected).abs() < 1e-14, "xlogy(3, 2) = 3*ln(2)");
}

// ======================== XLogY Zero X Cases ========================

#[test]
fn oracle_xlogy_zero_x_positive_y() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "xlogy(0, 5) = 0");
}

#[test]
fn oracle_xlogy_zero_x_zero_y() {
    // Key case: 0 * log(0) should be 0, not NaN
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "xlogy(0, 0) = 0 (special case)"
    );
}

#[test]
fn oracle_xlogy_zero_x_inf_y() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "xlogy(0, inf) = 0");
}

// ======================== XLogY Negative X ========================

#[test]
fn oracle_xlogy_negative_x() {
    let x = make_f64_tensor(&[], vec![-2.0]);
    let y = make_f64_tensor(&[], vec![std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - (-2.0)).abs() < 1e-14, "xlogy(-2, e) = -2");
}

// ======================== XLogY Special Values ========================

#[test]
fn oracle_xlogy_positive_x_zero_y() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "xlogy(2, 0) = -inf"
    );
}

#[test]
fn oracle_xlogy_nan() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "xlogy(NaN, 2) = NaN");
}

#[test]
fn oracle_xlogy_domain_edges_vector() {
    let x = make_f64_tensor(&[5], vec![0.0, 2.0, -2.0, 3.0, 4.0]);
    let y = make_f64_tensor(&[5], vec![-1.0, 0.0, 0.0, -2.0, 1.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0, "zero x masks log(negative) to zero");
    assert_eq!(vals[1], f64::NEG_INFINITY, "positive x times log(0)");
    assert_eq!(vals[2], f64::INFINITY, "negative x times log(0)");
    assert!(vals[3].is_nan(), "nonzero x with y < 0 is NaN");
    assert_eq!(vals[4], 0.0, "xlogy(x, 1) = 0");
}

// ======================== XLog1PY Basic Cases ========================

#[test]
fn oracle_xlog1py_basic() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let y = make_f64_tensor(&[], vec![std::f64::consts::E - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // xlog1py(2, e-1) = 2 * log(1 + (e-1)) = 2 * log(e) = 2
    assert!((actual - 2.0).abs() < 1e-14, "xlog1py(2, e-1) = 2");
}

#[test]
fn oracle_xlog1py_zero_x() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "xlog1py(0, 5) = 0");
}

#[test]
fn oracle_xlog1py_zero_x_neg_one_y() {
    // Key case: 0 * log1p(-1) = 0 * log(0) = 0, not NaN
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "xlog1py(0, -1) = 0 (special case)"
    );
}

#[test]
fn oracle_xlog1py_domain_edges_vector() {
    let x = make_f64_tensor(&[5], vec![0.0, 2.0, -2.0, 3.0, 4.0]);
    let y = make_f64_tensor(&[5], vec![-1.0, -1.0, -1.0, -2.0, 0.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0, "zero x masks log1p(-1) to zero");
    assert_eq!(vals[1], f64::NEG_INFINITY, "positive x times -inf");
    assert_eq!(vals[2], f64::INFINITY, "negative x times -inf");
    assert!(vals[3].is_nan(), "nonzero x with y < -1 is NaN");
    assert_eq!(vals[4], 0.0, "xlog1py(x, 0) = 0");
}

#[test]
fn oracle_xlog1py_zero_x_masks_invalid_log1p_lanes() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 0.0, 2.0]);
    let y = make_f64_tensor(&[2, 2], vec![-2.0, -2.0, f64::INFINITY, f64::INFINITY]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0, "zero x masks y < -1");
    assert!(vals[1].is_nan(), "nonzero x with y < -1 is NaN");
    assert_eq!(vals[2], 0.0, "zero x masks infinite log1p lane");
    assert_eq!(vals[3], f64::INFINITY, "positive x times log1p(inf)");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_xlogy_vector() {
    let x = make_f64_tensor(&[4], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[4], vec![0.0, 1.0, 1.0, std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0, "xlogy(0, 0) = 0");
    assert_eq!(vals[1], 0.0, "xlogy(1, 1) = 0");
    assert_eq!(vals[2], 0.0, "xlogy(2, 1) = 0");
    assert!((vals[3] - 3.0).abs() < 1e-14, "xlogy(3, e) = 3");
}

#[test]
fn oracle_xlogy_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 0.0]);
    let y = make_f64_tensor(&[2, 2], vec![0.0, 2.0, 2.0, 0.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - 2.0_f64.ln()).abs() < 1e-14);
    assert!((vals[2] - 2.0 * 2.0_f64.ln()).abs() < 1e-14);
    assert_eq!(vals[3], 0.0);
}

// ======================== Broadcasting ========================

#[test]
fn oracle_xlogy_vector_scalar_y_broadcast() {
    let x_values = [0.0, 1.0, 2.0, 3.0];
    let x = make_f64_tensor(&[4], x_values.to_vec());
    let y = make_f64_tensor(&[], vec![std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlogy scalar y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlogy_matrix_row_y_broadcast() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2], vec![1.0, std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let actual = extract_f64_vec(&result);
    let expected = [0.0, 1.0, 0.0, 3.0];
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlogy row y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlog1py_vector_scalar_y_broadcast() {
    let x_values = [0.0, 1.0, 2.0, 3.0];
    let x = make_f64_tensor(&[4], x_values.to_vec());
    let y = make_f64_tensor(&[], vec![std::f64::consts::E - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlog1py scalar y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlog1py_matrix_row_y_broadcast() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2], vec![0.0, std::f64::consts::E - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let actual = extract_f64_vec(&result);
    let expected = [0.0, 1.0, 0.0, 3.0];
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlog1py row y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}
