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

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

// -- XLogY Broadcast Tests --

#[test]
fn oracle_xlogy_all_scalars_broadcast() {
    let x = scalar_f64(2.0);
    let y = scalar_f64(std::f64::consts::E);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
    assert!((extract_f64_scalar(&result) - 2.0).abs() < 1e-14);
}

#[test]
fn oracle_xlogy_scalar_x_tensor_y_broadcast() {
    let x = scalar_f64(2.0);
    let y = make_f64_tensor(
        &[3],
        vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ],
    );
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14, "xlogy(2, 1) = 0");
    assert!((vals[1] - 2.0).abs() < 1e-14, "xlogy(2, e) = 2");
    assert!((vals[2] - 4.0).abs() < 1e-14, "xlogy(2, e^2) = 4");
}

#[test]
fn oracle_xlogy_tensor_x_scalar_y_broadcast() {
    let x_values: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
    let x = make_f64_tensor(&[4], x_values.to_vec());
    let y = scalar_f64(std::f64::consts::E);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlogy tensor_x scalar_y element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlogy_vector_scalar_y_broadcast() {
    let x_values: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
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
fn oracle_xlogy_singleton_x_vector_y_broadcast() {
    let x = make_f64_tensor(&[1], vec![2.0]);
    let y = make_f64_tensor(
        &[3],
        vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ],
    );
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 2.0).abs() < 1e-14);
    assert!((vals[2] - 4.0).abs() < 1e-14);
}

#[test]
fn oracle_xlogy_column_x_matrix_y_broadcast() {
    let x = make_f64_tensor(&[2, 1], vec![1.0, 2.0]);
    let y = make_f64_tensor(
        &[2, 3],
        vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E,
            1.0,
            std::f64::consts::E,
            std::f64::consts::E,
        ],
    );
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: xlogy(1, 1)=0, xlogy(1, e)=1, xlogy(1, e)=1
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 1.0).abs() < 1e-14);
    assert!((vals[2] - 1.0).abs() < 1e-14);
    // Row 1: xlogy(2, 1)=0, xlogy(2, e)=2, xlogy(2, e)=2
    assert!((vals[3] - 0.0).abs() < 1e-14);
    assert!((vals[4] - 2.0).abs() < 1e-14);
    assert!((vals[5] - 2.0).abs() < 1e-14);
}

#[test]
fn oracle_xlogy_matrix_row_y_broadcast() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2], vec![1.0, std::f64::consts::E]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let actual = extract_f64_vec(&result);
    let expected: [f64; 4] = [0.0, 1.0, 0.0, 3.0];
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlogy row y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlogy_different_ranks_broadcast() {
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let y = make_f64_tensor(
        &[2, 3],
        vec![
            std::f64::consts::E,
            std::f64::consts::E,
            std::f64::consts::E,
            1.0,
            1.0,
            1.0,
        ],
    );
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: xlogy(1,e)=1, xlogy(2,e)=2, xlogy(3,e)=3
    assert!((vals[0] - 1.0).abs() < 1e-14);
    assert!((vals[1] - 2.0).abs() < 1e-14);
    assert!((vals[2] - 3.0).abs() < 1e-14);
    // Row 1: xlogy(1,1)=0, xlogy(2,1)=0, xlogy(3,1)=0
    assert!((vals[3] - 0.0).abs() < 1e-14);
    assert!((vals[4] - 0.0).abs() < 1e-14);
    assert!((vals[5] - 0.0).abs() < 1e-14);
}

#[test]
fn oracle_xlogy_incompatible_shapes_error() {
    let x = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// -- XLog1PY Broadcast Tests --

#[test]
fn oracle_xlog1py_all_scalars_broadcast() {
    let x = scalar_f64(2.0);
    let y = scalar_f64(std::f64::consts::E - 1.0);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();
    assert!((extract_f64_scalar(&result) - 2.0).abs() < 1e-14);
}

#[test]
fn oracle_xlog1py_scalar_x_tensor_y_broadcast() {
    let x = scalar_f64(2.0);
    let e = std::f64::consts::E;
    let y = make_f64_tensor(&[3], vec![0.0, e - 1.0, e * e - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14, "xlog1py(2, 0) = 0");
    assert!((vals[1] - 2.0).abs() < 1e-14, "xlog1py(2, e-1) = 2");
    assert!((vals[2] - 4.0).abs() < 1e-12, "xlog1py(2, e^2-1) = 4");
}

#[test]
fn oracle_xlog1py_tensor_x_scalar_y_broadcast() {
    let x_values: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
    let x = make_f64_tensor(&[4], x_values.to_vec());
    let y = scalar_f64(std::f64::consts::E - 1.0);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlog1py tensor_x scalar_y element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlog1py_vector_scalar_y_broadcast() {
    let x_values: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
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
fn oracle_xlog1py_singleton_x_vector_y_broadcast() {
    let x = make_f64_tensor(&[1], vec![2.0]);
    let e = std::f64::consts::E;
    let y = make_f64_tensor(&[3], vec![0.0, e - 1.0, e * e - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 2.0).abs() < 1e-14);
    assert!((vals[2] - 4.0).abs() < 1e-12);
}

#[test]
fn oracle_xlog1py_matrix_row_y_broadcast() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2], vec![0.0, std::f64::consts::E - 1.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let actual = extract_f64_vec(&result);
    let expected: [f64; 4] = [0.0, 1.0, 0.0, 3.0];
    for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "xlog1py row y broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_xlog1py_incompatible_shapes_error() {
    let x = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_xlogy_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let x_values = [0.0_f64, 1.0, 2.0];
    let y_values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let x = make_vec(dtype, &x_values);
        let y = make_vec(dtype, &y_values);
        let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "xlogy {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex_vec(v: &Value) -> Vec<(f64, f64)> {
    let tensor = v.as_tensor().expect("expected tensor result");
    tensor
        .elements
        .iter()
        .map(|literal| match *literal {
            Literal::Complex64Bits(re, im) => {
                (f32::from_bits(re) as f64, f32::from_bits(im) as f64)
            }
            Literal::Complex128Bits(re, im) => (f64::from_bits(re), f64::from_bits(im)),
            other => panic!("expected complex literal, got {other:?}"),
        })
        .collect()
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64), tol: f64, label: &str) {
    assert!(
        (actual.0 - expected.0).abs() <= tol,
        "{label} real: expected {}, got {}",
        expected.0,
        actual.0
    );
    assert!(
        (actual.1 - expected.1).abs() <= tol,
        "{label} imag: expected {}, got {}",
        expected.1,
        actual.1
    );
}

#[test]

fn oracle_xlogy_complex64_basic() {
    // xlogy(1+0i, e+0i) = 1 * log(e) = 1
    let x = make_complex64_tensor(&[1], vec![(1.0, 0.0)]);
    let y = make_complex64_tensor(&[1], vec![(std::f32::consts::E, 0.0)]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params())
        .expect("xlogy complex64 should succeed");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]

fn oracle_xlogy_complex128_preserves_dtype() {
    let x = make_complex128_tensor(&[1], vec![(1.0, 0.0)]);
    let y = make_complex128_tensor(&[1], vec![(1.0, 0.0)]);
    let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params())
        .expect("xlogy complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]

fn property_xlogy_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (x, y) = match dtype {
            DType::Complex64 => (
                make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
                make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
                make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::XLogY, &[x, y], &no_params())
            .expect("xlogy should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "xlogy {dtype:?}: dtype mismatch");
    }
}

#[test]
fn oracle_xlog1py_complex64_basic_and_zero_mask() {
    let x = make_complex64_tensor(&[2], vec![(1.0, 0.0), (0.0, 0.0)]);
    let y = make_complex64_tensor(&[2], vec![(0.0, 1.0), (-1.0, 0.0)]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params())
        .expect("xlog1py complex64 should succeed");

    assert_eq!(result.dtype(), DType::Complex64);
    let actual = extract_complex_vec(&result);
    assert_complex_close(
        actual[0],
        (0.5 * 2.0_f64.ln(), std::f64::consts::FRAC_PI_4),
        1e-6,
        "complex64 xlog1py(1, i)",
    );
    assert_complex_close(actual[1], (0.0, 0.0), 1e-6, "complex64 xlog1py(0, -1)");
}

#[test]
fn oracle_xlog1py_complex128_basic() {
    let x = make_complex128_tensor(&[1], vec![(2.0, 0.0)]);
    let y = make_complex128_tensor(&[1], vec![(0.0, 1.0)]);
    let result = eval_primitive(Primitive::XLog1PY, &[x, y], &no_params())
        .expect("xlog1py complex128 should succeed");

    assert_eq!(result.dtype(), DType::Complex128);
    let actual = extract_complex_vec(&result);
    assert_complex_close(
        actual[0],
        (2.0_f64.ln(), std::f64::consts::FRAC_PI_2),
        1e-14,
        "complex128 xlog1py(2, i)",
    );
}
