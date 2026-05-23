//! Oracle tests for Deg2Rad and Rad2Deg primitives.
//!
//! deg2rad(x) = x * (pi / 180)
//! rad2deg(x) = x * (180 / pi)
//!
//! Tests:
//! - Common angles: 0, 90, 180, 270, 360 degrees
//! - Inverse relationship
//! - Negative angles
//! - Special values
//! - Tensor shapes

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::f64::consts::PI;

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

// ======================== Deg2Rad: Common Angles ========================

#[test]
fn oracle_deg2rad_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "deg2rad(0) = +0");
}

#[test]
fn oracle_deg2rad_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "deg2rad(-0.0) = -0");
}

#[test]
fn oracle_deg2rad_90() {
    let input = make_f64_tensor(&[], vec![90.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - PI / 2.0).abs() < 1e-15, "deg2rad(90) = pi/2");
}

#[test]
fn oracle_deg2rad_180() {
    let input = make_f64_tensor(&[], vec![180.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - PI).abs() < 1e-15, "deg2rad(180) = pi");
}

#[test]
fn oracle_deg2rad_270() {
    let input = make_f64_tensor(&[], vec![270.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        (actual - 3.0 * PI / 2.0).abs() < 1e-14,
        "deg2rad(270) = 3pi/2"
    );
}

#[test]
fn oracle_deg2rad_360() {
    let input = make_f64_tensor(&[], vec![360.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 2.0 * PI).abs() < 1e-14, "deg2rad(360) = 2pi");
}

// ======================== Rad2Deg: Common Angles ========================

#[test]
fn oracle_rad2deg_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "rad2deg(0) = +0");
}

#[test]
fn oracle_rad2deg_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "rad2deg(-0.0) = -0");
}

#[test]
fn oracle_rad2deg_pi_over_2() {
    let input = make_f64_tensor(&[], vec![PI / 2.0]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 90.0).abs() < 1e-13, "rad2deg(pi/2) = 90");
}

#[test]
fn oracle_rad2deg_pi() {
    let input = make_f64_tensor(&[], vec![PI]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 180.0).abs() < 1e-13, "rad2deg(pi) = 180");
}

#[test]
fn oracle_rad2deg_2pi() {
    let input = make_f64_tensor(&[], vec![2.0 * PI]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 360.0).abs() < 1e-12, "rad2deg(2pi) = 360");
}

// ======================== Inverse Relationship ========================

#[test]
fn oracle_deg2rad_rad2deg_inverse() {
    for deg in [0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0, 360.0] {
        let input = make_f64_tensor(&[], vec![deg]);
        let rad = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
        let roundtrip = eval_primitive(Primitive::Rad2Deg, &[rad], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - deg).abs() < 1e-12,
            "rad2deg(deg2rad({})) = {}",
            deg,
            actual
        );
    }
}

#[test]
fn oracle_rad2deg_deg2rad_inverse() {
    for rad in [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, 2.0 * PI] {
        let input = make_f64_tensor(&[], vec![rad]);
        let deg = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
        let roundtrip = eval_primitive(Primitive::Deg2Rad, &[deg], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - rad).abs() < 1e-14,
            "deg2rad(rad2deg({})) = {}",
            rad,
            actual
        );
    }
}

// ======================== Negative Angles ========================

#[test]
fn oracle_deg2rad_negative() {
    let input = make_f64_tensor(&[], vec![-90.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - (-PI / 2.0)).abs() < 1e-15, "deg2rad(-90) = -pi/2");
}

#[test]
fn oracle_rad2deg_negative() {
    let input = make_f64_tensor(&[], vec![-PI]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - (-180.0)).abs() < 1e-13, "rad2deg(-pi) = -180");
}

// ======================== Special Values ========================

#[test]
fn oracle_deg2rad_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "deg2rad(inf) = inf"
    );
}

#[test]
fn oracle_deg2rad_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "deg2rad(NaN) = NaN");
}

#[test]
fn oracle_rad2deg_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "rad2deg(inf) = inf"
    );
}

#[test]
fn oracle_rad2deg_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "rad2deg(NaN) = NaN");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_deg2rad_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 90.0, 180.0, 360.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-15);
    assert!((vals[1] - PI / 2.0).abs() < 1e-15);
    assert!((vals[2] - PI).abs() < 1e-15);
    assert!((vals[3] - 2.0 * PI).abs() < 1e-14);
}

#[test]
fn oracle_rad2deg_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, PI / 2.0, PI, 2.0 * PI]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-13);
    assert!((vals[1] - 90.0).abs() < 1e-13);
    assert!((vals[2] - 180.0).abs() < 1e-13);
    assert!((vals[3] - 360.0).abs() < 1e-12);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_deg2rad_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 90.0, 180.0, 270.0, 360.0, -90.0, -180.0, 45.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-15);
    assert!((vals[7] - PI / 4.0).abs() < 1e-15);
}

#[test]
fn oracle_rad2deg_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI, -PI / 2.0, -PI, PI / 4.0]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-13);
    assert!((vals[7] - 45.0).abs() < 1e-13);
}

#[test]
fn oracle_deg2rad_empty() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_rad2deg_empty() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_deg2rad_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 90.0, 180.0]);
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_rad2deg_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, PI / 2.0, PI]);
    let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_deg2rad_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_deg2rad_preserves_all_float_dtypes() {
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
        Value::Tensor(
            TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap(),
        )
    }

    let values = [0.0_f64, 90.0, 180.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Deg2Rad, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "deg2rad {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_rad2deg_preserves_all_float_dtypes() {
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
        Value::Tensor(
            TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap(),
        )
    }

    let values = [0.0_f64, std::f64::consts::PI / 2.0, std::f64::consts::PI];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Rad2Deg, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "rad2deg {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
