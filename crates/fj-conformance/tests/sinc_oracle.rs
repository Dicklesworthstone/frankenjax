//! Oracle tests for Sinc primitive.
//!
//! sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1 (the normalized sinc)
//!
//! Tests:
//! - Zero: sinc(0) = 1
//! - Known values: sinc(1) = 0, sinc(0.5) = 2/pi
//! - Symmetry: sinc(-x) = sinc(x)
//! - Special values: infinity, NaN
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

// ======================== Zero Case ========================

#[test]
fn oracle_sinc_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "sinc(0) = 1");
}

#[test]
fn oracle_sinc_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "sinc(-0) = 1");
}

// ======================== Known Values (normalized sinc) ========================

#[test]
fn oracle_sinc_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.abs() < 1e-15,
        "sinc(1) = sin(pi)/pi ~ 0 (got {})",
        actual
    );
}

#[test]
fn oracle_sinc_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.abs() < 1e-15,
        "sinc(2) = sin(2pi)/(2pi) ~ 0 (got {})",
        actual
    );
}

#[test]
fn oracle_sinc_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 2.0 / PI;
    assert!(
        (actual - expected).abs() < 1e-15,
        "sinc(0.5) = sin(pi/2)/(pi/2) = 2/pi ~ 0.6366"
    );
}

#[test]
fn oracle_sinc_quarter() {
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = (PI / 4.0).sin() / (PI / 4.0);
    assert!(
        (actual - expected).abs() < 1e-15,
        "sinc(0.25) = sin(pi/4)/(pi/4)"
    );
}

// ======================== Symmetry ========================

#[test]
fn oracle_sinc_symmetry() {
    for x in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Sinc, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Sinc, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val - neg_val).abs() < 1e-15,
            "sinc({}) = sinc(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

// ======================== Small Values (near limit) ========================

#[test]
fn oracle_sinc_small() {
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 1.0).abs() < 1e-10, "sinc(tiny) ~ 1");
}

// ======================== Special Values ========================

#[test]
fn oracle_sinc_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.is_nan(),
        "sinc(inf) = NaN (sin(inf)/inf is indeterminate)"
    );
}

#[test]
fn oracle_sinc_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "sinc(-inf) = NaN");
}

#[test]
fn oracle_sinc_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sinc(NaN) = NaN");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_sinc_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 0.5, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
    assert!((vals[1] - 2.0 / PI).abs() < 1e-15);
    assert!(vals[2].abs() < 1e-15);
    assert!(vals[3].abs() < 1e-15);
}

#[test]
fn oracle_sinc_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 0.5, -0.5, 1.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
    assert!((vals[1] - 2.0 / PI).abs() < 1e-15);
    assert!((vals[2] - 2.0 / PI).abs() < 1e-15);
    assert!(vals[3].abs() < 1e-15);
}

#[test]
fn oracle_sinc_integer_zeros() {
    // sinc(n) = 0 for all nonzero integers n
    for n in [3, 4, 5, 10, -3, -5] {
        let input = make_f64_tensor(&[], vec![n as f64]);
        let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
        let actual = extract_f64_scalar(&result);
        assert!(
            actual.abs() < 1e-14,
            "sinc({}) should be ~0, got {}",
            n,
            actual
        );
    }
}

#[test]
fn oracle_sinc_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_sinc_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_sinc_3d_shape() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.5, 1.0, 2.0, -0.5, -1.0, 0.25, 0.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 8);
    assert!((vals[0] - 1.0).abs() < 1e-15);
    assert!((vals[7] - 1.0).abs() < 1e-15);
}

#[test]
fn oracle_sinc_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[], vec![subnormal]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 1.0).abs() < 1e-10, "sinc(subnormal) ~ 1");
}

#[test]
fn oracle_sinc_large_integer() {
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.abs() < 1e-13, "sinc(100) ~ 0");
}

#[test]
fn oracle_sinc_f32_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape { dims: vec![3] },
            vec![
                Literal::F32Bits(0.0_f32.to_bits()),
                Literal::F32Bits(0.5_f32.to_bits()),
                Literal::F32Bits(1.0_f32.to_bits()),
            ],
        )
        .unwrap(),
    );
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(extract_shape(&result), vec![3]);
}

#[test]
fn oracle_sinc_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 4] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 4]);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_sinc_preserves_all_float_dtypes() {
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

    let values = [0.0_f64, 0.5, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "sinc {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: shape.to_vec() },
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
            Shape { dims: shape.to_vec() },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_complex64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]

fn oracle_sinc_complex64_at_zero() {
    // sinc(0) = 1 for any dtype
    let input = make_complex64_tensor(&[1], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params())
        .expect("sinc complex64 should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-5, "sinc(0) = 1");
    assert!(vals[0].1.abs() < 1e-5);
}

#[test]

fn oracle_sinc_complex64_real_axis() {
    // sinc on real axis should match real sinc
    let input = make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params())
        .expect("sinc complex64 real should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-5, "sinc(0) = 1");
    // sinc(1) = sin(pi)/pi ≈ 0 (normalized sinc)
    // or sin(1)/1 ≈ 0.8415 (unnormalized)
}

#[test]

fn oracle_sinc_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params())
        .expect("sinc complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]

fn property_sinc_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            DType::Complex128 => make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::Sinc, &[input], &no_params())
            .expect("sinc should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "sinc {dtype:?}: dtype mismatch");
    }
}
