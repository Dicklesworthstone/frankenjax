//! Oracle tests for ErfInv primitive.
//!
//! Tests against expected behavior matching JAX/scipy.special.erfinv:
//! - erfinv is the inverse of erf
//! - erfinv(erf(x)) = x for x in reasonable range
//! - erfinv(0) = 0
//! - erfinv(-1) = -inf, erfinv(1) = +inf

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
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
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

// Helper: compute erf using Horner approximation (matches fj-lax)
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_erf_inv_zero() {
    // erfinv(0) = 0
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10);
}

#[test]
fn oracle_erf_inv_signed_zero_bits() {
    let positive_zero = Value::Scalar(Literal::from_f64(0.0));
    let positive_result =
        eval_primitive(Primitive::ErfInv, &[positive_zero], &no_params()).unwrap();
    let positive_vals = extract_f64_vec(&positive_result);
    assert_eq!(positive_vals[0].to_bits(), 0.0_f64.to_bits());

    let negative_zero = Value::Scalar(Literal::from_f64(-0.0));
    let negative_result =
        eval_primitive(Primitive::ErfInv, &[negative_zero], &no_params()).unwrap();
    let negative_vals = extract_f64_vec(&negative_result);
    assert_eq!(negative_vals[0].to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn oracle_erf_inv_small_positive() {
    // erfinv(0.5) ≈ 0.4769
    let input = Value::Scalar(Literal::from_f64(0.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.4769).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_small_negative() {
    // erfinv(-0.5) ≈ -0.4769
    let input = Value::Scalar(Literal::from_f64(-0.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-0.4769)).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_close_to_one() {
    // erfinv(0.9) ≈ 1.1631
    let input = Value::Scalar(Literal::from_f64(0.9));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1631).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_close_to_neg_one() {
    // erfinv(-0.9) ≈ -1.1631
    let input = Value::Scalar(Literal::from_f64(-0.9));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.1631)).abs() < 0.01);
}

// ======================== Inverse Property Tests ========================

#[test]
fn oracle_erf_inv_inverse_property() {
    // erfinv(erf(x)) ≈ x for small x
    let x_values = [0.0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0];
    let erf_values: Vec<f64> = x_values.iter().map(|&x| erf_approx(x)).collect();
    let input = make_f64_tensor(&[7], erf_values);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    for (v, &x) in vals.iter().zip(x_values.iter()) {
        assert!((v - x).abs() < 0.05, "erfinv(erf({x})) = {v}, expected {x}");
    }
}

// ======================== 1D Tests ========================

#[test]
fn oracle_erf_inv_1d() {
    let input = make_f64_tensor(&[5], vec![-0.5, -0.25, 0.0, 0.25, 0.5]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    // erfinv is odd: erfinv(-x) = -erfinv(x)
    assert!((vals[0] + vals[4]).abs() < 0.01);
    assert!((vals[1] + vals[3]).abs() < 0.01);
    assert!(vals[2].abs() < 1e-10);
}

#[test]
fn oracle_erf_inv_1d_symmetric() {
    // Test symmetry: erfinv(-x) = -erfinv(x)
    let input = make_f64_tensor(&[4], vec![0.3, -0.3, 0.7, -0.7]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] + vals[1]).abs() < 1e-10);
    assert!((vals[2] + vals[3]).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_erf_inv_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 0.5, -0.5, 0.8]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10); // erfinv(0) = 0
    assert!((vals[1] + vals[2]).abs() < 1e-10); // symmetry
}

// ======================== Edge Cases ========================

#[test]
fn oracle_erf_inv_at_one() {
    // erfinv(1) = +inf
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
}

#[test]
fn oracle_erf_inv_at_neg_one() {
    // erfinv(-1) = -inf
    let input = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] < 0.0);
}

#[test]
fn oracle_erf_inv_single_element() {
    let input = make_f64_tensor(&[1], vec![0.5]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.4769).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_known_values() {
    // Known erfinv values (approximate)
    // erfinv(0.1) ≈ 0.0889
    // erfinv(0.2) ≈ 0.1791
    // erfinv(0.3) ≈ 0.2725
    let input = make_f64_tensor(&[3], vec![0.1, 0.2, 0.3]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0889).abs() < 0.01);
    assert!((vals[1] - 0.1791).abs() < 0.01);
    assert!((vals[2] - 0.2725).abs() < 0.01);
}

// ======================== Special Values ========================

#[test]
fn oracle_erf_inv_outside_domain_positive() {
    // erfinv(x) for x > 1 is undefined (NaN)
    let input = Value::Scalar(Literal::from_f64(1.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "erfinv(1.5) should be NaN (outside domain)");
}

#[test]
fn oracle_erf_inv_outside_domain_negative() {
    // erfinv(x) for x < -1 is undefined (NaN)
    let input = Value::Scalar(Literal::from_f64(-1.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "erfinv(-1.5) should be NaN (outside domain)");
}

#[test]
fn oracle_erf_inv_nan() {
    // erfinv(NaN) = NaN
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "erfinv(NaN) should be NaN");
}

#[test]
fn oracle_erf_inv_positive_infinity() {
    // erfinv(+inf) is outside domain, should be NaN
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "erfinv(+inf) should be NaN (outside domain)");
}

#[test]
fn oracle_erf_inv_negative_infinity() {
    // erfinv(-inf) is outside domain, should be NaN
    let input = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "erfinv(-inf) should be NaN (outside domain)");
}

#[test]
fn oracle_erf_inv_tensor_special_values() {
    // Test mixed special values in tensor form
    let input = make_f64_tensor(&[4], vec![1.5, -1.5, f64::NAN, f64::INFINITY]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan(), "erfinv(1.5) = NaN");
    assert!(vals[1].is_nan(), "erfinv(-1.5) = NaN");
    assert!(vals[2].is_nan(), "erfinv(NaN) = NaN");
    assert!(vals[3].is_nan(), "erfinv(+inf) = NaN");
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_erf_inv_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.5]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10); // erfinv(0) = 0
}

#[test]
fn oracle_erf_inv_empty() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_erf_inv_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_erf_inv_large_tensor() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 100.0).collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_f64_vec(&result);
    assert!(vals[50].abs() < 1e-10); // erfinv(0) = 0
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_erf_inv_preserves_all_float_dtypes() {
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

    // erfinv domain is (-1, 1)
    let values = [-0.5_f64, 0.0, 0.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "erfinv {dtype:?}: dtype mismatch");
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

#[test]
#[ignore = "PARITY GAP: erf_inv is not supported for complex dtypes"]
fn oracle_erf_inv_complex64_real_axis() {
    // erf_inv on real axis: erf_inv(0) = 0
    let input = make_complex64_tensor(&[2], vec![(0.0, 0.0), (0.5, 0.0)]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params())
        .expect("erf_inv complex64 should succeed");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
#[ignore = "PARITY GAP: erf_inv is not supported for complex dtypes"]
fn oracle_erf_inv_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(0.0, 0.0), (0.5, 0.0)]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params())
        .expect("erf_inv complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
#[ignore = "PARITY GAP: erf_inv is not supported for complex dtypes"]
fn property_erf_inv_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[2], vec![(0.0, 0.0), (0.5, 0.0)]),
            DType::Complex128 => make_complex128_tensor(&[2], vec![(0.0, 0.0), (0.5, 0.0)]),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::ErfInv, &[input], &no_params())
            .expect("erf_inv should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "erf_inv {dtype:?}: dtype mismatch");
    }
}
