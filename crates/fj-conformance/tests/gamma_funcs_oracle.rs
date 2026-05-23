//! Oracle tests for Lgamma and Digamma primitives.
//!
//! Tests against expected behavior matching scipy.special:
//! - Lgamma: log of gamma function, ln(Γ(x))
//! - Digamma: derivative of log gamma, ψ(x) = d/dx ln(Γ(x))

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

// ======================== Lgamma Tests ========================
// lgamma(x) = ln(Γ(x))
// Known values:
// lgamma(1) = ln(0!) = 0
// lgamma(2) = ln(1!) = 0
// lgamma(3) = ln(2!) = ln(2) ≈ 0.693
// lgamma(4) = ln(3!) = ln(6) ≈ 1.791
// lgamma(5) = ln(4!) = ln(24) ≈ 3.178

#[test]
fn oracle_lgamma_one() {
    // lgamma(1) = ln(Γ(1)) = ln(1) = 0
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10);
}

#[test]
fn oracle_lgamma_two() {
    // lgamma(2) = ln(Γ(2)) = ln(1) = 0
    let input = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10);
}

#[test]
fn oracle_lgamma_three() {
    // lgamma(3) = ln(Γ(3)) = ln(2!) = ln(2) ≈ 0.693
    let input = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0_f64.ln()).abs() < 0.01);
}

#[test]
fn oracle_lgamma_four() {
    // lgamma(4) = ln(Γ(4)) = ln(3!) = ln(6) ≈ 1.791
    let input = Value::Scalar(Literal::from_f64(4.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 6.0_f64.ln()).abs() < 0.01);
}

#[test]
fn oracle_lgamma_five() {
    // lgamma(5) = ln(Γ(5)) = ln(4!) = ln(24) ≈ 3.178
    let input = Value::Scalar(Literal::from_f64(5.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 24.0_f64.ln()).abs() < 0.01);
}

#[test]
fn oracle_lgamma_half() {
    // lgamma(0.5) = ln(Γ(0.5)) = ln(√π) ≈ 0.5723
    let input = Value::Scalar(Literal::from_f64(0.5));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let expected = (std::f64::consts::PI.sqrt()).ln();
    assert!((vals[0] - expected).abs() < 0.01);
}

#[test]
fn oracle_lgamma_negative_half() {
    // lgamma(-0.5) = ln(abs(Gamma(-0.5))) = ln(2 * sqrt(pi)).
    let input = Value::Scalar(Literal::from_f64(-0.5));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let expected = (2.0 * std::f64::consts::PI.sqrt()).ln();
    assert!((vals[0] - expected).abs() < 0.01);
}

#[test]
fn oracle_lgamma_1d() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 0.01); // lgamma(1) = 0
    assert!(vals[1].abs() < 0.01); // lgamma(2) = 0
    assert!((vals[2] - 2.0_f64.ln()).abs() < 0.01); // lgamma(3) = ln(2)
    assert!((vals[3] - 6.0_f64.ln()).abs() < 0.01); // lgamma(4) = ln(6)
}

#[test]
fn oracle_lgamma_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn oracle_lgamma_large() {
    // For large x, lgamma(x) ≈ (x - 0.5) * ln(x) - x + 0.5 * ln(2π)
    let input = Value::Scalar(Literal::from_f64(10.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // lgamma(10) = ln(9!) = ln(362880) ≈ 12.8018
    assert!((vals[0] - 362880.0_f64.ln()).abs() < 0.01);
}

// ======================== Digamma Tests ========================
// digamma(x) = ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
// Known values:
// digamma(1) = -γ ≈ -0.5772 (Euler-Mascheroni constant)
// digamma(2) = 1 - γ ≈ 0.4228
// digamma(n) = -γ + H_{n-1} for positive integers

#[test]
fn oracle_digamma_one() {
    // digamma(1) = -γ ≈ -0.5772
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    assert!((vals[0] - (-euler_mascheroni)).abs() < 0.01);
}

#[test]
fn oracle_digamma_two() {
    // digamma(2) = 1 - γ ≈ 0.4228
    let input = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    assert!((vals[0] - (1.0 - euler_mascheroni)).abs() < 0.01);
}

#[test]
fn oracle_digamma_three() {
    // digamma(3) = -γ + 1 + 1/2 = 1.5 - γ ≈ 0.9228
    let input = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    assert!((vals[0] - (1.5 - euler_mascheroni)).abs() < 0.01);
}

#[test]
fn oracle_digamma_half() {
    // digamma(0.5) = -γ - 2*ln(2) ≈ -1.9635
    let input = Value::Scalar(Literal::from_f64(0.5));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    let expected = -euler_mascheroni - 2.0 * 2.0_f64.ln();
    assert!((vals[0] - expected).abs() < 0.05);
}

#[test]
fn oracle_digamma_negative_half() {
    // digamma(-0.5) = 2 - gamma - 2*ln(2) by reflection.
    let input = Value::Scalar(Literal::from_f64(-0.5));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    let expected = 2.0 - euler_mascheroni - 2.0 * 2.0_f64.ln();
    assert!((vals[0] - expected).abs() < 0.05);
}

#[test]
fn oracle_digamma_1d() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // All values should be finite
    assert!(vals.iter().all(|v| v.is_finite()));
    // digamma is increasing for x > 0
    assert!(vals[1] > vals[0]);
    assert!(vals[2] > vals[1]);
}

#[test]
fn oracle_digamma_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn oracle_digamma_large() {
    // For large x, digamma(x) ≈ ln(x) - 1/(2x)
    let input = Value::Scalar(Literal::from_f64(100.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let approx = 100.0_f64.ln() - 1.0 / 200.0;
    assert!((vals[0] - approx).abs() < 0.01);
}

#[test]
fn oracle_digamma_increasing() {
    // digamma is strictly increasing for x > 0
    let input = make_f64_tensor(&[5], vec![0.5, 1.0, 2.0, 5.0, 10.0]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    for i in 1..5 {
        assert!(vals[i] > vals[i - 1], "digamma should be increasing");
    }
}

// ======================== Edge Cases ========================

#[test]
fn oracle_lgamma_single_element() {
    let input = make_f64_tensor(&[1], vec![2.0]);
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 0.01);
}

#[test]
fn oracle_digamma_single_element() {
    let input = make_f64_tensor(&[1], vec![1.0]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    let euler_mascheroni = 0.5772156649;
    assert!((vals[0] - (-euler_mascheroni)).abs() < 0.01);
}

// ======================== Special Values ========================

#[test]
fn oracle_lgamma_positive_infinity() {
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val > 0.0, "lgamma(+inf) = +inf");
}

#[test]
fn oracle_lgamma_nan() {
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "lgamma(NaN) = NaN");
}

#[test]
fn oracle_lgamma_zero() {
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val > 0.0, "lgamma(0) = +inf (pole)");
}

#[test]
fn oracle_lgamma_negative_integer() {
    let input = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val > 0.0, "lgamma(-1) = +inf (pole)");
}

#[test]
fn oracle_digamma_positive_infinity() {
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val > 0.0, "digamma(+inf) = +inf");
}

#[test]
fn oracle_digamma_nan() {
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "digamma(NaN) = NaN");
}

#[test]
fn oracle_digamma_zero() {
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val < 0.0, "digamma(0) = -inf (pole)");
}

#[test]
fn oracle_digamma_negative_integer() {
    let input = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite(), "digamma(-1) = +/-inf (pole)");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_lgamma_preserves_all_float_dtypes() {
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

    // lgamma domain: positive values (avoid poles at non-positive integers)
    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Lgamma, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "lgamma {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_digamma_preserves_all_float_dtypes() {
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

    // digamma domain: positive values (avoid poles at non-positive integers)
    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Digamma, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "digamma {dtype:?}: dtype mismatch");
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
#[ignore = "PARITY GAP: lgamma/digamma not supported for complex dtypes"]
fn oracle_lgamma_complex64_positive_real() {
    // lgamma on positive real axis should match real lgamma
    // lgamma(1) = 0, lgamma(2) = 0
    let input = make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params())
        .expect("lgamma complex64 should succeed");
    let vals = extract_complex64_vec(&result);
    assert!(vals[0].0.abs() < 1e-5, "lgamma(1) = 0");
    assert!(vals[1].0.abs() < 1e-5, "lgamma(2) = 0");
}

#[test]
#[ignore = "PARITY GAP: lgamma/digamma not supported for complex dtypes"]
fn oracle_lgamma_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Lgamma, &[input], &no_params())
        .expect("lgamma complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
#[ignore = "PARITY GAP: lgamma/digamma not supported for complex dtypes"]
fn oracle_digamma_complex64_positive_real() {
    // digamma(1) = -gamma (Euler-Mascheroni) ≈ -0.5772
    let input = make_complex64_tensor(&[1], vec![(1.0, 0.0)]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params())
        .expect("digamma complex64 should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - (-0.5772)).abs() < 0.01, "digamma(1) ≈ -0.5772");
}

#[test]
#[ignore = "PARITY GAP: lgamma/digamma not supported for complex dtypes"]
fn oracle_digamma_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[1], vec![(1.0, 0.0)]);
    let result = eval_primitive(Primitive::Digamma, &[input], &no_params())
        .expect("digamma complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
#[ignore = "PARITY GAP: lgamma/digamma not supported for complex dtypes"]
fn property_gamma_funcs_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
            DType::Complex128 => make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
            _ => unreachable!(),
        };
        for primitive in [Primitive::Lgamma, Primitive::Digamma] {
            let result = eval_primitive(primitive, std::slice::from_ref(&input), &no_params())
                .expect("gamma func should succeed for complex dtype");
            assert_eq!(result.dtype(), dtype, "{primitive:?} {dtype:?}: dtype mismatch");
        }
    }
}
