//! Oracle tests for Cbrt primitive.
//!
//! Tests against expected behavior matching JAX/lax.cbrt:
//! - Computes cube root of each element
//! - cbrt(x^3) = x for all real x
//! - Preserves IEEE signed-zero bit patterns

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

// ======================== Scalar Tests ========================

#[test]
fn oracle_cbrt_scalar_8() {
    // cbrt(8) = 2
    let input = Value::Scalar(Literal::from_f64(8.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_scalar_27() {
    // cbrt(27) = 3
    let input = Value::Scalar(Literal::from_f64(27.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_scalar_1() {
    // cbrt(1) = 1
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_scalar_0() {
    // cbrt(0) = 0
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0].to_bits(), 0.0_f64.to_bits(), "cbrt(+0.0) = +0.0");
}

#[test]
fn oracle_cbrt_scalar_negative() {
    // cbrt(-8) = -2 (unlike sqrt, cbrt handles negatives)
    let input = Value::Scalar(Literal::from_f64(-8.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-2.0)).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_scalar_negative_27() {
    // cbrt(-27) = -3
    let input = Value::Scalar(Literal::from_f64(-27.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-3.0)).abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_cbrt_1d_perfect_cubes() {
    let input = make_f64_tensor(&[4], vec![1.0, 8.0, 27.0, 64.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 3.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_1d_mixed() {
    let input = make_f64_tensor(&[5], vec![-27.0, -8.0, 0.0, 8.0, 27.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-3.0)).abs() < 1e-10);
    assert!((vals[1] - (-2.0)).abs() < 1e-10);
    assert_eq!(vals[2].to_bits(), 0.0_f64.to_bits(), "cbrt(+0.0) = +0.0");
    assert!((vals[3] - 2.0).abs() < 1e-10);
    assert!((vals[4] - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_1d_fractional() {
    // cbrt(0.125) = 0.5
    let input = make_f64_tensor(&[2], vec![0.125, 0.001]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.5).abs() < 1e-10);
    assert!((vals[1] - 0.1).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_cbrt_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 8.0, 27.0, 64.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 3.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_2d_negative() {
    let input = make_f64_tensor(&[2, 2], vec![-1.0, -8.0, -27.0, -64.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.0)).abs() < 1e-10);
    assert!((vals[1] - (-2.0)).abs() < 1e-10);
    assert!((vals[2] - (-3.0)).abs() < 1e-10);
    assert!((vals[3] - (-4.0)).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_cbrt_large() {
    // cbrt(1000000) = 100
    let input = Value::Scalar(Literal::from_f64(1000000.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 100.0).abs() < 1e-6);
}

#[test]
fn oracle_cbrt_small() {
    // cbrt(1e-9) = 1e-3
    let input = Value::Scalar(Literal::from_f64(1e-9));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1e-3).abs() < 1e-12);
}

#[test]
fn oracle_cbrt_inverse_of_cube() {
    // cbrt(x^3) = x
    let bases: Vec<f64> = vec![2.5, 3.7, -1.5, 0.0, 100.0];
    let cubes: Vec<f64> = bases.iter().map(|x| x.powi(3)).collect();
    let input = make_f64_tensor(&[5], cubes);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    for (v, b) in vals.iter().zip(bases.iter()) {
        assert!((v - b).abs() < 1e-10);
    }
}

#[test]
fn oracle_cbrt_single_element() {
    let input = make_f64_tensor(&[1], vec![125.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.0).abs() < 1e-10);
}

// ======================== Special Values ========================

#[test]
fn oracle_cbrt_positive_infinity() {
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val > 0.0, "cbrt(+inf) = +inf");
}

#[test]
fn oracle_cbrt_negative_infinity() {
    let input = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_infinite() && val < 0.0, "cbrt(-inf) = -inf");
}

#[test]
fn oracle_cbrt_nan() {
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert!(val.is_nan(), "cbrt(NaN) = NaN");
}

#[test]
fn oracle_cbrt_negative_zero() {
    let input = Value::Scalar(Literal::from_f64(-0.0));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    assert_eq!(val.to_bits(), (-0.0_f64).to_bits(), "cbrt(-0.0) = -0.0");
}

#[test]
fn oracle_cbrt_tensor_special_values() {
    let input = make_f64_tensor(&[4], vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert!(vals[0].is_infinite() && vals[0] > 0.0, "cbrt(+inf) = +inf");
    assert!(vals[1].is_infinite() && vals[1] < 0.0, "cbrt(-inf) = -inf");
    assert!(vals[2].is_nan(), "cbrt(NaN) = NaN");
    assert_eq!(vals[3].to_bits(), (-0.0_f64).to_bits(), "cbrt(-0.0) = -0.0");
}

// ======================== METAMORPHIC: cbrt(x)^3 = x ========================

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

#[test]
fn metamorphic_cbrt_cubed_identity() {
    // cbrt(x)^3 = x for all real x, using Mul primitive for cubing
    for x in [-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0, 64.0, 125.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let cbrt_result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
        // Cube: cbrt(x) * cbrt(x) * cbrt(x)
        let squared = eval_primitive(
            Primitive::Mul,
            &[cbrt_result.clone(), cbrt_result.clone()],
            &no_params(),
        )
        .unwrap();
        let cubed = eval_primitive(Primitive::Mul, &[squared, cbrt_result], &no_params()).unwrap();

        assert_close(
            extract_f64_vec(&cubed)[0],
            x,
            1e-10,
            &format!("cbrt({})^3 = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: cbrt(x*x*x) = x ========================

#[test]
fn metamorphic_cube_cbrt_identity() {
    // cbrt(x*x*x) = x for all real x, using Mul primitive for cubing
    for x in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        // Cube: x * x * x
        let squared = eval_primitive(
            Primitive::Mul,
            &[input.clone(), input.clone()],
            &no_params(),
        )
        .unwrap();
        let cubed = eval_primitive(Primitive::Mul, &[squared, input], &no_params()).unwrap();
        let cbrt_cubed = eval_primitive(Primitive::Cbrt, &[cubed], &no_params()).unwrap();

        assert_close(
            extract_f64_vec(&cbrt_cubed)[0],
            x,
            1e-10,
            &format!("cbrt({}^3) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_cbrt_tensor_roundtrip() {
    // For a tensor: cbrt(x)^3 = x
    let input = make_f64_tensor(&[6], vec![-27.0, -8.0, 0.0, 1.0, 8.0, 27.0]);
    let cbrt_result =
        eval_primitive(Primitive::Cbrt, std::slice::from_ref(&input), &no_params()).unwrap();
    let squared = eval_primitive(
        Primitive::Mul,
        &[cbrt_result.clone(), cbrt_result.clone()],
        &no_params(),
    )
    .unwrap();
    let cubed = eval_primitive(Primitive::Mul, &[squared, cbrt_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&cubed);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(*rt, *orig, 1e-10, &format!("cbrt({})^3 = {}", orig, orig));
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_cbrt_3d_shape() {
    let input = make_f64_tensor(
        &[2, 2, 2],
        vec![1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0],
    );
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[7] - 8.0).abs() < 1e-10);
}

#[test]
fn oracle_cbrt_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_cbrt_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_cbrt_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![1.0, 8.0, 27.0]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_cbrt_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = Value::Scalar(Literal::from_f64(subnormal));
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let val = extract_f64_vec(&result)[0];
    // cbrt of subnormal should be a small positive value
    assert!(
        val > 0.0 && val.is_finite(),
        "cbrt(subnormal) should be small positive finite"
    );
}

#[test]
fn oracle_cbrt_4d_shape() {
    let data: Vec<f64> = (1..=16).map(|x| (x as f64).powi(3)).collect();
    let input = make_f64_tensor(&[2, 2, 2, 2], data);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[15] - 16.0).abs() < 1e-10);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_cbrt_preserves_all_float_dtypes() {
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

    let values = [-1.0_f64, 0.0, 8.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "cbrt {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex cbrt: principal cube root of a complex number

fn make_complex64_tensor(shape: &[u32], data: &[(f32, f32)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: &[(f64, f64)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex64Bits(re, im) => (f32::from_bits(*re), f32::from_bits(*im)),
                _ => panic!("expected Complex64"),
            })
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let (ar, ai) = actual;
    let (er, ei) = expected;
    let re_diff = (ar - er).abs();
    let im_diff = (ai - ei).abs();
    assert!(
        re_diff < tol && im_diff < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        er,
        ei,
        ar,
        ai,
        re_diff,
        im_diff
    );
}

#[test]
fn oracle_cbrt_complex128_real_positive() {
    // cbrt(8+0i) = 2+0i
    let input = make_complex128_tensor(&[], &[(8.0, 0.0)]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (2.0, 0.0), 1e-10, "cbrt(8+0i) = 2");
}

#[test]
fn oracle_cbrt_complex128_real_negative() {
    // cbrt(-8+0i) should give principal root (not -2 which is real but not principal)
    // Principal cube root of -8 = 8 * e^(i*pi) has principal root 2 * e^(i*pi/3)
    // = 2 * (cos(pi/3) + i*sin(pi/3)) = 2 * (0.5 + i*sqrt(3)/2) = 1 + i*sqrt(3)
    let input = make_complex128_tensor(&[], &[(-8.0, 0.0)]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    let expected = (1.0, 3.0_f64.sqrt());
    assert_complex_close(vec[0], expected, 1e-10, "cbrt(-8+0i) = 1 + sqrt(3)i");
}

#[test]
fn oracle_cbrt_complex128_pure_imaginary() {
    // cbrt(i) = cbrt(e^(i*pi/2)) = e^(i*pi/6) = cos(pi/6) + i*sin(pi/6) = sqrt(3)/2 + 0.5i
    let input = make_complex128_tensor(&[], &[(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    let expected = (3.0_f64.sqrt() / 2.0, 0.5);
    assert_complex_close(vec[0], expected, 1e-10, "cbrt(i)");
}

#[test]
fn oracle_cbrt_complex64_vector() {
    let data: &[(f32, f32)] = &[(8.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
    let input = make_complex64_tensor(&[3], data);
    let result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 3);

    // cbrt(8) = 2
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (2.0, 0.0),
        1e-4,
        "cbrt(8)",
    );

    // cbrt(1) = 1
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (1.0, 0.0),
        1e-4,
        "cbrt(1)",
    );

    // cbrt(i) ≈ (sqrt(3)/2, 0.5)
    let expected_i = (3.0_f64.sqrt() / 2.0, 0.5);
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        expected_i,
        1e-4,
        "cbrt(i)",
    );
}

#[test]
fn oracle_cbrt_complex_cubed_identity() {
    // cbrt(z)³ = z
    let values: &[(f64, f64)] = &[(8.0, 0.0), (1.0, 1.0), (2.0, 3.0)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);
        let cbrt_result = eval_primitive(Primitive::Cbrt, &[input], &no_params()).unwrap();
        let cbrt_val = extract_complex128_vec(&cbrt_result)[0];

        // Cube the result: z³ = z * z * z
        let z_sq_re = cbrt_val.0 * cbrt_val.0 - cbrt_val.1 * cbrt_val.1;
        let z_sq_im = 2.0 * cbrt_val.0 * cbrt_val.1;
        let z_cubed_re = z_sq_re * cbrt_val.0 - z_sq_im * cbrt_val.1;
        let z_cubed_im = z_sq_re * cbrt_val.1 + z_sq_im * cbrt_val.0;

        assert_complex_close(
            (z_cubed_re, z_cubed_im),
            (a, b),
            1e-9,
            &format!("cbrt({a}+{b}i)³ = {a}+{b}i"),
        );
    }
}

#[test]
fn oracle_cbrt_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(8.0, 0.0), (1.0, 1.0)]);
    let c64_result = eval_primitive(Primitive::Cbrt, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "cbrt should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(8.0, 0.0), (1.0, 1.0)]);
    let c128_result = eval_primitive(Primitive::Cbrt, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "cbrt should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
