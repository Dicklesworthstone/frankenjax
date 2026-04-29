//! Oracle tests for Real and Imag primitives.
//!
//! Real: Extract real part from complex numbers
//! Imag: Extract imaginary part from complex numbers
//!
//! For complex number z = a + bi:
//! - Real(z) = a
//! - Imag(z) = b
//!
//! Tests:
//! - Pure real numbers
//! - Pure imaginary numbers
//! - Mixed complex numbers
//! - Zero
//! - Tensor shapes

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            match &t.elements[0] {
                Literal::F32Bits(bits) => f32::from_bits(*bits) as f64,
                Literal::F64Bits(bits) => f64::from_bits(*bits),
                _ => panic!("expected float"),
            }
        }
        Value::Scalar(Literal::F32Bits(bits)) => f32::from_bits(*bits) as f64,
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => f32::from_bits(*bits) as f64,
                Literal::F64Bits(bits) => f64::from_bits(*bits),
                _ => panic!("expected float"),
            })
            .collect(),
        _ => panic!("expected tensor"),
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

fn assert_close_f64(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}",
        msg,
        expected,
        actual
    );
}

// ====================== REAL: PURE REAL NUMBERS ======================

#[test]
fn oracle_real_pure_real_c64() {
    // Real(3 + 0i) = 3
    let input = make_complex64_tensor(&[], vec![(3.0, 0.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_close_f64(extract_f64_scalar(&result), 3.0, 1e-6, "Real(3+0i)");
}

#[test]
fn oracle_real_pure_real_c128() {
    let input = make_complex128_tensor(&[], vec![(5.0, 0.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 5.0, 1e-14, "Real(5+0i)");
}

// ====================== REAL: PURE IMAGINARY ======================

#[test]
fn oracle_real_pure_imaginary_c64() {
    // Real(0 + 4i) = 0
    let input = make_complex64_tensor(&[], vec![(0.0, 4.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "Real(0+4i) = 0");
}

#[test]
fn oracle_real_pure_imaginary_c128() {
    let input = make_complex128_tensor(&[], vec![(0.0, 7.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "Real(0+7i) = 0");
}

// ====================== REAL: MIXED ======================

#[test]
fn oracle_real_mixed_c64() {
    // Real(3 + 4i) = 3
    let input = make_complex64_tensor(&[], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 3.0, 1e-6, "Real(3+4i)");
}

#[test]
fn oracle_real_mixed_c128() {
    let input = make_complex128_tensor(&[], vec![(1.5, 2.5)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 1.5, 1e-14, "Real(1.5+2.5i)");
}

// ====================== IMAG: PURE REAL NUMBERS ======================

#[test]
fn oracle_imag_pure_real_c64() {
    // Imag(3 + 0i) = 0
    let input = make_complex64_tensor(&[], vec![(3.0, 0.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "Imag(3+0i) = 0");
}

#[test]
fn oracle_imag_pure_real_c128() {
    let input = make_complex128_tensor(&[], vec![(5.0, 0.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "Imag(5+0i) = 0");
}

// ====================== IMAG: PURE IMAGINARY ======================

#[test]
fn oracle_imag_pure_imaginary_c64() {
    // Imag(0 + 4i) = 4
    let input = make_complex64_tensor(&[], vec![(0.0, 4.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 4.0, 1e-6, "Imag(0+4i)");
}

#[test]
fn oracle_imag_pure_imaginary_c128() {
    let input = make_complex128_tensor(&[], vec![(0.0, 7.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 7.0, 1e-14, "Imag(0+7i)");
}

// ====================== IMAG: MIXED ======================

#[test]
fn oracle_imag_mixed_c64() {
    // Imag(3 + 4i) = 4
    let input = make_complex64_tensor(&[], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 4.0, 1e-6, "Imag(3+4i)");
}

#[test]
fn oracle_imag_mixed_c128() {
    let input = make_complex128_tensor(&[], vec![(1.5, 2.5)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), 2.5, 1e-14, "Imag(1.5+2.5i)");
}

// ====================== ZERO ======================

#[test]
fn oracle_real_zero_c64() {
    let input = make_complex64_tensor(&[], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

#[test]
fn oracle_imag_zero_c64() {
    let input = make_complex64_tensor(&[], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

// ====================== NEGATIVE VALUES ======================

#[test]
fn oracle_real_negative_c64() {
    let input = make_complex64_tensor(&[], vec![(-3.0, 4.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), -3.0, 1e-6, "Real(-3+4i)");
}

#[test]
fn oracle_imag_negative_c64() {
    let input = make_complex64_tensor(&[], vec![(3.0, -4.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_close_f64(extract_f64_scalar(&result), -4.0, 1e-6, "Imag(3-4i)");
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_real_1d_c64() {
    let input = make_complex64_tensor(&[4], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close_f64(vals[0], 1.0, 1e-6, "");
    assert_close_f64(vals[1], 3.0, 1e-6, "");
    assert_close_f64(vals[2], 5.0, 1e-6, "");
    assert_close_f64(vals[3], 7.0, 1e-6, "");
}

#[test]
fn oracle_imag_1d_c64() {
    let input = make_complex64_tensor(&[4], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close_f64(vals[0], 2.0, 1e-6, "");
    assert_close_f64(vals[1], 4.0, 1e-6, "");
    assert_close_f64(vals[2], 6.0, 1e-6, "");
    assert_close_f64(vals[3], 8.0, 1e-6, "");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_real_2d_c128() {
    let input = make_complex128_tensor(&[2, 2], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Real, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close_f64(vals[0], 1.0, 1e-14, "");
    assert_close_f64(vals[1], 3.0, 1e-14, "");
    assert_close_f64(vals[2], 5.0, 1e-14, "");
    assert_close_f64(vals[3], 7.0, 1e-14, "");
}

#[test]
fn oracle_imag_2d_c128() {
    let input = make_complex128_tensor(&[2, 2], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close_f64(vals[0], 2.0, 1e-14, "");
    assert_close_f64(vals[1], 4.0, 1e-14, "");
    assert_close_f64(vals[2], 6.0, 1e-14, "");
    assert_close_f64(vals[3], 8.0, 1e-14, "");
}

// ====================== MATHEMATICAL PROPERTIES ======================

#[test]
fn oracle_real_imag_magnitude() {
    // |z|^2 = Real(z)^2 + Imag(z)^2
    let z = (3.0, 4.0); // |z| = 5
    let input = make_complex64_tensor(&[], vec![z]);
    let real_result = eval_primitive(Primitive::Real, &[input.clone()], &no_params()).unwrap();
    let imag_result = eval_primitive(Primitive::Imag, &[input], &no_params()).unwrap();
    let re = extract_f64_scalar(&real_result);
    let im = extract_f64_scalar(&imag_result);
    let mag_squared = re * re + im * im;
    assert_close_f64(mag_squared, 25.0, 1e-5, "|3+4i|^2 = 25");
}

#[test]
fn oracle_real_imag_conjugate_property() {
    // For z = a + bi, conj(z) = a - bi
    // So Real(conj(z)) = Real(z) and Imag(conj(z)) = -Imag(z)
    // This tests that Real and Imag correctly extract parts
    let z = (3.0, 4.0);
    let conj_z = (3.0, -4.0);

    let z_input = make_complex64_tensor(&[], vec![z]);
    let conj_input = make_complex64_tensor(&[], vec![conj_z]);

    let real_z = extract_f64_scalar(&eval_primitive(Primitive::Real, &[z_input.clone()], &no_params()).unwrap());
    let real_conj = extract_f64_scalar(&eval_primitive(Primitive::Real, &[conj_input.clone()], &no_params()).unwrap());
    let imag_z = extract_f64_scalar(&eval_primitive(Primitive::Imag, &[z_input], &no_params()).unwrap());
    let imag_conj = extract_f64_scalar(&eval_primitive(Primitive::Imag, &[conj_input], &no_params()).unwrap());

    assert_close_f64(real_z, real_conj, 1e-6, "Real(z) = Real(conj(z))");
    assert_close_f64(imag_z, -imag_conj, 1e-6, "Imag(z) = -Imag(conj(z))");
}
