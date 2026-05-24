//! Oracle tests for Atan primitive.
//!
//! atan(x) = arctangent of x, returns angle in radians
//!
//! Domain: (-inf, inf)
//! Range: (-π/2, π/2)
//!
//! Tests:
//! - atan(0) = 0
//! - atan(1) = π/4
//! - atan(-1) = -π/4
//! - atan(inf) = π/2
//! - atan(-inf) = -π/2
//! - atan(NaN) = NaN
//! - Odd function: atan(-x) = -atan(x)
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
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

fn assert_same_f64_bits(actual: f64, expected: f64, msg: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{msg}: expected bits {:#018x}, got {:#018x}",
        expected.to_bits(),
        actual.to_bits()
    );
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_atan_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "atan(0) = +0");
}

#[test]
fn oracle_atan_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "atan(-0.0) = -0");
}

#[test]
fn oracle_atan_tensor_signed_zero_bits() {
    let input = make_f64_tensor(&[2], vec![0.0, -0.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);

    assert_same_f64_bits(vals[0], 0.0, "atan(+0.0) tensor lane");
    assert_same_f64_bits(vals[1], -0.0, "atan(-0.0) tensor lane");
}

#[test]
fn oracle_atan_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan(1) = π/4",
    );
}

#[test]
fn oracle_atan_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_4,
        1e-14,
        "atan(-1) = -π/4",
    );
}

#[test]
fn oracle_atan_sqrt3() {
    // atan(sqrt(3)) = π/3
    let input = make_f64_tensor(&[], vec![3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_3,
        1e-14,
        "atan(sqrt(3)) = π/3",
    );
}

#[test]
fn oracle_atan_one_over_sqrt3() {
    // atan(1/sqrt(3)) = π/6
    let input = make_f64_tensor(&[], vec![1.0 / 3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_6,
        1e-14,
        "atan(1/sqrt(3)) = π/6",
    );
}

// ====================== INFINITY ======================

#[test]
fn oracle_atan_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan(inf) = π/2",
    );
}

#[test]
fn oracle_atan_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "atan(-inf) = -π/2",
    );
}

// ====================== NaN ======================

#[test]
fn oracle_atan_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atan(NaN) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_atan_range() {
    // atan(x) should always be in (-π/2, π/2)
    let test_values = [-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0];
    for x in test_values {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            val > -std::f64::consts::FRAC_PI_2 && val < std::f64::consts::FRAC_PI_2,
            "atan({}) = {} should be in (-π/2, π/2)",
            x,
            val
        );
    }
}

// ====================== ODD FUNCTION ======================

#[test]
fn oracle_atan_odd_function() {
    // atan(-x) = -atan(x)
    for x in [0.5, 1.0, 2.0, 10.0, 100.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Atan, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Atan, &[neg_input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&neg_result),
            -extract_f64_scalar(&pos_result),
            1e-14,
            &format!("atan(-{}) = -atan({})", x, x),
        );
    }
}

// ====================== TAN INVERSE ======================

#[test]
fn oracle_atan_tan_inverse() {
    // tan(atan(x)) = x for all x
    for x in [0.0, 0.5, 1.0, 2.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let atan_result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let tan_result = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&tan_result),
            x,
            1e-13,
            &format!("tan(atan({})) = {}", x, x),
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_atan_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, 0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -std::f64::consts::FRAC_PI_4, 1e-14, "atan(-1)");
    assert_eq!(vals[1], 0.0, "atan(0)");
    assert_close(vals[2], std::f64::consts::FRAC_PI_4, 1e-14, "atan(1)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_2, 1e-14, "atan(inf)");
    assert_close(vals[4], -std::f64::consts::FRAC_PI_2, 1e-14, "atan(-inf)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_atan_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 3.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0);
    assert_close(vals[1], std::f64::consts::FRAC_PI_4, 1e-14, "atan(1)");
    assert_close(vals[2], -std::f64::consts::FRAC_PI_4, 1e-14, "atan(-1)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_3, 1e-14, "atan(sqrt(3))");
}

// ====================== ASYMPTOTIC BEHAVIOR ======================

#[test]
fn oracle_atan_large_values() {
    // For large |x|, atan(x) approaches ±π/2
    for x in [100.0, 1000.0, 1e10] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val - std::f64::consts::FRAC_PI_2).abs() < 0.01,
            "atan({}) should be close to π/2",
            x
        );
    }
}

#[test]
fn oracle_atan_small_values() {
    // For small |x|, atan(x) ≈ x
    for x in [0.001, 0.0001, 1e-10] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert_close(val, x, x * 0.01, &format!("atan({}) ≈ {}", x, x));
    }
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_atan_monotonic() {
    // atan is strictly increasing
    let values: Vec<f64> = vec![-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "atan should be strictly increasing: atan[{}] = {} > atan[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ======================== METAMORPHIC: tan(atan(x)) = x ========================

#[test]
fn metamorphic_tan_atan_identity() {
    // tan(atan(x)) = x for all real x
    // Use relative tolerance for large values where absolute error grows
    for x in [-100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let atan_result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let tan_atan = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();

        let tol = if x.abs() > 10.0 { 1e-10 } else { 1e-12 };
        assert_close(
            extract_f64_scalar(&tan_atan),
            x,
            tol,
            &format!("tan(atan({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: atan(tan(x)) = x for x in (-π/2, π/2) ========================

#[test]
fn metamorphic_atan_tan_identity() {
    // atan(tan(x)) = x for x in (-π/2, π/2)
    // Use values well within the domain to avoid precision issues near boundaries
    for x in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let tan_result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
        let atan_tan = eval_primitive(Primitive::Atan, &[tan_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&atan_tan),
            x,
            1e-12,
            &format!("atan(tan({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_atan_tensor_roundtrip() {
    let input = make_f64_tensor(&[5], vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let atan_result =
        eval_primitive(Primitive::Atan, std::slice::from_ref(&input), &no_params()).unwrap();
    let tan_atan = eval_primitive(Primitive::Tan, &[atan_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&tan_atan);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(
            *rt,
            *orig,
            1e-12,
            &format!("tan(atan({})) = {}", orig, orig),
        );
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_atan_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-1.0, 0.0, 1.0, 2.0, -2.0, 0.5, -0.5, 3.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], (-1.0_f64).atan(), 1e-14, "atan(-1)");
    assert_close(vals[7], 3.0_f64.atan(), 1e-14, "atan(3)");
}

#[test]
fn oracle_atan_empty() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_atan_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_atan_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_atan_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![subnormal, -subnormal]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // For tiny x, atan(x) ≈ x
    assert_close(vals[0], subnormal.atan(), 1e-30, "atan(subnormal)");
    assert_close(vals[1], (-subnormal).atan(), 1e-30, "atan(-subnormal)");
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================

fn make_complex64_scalar(re: f32, im: f32) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![] },
            vec![Literal::from_complex64(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex128_scalar(re: f64, im: f64) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![] },
            vec![Literal::from_complex128(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex64_tensor(shape: &[u32], pairs: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

fn assert_complex128_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

#[test]
fn oracle_atan_complex64_zero() {
    // atan(0+0i) = 0+0i
    let input = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 0.0), 1e-6, "atan(0+0i)");
}

#[test]
fn oracle_atan_complex64_real() {
    // atan(1+0i) = pi/4 + 0i
    let input = make_complex64_scalar(1.0, 0.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close(
        (re, im),
        (std::f32::consts::FRAC_PI_4, 0.0),
        1e-6,
        "atan(1+0i) = pi/4",
    );
}

#[test]
fn oracle_atan_complex64_pure_imaginary() {
    // For pure imaginary z = iy: atan(iy) = i * atanh(y)
    // atan(0.5i) = i * atanh(0.5) ≈ 0 + 0.5493i
    let y = 0.5_f32;
    let atanh_y = 0.5 * ((1.0 + y) / (1.0 - y)).ln();
    let input = make_complex64_scalar(0.0, y);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, atanh_y), 1e-5, "atan(0.5i) = i*atanh(0.5)");
}

#[test]
fn oracle_atan_complex64_general() {
    // atan(1+i): compute expected using formula
    // atan(z) = (1/2i) * ln((i+z)/(i-z))
    // For z = 1+i:
    // i+z = 1+2i, i-z = -1, (i+z)/(i-z) = -(1+2i) = -1-2i
    // ln(-1-2i) = ln(sqrt(5)) + i*atan2(-2,-1) = ln(sqrt(5)) + i*(-2.034...)
    // atan(1+i) ≈ 1.0172 + 0.4024i
    let input = make_complex64_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    // Reference values from numpy: np.arctan(1+1j) ≈ 1.0172 + 0.4024i
    assert_complex64_close((re, im), (1.0172, 0.4024), 0.001, "atan(1+i)");
}

#[test]
fn oracle_atan_complex64_vector() {
    let input = make_complex64_tensor(&[3], vec![(0.0, 0.0), (1.0, 0.0), (0.0, 0.5)]);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex64_vec(&result);

    // atan(0) = 0
    assert_complex64_close(vals[0], (0.0, 0.0), 1e-6, "atan(0)");
    // atan(1) = pi/4
    assert_complex64_close(vals[1], (std::f32::consts::FRAC_PI_4, 0.0), 1e-5, "atan(1)");
    // atan(0.5i) = i*atanh(0.5)
    let atanh_05 = 0.5_f32 * ((1.0_f32 + 0.5) / (1.0_f32 - 0.5)).ln();
    assert_complex64_close(vals[2], (0.0, atanh_05), 1e-5, "atan(0.5i)");
}

#[test]
fn oracle_atan_complex128_general() {
    // atan(1+i) ≈ 1.0172219678978514 + 0.40235947810852507i
    let input = make_complex128_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close(
        (re, im),
        (1.0172219678978514, 0.40235947810852507),
        1e-10,
        "atan(1+i) Complex128",
    );
}

#[test]
fn oracle_atan_complex64_tan_inverse_identity() {
    // tan(atan(z)) = z for complex z (away from branch cuts)
    let z = make_complex64_scalar(0.5, 0.3);
    let atan_z = eval_primitive(Primitive::Atan, &[z], &no_params()).unwrap();
    let tan_atan_z = eval_primitive(Primitive::Tan, &[atan_z], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&tan_atan_z);
    assert_complex64_close((re, im), (0.5, 0.3), 1e-5, "tan(atan(0.5+0.3i)) = 0.5+0.3i");
}

#[test]
fn oracle_atan_complex64_preserves_dtype() {
    let input = make_complex64_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_atan_complex128_preserves_dtype() {
    let input = make_complex128_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

// ======================== Property: dtype preservation across all float types ========================

#[test]
fn property_atan_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![values.len() as u32],
                },
                values.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    // atan domain is all reals
    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Atan, &[input], &no_params())
            .unwrap_or_else(|e| panic!("atan {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("atan {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "atan {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("atan {dtype:?}: validate_dtype_consistency failed: {e}"));
    }
}
