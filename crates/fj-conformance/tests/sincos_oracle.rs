//! Oracle tests for Sin and Cos primitives.
//!
//! sin(x), cos(x) - Trigonometric functions
//!
//! Properties:
//! - sin(0) = 0, cos(0) = 1
//! - sin(π/2) = 1, cos(π/2) = 0
//! - sin(π) = 0, cos(π) = -1
//! - sin^2(x) + cos^2(x) = 1 (Pythagorean identity)
//! - sin(-x) = -sin(x) (odd), cos(-x) = cos(x) (even)
//! - sin(x + 2π) = sin(x) (period 2π)
//!
//! Tests:
//! - Special values at multiples of π/6
//! - Symmetry properties
//! - Pythagorean identity
//! - Periodicity
//! - NaN/infinity propagation
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
            assert!(t.shape.dims.is_empty(), "expected scalar");
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

// ====================== SIN SPECIAL VALUES ======================

#[test]
fn oracle_sin_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "sin(0) = 0");
}

#[test]
fn oracle_sin_pi_over_6() {
    // sin(π/6) = 0.5
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_6]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "sin(π/6) = 0.5");
}

#[test]
fn oracle_sin_pi_over_4() {
    // sin(π/4) = sqrt(2)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_1_SQRT_2,
        1e-14,
        "sin(π/4)",
    );
}

#[test]
fn oracle_sin_pi_over_3() {
    // sin(π/3) = sqrt(3)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_3]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt() / 2.0,
        1e-14,
        "sin(π/3)",
    );
}

#[test]
fn oracle_sin_pi_over_2() {
    // sin(π/2) = 1
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "sin(π/2) = 1");
}

#[test]
fn oracle_sin_pi() {
    // sin(π) = 0
    let input = make_f64_tensor(&[], vec![std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "sin(π) = 0");
}

#[test]
fn oracle_sin_3pi_over_2() {
    // sin(3π/2) = -1
    let input = make_f64_tensor(&[], vec![3.0 * std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "sin(3π/2) = -1");
}

#[test]
fn oracle_sin_2pi() {
    // sin(2π) = 0
    let input = make_f64_tensor(&[], vec![2.0 * std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-13, "sin(2π) = 0");
}

// ====================== COS SPECIAL VALUES ======================

#[test]
fn oracle_cos_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "cos(0) = 1");
}

#[test]
fn oracle_cos_pi_over_6() {
    // cos(π/6) = sqrt(3)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_6]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt() / 2.0,
        1e-14,
        "cos(π/6)",
    );
}

#[test]
fn oracle_cos_pi_over_4() {
    // cos(π/4) = sqrt(2)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_1_SQRT_2,
        1e-14,
        "cos(π/4)",
    );
}

#[test]
fn oracle_cos_pi_over_3() {
    // cos(π/3) = 0.5
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_3]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "cos(π/3) = 0.5");
}

#[test]
fn oracle_cos_pi_over_2() {
    // cos(π/2) = 0
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "cos(π/2) = 0");
}

#[test]
fn oracle_cos_pi() {
    // cos(π) = -1
    let input = make_f64_tensor(&[], vec![std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "cos(π) = -1");
}

#[test]
fn oracle_cos_2pi() {
    // cos(2π) = 1
    let input = make_f64_tensor(&[], vec![2.0 * std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-13, "cos(2π) = 1");
}

// ====================== SIN ODD FUNCTION ======================

#[test]
fn oracle_sin_odd() {
    // sin(-x) = -sin(x)
    for x in [0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4, 2.0] {
        let pos = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let neg = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![-x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(neg, -pos, 1e-14, &format!("sin(-{}) = -sin({})", x, x));
    }
}

// ====================== COS EVEN FUNCTION ======================

#[test]
fn oracle_cos_even() {
    // cos(-x) = cos(x)
    for x in [0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4, 2.0] {
        let pos = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let neg = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![-x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(neg, pos, 1e-14, &format!("cos(-{}) = cos({})", x, x));
    }
}

// ====================== PYTHAGOREAN IDENTITY ======================

#[test]
fn oracle_sin_cos_pythagorean() {
    // sin^2(x) + cos^2(x) = 1
    for x in [0.0, 0.5, 1.0, std::f64::consts::PI, 2.5, -1.0] {
        let sin_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let cos_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let sum = sin_x * sin_x + cos_x * cos_x;
        assert_close(sum, 1.0, 1e-14, &format!("sin^2({}) + cos^2({}) = 1", x, x));
    }
}

// ====================== PERIODICITY ======================

#[test]
fn oracle_sin_periodic() {
    // sin(x + 2π) = sin(x)
    for x in [0.0, 0.5, 1.0, 2.0] {
        let sin_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let sin_x_2pi = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x + 2.0 * std::f64::consts::PI])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(
            sin_x_2pi,
            sin_x,
            1e-13,
            &format!("sin({} + 2π) = sin({})", x, x),
        );
    }
}

#[test]
fn oracle_cos_periodic() {
    // cos(x + 2π) = cos(x)
    for x in [0.0, 0.5, 1.0, 2.0] {
        let cos_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let cos_x_2pi = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x + 2.0 * std::f64::consts::PI])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(
            cos_x_2pi,
            cos_x,
            1e-13,
            &format!("cos({} + 2π) = cos({})", x, x),
        );
    }
}

// ====================== PHASE SHIFT ======================

#[test]
fn oracle_cos_sin_phase_shift() {
    // cos(x) = sin(x + π/2)
    for x in [0.0, 0.5, 1.0, -0.5] {
        let cos_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let sin_x_plus = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x + std::f64::consts::FRAC_PI_2])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(
            cos_x,
            sin_x_plus,
            1e-14,
            &format!("cos({}) = sin({} + π/2)", x, x),
        );
    }
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_sin_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sin(NaN) = NaN");
}

#[test]
fn oracle_cos_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "cos(NaN) = NaN");
}

#[test]
fn oracle_sin_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sin(inf) = NaN");
}

#[test]
fn oracle_cos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "cos(inf) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_sin_range() {
    // sin(x) is always in [-1, 1]
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0] {
        let val = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            (-1.0..=1.0).contains(&val),
            "sin({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

#[test]
fn oracle_cos_range() {
    // cos(x) is always in [-1, 1]
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0] {
        let val = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            (-1.0..=1.0).contains(&val),
            "cos({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_sin_1d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[4], vec![0.0, pi / 6.0, pi / 2.0, pi]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.0, 1e-14, "sin(0)");
    assert_close(vals[1], 0.5, 1e-14, "sin(π/6)");
    assert_close(vals[2], 1.0, 1e-14, "sin(π/2)");
    assert_close(vals[3], 0.0, 1e-14, "sin(π)");
}

#[test]
fn oracle_cos_1d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[4], vec![0.0, pi / 3.0, pi / 2.0, pi]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "cos(0)");
    assert_close(vals[1], 0.5, 1e-14, "cos(π/3)");
    assert_close(vals[2], 0.0, 1e-14, "cos(π/2)");
    assert_close(vals[3], -1.0, 1e-14, "cos(π)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_sincos_2d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[2, 2], vec![0.0, pi / 2.0, pi, 3.0 * pi / 2.0]);
    let sin_result =
        eval_primitive(Primitive::Sin, std::slice::from_ref(&input), &no_params()).unwrap();
    let cos_result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();

    assert_eq!(extract_shape(&sin_result), vec![2, 2]);
    let sin_vals = extract_f64_vec(&sin_result);
    let cos_vals = extract_f64_vec(&cos_result);

    assert_close(sin_vals[0], 0.0, 1e-14, "sin(0)");
    assert_close(sin_vals[1], 1.0, 1e-14, "sin(π/2)");
    assert_close(sin_vals[2], 0.0, 1e-14, "sin(π)");
    assert_close(sin_vals[3], -1.0, 1e-14, "sin(3π/2)");

    assert_close(cos_vals[0], 1.0, 1e-14, "cos(0)");
    assert_close(cos_vals[1], 0.0, 1e-14, "cos(π/2)");
    assert_close(cos_vals[2], -1.0, 1e-14, "cos(π)");
    assert_close(cos_vals[3], 0.0, 1e-14, "cos(3π/2)");
}

// Property sweep across all float dtypes for Sin and Cos. The unary
// transcendental path goes through `eval_unary_elementwise`, whose
// tensor arm was fixed in eldm to preserve input dtype across
// BF16/F16/F32/F64. Pin the dispatch helper against per-arm regressions.
#[test]
fn property_sin_cos_preserves_all_float_dtypes() {
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

    let values = [0.0_f64, 0.5, 1.0, 1.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        for primitive in [Primitive::Sin, Primitive::Cos] {
            let input = make_vec(dtype, &values);
            let result = eval_primitive(primitive, &[input], &BTreeMap::new())
                .unwrap_or_else(|e| panic!("{primitive:?} {dtype:?} failed: {e}"));
            let Value::Tensor(t) = result else {
                panic!("{primitive:?} {dtype:?}: expected tensor");
            };
            assert_eq!(
                t.dtype, dtype,
                "{primitive:?} {dtype:?}: tensor dtype mismatch"
            );
            t.validate_dtype_consistency().unwrap_or_else(|e| {
                panic!("{primitive:?} {dtype:?}: validate_dtype_consistency failed: {e}")
            });
        }
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex sin: sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
// Complex cos: cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)

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
fn oracle_sin_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Sin, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-6,
        "sin(0+0i)",
    );
}

#[test]
fn oracle_cos_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Cos, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (1.0, 0.0),
        1e-6,
        "cos(0+0i)",
    );
}

#[test]
fn oracle_sin_complex128_pure_imaginary() {
    // sin(i*x) = i*sinh(x)
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(0.0, x)]);
    let result = eval_primitive(Primitive::Sin, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.0, x.sinh()), 1e-10, "sin(0+0.5i) = i*sinh(0.5)");
}

#[test]
fn oracle_cos_complex128_pure_imaginary() {
    // cos(i*x) = cosh(x)
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(0.0, x)]);
    let result = eval_primitive(Primitive::Cos, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.cosh(), 0.0), 1e-10, "cos(0+0.5i) = cosh(0.5)");
}

#[test]
fn oracle_sin_complex128_pure_real() {
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Sin, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.sin(), 0.0), 1e-10, "sin(0.5+0i) = sin(0.5)");
}

#[test]
fn oracle_cos_complex128_pure_real() {
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Cos, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.cos(), 0.0), 1e-10, "cos(0.5+0i) = cos(0.5)");
}

#[test]
fn oracle_sin_complex128_general() {
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected_re = a.sin() * b.cosh();
    let expected_im = a.cos() * b.sinh();

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Sin, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (expected_re, expected_im), 1e-10, "sin(0.5+0.3i)");
}

#[test]
fn oracle_cos_complex128_general() {
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected_re = a.cos() * b.cosh();
    let expected_im = -a.sin() * b.sinh();

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Cos, &[input], &BTreeMap::new()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (expected_re, expected_im), 1e-10, "cos(0.5+0.3i)");
}

#[test]
fn oracle_sincos_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.3)];
    let input_sin = make_complex64_tensor(&[4], data);
    let input_cos = make_complex64_tensor(&[4], data);

    let sin_result = eval_primitive(Primitive::Sin, &[input_sin], &BTreeMap::new()).unwrap();
    let cos_result = eval_primitive(Primitive::Cos, &[input_cos], &BTreeMap::new()).unwrap();

    let sin_vec = extract_complex64_vec(&sin_result);
    let cos_vec = extract_complex64_vec(&cos_result);
    assert_eq!(sin_vec.len(), 4);
    assert_eq!(cos_vec.len(), 4);

    // sin(0) = 0, cos(0) = 1
    assert_complex_close(
        (sin_vec[0].0 as f64, sin_vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "sin(0)",
    );
    assert_complex_close(
        (cos_vec[0].0 as f64, cos_vec[0].1 as f64),
        (1.0, 0.0),
        1e-5,
        "cos(0)",
    );

    // sin(0.5) pure real
    assert_complex_close(
        (sin_vec[1].0 as f64, sin_vec[1].1 as f64),
        (0.5_f64.sin(), 0.0),
        1e-4,
        "sin(0.5)",
    );

    // sin(0.5i) = i*sinh(0.5)
    assert_complex_close(
        (sin_vec[2].0 as f64, sin_vec[2].1 as f64),
        (0.0, 0.5_f64.sinh()),
        1e-4,
        "sin(0.5i)",
    );

    // cos(0.5i) = cosh(0.5)
    assert_complex_close(
        (cos_vec[2].0 as f64, cos_vec[2].1 as f64),
        (0.5_f64.cosh(), 0.0),
        1e-4,
        "cos(0.5i)",
    );
}

#[test]
fn oracle_sincos_complex_dtype_preservation() {
    for prim in [Primitive::Sin, Primitive::Cos] {
        // Complex64 -> Complex64
        let c64_input = make_complex64_tensor(&[2], &[(0.5, 0.3), (-0.3, 0.5)]);
        let c64_result = eval_primitive(prim, &[c64_input], &BTreeMap::new()).unwrap();
        match &c64_result {
            Value::Tensor(t) => {
                assert_eq!(
                    t.dtype,
                    DType::Complex64,
                    "{prim:?} should preserve Complex64"
                );
                t.validate_dtype_consistency().unwrap();
            }
            _ => panic!("expected tensor"),
        }

        // Complex128 -> Complex128
        let c128_input = make_complex128_tensor(&[2], &[(0.5, 0.3), (-0.3, 0.5)]);
        let c128_result = eval_primitive(prim, &[c128_input], &BTreeMap::new()).unwrap();
        match &c128_result {
            Value::Tensor(t) => {
                assert_eq!(
                    t.dtype,
                    DType::Complex128,
                    "{prim:?} should preserve Complex128"
                );
                t.validate_dtype_consistency().unwrap();
            }
            _ => panic!("expected tensor"),
        }
    }
}

#[test]
fn oracle_sincos_pythagorean_identity_complex() {
    // sin²(z) + cos²(z) = 1 holds for all complex z
    let z_values: &[(f64, f64)] = &[(0.0, 0.0), (0.5, 0.3), (1.0, -0.5), (-0.3, 0.7)];

    for &z in z_values {
        let input_sin = make_complex128_tensor(&[], &[z]);
        let input_cos = make_complex128_tensor(&[], &[z]);

        let sin_result = eval_primitive(Primitive::Sin, &[input_sin], &BTreeMap::new()).unwrap();
        let cos_result = eval_primitive(Primitive::Cos, &[input_cos], &BTreeMap::new()).unwrap();

        let sin_z = extract_complex128_vec(&sin_result)[0];
        let cos_z = extract_complex128_vec(&cos_result)[0];

        // sin²(z) = sin(z) * sin(z)
        let sin_sq_re = sin_z.0 * sin_z.0 - sin_z.1 * sin_z.1;
        let sin_sq_im = 2.0 * sin_z.0 * sin_z.1;

        // cos²(z) = cos(z) * cos(z)
        let cos_sq_re = cos_z.0 * cos_z.0 - cos_z.1 * cos_z.1;
        let cos_sq_im = 2.0 * cos_z.0 * cos_z.1;

        // sin²(z) + cos²(z) should equal 1 + 0i
        let sum_re = sin_sq_re + cos_sq_re;
        let sum_im = sin_sq_im + cos_sq_im;

        assert_complex_close(
            (sum_re, sum_im),
            (1.0, 0.0),
            1e-10,
            &format!("sin²({:?}) + cos²({:?}) = 1", z, z),
        );
    }
}

// ======================== METAMORPHIC: mathematical identities ========================

#[test]
fn metamorphic_sin_cos_pythagorean() {
    // sin²(x) + cos²(x) = 1 for real x
    let x = make_f64_tensor(&[6], vec![0.0, 0.5, 1.0, 2.0, -1.0, std::f64::consts::PI]);
    let sin_result =
        eval_primitive(Primitive::Sin, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let cos_result =
        eval_primitive(Primitive::Cos, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let sin_vals = extract_f64_vec(&sin_result);
    let cos_vals = extract_f64_vec(&cos_result);
    for (i, (&s, &c)) in sin_vals.iter().zip(cos_vals.iter()).enumerate() {
        let sum = s * s + c * c;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "sin²(x) + cos²(x) should equal 1 at index {i}: got {sum}"
        );
    }
}

#[test]
fn metamorphic_sin_odd_symmetry() {
    // sin(-x) = -sin(x)
    let x = make_f64_tensor(&[5], vec![0.5, 1.0, 2.0, -1.5, 3.0]);
    let neg_x = make_f64_tensor(&[5], vec![-0.5, -1.0, -2.0, 1.5, -3.0]);
    let sin_x = eval_primitive(Primitive::Sin, &[x], &BTreeMap::new()).unwrap();
    let sin_neg_x = eval_primitive(Primitive::Sin, &[neg_x], &BTreeMap::new()).unwrap();
    let vals_x = extract_f64_vec(&sin_x);
    let vals_neg_x = extract_f64_vec(&sin_neg_x);
    for (i, (&sx, &snx)) in vals_x.iter().zip(vals_neg_x.iter()).enumerate() {
        assert!(
            (sx + snx).abs() < 1e-10,
            "sin(-x) should equal -sin(x) at index {i}: sin(x)={sx}, sin(-x)={snx}"
        );
    }
}

#[test]
fn metamorphic_cos_even_symmetry() {
    // cos(-x) = cos(x)
    let x = make_f64_tensor(&[5], vec![0.5, 1.0, 2.0, -1.5, 3.0]);
    let neg_x = make_f64_tensor(&[5], vec![-0.5, -1.0, -2.0, 1.5, -3.0]);
    let cos_x = eval_primitive(Primitive::Cos, &[x], &BTreeMap::new()).unwrap();
    let cos_neg_x = eval_primitive(Primitive::Cos, &[neg_x], &BTreeMap::new()).unwrap();
    let vals_x = extract_f64_vec(&cos_x);
    let vals_neg_x = extract_f64_vec(&cos_neg_x);
    for (i, (&cx, &cnx)) in vals_x.iter().zip(vals_neg_x.iter()).enumerate() {
        assert!(
            (cx - cnx).abs() < 1e-10,
            "cos(-x) should equal cos(x) at index {i}: cos(x)={cx}, cos(-x)={cnx}"
        );
    }
}

#[test]
fn metamorphic_sincos_bounded_output() {
    // |sin(x)| <= 1 and |cos(x)| <= 1 for real x
    let x = make_f64_tensor(&[6], vec![0.0, 1.0, 10.0, -100.0, 1000.0, -0.001]);
    let sin_result =
        eval_primitive(Primitive::Sin, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let cos_result =
        eval_primitive(Primitive::Cos, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let sin_vals = extract_f64_vec(&sin_result);
    let cos_vals = extract_f64_vec(&cos_result);
    for (i, (&s, &c)) in sin_vals.iter().zip(cos_vals.iter()).enumerate() {
        assert!(
            s.abs() <= 1.0 + 1e-10,
            "|sin(x)| should be <= 1 at index {i}: got {}",
            s.abs()
        );
        assert!(
            c.abs() <= 1.0 + 1e-10,
            "|cos(x)| should be <= 1 at index {i}: got {}",
            c.abs()
        );
    }
}
