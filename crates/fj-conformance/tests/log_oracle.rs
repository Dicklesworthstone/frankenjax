//! Oracle tests for Log primitive.
//!
//! log(x) = ln(x) (natural logarithm)
//!
//! Domain: (0, inf)
//! Range: (-inf, inf)
//!
//! Properties:
//! - log(1) = 0
//! - log(e) = 1
//! - log(x * y) = log(x) + log(y) (product rule)
//! - log(x / y) = log(x) - log(y)
//! - log(x^n) = n * log(x)
//! - log(exp(x)) = x
//!
//! Tests:
//! - Special values (1, e)
//! - Domain boundaries
//! - Mathematical properties
//! - Monotonicity
//! - Complex numbers
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
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_complex_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            match &t.elements[0] {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                _ => panic!("expected complex128"),
            }
        }
        Value::Scalar(Literal::Complex128Bits(re, im)) => (f64::from_bits(*re), f64::from_bits(*im)),
        _ => panic!("expected complex128"),
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

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_log_one() {
    // log(1) = 0
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "log(1) = 0");
}

#[test]
fn oracle_log_e() {
    // log(e) = 1
    let input = make_f64_tensor(&[], vec![std::f64::consts::E]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "log(e) = 1");
}

#[test]
fn oracle_log_e_squared() {
    // log(e^2) = 2
    let e = std::f64::consts::E;
    let input = make_f64_tensor(&[], vec![e * e]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "log(e^2) = 2");
}

#[test]
fn oracle_log_10() {
    // log(10) = ln(10) ≈ 2.302585
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::LN_10,
        1e-14,
        "log(10)",
    );
}

#[test]
fn oracle_log_2() {
    // log(2) = ln(2) ≈ 0.693147
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::LN_2,
        1e-14,
        "log(2)",
    );
}

// ====================== DOMAIN BOUNDARIES ======================

#[test]
fn oracle_log_zero() {
    // log(0) = -infinity
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY, "log(0) = -inf");
}

#[test]
fn oracle_log_negative() {
    // log(-1) = NaN (for real numbers)
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log(-1) = NaN");
}

#[test]
fn oracle_log_very_small() {
    // log of very small positive number is very negative
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val < -200.0, "log(1e-100) should be very negative");
}

#[test]
fn oracle_log_very_large() {
    // log of very large number is moderately large
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 100.0 * std::f64::consts::LN_10, 1e-10, "log(1e100)");
}

// ====================== SPECIAL FLOAT VALUES ======================

#[test]
fn oracle_log_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY, "log(inf) = inf");
}

#[test]
fn oracle_log_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log(NaN) = NaN");
}

// ====================== PRODUCT RULE ======================

#[test]
fn oracle_log_product_rule() {
    // log(x * y) = log(x) + log(y)
    let test_pairs = [(2.0, 3.0), (10.0, 100.0), (std::f64::consts::E, 2.0)];
    for (x, y) in test_pairs {
        let product = x * y;
        let log_product = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![product])], &no_params()).unwrap(),
        );
        let log_x = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let log_y = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![y])], &no_params()).unwrap(),
        );
        assert_close(
            log_product,
            log_x + log_y,
            1e-13,
            &format!("log({} * {}) = log({}) + log({})", x, y, x, y),
        );
    }
}

// ====================== QUOTIENT RULE ======================

#[test]
fn oracle_log_quotient_rule() {
    // log(x / y) = log(x) - log(y)
    let test_pairs = [(10.0, 2.0), (100.0, 10.0), (std::f64::consts::E, 2.0)];
    for (x, y) in test_pairs {
        let quotient = x / y;
        let log_quotient = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![quotient])], &no_params()).unwrap(),
        );
        let log_x = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let log_y = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![y])], &no_params()).unwrap(),
        );
        assert_close(
            log_quotient,
            log_x - log_y,
            1e-13,
            &format!("log({} / {}) = log({}) - log({})", x, y, x, y),
        );
    }
}

// ====================== POWER RULE ======================

#[test]
fn oracle_log_power_rule() {
    // log(x^n) = n * log(x)
    let x: f64 = 2.0;
    for n in [2, 3, 4, 5] {
        let x_pow_n = x.powi(n);
        let log_x_pow_n = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![x_pow_n])], &no_params()).unwrap(),
        );
        let log_x = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        assert_close(
            log_x_pow_n,
            (n as f64) * log_x,
            1e-13,
            &format!("log({}^{}) = {} * log({})", x, n, n, x),
        );
    }
}

// ====================== LOG-EXP RELATIONSHIP ======================

#[test]
fn oracle_log_exp_inverse() {
    // log(exp(x)) = x
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0, 10.0] {
        let exp_x = extract_f64_scalar(
            &eval_primitive(Primitive::Exp, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let log_exp_x = extract_f64_scalar(
            &eval_primitive(Primitive::Log, &[make_f64_tensor(&[], vec![exp_x])], &no_params()).unwrap(),
        );
        assert_close(log_exp_x, x, 1e-14, &format!("log(exp({})) = {}", x, x));
    }
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_log_monotonic() {
    // log is strictly increasing on (0, inf)
    let values = vec![0.1, 0.5, 1.0, 2.0, 10.0, 100.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "log should be strictly increasing: log[{}] = {} > log[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ====================== COMPLEX NUMBERS ======================

#[test]
fn oracle_log_complex_positive_real() {
    // log(e + 0i) = 1 + 0i
    let input = make_complex128_tensor(&[], vec![(std::f64::consts::E, 0.0)]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, 1.0, 1e-14, "log(e + 0i) real part");
    assert_close(im, 0.0, 1e-14, "log(e + 0i) imag part");
}

#[test]
fn oracle_log_complex_negative_real() {
    // log(-1) = i*π (principal value)
    let input = make_complex128_tensor(&[], vec![(-1.0, 0.0)]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, 0.0, 1e-14, "log(-1) real part");
    assert_close(im, std::f64::consts::PI, 1e-14, "log(-1) = i*π");
}

#[test]
fn oracle_log_complex_pure_imag() {
    // log(i) = i*π/2 (principal value)
    let input = make_complex128_tensor(&[], vec![(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, 0.0, 1e-14, "log(i) real part");
    assert_close(im, std::f64::consts::FRAC_PI_2, 1e-14, "log(i) = i*π/2");
}

#[test]
fn oracle_log_complex_general() {
    // log(z) = log|z| + i*arg(z)
    let z = (3.0, 4.0); // |z| = 5, arg(z) = atan2(4, 3)
    let input = make_complex128_tensor(&[], vec![z]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    let magnitude = (z.0 * z.0 + z.1 * z.1).sqrt(); // 5
    let arg = z.1.atan2(z.0);
    assert_close(re, magnitude.ln(), 1e-14, "log(3+4i) real part = ln(5)");
    assert_close(im, arg, 1e-14, "log(3+4i) imag part = atan2(4,3)");
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_log_1d() {
    let e = std::f64::consts::E;
    let input = make_f64_tensor(&[5], vec![1.0, e, e * e, 10.0, 100.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.0, 1e-14, "log(1)");
    assert_close(vals[1], 1.0, 1e-14, "log(e)");
    assert_close(vals[2], 2.0, 1e-14, "log(e^2)");
    assert_close(vals[3], std::f64::consts::LN_10, 1e-14, "log(10)");
    assert_close(vals[4], 2.0 * std::f64::consts::LN_10, 1e-14, "log(100)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_log_2d() {
    let e = std::f64::consts::E;
    let input = make_f64_tensor(&[2, 2], vec![1.0, e, 2.0, 10.0]);
    let result = eval_primitive(Primitive::Log, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.0, 1e-14, "");
    assert_close(vals[1], 1.0, 1e-14, "");
    assert_close(vals[2], std::f64::consts::LN_2, 1e-14, "");
    assert_close(vals[3], std::f64::consts::LN_10, 1e-14, "");
}
