//! Oracle tests for Pow primitive.
//!
//! pow(x, y) = x^y
//!
//! Tests:
//! - Basic powers: 2^3 = 8, 3^2 = 9
//! - Zero exponent: x^0 = 1
//! - One exponent: x^1 = x
//! - Zero base: 0^y = 0 for y > 0
//! - Negative exponents: x^(-y) = 1/x^y
//! - Fractional exponents (roots): x^0.5 = sqrt(x)
//! - Negative bases with integer exponents
//! - Infinity and NaN cases
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

// ======================== Basic Powers ========================

#[test]
fn oracle_pow_two_cubed() {
    // 2^3 = 8
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 8.0, 1e-14, "2^3 = 8");
}

#[test]
fn oracle_pow_three_squared() {
    // 3^2 = 9
    let base = make_f64_tensor(&[], vec![3.0]);
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 9.0, 1e-14, "3^2 = 9");
}

#[test]
fn oracle_pow_ten_squared() {
    let base = make_f64_tensor(&[], vec![10.0]);
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 100.0, 1e-14, "10^2 = 100");
}

// ======================== Zero Exponent ========================

#[test]
fn oracle_pow_x_to_zero() {
    // x^0 = 1 for any non-zero x
    for x in [1.0, 2.0, 10.0, 100.0, -1.0, -5.0] {
        let base = make_f64_tensor(&[], vec![x]);
        let exp = make_f64_tensor(&[], vec![0.0]);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), 1.0, "{}^0 = 1", x);
    }
}

#[test]
fn oracle_pow_zero_to_zero() {
    // 0^0 is often defined as 1 in numerical contexts
    let base = make_f64_tensor(&[], vec![0.0]);
    let exp = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val == 1.0 || val.is_nan(), "0^0 = 1 or NaN");
}

// ======================== One Exponent ========================

#[test]
fn oracle_pow_x_to_one() {
    // x^1 = x
    for x in [0.0, 1.0, 2.5, -3.0, 100.0] {
        let base = make_f64_tensor(&[], vec![x]);
        let exp = make_f64_tensor(&[], vec![1.0]);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x,
            1e-14,
            &format!("{}^1 = {}", x, x),
        );
    }
}

// ======================== One Base ========================

#[test]
fn oracle_pow_one_to_x() {
    // 1^x = 1 for any x
    for x in [0.0, 1.0, 2.0, -1.0, 100.0, -100.0] {
        let base = make_f64_tensor(&[], vec![1.0]);
        let exp = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), 1.0, "1^{} = 1", x);
    }
}

// ======================== Zero Base ========================

#[test]
fn oracle_pow_zero_positive_exp() {
    // 0^y = 0 for y > 0
    for y in [1.0, 2.0, 0.5, 10.0] {
        let base = make_f64_tensor(&[], vec![0.0]);
        let exp = make_f64_tensor(&[], vec![y]);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), 0.0, "0^{} = 0", y);
    }
}

#[test]
fn oracle_pow_zero_negative_exp() {
    // 0^(-y) = inf for y > 0
    let base = make_f64_tensor(&[], vec![0.0]);
    let exp = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "0^(-1) = +inf");
}

// ======================== Negative Exponents ========================

#[test]
fn oracle_pow_negative_exp() {
    // x^(-y) = 1/x^y
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "2^(-1) = 0.5");
}

#[test]
fn oracle_pow_negative_exp_two() {
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "2^(-2) = 0.25");
}

#[test]
fn oracle_pow_negative_exp_three() {
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![-3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.125, 1e-14, "2^(-3) = 0.125");
}

// ======================== Fractional Exponents (Roots) ========================

#[test]
fn oracle_pow_sqrt() {
    // x^0.5 = sqrt(x)
    let base = make_f64_tensor(&[], vec![4.0]);
    let exp = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "4^0.5 = 2");
}

#[test]
fn oracle_pow_cube_root() {
    // x^(1/3) = cbrt(x)
    let base = make_f64_tensor(&[], vec![8.0]);
    let exp = make_f64_tensor(&[], vec![1.0 / 3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "8^(1/3) = 2");
}

#[test]
fn oracle_pow_fourth_root() {
    let base = make_f64_tensor(&[], vec![16.0]);
    let exp = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "16^0.25 = 2");
}

// ======================== Negative Bases with Integer Exponents ========================

#[test]
fn oracle_pow_neg_base_even_exp() {
    // (-2)^2 = 4
    let base = make_f64_tensor(&[], vec![-2.0]);
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 4.0, 1e-14, "(-2)^2 = 4");
}

#[test]
fn oracle_pow_neg_base_odd_exp() {
    // (-2)^3 = -8
    let base = make_f64_tensor(&[], vec![-2.0]);
    let exp = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -8.0, 1e-14, "(-2)^3 = -8");
}

// ======================== Infinity ========================

#[test]
fn oracle_pow_inf_positive_exp() {
    // inf^y = inf for y > 0
    let base = make_f64_tensor(&[], vec![f64::INFINITY]);
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "inf^2 = inf");
}

#[test]
fn oracle_pow_inf_negative_exp() {
    // inf^(-y) = 0 for y > 0
    let base = make_f64_tensor(&[], vec![f64::INFINITY]);
    let exp = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "inf^(-1) = 0");
}

#[test]
fn oracle_pow_x_to_inf() {
    // x^inf = inf for x > 1
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "2^inf = inf");
}

#[test]
fn oracle_pow_fraction_to_inf() {
    // x^inf = 0 for 0 < x < 1
    let base = make_f64_tensor(&[], vec![0.5]);
    let exp = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "0.5^inf = 0");
}

// ======================== NaN ========================

#[test]
fn oracle_pow_nan_base() {
    let base = make_f64_tensor(&[], vec![f64::NAN]);
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "NaN^2 = NaN");
}

#[test]
fn oracle_pow_nan_exp() {
    let base = make_f64_tensor(&[], vec![2.0]);
    let exp = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "2^NaN = NaN");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_pow_1d() {
    let base = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 10.0]);
    let exp = make_f64_tensor(&[4], vec![2.0, 2.0, 0.5, 3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 4.0, 1e-14, "2^2");
    assert_close(vals[1], 9.0, 1e-14, "3^2");
    assert_close(vals[2], 2.0, 1e-14, "4^0.5");
    assert_close(vals[3], 1000.0, 1e-14, "10^3");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_pow_2d() {
    let base = make_f64_tensor(&[2, 2], vec![2.0, 3.0, 4.0, 5.0]);
    let exp = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 0.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 2.0, 1e-14, "2^1");
    assert_close(vals[1], 9.0, 1e-14, "3^2");
    assert_close(vals[2], 64.0, 1e-14, "4^3");
    assert_eq!(vals[3], 1.0, "5^0");
}

// ======================== Identity: x^y = e^(y*ln(x)) ========================

#[test]
fn oracle_pow_identity() {
    for (x, y) in [(2.0, 3.0), (3.0, 2.0), (4.0, 0.5), (10.0, 2.0)] {
        let base = make_f64_tensor(&[], vec![x]);
        let exp = make_f64_tensor(&[], vec![y]);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = (y * x.ln()).exp();
        assert_close(
            val,
            expected,
            1e-13,
            &format!("{}^{} = e^({}*ln({}))", x, y, y, x),
        );
    }
}
