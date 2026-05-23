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
//! - Broadcast-compatible operands

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

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_pow_scalar_base_tensor_exp_broadcast() {
    // scalar base ^ tensor exponent
    let base = scalar_f64(2.0);
    let exp = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 2.0, 1e-14, "2^1");
    assert_close(vals[1], 4.0, 1e-14, "2^2");
    assert_close(vals[2], 8.0, 1e-14, "2^3");
    assert_close(vals[3], 16.0, 1e-14, "2^4");
}

#[test]
fn oracle_pow_tensor_base_scalar_exp_broadcast() {
    // tensor base ^ scalar exponent
    let base = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 5.0]);
    let exp = scalar_f64(2.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 4.0, 1e-14, "2^2");
    assert_close(vals[1], 9.0, 1e-14, "3^2");
    assert_close(vals[2], 16.0, 1e-14, "4^2");
    assert_close(vals[3], 25.0, 1e-14, "5^2");
}

#[test]
fn oracle_pow_singleton_base_vector_exp_broadcast() {
    // [1] base ^ [3] exp -> [3]
    let base = make_f64_tensor(&[1], vec![2.0]);
    let exp = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 2.0, 1e-14, "2^1");
    assert_close(vals[1], 4.0, 1e-14, "2^2");
    assert_close(vals[2], 8.0, 1e-14, "2^3");
}

#[test]
fn oracle_pow_vector_base_singleton_exp_broadcast() {
    // [3] base ^ [1] exp -> [3]
    let base = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let exp = make_f64_tensor(&[1], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 4.0, 1e-14, "2^2");
    assert_close(vals[1], 9.0, 1e-14, "3^2");
    assert_close(vals[2], 16.0, 1e-14, "4^2");
}

#[test]
fn oracle_pow_column_vector_base_broadcast() {
    // [2, 1] base ^ [2, 3] exp -> [2, 3]
    let base = make_f64_tensor(&[2, 1], vec![2.0, 3.0]);
    let exp = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: 2^1, 2^2, 2^3 = 2, 4, 8
    assert_close(vals[0], 2.0, 1e-14, "2^1");
    assert_close(vals[1], 4.0, 1e-14, "2^2");
    assert_close(vals[2], 8.0, 1e-14, "2^3");
    // Row 1: 3^2, 3^3, 3^4 = 9, 27, 81
    assert_close(vals[3], 9.0, 1e-14, "3^2");
    assert_close(vals[4], 27.0, 1e-14, "3^3");
    assert_close(vals[5], 81.0, 1e-14, "3^4");
}

#[test]
fn oracle_pow_row_vector_exp_broadcast() {
    // [2, 3] base ^ [1, 3] exp -> [2, 3]
    let base = make_f64_tensor(&[2, 3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let exp = make_f64_tensor(&[1, 3], vec![2.0, 2.0, 2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 4.0, 1e-14, "2^2");
    assert_close(vals[1], 9.0, 1e-14, "3^2");
    assert_close(vals[2], 16.0, 1e-14, "4^2");
    assert_close(vals[3], 25.0, 1e-14, "5^2");
    assert_close(vals[4], 36.0, 1e-14, "6^2");
    assert_close(vals[5], 49.0, 1e-14, "7^2");
}

#[test]
fn oracle_pow_different_ranks_broadcast() {
    // [3] base ^ [2, 3] exp -> [2, 3]
    let base = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let exp = make_f64_tensor(&[2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: 2^1, 3^1, 4^1 = 2, 3, 4
    assert_close(vals[0], 2.0, 1e-14, "2^1");
    assert_close(vals[1], 3.0, 1e-14, "3^1");
    assert_close(vals[2], 4.0, 1e-14, "4^1");
    // Row 1: 2^2, 3^2, 4^2 = 4, 9, 16
    assert_close(vals[3], 4.0, 1e-14, "2^2");
    assert_close(vals[4], 9.0, 1e-14, "3^2");
    assert_close(vals[5], 16.0, 1e-14, "4^2");
}

#[test]
fn oracle_pow_all_scalars_broadcast() {
    // scalar ^ scalar -> scalar
    let base = scalar_f64(2.0);
    let exp = scalar_f64(3.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 8.0, 1e-14, "2^3 = 8");
}

#[test]
fn oracle_pow_incompatible_shapes_error() {
    // [2] ^ [3] should error
    let base = make_f64_tensor(&[2], vec![2.0, 3.0]);
    let exp = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_pow_vector_scalar_exp_broadcast() {
    let base_values = [2.0, 3.0, 4.0, 5.0];
    let base = make_f64_tensor(&[4], base_values.to_vec());
    let exp = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    for (i, (&actual, &base_value)) in vals.iter().zip(base_values.iter()).enumerate() {
        let expected = base_value.powf(2.0);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast scalar exponent element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_pow_row_base_matrix_exp_broadcast() {
    let base_values = [2.0, 3.0, 1.0];
    let exp_values = [4.0, 1.0, 6.0, 2.0, 3.0, 5.0];
    let base = make_f64_tensor(&[3], base_values.to_vec());
    let exp = make_f64_tensor(&[2, 3], exp_values.to_vec());
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    for (i, ((&actual, &base_value), &exp_value)) in vals
        .iter()
        .zip(base_values.iter().cycle())
        .zip(exp_values.iter())
        .enumerate()
    {
        let expected = base_value.powf(exp_value);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast row base element {i}: expected {expected}, got {actual}"
        );
    }
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

// ======================== Metamorphic: Power Law Properties ========================

#[test]
fn metamorphic_pow_exponent_sum() {
    // x^(a+b) = x^a * x^b
    for (x, a, b) in [
        (2.0, 3.0, 2.0),
        (3.0, 1.5, 0.5),
        (1.5, 2.0, 3.0),
        (4.0, 0.5, 1.5),
    ] {
        let base = make_f64_tensor(&[], vec![x]);
        let exp_sum = make_f64_tensor(&[], vec![a + b]);
        let exp_a = make_f64_tensor(&[], vec![a]);
        let exp_b = make_f64_tensor(&[], vec![b]);

        let lhs = eval_primitive(Primitive::Pow, &[base.clone(), exp_sum], &no_params()).unwrap();
        let pow_a = eval_primitive(Primitive::Pow, &[base.clone(), exp_a], &no_params()).unwrap();
        let pow_b = eval_primitive(Primitive::Pow, &[base, exp_b], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::Mul, &[pow_a, pow_b], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&lhs),
            extract_f64_scalar(&rhs),
            1e-12,
            &format!("{}^({} + {}) = {}^{} * {}^{}", x, a, b, x, a, x, b),
        );
    }
}

#[test]
fn metamorphic_pow_product_base() {
    // (x*y)^a = x^a * y^a
    for (x, y, a) in [(2.0, 3.0, 2.0), (1.5, 2.0, 3.0), (4.0, 0.5, 2.5)] {
        let base_x = make_f64_tensor(&[], vec![x]);
        let base_y = make_f64_tensor(&[], vec![y]);
        let exp = make_f64_tensor(&[], vec![a]);

        let product = eval_primitive(
            Primitive::Mul,
            &[base_x.clone(), base_y.clone()],
            &no_params(),
        )
        .unwrap();
        let lhs = eval_primitive(Primitive::Pow, &[product, exp.clone()], &no_params()).unwrap();

        let pow_x = eval_primitive(Primitive::Pow, &[base_x, exp.clone()], &no_params()).unwrap();
        let pow_y = eval_primitive(Primitive::Pow, &[base_y, exp], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::Mul, &[pow_x, pow_y], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&lhs),
            extract_f64_scalar(&rhs),
            1e-12,
            &format!("({} * {})^{} = {}^{} * {}^{}", x, y, a, x, a, y, a),
        );
    }
}

#[test]
fn metamorphic_pow_power_of_power() {
    // (x^a)^b = x^(a*b)
    for (x, a, b) in [(2.0, 2.0, 3.0), (3.0, 0.5, 4.0), (1.5, 2.0, 1.5)] {
        let base = make_f64_tensor(&[], vec![x]);
        let exp_a = make_f64_tensor(&[], vec![a]);
        let exp_b = make_f64_tensor(&[], vec![b]);
        let exp_ab = make_f64_tensor(&[], vec![a * b]);

        let inner = eval_primitive(Primitive::Pow, &[base.clone(), exp_a], &no_params()).unwrap();
        let lhs = eval_primitive(Primitive::Pow, &[inner, exp_b], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::Pow, &[base, exp_ab], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&lhs),
            extract_f64_scalar(&rhs),
            1e-11,
            &format!("({}^{})^{} = {}^({} * {})", x, a, b, x, a, b),
        );
    }
}

#[test]
fn metamorphic_pow_exponent_sum_tensor() {
    // x^(a+b) = x^a * x^b for 1D tensors
    let x = make_f64_tensor(&[4], vec![2.0, 3.0, 4.0, 5.0]);
    let a = make_f64_tensor(&[4], vec![2.0, 1.5, 0.5, 2.0]);
    let b = make_f64_tensor(&[4], vec![1.0, 0.5, 1.5, 1.0]);

    let sum_ab = eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap();
    let lhs = eval_primitive(Primitive::Pow, &[x.clone(), sum_ab], &no_params()).unwrap();

    let pow_a = eval_primitive(Primitive::Pow, &[x.clone(), a], &no_params()).unwrap();
    let pow_b = eval_primitive(Primitive::Pow, &[x, b], &no_params()).unwrap();
    let rhs = eval_primitive(Primitive::Mul, &[pow_a, pow_b], &no_params()).unwrap();

    let lhs_vals = extract_f64_vec(&lhs);
    let rhs_vals = extract_f64_vec(&rhs);

    for i in 0..4 {
        assert_close(
            lhs_vals[i],
            rhs_vals[i],
            1e-12,
            &format!("exponent sum rule element {}", i),
        );
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_pow_preserves_all_float_dtypes() {
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

    // Pow requires positive base for fractional exponents
    let base_values = [1.0_f64, 2.0, 4.0];
    let exp_values = [1.0_f64, 0.5, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let base = make_vec(dtype, &base_values);
        let exp = make_vec(dtype, &exp_values);
        let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "pow {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
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
            Shape { dims: shape.to_vec() },
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
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
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
fn oracle_pow_complex64_integer_exponent() {
    // (1+i)^2 = 1 + 2i - 1 = 2i
    let base = make_complex64_scalar(1.0, 1.0);
    let exp = make_complex64_scalar(2.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 2.0), 1e-5, "(1+i)^2 = 2i");
}

#[test]
fn oracle_pow_complex64_zero_exponent() {
    // z^0 = 1 for any z != 0
    let base = make_complex64_scalar(3.0, 4.0);
    let exp = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (1.0, 0.0), 1e-6, "(3+4i)^0 = 1");
}

#[test]
fn oracle_pow_complex64_one_exponent() {
    // z^1 = z
    let base = make_complex64_scalar(3.0, 4.0);
    let exp = make_complex64_scalar(1.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-5, "(3+4i)^1 = 3+4i");
}

#[test]
fn oracle_pow_complex64_i_squared() {
    // i^2 = -1
    let base = make_complex64_scalar(0.0, 1.0);
    let exp = make_complex64_scalar(2.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (-1.0, 0.0), 1e-5, "i^2 = -1");
}

#[test]
fn oracle_pow_complex64_cube() {
    // i^3 = -i
    let base = make_complex64_scalar(0.0, 1.0);
    let exp = make_complex64_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, -1.0), 1e-5, "i^3 = -i");
}

#[test]
fn oracle_pow_complex64_fourth_power() {
    // i^4 = 1
    let base = make_complex64_scalar(0.0, 1.0);
    let exp = make_complex64_scalar(4.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (1.0, 0.0), 1e-5, "i^4 = 1");
}

#[test]
fn oracle_pow_complex64_half_power() {
    // 4^0.5 = 2 (principal square root)
    let base = make_complex64_scalar(4.0, 0.0);
    let exp = make_complex64_scalar(0.5, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (2.0, 0.0), 1e-5, "4^0.5 = 2");
}

#[test]
fn oracle_pow_complex64_negative_one() {
    // (2+0i)^(-1) = 0.5
    let base = make_complex64_scalar(2.0, 0.0);
    let exp = make_complex64_scalar(-1.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.5, 0.0), 1e-5, "2^(-1) = 0.5");
}

#[test]
fn oracle_pow_complex64_complex_exponent() {
    // e^(i*pi) = -1 (Euler's identity)
    // base = e, exp = i*pi
    let e = std::f32::consts::E;
    let pi = std::f32::consts::PI;
    let base = make_complex64_scalar(e, 0.0);
    let exp = make_complex64_scalar(0.0, pi);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (-1.0, 0.0), 1e-4, "e^(i*pi) = -1");
}

#[test]
fn oracle_pow_complex64_vector() {
    let base = make_complex64_tensor(&[3], vec![(2.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
    let exp = make_complex64_tensor(&[3], vec![(2.0, 0.0), (2.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    // 2^2 = 4
    assert_complex64_close(vals[0], (4.0, 0.0), 1e-5, "2^2");
    // i^2 = -1
    assert_complex64_close(vals[1], (-1.0, 0.0), 1e-5, "i^2");
    // (1+i)^2 = 2i
    assert_complex64_close(vals[2], (0.0, 2.0), 1e-5, "(1+i)^2");
}

#[test]
fn oracle_pow_complex128_integer_exponent() {
    // (1+i)^2 = 2i
    let base = make_complex128_scalar(1.0, 1.0);
    let exp = make_complex128_scalar(2.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (0.0, 2.0), 1e-12, "(1+i)^2 = 2i Complex128");
}

#[test]
fn oracle_pow_complex128_eulers_identity() {
    // e^(i*pi) = -1
    let e = std::f64::consts::E;
    let pi = std::f64::consts::PI;
    let base = make_complex128_scalar(e, 0.0);
    let exp = make_complex128_scalar(0.0, pi);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (-1.0, 0.0), 1e-10, "e^(i*pi) = -1 Complex128");
}

#[test]
fn oracle_pow_complex64_preserves_dtype() {
    let base = make_complex64_scalar(2.0, 1.0);
    let exp = make_complex64_scalar(2.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_pow_complex128_preserves_dtype() {
    let base = make_complex128_scalar(2.0, 1.0);
    let exp = make_complex128_scalar(2.0, 0.0);
    let result = eval_primitive(Primitive::Pow, &[base, exp], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
