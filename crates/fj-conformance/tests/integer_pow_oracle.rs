//! Oracle tests for IntegerPow primitive.
//!
//! Tests against expected behavior matching JAX/lax.integer_pow:
//! - exponent: integer power to raise each element to
//! - Equivalent to x^n for integer n

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

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::I64).collect(),
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

fn pow_params(exponent: i32) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("exponent".to_string(), exponent.to_string());
    p
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_integer_pow_scalar_square() {
    // 3^2 = 9
    let input = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 9.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_cube() {
    // 2^3 = 8
    let input = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(3)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 8.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_zero_exp() {
    // x^0 = 1
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(0)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_one_exp() {
    // x^1 = x
    let input = Value::Scalar(Literal::from_f64(5.5));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(1)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.5).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_negative_exp() {
    // 2^(-2) = 0.25
    let input = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(-2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.25).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_negative_base() {
    // (-2)^3 = -8
    let input = Value::Scalar(Literal::from_f64(-2.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(3)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-8.0)).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_scalar_negative_base_even() {
    // (-2)^4 = 16
    let input = Value::Scalar(Literal::from_f64(-2.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(4)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 16.0).abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_integer_pow_1d_square() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 4.0).abs() < 1e-10);
    assert!((vals[2] - 9.0).abs() < 1e-10);
    assert!((vals[3] - 16.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_1d_cube() {
    let input = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(3)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 8.0).abs() < 1e-10);
    assert!((vals[1] - 27.0).abs() < 1e-10);
    assert!((vals[2] - 64.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_1d_zero_exp() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(0)).unwrap();
    let vals = extract_f64_vec(&result);
    for v in vals {
        assert!((v - 1.0).abs() < 1e-10);
    }
}

#[test]
fn oracle_integer_pow_1d_negative_exp() {
    let input = make_f64_tensor(&[3], vec![2.0, 4.0, 10.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(-1)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.5).abs() < 1e-10);
    assert!((vals[1] - 0.25).abs() < 1e-10);
    assert!((vals[2] - 0.1).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_1d_mixed_signs() {
    let input = make_f64_tensor(&[4], vec![-2.0, -1.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
    assert!((vals[2] - 1.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_integer_pow_2d_square() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 4.0).abs() < 1e-10);
    assert!((vals[2] - 9.0).abs() < 1e-10);
    assert!((vals[3] - 16.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_2d_high_exp() {
    let input = make_f64_tensor(&[2, 2], vec![2.0, 2.0, 2.0, 2.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(10)).unwrap();
    let vals = extract_f64_vec(&result);
    for v in vals {
        assert!((v - 1024.0).abs() < 1e-10);
    }
}

// ======================== Edge Cases ========================

#[test]
fn oracle_integer_pow_zero_base() {
    // 0^n = 0 for n > 0
    let input = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(5)).unwrap();
    let vals = extract_f64_vec(&result);
    for v in vals {
        assert!((v - 0.0).abs() < 1e-10);
    }
}

#[test]
fn oracle_integer_pow_zero_to_zero() {
    // 0^0 = 1 (mathematical convention)
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(0)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_one_base() {
    // 1^n = 1 for any n
    let input = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(100)).unwrap();
    let vals = extract_f64_vec(&result);
    for v in vals {
        assert!((v - 1.0).abs() < 1e-10);
    }
}

#[test]
fn oracle_integer_pow_fractional_base() {
    // 0.5^2 = 0.25
    let input = Value::Scalar(Literal::from_f64(0.5));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.25).abs() < 1e-10);
}

#[test]
fn oracle_integer_pow_large_exp() {
    // 2^20 = 1048576
    let input = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(20)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1048576.0).abs() < 1e-5);
}

#[test]
fn oracle_integer_pow_i64_tensor() {
    // Integer input gets converted to f64 for powi
    let input = make_i64_tensor(&[3], vec![2, 3, 4]);
    let result = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10);
    assert!((vals[1] - 9.0).abs() < 1e-10);
    assert!((vals[2] - 16.0).abs() < 1e-10);
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

// ======================== METAMORPHIC: IntegerPow(x, 2) = Square(x) ========================

#[test]
fn metamorphic_integer_pow_2_equals_square() {
    // IntegerPow(x, 2) = Square(x) using primitives
    for x in [0.5, 1.0, 2.0, 3.0, -2.0, -3.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let pow2 = eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&input), &pow_params(2)).unwrap();
        let squared = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&pow2),
            extract_f64_scalar(&squared),
            1e-14,
            &format!("IntegerPow({}, 2) = Square({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: IntegerPow(x, 1) = x ========================

#[test]
fn metamorphic_integer_pow_1_identity() {
    // IntegerPow(x, 1) = x
    for x in [0.5, 1.0, 2.0, 3.0, -2.0, -3.0, 0.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let pow1 = eval_primitive(Primitive::IntegerPow, &[input], &pow_params(1)).unwrap();

        assert_close(
            extract_f64_scalar(&pow1),
            x,
            1e-14,
            &format!("IntegerPow({}, 1) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: IntegerPow(x, 3) = Mul(Square(x), x) ========================

#[test]
fn metamorphic_integer_pow_3_equals_square_mul() {
    // IntegerPow(x, 3) = Mul(Square(x), x)
    for x in [0.5, 1.0, 2.0, 3.0, -2.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let pow3 = eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&input), &pow_params(3)).unwrap();
        let squared = eval_primitive(Primitive::Square, std::slice::from_ref(&input), &no_params()).unwrap();
        let cubed = eval_primitive(Primitive::Mul, &[squared, input], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&pow3),
            extract_f64_scalar(&cubed),
            1e-14,
            &format!("IntegerPow({}, 3) = Mul(Square({}), {})", x, x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor IntegerPow(x, 2) = Square(x) ========================

#[test]
fn metamorphic_integer_pow_tensor_square() {
    // For tensor: IntegerPow(x, 2) = Square(x)
    let data = vec![0.5, 1.0, 2.0, 3.0, -2.0];
    let input = make_f64_tensor(&[5], data);

    let pow2 = eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&input), &pow_params(2)).unwrap();
    let squared = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();

    assert_eq!(extract_shape(&pow2), vec![5]);
    let pow2_vec = extract_f64_vec(&pow2);
    let sq_vec = extract_f64_vec(&squared);
    for (i, (&p, &s)) in pow2_vec.iter().zip(sq_vec.iter()).enumerate() {
        assert_close(p, s, 1e-14, &format!("element {}", i));
    }
}
