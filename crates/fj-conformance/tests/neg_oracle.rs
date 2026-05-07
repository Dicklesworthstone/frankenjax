//! Oracle tests for Neg primitive.
//!
//! neg(x) = -x (unary negation)
//!
//! Properties:
//! - neg(neg(x)) = x (double negation / involution)
//! - neg(0) = 0 (but neg(-0) = 0 in IEEE 754)
//! - neg(x) + x = 0
//! - neg(x + y) = neg(x) + neg(y) (linearity)
//! - neg(x * y) = neg(x) * y = x * neg(y)
//!
//! Tests:
//! - Positive, negative, zero values
//! - Special float values (infinity, NaN, signed zeros)
//! - Integer types
//! - Complex numbers
//! - Mathematical properties
//! - Tensor shapes

#![allow(clippy::approx_constant)]

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

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            match &t.elements[0] {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                _ => unreachable!("expected complex128"),
            }
        }
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            (f64::from_bits(*re), f64::from_bits(*im))
        }
        _ => unreachable!("expected complex128"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::Complex64Bits(re, im) => (f32::from_bits(*re), f32::from_bits(*im)),
                _ => unreachable!("expected complex64"),
            })
            .collect(),
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

// ====================== BASIC FLOAT VALUES ======================

#[test]
fn oracle_neg_positive_f64() {
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0);
}

#[test]
fn oracle_neg_negative_f64() {
    let input = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_neg_zero_f64() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val == 0.0, "neg(0) = 0 or -0");
}

#[test]
fn oracle_neg_fractional() {
    let input = make_f64_tensor(&[], vec![3.17]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -3.17, 1e-14, "neg(pi)");
}

// ====================== INTEGER VALUES ======================

#[test]
fn oracle_neg_positive_i64() {
    let input = make_i64_tensor(&[], vec![42]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -42);
}

#[test]
fn oracle_neg_negative_i64() {
    let input = make_i64_tensor(&[], vec![-42]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 42);
}

#[test]
fn oracle_neg_zero_i64() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_neg_large_i64() {
    let input = make_i64_tensor(&[], vec![i64::MAX / 2]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -(i64::MAX / 2));
}

// ====================== SPECIAL FLOAT VALUES ======================

#[test]
fn oracle_neg_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY);
}

#[test]
fn oracle_neg_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_neg_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "neg(NaN) = NaN");
}

// ====================== DOUBLE NEGATION (INVOLUTION) ======================

#[test]
fn oracle_neg_involution() {
    // neg(neg(x)) = x
    for x in [
        1.0,
        -1.0,
        0.0,
        3.17,
        -2.718,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg1 = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let neg2 = eval_primitive(Primitive::Neg, &[neg1], &no_params()).unwrap();
        let result = extract_f64_scalar(&neg2);
        if x.is_nan() {
            assert!(result.is_nan());
        } else {
            assert_eq!(result, x, "neg(neg({})) = {}", x, x);
        }
    }
}

// ====================== ADDITIVE INVERSE ======================

#[test]
fn oracle_neg_additive_inverse() {
    // x + neg(x) = 0
    for x in [1.0, -1.0, 3.17, -2.718, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg_x =
            extract_f64_scalar(&eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap());
        let sum = x + neg_x;
        assert_close(sum, 0.0, 1e-14, &format!("{} + neg({}) = 0", x, x));
    }
}

// ====================== LINEARITY ======================

#[test]
fn oracle_neg_distributes_over_sum() {
    // neg(x + y) = neg(x) + neg(y)
    let test_pairs = [(2.0, 3.0), (-2.0, 3.0), (2.0, -3.0), (-2.0, -3.0)];
    for (x, y) in test_pairs {
        let sum = x + y;
        let neg_sum = extract_f64_scalar(
            &eval_primitive(
                Primitive::Neg,
                &[make_f64_tensor(&[], vec![sum])],
                &no_params(),
            )
            .unwrap(),
        );
        let neg_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Neg,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let neg_y = extract_f64_scalar(
            &eval_primitive(
                Primitive::Neg,
                &[make_f64_tensor(&[], vec![y])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(
            neg_sum,
            neg_x + neg_y,
            1e-14,
            &format!("neg({} + {}) = neg({}) + neg({})", x, y, x, y),
        );
    }
}

// ====================== COMPLEX NUMBERS ======================

#[test]
fn oracle_neg_complex() {
    // neg(a + bi) = -a - bi
    let input = make_complex128_tensor(&[], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, -3.0, 1e-14, "neg(3+4i) real part");
    assert_close(im, -4.0, 1e-14, "neg(3+4i) imag part");
}

#[test]
fn oracle_neg_complex_pure_real() {
    let input = make_complex128_tensor(&[], vec![(5.0, 0.0)]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, -5.0, 1e-14, "neg(5+0i) real part");
    assert_close(im, 0.0, 1e-14, "neg(5+0i) imag part");
}

#[test]
fn oracle_neg_complex_pure_imag() {
    let input = make_complex128_tensor(&[], vec![(0.0, 5.0)]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, 0.0, 1e-14, "neg(0+5i) real part");
    assert_close(im, -5.0, 1e-14, "neg(0+5i) imag part");
}

#[test]
fn oracle_neg_complex64_preserves_complex64_literals() {
    let input = make_complex64_tensor(&[2], vec![(3.0, 4.0), (-5.0, 12.0)]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let Value::Tensor(tensor) = &result else {
        unreachable!("expected tensor result");
    };
    assert_eq!(tensor.dtype, DType::Complex64);
    assert_eq!(tensor.shape.dims, vec![2]);
    assert_eq!(
        extract_complex64_vec(&result),
        vec![(-3.0, -4.0), (5.0, -12.0)]
    );
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_neg_1d_f64() {
    let input = make_f64_tensor(&[5], vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 1.0, 0.0, -1.0, -3.0]);
}

#[test]
fn oracle_neg_1d_i64() {
    let input = make_i64_tensor(&[5], vec![-3, -1, 0, 1, 3]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 0, -1, -3]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_neg_2d_f64() {
    let input = make_f64_tensor(&[2, 3], vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 3.0, -4.0, -5.0, -6.0]
    );
}

// ====================== RELATIONSHIP WITH OTHER OPS ======================

#[test]
fn oracle_neg_vs_mul_minus_one() {
    // neg(x) = x * (-1)
    for x in [1.0, -1.0, 0.0, 3.17, -2.718] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg_result =
            extract_f64_scalar(&eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap());
        let expected = -x;
        assert_close(
            neg_result,
            expected,
            1e-14,
            &format!("neg({}) = {} * (-1)", x, x),
        );
    }
}

// ====================== SIGN PRESERVATION ======================

#[test]
fn oracle_neg_flips_sign() {
    // sign(neg(x)) = -sign(x) for x != 0
    for x in [1.0, -1.0, 100.0, -100.0, 0.001, -0.001] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg_result =
            extract_f64_scalar(&eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap());
        assert_eq!(
            neg_result.signum(),
            -(x.signum()),
            "sign(neg({})) = -sign({})",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: Neg(Neg(x)) = x ========================

#[test]
fn metamorphic_neg_involution() {
    // Neg(Neg(x)) = x (involution)
    for x in [-5.5, -1.0, 0.0, 1.0, 5.5, f64::INFINITY, f64::NEG_INFINITY] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg1 = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let neg2 = eval_primitive(Primitive::Neg, &[neg1], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&neg2),
            x,
            "Neg(Neg({})) = {}",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: Add(x, Neg(x)) = 0 ========================

#[test]
fn metamorphic_neg_add_zero() {
    // Add(x, Neg(x)) = 0 using Add primitive
    for x in [-5.5, -1.0, 1.0, 5.5, 100.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg_x = eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();
        let sum = eval_primitive(Primitive::Add, &[input, neg_x], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&sum),
            0.0,
            "Add({}, Neg({})) = 0",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: Neg distributes over Add ========================

#[test]
fn metamorphic_neg_distributes_add() {
    // Neg(Add(x, y)) = Add(Neg(x), Neg(y))
    let pairs = [(1.0, 2.0), (-1.0, 3.0), (2.5, -4.5)];
    for (x, y) in pairs {
        let x_tensor = make_f64_tensor(&[], vec![x]);
        let y_tensor = make_f64_tensor(&[], vec![y]);

        // Neg(Add(x, y))
        let sum = eval_primitive(Primitive::Add, &[x_tensor.clone(), y_tensor.clone()], &no_params()).unwrap();
        let neg_sum = eval_primitive(Primitive::Neg, &[sum], &no_params()).unwrap();

        // Add(Neg(x), Neg(y))
        let neg_x = eval_primitive(Primitive::Neg, &[x_tensor], &no_params()).unwrap();
        let neg_y = eval_primitive(Primitive::Neg, &[y_tensor], &no_params()).unwrap();
        let sum_neg = eval_primitive(Primitive::Add, &[neg_x, neg_y], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&neg_sum),
            extract_f64_scalar(&sum_neg),
            1e-14,
            &format!("Neg(Add({}, {})) = Add(Neg({}), Neg({}))", x, y, x, y),
        );
    }
}

// ======================== METAMORPHIC: tensor Neg involution ========================

#[test]
fn metamorphic_neg_tensor_involution() {
    // For tensor: Neg(Neg(x)) = x
    let data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let input = make_f64_tensor(&[5], data.clone());

    let neg1 = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let neg2 = eval_primitive(Primitive::Neg, &[neg1], &no_params()).unwrap();

    let result = extract_f64_vec(&neg2);
    for (i, (&orig, &rec)) in data.iter().zip(result.iter()).enumerate() {
        assert_eq!(rec, orig, "element {}: Neg(Neg(x)) = x", i);
    }
}
