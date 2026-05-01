//! Oracle tests for Abs primitive.
//!
//! abs(x) = |x| = x if x >= 0, -x if x < 0
//!
//! Properties:
//! - abs(x) >= 0 for all x (non-negativity)
//! - abs(-x) = abs(x) (symmetry/evenness)
//! - abs(x * y) = abs(x) * abs(y) (multiplicativity)
//! - abs(x) = 0 iff x = 0
//! - For complex: abs(z) = sqrt(re^2 + im^2) (magnitude)
//!
//! Tests:
//! - Positive, negative, zero values
//! - Mathematical properties
//! - Float special values (inf, NaN)
//! - Integer types
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
fn oracle_abs_positive_f64() {
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_abs_negative_f64() {
    let input = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_abs_zero_f64() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

#[test]
fn oracle_abs_neg_zero_f64() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

// ====================== INTEGER VALUES ======================

#[test]
fn oracle_abs_positive_i64() {
    let input = make_i64_tensor(&[], vec![42]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 42);
}

#[test]
fn oracle_abs_negative_i64() {
    let input = make_i64_tensor(&[], vec![-42]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 42);
}

#[test]
fn oracle_abs_zero_i64() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ====================== SPECIAL FLOAT VALUES ======================

#[test]
fn oracle_abs_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_abs_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_abs_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "abs(NaN) = NaN");
}

// ====================== SYMMETRY PROPERTY ======================

#[test]
fn oracle_abs_symmetry() {
    // abs(-x) = abs(x)
    for x in [1.0, 2.5, 100.0, 0.001, f64::INFINITY] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Abs, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Abs, &[neg_input], &no_params()).unwrap();
        assert_eq!(
            extract_f64_scalar(&pos_result),
            extract_f64_scalar(&neg_result),
            "abs({}) = abs(-{})",
            x,
            x
        );
    }
}

// ====================== NON-NEGATIVITY PROPERTY ======================

#[test]
fn oracle_abs_non_negative() {
    // abs(x) >= 0 for all x
    for x in [-100.0, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val >= 0.0, "abs({}) = {} should be non-negative", x, val);
    }
}

// ====================== MULTIPLICATIVITY PROPERTY ======================

#[test]
fn oracle_abs_multiplicative() {
    // abs(x * y) = abs(x) * abs(y)
    let test_pairs = [
        (2.0, 3.0),
        (-2.0, 3.0),
        (2.0, -3.0),
        (-2.0, -3.0),
        (0.5, 4.0),
    ];
    for (x, y) in test_pairs {
        let product = x * y;
        let abs_product_input = make_f64_tensor(&[], vec![product]);
        let abs_x_input = make_f64_tensor(&[], vec![x]);
        let abs_y_input = make_f64_tensor(&[], vec![y]);

        let abs_product = extract_f64_scalar(
            &eval_primitive(Primitive::Abs, &[abs_product_input], &no_params()).unwrap(),
        );
        let abs_x = extract_f64_scalar(
            &eval_primitive(Primitive::Abs, &[abs_x_input], &no_params()).unwrap(),
        );
        let abs_y = extract_f64_scalar(
            &eval_primitive(Primitive::Abs, &[abs_y_input], &no_params()).unwrap(),
        );

        assert_close(
            abs_product,
            abs_x * abs_y,
            1e-14,
            &format!("abs({} * {}) = abs({}) * abs({})", x, y, x, y),
        );
    }
}

// ====================== ZERO IFF ZERO PROPERTY ======================

#[test]
fn oracle_abs_zero_iff_zero() {
    // abs(x) = 0 iff x = 0
    let zero_input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Abs, &[zero_input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);

    for x in [0.001, -0.001, 1e-100, -1e-100] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > 0.0, "abs({}) = {} should be positive", x, val);
    }
}

// ====================== COMPLEX MAGNITUDE ======================

#[test]
fn oracle_abs_complex_3_4i() {
    // abs(3 + 4i) = 5 (3-4-5 triangle)
    let input = make_complex128_tensor(&[], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 5.0, 1e-14, "abs(3+4i)");
}

#[test]
fn oracle_abs_complex_pure_real() {
    // abs(5 + 0i) = 5
    let input = make_complex128_tensor(&[], vec![(5.0, 0.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 5.0, 1e-14, "abs(5+0i)");
}

#[test]
fn oracle_abs_complex_pure_imag() {
    // abs(0 + 5i) = 5
    let input = make_complex128_tensor(&[], vec![(0.0, 5.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 5.0, 1e-14, "abs(0+5i)");
}

#[test]
fn oracle_abs_complex_negative() {
    // abs(-3 - 4i) = 5
    let input = make_complex128_tensor(&[], vec![(-3.0, -4.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 5.0, 1e-14, "abs(-3-4i)");
}

#[test]
fn oracle_abs_complex_unit() {
    // abs(cos(θ) + i*sin(θ)) = 1
    let theta = std::f64::consts::FRAC_PI_4;
    let re = theta.cos();
    let im = theta.sin();
    let input = make_complex128_tensor(&[], vec![(re, im)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "abs(e^(iπ/4))");
}

#[test]
fn oracle_abs_complex64_returns_float32_literals() {
    let input = make_complex64_tensor(&[2], vec![(3.0, 4.0), (5.0, 12.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    let Value::Tensor(tensor) = result else {
        unreachable!("expected tensor result");
    };
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.shape.dims, vec![2]);
    assert_eq!(
        tensor.elements,
        vec![Literal::from_f32(5.0), Literal::from_f32(13.0)]
    );
}

#[test]
fn oracle_abs_complex128_returns_float64_literals() {
    let input = make_complex128_tensor(&[2], vec![(3.0, 4.0), (5.0, 12.0)]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    let Value::Tensor(tensor) = result else {
        unreachable!("expected tensor result");
    };
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape.dims, vec![2]);
    assert_eq!(
        tensor.elements,
        vec![Literal::from_f64(5.0), Literal::from_f64(13.0)]
    );
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_abs_1d_f64() {
    let input = make_f64_tensor(&[5], vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 1.0, 0.0, 1.0, 3.0]);
}

#[test]
fn oracle_abs_1d_i64() {
    let input = make_i64_tensor(&[5], vec![-3, -1, 0, 1, 3]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 0, 1, 3]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_abs_2d_f64() {
    let input = make_f64_tensor(&[2, 3], vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ====================== IDEMPOTENCE ======================

#[test]
fn oracle_abs_idempotent() {
    // abs(abs(x)) = abs(x)
    for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let abs1 = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
        let abs2 =
            eval_primitive(Primitive::Abs, std::slice::from_ref(&abs1), &no_params()).unwrap();
        assert_eq!(
            extract_f64_scalar(&abs1),
            extract_f64_scalar(&abs2),
            "abs(abs({})) = abs({})",
            x,
            x
        );
    }
}

// ====================== TRIANGLE INEQUALITY VERIFICATION ======================

#[test]
fn oracle_abs_triangle_inequality_check() {
    // Verify: |x + y| <= |x| + |y|
    let test_pairs = [
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (1.5, 2.5),
    ];
    for (x, y) in test_pairs {
        let sum = x + y;
        let abs_sum = extract_f64_scalar(
            &eval_primitive(
                Primitive::Abs,
                &[make_f64_tensor(&[], vec![sum])],
                &no_params(),
            )
            .unwrap(),
        );
        let abs_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Abs,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let abs_y = extract_f64_scalar(
            &eval_primitive(
                Primitive::Abs,
                &[make_f64_tensor(&[], vec![y])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            abs_sum <= abs_x + abs_y + 1e-14,
            "|{} + {}| = {} <= |{}| + |{}| = {}",
            x,
            y,
            abs_sum,
            x,
            y,
            abs_x + abs_y
        );
    }
}
