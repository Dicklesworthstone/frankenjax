//! Oracle tests for Square primitive.
//!
//! square(x) = x * x
//!
//! Tests:
//! - Basic values
//! - Negative values (square always positive)
//! - Zero
//! - Infinity
//! - NaN propagation
//! - Identity: square(x) = x * x

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
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
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
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

// ======================== Integer Positive ========================

#[test]
fn oracle_square_i64_zero() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_square_i64_one() {
    let input = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_square_i64_two() {
    let input = make_i64_tensor(&[], vec![2]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4);
}

#[test]
fn oracle_square_i64_three() {
    let input = make_i64_tensor(&[], vec![3]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 9);
}

#[test]
fn oracle_square_i64_ten() {
    let input = make_i64_tensor(&[], vec![10]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 100);
}

// ======================== Integer Negative ========================

#[test]
fn oracle_square_i64_neg_one() {
    // square(-1) = 1
    let input = make_i64_tensor(&[], vec![-1]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_square_i64_neg_two() {
    // square(-2) = 4
    let input = make_i64_tensor(&[], vec![-2]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4);
}

#[test]
fn oracle_square_i64_neg_ten() {
    // square(-10) = 100
    let input = make_i64_tensor(&[], vec![-10]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 100);
}

// ======================== Float Positive ========================

#[test]
fn oracle_square_f64_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

#[test]
fn oracle_square_f64_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "square(1.0)");
}

#[test]
fn oracle_square_f64_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 4.0, 1e-14, "square(2.0)");
}

#[test]
fn oracle_square_f64_half() {
    // square(0.5) = 0.25
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "square(0.5)");
}

#[test]
fn oracle_square_f64_sqrt2() {
    // square(sqrt(2)) = 2
    let input = make_f64_tensor(&[], vec![2.0_f64.sqrt()]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "square(sqrt(2))");
}

// ======================== Float Negative ========================

#[test]
fn oracle_square_f64_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "square(-1.0)");
}

#[test]
fn oracle_square_f64_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 4.0, 1e-14, "square(-2.0)");
}

#[test]
fn oracle_square_f64_neg_half() {
    // square(-0.5) = 0.25
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "square(-0.5)");
}

// ======================== Float Zero Signs ========================

#[test]
fn oracle_square_f64_neg_zero() {
    // square(-0.0) = 0.0 (positive zero)
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

// ======================== Float Infinity ========================

#[test]
fn oracle_square_f64_pos_infinity() {
    // square(+inf) = +inf
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "square(+inf) = +inf");
}

#[test]
fn oracle_square_f64_neg_infinity() {
    // square(-inf) = +inf (negative squared is positive)
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "square(-inf) = +inf");
}

// ======================== Float NaN ========================

#[test]
fn oracle_square_f64_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "square(NaN) = NaN");
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_square_f64_very_small() {
    // square(1e-100) = 1e-200
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-200,
        1e-214,
        "square(1e-100)",
    );
}

#[test]
fn oracle_square_f64_very_large() {
    // square(1e100) = 1e200
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1e200, 1e186, "square(1e100)");
}

// ======================== 1D Tensor Integer ========================

#[test]
fn oracle_square_i64_1d() {
    let input = make_i64_tensor(&[5], vec![-2, -1, 0, 1, 2]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![4, 1, 0, 1, 4]);
}

// ======================== 1D Tensor Float ========================

#[test]
fn oracle_square_f64_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 4.0, 1e-14, "square(-2)");
    assert_close(vals[1], 1.0, 1e-14, "square(-1)");
    assert_eq!(vals[2], 0.0);
    assert_close(vals[3], 1.0, 1e-14, "square(1)");
    assert_close(vals[4], 4.0, 1e-14, "square(2)");
}

#[test]
fn oracle_square_f64_1d_mixed() {
    let input = make_f64_tensor(&[4], vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
    assert!(vals[1].is_infinite() && vals[1] > 0.0);
    assert_eq!(vals[2], 0.0);
    assert!(vals[3].is_nan());
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_square_i64_2d() {
    let input = make_i64_tensor(&[2, 3], vec![-3, -2, -1, 0, 1, 2]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![9, 4, 1, 0, 1, 4]);
}

#[test]
fn oracle_square_f64_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "square(1)");
    assert_close(vals[1], 4.0, 1e-14, "square(2)");
    assert_close(vals[2], 9.0, 1e-14, "square(3)");
    assert_close(vals[3], 16.0, 1e-14, "square(4)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_square_i64_3d() {
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 4, 9, 16, 25, 36, 49, 64]);
}

// ======================== Identity: square(x) = x * x ========================

#[test]
fn oracle_square_identity_i64() {
    for x in [-5, -2, -1, 0, 1, 2, 5, 10] {
        let input = make_i64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
        assert_eq!(
            extract_i64_scalar(&result),
            x * x,
            "square({}) = {} * {}",
            x,
            x,
            x
        );
    }
}

#[test]
fn oracle_square_identity_f64() {
    for x in [-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x * x,
            1e-14,
            &format!("square({}) = {} * {}", x, x, x),
        );
    }
}

// ======================== Symmetry: square(x) = square(-x) ========================

#[test]
fn oracle_square_symmetry() {
    for x in [1.0, 2.0, 0.5, 10.0, 100.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Square, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Square, &[input_neg], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&result_pos),
            extract_f64_scalar(&result_neg),
            1e-14,
            &format!("square({}) = square(-{})", x, x),
        );
    }
}

// ======================== METAMORPHIC: square(x) = Mul(x, x) ========================

#[test]
fn metamorphic_square_equals_mul() {
    // square(x) = Mul(x, x) using Mul primitive
    for x in [-2.5, -1.0, 0.0, 1.0, 2.5, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let square_result = eval_primitive(Primitive::Square, std::slice::from_ref(&input), &no_params()).unwrap();
        let mul_result = eval_primitive(Primitive::Mul, &[input.clone(), input], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&square_result),
            extract_f64_scalar(&mul_result),
            1e-14,
            &format!("square({}) = Mul({}, {})", x, x, x),
        );
    }
}

// ======================== METAMORPHIC: sqrt(square(x)) = abs(x) ========================

#[test]
fn metamorphic_sqrt_square_abs() {
    // sqrt(square(x)) = abs(x) using Sqrt and Abs primitives
    for x in [-2.5, -1.0, 0.0, 1.0, 2.5, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let squared = eval_primitive(Primitive::Square, std::slice::from_ref(&input), &no_params()).unwrap();
        let sqrt_squared = eval_primitive(Primitive::Sqrt, &[squared], &no_params()).unwrap();
        let abs_x = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sqrt_squared),
            extract_f64_scalar(&abs_x),
            1e-14,
            &format!("sqrt(square({})) = abs({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: square(Neg(x)) = square(x) ========================

#[test]
fn metamorphic_square_negation_invariant() {
    // square(Neg(x)) = square(x) (even function) using Neg primitive
    for x in [1.0, 2.0, 0.5, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let neg_x = eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();

        let square_x = eval_primitive(Primitive::Square, &[input], &no_params()).unwrap();
        let square_neg_x = eval_primitive(Primitive::Square, &[neg_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&square_neg_x),
            extract_f64_scalar(&square_x),
            1e-14,
            &format!("square(Neg({})) = square({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor square = Mul ========================

#[test]
fn metamorphic_square_tensor_mul() {
    // For tensor: square(x) = Mul(x, x)
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let input = make_f64_tensor(&[5], data);

    let square_result = eval_primitive(Primitive::Square, std::slice::from_ref(&input), &no_params()).unwrap();
    let mul_result = eval_primitive(Primitive::Mul, &[input.clone(), input], &no_params()).unwrap();

    assert_eq!(extract_shape(&square_result), vec![5]);
    let sq_vec = extract_f64_vec(&square_result);
    let mul_vec = extract_f64_vec(&mul_result);
    for (i, (&sq, &mul)) in sq_vec.iter().zip(mul_vec.iter()).enumerate() {
        assert_close(sq, mul, 1e-14, &format!("element {}", i));
    }
}
