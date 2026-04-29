//! Oracle tests for Sinh (hyperbolic sine) primitive.
//!
//! sinh(x) = (e^x - e^-x) / 2
//!
//! Properties:
//! - sinh(0) = 0
//! - sinh is odd: sinh(-x) = -sinh(x)
//! - sinh(x) → +∞ as x → +∞
//! - sinh(x) → -∞ as x → -∞
//!
//! Tests:
//! - Zero: sinh(0) = 0
//! - Positive/negative values
//! - Odd function property
//! - Infinity: sinh(+inf) = +inf, sinh(-inf) = -inf
//! - NaN propagation
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

// ======================== Zero ========================

#[test]
fn oracle_sinh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "sinh(0) = 0");
}

#[test]
fn oracle_sinh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "sinh(-0.0) = 0");
}

// ======================== Positive Values ========================

#[test]
fn oracle_sinh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0_f64.sinh(), 1e-14, "sinh(1)");
}

#[test]
fn oracle_sinh_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0_f64.sinh(), 1e-14, "sinh(2)");
}

#[test]
fn oracle_sinh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5_f64.sinh(), 1e-14, "sinh(0.5)");
}

// ======================== Negative Values ========================

#[test]
fn oracle_sinh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-1.0_f64).sinh(),
        1e-14,
        "sinh(-1)",
    );
}

#[test]
fn oracle_sinh_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-2.0_f64).sinh(),
        1e-14,
        "sinh(-2)",
    );
}

// ======================== Large Values ========================

#[test]
fn oracle_sinh_large_positive() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        10.0_f64.sinh(),
        1e-10,
        "sinh(10)",
    );
}

#[test]
fn oracle_sinh_large_negative() {
    let input = make_f64_tensor(&[], vec![-10.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-10.0_f64).sinh(),
        1e-10,
        "sinh(-10)",
    );
}

// ======================== Infinity ========================

#[test]
fn oracle_sinh_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "sinh(+inf) = +inf");
}

#[test]
fn oracle_sinh_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "sinh(-inf) = -inf");
}

// ======================== NaN ========================

#[test]
fn oracle_sinh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sinh(NaN) = NaN");
}

// ======================== Odd Function: sinh(-x) = -sinh(x) ========================

#[test]
fn oracle_sinh_odd_function() {
    for x in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Sinh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Sinh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("sinh(-{}) = -sinh({})", x, x),
        );
    }
}

// ======================== Identity: cosh^2(x) - sinh^2(x) = 1 ========================

#[test]
fn oracle_sinh_cosh_identity() {
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let cosh_result = eval_primitive(Primitive::Cosh, &[input.clone()], &no_params()).unwrap();
        let sinh_result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();

        let cosh_val = extract_f64_scalar(&cosh_result);
        let sinh_val = extract_f64_scalar(&sinh_result);

        let identity = cosh_val * cosh_val - sinh_val * sinh_val;
        assert_close(
            identity,
            1.0,
            1e-13,
            &format!("cosh^2({}) - sinh^2({}) = 1", x, x),
        );
    }
}

// ======================== Monotonicity ========================

#[test]
fn oracle_sinh_monotonic() {
    let inputs = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "sinh should be monotonically increasing"
        );
    }
}

// ======================== Small Values ========================

#[test]
fn oracle_sinh_small() {
    // For small x, sinh(x) ≈ x
    let x = 1e-10;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, x, 1e-20, "sinh(1e-10) ≈ 1e-10");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_sinh_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).sinh(), 1e-14, "sinh(-2)");
    assert_close(vals[1], (-1.0_f64).sinh(), 1e-14, "sinh(-1)");
    assert_eq!(vals[2], 0.0, "sinh(0)");
    assert_close(vals[3], 1.0_f64.sinh(), 1e-14, "sinh(1)");
    assert_close(vals[4], 2.0_f64.sinh(), 1e-14, "sinh(2)");
}

#[test]
fn oracle_sinh_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "sinh(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "sinh(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "sinh(-inf)");
    assert!(vals[3].is_nan(), "sinh(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_sinh_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).sinh(), 1e-14, "sinh(-2)");
    assert_eq!(vals[2], 0.0, "sinh(0)");
    assert_close(vals[5], 3.0_f64.sinh(), 1e-14, "sinh(3)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_sinh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-3.0_f64).sinh(), 1e-14, "sinh(-3)");
    assert_eq!(vals[3], 0.0, "sinh(0)");
    assert_close(vals[7], 4.0_f64.sinh(), 1e-14, "sinh(4)");
}

// ======================== Identity: computed vs. formula ========================

#[test]
fn oracle_sinh_identity_formula() {
    for x in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = (x.exp() - (-x).exp()) / 2.0;
        assert_close(
            val,
            expected,
            1e-14,
            &format!("sinh({}) = (e^{} - e^-{}) / 2", x, x, x),
        );
    }
}
