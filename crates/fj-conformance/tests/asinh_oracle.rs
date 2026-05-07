//! Oracle tests for Asinh (inverse hyperbolic sine) primitive.
//!
//! asinh(x) = ln(x + sqrt(x² + 1))
//!
//! Properties:
//! - asinh(0) = 0
//! - asinh is odd: asinh(-x) = -asinh(x)
//! - Defined for all real x
//! - Metamorphic: sinh(asinh(x)) = x
//! - Metamorphic: asinh(sinh(x)) = x

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
fn oracle_asinh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "asinh(0) = 0");
}

// ======================== Basic Values ========================

#[test]
fn oracle_asinh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0_f64.asinh(),
        1e-14,
        "asinh(1)",
    );
}

#[test]
fn oracle_asinh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-1.0_f64).asinh(),
        1e-14,
        "asinh(-1)",
    );
}

// ======================== Odd Function: asinh(-x) = -asinh(x) ========================

#[test]
fn oracle_asinh_odd_function() {
    for x in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Asinh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Asinh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("asinh(-{}) = -asinh({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: sinh(asinh(x)) = x ========================

#[test]
fn metamorphic_sinh_asinh_identity() {
    for x in [-100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let asinh_result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
        let sinh_asinh = eval_primitive(Primitive::Sinh, &[asinh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sinh_asinh),
            x,
            1e-12,
            &format!("sinh(asinh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: asinh(sinh(x)) = x ========================

#[test]
fn metamorphic_asinh_sinh_identity() {
    for x in [-10.0, -5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let sinh_result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
        let asinh_sinh = eval_primitive(Primitive::Asinh, &[sinh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&asinh_sinh),
            x,
            1e-12,
            &format!("asinh(sinh({})) = {}", x, x),
        );
    }
}

// ======================== Large Values ========================

#[test]
fn oracle_asinh_large() {
    let input = make_f64_tensor(&[], vec![1e10]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e10_f64.asinh(),
        1e-5,
        "asinh(1e10)",
    );
}

// ======================== Infinity ========================

#[test]
fn oracle_asinh_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_infinite() && extract_f64_scalar(&result) > 0.0,
        "asinh(+inf) = +inf"
    );
}

#[test]
fn oracle_asinh_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_infinite() && extract_f64_scalar(&result) < 0.0,
        "asinh(-inf) = -inf"
    );
}

// ======================== NaN ========================

#[test]
fn oracle_asinh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "asinh(NaN) = NaN");
}

// ======================== Stdlib comparison ========================

#[test]
fn oracle_asinh_stdlib() {
    for x in [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x.asinh(),
            1e-14,
            &format!("asinh({}) vs stdlib", x),
        );
    }
}

// ======================== Tensor Shape Tests ========================

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

#[test]
fn oracle_asinh_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    let expected: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for (v, &x) in vals.iter().zip(expected.iter()) {
        assert_close(*v, x.asinh(), 1e-14, &format!("asinh({})", x));
    }
}

#[test]
fn oracle_asinh_2d() {
    let input = make_f64_tensor(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    let expected: [f64; 4] = [-1.0, 0.0, 1.0, 2.0];
    for (v, &x) in vals.iter().zip(expected.iter()) {
        assert_close(*v, x.asinh(), 1e-14, &format!("asinh({})", x));
    }
}

#[test]
fn oracle_asinh_tensor_special_values() {
    let input = make_f64_tensor(&[4], vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0]);
    let result = eval_primitive(Primitive::Asinh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert!(vals[0].is_infinite() && vals[0] > 0.0, "asinh(+inf) = +inf");
    assert!(vals[1].is_infinite() && vals[1] < 0.0, "asinh(-inf) = -inf");
    assert!(vals[2].is_nan(), "asinh(NaN) = NaN");
    assert_eq!(vals[3], 0.0, "asinh(0) = 0");
}

// ======================== Tensor Metamorphic Tests ========================

#[test]
fn metamorphic_asinh_tensor_odd_function() {
    let input_pos = make_f64_tensor(&[4], vec![0.5, 1.0, 2.0, 5.0]);
    let input_neg = make_f64_tensor(&[4], vec![-0.5, -1.0, -2.0, -5.0]);

    let result_pos = eval_primitive(Primitive::Asinh, &[input_pos], &no_params()).unwrap();
    let result_neg = eval_primitive(Primitive::Asinh, &[input_neg], &no_params()).unwrap();

    let vals_pos = extract_f64_vec(&result_pos);
    let vals_neg = extract_f64_vec(&result_neg);

    for (vp, vn) in vals_pos.iter().zip(vals_neg.iter()) {
        assert_close(*vn, -(*vp), 1e-14, "asinh(-x) = -asinh(x)");
    }
}

#[test]
fn metamorphic_asinh_tensor_sinh_inverse() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let asinh_result = eval_primitive(Primitive::Asinh, std::slice::from_ref(&input), &no_params()).unwrap();
    let sinh_asinh = eval_primitive(Primitive::Sinh, &[asinh_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&sinh_asinh);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(*rt, *orig, 1e-12, &format!("sinh(asinh({})) = {}", orig, orig));
    }
}
