//! Oracle tests for Tanh (hyperbolic tangent) primitive.
//!
//! tanh(x) = (e^x - e^-x) / (e^x + e^-x) = sinh(x) / cosh(x)
//!
//! Properties:
//! - tanh(0) = 0
//! - tanh(x) → 1 as x → +∞
//! - tanh(x) → -1 as x → -∞
//! - tanh is odd: tanh(-x) = -tanh(x)
//! - Output always in (-1, 1) for finite x
//!
//! Tests:
//! - Zero: tanh(0) = 0
//! - Positive/negative values
//! - Large values (saturation)
//! - Infinity: tanh(+inf) = 1, tanh(-inf) = -1
//! - NaN propagation
//! - Odd function property
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
fn oracle_tanh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "tanh(0) = 0");
}

#[test]
fn oracle_tanh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "tanh(-0.0) = 0");
}

// ======================== Positive Values ========================

#[test]
fn oracle_tanh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0_f64.tanh(),
        1e-14,
        "tanh(1)",
    );
}

#[test]
fn oracle_tanh_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.tanh(),
        1e-14,
        "tanh(2)",
    );
}

#[test]
fn oracle_tanh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5_f64.tanh(),
        1e-14,
        "tanh(0.5)",
    );
}

// ======================== Negative Values ========================

#[test]
fn oracle_tanh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-1.0_f64).tanh(),
        1e-14,
        "tanh(-1)",
    );
}

#[test]
fn oracle_tanh_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-2.0_f64).tanh(),
        1e-14,
        "tanh(-2)",
    );
}

#[test]
fn oracle_tanh_neg_half() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-0.5_f64).tanh(),
        1e-14,
        "tanh(-0.5)",
    );
}

// ======================== Large Values (Saturation) ========================

#[test]
fn oracle_tanh_large_positive() {
    // tanh(20) ≈ 1
    let input = make_f64_tensor(&[], vec![20.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1.0, 1e-14, "tanh(20) ≈ 1");
}

#[test]
fn oracle_tanh_large_negative() {
    // tanh(-20) ≈ -1
    let input = make_f64_tensor(&[], vec![-20.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, -1.0, 1e-14, "tanh(-20) ≈ -1");
}

#[test]
fn oracle_tanh_very_large_positive() {
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "tanh(100) = 1");
}

#[test]
fn oracle_tanh_very_large_negative() {
    let input = make_f64_tensor(&[], vec![-100.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "tanh(-100) = -1");
}

// ======================== Infinity ========================

#[test]
fn oracle_tanh_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "tanh(+inf) = 1");
}

#[test]
fn oracle_tanh_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "tanh(-inf) = -1");
}

// ======================== NaN ========================

#[test]
fn oracle_tanh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "tanh(NaN) = NaN");
}

// ======================== Bounds: output in (-1, 1) for finite x ========================

#[test]
fn oracle_tanh_bounds_positive() {
    for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > 0.0, "tanh({}) > 0", x);
        assert!(val < 1.0, "tanh({}) < 1", x);
    }
}

#[test]
fn oracle_tanh_bounds_negative() {
    for x in [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > -1.0, "tanh({}) > -1", x);
        assert!(val < 0.0, "tanh({}) < 0", x);
    }
}

// ======================== Odd Function: tanh(-x) = -tanh(x) ========================

#[test]
fn oracle_tanh_odd_function() {
    for x in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Tanh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Tanh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("tanh(-{}) = -tanh({})", x, x),
        );
    }
}

// ======================== Monotonicity ========================

#[test]
fn oracle_tanh_monotonic() {
    let inputs = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "tanh should be monotonically increasing"
        );
    }
}

// ======================== Small Values (Linear Approximation) ========================

#[test]
fn oracle_tanh_small() {
    // For small x, tanh(x) ≈ x
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e-10, 1e-20, "tanh(1e-10) ≈ 1e-10");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_tanh_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).tanh(), 1e-14, "tanh(-2)");
    assert_close(vals[1], (-1.0_f64).tanh(), 1e-14, "tanh(-1)");
    assert_eq!(vals[2], 0.0, "tanh(0)");
    assert_close(vals[3], 1.0_f64.tanh(), 1e-14, "tanh(1)");
    assert_close(vals[4], 2.0_f64.tanh(), 1e-14, "tanh(2)");
}

#[test]
fn oracle_tanh_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "tanh(0)");
    assert_eq!(vals[1], 1.0, "tanh(+inf)");
    assert_eq!(vals[2], -1.0, "tanh(-inf)");
    assert!(vals[3].is_nan(), "tanh(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_tanh_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-3.0_f64).tanh(), 1e-14, "tanh(-3)");
    assert_eq!(vals[2], 0.0, "tanh(0)");
    assert_close(vals[5], 5.0_f64.tanh(), 1e-14, "tanh(5)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_tanh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-4.0_f64).tanh(), 1e-14, "tanh(-4)");
    assert_eq!(vals[3], 0.0, "tanh(0)");
    assert_close(vals[7], 8.0_f64.tanh(), 1e-14, "tanh(8)");
}

// ======================== Identity: tanh computed vs. formula ========================

#[test]
fn oracle_tanh_identity() {
    for x in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = x.tanh();
        assert_close(val, expected, 1e-14, &format!("tanh({}) = stdlib tanh", x));
    }
}

// ======================== Derivative at zero: tanh'(0) = 1 ========================

#[test]
fn oracle_tanh_derivative_at_zero() {
    // Using small h to approximate derivative: (tanh(h) - tanh(0)) / h ≈ 1
    let h = 1e-8;
    let input = make_f64_tensor(&[], vec![h]);
    let result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    let approx_derivative = val / h;
    assert_close(approx_derivative, 1.0, 1e-6, "tanh'(0) ≈ 1");
}

// ======================== METAMORPHIC: atanh(tanh(x)) = x ========================

#[test]
fn metamorphic_atanh_tanh_identity() {
    // atanh(tanh(x)) = x for x in reasonable range (|x| < ~18 to avoid saturation)
    for x in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let tanh_result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
        let atanh_tanh = eval_primitive(Primitive::Atanh, &[tanh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&atanh_tanh),
            x,
            1e-12,
            &format!("atanh(tanh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tanh(atanh(x)) = x ========================

#[test]
fn metamorphic_tanh_atanh_identity() {
    // tanh(atanh(x)) = x for x in (-1, 1)
    for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let atanh_result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let tanh_atanh = eval_primitive(Primitive::Tanh, &[atanh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&tanh_atanh),
            x,
            1e-14,
            &format!("tanh(atanh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_tanh_tensor_roundtrip() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let tanh_result = eval_primitive(Primitive::Tanh, std::slice::from_ref(&input), &no_params()).unwrap();
    let atanh_tanh = eval_primitive(Primitive::Atanh, &[tanh_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&atanh_tanh);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(*rt, *orig, 1e-12, &format!("atanh(tanh({})) = {}", orig, orig));
    }
}
