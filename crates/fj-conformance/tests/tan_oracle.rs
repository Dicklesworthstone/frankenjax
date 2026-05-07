//! Oracle tests for Tan primitive.
//!
//! tan(x) = sin(x)/cos(x)
//!
//! Domain: All real numbers except x = (2n+1)*π/2
//! Range: (-inf, inf)
//!
//! Tests:
//! - tan(0) = 0
//! - tan(π/4) = 1
//! - tan(-π/4) = -1
//! - tan(π/3) = sqrt(3)
//! - tan(π/6) = 1/sqrt(3)
//! - tan(π) = 0
//! - Odd function: tan(-x) = -tan(x)
//! - Period: tan(x + π) = tan(x)
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
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
fn oracle_tan_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "tan(0) = 0");
}

#[test]
fn oracle_tan_pi_over_4() {
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "tan(π/4) = 1");
}

#[test]
fn oracle_tan_neg_pi_over_4() {
    let input = make_f64_tensor(&[], vec![-std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "tan(-π/4) = -1");
}

#[test]
fn oracle_tan_pi_over_3() {
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_3]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt(),
        1e-14,
        "tan(π/3) = sqrt(3)",
    );
}

#[test]
fn oracle_tan_pi_over_6() {
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_6]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 3.0_f64.sqrt(),
        1e-14,
        "tan(π/6) = 1/sqrt(3)",
    );
}

#[test]
fn oracle_tan_pi() {
    let input = make_f64_tensor(&[], vec![std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "tan(π) = 0");
}

#[test]
fn oracle_tan_two_pi() {
    let input = make_f64_tensor(&[], vec![2.0 * std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-13, "tan(2π) = 0");
}

// ====================== NEAR ASYMPTOTES ======================

#[test]
fn oracle_tan_near_pi_over_2() {
    // tan approaches +infinity as x approaches π/2 from below
    let slightly_less = std::f64::consts::FRAC_PI_2 - 0.001;
    let input = make_f64_tensor(&[], vec![slightly_less]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val > 100.0, "tan(π/2 - ε) should be very large positive");
}

#[test]
fn oracle_tan_near_neg_pi_over_2() {
    // tan approaches -infinity as x approaches -π/2 from above
    let slightly_more = -std::f64::consts::FRAC_PI_2 + 0.001;
    let input = make_f64_tensor(&[], vec![slightly_more]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val < -100.0, "tan(-π/2 + ε) should be very large negative");
}

// ====================== NaN ======================

#[test]
fn oracle_tan_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "tan(NaN) = NaN");
}

// ====================== INFINITY ======================

#[test]
fn oracle_tan_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "tan(inf) = NaN");
}

// ====================== ODD FUNCTION ======================

#[test]
fn oracle_tan_odd_function() {
    // tan(-x) = -tan(x)
    for x in [0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4, 2.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Tan, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Tan, &[neg_input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&neg_result),
            -extract_f64_scalar(&pos_result),
            1e-14,
            &format!("tan(-{}) = -tan({})", x, x),
        );
    }
}

// ====================== PERIOD ======================

#[test]
fn oracle_tan_period() {
    // tan(x + π) = tan(x)
    for x in [0.0, 0.5, 1.0, -0.5, -1.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let input2 = make_f64_tensor(&[], vec![x + std::f64::consts::PI]);
        let result1 = eval_primitive(Primitive::Tan, &[input1], &no_params()).unwrap();
        let result2 = eval_primitive(Primitive::Tan, &[input2], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result1),
            extract_f64_scalar(&result2),
            1e-13,
            &format!("tan({}) = tan({} + π)", x, x),
        );
    }
}

// ====================== SIN/COS RELATIONSHIP ======================

#[test]
fn oracle_tan_sin_cos_relationship() {
    // tan(x) = sin(x) / cos(x)
    for x in [0.0, 0.25, 0.5, 1.0, 2.0] {
        let tan_input = make_f64_tensor(&[], vec![x]);
        let sin_input = make_f64_tensor(&[], vec![x]);
        let cos_input = make_f64_tensor(&[], vec![x]);

        let tan_result = eval_primitive(Primitive::Tan, &[tan_input], &no_params()).unwrap();
        let sin_result = eval_primitive(Primitive::Sin, &[sin_input], &no_params()).unwrap();
        let cos_result = eval_primitive(Primitive::Cos, &[cos_input], &no_params()).unwrap();

        let tan_val = extract_f64_scalar(&tan_result);
        let sin_val = extract_f64_scalar(&sin_result);
        let cos_val = extract_f64_scalar(&cos_result);

        assert_close(
            tan_val,
            sin_val / cos_val,
            1e-14,
            &format!("tan({}) = sin({})/cos({})", x, x, x),
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_tan_1d() {
    let input = make_f64_tensor(
        &[5],
        vec![
            0.0,
            std::f64::consts::FRAC_PI_4,
            -std::f64::consts::FRAC_PI_4,
            std::f64::consts::PI,
            f64::NAN,
        ],
    );
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "tan(0)");
    assert_close(vals[1], 1.0, 1e-14, "tan(π/4)");
    assert_close(vals[2], -1.0, 1e-14, "tan(-π/4)");
    assert_close(vals[3], 0.0, 1e-14, "tan(π)");
    assert!(vals[4].is_nan(), "tan(NaN)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_tan_2d() {
    let input = make_f64_tensor(
        &[2, 2],
        vec![
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_3,
        ],
    );
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0);
    assert_close(vals[1], 1.0 / 3.0_f64.sqrt(), 1e-14, "tan(π/6)");
    assert_close(vals[2], 1.0, 1e-14, "tan(π/4)");
    assert_close(vals[3], 3.0_f64.sqrt(), 1e-14, "tan(π/3)");
}

// ====================== SMALL VALUES ======================

#[test]
fn oracle_tan_small_values() {
    // For small |x|, tan(x) ≈ x
    for x in [0.001, 0.0001, 1e-10] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert_close(val, x, x * 0.01, &format!("tan({}) ≈ {}", x, x));
    }
}

// ====================== SEC^2 IDENTITY ======================

#[test]
fn oracle_tan_derivative_identity() {
    // 1 + tan^2(x) = sec^2(x) = 1/cos^2(x)
    for x in [0.0, 0.25, 0.5, 1.0] {
        let tan_input = make_f64_tensor(&[], vec![x]);
        let cos_input = make_f64_tensor(&[], vec![x]);

        let tan_result = eval_primitive(Primitive::Tan, &[tan_input], &no_params()).unwrap();
        let cos_result = eval_primitive(Primitive::Cos, &[cos_input], &no_params()).unwrap();

        let tan_val = extract_f64_scalar(&tan_result);
        let cos_val = extract_f64_scalar(&cos_result);

        let lhs = 1.0 + tan_val * tan_val;
        let rhs = 1.0 / (cos_val * cos_val);
        assert_close(lhs, rhs, 1e-13, &format!("1 + tan^2({}) = sec^2({})", x, x));
    }
}

// ====================== METAMORPHIC PROPERTIES ======================

#[test]
fn metamorphic_tan_equals_sin_over_cos() {
    // tan(x) = sin(x) / cos(x) for all x where cos(x) != 0
    for x in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let tan_result = eval_primitive(Primitive::Tan, std::slice::from_ref(&input), &no_params()).unwrap();
        let sin_result = eval_primitive(Primitive::Sin, std::slice::from_ref(&input), &no_params()).unwrap();
        let cos_result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();

        let tan_val = extract_f64_scalar(&tan_result);
        let sin_val = extract_f64_scalar(&sin_result);
        let cos_val = extract_f64_scalar(&cos_result);

        if cos_val.abs() > 1e-10 {
            let sin_over_cos = sin_val / cos_val;
            assert_close(
                tan_val,
                sin_over_cos,
                1e-12,
                &format!("tan({}) = sin({})/cos({})", x, x, x),
            );
        }
    }
}

#[test]
fn metamorphic_tan_odd_function() {
    // tan(-x) = -tan(x) for all x
    for x in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);

        let tan_pos = eval_primitive(Primitive::Tan, &[pos_input], &no_params()).unwrap();
        let tan_neg = eval_primitive(Primitive::Tan, &[neg_input], &no_params()).unwrap();

        let pos_val = extract_f64_scalar(&tan_pos);
        let neg_val = extract_f64_scalar(&tan_neg);

        assert_close(
            neg_val,
            -pos_val,
            1e-12,
            &format!("tan(-{}) = -tan({})", x, x),
        );
    }
}

#[test]
fn metamorphic_tan_period_pi() {
    // tan(x + π) = tan(x) for all x
    let pi = std::f64::consts::PI;
    for x in [0.0, 0.25, 0.5, 1.0, 1.2] {
        let input_x = make_f64_tensor(&[], vec![x]);
        let input_x_plus_pi = make_f64_tensor(&[], vec![x + pi]);

        let tan_x = eval_primitive(Primitive::Tan, &[input_x], &no_params()).unwrap();
        let tan_x_pi = eval_primitive(Primitive::Tan, &[input_x_plus_pi], &no_params()).unwrap();

        let val_x = extract_f64_scalar(&tan_x);
        let val_x_pi = extract_f64_scalar(&tan_x_pi);

        assert_close(
            val_x,
            val_x_pi,
            1e-12,
            &format!("tan({}) = tan({} + π)", x, x),
        );
    }
}

#[test]
fn metamorphic_tan_tensor_sin_over_cos() {
    // tan(x) = sin(x) / cos(x) for 1D tensor
    let x = make_f64_tensor(&[5], vec![0.1, 0.5, 1.0, 1.5, 2.0]);

    let tan_result = eval_primitive(Primitive::Tan, std::slice::from_ref(&x), &no_params()).unwrap();
    let sin_result = eval_primitive(Primitive::Sin, std::slice::from_ref(&x), &no_params()).unwrap();
    let cos_result = eval_primitive(Primitive::Cos, &[x], &no_params()).unwrap();

    let tan_vals = extract_f64_vec(&tan_result);
    let sin_vals = extract_f64_vec(&sin_result);
    let cos_vals = extract_f64_vec(&cos_result);

    for i in 0..5 {
        if cos_vals[i].abs() > 1e-10 {
            let expected = sin_vals[i] / cos_vals[i];
            assert_close(
                tan_vals[i],
                expected,
                1e-12,
                &format!("tan = sin/cos element {}", i),
            );
        }
    }
}
