//! Oracle tests for Acos (arc cosine) primitive.
//!
//! acos(x) = inverse cosine of x
//!
//! Domain: [-1, 1]
//! Range: [0, π]
//!
//! Tests:
//! - Boundary: acos(-1) = π, acos(1) = 0
//! - Special: acos(0) = π/2
//! - Out of domain: acos(x) = NaN for |x| > 1
//! - Infinity: acos(±inf) = NaN
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

// ======================== Boundary Values ========================

#[test]
fn oracle_acos_one() {
    // acos(1) = 0
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "acos(1)");
}

#[test]
fn oracle_acos_neg_one() {
    // acos(-1) = π
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::PI,
        1e-14,
        "acos(-1)",
    );
}

#[test]
fn oracle_acos_zero() {
    // acos(0) = π/2
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "acos(0)",
    );
}

#[test]
fn oracle_acos_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "acos(-0.0)",
    );
}

// ======================== Common Values ========================

#[test]
fn oracle_acos_half() {
    // acos(0.5) = π/3
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_3,
        1e-14,
        "acos(0.5)",
    );
}

#[test]
fn oracle_acos_neg_half() {
    // acos(-0.5) = 2π/3
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0 * std::f64::consts::FRAC_PI_3,
        1e-14,
        "acos(-0.5)",
    );
}

#[test]
fn oracle_acos_sqrt2_over_2() {
    // acos(√2/2) = π/4
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_1_SQRT_2]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_4,
        1e-14,
        "acos(√2/2)",
    );
}

#[test]
fn oracle_acos_neg_sqrt2_over_2() {
    // acos(-√2/2) = 3π/4
    let input = make_f64_tensor(&[], vec![-std::f64::consts::FRAC_1_SQRT_2]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0 * std::f64::consts::FRAC_PI_4,
        1e-14,
        "acos(-√2/2)",
    );
}

#[test]
fn oracle_acos_sqrt3_over_2() {
    // acos(√3/2) = π/6
    let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
    let input = make_f64_tensor(&[], vec![sqrt3_over_2]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_6,
        1e-14,
        "acos(√3/2)",
    );
}

// ======================== Out of Domain ========================

#[test]
fn oracle_acos_greater_than_one() {
    for x in [1.1, 2.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result).is_nan(),
            "acos({}) should be NaN",
            x
        );
    }
}

#[test]
fn oracle_acos_less_than_neg_one() {
    for x in [-1.1, -2.0, -10.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result).is_nan(),
            "acos({}) should be NaN",
            x
        );
    }
}

// ======================== Infinity ========================

#[test]
fn oracle_acos_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "acos(+inf) = NaN"
    );
}

#[test]
fn oracle_acos_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "acos(-inf) = NaN"
    );
}

// ======================== NaN ========================

#[test]
fn oracle_acos_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "acos(NaN) = NaN");
}

// ======================== Range: output in [0, π] ========================

#[test]
fn oracle_acos_range() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val >= 0.0, "acos({}) >= 0", x);
        assert!(val <= std::f64::consts::PI, "acos({}) <= π", x);
    }
}

// ======================== Monotonicity: acos is decreasing ========================

#[test]
fn oracle_acos_monotonic_decreasing() {
    let inputs = vec![-0.9, -0.5, 0.0, 0.5, 0.9];
    let input = make_f64_tensor(&[5], inputs);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] < vals[i - 1],
            "acos should be monotonically decreasing: {} should be < {}",
            vals[i],
            vals[i - 1]
        );
    }
}

// ======================== Symmetry: acos(-x) = π - acos(x) ========================

#[test]
fn oracle_acos_symmetry() {
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Acos, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Acos, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            std::f64::consts::PI - val_pos,
            1e-14,
            &format!("acos(-{}) = π - acos({})", x, x),
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_acos_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], std::f64::consts::PI, 1e-14, "acos(-1)");
    assert_close(vals[1], 2.0 * std::f64::consts::FRAC_PI_3, 1e-14, "acos(-0.5)");
    assert_close(vals[2], std::f64::consts::FRAC_PI_2, 1e-14, "acos(0)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_3, 1e-14, "acos(0.5)");
    assert_close(vals[4], 0.0, 1e-14, "acos(1)");
}

#[test]
fn oracle_acos_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, 2.0, f64::INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], std::f64::consts::FRAC_PI_2, 1e-14, "acos(0)");
    assert!(vals[1].is_nan(), "acos(2) = NaN");
    assert!(vals[2].is_nan(), "acos(inf) = NaN");
    assert!(vals[3].is_nan(), "acos(NaN) = NaN");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_acos_2d() {
    let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
    let input = make_f64_tensor(&[2, 3], vec![-1.0, -sqrt2_2, 0.0, sqrt2_2, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], std::f64::consts::PI, 1e-14, "acos(-1)");
    assert_close(vals[2], std::f64::consts::FRAC_PI_2, 1e-14, "acos(0)");
    assert_close(vals[5], 0.0, 1e-14, "acos(1)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_acos_3d() {
    let input = make_f64_tensor(
        &[2, 2, 2],
        vec![-1.0, -0.5, 0.0, 0.5, -0.9, -0.1, 0.1, 0.9],
    );
    let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], std::f64::consts::PI, 1e-14, "acos(-1)");
    assert_close(vals[2], std::f64::consts::FRAC_PI_2, 1e-14, "acos(0)");
}

// ======================== Identity: cos(acos(x)) = x ========================

#[test]
fn oracle_acos_cos_identity() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();
        let acos_val = extract_f64_scalar(&result);

        let input2 = make_f64_tensor(&[], vec![acos_val]);
        let result2 = eval_primitive(Primitive::Cos, &[input2], &no_params()).unwrap();
        let roundtrip = extract_f64_scalar(&result2);

        assert_close(roundtrip, x, 1e-14, &format!("cos(acos({})) = {}", x, x));
    }
}

// ======================== Relationship: acos(x) + asin(x) = π/2 ========================

#[test]
fn oracle_acos_asin_relationship() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let acos_result = eval_primitive(Primitive::Acos, &[input.clone()], &no_params()).unwrap();
        let asin_result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();

        let acos_val = extract_f64_scalar(&acos_result);
        let asin_val = extract_f64_scalar(&asin_result);

        assert_close(
            acos_val + asin_val,
            std::f64::consts::FRAC_PI_2,
            1e-14,
            &format!("acos({}) + asin({}) = π/2", x, x),
        );
    }
}
