//! Oracle tests for Acosh (inverse hyperbolic cosine) primitive.
//!
//! acosh(x) = ln(x + sqrt(x² - 1))
//!
//! Properties:
//! - acosh(1) = 0
//! - Domain: x >= 1 (real-valued), returns NaN for x < 1 on reals
//! - Metamorphic: cosh(acosh(x)) = x for x >= 1
//! - Metamorphic: acosh(cosh(x)) = |x| for x >= 0

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

// ======================== Boundary: acosh(1) = 0 ========================

#[test]
fn oracle_acosh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "acosh(1) = 0");
}

// ======================== Basic Values (x > 1) ========================

#[test]
fn oracle_acosh_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.acosh(),
        1e-14,
        "acosh(2)",
    );
}

#[test]
fn oracle_acosh_ten() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        10.0_f64.acosh(),
        1e-14,
        "acosh(10)",
    );
}

// ======================== Domain: x < 1 returns NaN ========================

#[test]
fn oracle_acosh_below_domain() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "acosh(0.5) = NaN (below domain)"
    );
}

#[test]
fn oracle_acosh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "acosh(0) = NaN (below domain)"
    );
}

#[test]
fn oracle_acosh_negative() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "acosh(-1) = NaN (below domain)"
    );
}

// ======================== METAMORPHIC: cosh(acosh(x)) = x for x >= 1 ========================

#[test]
fn metamorphic_cosh_acosh_identity() {
    for x in [1.0, 1.5, 2.0, 5.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let acosh_result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
        let cosh_acosh = eval_primitive(Primitive::Cosh, &[acosh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&cosh_acosh),
            x,
            1e-12,
            &format!("cosh(acosh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: acosh(cosh(x)) = |x| for x >= 0 ========================

#[test]
fn metamorphic_acosh_cosh_identity() {
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let cosh_result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
        let acosh_cosh = eval_primitive(Primitive::Acosh, &[cosh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&acosh_cosh),
            x.abs(),
            1e-12,
            &format!("acosh(cosh({})) = |{}|", x, x),
        );
    }
}

// ======================== Large Values ========================

#[test]
fn oracle_acosh_large() {
    let input = make_f64_tensor(&[], vec![1e10]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e10_f64.acosh(),
        1e-5,
        "acosh(1e10)",
    );
}

// ======================== Infinity ========================

#[test]
fn oracle_acosh_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_infinite() && extract_f64_scalar(&result) > 0.0,
        "acosh(+inf) = +inf"
    );
}

// ======================== NaN ========================

#[test]
fn oracle_acosh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "acosh(NaN) = NaN");
}

// ======================== Stdlib comparison ========================

#[test]
fn oracle_acosh_stdlib() {
    for x in [1.0, 1.1, 2.0, 5.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x.acosh(),
            1e-14,
            &format!("acosh({}) vs stdlib", x),
        );
    }
}

// ======================== Result is always non-negative ========================

#[test]
fn oracle_acosh_non_negative() {
    for x in [1.0, 1.001, 2.0, 10.0, 1000.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            val >= 0.0,
            "acosh({}) should be non-negative, got {}",
            x,
            val
        );
    }
}
