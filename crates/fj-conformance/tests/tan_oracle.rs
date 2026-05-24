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
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "tan(0) = +0");
}

#[test]
fn oracle_tan_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "tan(-0.0) = -0");
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

        let tan_result =
            eval_primitive(Primitive::Tan, std::slice::from_ref(&input), &no_params()).unwrap();
        let sin_result =
            eval_primitive(Primitive::Sin, std::slice::from_ref(&input), &no_params()).unwrap();
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

    let tan_result =
        eval_primitive(Primitive::Tan, std::slice::from_ref(&x), &no_params()).unwrap();
    let sin_result =
        eval_primitive(Primitive::Sin, std::slice::from_ref(&x), &no_params()).unwrap();
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

// ====================== ADDITIONAL COVERAGE ======================

#[test]
fn oracle_tan_3d_shape() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0);
    assert_close(vals[7], 0.7_f64.tan(), 1e-14, "tan(0.7)");
}

#[test]
fn oracle_tan_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_tan_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_tan_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_tan_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "tan(-inf) = NaN");
}

#[test]
fn oracle_tan_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[], vec![subnormal]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    // For small x, tan(x) ≈ x
    assert!(
        (val - subnormal).abs() < subnormal * 0.01,
        "tan(subnormal) ≈ subnormal"
    );
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_tan_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let values = [0.0_f64, 0.5, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "tan {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex tan: tan(a + bi) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))

fn make_complex64_tensor(shape: &[u32], data: &[(f32, f32)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: &[(f64, f64)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex64Bits(re, im) => (f32::from_bits(*re), f32::from_bits(*im)),
                _ => panic!("expected Complex64"),
            })
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let (ar, ai) = actual;
    let (er, ei) = expected;
    let re_diff = (ar - er).abs();
    let im_diff = (ai - ei).abs();
    assert!(
        re_diff < tol && im_diff < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        er,
        ei,
        ar,
        ai,
        re_diff,
        im_diff
    );
}

#[test]
fn oracle_tan_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-6,
        "tan(0+0i)",
    );
}

#[test]
fn oracle_tan_complex128_pure_imaginary() {
    // tan(i*x) = i*tanh(x)
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(0.0, x)]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.0, x.tanh()), 1e-10, "tan(0+0.5i) = i*tanh(0.5)");
}

#[test]
fn oracle_tan_complex128_pure_real() {
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.tan(), 0.0), 1e-10, "tan(0.5+0i) = tan(0.5)");
}

#[test]
fn oracle_tan_complex128_general() {
    // tan(a+bi) = (sin(2a) + i*sinh(2b)) / (cos(2a) + cosh(2b))
    let (a, b) = (0.3_f64, 0.2_f64);
    let denom = (2.0 * a).cos() + (2.0 * b).cosh();
    let expected_re = (2.0 * a).sin() / denom;
    let expected_im = (2.0 * b).sinh() / denom;

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (expected_re, expected_im), 1e-10, "tan(0.3+0.2i)");
}

#[test]
fn oracle_tan_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.3, 0.2)];
    let input = make_complex64_tensor(&[4], data);
    let result = eval_primitive(Primitive::Tan, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 4);

    // tan(0) = 0
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "tan(0)",
    );

    // tan(0.5+0i) = tan(0.5)
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (0.5_f64.tan(), 0.0),
        1e-4,
        "tan(0.5+0i)",
    );

    // tan(0.5i) = i*tanh(0.5)
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.0, 0.5_f64.tanh()),
        1e-4,
        "tan(0.5i)",
    );

    // tan(0.3+0.2i) using the formula
    let (a, b) = (0.3_f64, 0.2_f64);
    let denom = (2.0 * a).cos() + (2.0 * b).cosh();
    let expected = ((2.0 * a).sin() / denom, (2.0 * b).sinh() / denom);
    assert_complex_close(
        (vec[3].0 as f64, vec[3].1 as f64),
        expected,
        1e-4,
        "tan(0.3+0.2i)",
    );
}

#[test]
fn oracle_tan_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(0.3, 0.2), (-0.2, 0.3)]);
    let c64_result = eval_primitive(Primitive::Tan, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "tan should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(0.3, 0.2), (-0.2, 0.3)]);
    let c128_result = eval_primitive(Primitive::Tan, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex128, "tan should preserve Complex128");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
