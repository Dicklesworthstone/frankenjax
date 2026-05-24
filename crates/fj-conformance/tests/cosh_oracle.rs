//! Oracle tests for Cosh (hyperbolic cosine) primitive.
//!
//! cosh(x) = (e^x + e^-x) / 2
//!
//! Properties:
//! - cosh(0) = 1
//! - cosh(x) >= 1 for all real x
//! - cosh is even: cosh(-x) = cosh(x)
//! - cosh(x) → +∞ as x → ±∞
//!
//! Tests:
//! - Zero: cosh(0) = 1
//! - Positive/negative values
//! - Even function property
//! - Infinity: cosh(±inf) = +inf
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
fn oracle_cosh_zero() {
    // cosh(0) = 1
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "cosh(0)");
}

#[test]
fn oracle_cosh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "cosh(-0.0)");
}

// ======================== Positive Values ========================

#[test]
fn oracle_cosh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0_f64.cosh(),
        1e-14,
        "cosh(1)",
    );
}

#[test]
fn oracle_cosh_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.cosh(),
        1e-14,
        "cosh(2)",
    );
}

#[test]
fn oracle_cosh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5_f64.cosh(),
        1e-14,
        "cosh(0.5)",
    );
}

// ======================== Negative Values ========================

#[test]
fn oracle_cosh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-1.0_f64).cosh(),
        1e-14,
        "cosh(-1)",
    );
}

#[test]
fn oracle_cosh_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-2.0_f64).cosh(),
        1e-14,
        "cosh(-2)",
    );
}

// ======================== Large Values ========================

#[test]
fn oracle_cosh_large_positive() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        10.0_f64.cosh(),
        1e-10,
        "cosh(10)",
    );
}

#[test]
fn oracle_cosh_large_negative() {
    let input = make_f64_tensor(&[], vec![-10.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-10.0_f64).cosh(),
        1e-10,
        "cosh(-10)",
    );
}

// ======================== Infinity ========================

#[test]
fn oracle_cosh_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "cosh(+inf) = +inf");
}

#[test]
fn oracle_cosh_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "cosh(-inf) = +inf");
}

// ======================== NaN ========================

#[test]
fn oracle_cosh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "cosh(NaN) = NaN");
}

// ======================== Bounds: cosh(x) >= 1 for all finite x ========================

#[test]
fn oracle_cosh_bounds() {
    for x in [-10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val >= 1.0, "cosh({}) >= 1", x);
    }
}

// ======================== Even Function: cosh(-x) = cosh(x) ========================

#[test]
fn oracle_cosh_even_function() {
    for x in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Cosh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Cosh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_pos,
            val_neg,
            1e-14,
            &format!("cosh({}) = cosh(-{})", x, x),
        );
    }
}

// ======================== Identity: cosh^2(x) - sinh^2(x) = 1 ========================

#[test]
fn oracle_cosh_sinh_identity() {
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let cosh_result =
            eval_primitive(Primitive::Cosh, std::slice::from_ref(&input), &no_params()).unwrap();
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

// ======================== Small Values ========================

#[test]
fn oracle_cosh_small() {
    // For small x, cosh(x) ≈ 1 + x^2/2
    let x = 1e-10;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1.0, 1e-19, "cosh(1e-10) ≈ 1");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_cosh_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).cosh(), 1e-14, "cosh(-2)");
    assert_close(vals[1], (-1.0_f64).cosh(), 1e-14, "cosh(-1)");
    assert_close(vals[2], 1.0, 1e-14, "cosh(0)");
    assert_close(vals[3], 1.0_f64.cosh(), 1e-14, "cosh(1)");
    assert_close(vals[4], 2.0_f64.cosh(), 1e-14, "cosh(2)");
}

#[test]
fn oracle_cosh_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 1.0, 1e-14, "cosh(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "cosh(+inf)");
    assert!(vals[2].is_infinite() && vals[2] > 0.0, "cosh(-inf)");
    assert!(vals[3].is_nan(), "cosh(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_cosh_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).cosh(), 1e-14, "cosh(-2)");
    assert_close(vals[2], 1.0, 1e-14, "cosh(0)");
    assert_close(vals[5], 3.0_f64.cosh(), 1e-14, "cosh(3)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_cosh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-3.0_f64).cosh(), 1e-14, "cosh(-3)");
    assert_close(vals[3], 1.0, 1e-14, "cosh(0)");
    assert_close(vals[7], 4.0_f64.cosh(), 1e-14, "cosh(4)");
}

// ======================== Identity: computed vs. formula ========================

#[test]
fn oracle_cosh_identity_formula() {
    for x in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = (x.exp() + (-x).exp()) / 2.0;
        assert_close(
            val,
            expected,
            1e-14,
            &format!("cosh({}) = (e^{} + e^-{}) / 2", x, x, x),
        );
    }
}

// ======================== METAMORPHIC: cosh(acosh(x)) = x for x >= 1 ========================

#[test]
fn metamorphic_cosh_acosh_identity() {
    // cosh(acosh(x)) = x for x >= 1 (domain of acosh)
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

// ======================== METAMORPHIC: acosh(cosh(x)) = |x| ========================

#[test]
fn metamorphic_acosh_cosh_abs_identity() {
    // acosh(cosh(x)) = |x| since cosh is even (cosh(-x) = cosh(x))
    for x in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let cosh_result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
        let acosh_cosh = eval_primitive(Primitive::Acosh, &[cosh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&acosh_cosh),
            x.abs(),
            1e-12,
            &format!("acosh(cosh({})) = |{}| = {}", x, x, x.abs()),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_cosh_tensor_roundtrip() {
    // For positive values, acosh(cosh(x)) = x
    let input = make_f64_tensor(&[5], vec![0.0, 0.5, 1.0, 2.0, 3.0]);
    let cosh_result =
        eval_primitive(Primitive::Cosh, std::slice::from_ref(&input), &no_params()).unwrap();
    let acosh_cosh = eval_primitive(Primitive::Acosh, &[cosh_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&acosh_cosh);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(
            *rt,
            orig.abs(),
            1e-12,
            &format!("acosh(cosh({})) = {}", orig, orig.abs()),
        );
    }
}

// ======================== Property: dtype preservation across all float types ========================

#[test]
fn property_cosh_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![values.len() as u32],
                },
                values.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    // Use small values to avoid overflow in lower precision types
    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Cosh, &[input], &no_params())
            .unwrap_or_else(|e| panic!("cosh {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("cosh {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "cosh {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("cosh {dtype:?}: validate_dtype_consistency failed: {e}"));
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex cosh: cosh(a + bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)

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
fn oracle_cosh_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    // cosh(0) = 1
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (1.0, 0.0),
        1e-6,
        "cosh(0+0i)",
    );
}

#[test]
fn oracle_cosh_complex128_pure_imaginary() {
    // cosh(i*x) = cos(x)
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(0.0, x)]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.cos(), 0.0), 1e-10, "cosh(0+0.5i) = cos(0.5)");
}

#[test]
fn oracle_cosh_complex128_pure_real() {
    // cosh(x+0i) = cosh(x)+0i
    let x = 1.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.cosh(), 0.0), 1e-10, "cosh(1.5+0i) = cosh(1.5)");
}

#[test]
fn oracle_cosh_complex128_general() {
    // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected_re = a.cosh() * b.cos();
    let expected_im = a.sinh() * b.sin();

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (expected_re, expected_im), 1e-10, "cosh(0.5+0.3i)");
}

#[test]
fn oracle_cosh_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.3)];
    let input = make_complex64_tensor(&[4], data);
    let result = eval_primitive(Primitive::Cosh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 4);

    // cosh(0+0i) = 1+0i
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (1.0, 0.0),
        1e-5,
        "cosh(0)",
    );

    // cosh(1+0i) = cosh(1)+0i ≈ 1.5431
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (1.0_f64.cosh(), 0.0),
        1e-4,
        "cosh(1+0i)",
    );

    // cosh(0+i) = cos(1) ≈ 0.5403
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (1.0_f64.cos(), 0.0),
        1e-4,
        "cosh(i)",
    );

    // cosh(0.5+0.3i) using the formula
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected = (a.cosh() * b.cos(), a.sinh() * b.sin());
    assert_complex_close(
        (vec[3].0 as f64, vec[3].1 as f64),
        expected,
        1e-4,
        "cosh(0.5+0.3i)",
    );
}

#[test]
fn oracle_cosh_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(1.0, 0.5), (-0.5, 1.0)]);
    let c64_result = eval_primitive(Primitive::Cosh, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "cosh should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(1.0, 0.5), (-0.5, 1.0)]);
    let c128_result = eval_primitive(Primitive::Cosh, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "cosh should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
