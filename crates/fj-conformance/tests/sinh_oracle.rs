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
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "sinh(0) = +0");
}

#[test]
fn oracle_sinh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "sinh(-0.0) = -0");
}

// ======================== Positive Values ========================

#[test]
fn oracle_sinh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0_f64.sinh(),
        1e-14,
        "sinh(1)",
    );
}

#[test]
fn oracle_sinh_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.sinh(),
        1e-14,
        "sinh(2)",
    );
}

#[test]
fn oracle_sinh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5_f64.sinh(),
        1e-14,
        "sinh(0.5)",
    );
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

// ======================== METAMORPHIC: asinh(sinh(x)) = x ========================

#[test]
fn metamorphic_asinh_sinh_identity() {
    // asinh(sinh(x)) = x for all x
    for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
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

// ======================== METAMORPHIC: sinh(asinh(x)) = x ========================

#[test]
fn metamorphic_sinh_asinh_identity() {
    // sinh(asinh(x)) = x for all real x
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

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_sinh_tensor_roundtrip() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let sinh_result =
        eval_primitive(Primitive::Sinh, std::slice::from_ref(&input), &no_params()).unwrap();
    let asinh_sinh = eval_primitive(Primitive::Asinh, &[sinh_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&asinh_sinh);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(
            *rt,
            *orig,
            1e-12,
            &format!("asinh(sinh({})) = {}", orig, orig),
        );
    }
}

// ======================== Property: dtype preservation across all float types ========================

#[test]
fn property_sinh_preserves_all_float_dtypes() {
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
        let result = eval_primitive(Primitive::Sinh, &[input], &no_params())
            .unwrap_or_else(|e| panic!("sinh {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("sinh {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "sinh {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("sinh {dtype:?}: validate_dtype_consistency failed: {e}"));
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex sinh: sinh(a + bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)

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
fn oracle_sinh_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-6,
        "sinh(0+0i)",
    );
}

#[test]
fn oracle_sinh_complex128_pure_imaginary() {
    // sinh(i*x) = i*sin(x)
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(0.0, x)]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.0, x.sin()), 1e-10, "sinh(0+0.5i) = i*sin(0.5)");
}

#[test]
fn oracle_sinh_complex128_pure_real() {
    // sinh(x+0i) = sinh(x)+0i
    let x = 1.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.sinh(), 0.0), 1e-10, "sinh(1.5+0i) = sinh(1.5)");
}

#[test]
fn oracle_sinh_complex128_general() {
    // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected_re = a.sinh() * b.cos();
    let expected_im = a.cosh() * b.sin();

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (expected_re, expected_im), 1e-10, "sinh(0.5+0.3i)");
}

#[test]
fn oracle_sinh_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.3)];
    let input = make_complex64_tensor(&[4], data);
    let result = eval_primitive(Primitive::Sinh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 4);

    // sinh(0+0i) = 0+0i
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "sinh(0)",
    );

    // sinh(1+0i) = sinh(1)+0i ≈ 1.1752
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (1.0_f64.sinh(), 0.0),
        1e-4,
        "sinh(1+0i)",
    );

    // sinh(0+i) = i*sin(1) ≈ 0.8415i
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.0, 1.0_f64.sin()),
        1e-4,
        "sinh(i)",
    );

    // sinh(0.5+0.3i) using the formula
    let (a, b) = (0.5_f64, 0.3_f64);
    let expected = (a.sinh() * b.cos(), a.cosh() * b.sin());
    assert_complex_close(
        (vec[3].0 as f64, vec[3].1 as f64),
        expected,
        1e-4,
        "sinh(0.5+0.3i)",
    );
}

#[test]
fn oracle_sinh_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(1.0, 0.5), (-0.5, 1.0)]);
    let c64_result = eval_primitive(Primitive::Sinh, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "sinh should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(1.0, 0.5), (-0.5, 1.0)]);
    let c128_result = eval_primitive(Primitive::Sinh, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "sinh should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
