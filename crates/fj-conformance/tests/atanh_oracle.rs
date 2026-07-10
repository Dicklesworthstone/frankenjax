//! Oracle tests for Atanh (inverse hyperbolic tangent) primitive.
//!
//! atanh(x) = 0.5 * ln((1 + x) / (1 - x))
//!
//! Properties:
//! - atanh(0) = 0
//! - Domain: |x| < 1 (real-valued), ±inf at ±1, NaN outside
//! - atanh is odd: atanh(-x) = -atanh(x)
//! - Metamorphic: tanh(atanh(x)) = x for |x| < 1
//! - Metamorphic: atanh(tanh(x)) = x

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
fn oracle_atanh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "atanh(0) = +0");
}

#[test]
fn oracle_atanh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "atanh(-0.0) = -0");
}

// ======================== Basic Values (|x| < 1) ========================

#[test]
fn oracle_atanh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5_f64.atanh(),
        1e-14,
        "atanh(0.5)",
    );
}

#[test]
fn oracle_atanh_neg_half() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-0.5_f64).atanh(),
        1e-14,
        "atanh(-0.5)",
    );
}

// ======================== Odd Function: atanh(-x) = -atanh(x) ========================

#[test]
fn oracle_atanh_odd_function() {
    for x in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Atanh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Atanh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("atanh(-{}) = -atanh({})", x, x),
        );
    }
}

// ======================== Domain boundary: atanh(±1) = ±inf ========================

#[test]
fn oracle_atanh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "atanh(1) = +inf");
}

#[test]
fn oracle_atanh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "atanh(-1) = -inf");
}

// ======================== Outside domain: |x| > 1 returns NaN ========================

#[test]
fn oracle_atanh_above_domain() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "atanh(1.5) = NaN (outside domain)"
    );
}

#[test]
fn oracle_atanh_below_domain() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "atanh(-1.5) = NaN (outside domain)"
    );
}

// ======================== METAMORPHIC: tanh(atanh(x)) = x for |x| < 1 ========================

#[test]
fn metamorphic_tanh_atanh_identity() {
    for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let atanh_result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let tanh_atanh = eval_primitive(Primitive::Tanh, &[atanh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&tanh_atanh),
            x,
            1e-12,
            &format!("tanh(atanh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: atanh(tanh(x)) = x ========================

#[test]
fn metamorphic_atanh_tanh_identity() {
    for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
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

// ======================== Near boundary ========================

#[test]
fn oracle_atanh_near_one() {
    let input = make_f64_tensor(&[], vec![0.999]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.999_f64.atanh(),
        1e-12,
        "atanh(0.999)",
    );
}

#[test]
fn oracle_atanh_near_neg_one() {
    let input = make_f64_tensor(&[], vec![-0.999]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-0.999_f64).atanh(),
        1e-12,
        "atanh(-0.999)",
    );
}

// ======================== NaN ========================

#[test]
fn oracle_atanh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atanh(NaN) = NaN");
}

// ======================== Stdlib comparison ========================

#[test]
fn oracle_atanh_stdlib() {
    for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x.atanh(),
            1e-14,
            &format!("atanh({}) vs stdlib", x),
        );
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 0.5, -0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 0.5, -0.5, 0.9]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![4]);
            let vals = extract_f64_vec(&result);
            assert_close(vals[0], 0.0, 1e-14, "atanh(0)");
            assert_close(vals[1], 0.5_f64.atanh(), 1e-14, "atanh(0.5)");
            assert_close(vals[2], (-0.5_f64).atanh(), 1e-14, "atanh(-0.5)");
            assert_close(vals[3], 0.9_f64.atanh(), 1e-14, "atanh(0.9)");
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![0]);
            assert!(t.elements.is_empty());
        }
        _ => panic!("expected tensor"),
    }
}

// ======================== Additional Coverage ========================

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.5, -0.5, 0.9, -0.9, 0.1, -0.1, 0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

#[test]
fn oracle_atanh_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_atanh_very_small() {
    let input = make_f64_tensor(&[], vec![1e-15]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-15,
        1e-20,
        "atanh(tiny) ≈ tiny for |x| << 1",
    );
}

#[test]
fn oracle_atanh_subnormal() {
    let tiny = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![tiny, -tiny]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // For very small x, atanh(x) ≈ x
    assert_close(vals[0], tiny.atanh(), 1e-30, "atanh(subnormal)");
    assert_close(vals[1], (-tiny).atanh(), 1e-30, "atanh(-subnormal)");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_atanh_preserves_all_float_dtypes() {
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

    // atanh domain is |x| < 1
    let values = [-0.5_f64, 0.0, 0.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "atanh {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex atanh: atanh(z) = (1/2) * log((1+z)/(1-z))

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
fn oracle_atanh_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-6,
        "atanh(0+0i)",
    );
}

#[test]
fn oracle_atanh_complex128_pure_imaginary() {
    // atanh(i) = i * atan(1) = i * pi/4
    let input = make_complex128_tensor(&[], &[(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        vec[0],
        (0.0, std::f64::consts::FRAC_PI_4),
        1e-10,
        "atanh(i) = i*pi/4",
    );
}

#[test]
fn oracle_atanh_complex128_real_small() {
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        vec[0],
        (x.atanh(), 0.0),
        1e-10,
        "atanh(0.5+0i) = atanh(0.5)",
    );
}

#[test]
fn oracle_atanh_complex128_large_no_overflow() {
    // Robustness: complex atanh of a large magnitude must stay finite. atanh uses the
    // log-form 0.5*log((1+z)/(1-z)) (no z²), so it doesn't overflow. As z -> +inf along
    // the real axis, (1+z)/(1-z) -> -1, so atanh -> 0.5*log(-1) = i*pi/2: real part ~0,
    // |imag| ~ pi/2. (Sign of imag is branch-dependent, so assert its magnitude.)
    let input = make_complex128_tensor(&[], &[(1e200, 0.0)]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert!(
        vec[0].0.is_finite() && vec[0].1.is_finite(),
        "atanh(1e200) must be finite, got {:?}",
        vec[0]
    );
    assert!(
        vec[0].0.abs() < 1e-9,
        "atanh(1e200) real part ~ 0, got {}",
        vec[0].0
    );
    assert!(
        (vec[0].1.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "|imag(atanh(1e200))| ~ pi/2, got {}",
        vec[0].1
    );
}

#[test]
fn oracle_atanh_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (0.5, 0.0), (0.0, 1.0)];
    let input = make_complex64_tensor(&[3], data);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 3);

    // atanh(0) = 0
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "atanh(0)",
    );

    // atanh(0.5) = atanh(0.5)
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (0.5_f64.atanh(), 0.0),
        1e-4,
        "atanh(0.5)",
    );

    // atanh(i) = i*pi/4
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.0, std::f64::consts::FRAC_PI_4),
        1e-4,
        "atanh(i)",
    );
}

#[test]
fn oracle_atanh_complex_tanh_inverse_identity() {
    // tanh(atanh(z)) = z for |z| < 1
    let values: &[(f64, f64)] = &[(0.3, 0.2), (0.5, 0.0), (0.0, 0.5)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);
        let atanh_result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let tanh_atanh = eval_primitive(Primitive::Tanh, &[atanh_result], &no_params()).unwrap();

        let result = extract_complex128_vec(&tanh_atanh)[0];
        assert_complex_close(
            result,
            (a, b),
            1e-9,
            &format!("tanh(atanh({a}+{b}i)) = {a}+{b}i"),
        );
    }
}

#[test]
fn oracle_atanh_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(0.3, 0.2), (0.5, 0.0)]);
    let c64_result = eval_primitive(Primitive::Atanh, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "atanh should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(0.3, 0.2), (0.5, 0.0)]);
    let c128_result = eval_primitive(Primitive::Atanh, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "atanh should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
