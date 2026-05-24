//! Oracle tests for Sign primitive.
//!
//! sign(x) returns:
//! - -1 if x < 0
//! - 0 if x == 0
//! - +1 if x > 0
//! - NaN if x is NaN
//!
//! For integers, returns -1, 0, or 1.
//! For floats, returns -1.0, 0.0, or 1.0 (preserving -0.0 sign).

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

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

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
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

// ======================== Integer Positive ========================

#[test]
fn oracle_sign_i64_positive_one() {
    let input = make_i64_tensor(&[], vec![1]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_sign_i64_positive_large() {
    let input = make_i64_tensor(&[], vec![1_000_000]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_sign_i64_max() {
    let input = make_i64_tensor(&[], vec![i64::MAX]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

// ======================== Integer Negative ========================

#[test]
fn oracle_sign_i64_negative_one() {
    let input = make_i64_tensor(&[], vec![-1]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

#[test]
fn oracle_sign_i64_negative_large() {
    let input = make_i64_tensor(&[], vec![-1_000_000]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

#[test]
fn oracle_sign_i64_min() {
    let input = make_i64_tensor(&[], vec![i64::MIN]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

// ======================== Integer Zero ========================

#[test]
fn oracle_sign_i64_zero() {
    let input = make_i64_tensor(&[], vec![0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Float Positive ========================

#[test]
fn oracle_sign_f64_positive_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_fraction() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_small() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_large() {
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_sign_f64_positive_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

// ======================== Float Negative ========================

#[test]
fn oracle_sign_f64_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_fraction() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_small() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_large() {
    let input = make_f64_tensor(&[], vec![-1e100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_sign_f64_negative_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

// ======================== Float Zero ========================

#[test]
fn oracle_sign_f64_positive_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_sign_f64_negative_zero() {
    // sign(-0.0) should preserve the sign: -0.0
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_negative(), "sign(-0.0) should be -0.0");
}

// ======================== Float NaN ========================

#[test]
fn oracle_sign_f64_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "sign(NaN) should be NaN"
    );
}

// ======================== 1D Tensor Integer ========================

#[test]
fn oracle_sign_i64_1d() {
    let input = make_i64_tensor(&[5], vec![-5, -1, 0, 1, 100]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, 0, 1, 1]);
}

#[test]
fn oracle_sign_i64_all_positive() {
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1]);
}

#[test]
fn oracle_sign_i64_all_negative() {
    let input = make_i64_tensor(&[4], vec![-1, -2, -3, -4]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, -1]);
}

#[test]
fn oracle_sign_i64_all_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

// ======================== 1D Tensor Float ========================

#[test]
fn oracle_sign_f64_1d() {
    let input = make_f64_tensor(&[5], vec![-3.5, -0.1, 0.0, 0.1, 7.0]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
}

#[test]
fn oracle_sign_f64_1d_mixed_special() {
    let input = make_f64_tensor(
        &[6],
        vec![-f64::INFINITY, -1.0, 0.0, 1.0, f64::INFINITY, f64::NAN],
    );
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
    assert!(vals[5].is_nan());
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_sign_i64_2d() {
    let input = make_i64_tensor(&[2, 3], vec![-3, -2, -1, 0, 1, 2]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, 0, 1, 1]);
}

#[test]
fn oracle_sign_f64_2d() {
    let input = make_f64_tensor(&[2, 2], vec![-1.5, 0.0, 0.0, 2.5]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert_eq!(vals[3], 1.0);
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_sign_i64_3d() {
    let input = make_i64_tensor(&[2, 2, 2], vec![-4, -3, -2, -1, 0, 1, 2, 3]);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![-1, -1, -1, -1, 0, 1, 1, 1]);
}

// ======================== Identity: sign(x) * |x| = x ========================

#[test]
fn oracle_sign_identity_positive() {
    // For positive x: sign(x) = 1, so sign(x) * |x| = x
    for x in [1.0, 2.5, 100.0, 1e50] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let sign_val = extract_f64_scalar(&result);
        assert_eq!(sign_val * x.abs(), x, "sign({}) * |{}| = {}", x, x, x);
    }
}

#[test]
fn oracle_sign_identity_negative() {
    // For negative x: sign(x) = -1, so sign(x) * |x| = x
    for x in [-1.0, -2.5, -100.0, -1e50] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let sign_val = extract_f64_scalar(&result);
        assert_eq!(sign_val * x.abs(), x, "sign({}) * |{}| = {}", x, x, x);
    }
}

// ======================== Idempotence: sign(sign(x)) = sign(x) ========================

#[test]
fn oracle_sign_idempotent() {
    // sign(sign(x)) = sign(x) for non-zero x
    for x in [-5.0, -1.0, 1.0, 5.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Sign, &[input1], &no_params()).unwrap();
        let sign1 = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![sign1]);
        let result2 = eval_primitive(Primitive::Sign, &[input2], &no_params()).unwrap();
        let sign2 = extract_f64_scalar(&result2);

        assert_eq!(sign1, sign2, "sign(sign({})) should equal sign({})", x, x);
    }
}

// ======================== METAMORPHIC: sign(Neg(x)) = Neg(sign(x)) ========================

#[test]
fn metamorphic_sign_negation() {
    // sign(-x) = -sign(x) for x != 0
    for x in [-5.5, -1.0, 1.0, 5.5, f64::INFINITY, f64::NEG_INFINITY] {
        let input = make_f64_tensor(&[], vec![x]);

        // sign(Neg(x))
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();
        let sign_neg_x = eval_primitive(Primitive::Sign, &[neg_x], &no_params()).unwrap();

        // Neg(sign(x))
        let sign_x = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let neg_sign_x = eval_primitive(Primitive::Neg, &[sign_x], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&sign_neg_x),
            extract_f64_scalar(&neg_sign_x),
            "sign(Neg({})) = Neg(sign({}))",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: Mul(sign(x), abs(x)) = x ========================

#[test]
fn metamorphic_sign_abs_reconstruction() {
    // x = sign(x) * abs(x) for finite non-zero x
    for x in [-5.5, -1.0, 1.0, 5.5, 100.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let sign_x =
            eval_primitive(Primitive::Sign, std::slice::from_ref(&input), &no_params()).unwrap();
        let abs_x = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
        let reconstructed = eval_primitive(Primitive::Mul, &[sign_x, abs_x], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&reconstructed),
            x,
            "Mul(sign({}), abs({})) = {}",
            x,
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: tensor sign reconstruction ========================

#[test]
fn metamorphic_sign_abs_tensor_reconstruction() {
    // For tensor: Mul(sign(x), abs(x)) = x
    let data = vec![-3.0, -1.0, 1.0, 3.0, -2.5, 2.5];
    let input = make_f64_tensor(&[6], data.clone());

    let sign_x =
        eval_primitive(Primitive::Sign, std::slice::from_ref(&input), &no_params()).unwrap();
    let abs_x = eval_primitive(Primitive::Abs, &[input], &no_params()).unwrap();
    let reconstructed = eval_primitive(Primitive::Mul, &[sign_x, abs_x], &no_params()).unwrap();

    assert_eq!(extract_shape(&reconstructed), vec![6]);
    let result = extract_f64_vec(&reconstructed);
    for (i, (&orig, &rec)) in data.iter().zip(result.iter()).enumerate() {
        assert_eq!(
            rec, orig,
            "element {}: Mul(sign, abs) should reconstruct",
            i
        );
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_sign_preserves_all_float_dtypes() {
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

    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "sign {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================

fn make_complex64_scalar(re: f32, im: f32) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![] },
            vec![Literal::from_complex64(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex128_scalar(re: f64, im: f64) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![] },
            vec![Literal::from_complex128(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex64_tensor(shape: &[u32], pairs: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

fn assert_complex128_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

#[test]
fn oracle_sign_complex64_zero() {
    // sign(0+0i) = 0+0i
    let input = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 0.0), 1e-6, "sign(0)");
}

#[test]
fn oracle_sign_complex64_positive_real() {
    // sign(3+0i) = 1+0i (unit vector in positive real direction)
    let input = make_complex64_scalar(3.0, 0.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (1.0, 0.0), 1e-6, "sign(3) = 1");
}

#[test]
fn oracle_sign_complex64_negative_real() {
    // sign(-3+0i) = -1+0i
    let input = make_complex64_scalar(-3.0, 0.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (-1.0, 0.0), 1e-6, "sign(-3) = -1");
}

#[test]
fn oracle_sign_complex64_positive_imaginary() {
    // sign(0+2i) = 0+i (unit vector in positive imaginary direction)
    let input = make_complex64_scalar(0.0, 2.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 1.0), 1e-6, "sign(2i) = i");
}

#[test]
fn oracle_sign_complex64_negative_imaginary() {
    // sign(0-2i) = 0-i
    let input = make_complex64_scalar(0.0, -2.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, -1.0), 1e-6, "sign(-2i) = -i");
}

#[test]
fn oracle_sign_complex64_unit_circle() {
    // sign(z) = z/|z| gives a point on the unit circle
    // sign(3+4i): |3+4i| = 5, so sign = (3/5, 4/5) = (0.6, 0.8)
    let input = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.6, 0.8), 1e-5, "sign(3+4i) = (0.6, 0.8)");

    // Verify it's on the unit circle: |sign(z)| = 1
    let magnitude = (re * re + im * im).sqrt();
    assert!(
        (magnitude - 1.0).abs() < 1e-5,
        "sign result should have magnitude 1"
    );
}

#[test]
fn oracle_sign_complex64_diagonal() {
    // sign(1+i): |1+i| = sqrt(2), so sign = (1/sqrt(2), 1/sqrt(2))
    let sqrt2_inv = 1.0 / 2.0_f32.sqrt();
    let input = make_complex64_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (sqrt2_inv, sqrt2_inv), 1e-5, "sign(1+i)");
}

#[test]
fn oracle_sign_complex64_vector() {
    let input = make_complex64_tensor(
        &[4],
        vec![
            (0.0, 0.0), // zero
            (5.0, 0.0), // positive real
            (0.0, 3.0), // positive imaginary
            (3.0, 4.0), // 3-4-5 triangle
        ],
    );
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    assert_complex64_close(vals[0], (0.0, 0.0), 1e-5, "sign(0)");
    assert_complex64_close(vals[1], (1.0, 0.0), 1e-5, "sign(5)");
    assert_complex64_close(vals[2], (0.0, 1.0), 1e-5, "sign(3i)");
    assert_complex64_close(vals[3], (0.6, 0.8), 1e-5, "sign(3+4i)");
}

#[test]
fn oracle_sign_complex64_idempotent() {
    // sign(sign(z)) = sign(z) for z != 0 (since |sign(z)| = 1)
    let z = make_complex64_scalar(3.0, 4.0);
    let sign_z = eval_primitive(Primitive::Sign, &[z], &no_params()).unwrap();
    let sign_sign_z = eval_primitive(Primitive::Sign, &[sign_z.clone()], &no_params()).unwrap();

    let (s1_re, s1_im) = extract_complex64_scalar(&sign_z);
    let (s2_re, s2_im) = extract_complex64_scalar(&sign_sign_z);

    assert_complex64_close(
        (s2_re, s2_im),
        (s1_re, s1_im),
        1e-5,
        "sign(sign(z)) = sign(z)",
    );
}

#[test]
fn oracle_sign_complex128_unit_circle() {
    // sign(3+4i) = (0.6, 0.8) with higher precision
    let input = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (0.6, 0.8), 1e-12, "sign(3+4i) Complex128");
}

#[test]
fn oracle_sign_complex64_preserves_dtype() {
    let input = make_complex64_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_sign_complex128_preserves_dtype() {
    let input = make_complex128_scalar(3.0, 4.0);
    let result = eval_primitive(Primitive::Sign, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
