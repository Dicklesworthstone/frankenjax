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

#[test]
fn oracle_acosh_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 5.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_acosh_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![0]);
            assert!(t.elements.is_empty());
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_acosh_vector() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 5.0, 10.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![4]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_close(vals[0], 0.0, 1e-14, "acosh(1)");
            assert_close(vals[1], 2.0_f64.acosh(), 1e-14, "acosh(2)");
            assert_close(vals[2], 5.0_f64.acosh(), 1e-14, "acosh(5)");
            assert_close(vals[3], 10.0_f64.acosh(), 1e-14, "acosh(10)");
        }
        _ => panic!("expected tensor"),
    }
}

// ======================== Additional Coverage ========================

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_acosh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.0, 1e-14, "acosh(1)");
}

#[test]
fn oracle_acosh_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_acosh_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "acosh(-inf) = NaN");
}

#[test]
fn oracle_acosh_near_one() {
    let input = make_f64_tensor(&[], vec![1.0 + 1e-10]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(
        val > 0.0 && val < 1e-4,
        "acosh(1+tiny) should be small positive"
    );
}

#[test]
fn oracle_acosh_4d() {
    let input = make_f64_tensor(
        &[2, 2, 2, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
        ],
    );
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_acosh_preserves_all_float_dtypes() {
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

    // acosh domain is x >= 1
    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "acosh {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex acosh: acosh(z) = log(z + sqrt(z² - 1))

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
fn oracle_acosh_complex128_real_ge_one() {
    // acosh(2+0i) = acosh(2)+0i
    let x = 2.0_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.acosh(), 0.0), 1e-10, "acosh(2+0i) = acosh(2)");
}

#[test]
fn oracle_acosh_complex128_one() {
    // acosh(1+0i) = 0+0i
    let input = make_complex128_tensor(&[], &[(1.0, 0.0)]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.0, 0.0), 1e-10, "acosh(1) = 0");
}

#[test]
fn oracle_acosh_complex128_pure_imaginary() {
    // acosh(i) = log(i + sqrt(-1-1)) = log(i + sqrt(-2))
    // acosh(i) = acosh(0+i) ≈ 0.881374... + 1.570796...i
    let input = make_complex128_tensor(&[], &[(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    // Approximate expected value
    let expected = (0.881_373_587_019_543, std::f64::consts::FRAC_PI_2);
    assert_complex_close(vec[0], expected, 1e-10, "acosh(i)");
}

#[test]
fn oracle_acosh_complex64_vector() {
    let data: &[(f32, f32)] = &[(1.0, 0.0), (2.0, 0.0), (0.0, 1.0)];
    let input = make_complex64_tensor(&[3], data);
    let result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 3);

    // acosh(1) = 0
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "acosh(1)",
    );

    // acosh(2) = acosh(2)
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (2.0_f64.acosh(), 0.0),
        1e-4,
        "acosh(2)",
    );
}

#[test]
fn oracle_acosh_complex_cosh_inverse_identity() {
    // cosh(acosh(z)) = z for z with real part >= 1
    let values: &[(f64, f64)] = &[(2.0, 0.5), (1.5, 0.0), (3.0, 1.0)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);
        let acosh_result = eval_primitive(Primitive::Acosh, &[input], &no_params()).unwrap();
        let cosh_acosh = eval_primitive(Primitive::Cosh, &[acosh_result], &no_params()).unwrap();

        let result = extract_complex128_vec(&cosh_acosh)[0];
        assert_complex_close(
            result,
            (a, b),
            1e-9,
            &format!("cosh(acosh({a}+{b}i)) = {a}+{b}i"),
        );
    }
}

#[test]
fn oracle_acosh_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(2.0, 0.5), (1.5, 0.0)]);
    let c64_result = eval_primitive(Primitive::Acosh, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "acosh should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(2.0, 0.5), (1.5, 0.0)]);
    let c128_result = eval_primitive(Primitive::Acosh, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "acosh should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
