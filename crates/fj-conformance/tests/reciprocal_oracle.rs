//! Oracle tests for Reciprocal primitive.
//!
//! reciprocal(x) = 1/x
//!
//! Tests:
//! - Basic: reciprocal(2) = 0.5, reciprocal(4) = 0.25
//! - Zero: reciprocal(0) = +infinity, reciprocal(-0) = -infinity
//! - Negative: reciprocal(-x) = -reciprocal(x)
//! - Infinity: reciprocal(inf) = exact signed zero
//! - NaN propagation
//! - Identity: reciprocal(reciprocal(x)) = x

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

// ======================== Basic Positive Values ========================

#[test]
fn oracle_reciprocal_one() {
    // reciprocal(1) = 1
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "reciprocal(1)");
}

#[test]
fn oracle_reciprocal_two() {
    // reciprocal(2) = 0.5
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "reciprocal(2)");
}

#[test]
fn oracle_reciprocal_four() {
    // reciprocal(4) = 0.25
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "reciprocal(4)");
}

#[test]
fn oracle_reciprocal_ten() {
    // reciprocal(10) = 0.1
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.1, 1e-14, "reciprocal(10)");
}

#[test]
fn oracle_reciprocal_hundred() {
    // reciprocal(100) = 0.01
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.01, 1e-14, "reciprocal(100)");
}

// ======================== Fractions (reciprocal > 1) ========================

#[test]
fn oracle_reciprocal_half() {
    // reciprocal(0.5) = 2
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "reciprocal(0.5)");
}

#[test]
fn oracle_reciprocal_quarter() {
    // reciprocal(0.25) = 4
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 4.0, 1e-14, "reciprocal(0.25)");
}

#[test]
fn oracle_reciprocal_tenth() {
    // reciprocal(0.1) = 10
    let input = make_f64_tensor(&[], vec![0.1]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 10.0, 1e-12, "reciprocal(0.1)");
}

// ======================== Negative Values ========================

#[test]
fn oracle_reciprocal_negative_one() {
    // reciprocal(-1) = -1
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "reciprocal(-1)");
}

#[test]
fn oracle_reciprocal_negative_two() {
    // reciprocal(-2) = -0.5
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -0.5, 1e-14, "reciprocal(-2)");
}

#[test]
fn oracle_reciprocal_negative_half() {
    // reciprocal(-0.5) = -2
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -2.0, 1e-14, "reciprocal(-0.5)");
}

// ======================== Zero (Infinity) ========================

#[test]
fn oracle_reciprocal_positive_zero() {
    // reciprocal(+0) = +infinity
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "reciprocal(+0) = +inf");
}

#[test]
fn oracle_reciprocal_negative_zero() {
    // reciprocal(-0) = -infinity
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "reciprocal(-0) = -inf");
}

// ======================== Infinity ========================

#[test]
fn oracle_reciprocal_positive_infinity() {
    // reciprocal(+inf) = +0
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val.to_bits(), 0.0_f64.to_bits(), "reciprocal(+inf) = +0");
}

#[test]
fn oracle_reciprocal_negative_infinity() {
    // reciprocal(-inf) = -0
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val.to_bits(), (-0.0_f64).to_bits(), "reciprocal(-inf) = -0");
}

// ======================== NaN ========================

#[test]
fn oracle_reciprocal_nan() {
    // reciprocal(NaN) = NaN
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "reciprocal(NaN) = NaN"
    );
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_reciprocal_very_small() {
    // reciprocal(1e-100) = 1e100
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e100,
        1e86,
        "reciprocal(1e-100)",
    );
}

#[test]
fn oracle_reciprocal_very_large() {
    // reciprocal(1e100) = 1e-100
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-100,
        1e-114,
        "reciprocal(1e100)",
    );
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_reciprocal_1d() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 4.0, 10.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[1], 0.5, 1e-14, "reciprocal(2)");
    assert_close(vals[2], 0.25, 1e-14, "reciprocal(4)");
    assert_close(vals[3], 0.1, 1e-14, "reciprocal(10)");
}

#[test]
fn oracle_reciprocal_1d_mixed() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.5, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], -0.5, 1e-14, "reciprocal(-2)");
    assert_close(vals[1], -1.0, 1e-14, "reciprocal(-1)");
    assert_close(vals[2], 2.0, 1e-14, "reciprocal(0.5)");
    assert_close(vals[3], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[4], 0.5, 1e-14, "reciprocal(2)");
}

#[test]
fn oracle_reciprocal_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
    assert!(vals[3].is_nan());
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_reciprocal_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 4.0, 5.0, 10.0, 20.0]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[1], 0.5, 1e-14, "reciprocal(2)");
    assert_close(vals[2], 0.25, 1e-14, "reciprocal(4)");
    assert_close(vals[3], 0.2, 1e-14, "reciprocal(5)");
    assert_close(vals[4], 0.1, 1e-14, "reciprocal(10)");
    assert_close(vals[5], 0.05, 1e-14, "reciprocal(20)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_reciprocal_3d() {
    let input = make_f64_tensor(
        &[2, 2, 2],
        vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    );
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "reciprocal(1)");
    assert_close(vals[7], 1.0 / 128.0, 1e-14, "reciprocal(128)");
}

// ======================== Identity: reciprocal(reciprocal(x)) = x ========================

#[test]
fn oracle_reciprocal_double_reciprocal() {
    // reciprocal(reciprocal(x)) = x for non-zero, non-inf values
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, 100.0, -1.0, -0.5] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Reciprocal, &[input1], &no_params()).unwrap();
        let recip1 = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![recip1]);
        let result2 = eval_primitive(Primitive::Reciprocal, &[input2], &no_params()).unwrap();
        let recip2 = extract_f64_scalar(&result2);

        assert_close(
            recip2,
            x,
            1e-12,
            &format!("reciprocal(reciprocal({})) = {}", x, x),
        );
    }
}

// ======================== Identity: x * reciprocal(x) = 1 ========================

#[test]
fn oracle_reciprocal_product_identity() {
    // x * reciprocal(x) = 1 for non-zero values
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, -1.0, -5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let recip = extract_f64_scalar(&result);
        let product = x * recip;
        assert_close(
            product,
            1.0,
            1e-14,
            &format!("{} * reciprocal({}) = 1", x, x),
        );
    }
}

// ======================== Symmetry: reciprocal(-x) = -reciprocal(x) ========================

#[test]
fn oracle_reciprocal_negative_symmetry() {
    for x in [1.0, 2.0, 0.5, 10.0, 100.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Reciprocal, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Reciprocal, &[input_neg], &no_params()).unwrap();

        let recip_pos = extract_f64_scalar(&result_pos);
        let recip_neg = extract_f64_scalar(&result_neg);

        assert_close(
            recip_neg,
            -recip_pos,
            1e-14,
            &format!("reciprocal(-{}) = -reciprocal({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: reciprocal(reciprocal(x)) = x ========================

#[test]
fn metamorphic_reciprocal_involution() {
    // reciprocal is an involution: reciprocal(reciprocal(x)) = x
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, -1.0, -5.0, 0.001, 1000.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let recip1 = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let recip2 = eval_primitive(Primitive::Reciprocal, &[recip1], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&recip2),
            x,
            1e-12,
            &format!("reciprocal(reciprocal({}))", x),
        );
    }
}

// ======================== METAMORPHIC: Mul(x, reciprocal(x)) = 1 ========================

#[test]
fn metamorphic_reciprocal_mul_identity() {
    // Mul(x, reciprocal(x)) = 1 using Mul primitive
    for x in [0.5, 1.0, 2.0, 3.0, 10.0, -1.0, -5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let recip = eval_primitive(
            Primitive::Reciprocal,
            std::slice::from_ref(&input),
            &no_params(),
        )
        .unwrap();
        let product = eval_primitive(Primitive::Mul, &[input, recip], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&product),
            1.0,
            1e-14,
            &format!("Mul({}, reciprocal({}))", x, x),
        );
    }
}

// ======================== METAMORPHIC: reciprocal(Neg(x)) = Neg(reciprocal(x)) ========================

#[test]
fn metamorphic_reciprocal_negation() {
    // reciprocal(-x) = -reciprocal(x) using Neg primitive
    for x in [1.0, 2.0, 0.5, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);

        // reciprocal(Neg(x))
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();
        let recip_neg = eval_primitive(Primitive::Reciprocal, &[neg_x], &no_params()).unwrap();

        // Neg(reciprocal(x))
        let recip_x = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let neg_recip = eval_primitive(Primitive::Neg, &[recip_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&recip_neg),
            extract_f64_scalar(&neg_recip),
            1e-14,
            &format!("reciprocal(Neg({})) = Neg(reciprocal({}))", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor reciprocal involution ========================

#[test]
fn metamorphic_reciprocal_tensor_involution() {
    // For tensor: reciprocal(reciprocal(x)) = x
    let data = vec![0.5, 1.0, 2.0, -0.5, -1.0, -2.0];
    let input = make_f64_tensor(&[6], data.clone());

    let recip1 = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let recip2 = eval_primitive(Primitive::Reciprocal, &[recip1], &no_params()).unwrap();

    assert_eq!(extract_shape(&recip2), vec![6]);
    let result = extract_f64_vec(&recip2);
    for (i, (&orig, &rec)) in data.iter().zip(result.iter()).enumerate() {
        assert_close(rec, orig, 1e-12, &format!("element {}", i));
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_reciprocal_preserves_all_float_dtypes() {
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

    // Avoid zero
    let values = [0.5_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "reciprocal {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex reciprocal: 1/(a + bi) = (a - bi)/(a² + b²)

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

fn complex_reciprocal(a: f64, b: f64) -> (f64, f64) {
    let denom = a * a + b * b;
    (a / denom, -b / denom)
}

#[test]
fn oracle_reciprocal_complex128_real() {
    // reciprocal(2+0i) = 0.5+0i
    let input = make_complex128_tensor(&[], &[(2.0, 0.0)]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.5, 0.0), 1e-10, "reciprocal(2+0i) = 0.5");
}

#[test]
fn oracle_reciprocal_complex128_pure_imaginary() {
    // reciprocal(0+2i) = 0-0.5i
    let input = make_complex128_tensor(&[], &[(0.0, 2.0)]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.0, -0.5), 1e-10, "reciprocal(2i) = -0.5i");
}

#[test]
fn oracle_reciprocal_complex128_unit_plus_i() {
    // reciprocal(1+i) = (1-i)/2 = 0.5-0.5i
    let input = make_complex128_tensor(&[], &[(1.0, 1.0)]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.5, -0.5), 1e-10, "reciprocal(1+i) = 0.5-0.5i");
}

#[test]
fn oracle_reciprocal_complex128_general() {
    // reciprocal(3+4i) = (3-4i)/25 = 0.12-0.16i
    let (a, b) = (3.0_f64, 4.0_f64);
    let expected = complex_reciprocal(a, b);

    let input = make_complex128_tensor(&[], &[(a, b)]);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], expected, 1e-10, "reciprocal(3+4i)");
}

#[test]
fn oracle_reciprocal_complex64_vector() {
    let data: &[(f32, f32)] = &[(2.0, 0.0), (0.0, 2.0), (1.0, 1.0), (3.0, 4.0)];
    let input = make_complex64_tensor(&[4], data);
    let result = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 4);

    // reciprocal(2) = 0.5
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.5, 0.0),
        1e-4,
        "reciprocal(2)",
    );

    // reciprocal(2i) = -0.5i
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (0.0, -0.5),
        1e-4,
        "reciprocal(2i)",
    );

    // reciprocal(1+i) = 0.5-0.5i
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.5, -0.5),
        1e-4,
        "reciprocal(1+i)",
    );

    // reciprocal(3+4i)
    let expected = complex_reciprocal(3.0, 4.0);
    assert_complex_close(
        (vec[3].0 as f64, vec[3].1 as f64),
        expected,
        1e-4,
        "reciprocal(3+4i)",
    );
}

#[test]
fn oracle_reciprocal_complex_involution() {
    // reciprocal(reciprocal(z)) = z
    let values: &[(f64, f64)] = &[(2.0, 0.0), (0.0, 2.0), (1.0, 1.0), (3.0, 4.0)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);
        let recip1 = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();
        let recip2 = eval_primitive(Primitive::Reciprocal, &[recip1], &no_params()).unwrap();
        let result = extract_complex128_vec(&recip2)[0];

        assert_complex_close(
            result,
            (a, b),
            1e-10,
            &format!("reciprocal(reciprocal({a}+{b}i)) = {a}+{b}i"),
        );
    }
}

#[test]
fn oracle_reciprocal_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(1.0, 1.0), (3.0, 4.0)]);
    let c64_result = eval_primitive(Primitive::Reciprocal, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex64,
                "reciprocal should preserve Complex64"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(1.0, 1.0), (3.0, 4.0)]);
    let c128_result = eval_primitive(Primitive::Reciprocal, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "reciprocal should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
