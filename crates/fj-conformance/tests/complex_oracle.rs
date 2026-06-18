//! Oracle tests for Complex primitive.
//!
//! Tests against expected behavior for creating complex numbers:
//! - Takes two inputs: real part and imaginary part
//! - Creates complex128 output

#![allow(clippy::approx_constant)]

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

fn extract_complex_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                Literal::Complex64Bits(re, im) => {
                    (f32::from_bits(*re) as f64, f32::from_bits(*im) as f64)
                }
                _ => unreachable!("expected complex"),
            })
            .collect(),
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            vec![(f64::from_bits(*re), f64::from_bits(*im))]
        }
        Value::Scalar(Literal::Complex64Bits(re, im)) => {
            vec![(f32::from_bits(*re) as f64, f32::from_bits(*im) as f64)]
        }
        _ => unreachable!("expected complex"),
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

// ======================== Scalar Tests ========================

#[test]
fn oracle_complex_scalar_basic() {
    // complex(3, 4) = 3 + 4i
    let re = Value::Scalar(Literal::from_f64(3.0));
    let im = Value::Scalar(Literal::from_f64(4.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 3.0).abs() < 1e-10);
    assert!((vals[0].1 - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_negative() {
    // complex(-2, -5) = -2 - 5i
    let re = Value::Scalar(Literal::from_f64(-2.0));
    let im = Value::Scalar(Literal::from_f64(-5.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - (-2.0)).abs() < 1e-10);
    assert!((vals[0].1 - (-5.0)).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zero_imag() {
    // complex(5, 0) = 5 + 0i (real number)
    let re = Value::Scalar(Literal::from_f64(5.0));
    let im = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 5.0).abs() < 1e-10);
    assert!(vals[0].1.abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zero_real() {
    // complex(0, 7) = 0 + 7i (pure imaginary)
    let re = Value::Scalar(Literal::from_f64(0.0));
    let im = Value::Scalar(Literal::from_f64(7.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.abs() < 1e-10);
    assert!((vals[0].1 - 7.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_scalar_zeros() {
    // complex(0, 0) = 0 + 0i
    let re = Value::Scalar(Literal::from_f64(0.0));
    let im = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.abs() < 1e-10);
    assert!(vals[0].1.abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_complex_1d() {
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 4.0).abs() < 1e-10);
    assert!((vals[2].0 - 3.0).abs() < 1e-10);
    assert!((vals[2].1 - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_1d_mixed_signs() {
    let re = make_f64_tensor(&[4], vec![-1.0, 1.0, -1.0, 1.0]);
    let im = make_f64_tensor(&[4], vec![1.0, -1.0, -1.0, 1.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - (-1.0)).abs() < 1e-10);
    assert!((vals[0].1 - 1.0).abs() < 1e-10);
    assert!((vals[1].0 - 1.0).abs() < 1e-10);
    assert!((vals[1].1 - (-1.0)).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_complex_2d() {
    let re = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let im = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 5.0).abs() < 1e-10);
    assert!((vals[3].0 - 4.0).abs() < 1e-10);
    assert!((vals[3].1 - 8.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_complex_single_element() {
    let re = make_f64_tensor(&[1], vec![3.17]);
    let im = make_f64_tensor(&[1], vec![2.71]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 3.17).abs() < 1e-10);
    assert!((vals[0].1 - 2.71).abs() < 1e-10);
}

#[test]
fn oracle_complex_large() {
    let re = make_f64_tensor(&[10], (1..=10).map(|x| x as f64).collect());
    let im = make_f64_tensor(&[10], (11..=20).map(|x| x as f64).collect());
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![10]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 11.0).abs() < 1e-10);
    assert!((vals[9].0 - 10.0).abs() < 1e-10);
    assert!((vals[9].1 - 20.0).abs() < 1e-10);
}

// ======================== Broadcast Tests ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_complex_broadcast_scalar_real_tensor_imag() {
    let re = scalar_f64(1.0);
    let im = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 2.0).abs() < 1e-10);
    assert!((vals[1].1 - 3.0).abs() < 1e-10);
    assert!((vals[2].1 - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_broadcast_tensor_real_scalar_imag() {
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = scalar_f64(5.0);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 5.0).abs() < 1e-10);
    assert!((vals[1].0 - 2.0).abs() < 1e-10);
    assert!((vals[2].0 - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_broadcast_singleton_to_vector() {
    let re = make_f64_tensor(&[1], vec![7.0]);
    let im = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 7.0).abs() < 1e-10);
    assert!((vals[1].0 - 7.0).abs() < 1e-10);
    assert!((vals[2].0 - 7.0).abs() < 1e-10);
    assert!((vals[0].1 - 1.0).abs() < 1e-10);
    assert!((vals[2].1 - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_broadcast_vector_to_singleton() {
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = make_f64_tensor(&[1], vec![9.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[1].0 - 2.0).abs() < 1e-10);
    assert!((vals[2].0 - 3.0).abs() < 1e-10);
    for v in &vals {
        assert!((v.1 - 9.0).abs() < 1e-10);
    }
}

#[test]
fn oracle_complex_broadcast_column_to_row() {
    // [2,1] x [1,3] → [2,3]
    let re = make_f64_tensor(&[2, 1], vec![1.0, 2.0]);
    let im = make_f64_tensor(&[1, 3], vec![10.0, 20.0, 30.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_complex_vec(&result);
    // Row 0: re=1, im=[10,20,30]
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 10.0).abs() < 1e-10);
    assert!((vals[1].1 - 20.0).abs() < 1e-10);
    assert!((vals[2].1 - 30.0).abs() < 1e-10);
    // Row 1: re=2, im=[10,20,30]
    assert!((vals[3].0 - 2.0).abs() < 1e-10);
    assert!((vals[4].0 - 2.0).abs() < 1e-10);
    assert!((vals[5].0 - 2.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_broadcast_different_ranks() {
    // [3] x [2,3] → [2,3]
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = make_f64_tensor(&[2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_complex_vec(&result);
    // Row 0: re=[1,2,3] broadcast, im=[10,20,30]
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 10.0).abs() < 1e-10);
    // Row 1: re=[1,2,3] broadcast, im=[40,50,60]
    assert!((vals[3].0 - 1.0).abs() < 1e-10);
    assert!((vals[3].1 - 40.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_broadcast_incompatible_shapes_error() {
    let re = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let im = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params());
    assert!(
        result.is_err(),
        "incompatible shapes [2] vs [3] should error"
    );
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_complex_empty_tensor() {
    let re = make_f64_tensor(&[0], vec![]);
    let im = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_complex_output_dtype() {
    let re = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let im = make_f64_tensor(&[2], vec![3.0, 4.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_complex_3d() {
    let re = make_f64_tensor(&[2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let im = make_f64_tensor(&[2, 1, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    let vals = extract_complex_vec(&result);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[0].1 - 5.0).abs() < 1e-10);
    assert!((vals[3].0 - 4.0).abs() < 1e-10);
    assert!((vals[3].1 - 8.0).abs() < 1e-10);
}

#[test]
fn oracle_complex_special_values() {
    let re = make_f64_tensor(&[4], vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0]);
    let im = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    assert!(vals[0].0.is_infinite() && vals[0].0 > 0.0);
    assert!(vals[1].0.is_infinite() && vals[1].0 < 0.0);
    assert!(vals[1].1.is_infinite() && vals[1].1 > 0.0);
    assert!(vals[2].0.is_nan());
    assert!(vals[3].1.is_infinite() && vals[3].1 < 0.0);
}

#[test]
fn oracle_complex_2d_empty() {
    let re =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let im =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

// ======================== METAMORPHIC: mathematical properties ========================

#[test]
fn metamorphic_complex_zero_is_zero_element() {
    // Complex(0, 0) should be the zero element for complex addition
    let zeros = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Complex, &[zeros.clone(), zeros], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    for (re, im) in vals {
        assert!(re.abs() < 1e-10, "real part should be zero");
        assert!(im.abs() < 1e-10, "imag part should be zero");
    }
}

#[test]
fn metamorphic_complex_purely_real_has_zero_imag() {
    // Complex(a, 0) should have zero imaginary part
    let reals = make_f64_tensor(&[4], vec![1.0, -2.5, 0.0, 100.0]);
    let zeros = make_f64_tensor(&[4], vec![0.0, 0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Complex, &[reals, zeros], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    let expected = [1.0, -2.5, 0.0, 100.0];
    for (i, (re, im)) in vals.iter().enumerate() {
        assert!((*re - expected[i]).abs() < 1e-10, "real part mismatch");
        assert!(im.abs() < 1e-10, "imag should be zero for purely real");
    }
}

#[test]
fn metamorphic_complex_purely_imag_has_zero_real() {
    // Complex(0, b) should have zero real part
    let zeros = make_f64_tensor(&[4], vec![0.0, 0.0, 0.0, 0.0]);
    let imags = make_f64_tensor(&[4], vec![1.0, -2.5, 0.0, 100.0]);
    let result = eval_primitive(Primitive::Complex, &[zeros, imags], &no_params()).unwrap();
    let vals = extract_complex_vec(&result);
    let expected = [1.0, -2.5, 0.0, 100.0];
    for (i, (re, im)) in vals.iter().enumerate() {
        assert!(re.abs() < 1e-10, "real should be zero for purely imaginary");
        assert!((*im - expected[i]).abs() < 1e-10, "imag part mismatch");
    }
}

#[test]
fn metamorphic_complex_negation_symmetry() {
    // Complex(-a, -b) = -Complex(a, b)
    let re = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let im = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let neg_re = make_f64_tensor(&[3], vec![-1.0, -2.0, -3.0]);
    let neg_im = make_f64_tensor(&[3], vec![-4.0, -5.0, -6.0]);
    let pos_result = eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap();
    let neg_result = eval_primitive(Primitive::Complex, &[neg_re, neg_im], &no_params()).unwrap();
    let pos_vals = extract_complex_vec(&pos_result);
    let neg_vals = extract_complex_vec(&neg_result);
    for ((pos_re, pos_im), (neg_re, neg_im)) in pos_vals.iter().zip(neg_vals.iter()) {
        assert!(
            (*pos_re + *neg_re).abs() < 1e-10,
            "negation: real parts should sum to zero"
        );
        assert!(
            (*pos_im + *neg_im).abs() < 1e-10,
            "negation: imag parts should sum to zero"
        );
    }
}

// ============ METAMORPHIC: complex transcendental inverse identities ============
// Oracle-free mathematical identities (no JAX runtime needed) that exercise the
// complex exp/log/sqrt/reciprocal/mul/add evaluators end-to-end. These catch
// divergence in the complex transcendental implementations — the same family where
// complex bessel had a real truncation bug (fixed in 0f1ed8b9) and which the
// frankenjax-w8u0a fail-close touched. All ops used here are JAX `_float|_complex`.

fn complex_from_pairs(pairs: &[(f64, f64)]) -> Value {
    let n = pairs.len() as u32;
    let re = make_f64_tensor(&[n], pairs.iter().map(|p| p.0).collect());
    let im = make_f64_tensor(&[n], pairs.iter().map(|p| p.1).collect());
    eval_primitive(Primitive::Complex, &[re, im], &no_params()).unwrap()
}

fn unary(prim: Primitive, z: &Value) -> Value {
    eval_primitive(prim, std::slice::from_ref(z), &no_params()).unwrap()
}

fn assert_complex_close(got: &Value, expected: &[(f64, f64)], tol: f64, msg: &str) {
    let g = extract_complex_vec(got);
    assert_eq!(g.len(), expected.len(), "{msg}: length");
    for (i, ((gre, gim), (ere, eim))) in g.iter().zip(expected).enumerate() {
        assert!(
            (gre - ere).abs() < tol && (gim - eim).abs() < tol,
            "{msg} [{i}]: got ({gre}, {gim}) expected ({ere}, {eim})"
        );
    }
}

#[test]
fn metamorphic_complex_exp_log_roundtrip() {
    // exp(log(z)) == z for every z != 0 (principal log returns the branch exp inverts).
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let back = unary(Primitive::Exp, &unary(Primitive::Log, &z));
    assert_complex_close(&back, &pts, 1e-9, "exp(log(z)) == z");
}

#[test]
fn metamorphic_complex_sqrt_squared_roundtrip() {
    // sqrt(z) * sqrt(z) == z for every z (principal sqrt).
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let s = unary(Primitive::Sqrt, &z);
    let sq = eval_primitive(Primitive::Mul, &[s.clone(), s], &no_params()).unwrap();
    assert_complex_close(&sq, &pts, 1e-9, "sqrt(z)^2 == z");
}

#[test]
fn metamorphic_complex_reciprocal_involution() {
    // reciprocal(reciprocal(z)) == z for every z != 0.
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let back = unary(Primitive::Reciprocal, &unary(Primitive::Reciprocal, &z));
    assert_complex_close(&back, &pts, 1e-9, "1/(1/z) == z");
}

#[test]
fn metamorphic_complex_exp_addition_is_multiplication() {
    // exp(a + b) == exp(a) * exp(b). Small arguments keep values O(1).
    let a_pts = [(0.5, 0.3), (-0.2, 0.7), (0.1, -0.4)];
    let b_pts = [(-0.2, 0.6), (0.4, -0.1), (0.3, 0.2)];
    let a = complex_from_pairs(&a_pts);
    let b = complex_from_pairs(&b_pts);
    let sum = eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap();
    let exp_sum = unary(Primitive::Exp, &sum);
    let exp_a = unary(Primitive::Exp, &a);
    let exp_b = unary(Primitive::Exp, &b);
    let prod = eval_primitive(Primitive::Mul, &[exp_a, exp_b], &no_params()).unwrap();
    let expected = extract_complex_vec(&prod);
    assert_complex_close(&exp_sum, &expected, 1e-9, "exp(a+b) == exp(a)*exp(b)");
}

#[test]
fn metamorphic_complex_pythagorean_identity() {
    // sin(z)^2 + cos(z)^2 == 1 for every complex z. Exercises complex_sin and
    // complex_cos (eval_sin/eval_cos route complex through eval_unary_complex_map)
    // which the exp/log/sqrt/reciprocal suite does not touch. Imag parts kept
    // moderate so the cosh^2 - sinh^2 cancellation stays well-conditioned.
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let s = unary(Primitive::Sin, &z);
    let c = unary(Primitive::Cos, &z);
    let s2 = eval_primitive(Primitive::Mul, &[s.clone(), s], &no_params()).unwrap();
    let c2 = eval_primitive(Primitive::Mul, &[c.clone(), c], &no_params()).unwrap();
    let sum = eval_primitive(Primitive::Add, &[s2, c2], &no_params()).unwrap();
    let ones: Vec<(f64, f64)> = pts.iter().map(|_| (1.0, 0.0)).collect();
    assert_complex_close(&sum, &ones, 1e-9, "sin(z)^2 + cos(z)^2 == 1");
}

#[test]
fn metamorphic_complex_hyperbolic_identity() {
    // cosh(z)^2 - sinh(z)^2 == 1 for every complex z. Exercises complex_sinh and
    // complex_cosh (eval_sinh/eval_cosh route complex through eval_unary_complex_map)
    // — distinct code paths from the sin/cos Pythagorean test. Imag parts kept
    // moderate so the cancellation stays well-conditioned.
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let sh = unary(Primitive::Sinh, &z);
    let ch = unary(Primitive::Cosh, &z);
    let sh2 = eval_primitive(Primitive::Mul, &[sh.clone(), sh], &no_params()).unwrap();
    let ch2 = eval_primitive(Primitive::Mul, &[ch.clone(), ch], &no_params()).unwrap();
    let diff = eval_primitive(Primitive::Sub, &[ch2, sh2], &no_params()).unwrap();
    let ones: Vec<(f64, f64)> = pts.iter().map(|_| (1.0, 0.0)).collect();
    assert_complex_close(&diff, &ones, 1e-9, "cosh(z)^2 - sinh(z)^2 == 1");
}

#[test]
fn metamorphic_complex_tan_equals_sin_over_cos() {
    // tan(z) == sin(z) / cos(z). complex_tan SATURATES for |Im(z)| > 20; these
    // moderate points stay in the standard branch, so it must match the ratio of
    // the independently-implemented complex sin and cos. Catches divergence between
    // complex_tan and its definition in the normal regime.
    let pts = [(1.5, 0.7), (-0.8, 1.2), (2.0, -1.0), (0.3, 0.4)];
    let z = complex_from_pairs(&pts);
    let tan = unary(Primitive::Tan, &z);
    let s = unary(Primitive::Sin, &z);
    let c = unary(Primitive::Cos, &z);
    let ratio = eval_primitive(Primitive::Div, &[s, c], &no_params()).unwrap();
    let expected = extract_complex_vec(&ratio);
    assert_complex_close(&tan, &expected, 1e-9, "tan(z) == sin(z)/cos(z)");
}

#[test]
fn metamorphic_complex_tanh_equals_sinh_over_cosh() {
    // tanh(z) == sinh(z) / cosh(z). complex_tanh SATURATES for large |Re(z)|; these
    // moderate points stay in the standard branch, so it must match the ratio of the
    // independently-implemented complex sinh and cosh.
    let pts = [(0.7, 1.5), (1.2, -0.8), (-1.0, 2.0), (0.4, 0.3)];
    let z = complex_from_pairs(&pts);
    let tanh = unary(Primitive::Tanh, &z);
    let sh = unary(Primitive::Sinh, &z);
    let ch = unary(Primitive::Cosh, &z);
    let ratio = eval_primitive(Primitive::Div, &[sh, ch], &no_params()).unwrap();
    let expected = extract_complex_vec(&ratio);
    assert_complex_close(&tanh, &expected, 1e-9, "tanh(z) == sinh(z)/cosh(z)");
}

#[test]
fn metamorphic_complex_inverse_trig_round_trips() {
    // Forward-of-inverse round trips for the recently-stabilized complex_asin/acos/
    // atan (commit 01152b59, HFT): sin(asin(z)) == z, cos(acos(z)) == z, and
    // tan(atan(z)) == z. The forward function recovers z from the inverse's
    // principal value for these points (away from the atan singularities ±i).
    // Regression coverage on recently-modified inverse-trig code.
    let pts = [(0.5, 0.3), (-0.8, 0.6), (1.2, -0.4), (0.3, 0.7)];
    let z = complex_from_pairs(&pts);

    let sin_asin = unary(Primitive::Sin, &unary(Primitive::Asin, &z));
    assert_complex_close(&sin_asin, &pts, 1e-9, "sin(asin(z)) == z");

    let cos_acos = unary(Primitive::Cos, &unary(Primitive::Acos, &z));
    assert_complex_close(&cos_acos, &pts, 1e-9, "cos(acos(z)) == z");

    let tan_atan = unary(Primitive::Tan, &unary(Primitive::Atan, &z));
    assert_complex_close(&tan_atan, &pts, 1e-9, "tan(atan(z)) == z");
}

#[test]
fn metamorphic_complex_inverse_hyperbolic_round_trips() {
    // Forward-of-inverse round trips for complex_asinh/acosh/atanh (eval_asinh/
    // acosh/atanh route complex through eval_unary_complex_map): sinh(asinh(z))==z,
    // cosh(acosh(z))==z, tanh(atanh(z))==z. Points chosen away from the branch cuts
    // (acosh cut on (-inf, 1], atanh poles at +/-1) so the principal inverse is
    // recovered exactly. Completes the complex inverse-function metamorphic family.
    let pts = [(1.5, 0.5), (2.0, -0.8), (1.2, 0.9), (0.6, 0.4)];
    let z = complex_from_pairs(&pts);

    let sinh_asinh = unary(Primitive::Sinh, &unary(Primitive::Asinh, &z));
    assert_complex_close(&sinh_asinh, &pts, 1e-9, "sinh(asinh(z)) == z");

    let cosh_acosh = unary(Primitive::Cosh, &unary(Primitive::Acosh, &z));
    assert_complex_close(&cosh_acosh, &pts, 1e-9, "cosh(acosh(z)) == z");

    // atanh poles at +/-1; keep |z| modest and away from the real axis >= 1.
    let tanh_pts = [(0.5, 0.3), (-0.4, 0.6), (0.2, -0.5), (0.3, 0.4)];
    let tz = complex_from_pairs(&tanh_pts);
    let tanh_atanh = unary(Primitive::Tanh, &unary(Primitive::Atanh, &tz));
    assert_complex_close(&tanh_atanh, &tanh_pts, 1e-9, "tanh(atanh(z)) == z");
}
