//! Oracle tests for Fma (fused multiply-add) primitive.
//!
//! fma(a, b, c) = a * b + c (with single rounding for better precision)
//!
//! Tests:
//! - Basic: fma(2, 3, 1) = 7
//! - Zero cases
//! - Negative values
//! - Associativity with mul/add
//! - Special values: infinity, NaN
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

// ======================== Basic Cases ========================

#[test]
fn oracle_fma_basic() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 7.0, "fma(2, 3, 1) = 7");
}

#[test]
fn oracle_fma_multiply_only() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let c = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 6.0, "fma(2, 3, 0) = 6");
}

#[test]
fn oracle_fma_add_only() {
    let a = make_f64_tensor(&[], vec![1.0]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let c = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "fma(1, 0, 5) = 5");
}

#[test]
fn oracle_fma_all_zeros() {
    let a = make_f64_tensor(&[], vec![0.0]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let c = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "fma(0, 0, 0) = 0");
}

// ======================== Negative Values ========================

#[test]
fn oracle_fma_negative_a() {
    let a = make_f64_tensor(&[], vec![-2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "fma(-2, 3, 1) = -5");
}

#[test]
fn oracle_fma_negative_b() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![-3.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "fma(2, -3, 1) = -5");
}

#[test]
fn oracle_fma_negative_c() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let c = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "fma(2, 3, -1) = 5");
}

#[test]
fn oracle_fma_all_negative() {
    let a = make_f64_tensor(&[], vec![-2.0]);
    let b = make_f64_tensor(&[], vec![-3.0]);
    let c = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "fma(-2, -3, -1) = 5");
}

// ======================== Special Values ========================

#[test]
fn oracle_fma_inf_finite() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "fma(inf, 2, 1) = inf"
    );
}

#[test]
fn oracle_fma_finite_inf() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let c = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "fma(2, 3, inf) = inf"
    );
}

#[test]
fn oracle_fma_nan() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "fma(NaN, 2, 1) = NaN");
}

#[test]
fn oracle_fma_inf_times_zero() {
    let a = make_f64_tensor(&[], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[], vec![0.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "fma(inf, 0, 1) = NaN (inf * 0)"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_fma_vector() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 2.0, 2.0]);
    let c = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 7.0]);
}

#[test]
fn oracle_fma_matrix() {
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[2, 2], vec![2.0, 2.0, 2.0, 2.0]);
    let c = make_f64_tensor(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 7.0, 9.0]);
}

// ======================== Comparison with mul+add ========================

#[test]
fn oracle_fma_equals_mul_add() {
    let a = make_f64_tensor(&[], vec![3.14159]);
    let b = make_f64_tensor(&[], vec![2.71828]);
    let c = make_f64_tensor(&[], vec![1.41421]);

    let fma_result = eval_primitive(Primitive::Fma, &[a.clone(), b.clone(), c.clone()], &no_params()).unwrap();

    let mul_result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    let add_result = eval_primitive(Primitive::Add, &[mul_result, c], &no_params()).unwrap();

    let fma_val = extract_f64_scalar(&fma_result);
    let separate_val = extract_f64_scalar(&add_result);

    assert!(
        (fma_val - separate_val).abs() < 1e-10,
        "fma should equal separate mul+add"
    );
}

// ======================== Broadcast Tests ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_fma_scalar_tensor_tensor_broadcast() {
    // scalar * tensor + tensor
    let a = scalar_f64(2.0);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let c = make_f64_tensor(&[3], vec![10.0, 20.0, 30.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![12.0, 24.0, 36.0]);
}

#[test]
fn oracle_fma_tensor_scalar_tensor_broadcast() {
    // tensor * scalar + tensor
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = scalar_f64(2.0);
    let c = make_f64_tensor(&[3], vec![10.0, 20.0, 30.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![12.0, 24.0, 36.0]);
}

#[test]
fn oracle_fma_tensor_tensor_scalar_broadcast() {
    // tensor * tensor + scalar
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 2.0, 2.0]);
    let c = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![12.0, 14.0, 16.0]);
}

#[test]
fn oracle_fma_scalar_scalar_tensor_broadcast() {
    // scalar * scalar + tensor
    let a = scalar_f64(2.0);
    let b = scalar_f64(3.0);
    let c = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![7.0, 8.0, 9.0]);
}

#[test]
fn oracle_fma_row_matrix_broadcast() {
    // [1, 3] * [2, 3] + [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    let c = make_f64_tensor(&[2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 7.0, 4.0, 7.0, 10.0]);
}

#[test]
fn oracle_fma_column_matrix_broadcast() {
    // [2, 1] * [2, 3] + [2, 3] -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let c = make_f64_tensor(&[2, 3], vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![12.0, 14.0, 16.0, 23.0, 26.0, 29.0]);
}

#[test]
fn oracle_fma_different_ranks_broadcast() {
    // [3] * [2, 3] + [1] -> [2, 3]
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[2, 3], vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 5.0, 7.0, 4.0, 7.0, 10.0]);
}

#[test]
fn oracle_fma_all_scalars() {
    let a = scalar_f64(2.0);
    let b = scalar_f64(3.0);
    let c = scalar_f64(1.0);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 7.0);
}

#[test]
fn oracle_fma_incompatible_shapes_error() {
    // [2] * [3] + [1] should error
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_fma_3d_shape() {
    let a = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = make_f64_tensor(&[2, 2, 2], vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let c = make_f64_tensor(&[2, 2, 2], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 3.0);  // 1*2+1
    assert_eq!(vals[7], 17.0); // 8*2+1
}

#[test]
fn oracle_fma_empty_tensor() {
    let a = make_f64_tensor(&[0], vec![]);
    let b = make_f64_tensor(&[0], vec![]);
    let c = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_fma_2d_empty() {
    let a = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let b = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let c = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_fma_preserves_dtype() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 2.0, 2.0]);
    let c = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_fma_neg_inf() {
    let a = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let c = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY, "fma(-inf, 2, 1) = -inf");
}

#[test]
fn oracle_fma_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let a = make_f64_tensor(&[], vec![subnormal]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let c = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - subnormal * 2.0).abs() < 1e-300, "fma(subnormal, 2, 0) = 2*subnormal");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_fma_preserves_f32_f64_dtypes() {
    // Note: BF16/F16 have dtype consistency issues (literals stored as F64Bits).
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not F32 or F64"),
            })
            .collect();
        Value::Tensor(
            TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap(),
        )
    }

    let a_values = [1.0_f64, 2.0, 3.0];
    let b_values = [2.0_f64, 3.0, 4.0];
    let c_values = [0.5_f64, 1.0, 1.5];
    for dtype in [DType::F32, DType::F64] {
        let a = make_vec(dtype, &a_values);
        let b = make_vec(dtype, &b_values);
        let c = make_vec(dtype, &c_values);
        let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "fma {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: shape.to_vec() },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: shape.to_vec() },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_complex64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]
#[ignore = "PARITY GAP: fma not supported for complex operands"]
fn oracle_fma_complex64_basic() {
    // fma(1+i, 1+i, 0) = (1+i)*(1+i) + 0 = 2i
    let a = make_complex64_tensor(&[1], vec![(1.0, 1.0)]);
    let b = make_complex64_tensor(&[1], vec![(1.0, 1.0)]);
    let c = make_complex64_tensor(&[1], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params())
        .expect("fma complex64 should succeed");
    let vals = extract_complex64_vec(&result);
    assert!(vals[0].0.abs() < 1e-5, "expected 0, got {}", vals[0].0);
    assert!((vals[0].1 - 2.0).abs() < 1e-5, "expected 2, got {}", vals[0].1);
}

#[test]
#[ignore = "PARITY GAP: fma not supported for complex operands"]
fn oracle_fma_complex64_with_addend() {
    // fma(1, 2, 3+4i) = 1*2 + (3+4i) = 5+4i
    let a = make_complex64_tensor(&[1], vec![(1.0, 0.0)]);
    let b = make_complex64_tensor(&[1], vec![(2.0, 0.0)]);
    let c = make_complex64_tensor(&[1], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params())
        .expect("fma complex64 with addend should succeed");
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 5.0).abs() < 1e-5);
    assert!((vals[0].1 - 4.0).abs() < 1e-5);
}

#[test]
#[ignore = "PARITY GAP: fma not supported for complex operands"]
fn oracle_fma_complex128_preserves_dtype() {
    let a = make_complex128_tensor(&[1], vec![(1.0, 0.0)]);
    let b = make_complex128_tensor(&[1], vec![(2.0, 0.0)]);
    let c = make_complex128_tensor(&[1], vec![(3.0, 0.0)]);
    let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params())
        .expect("fma complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
#[ignore = "PARITY GAP: fma not supported for complex operands"]
fn property_fma_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (a, b, c) = match dtype {
            DType::Complex64 => (
                make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
                make_complex64_tensor(&[2], vec![(2.0, 0.0), (3.0, 0.0)]),
                make_complex64_tensor(&[2], vec![(0.0, 0.0), (0.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]),
                make_complex128_tensor(&[2], vec![(2.0, 0.0), (3.0, 0.0)]),
                make_complex128_tensor(&[2], vec![(0.0, 0.0), (0.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::Fma, &[a, b, c], &no_params())
            .expect("fma should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "fma {dtype:?}: dtype mismatch");
    }
}
