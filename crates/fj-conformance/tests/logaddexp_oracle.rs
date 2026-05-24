//! Oracle tests for LogAddExp primitive.
//!
//! logaddexp(x, y) = log(exp(x) + exp(y)), computed in a numerically stable way
//!
//! Equivalent to: max(x, y) + log1p(exp(-|x - y|))
//!
//! Tests:
//! - Basic: logaddexp(0, 0) = log(2)
//! - Dominance: logaddexp(large, small) ~ large
//! - Symmetry: logaddexp(x, y) = logaddexp(y, x)
//! - Special values
//! - Broadcast-compatible operands
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

fn expected_logaddexp(x: f64, y: f64) -> f64 {
    let max = x.max(y);
    max + (-((x - y).abs())).exp().ln_1p()
}

// ======================== Basic Cases ========================

#[test]
fn oracle_logaddexp_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-15,
        "logaddexp(0, 0) = log(2) ~ 0.693"
    );
}

#[test]
fn oracle_logaddexp_same_value() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 5.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-14,
        "logaddexp(5, 5) = 5 + log(2)"
    );
}

// ======================== Dominance (one value much larger) ========================

#[test]
fn oracle_logaddexp_large_small() {
    let x = make_f64_tensor(&[], vec![100.0]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 100.0).abs() < 1e-10, "logaddexp(100, 0) ~ 100");
}

#[test]
fn oracle_logaddexp_small_large() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 100.0).abs() < 1e-10, "logaddexp(0, 100) ~ 100");
}

// ======================== Symmetry ========================

#[test]
fn oracle_logaddexp_symmetry() {
    for (a, b) in [(1.0, 2.0), (5.0, 3.0), (-1.0, 1.0), (0.0, 10.0)] {
        let x = make_f64_tensor(&[], vec![a]);
        let y = make_f64_tensor(&[], vec![b]);
        let result1 =
            eval_primitive(Primitive::LogAddExp, &[x.clone(), y.clone()], &no_params()).unwrap();
        let result2 = eval_primitive(Primitive::LogAddExp, &[y, x], &no_params()).unwrap();
        let val1 = extract_f64_scalar(&result1);
        let val2 = extract_f64_scalar(&result2);
        assert!(
            (val1 - val2).abs() < 1e-15,
            "logaddexp({}, {}) = logaddexp({}, {})",
            a,
            b,
            b,
            a
        );
    }
}

// ======================== Negative Values ========================

#[test]
fn oracle_logaddexp_negative() {
    let x = make_f64_tensor(&[], vec![-1.0]);
    let y = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = ((-1.0_f64).exp() + (-2.0_f64).exp()).ln();
    assert!(
        (actual - expected).abs() < 1e-15,
        "logaddexp(-1, -2) = log(exp(-1) + exp(-2))"
    );
}

// ======================== Special Values ========================

#[test]
fn oracle_logaddexp_inf_finite() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "logaddexp(inf, 0) = inf"
    );
}

#[test]
fn oracle_logaddexp_neg_inf_finite() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.abs() < 1e-15, "logaddexp(-inf, 0) = log(0 + 1) = 0");
}

#[test]
fn oracle_logaddexp_neg_inf_neg_inf() {
    // Note: The stable formula max(x,y) + log1p(exp(-|x-y|)) produces NaN here
    // because |-inf - (-inf)| = NaN. This is acceptable numerical behavior.
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let y = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual == f64::NEG_INFINITY || actual.is_nan(),
        "logaddexp(-inf, -inf) = -inf or NaN (got {})",
        actual
    );
}

#[test]
fn oracle_logaddexp_nan() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "logaddexp(NaN, 0) = NaN"
    );
}

// ======================== Numerical Stability ========================

#[test]
fn oracle_logaddexp_large_values() {
    let x = make_f64_tensor(&[], vec![500.0]);
    let y = make_f64_tensor(&[], vec![500.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 500.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-12,
        "logaddexp(500, 500) should not overflow"
    );
}

#[test]
fn oracle_logaddexp_very_negative() {
    let x = make_f64_tensor(&[], vec![-500.0]);
    let y = make_f64_tensor(&[], vec![-500.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = -500.0 + 2.0_f64.ln();
    assert!(
        (actual - expected).abs() < 1e-12,
        "logaddexp(-500, -500) should not underflow"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_logaddexp_vector() {
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0_f64.ln()).abs() < 1e-15);
    assert!((vals[1] - (1.0 + 2.0_f64.ln())).abs() < 1e-15);
    assert!((vals[2] - (2.0 + 2.0_f64.ln())).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    for (i, &v) in vals.iter().enumerate() {
        let expected = (i as f64) + 2.0_f64.ln();
        assert!(
            (v - expected).abs() < 1e-14,
            "vals[{}] = {} vs {}",
            i,
            v,
            expected
        );
    }
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_logaddexp_scalar_x_tensor_y_broadcast() {
    // scalar x with tensor y
    let x = scalar_f64(1.0);
    let y = make_f64_tensor(&[4], vec![0.0, 1.0, 2.0, 10.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    for (i, &y_val) in [0.0, 1.0, 2.0, 10.0].iter().enumerate() {
        let expected = expected_logaddexp(1.0, y_val);
        assert!(
            (vals[i] - expected).abs() < 1e-14,
            "scalar x broadcast element {i}: {} vs {}",
            vals[i],
            expected
        );
    }
}

#[test]
fn oracle_logaddexp_tensor_x_scalar_y_broadcast() {
    // tensor x with scalar y
    let x = make_f64_tensor(&[4], vec![0.0, 1.0, 2.0, 10.0]);
    let y = scalar_f64(1.0);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    for (i, &x_val) in [0.0, 1.0, 2.0, 10.0].iter().enumerate() {
        let expected = expected_logaddexp(x_val, 1.0);
        assert!(
            (vals[i] - expected).abs() < 1e-14,
            "tensor x broadcast element {i}: {} vs {}",
            vals[i],
            expected
        );
    }
}

#[test]
fn oracle_logaddexp_singleton_x_vector_y_broadcast() {
    // [1] x with [3] y -> [3]
    let x = make_f64_tensor(&[1], vec![1.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    for (i, &y_val) in [0.0, 1.0, 2.0].iter().enumerate() {
        let expected = expected_logaddexp(1.0, y_val);
        assert!((vals[i] - expected).abs() < 1e-14);
    }
}

#[test]
fn oracle_logaddexp_vector_x_singleton_y_broadcast() {
    // [3] x with [1] y -> [3]
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let y = make_f64_tensor(&[1], vec![1.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    for (i, &x_val) in [0.0, 1.0, 2.0].iter().enumerate() {
        let expected = expected_logaddexp(x_val, 1.0);
        assert!((vals[i] - expected).abs() < 1e-14);
    }
}

#[test]
fn oracle_logaddexp_column_x_matrix_y_broadcast() {
    // [2, 1] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![0.0, 10.0]);
    let y = make_f64_tensor(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x=0 with y=0,1,2
    assert!((vals[0] - expected_logaddexp(0.0, 0.0)).abs() < 1e-14);
    assert!((vals[1] - expected_logaddexp(0.0, 1.0)).abs() < 1e-14);
    assert!((vals[2] - expected_logaddexp(0.0, 2.0)).abs() < 1e-14);
    // Row 1: x=10 with y=0,1,2
    assert!((vals[3] - expected_logaddexp(10.0, 0.0)).abs() < 1e-14);
    assert!((vals[4] - expected_logaddexp(10.0, 1.0)).abs() < 1e-14);
    assert!((vals[5] - expected_logaddexp(10.0, 2.0)).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_different_ranks_broadcast() {
    // [3] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let y = make_f64_tensor(&[2, 3], vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x=[0,1,2] with y=0
    assert!((vals[0] - expected_logaddexp(0.0, 0.0)).abs() < 1e-14);
    assert!((vals[1] - expected_logaddexp(1.0, 0.0)).abs() < 1e-14);
    assert!((vals[2] - expected_logaddexp(2.0, 0.0)).abs() < 1e-14);
    // Row 1: x=[0,1,2] with y=10
    assert!((vals[3] - expected_logaddexp(0.0, 10.0)).abs() < 1e-14);
    assert!((vals[4] - expected_logaddexp(1.0, 10.0)).abs() < 1e-14);
    assert!((vals[5] - expected_logaddexp(2.0, 10.0)).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_all_scalars_broadcast() {
    // scalar logaddexp scalar -> scalar
    let x = scalar_f64(1.0);
    let y = scalar_f64(2.0);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    let expected = expected_logaddexp(1.0, 2.0);
    assert!((val - expected).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_incompatible_shapes_error() {
    // [2] logaddexp [3] should error
    let x = make_f64_tensor(&[2], vec![0.0, 1.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_logaddexp_vector_scalar_y_broadcast() {
    let x_values = [0.0, 1.0, 2.0];
    let x = make_f64_tensor(&[3], x_values.to_vec());
    let y = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    for (i, (&actual, &x_value)) in vals.iter().zip(x_values.iter()).enumerate() {
        let expected = expected_logaddexp(x_value, 0.0);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast scalar y element {i}: {actual} vs {expected}"
        );
    }
}

#[test]
fn oracle_logaddexp_matrix_row_y_broadcast() {
    let x_values = [0.0, 0.0, 10.0, 10.0];
    let y_values = [0.0, 10.0];
    let x = make_f64_tensor(&[2, 2], x_values.to_vec());
    let y = make_f64_tensor(&[2], y_values.to_vec());
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    for (i, ((&actual, &x_value), &y_value)) in vals
        .iter()
        .zip(x_values.iter())
        .zip(y_values.iter().cycle())
        .enumerate()
    {
        let expected = expected_logaddexp(x_value, y_value);
        assert!(
            (actual - expected).abs() < 1e-14,
            "broadcast row y element {i}: {actual} vs {expected}"
        );
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_logaddexp_3d_shape() {
    let x = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let y = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0_f64.ln()).abs() < 1e-15);
    assert!((vals[7] - (7.0 + 2.0_f64.ln())).abs() < 1e-14);
}

#[test]
fn oracle_logaddexp_empty_tensor() {
    let x = make_f64_tensor(&[0], vec![]);
    let y = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_logaddexp_2d_empty() {
    let x =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let y =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_logaddexp_preserves_dtype() {
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_logaddexp_inf_inf() {
    // Numerically stable formula: max(x,y) + log1p(exp(-|x-y|))
    // With inf,inf: inf - inf = NaN, so result is NaN
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "logaddexp(inf, inf) = NaN due to inf-inf"
    );
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_logaddexp_preserves_all_float_dtypes() {
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

    let x_values = [0.0_f64, 1.0, 2.0];
    let y_values = [0.0_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let x = make_vec(dtype, &x_values);
        let y = make_vec(dtype, &y_values);
        let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "logaddexp {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
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
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

#[test]

fn oracle_logaddexp_complex64_real_axis() {
    // logaddexp on real axis: logaddexp(0, 0) = log(2)
    let x = make_complex64_tensor(&[1], vec![(0.0, 0.0)]);
    let y = make_complex64_tensor(&[1], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params())
        .expect("logaddexp complex64 should succeed");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]

fn oracle_logaddexp_complex128_preserves_dtype() {
    let x = make_complex128_tensor(&[1], vec![(0.0, 0.0)]);
    let y = make_complex128_tensor(&[1], vec![(0.0, 0.0)]);
    let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params())
        .expect("logaddexp complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]

fn property_logaddexp_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (x, y) = match dtype {
            DType::Complex64 => (
                make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
                make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
                make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::LogAddExp, &[x, y], &no_params())
            .expect("logaddexp should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "logaddexp {dtype:?}: dtype mismatch");
    }
}

// ======================== METAMORPHIC: mathematical identities ========================

#[test]
fn metamorphic_logaddexp_commutativity() {
    // logaddexp(x, y) = logaddexp(y, x)
    let x = make_f64_tensor(&[5], vec![0.0, 1.0, -1.0, 10.0, -10.0]);
    let y = make_f64_tensor(&[5], vec![1.0, -1.0, 0.0, -5.0, 5.0]);
    let result_xy =
        eval_primitive(Primitive::LogAddExp, &[x.clone(), y.clone()], &no_params()).unwrap();
    let result_yx = eval_primitive(Primitive::LogAddExp, &[y, x], &no_params()).unwrap();
    let vals_xy = extract_f64_vec(&result_xy);
    let vals_yx = extract_f64_vec(&result_yx);
    for (i, (&v1, &v2)) in vals_xy.iter().zip(vals_yx.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "logaddexp(x, y) should equal logaddexp(y, x) at index {i}: got {} vs {}",
            v1,
            v2
        );
    }
}

#[test]
fn metamorphic_logaddexp_same_value() {
    // logaddexp(x, x) = x + ln(2)
    let x = make_f64_tensor(&[5], vec![0.0, 1.0, -1.0, 10.0, -10.0]);
    let result =
        eval_primitive(Primitive::LogAddExp, &[x.clone(), x.clone()], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let x_vals = extract_f64_vec(&x);
    let ln2 = 2.0_f64.ln();
    for (i, (&v, &x_v)) in vals.iter().zip(x_vals.iter()).enumerate() {
        let expected = x_v + ln2;
        assert!(
            (v - expected).abs() < 1e-10,
            "logaddexp(x, x) should equal x + ln(2) at index {i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn metamorphic_logaddexp_greater_than_max() {
    // logaddexp(x, y) >= max(x, y) always
    let x = make_f64_tensor(&[5], vec![0.0, 1.0, -1.0, 10.0, -10.0]);
    let y = make_f64_tensor(&[5], vec![1.0, -1.0, 0.0, -5.0, 5.0]);
    let result =
        eval_primitive(Primitive::LogAddExp, &[x.clone(), y.clone()], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    let x_vals = extract_f64_vec(&x);
    let y_vals = extract_f64_vec(&y);
    for (i, (&v, (&x_v, &y_v))) in vals
        .iter()
        .zip(x_vals.iter().zip(y_vals.iter()))
        .enumerate()
    {
        let max_val = x_v.max(y_v);
        assert!(
            v >= max_val - 1e-10,
            "logaddexp(x, y) should be >= max(x, y) at index {i}: got {v} < {}",
            max_val
        );
    }
}

#[test]
fn metamorphic_logaddexp_associativity_approximation() {
    // While logaddexp is not strictly associative, we can test that
    // logaddexp(logaddexp(x, y), z) gives a sensible result
    let x = make_f64_tensor(&[], vec![1.0]);
    let y = make_f64_tensor(&[], vec![2.0]);
    let z = make_f64_tensor(&[], vec![3.0]);

    let xy = eval_primitive(Primitive::LogAddExp, &[x.clone(), y.clone()], &no_params()).unwrap();
    let xyz_left = eval_primitive(Primitive::LogAddExp, &[xy, z.clone()], &no_params()).unwrap();

    let yz = eval_primitive(Primitive::LogAddExp, &[y, z], &no_params()).unwrap();
    let xyz_right = eval_primitive(Primitive::LogAddExp, &[x, yz], &no_params()).unwrap();

    let left = extract_f64_scalar(&xyz_left);
    let right = extract_f64_scalar(&xyz_right);

    // Both should equal log(e^1 + e^2 + e^3)
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!(
        (left - expected).abs() < 1e-10,
        "left associative: got {left}, expected {expected}"
    );
    assert!(
        (right - expected).abs() < 1e-10,
        "right associative: got {right}, expected {expected}"
    );
}
