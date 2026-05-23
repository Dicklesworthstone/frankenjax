//! Oracle tests for Betainc primitive.
//!
//! betainc(a, b, x) = I_x(a,b) (regularized incomplete beta function)
//!
//! Key properties:
//! - betainc(a, b, 0) = 0
//! - betainc(a, b, 1) = 1
//! - betainc(1, 1, x) = x (uniform CDF)
//! - betainc(a, b, x) + betainc(b, a, 1-x) = 1
//!
//! Tests:
//! - Boundary values
//! - Uniform case
//! - Symmetry property
//! - Tensor shapes
//! - Broadcast-compatible operands

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

// ======================== Boundary Cases ========================

#[test]
fn oracle_betainc_x_zero() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "betainc(a, b, 0) = 0");
}

#[test]
fn oracle_betainc_x_one() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "betainc(a, b, 1) = 1");
}

// ======================== Uniform Case (a=1, b=1) ========================

#[test]
fn oracle_betainc_uniform() {
    // betainc(1, 1, x) = x (uniform distribution CDF)
    for x_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let a = make_f64_tensor(&[], vec![1.0]);
        let b = make_f64_tensor(&[], vec![1.0]);
        let x = make_f64_tensor(&[], vec![x_val]);
        let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
        let actual = extract_f64_scalar(&result);
        assert!(
            (actual - x_val).abs() < 1e-14,
            "betainc(1, 1, {}) = {}, got {}",
            x_val,
            x_val,
            actual
        );
    }
}

// ======================== Symmetry Property ========================

#[test]
fn oracle_betainc_symmetry() {
    // betainc(a, b, x) + betainc(b, a, 1-x) = 1
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![0.4]);
    let one_minus_x = make_f64_tensor(&[], vec![0.6]);

    let r1 = eval_primitive(Primitive::Betainc, &[a.clone(), b.clone(), x], &no_params()).unwrap();
    let r2 = eval_primitive(Primitive::Betainc, &[b, a, one_minus_x], &no_params()).unwrap();

    let sum = extract_f64_scalar(&r1) + extract_f64_scalar(&r2);
    assert!(
        (sum - 1.0).abs() < 1e-14,
        "betainc(a,b,x) + betainc(b,a,1-x) = 1, got {}",
        sum
    );
}

// ======================== Known Values ========================

#[test]
fn oracle_betainc_half() {
    // betainc(a, a, 0.5) = 0.5 for any a (symmetric case)
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 0.5).abs() < 1e-14, "betainc(a, a, 0.5) = 0.5");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_betainc_vector() {
    let a = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 0.5).abs() < 1e-14);
    assert!((vals[2] - 1.0).abs() < 1e-14);
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_betainc_all_scalars_broadcast() {
    // scalar betainc with all scalars -> scalar
    let a = scalar_f64(1.0);
    let b = scalar_f64(1.0);
    let x = scalar_f64(0.5);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert!((extract_f64_scalar(&result) - 0.5).abs() < 1e-14);
}

#[test]
fn oracle_betainc_scalar_a_scalar_b_tensor_x_broadcast() {
    // scalar a, scalar b, tensor x
    let a = scalar_f64(1.0);
    let b = scalar_f64(1.0);
    let x = make_f64_tensor(&[4], vec![0.0, 0.25, 0.75, 1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    // betainc(1, 1, x) = x
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 0.25).abs() < 1e-14);
    assert!((vals[2] - 0.75).abs() < 1e-14);
    assert!((vals[3] - 1.0).abs() < 1e-14);
}

#[test]
fn oracle_betainc_tensor_a_scalar_b_scalar_x_broadcast() {
    // tensor a, scalar b, scalar x
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = scalar_f64(1.0);
    let x = scalar_f64(0.0);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // betainc(a, b, 0) = 0 for all a, b
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.0);
    assert_eq!(vals[2], 0.0);
}

#[test]
fn oracle_betainc_scalar_a_tensor_b_scalar_x_broadcast() {
    // scalar a, tensor b, scalar x
    let a = scalar_f64(1.0);
    let b = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let x = scalar_f64(1.0);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // betainc(a, b, 1) = 1 for all a, b
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 1.0);
}

#[test]
fn oracle_betainc_singleton_a_vector_x_broadcast() {
    // [1] a with [1] b and [3] x -> [3]
    let a = make_f64_tensor(&[1], vec![1.0]);
    let b = make_f64_tensor(&[1], vec![1.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    // betainc(1, 1, x) = x
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - 0.5).abs() < 1e-14);
    assert_eq!(vals[2], 1.0);
}

#[test]
fn oracle_betainc_column_params_matrix_x_broadcast() {
    // [2, 1] a,b with [2, 3] x -> [2, 3]
    let a = make_f64_tensor(&[2, 1], vec![1.0, 1.0]);
    let b = make_f64_tensor(&[2, 1], vec![1.0, 1.0]);
    let x = make_f64_tensor(&[2, 3], vec![0.0, 0.5, 1.0, 0.25, 0.5, 0.75]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // betainc(1, 1, x) = x for all values
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - 0.5).abs() < 1e-14);
    assert_eq!(vals[2], 1.0);
    assert!((vals[3] - 0.25).abs() < 1e-14);
    assert!((vals[4] - 0.5).abs() < 1e-14);
    assert!((vals[5] - 0.75).abs() < 1e-14);
}

#[test]
fn oracle_betainc_different_ranks_broadcast() {
    // [3] params with [2, 3] x -> [2, 3]
    let a = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let b = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let x = make_f64_tensor(&[2, 3], vec![0.0, 0.5, 1.0, 0.25, 0.75, 0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: 0.0, 0.5, 1.0
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - 0.5).abs() < 1e-14);
    assert_eq!(vals[2], 1.0);
    // Row 1: 0.25, 0.75, 0.5
    assert!((vals[3] - 0.25).abs() < 1e-14);
    assert!((vals[4] - 0.75).abs() < 1e-14);
    assert!((vals[5] - 0.5).abs() < 1e-14);
}

#[test]
fn oracle_betainc_incompatible_shapes_error() {
    // [2] params with [3] x should error
    let a = make_f64_tensor(&[2], vec![1.0, 1.0]);
    let b = make_f64_tensor(&[2], vec![1.0, 1.0]);
    let x = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_betainc_scalar_params_vector_x_broadcast() {
    let a = scalar_f64(1.0);
    let b = scalar_f64(1.0);
    let x_values: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    let x = make_f64_tensor(&[5], x_values.to_vec());
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![5]);
    let actual = extract_f64_vec(&result);
    for (i, (&actual, &expected)) in actual.iter().zip(x_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-14,
            "betainc scalar-parameter broadcast element {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn oracle_betainc_preserves_dtype() {
    let a = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let b = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let x = make_f64_tensor(&[2], vec![0.5, 0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_betainc_empty_tensor() {
    let a = make_f64_tensor(&[0], vec![]);
    let b = make_f64_tensor(&[0], vec![]);
    let x = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}

#[test]
fn oracle_betainc_nan_propagates() {
    let a = make_f64_tensor(&[], vec![f64::NAN]);
    let b = make_f64_tensor(&[], vec![2.0]);
    let x = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "NaN should propagate");
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_betainc_3d() {
    let a = make_f64_tensor(&[2, 2, 2], vec![1.0; 8]);
    let b = make_f64_tensor(&[2, 2, 2], vec![1.0; 8]);
    let x = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.125, 0.375, 0.625]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

#[test]
fn oracle_betainc_nan_in_b() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![f64::NAN]);
    let x = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "NaN in b should propagate");
}

#[test]
fn oracle_betainc_nan_in_x() {
    let a = make_f64_tensor(&[], vec![2.0]);
    let b = make_f64_tensor(&[], vec![3.0]);
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "NaN in x should propagate");
}

#[test]
fn oracle_betainc_large_params() {
    let a = make_f64_tensor(&[], vec![100.0]);
    let b = make_f64_tensor(&[], vec![100.0]);
    let x = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 0.5).abs() < 0.01, "betainc(100, 100, 0.5) ~ 0.5");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_betainc_outputs_f64() {
    // Betainc always outputs F64 regardless of input dtype for numerical precision.
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
        Value::Tensor(
            TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap(),
        )
    }

    let a_values = [1.0_f64, 2.0, 3.0];
    let b_values = [2.0_f64, 3.0, 4.0];
    let x_values = [0.25_f64, 0.5, 0.75];
    for input_dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_vec(input_dtype, &a_values);
        let b = make_vec(input_dtype, &b_values);
        let x = make_vec(input_dtype, &x_values);
        let result = eval_primitive(Primitive::Betainc, &[a, b, x], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        // Betainc always promotes to F64 for precision
        assert_eq!(t.dtype, DType::F64, "betainc {input_dtype:?}->F64: output should be F64");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
