//! Oracle tests for ExpandDims primitive.
//!
//! Tests against expected behavior matching JAX/lax.expand_dims:
//! - axis: position to insert new dimension of size 1
//! - Preserves data, only changes shape

#![allow(clippy::approx_constant)]

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
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn expand_params(axis: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_expand_dims_scalar() {
    // JAX: lax.expand_dims(42, dimensions=(0,)) => shape [1]
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_scalar_f64() {
    let input = Value::Scalar(Literal::from_f64(3.17));
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.17).abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_expand_dims_1d_axis0() {
    // JAX: lax.expand_dims(jnp.array([1, 2, 3]), dimensions=(0,)) => shape [1, 3]
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_expand_dims_1d_axis1() {
    // JAX: lax.expand_dims(jnp.array([1, 2, 3]), dimensions=(1,)) => shape [3, 1]
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_expand_dims_1d_single_element() {
    // [1] -> [1, 1] at axis 0
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_1d_large() {
    let input = make_i64_tensor(&[10], (1..=10).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 10]);
    assert_eq!(extract_i64_vec(&result), (1..=10).collect::<Vec<_>>());
}

// ======================== 2D Tests ========================

#[test]
fn oracle_expand_dims_2d_axis0() {
    // [2, 3] -> [1, 2, 3]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_axis1() {
    // [2, 3] -> [2, 1, 3]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_axis2() {
    // [2, 3] -> [2, 3, 1]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_negative_axis() {
    use std::collections::BTreeMap;
    // Negative axes normalize against the OUTPUT rank (numpy/jnp expand_dims):
    // [2,3] axis=-1 -> [2,3,1]; axis=-3 -> [1,2,3]. (Was rejected: usize parse.)
    let mut p_neg1 = BTreeMap::new();
    p_neg1.insert("axis".to_string(), "-1".to_string());
    let r1 = eval_primitive(
        Primitive::ExpandDims,
        &[make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6])],
        &p_neg1,
    )
    .unwrap();
    assert_eq!(extract_shape(&r1), vec![2, 3, 1]);
    assert_eq!(extract_i64_vec(&r1), vec![1, 2, 3, 4, 5, 6]);

    let mut p_neg3 = BTreeMap::new();
    p_neg3.insert("axis".to_string(), "-3".to_string());
    let r2 = eval_primitive(
        Primitive::ExpandDims,
        &[make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6])],
        &p_neg3,
    )
    .unwrap();
    assert_eq!(extract_shape(&r2), vec![1, 2, 3]);
}

#[test]
fn oracle_expand_dims_2d_square() {
    // [3, 3] -> [1, 3, 3]
    let input = make_i64_tensor(&[3, 3], (1..=9).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 3]);
}

#[test]
fn oracle_expand_dims_2d_single_row() {
    // [1, 5] -> [1, 1, 5]
    let input = make_i64_tensor(&[1, 5], (1..=5).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 5]);
}

#[test]
fn oracle_expand_dims_2d_single_col() {
    // [5, 1] -> [5, 1, 1]
    let input = make_i64_tensor(&[5, 1], (1..=5).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1, 1]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_expand_dims_3d_axis0() {
    // [2, 3, 4] -> [1, 2, 3, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3, 4]);
    assert_eq!(extract_i64_vec(&result).len(), 24);
}

#[test]
fn oracle_expand_dims_3d_axis1() {
    // [2, 3, 4] -> [2, 1, 3, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3, 4]);
}

#[test]
fn oracle_expand_dims_3d_axis2() {
    // [2, 3, 4] -> [2, 3, 1, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 1, 4]);
}

#[test]
fn oracle_expand_dims_3d_axis3() {
    // [2, 3, 4] -> [2, 3, 4, 1]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4, 1]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_expand_dims_f64_1d() {
    let input = make_f64_tensor(&[3], vec![1.1, 2.2, 3.3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

#[test]
fn oracle_expand_dims_f64_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_expand_dims_with_negatives() {
    let input = make_i64_tensor(&[3], vec![-5, 0, 5]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, 0, 5]);
}

#[test]
fn oracle_expand_dims_preserves_data_order() {
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

#[test]
fn oracle_expand_dims_unit_tensor() {
    // [1, 1] -> [1, 1, 1]
    let input = make_i64_tensor(&[1, 1], vec![42]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_4d() {
    // [2, 2, 2, 2] -> [1, 2, 2, 2, 2]
    let input = make_i64_tensor(&[2, 2, 2, 2], (1..=16).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), (1..=16).collect::<Vec<_>>());
}

// ======================== Compose with Squeeze ========================

#[test]
fn oracle_expand_dims_squeeze_identity() {
    // ExpandDims then Squeeze should give back original
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let expanded = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&expanded), vec![1, 2, 3]);

    let mut squeeze_params = BTreeMap::new();
    squeeze_params.insert("dimensions".to_string(), "0".to_string());
    let squeezed = eval_primitive(Primitive::Squeeze, &[expanded], &squeeze_params).unwrap();
    assert_eq!(extract_shape(&squeezed), vec![2, 3]);
    assert_eq!(extract_i64_vec(&squeezed), vec![1, 2, 3, 4, 5, 6]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_expand_dims_empty() {
    let input = make_i64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0]);
}

#[test]
fn oracle_expand_dims_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0, 3]);
}

fn make_bool_tensor(shape: &[u32], data: Vec<bool>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool"),
            })
            .collect(),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_expand_dims_bool_dtype() {
    let input = make_bool_tensor(&[3], vec![true, false, true]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true]);
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn oracle_expand_dims_special_values() {
    let input = make_f64_tensor(&[4], vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 1]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
    assert_eq!(vals[1], f64::INFINITY);
    assert_eq!(vals[2], f64::NEG_INFINITY);
    assert_eq!(vals[3], 0.0);
}

#[test]
fn oracle_expand_dims_preserves_dtype_i64() {
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(result.dtype(), DType::I64);
}

#[test]
fn oracle_expand_dims_preserves_dtype_f64() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_expand_dims_preserves_all_float_dtypes() {
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

    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "expand_dims {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 Tests ========================

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

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        Value::Scalar(l) => vec![l.as_complex64().unwrap()],
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        Value::Scalar(l) => vec![l.as_complex128().unwrap()],
    }
}

#[test]
fn oracle_expand_dims_complex64_scalar() {
    let input = Value::Scalar(Literal::from_complex64(1.0, 2.0));
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 2.0)]);
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_expand_dims_complex128_scalar() {
    let input = Value::Scalar(Literal::from_complex128(3.0, 4.0));
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_complex128_vec(&result);
    assert_eq!(vals, vec![(3.0, 4.0)]);
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_expand_dims_complex64_1d_axis0() {
    // [3] -> [1, 3]
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
}

#[test]
fn oracle_expand_dims_complex64_1d_axis1() {
    // [3] -> [3, 1]
    let input = make_complex64_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 1]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
}

#[test]
fn oracle_expand_dims_complex64_2d_axis0() {
    // [2, 2] -> [1, 2, 2]
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
    );
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]);
}

#[test]
fn oracle_expand_dims_complex64_2d_axis1() {
    // [2, 2] -> [2, 1, 2]
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
    );
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
}

#[test]
fn oracle_expand_dims_complex64_2d_axis2() {
    // [2, 2] -> [2, 2, 1]
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
    );
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 1]);
}

#[test]
fn oracle_expand_dims_complex128_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    assert_eq!(result.dtype(), DType::Complex128);
    let vals = extract_complex128_vec(&result);
    assert_eq!(vals, vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
}

#[test]
fn oracle_expand_dims_complex128_2d() {
    let input = make_complex128_tensor(
        &[2, 2],
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
    );
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_expand_dims_complex64_preserves_dtype() {
    let input = make_complex64_tensor(&[2], vec![(1.0, 0.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_expand_dims_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(1.0, 0.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_expand_dims_complex64_preserves_data_order() {
    let input = make_complex64_tensor(&[4], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 1]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
}

#[test]
fn oracle_expand_dims_complex64_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::Complex64, Shape { dims: vec![0] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 0]);
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_expand_dims_complex64_unit_tensor() {
    let input = make_complex64_tensor(&[1, 1], vec![(42.0, -42.0)]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(42.0, -42.0)]);
}

#[test]
fn oracle_expand_dims_complex64_squeeze_identity() {
    let input = make_complex64_tensor(
        &[2, 3],
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ],
    );
    let expanded = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&expanded), vec![1, 2, 3]);

    let mut squeeze_params = BTreeMap::new();
    squeeze_params.insert("dimensions".to_string(), "0".to_string());
    let squeezed = eval_primitive(Primitive::Squeeze, &[expanded], &squeeze_params).unwrap();
    assert_eq!(extract_shape(&squeezed), vec![2, 3]);
    assert_eq!(squeezed.dtype(), DType::Complex64);
}

#[test]
fn property_expand_dims_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let lits: Vec<Literal> = match dtype {
            DType::Complex64 => vec![
                Literal::from_complex64(1.0, 2.0),
                Literal::from_complex64(3.0, 4.0),
                Literal::from_complex64(5.0, 6.0),
            ],
            DType::Complex128 => vec![
                Literal::from_complex128(1.0, 2.0),
                Literal::from_complex128(3.0, 4.0),
                Literal::from_complex128(5.0, 6.0),
            ],
            _ => unreachable!(),
        };
        let input = Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap());
        let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "expand_dims {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
