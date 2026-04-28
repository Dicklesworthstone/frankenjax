//! Oracle tests for Iota primitive.
//!
//! Tests against expected behavior matching JAX/lax.iota:
//! - Creates 1D tensor with incrementing values [0, 1, 2, ..., length-1]
//! - Supports multiple dtypes

use fj_core::{DType, Primitive, Shape, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn iota_params(length: u32, dtype: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("length".to_string(), length.to_string());
    p.insert("dtype".to_string(), dtype.to_string());
    p
}

fn iota_params_default(length: u32) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("length".to_string(), length.to_string());
    p
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

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

fn extract_dtype(v: &Value) -> DType {
    match v {
        Value::Tensor(t) => t.dtype,
        _ => panic!("expected tensor"),
    }
}

// ======================== Basic Iota Tests ========================

#[test]
fn oracle_iota_i64_5() {
    // JAX: lax.iota(jnp.int64, 5) => [0, 1, 2, 3, 4]
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "I64")).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_dtype(&result), DType::I64);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_iota_i32_5() {
    // JAX: lax.iota(jnp.int32, 5) => [0, 1, 2, 3, 4]
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "I32")).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_dtype(&result), DType::I32);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_iota_f64_5() {
    // JAX: lax.iota(jnp.float64, 5) => [0.0, 1.0, 2.0, 3.0, 4.0]
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "F64")).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_dtype(&result), DType::F64);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
    assert!((vals[2] - 2.0).abs() < 1e-10);
    assert!((vals[3] - 3.0).abs() < 1e-10);
    assert!((vals[4] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_iota_f32_5() {
    // JAX: lax.iota(jnp.float32, 5) => [0.0, 1.0, 2.0, 3.0, 4.0]
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "F32")).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_dtype(&result), DType::F32);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[4] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_iota_default_dtype() {
    // Default dtype should be I64
    let result = eval_primitive(Primitive::Iota, &[], &iota_params_default(3)).unwrap();
    assert_eq!(extract_dtype(&result), DType::I64);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_iota_length_1() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(1, "I64")).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_iota_length_0() {
    // Empty iota
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(0, "I64")).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

#[test]
fn oracle_iota_large() {
    // Larger iota
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(100, "I64")).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[99], 99);
    for i in 0..100 {
        assert_eq!(vals[i], i as i64);
    }
}

#[test]
fn oracle_iota_lowercase_dtype() {
    // Test lowercase dtype
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(3, "i64")).unwrap();
    assert_eq!(extract_dtype(&result), DType::I64);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2]);
}

#[test]
fn oracle_iota_f64_lowercase() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(4, "f64")).unwrap();
    assert_eq!(extract_dtype(&result), DType::F64);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 4);
}
