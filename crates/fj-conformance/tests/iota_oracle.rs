//! Oracle tests for Iota primitive.
//!
//! Tests against expected behavior matching JAX/lax.iota:
//! - Creates 1D tensor with incrementing values [0, 1, 2, ..., length-1]
//! - Supports multiple dtypes

use fj_core::{DType, Primitive, Value};
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
    for (i, value) in vals.iter().enumerate() {
        assert_eq!(*value, i as i64);
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

// Property sweep: every dtype that Iota supports (everything except
// Bool) round-trips through `eval_iota` + `literal_from_index_for_dtype`
// with declared dtype and element kinds in agreement. Bool is rejected
// per JAX semantics. Pins the dispatch helper against per-dtype
// regressions a single-dtype point test would miss.
#[test]
fn property_iota_preserves_all_supported_dtypes() {
    let cases: &[(&str, DType)] = &[
        ("i32", DType::I32),
        ("i64", DType::I64),
        ("u32", DType::U32),
        ("u64", DType::U64),
        ("bf16", DType::BF16),
        ("f16", DType::F16),
        ("f32", DType::F32),
        ("f64", DType::F64),
        ("complex64", DType::Complex64),
        ("complex128", DType::Complex128),
    ];
    for (token, expected_dtype) in cases {
        let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, token))
            .unwrap_or_else(|e| panic!("iota dtype={token} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("iota dtype={token}: expected tensor");
        };
        assert_eq!(
            t.dtype, *expected_dtype,
            "iota dtype={token}: declared dtype"
        );
        assert_eq!(t.shape.dims, vec![5]);
        t.validate_dtype_consistency().unwrap_or_else(|e| {
            panic!("iota dtype={token}: validate_dtype_consistency failed: {e}")
        });
    }
    // bool dtype must be rejected (JAX `lax.iota` does not accept Bool).
    assert!(
        eval_primitive(Primitive::Iota, &[], &iota_params(3, "bool")).is_err(),
        "iota with bool dtype must error"
    );
}

#[test]
fn oracle_iota_u32() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(4, "U32")).unwrap();
    assert_eq!(extract_dtype(&result), DType::U32);
    assert_eq!(extract_shape(&result), vec![4]);
    let vals: Vec<u64> = result
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_u64().unwrap())
        .collect();
    assert_eq!(vals, vec![0, 1, 2, 3]);
}

#[test]
fn oracle_iota_rejects_invalid_dtype() {
    let err = eval_primitive(Primitive::Iota, &[], &iota_params(5, "invalid"))
        .expect_err("invalid dtype should fail");
    assert!(
        err.to_string().contains("unsupported")
            || err.to_string().contains("dtype")
            || err.to_string().contains("unknown"),
        "unexpected error: {err}"
    );
}

#[test]
fn oracle_iota_values_are_contiguous() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(10, "I64")).unwrap();
    let vals = extract_i64_vec(&result);
    for i in 0..10 {
        assert_eq!(vals[i], i as i64, "iota value at index {i}");
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_iota_i64_large() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(1000, "I64")).unwrap();
    assert_eq!(extract_shape(&result), vec![1000]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[999], 999);
}

#[test]
fn oracle_iota_bf16() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "BF16")).unwrap();
    assert_eq!(extract_dtype(&result), DType::BF16);
    assert_eq!(extract_shape(&result), vec![5]);
}

#[test]
fn oracle_iota_f16() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "F16")).unwrap();
    assert_eq!(extract_dtype(&result), DType::F16);
    assert_eq!(extract_shape(&result), vec![5]);
}

#[test]
fn oracle_iota_u64() {
    let result = eval_primitive(Primitive::Iota, &[], &iota_params(5, "U64")).unwrap();
    assert_eq!(extract_dtype(&result), DType::U64);
    let vals: Vec<u64> = result
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_u64().unwrap())
        .collect();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_iota_output_shape_is_1d() {
    // Verify iota always produces 1D output
    for len in [1, 5, 10, 100] {
        let result = eval_primitive(Primitive::Iota, &[], &iota_params(len, "I64")).unwrap();
        let shape = extract_shape(&result);
        assert_eq!(shape.len(), 1, "iota should produce 1D tensor");
        assert_eq!(shape[0], len);
    }
}
