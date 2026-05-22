//! Oracle tests for BroadcastedIota primitive.
//!
//! Tests against expected behavior matching JAX/lax.broadcasted_iota:
//! - shape: output shape
//! - dimension: axis along which to broadcast indices
//! - No inputs, output is indices broadcasted across shape

use fj_core::{DType, Primitive, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn iota_params(shape: &[usize], dimension: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "shape".to_string(),
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert("dimension".to_string(), dimension.to_string());
    p
}

// ======================== 1D Tests ========================

#[test]
fn oracle_broadcasted_iota_1d() {
    // shape [5], dimension 0 -> [0, 1, 2, 3, 4]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[5], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_broadcasted_iota_1d_single() {
    // shape [1], dimension 0 -> [0]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_broadcasted_iota_2d_dim0() {
    // shape [3, 4], dimension 0 -> row indices broadcast across columns
    // [[0,0,0,0], [1,1,1,1], [2,2,2,2]]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 4], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
    let vals = extract_i64_vec(&result);
    // Row 0: all 0s
    assert_eq!(&vals[0..4], &[0, 0, 0, 0]);
    // Row 1: all 1s
    assert_eq!(&vals[4..8], &[1, 1, 1, 1]);
    // Row 2: all 2s
    assert_eq!(&vals[8..12], &[2, 2, 2, 2]);
}

#[test]
fn oracle_broadcasted_iota_2d_dim1() {
    // shape [3, 4], dimension 1 -> column indices broadcast across rows
    // [[0,1,2,3], [0,1,2,3], [0,1,2,3]]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 4], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
    let vals = extract_i64_vec(&result);
    // Each row has [0, 1, 2, 3]
    assert_eq!(&vals[0..4], &[0, 1, 2, 3]);
    assert_eq!(&vals[4..8], &[0, 1, 2, 3]);
    assert_eq!(&vals[8..12], &[0, 1, 2, 3]);
}

#[test]
fn oracle_broadcasted_iota_2d_square() {
    // shape [3, 3], dimension 0
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 3], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals, vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
}

#[test]
fn oracle_broadcasted_iota_2d_square_dim1() {
    // shape [3, 3], dimension 1
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 3], 1)).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_broadcasted_iota_3d_dim0() {
    // shape [2, 3, 4], dimension 0 -> indices [0, 0, ..., 1, 1, ...]
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // First half (12 elements): all 0s
    assert!(vals[0..12].iter().all(|&x| x == 0));
    // Second half (12 elements): all 1s
    assert!(vals[12..24].iter().all(|&x| x == 1));
}

#[test]
fn oracle_broadcasted_iota_3d_dim1() {
    // shape [2, 3, 4], dimension 1 -> row indices within each slice
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Within each 2D slice (12 elements), we have rows 0, 1, 2
    // First slice
    assert_eq!(&vals[0..4], &[0, 0, 0, 0]); // row 0
    assert_eq!(&vals[4..8], &[1, 1, 1, 1]); // row 1
    assert_eq!(&vals[8..12], &[2, 2, 2, 2]); // row 2
}

#[test]
fn oracle_broadcasted_iota_3d_dim2() {
    // shape [2, 3, 4], dimension 2 -> column indices
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Each row has [0, 1, 2, 3]
    for i in 0..6 {
        assert_eq!(&vals[i * 4..(i + 1) * 4], &[0, 1, 2, 3]);
    }
}

// ======================== Edge Cases ========================

#[test]
fn oracle_broadcasted_iota_unit_dims() {
    // shape [1, 5], dimension 1
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1, 5], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_broadcasted_iota_unit_dim_0() {
    // shape [1, 5], dimension 0 -> all zeros
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1, 5], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0, 0]);
}

#[test]
fn oracle_broadcasted_iota_tall() {
    // shape [5, 1], dimension 0
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[5, 1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

// Property sweep across every BroadcastedIota-supported dtype. Bool is
// rejected (mirrors the Iota path). Pins the dtype dispatch dispatch
// helper against per-arm regressions.
#[test]
fn property_broadcasted_iota_preserves_all_supported_dtypes() {
    fn iota_params_with_dtype(
        shape: &[usize],
        dimension: usize,
        dtype: &str,
    ) -> BTreeMap<String, String> {
        let mut p = iota_params(shape, dimension);
        p.insert("dtype".to_string(), dtype.to_string());
        p
    }

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
        let result = eval_primitive(
            Primitive::BroadcastedIota,
            &[],
            &iota_params_with_dtype(&[3, 2], 0, token),
        )
        .unwrap_or_else(|e| panic!("broadcasted_iota dtype={token} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("broadcasted_iota dtype={token}: expected tensor");
        };
        assert_eq!(
            t.dtype, *expected_dtype,
            "broadcasted_iota dtype={token}: declared dtype"
        );
        assert_eq!(t.shape.dims, vec![3, 2]);
        t.validate_dtype_consistency().unwrap_or_else(|e| {
            panic!(
                "broadcasted_iota dtype={token}: validate_dtype_consistency failed: {e}"
            )
        });
    }
    // Bool must be rejected (no integer ramp semantics for booleans).
    assert!(
        eval_primitive(
            Primitive::BroadcastedIota,
            &[],
            &iota_params_with_dtype(&[3], 0, "bool"),
        )
        .is_err(),
        "broadcasted_iota with bool dtype must error"
    );
}

#[test]
fn oracle_broadcasted_iota_rejects_dim_out_of_range() {
    let err = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 4], 5))
        .expect_err("dimension out of range should fail");
    assert!(
        err.to_string().contains("dimension")
            || err.to_string().contains("out of")
            || err.to_string().contains("range")
            || err.to_string().contains("rank"),
        "unexpected error: {err}"
    );
}

#[test]
fn oracle_broadcasted_iota_large_shape() {
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[100, 1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![100, 1]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[99], 99);
}

#[test]
fn oracle_broadcasted_iota_4d() {
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 2, 2, 2], 3)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
    let vals = extract_i64_vec(&result);
    // dimension 3 (innermost) should alternate [0, 1, 0, 1, ...]
    for chunk in vals.chunks(2) {
        assert_eq!(chunk, &[0, 1]);
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_broadcasted_iota_default_dtype_is_i64() {
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 2], 0)).unwrap();
    assert_eq!(result.dtype(), DType::I64);
}

#[test]
fn oracle_broadcasted_iota_empty_dim() {
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[0, 3], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
    assert!(extract_i64_vec(&result).is_empty());
}

#[test]
fn oracle_broadcasted_iota_empty_other_dim() {
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 0], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 0]);
    assert!(extract_i64_vec(&result).is_empty());
}

#[test]
fn oracle_broadcasted_iota_single_element_shape() {
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1, 1, 1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}
