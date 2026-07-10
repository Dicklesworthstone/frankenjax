//! Oracle tests for Tile primitive.
//!
//! tile(x, reps) repeats the input array according to reps

#![allow(clippy::cloned_ref_to_slice_refs)]
//!
//! For example:
//! - tile([1, 2], [2]) = [1, 2, 1, 2]
//! - tile([[1, 2]], [2, 3]) = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
//!
//! Tests:
//! - 1D tiling
//! - 2D tiling
//! - Identity (reps = [1, 1, ...])
//! - Expanding dimensions

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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

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

fn tile_params(reps: &[i64]) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    if !reps.is_empty() {
        params.insert(
            "reps".to_string(),
            reps.iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    } else {
        params.insert("reps".to_string(), "1".to_string());
    }
    params
}

// ======================== 1D Tiling ========================

#[test]
fn oracle_tile_1d_repeat_2() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn oracle_tile_1d_repeat_3() {
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn oracle_tile_1d_identity() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
}

// ======================== 2D Tiling ========================

#[test]
fn oracle_tile_2d_row_only() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn oracle_tile_2d_col_only() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );
}

#[test]
fn oracle_tile_2d_both() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 4]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 16);
    // First row of first block
    assert_eq!(vals[0..4], [1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn oracle_tile_2d_identity() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ======================== Integer Types ========================

#[test]
fn oracle_tile_i64() {
    let input = make_i64_tensor(&[2], vec![10, 20]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![10, 20, 10, 20, 10, 20]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_tile_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![42.0, 42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn oracle_tile_promotes_rank_when_reps_longer_than_input() {
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn oracle_tile_left_pads_reps_when_shorter_than_input() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );
}

#[test]
fn oracle_tile_scalar_promotes_to_array() {
    let input = Value::scalar_f64(5.0);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![5.0; 6]);
}

#[test]
fn oracle_tile_zero_repeat_returns_empty_axis() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[0, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 4]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_tile_preserves_dtype() {
    let input = make_i64_tensor(&[2], vec![1, 2]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_tile_large_reps() {
    let input = make_f64_tensor(&[1], vec![7.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[100])).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|&v| (v - 7.0).abs() < 1e-10));
}

#[test]
fn oracle_tile_3d() {
    let input = make_f64_tensor(&[1, 1, 2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    // All 4 2-element blocks should be [1, 2]
    assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn oracle_tile_negative_values() {
    let input = make_f64_tensor(&[3], vec![-1.0, -2.0, -3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![-1.0, -2.0, -3.0, -1.0, -2.0, -3.0]
    );
}

#[test]
fn oracle_tile_metamorphic_product() {
    // tile(x, [a]) then tile(result, [b]) should equal tile(x, [a*b])
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let once = eval_primitive(Primitive::Tile, &[input.clone()], &tile_params(&[6])).unwrap();
    let twice = {
        let step1 = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
        eval_primitive(Primitive::Tile, &[step1], &tile_params(&[3])).unwrap()
    };
    assert_eq!(extract_shape(&once), extract_shape(&twice));
    assert_eq!(extract_f64_vec(&once), extract_f64_vec(&twice));
}

#[test]
fn oracle_tile_empty_input() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}

#[test]
fn oracle_tile_4d() {
    let input = make_f64_tensor(&[1, 1, 1, 2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2, 2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 16);
    assert!(
        vals.iter()
            .all(|&v| (v - 1.0).abs() < 1e-10 || (v - 2.0).abs() < 1e-10)
    );
}

#[test]
fn oracle_tile_special_values() {
    let input = make_f64_tensor(&[3], vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
    assert!(vals[1].is_infinite() && vals[1] > 0.0);
    assert!(vals[2].is_infinite() && vals[2] < 0.0);
}

#[test]
fn oracle_tile_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 6]);
}

#[test]
fn oracle_tile_bool_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape { dims: vec![2] },
            vec![Literal::Bool(true), Literal::Bool(false)],
        )
        .unwrap(),
    );
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Bool);
            let vals: Vec<bool> = t
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Bool(b) => *b,
                    _ => panic!("expected bool"),
                })
                .collect();
            assert_eq!(vals, vec![true, false, true, false, true, false]);
        }
        _ => panic!("expected tensor"),
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_tile_preserves_all_float_dtypes() {
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
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![2] }, lits).unwrap())
    }

    let values = [1.0_f64, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "tile {dtype:?}: dtype mismatch");
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

#[test]
fn oracle_tile_complex64_1d() {
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
        ]
    );
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_tile_complex64_2d_rows() {
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
    );
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
        ]
    );
}

#[test]
fn oracle_tile_complex64_2d_cols() {
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
    );
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 6]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
        ]
    );
}

#[test]
fn oracle_tile_complex64_2d_both() {
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)],
    );
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 4]);
}

#[test]
fn oracle_tile_complex128_1d() {
    let input = make_complex128_tensor(&[2], vec![(1.0, 2.0), (3.0, 4.0)]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    let vals = extract_complex128_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 2.0),
            (3.0, 4.0),
            (1.0, 2.0),
            (3.0, 4.0),
            (1.0, 2.0),
            (3.0, 4.0),
        ]
    );
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn oracle_tile_complex64_identity() {
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
}

#[test]
fn oracle_tile_complex64_preserves_dtype() {
    let input = make_complex64_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_tile_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(1.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn property_tile_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let lits: Vec<Literal> = match dtype {
            DType::Complex64 => vec![
                Literal::from_complex64(1.0, 2.0),
                Literal::from_complex64(3.0, 4.0),
            ],
            DType::Complex128 => vec![
                Literal::from_complex128(1.0, 2.0),
                Literal::from_complex128(3.0, 4.0),
            ],
            _ => unreachable!(),
        };
        let input = Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![2] }, lits).unwrap());
        let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "tile {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
