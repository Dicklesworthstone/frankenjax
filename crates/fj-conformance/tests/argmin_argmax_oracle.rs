//! Oracle tests for Argmin and Argmax primitives.
//!
//! These pin JAX/NumPy-compatible index-of-extremum behavior:
//! - explicit positive and negative axes
//! - first index wins for ties
//! - scalar inputs return index 0

#![allow(clippy::cloned_ref_to_slice_refs)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), axis.to_string());
    params
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

fn extract_i64_scalar(value: &Value) -> i64 {
    if let Value::Scalar(Literal::I64(v)) = value {
        *v
    } else {
        assert!(
            matches!(value, Value::Scalar(Literal::I64(_))),
            "expected scalar i64, got {value:?}"
        );
        0
    }
}

fn extract_i64_vec(value: &Value) -> Vec<i64> {
    if let Value::Tensor(tensor) = value {
        tensor
            .elements
            .iter()
            .map(|literal| {
                if let Some(index) = literal.as_i64() {
                    index
                } else {
                    assert!(
                        literal.as_i64().is_some(),
                        "expected i64 literal, got {literal:?}"
                    );
                    0
                }
            })
            .collect()
    } else {
        assert!(
            matches!(value, Value::Tensor(_)),
            "expected tensor i64, got {value:?}"
        );
        Vec::new()
    }
}

fn extract_shape(value: &Value) -> Vec<u32> {
    match value {
        Value::Tensor(tensor) => tensor.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

#[test]
fn oracle_argmin_1d_returns_first_minimum_index() {
    // JAX: jnp.argmin([3, 1, 4, 1, 5]) == 1
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_argmax_1d_returns_first_maximum_index() {
    // JAX: jnp.argmax([3, 5, 4, 5, 1]) == 1
    let input = make_f64_tensor(&[5], vec![3.0, 5.0, 4.0, 5.0, 1.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_argmax_argmin_first_nan_wins_sign_agnostic() {
    // JAX _ArgMinMaxReducer: the FIRST NaN (either sign) wins outright and is sticky
    // for BOTH argmax and argmin (a NaN candidate replaces via `v != v`), verified
    // vs JAX CPU. This DIFFERS from total_cmp, which ranks -NaN below -inf — so a
    // total_cmp argmax would MISS a -NaN (returning 2 below) and a total_cmp argmin
    // would wrongly pick a +NaN's neighbor. project_total_cmp_vs_jax_float_ordering.
    let pos_nan = f64::NAN;
    let neg_nan = -f64::NAN;
    assert!(neg_nan.is_sign_negative());

    // argmax picks the first NaN even when it is -NaN (total_cmp would give 2).
    let inp = make_f64_tensor(&[3], vec![1.0, neg_nan, 2.0]);
    let r = eval_primitive(Primitive::Argmax, &[inp], &axis_params(0)).unwrap();
    assert_eq!(
        extract_i64_scalar(&r),
        1,
        "argmax: first NaN wins, sign-agnostic"
    );

    // argmin also picks the first NaN, even when it is +NaN (total_cmp would give 0).
    let inp = make_f64_tensor(&[3], vec![1.0, pos_nan, 2.0]);
    let r = eval_primitive(Primitive::Argmin, &[inp], &axis_params(0)).unwrap();
    assert_eq!(
        extract_i64_scalar(&r),
        1,
        "argmin: first NaN wins, sign-agnostic"
    );

    // First NaN is sticky: an earlier NaN beats a later NaN and all finite values.
    let inp = make_f64_tensor(&[4], vec![5.0, pos_nan, neg_nan, 9.0]);
    let r = eval_primitive(Primitive::Argmax, &[inp], &axis_params(0)).unwrap();
    assert_eq!(
        extract_i64_scalar(&r),
        1,
        "first NaN is sticky over later NaN"
    );
}

#[test]
fn argmax_index_holds_the_reduce_max_value() {
    // Cross-validate Argmax against ReduceMax: the element at argmax's index must
    // equal the reduce_max value. Catches a disagreement between the two extremum
    // code paths (e.g. divergent float ordering / total_cmp). Oracle-free.
    // NOTE: argmax reads the "axis" param; reduce_max reads "axes".
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let input = make_f64_tensor(&[data.len() as u32], data.clone());
    let idx = extract_i64_scalar(
        &eval_primitive(Primitive::Argmax, &[input.clone()], &axis_params(0)).unwrap(),
    ) as usize;
    let reduce_axes = BTreeMap::from([("axes".to_string(), "0".to_string())]);
    let max_result = eval_primitive(Primitive::ReduceMax, &[input], &reduce_axes).unwrap();
    let max_val = match &max_result {
        Value::Scalar(lit) => lit.as_f64().unwrap(),
        Value::Tensor(t) => t.elements[0].as_f64().unwrap(),
    };
    assert!(
        (data[idx] - max_val).abs() < 1e-12,
        "x[argmax]={} must equal reduce_max={}",
        data[idx],
        max_val
    );
}

#[test]
fn argmin_index_holds_the_reduce_min_value() {
    // Cross-validate Argmin against ReduceMin (sibling of the argmax/reduce_max
    // check): the element at argmin's index must equal the reduce_min value. Catches
    // a disagreement between the two extremum paths. Oracle-free.
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let input = make_f64_tensor(&[data.len() as u32], data.clone());
    let idx = extract_i64_scalar(
        &eval_primitive(Primitive::Argmin, &[input.clone()], &axis_params(0)).unwrap(),
    ) as usize;
    let reduce_axes = BTreeMap::from([("axes".to_string(), "0".to_string())]);
    let min_result = eval_primitive(Primitive::ReduceMin, &[input], &reduce_axes).unwrap();
    let min_val = match &min_result {
        Value::Scalar(lit) => lit.as_f64().unwrap(),
        Value::Tensor(t) => t.elements[0].as_f64().unwrap(),
    };
    assert!(
        (data[idx] - min_val).abs() < 1e-12,
        "x[argmin]={} must equal reduce_min={}",
        data[idx],
        min_val
    );
}

#[test]
fn oracle_argmin_2d_axis0_reduces_rows() {
    // JAX: jnp.argmin([[1, 4, 2], [3, 0, 5]], axis=0) == [0, 1, 0]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 0]);
}

#[test]
fn oracle_argmax_2d_axis0_reduces_rows() {
    // JAX: jnp.argmax([[1, 4, 2], [3, 0, 5]], axis=0) == [1, 0, 1]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 1]);
}

#[test]
fn oracle_argmax_2d_axis1_reduces_columns() {
    // JAX: jnp.argmax([[1, 4, 2], [3, 0, 5]], axis=1) == [1, 2]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2]);
}

#[test]
fn oracle_argmax_negative_axis_matches_last_axis() {
    // JAX: axis=-1 addresses the last axis.
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2]);
}

#[test]
fn oracle_argmin_negative_axis_matches_first_axis() {
    // JAX: axis=-2 addresses axis 0 for a rank-2 value.
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(-2)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 0]);
}

#[test]
fn oracle_argmin_scalar_returns_zero() {
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(Primitive::Argmin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmax_scalar_returns_zero() {
    let input = Value::Scalar(Literal::from_f64(42.0));
    let result = eval_primitive(Primitive::Argmax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmin_axis_out_of_bounds_errors() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let err = eval_primitive(Primitive::Argmin, &[input], &axis_params(2)).unwrap_err();
    assert!(
        err.to_string().contains("axis 2 out of bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn oracle_argmax_negative_axis_out_of_bounds_errors() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
    let err = eval_primitive(Primitive::Argmax, &[input], &axis_params(-3)).unwrap_err();
    assert!(
        err.to_string().contains("axis -3 out of bounds"),
        "unexpected error: {err}"
    );
}

// ======================== Additional Coverage ========================

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

#[test]
fn oracle_argmin_negative_values() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2); // -4 at index 2
}

#[test]
fn oracle_argmax_negative_values() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // -1 at index 1
}

#[test]
fn oracle_argmin_all_equal_returns_first() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0); // first index
}

#[test]
fn oracle_argmax_all_equal_returns_first() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0); // first index
}

#[test]
fn oracle_argmin_integer_dtype() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // 10 at index 1
}

#[test]
fn oracle_argmax_integer_dtype() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2); // 40 at index 2
}

#[test]
fn oracle_argmin_3d_axis1() {
    // [2, 3, 2] tensor, reduce along axis 1
    let input = make_f64_tensor(
        &[2, 3, 2],
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
    );
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn oracle_argmax_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_argmin_inf_values() {
    let input = make_f64_tensor(&[4], vec![1.0, f64::NEG_INFINITY, 2.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1); // -inf at index 1
}

#[test]
fn oracle_argmax_inf_values() {
    let input = make_f64_tensor(&[4], vec![1.0, f64::NEG_INFINITY, 2.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 3); // +inf at index 3
}

#[test]
fn oracle_argmin_empty_error() {
    let input = make_f64_tensor(&[0], vec![]);
    let err = eval_primitive(Primitive::Argmin, &[input], &axis_params(0));
    assert!(err.is_err(), "argmin on empty tensor should error");
}

#[test]
fn oracle_argmax_large_tensor() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_scalar(&result), 99);
}

#[test]
fn oracle_argmin_output_dtype_is_i64() {
    let input = make_f64_tensor(&[3], vec![3.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0)).unwrap();
    assert!(
        matches!(result.dtype(), DType::I32 | DType::I64),
        "argmin should return integer type"
    );
}

// ======================== PROPERTY: output dtype is always I64 indices ========================

#[test]
fn property_argmin_argmax_output_i64_for_all_float_inputs() {
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
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![4] }, lits).unwrap())
    }
    let values = [1.0_f64, 3.0, 2.0, 4.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);

        let min_result =
            eval_primitive(Primitive::Argmin, &[input.clone()], &axis_params(0)).unwrap();
        assert!(
            matches!(min_result.dtype(), DType::I32 | DType::I64),
            "argmin on {dtype:?} should return integer indices, got {:?}",
            min_result.dtype()
        );

        let max_result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0)).unwrap();
        assert!(
            matches!(max_result.dtype(), DType::I32 | DType::I64),
            "argmax on {dtype:?} should return integer indices, got {:?}",
            max_result.dtype()
        );
    }
}

// ====================== COMPLEX DTYPE TESTS ======================

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

#[test]
fn oracle_argmin_complex64_lexicographic() {
    // Complex argmin orders lexicographically by (real, imag), like JAX/NumPy.
    // (1,2) and (1,1) tie on real=1; imag 1 < 2 picks index 1.
    let input = make_complex64_tensor(&[3], vec![(1.0, 2.0), (1.0, 1.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Argmin, &[input], &axis_params(0))
        .expect("argmin should work on complex64");
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_argmax_complex64_lexicographic() {
    // (2,1) and (2,3) tie on real=2; imag 3 > 1 picks index 2.
    let input = make_complex64_tensor(&[3], vec![(1.0, 5.0), (2.0, 1.0), (2.0, 3.0)]);
    let result = eval_primitive(Primitive::Argmax, &[input], &axis_params(0))
        .expect("argmax should work on complex64");
    assert_eq!(extract_i64_scalar(&result), 2);
}
