//! Oracle tests for TopK primitive.
//!
//! top_k(x, k) returns the k largest elements and their indices
//!
//! Tests:
//! - Basic: top_k([3, 1, 4, 1, 5], 2) = ([5, 4], [4, 2])
//! - Full k: top_k(x, len(x)) returns sorted x
//! - k=1: returns max
//! - Negative values
//! - Tensor shapes (batched)

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
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
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_f64_bits(v: &Value) -> Vec<u64> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| {
            let Literal::F64Bits(bits) = literal else {
                return None;
            };
            Some(*bits)
        })
        .collect::<Option<Vec<_>>>()
        .expect("expected f64 literals")
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect()
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => Vec::new(),
    }
}

fn topk_params(k: usize) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("k".to_string(), k.to_string());
    params
}

// ======================== Basic Cases ========================

#[test]
fn oracle_topk_basic() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    assert!(vals.contains(&5.0) && vals.contains(&4.0));
}

#[test]
fn oracle_topk_k1() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(1)).unwrap();

    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_f64_vec(&result), vec![5.0]);
}

#[test]
fn oracle_topk_full() {
    let input = make_f64_tensor(&[4], vec![3.0, 1.0, 4.0, 2.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(4)).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 2.0, 1.0]);
}

// ======================== Negative Values ========================

#[test]
fn oracle_topk_negative() {
    let input = make_f64_tensor(&[5], vec![-3.0, -1.0, -4.0, -1.0, -5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|&v| v >= -3.0));
}

// ======================== Integer Types ========================

#[test]
fn oracle_topk_i64() {
    let input = make_i64_tensor(&[5], vec![30, 10, 40, 10, 50]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_i64_vec(&result);
    assert!(vals.contains(&50) && vals.contains(&40));
}

// ======================== 2D (Batched) ========================

#[test]
fn oracle_topk_2d() {
    // [[3, 1, 4], [1, 5, 9]] -> top 2 per row
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 9.0, 5.0]);
}

// ======================== Multi-output Values and Indices ========================

#[test]
fn oracle_topk_multi_output_values_and_indices_1d() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![2]);
    assert_eq!(extract_shape(&result[1]), vec![2]);
    assert_eq!(extract_f64_vec(&result[0]), vec![5.0, 4.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![4, 2]);
}

#[test]
fn oracle_topk_multi_output_indices_are_per_last_axis_slice() {
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![2, 2]);
    assert_eq!(extract_shape(&result[1]), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result[0]), vec![4.0, 3.0, 9.0, 5.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![2, 0, 2, 1]);
}

#[test]
fn oracle_topk_multi_output_nan_sorts_above_finite_values() {
    // JAX 0.10.1: lax.top_k(jnp.array([5.0, nan, 4.0]), 3)
    // yields values [nan, 5.0, 4.0] and indices [1, 0, 2].
    let input = make_f64_tensor(&[3], vec![5.0, f64::NAN, 4.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(3)).unwrap();

    let values = extract_f64_vec(&result[0]);
    assert!(values[0].is_nan(), "NaN should sort before finite values");
    assert_eq!(values[1..], [5.0, 4.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![1, 0, 2]);
}

#[test]
fn oracle_topk_multi_output_negative_nan_sorts_below_finite_values() {
    // Distinguishes top_k from sort (project_total_cmp_vs_jax_float_ordering): top_k
    // keys by total order, where -NaN is the MINIMUM (total_cmp: -NaN < -inf) — so a
    // -NaN sorts LAST in descending top_k, the opposite of +NaN (which is the max,
    // tested above). sort instead sends EVERY NaN to the end regardless of sign.
    let neg_nan = -f64::NAN;
    assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());
    let input = make_f64_tensor(&[3], vec![5.0, neg_nan, 4.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(3)).unwrap();

    let values = extract_f64_vec(&result[0]);
    assert_eq!(values[..2], [5.0, 4.0], "finite values rank above -NaN");
    assert!(
        values[2].is_nan(),
        "-NaN is the minimum -> sorts last in top_k"
    );
    assert_eq!(extract_i64_vec(&result[1]), vec![0, 2, 1]);
}

#[test]
fn oracle_topk_multi_output_positive_zero_sorts_above_negative_zero() {
    // JAX 0.10.1 total-order behavior puts +0.0 before -0.0 for top_k.
    let input = make_f64_tensor(&[3], vec![-0.0, 0.0, 1.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(3)).unwrap();

    assert_eq!(
        extract_f64_bits(&result[0]),
        vec![1.0_f64.to_bits(), 0.0_f64.to_bits(), (-0.0_f64).to_bits()]
    );
    assert_eq!(extract_i64_vec(&result[1]), vec![2, 1, 0]);
}

#[test]
fn oracle_topk_multi_output_equal_nonzero_ties_keep_lower_index_first() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 4.0, 4.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(3)).unwrap();

    assert_eq!(extract_f64_vec(&result[0]), vec![5.0, 5.0, 4.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![0, 1, 2]);
}

#[test]
fn oracle_topk_multi_output_allows_zero_k() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(0)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![0]);
    assert_eq!(extract_shape(&result[1]), vec![0]);
    assert!(extract_f64_vec(&result[0]).is_empty());
    assert!(extract_i64_vec(&result[1]).is_empty());
}

#[test]
fn oracle_topk_multi_output_rejects_k_larger_than_axis() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let err = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(4))
        .expect_err("k larger than axis size should fail");

    assert!(
        err.to_string().contains("exceeds axis size"),
        "unexpected oversized-k error: {err}",
    );
}

#[test]
fn oracle_topk_rejects_scalar_operand() {
    let err = eval_primitive(Primitive::TopK, &[Value::scalar_f64(42.0)], &topk_params(1))
        .expect_err("scalar operand should fail");

    assert!(
        err.to_string().contains(">= 1 dimension") || err.to_string().contains("ndim"),
        "unexpected scalar error: {err}",
    );
}

#[test]
fn oracle_topk_rejects_0d_tensor() {
    let input = make_f64_tensor(&[], vec![42.0]);
    let err = eval_primitive(Primitive::TopK, &[input], &topk_params(1))
        .expect_err("0-d tensor should fail");

    assert!(
        err.to_string().contains(">= 1 dimension") || err.to_string().contains("ndim"),
        "unexpected 0-d tensor error: {err}",
    );
}

#[test]
fn oracle_topk_3d_batched() {
    // [2, 2, 3] -> top 2 per innermost axis
    let input = make_f64_tensor(
        &[2, 2, 3],
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0],
    );
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

#[test]
fn oracle_topk_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_f64_vec(&result), vec![42.0]);
}

#[test]
fn oracle_topk_already_sorted_descending() {
    let input = make_f64_tensor(&[5], vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(3)).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![5.0, 4.0, 3.0]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_topk_preserves_dtype() {
    let input = make_f64_tensor(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_topk_large_tensor() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![99.0, 98.0, 97.0]);
}

#[test]
fn oracle_topk_inf_values() {
    let input = make_f64_tensor(&[4], vec![1.0, f64::INFINITY, 2.0, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
    assert_eq!(vals[1], 2.0);
}

#[test]
fn oracle_topk_2d_batch_empty() {
    // Empty batch dimension - [0, 5] input with k=2 should yield [0, 2]
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 5] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 2]);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_topk_preserves_dtype() {
    let input = make_f64_tensor(&[5], vec![1.0, 5.0, 3.0, 4.0, 2.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(3)).unwrap();
    assert_eq!(result.dtype(), DType::F64, "topk should preserve F64 dtype");
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
fn oracle_topk_complex64_lexicographic() {
    // top_k orders complex lexicographically by (real, imag), descending, like
    // JAX/NumPy. For [1+0j, 4+0j, 2+0j, 3+0j] the top-2 are 4+0j (idx 1) and
    // 3+0j (idx 3).
    let input = make_complex64_tensor(&[4], vec![(1.0, 0.0), (4.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(2))
        .expect("topk should work on complex64");
    let Value::Tensor(values) = &result[0] else {
        panic!("expected tensor values");
    };
    let got_vals: Vec<(f32, f32)> = values
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect();
    assert_eq!(got_vals, vec![(4.0, 0.0), (3.0, 0.0)]);
    let Value::Tensor(indices) = &result[1] else {
        panic!("expected tensor indices");
    };
    let got_idx: Vec<i64> = indices
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect();
    assert_eq!(got_idx, vec![1, 3]);
}
