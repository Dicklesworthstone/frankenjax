//! Oracle tests for Sort and Argsort primitives.
//!
//! Tests against expected behavior matching JAX/NumPy:
//! - jax.lax.sort: stable ascending sort by default
//! - jax.lax.sort with is_stable=True, dimension=-1
//! - jnp.argsort: returns indices that would sort the array

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

fn descending_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("descending".to_string(), "true".to_string());
    p
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
        _ => panic!("expected tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

// ======================== Sort Oracle Tests ========================

#[test]
fn oracle_sort_1d_i64_ascending() {
    // JAX: jax.lax.sort(jnp.array([3, 1, 4, 1, 5])) => [1, 1, 3, 4, 5]
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 3, 4, 5]);
}

#[test]
fn sort_extremes_equal_reduce_min_max() {
    // Cross-validate Sort against ReduceMin/ReduceMax on finite data: an ascending
    // sort puts the minimum first and the maximum last. i64 so it is exact.
    // Oracle-free — catches a disagreement between the ordering and reduction paths.
    let data = vec![3_i64, 1, 4, 1, 5, 9, 2];
    let input = make_i64_tensor(&[data.len() as u32], data.clone());
    let sorted = extract_i64_vec(
        &eval_primitive(Primitive::Sort, &[input.clone()], &no_params()).unwrap(),
    );
    let axes = BTreeMap::from([("axes".to_string(), "0".to_string())]);
    let scalar = |v: &Value| -> i64 {
        match v {
            Value::Scalar(l) => l.as_i64().unwrap(),
            Value::Tensor(t) => t.elements[0].as_i64().unwrap(),
        }
    };
    let rmin = scalar(&eval_primitive(Primitive::ReduceMin, &[input.clone()], &axes).unwrap());
    let rmax = scalar(&eval_primitive(Primitive::ReduceMax, &[input], &axes).unwrap());
    assert_eq!(*sorted.first().unwrap(), rmin, "sort[0] must equal reduce_min");
    assert_eq!(*sorted.last().unwrap(), rmax, "sort[-1] must equal reduce_max");
}

#[test]
fn oracle_sort_1d_f64_ascending() {
    // JAX: jax.lax.sort(jnp.array([3.5, 1.2, 4.8, 1.1])) => [1.1, 1.2, 3.5, 4.8]
    let input = make_f64_tensor(&[4u32], vec![3.5, 1.2, 4.8, 1.1]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[1] - 1.2).abs() < 1e-10);
    assert!((vals[2] - 3.5).abs() < 1e-10);
    assert!((vals[3] - 4.8).abs() < 1e-10);
}

#[test]
fn oracle_sort_1d_descending() {
    // JAX: jax.lax.sort(x, is_ascending=False) => [5, 4, 3, 1, 1]
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &descending_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 4, 3, 1, 1]);
}

#[test]
fn oracle_sort_2d_last_axis() {
    // JAX: jax.lax.sort(jnp.array([[3,1],[4,2]]), dimension=-1) => [[1,3],[2,4]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Sort, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 2, 4]);
}

#[test]
fn oracle_sort_2d_first_axis() {
    // JAX: jax.lax.sort(jnp.array([[3,1],[4,2]]), dimension=0) => [[3,1],[4,2]]
    // Actually sorts along axis 0: [[3,1],[4,2]] => for col 0: [3,4] sorted is [3,4]
    // for col 1: [1,2] sorted is [1,2] => result [[3,1],[4,2]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Sort, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 4, 2]);
}

#[test]
fn oracle_sort_2d_negative_first_axis() {
    // JAX: jax.lax.sort(jnp.array([[4,3],[1,2]]), dimension=-2) => [[1,2],[4,3]]
    let input = make_i64_tensor(&[2u32, 2], vec![4, 3, 1, 2]);
    let result = eval_primitive(Primitive::Sort, &[input], &axis_params(-2)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 4, 3]);
}

#[test]
fn oracle_sort_negative_axis_out_of_bounds_errors() {
    let input = make_i64_tensor(&[2u32, 2], vec![4, 3, 1, 2]);
    let err = eval_primitive(Primitive::Sort, &[input], &axis_params(-3)).unwrap_err();
    assert!(
        err.to_string().contains("axis -3 out of bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn oracle_sort_empty_array() {
    let input = make_i64_tensor(&[0u32], vec![]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), Vec::<i64>::new());
}

#[test]
fn oracle_sort_single_element() {
    let input = make_i64_tensor(&[1u32], vec![42]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_sort_already_sorted() {
    let input = make_i64_tensor(&[5u32], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_sort_reverse_sorted() {
    let input = make_i64_tensor(&[5u32], vec![5, 4, 3, 2, 1]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_sort_with_negatives() {
    let input = make_i64_tensor(&[5u32], vec![-3, 0, -1, 2, -5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, -3, -1, 0, 2]);
}

#[test]
fn oracle_sort_i64_large_values_preserves_exact_literals() {
    // Values above 2^53 must not be rounded through f64 during sort.
    let input = make_i64_tensor(&[2u32], vec![9_007_199_254_740_993, 9_007_199_254_740_992]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_i64_vec(&result),
        vec![9_007_199_254_740_992, 9_007_199_254_740_993]
    );
}

#[test]
fn oracle_sort_f64_with_special_values() {
    // NaN handling: JAX sorts NaN to the end
    let input = make_f64_tensor(&[4u32], vec![f64::NAN, 1.0, f64::NEG_INFINITY, 2.0]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], f64::NEG_INFINITY);
    assert!((vals[1] - 1.0).abs() < 1e-10);
    assert!((vals[2] - 2.0).abs() < 1e-10);
    assert!(vals[3].is_nan());
}

#[test]
fn oracle_sort_sends_negative_nan_to_end_too() {
    // JAX/numpy sort sends EVERY NaN to the end, regardless of sign bit — unlike
    // total_cmp, which orders -NaN as the minimum (it would sort FIRST). Guards the
    // -NaN sign edge that the +NaN-only test above misses
    // (project_total_cmp_vs_jax_float_ordering; note top_k keeps -NaN as min).
    let neg_nan = -f64::NAN;
    assert!(neg_nan.is_nan() && neg_nan.is_sign_negative());
    let input = make_f64_tensor(&[4u32], vec![neg_nan, 3.0, f64::NAN, 1.0]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10, "finite values sort first");
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!(vals[2].is_nan(), "both NaN (either sign) sort to the end");
    assert!(vals[3].is_nan());
}

#[test]
fn oracle_sort_orders_negative_zero_before_positive_zero() {
    // JAX sort uses a total order in which -0.0 < +0.0 (verified vs JAX 0.10.1),
    // so a -0.0 sorts ahead of a +0.0 even though they compare equal under `==`.
    let input = make_f64_tensor(&[2u32], vec![0.0, -0.0]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(
        vals[0].is_sign_negative() && vals[0] == 0.0,
        "-0.0 must sort before +0.0, got {:?}",
        vals
    );
    assert!(vals[1].is_sign_positive() && vals[1] == 0.0);
}

// ======================== Argsort Oracle Tests ========================

#[test]
fn oracle_argsort_1d_i64() {
    // JAX: jnp.argsort(jnp.array([3, 1, 4, 1, 5])) => [1, 3, 0, 2, 4]
    // indices that would sort: positions of [1,1,3,4,5] in original
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    let indices = extract_i64_vec(&result);

    // Verify: applying indices to original yields sorted array
    let original = [3i64, 1, 4, 1, 5];
    let sorted_via_indices: Vec<i64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert_eq!(sorted_via_indices, vec![1, 1, 3, 4, 5]);
}

#[test]
fn oracle_argsort_1d_f64() {
    let input = make_f64_tensor(&[4u32], vec![3.5, 1.2, 4.8, 1.1]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    let indices = extract_i64_vec(&result);

    let original = [3.5, 1.2, 4.8, 1.1];
    let sorted_via_indices: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!((sorted_via_indices[0] - 1.1).abs() < 1e-10);
    assert!((sorted_via_indices[1] - 1.2).abs() < 1e-10);
    assert!((sorted_via_indices[2] - 3.5).abs() < 1e-10);
    assert!((sorted_via_indices[3] - 4.8).abs() < 1e-10);
}

#[test]
fn oracle_argsort_descending() {
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &descending_params()).unwrap();
    let indices = extract_i64_vec(&result);

    let original = [3i64, 1, 4, 1, 5];
    let sorted_via_indices: Vec<i64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert_eq!(sorted_via_indices, vec![5, 4, 3, 1, 1]);
}

#[test]
fn oracle_argsort_2d_last_axis() {
    // JAX: jnp.argsort(jnp.array([[3,1],[4,2]]), axis=-1) => [[1,0],[1,0]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Argsort, &[input], &axis_params(-1)).unwrap();
    let indices = extract_i64_vec(&result);

    // Row 0: [3,1] -> argsort -> [1,0] (index 1 has smaller value)
    // Row 1: [4,2] -> argsort -> [1,0]
    assert_eq!(indices, vec![1, 0, 1, 0]);
}

#[test]
fn oracle_argsort_2d_negative_first_axis() {
    // JAX: jnp.argsort(jnp.array([[4,3],[1,2]]), axis=-2) => [[1,1],[0,0]]
    let input = make_i64_tensor(&[2u32, 2], vec![4, 3, 1, 2]);
    let result = eval_primitive(Primitive::Argsort, &[input], &axis_params(-2)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 0, 0]);
}

#[test]
fn oracle_argsort_i64_large_values_uses_exact_integer_ordering() {
    let input = make_i64_tensor(
        &[3u32],
        vec![
            9_007_199_254_740_993,
            9_007_199_254_740_992,
            9_007_199_254_740_994,
        ],
    );
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 2]);
}

#[test]
fn oracle_argsort_empty() {
    let input = make_i64_tensor(&[0u32], vec![]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), Vec::<i64>::new());
}

#[test]
fn oracle_argsort_single_element() {
    let input = make_i64_tensor(&[1u32], vec![42]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_argsort_already_sorted() {
    let input = make_i64_tensor(&[5u32], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_argsort_stability_check() {
    // When elements are equal, stable sort should preserve original order
    // JAX: jnp.argsort(jnp.array([1, 1, 1])) => [0, 1, 2]
    let input = make_i64_tensor(&[3u32], vec![1, 1, 1]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2]);
}

// ======================== Metamorphic Properties ========================

#[test]
fn metamorphic_sort_idempotent() {
    // Metamorphic: sort(sort(x)) = sort(x) — sorting is idempotent
    let data = vec![7i64, 2, 9, 1, 5, 3, 8, 4, 6];
    let input = make_i64_tensor(&[9u32], data);

    let sorted_once =
        eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &no_params()).unwrap();
    let sorted_twice = eval_primitive(
        Primitive::Sort,
        std::slice::from_ref(&sorted_once),
        &no_params(),
    )
    .unwrap();

    assert_eq!(
        extract_i64_vec(&sorted_once),
        extract_i64_vec(&sorted_twice),
        "sort should be idempotent"
    );
}

#[test]
fn metamorphic_sort_idempotent_f64() {
    // Idempotence with f64 including negative values
    let data = vec![-3.5, 2.1, -1.0, 0.0, 5.5, -2.2, 4.4];
    let input = make_f64_tensor(&[7u32], data);

    let sorted_once =
        eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &no_params()).unwrap();
    let sorted_twice = eval_primitive(
        Primitive::Sort,
        std::slice::from_ref(&sorted_once),
        &no_params(),
    )
    .unwrap();

    let vals1 = extract_f64_vec(&sorted_once);
    let vals2 = extract_f64_vec(&sorted_twice);
    for (a, b) in vals1.iter().zip(vals2.iter()) {
        assert!((a - b).abs() < 1e-15, "sort should be idempotent for f64");
    }
}

#[test]
fn metamorphic_sort_2d_idempotent() {
    // Idempotence for 2D tensors along default axis
    let data = vec![9i64, 3, 7, 1, 8, 2, 6, 4, 5];
    let input = make_i64_tensor(&[3u32, 3], data);

    let sorted_once =
        eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &no_params()).unwrap();
    let sorted_twice = eval_primitive(
        Primitive::Sort,
        std::slice::from_ref(&sorted_once),
        &no_params(),
    )
    .unwrap();

    assert_eq!(
        extract_i64_vec(&sorted_once),
        extract_i64_vec(&sorted_twice),
        "2D sort should be idempotent"
    );
}

#[test]
fn metamorphic_argsort_applied_is_sorted() {
    // Metamorphic: x[argsort(x)] produces a sorted array
    // This is stronger than consistency — the result must be monotonically increasing
    let data = vec![7i64, 2, 9, 1, 5, 3, 8, 4, 6];
    let input = make_i64_tensor(&[9u32], data.clone());

    let indices = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    let idx_vals = extract_i64_vec(&indices);
    let applied: Vec<i64> = idx_vals.iter().map(|&i| data[i as usize]).collect();

    // Verify monotonically non-decreasing
    for i in 1..applied.len() {
        assert!(
            applied[i] >= applied[i - 1],
            "x[argsort(x)] must be sorted: {} >= {} at index {}",
            applied[i],
            applied[i - 1],
            i
        );
    }
}

#[test]
fn metamorphic_sort_descending_reverses_ascending() {
    // Metamorphic: sort_desc(x) = reverse(sort_asc(x))
    let data = vec![7i64, 2, 9, 1, 5, 3, 8, 4, 6];
    let input = make_i64_tensor(&[9u32], data);

    let sorted_asc =
        eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &no_params()).unwrap();
    let sorted_desc = eval_primitive(Primitive::Sort, &[input], &descending_params()).unwrap();

    let asc_vals = extract_i64_vec(&sorted_asc);
    let desc_vals = extract_i64_vec(&sorted_desc);

    let reversed_asc: Vec<i64> = asc_vals.iter().rev().copied().collect();
    assert_eq!(
        reversed_asc, desc_vals,
        "descending sort should be reverse of ascending"
    );
}

// ======================== Sort+Argsort Consistency ========================

#[test]
fn oracle_sort_argsort_consistency() {
    // Sort and Argsort should be consistent:
    // sort(x) == x[argsort(x)]
    let data = vec![7i64, 2, 9, 1, 5, 3, 8, 4, 6];
    let input = make_i64_tensor(&[9u32], data.clone());

    let sorted =
        eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &no_params()).unwrap();
    let indices = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();

    let sorted_vals = extract_i64_vec(&sorted);
    let idx_vals = extract_i64_vec(&indices);

    let reconstructed: Vec<i64> = idx_vals.iter().map(|&i| data[i as usize]).collect();
    assert_eq!(sorted_vals, reconstructed);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_sort_preserves_all_float_dtypes() {
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

    let values = [3.0_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "sort {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
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
fn oracle_sort_complex64_lexicographic() {
    // JAX lax.sort sorts complex lexicographically by (real, imag) — its
    // comparator (`_operands_to_keys`) splits complex into [real, imag] float
    // keys. NumPy agrees:
    //   np.sort([3+1j, 1+2j, 1+1j, 2+0j]) == [1+1j, 1+2j, 2+0j, 3+1j]
    let input = make_complex64_tensor(&[4], vec![(3.0, 1.0), (1.0, 2.0), (1.0, 1.0), (2.0, 0.0)]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params())
        .expect("sort should work on complex64");
    let Value::Tensor(t) = result else {
        panic!("expected tensor output");
    };
    let got: Vec<(f32, f32)> = t
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect();
    assert_eq!(
        got,
        vec![(1.0, 1.0), (1.0, 2.0), (2.0, 0.0), (3.0, 1.0)],
        "complex64 sort must order lexicographically by (real, imag)"
    );
}
