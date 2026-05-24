//! Oracle tests for Argsort primitive.
//!
//! argsort(x) returns the indices that would sort the array
//!
//! Tests:
//! - Basic: argsort([3, 1, 2]) = [1, 2, 0]
//! - Already sorted
//! - Reverse sorted
//! - With duplicates
//! - Negative values
//! - 2D (per-row)

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

fn argsort_params(axis: i64, descending: bool) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), axis.to_string());
    params.insert("descending".to_string(), descending.to_string());
    params
}

// ======================== Basic Cases ========================

#[test]
fn oracle_argsort_basic() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let indices = extract_i64_vec(&result);
    // Sorted: 1.0, 1.0, 3.0, 4.0, 5.0 -> indices: 1 or 3, 3 or 1, 0, 2, 4
    // Check that applying these indices sorts the array
    let original = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let sorted: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "result should be sorted"
    );
}

#[test]
fn oracle_argsort_already_sorted() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn oracle_argsort_reverse_sorted() {
    let input = make_f64_tensor(&[4], vec![4.0, 3.0, 2.0, 1.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![3, 2, 1, 0]);
}

#[test]
fn oracle_argsort_descending() {
    let input = make_f64_tensor(&[4], vec![1.0, 4.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, true)).unwrap();
    let indices = extract_i64_vec(&result);
    // Descending: 4, 3, 2, 1 -> indices: 1, 3, 2, 0
    assert_eq!(indices, vec![1, 3, 2, 0]);
}

// ======================== Negative Values ========================

#[test]
fn oracle_argsort_negative() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -4, -3, -2, -1 -> indices: 2, 0, 3, 1
    assert_eq!(indices, vec![2, 0, 3, 1]);
}

// ======================== Integer Types ========================

#[test]
fn oracle_argsort_i64() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: 10, 20, 30, 40 -> indices: 1, 3, 0, 2
    assert_eq!(indices, vec![1, 3, 0, 2]);
}

// ======================== 2D (Per-Row) ========================

#[test]
fn oracle_argsort_2d() {
    // [[3, 1, 2], [6, 4, 5]]
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let indices = extract_i64_vec(&result);
    // Row 0: sorted [1, 2, 3] -> indices [1, 2, 0]
    // Row 1: sorted [4, 5, 6] -> indices [1, 2, 0]
    assert_eq!(indices, vec![1, 2, 0, 1, 2, 0]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_argsort_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_argsort_2d_axis0() {
    // Sort along axis 0 (columns)
    // [[3, 1], [1, 3]] -> sorted by columns: [[1, 1], [3, 3]]
    // Column 0: [3, 1] -> indices [1, 0]
    // Column 1: [1, 3] -> indices [0, 1]
    let input = make_f64_tensor(&[2, 2], vec![3.0, 1.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(0, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![1, 0, 0, 1]);
}

#[test]
fn oracle_argsort_preserves_index_dtype() {
    let input = make_f64_tensor(&[3], vec![3.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    match &result {
        Value::Tensor(t) => {
            // Argsort output should be integer type
            assert!(
                matches!(t.dtype, DType::I32 | DType::I64),
                "argsort should return integer indices"
            );
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_argsort_with_zeros() {
    let input = make_f64_tensor(&[5], vec![0.0, -1.0, 1.0, 0.0, -2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -2, -1, 0, 0, 1 -> indices: 4, 1, 0 or 3, 3 or 0, 2
    let original = vec![0.0, -1.0, 1.0, 0.0, -2.0];
    let sorted: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "result should be sorted"
    );
}

#[test]
fn oracle_argsort_all_equal() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // All equal: stable sort should preserve original order
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn oracle_argsort_large_values() {
    let input = make_f64_tensor(&[4], vec![1e10, -1e10, 1e-10, -1e-10]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -1e10, -1e-10, 1e-10, 1e10
    assert_eq!(indices, vec![1, 3, 2, 0]);
}

#[test]
fn oracle_argsort_descending_3d() {
    // 3D tensor, sort along last axis descending
    let input = make_f64_tensor(&[2, 1, 3], vec![1.0, 3.0, 2.0, 6.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, true)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3]);
    let indices = extract_i64_vec(&result);
    // [1, 3, 2] desc -> [3, 2, 1] -> indices [1, 2, 0]
    // [6, 4, 5] desc -> [6, 5, 4] -> indices [0, 2, 1]
    assert_eq!(indices, vec![1, 2, 0, 0, 2, 1]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_argsort_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

#[test]
fn oracle_argsort_preserves_shape() {
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_argsort_special_values() {
    // NaN should sort to the end in ascending order per IEEE 754
    let input = make_f64_tensor(
        &[5],
        vec![f64::NAN, 1.0, f64::INFINITY, 0.0, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Expected order: -inf (4), 0 (3), 1 (1), inf (2), nan (0)
    // NaN handling varies by implementation, but finite values should be ordered correctly
    assert_eq!(indices.len(), 5);
    // Check that -inf comes first
    assert_eq!(indices[0], 4);
    // And inf comes before nan
    assert_eq!(indices[3], 2);
}

#[test]
fn oracle_argsort_2d_both_axes() {
    // Test sorting along both axes
    let input = make_f64_tensor(&[3, 3], vec![9.0, 7.0, 8.0, 6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);

    // Sort along axis 0 (columns)
    let result_axis0 = eval_primitive(
        Primitive::Argsort,
        &[input.clone()],
        &argsort_params(0, false),
    )
    .unwrap();
    assert_eq!(extract_shape(&result_axis0), vec![3, 3]);
    let indices0 = extract_i64_vec(&result_axis0);
    // Column 0: [9, 6, 3] -> sorted indices [2, 1, 0]
    // Column 1: [7, 4, 1] -> sorted indices [2, 1, 0]
    // Column 2: [8, 5, 2] -> sorted indices [2, 1, 0]
    assert_eq!(indices0, vec![2, 2, 2, 1, 1, 1, 0, 0, 0]);

    // Sort along axis 1 (rows)
    let result_axis1 =
        eval_primitive(Primitive::Argsort, &[input], &argsort_params(1, false)).unwrap();
    assert_eq!(extract_shape(&result_axis1), vec![3, 3]);
}

#[test]
fn oracle_argsort_result_is_permutation() {
    // Verify that argsort result is a valid permutation
    let input = make_f64_tensor(&[6], vec![5.0, 3.0, 1.0, 4.0, 2.0, 6.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);

    // Check that indices form a valid permutation (each 0..6 appears exactly once)
    let mut sorted_indices = indices.clone();
    sorted_indices.sort();
    assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn oracle_argsort_4d() {
    let input = make_f64_tensor(
        &[2, 2, 2, 3],
        vec![
            3.0, 1.0, 2.0, 6.0, 4.0, 5.0, 9.0, 7.0, 8.0, 12.0, 10.0, 11.0, 15.0, 13.0, 14.0, 18.0,
            16.0, 17.0, 21.0, 19.0, 20.0, 24.0, 22.0, 23.0,
        ],
    );
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 3]);
}

#[test]
fn oracle_argsort_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_argsort_large_tensor() {
    let data: Vec<f64> = (0..200).map(|x| (100 - x % 200) as f64).collect();
    let input = make_f64_tensor(&[200], data.clone());
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![200]);
    let indices = extract_i64_vec(&result);
    let sorted: Vec<f64> = indices.iter().map(|&i| data[i as usize]).collect();
    assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "result should be sorted"
    );
}

#[test]
fn oracle_argsort_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[4], vec![subnormal, 0.0, -subnormal, 1.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Order: -subnormal (2), 0 (1), subnormal (0), 1 (3)
    assert_eq!(indices, vec![2, 1, 0, 3]);
}

// ======================== PROPERTY: output dtype is always I64 indices ========================

#[test]
fn property_argsort_output_i64_for_all_float_inputs() {
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
    let values = [3.0_f64, 1.0, 4.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result =
            eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
        assert!(
            matches!(result.dtype(), DType::I32 | DType::I64),
            "argsort on {dtype:?} should return integer indices, got {:?}",
            result.dtype()
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
#[ignore = "PARITY GAP: Argsort not supported for complex - no natural ordering"]
fn oracle_argsort_complex64_not_supported() {
    let input = make_complex64_tensor(&[3], vec![(3.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
    let _result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false))
        .expect("argsort should work on complex64");
}
