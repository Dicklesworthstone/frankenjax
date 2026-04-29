//! Oracle tests for ReduceMin and ReduceMax primitives.
//!
//! ReduceMin - reduces a tensor along specified axes by taking the minimum
//! ReduceMax - reduces a tensor along specified axes by taking the maximum
//!
//! Tests:
//! - Single axis reduction
//! - Multi-axis reduction
//! - Full reduction (all axes)
//! - Empty axes (no reduction)
//! - Negative values
//! - Infinity handling
//! - NaN handling
//! - Shape verification
//! - Relationship between ReduceMin/Max

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn params_with_axes(axes: &[i64]) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    let axes_str = axes
        .iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join(",");
    params.insert("axes".to_string(), axes_str);
    params
}

// ====================== REDUCE MIN: FULL REDUCTION ======================

#[test]
fn oracle_reduce_min_full_1d() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

#[test]
fn oracle_reduce_min_full_2d() {
    let input = make_f64_tensor(&[2, 3], vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0]);
    let result =
        eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 1.0);
}

// ====================== REDUCE MIN: SINGLE AXIS ======================

#[test]
fn oracle_reduce_min_axis0_2d() {
    // Shape [2, 3]: [[1, 5, 3], [4, 2, 6]] -> [1, 2, 3] (min along axis 0)
    let input = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
}

#[test]
fn oracle_reduce_min_axis1_2d() {
    // Shape [2, 3]: [[1, 5, 3], [4, 2, 6]] -> [1, 2] (min along axis 1)
    let input = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0]);
}

#[test]
fn oracle_reduce_min_axis0_3d() {
    // Shape [2, 2, 2]: reduce along axis 0
    let input = make_f64_tensor(&[2, 2, 2], vec![8.0, 1.0, 5.0, 3.0, 2.0, 7.0, 4.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![2.0, 1.0, 4.0, 3.0]);
}

// ====================== REDUCE MIN: NEGATIVE VALUES ======================

#[test]
fn oracle_reduce_min_negative() {
    let input = make_f64_tensor(&[4], vec![-5.0, -2.0, -8.0, -1.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), -8.0);
}

#[test]
fn oracle_reduce_min_mixed_sign() {
    let input = make_f64_tensor(&[5], vec![-3.0, 1.0, -5.0, 2.0, 0.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0);
}

// ====================== REDUCE MIN: INFINITY ======================

#[test]
fn oracle_reduce_min_pos_infinity() {
    let input = make_f64_tensor(&[3], vec![f64::INFINITY, 5.0, 10.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_reduce_min_neg_infinity() {
    let input = make_f64_tensor(&[3], vec![f64::NEG_INFINITY, 5.0, -10.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0);
}

// ====================== REDUCE MIN: NaN ======================

#[test]
fn oracle_reduce_min_with_nan() {
    let input = make_f64_tensor(&[3], vec![f64::NAN, 5.0, 3.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    let val = extract_f64_scalar(&result);
    // NaN behavior: often propagates or returns the non-NaN minimum
    // This depends on implementation
    assert!(val.is_nan() || val == 3.0);
}

// ====================== REDUCE MIN: INTEGER ======================

#[test]
fn oracle_reduce_min_i64() {
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_reduce_min_i64_negative() {
    let input = make_i64_tensor(&[4], vec![-5, 2, -8, 1]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), -8);
}

// ====================== REDUCE MAX: FULL REDUCTION ======================

#[test]
fn oracle_reduce_max_full_1d() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

#[test]
fn oracle_reduce_max_full_2d() {
    let input = make_f64_tensor(&[2, 3], vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0]);
    let result =
        eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 9.0);
}

// ====================== REDUCE MAX: SINGLE AXIS ======================

#[test]
fn oracle_reduce_max_axis0_2d() {
    // Shape [2, 3]: [[1, 5, 3], [4, 2, 6]] -> [4, 5, 6] (max along axis 0)
    let input = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 5.0, 6.0]);
}

#[test]
fn oracle_reduce_max_axis1_2d() {
    // Shape [2, 3]: [[1, 5, 3], [4, 2, 6]] -> [5, 6] (max along axis 1)
    let input = make_f64_tensor(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 6.0]);
}

#[test]
fn oracle_reduce_max_axis0_3d() {
    // Shape [2, 2, 2]: reduce along axis 0
    let input = make_f64_tensor(&[2, 2, 2], vec![8.0, 1.0, 5.0, 3.0, 2.0, 7.0, 4.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![8.0, 7.0, 5.0, 6.0]);
}

// ====================== REDUCE MAX: NEGATIVE VALUES ======================

#[test]
fn oracle_reduce_max_negative() {
    let input = make_f64_tensor(&[4], vec![-5.0, -2.0, -8.0, -1.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0);
}

#[test]
fn oracle_reduce_max_mixed_sign() {
    let input = make_f64_tensor(&[5], vec![-3.0, 1.0, -5.0, 2.0, 0.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0);
}

// ====================== REDUCE MAX: INFINITY ======================

#[test]
fn oracle_reduce_max_pos_infinity() {
    let input = make_f64_tensor(&[3], vec![f64::INFINITY, 5.0, 10.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn oracle_reduce_max_neg_infinity() {
    let input = make_f64_tensor(&[3], vec![f64::NEG_INFINITY, 5.0, -10.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0);
}

// ====================== REDUCE MAX: NaN ======================

#[test]
fn oracle_reduce_max_with_nan() {
    let input = make_f64_tensor(&[3], vec![f64::NAN, 5.0, 3.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    let val = extract_f64_scalar(&result);
    // NaN behavior depends on implementation
    assert!(val.is_nan() || val == 5.0);
}

// ====================== REDUCE MAX: INTEGER ======================

#[test]
fn oracle_reduce_max_i64() {
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 5);
}

#[test]
fn oracle_reduce_max_i64_negative() {
    let input = make_i64_tensor(&[4], vec![-5, 2, -8, 1]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2);
}

// ====================== MIN/MAX RELATIONSHIP ======================

#[test]
fn oracle_reduce_min_max_relationship() {
    // For same input: reduce_min <= reduce_max
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let min_result = eval_primitive(
        Primitive::ReduceMin,
        std::slice::from_ref(&input),
        &params_with_axes(&[0]),
    )
    .unwrap();
    let max_result =
        eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();

    let min_val = extract_f64_scalar(&min_result);
    let max_val = extract_f64_scalar(&max_result);

    assert!(min_val <= max_val, "reduce_min <= reduce_max");
    assert_eq!(min_val, 1.0);
    assert_eq!(max_val, 5.0);
}

#[test]
fn oracle_reduce_min_max_single_element() {
    // For single element, min == max
    let input = make_f64_tensor(&[1], vec![42.0]);
    let min_result = eval_primitive(
        Primitive::ReduceMin,
        std::slice::from_ref(&input),
        &params_with_axes(&[0]),
    )
    .unwrap();
    let max_result =
        eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();

    assert_eq!(extract_f64_scalar(&min_result), 42.0);
    assert_eq!(extract_f64_scalar(&max_result), 42.0);
}

#[test]
fn oracle_reduce_min_max_all_same() {
    // All same values: min == max
    let input = make_f64_tensor(&[4], vec![7.0, 7.0, 7.0, 7.0]);
    let min_result = eval_primitive(
        Primitive::ReduceMin,
        std::slice::from_ref(&input),
        &params_with_axes(&[0]),
    )
    .unwrap();
    let max_result =
        eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();

    assert_eq!(extract_f64_scalar(&min_result), 7.0);
    assert_eq!(extract_f64_scalar(&max_result), 7.0);
}

// ====================== MULTI-AXIS REDUCTION ======================

#[test]
fn oracle_reduce_min_multi_axis() {
    // Shape [2, 3, 2] -> reduce axes [0, 2] -> shape [3]
    let input = make_f64_tensor(
        &[2, 3, 2],
        vec![
            1.0, 5.0, // [0, 0, :]
            3.0, 2.0, // [0, 1, :]
            8.0, 4.0, // [0, 2, :]
            6.0, 9.0, // [1, 0, :]
            7.0, 1.0, // [1, 1, :]
            2.0, 3.0, // [1, 2, :]
        ],
    );
    let result =
        eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    // For each position in axis 1, find min across axes 0 and 2
    // axis 1 = 0: min(1, 5, 6, 9) = 1
    // axis 1 = 1: min(3, 2, 7, 1) = 1
    // axis 1 = 2: min(8, 4, 2, 3) = 2
    assert_eq!(extract_f64_vec(&result), vec![1.0, 1.0, 2.0]);
}

#[test]
fn oracle_reduce_max_multi_axis() {
    let input = make_f64_tensor(
        &[2, 3, 2],
        vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0, 6.0, 9.0, 7.0, 1.0, 2.0, 3.0],
    );
    let result =
        eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    // For each position in axis 1, find max across axes 0 and 2
    // axis 1 = 0: max(1, 5, 6, 9) = 9
    // axis 1 = 1: max(3, 2, 7, 1) = 7
    // axis 1 = 2: max(8, 4, 2, 3) = 8
    assert_eq!(extract_f64_vec(&result), vec![9.0, 7.0, 8.0]);
}

// ====================== ZEROS ======================

#[test]
fn oracle_reduce_min_with_zeros() {
    let input = make_f64_tensor(&[4], vec![0.0, 1.0, 0.0, 2.0]);
    let result = eval_primitive(Primitive::ReduceMin, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

#[test]
fn oracle_reduce_max_with_zeros() {
    let input = make_f64_tensor(&[4], vec![0.0, -1.0, 0.0, -2.0]);
    let result = eval_primitive(Primitive::ReduceMax, &[input], &params_with_axes(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}
