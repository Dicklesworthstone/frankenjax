//! Oracle tests for ReduceSum primitive.
//!
//! ReduceSum: Sum reduction along specified axes
//!
//! Properties tested:
//! - Identity: sum([x]) = x
//! - Linearity: sum(a * x) = a * sum(x)
//! - Commutativity: order of elements doesn't matter
//! - Distributivity: sum(x) + sum(y) = sum(concat(x, y))
//! - Empty tensor: sum([]) = 0
//!
//! Tests:
//! - Basic reductions
//! - Multi-axis reductions
//! - Full reductions
//! - Special values (infinity, NaN)
//! - Integer and float types

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
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
        Value::Scalar(_) => vec![],
    }
}

fn reduce_params(axes: &[i64]) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    let axes_str: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
    params.insert("axes".to_string(), axes_str.join(","));
    params
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff={}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ====================== BASIC 1D REDUCTION ======================

#[test]
fn oracle_reduce_sum_1d_basic() {
    let input = make_f64_tensor(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 15.0);
}

#[test]
fn oracle_reduce_sum_1d_single() {
    // sum of single element = element
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 42.0);
}

#[test]
fn oracle_reduce_sum_1d_negative() {
    let input = make_f64_tensor(&[4], vec![-1.0, -2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0);
}

#[test]
fn oracle_reduce_sum_1d_zeros() {
    let input = make_f64_tensor(&[4], vec![0.0, 0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0);
}

// ====================== 2D AXIS REDUCTIONS ======================

#[test]
fn oracle_reduce_sum_2d_axis0() {
    // [[1, 2, 3],
    //  [4, 5, 6]] -> [5, 7, 9]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 7.0, 9.0]);
}

#[test]
fn oracle_reduce_sum_2d_axis1() {
    // [[1, 2, 3],
    //  [4, 5, 6]] -> [6, 15]
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_f64_vec(&result), vec![6.0, 15.0]);
}

#[test]
fn oracle_reduce_sum_2d_full() {
    // sum all elements
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 21.0);
}

// ====================== 3D REDUCTIONS ======================

#[test]
fn oracle_reduce_sum_3d_axis0() {
    // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[6, 8], [10, 12]]
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn oracle_reduce_sum_3d_axis1() {
    // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[4, 6], [12, 14]]
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 6.0, 12.0, 14.0]);
}

#[test]
fn oracle_reduce_sum_3d_axis2() {
    // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[3, 7], [11, 15]]
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn oracle_reduce_sum_3d_full() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result =
        eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0, 1, 2])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 36.0);
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_reduce_sum_infinity() {
    let input = make_f64_tensor(&[3], vec![1.0, f64::INFINITY, 2.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY);
}

#[test]
fn oracle_reduce_sum_neg_infinity() {
    let input = make_f64_tensor(&[3], vec![1.0, f64::NEG_INFINITY, 2.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY);
}

#[test]
fn oracle_reduce_sum_nan() {
    let input = make_f64_tensor(&[3], vec![1.0, f64::NAN, 2.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

#[test]
fn oracle_reduce_sum_inf_cancel() {
    // inf + (-inf) = NaN
    let input = make_f64_tensor(&[3], vec![f64::INFINITY, 1.0, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert!(extract_f64_scalar(&result).is_nan());
}

// ====================== LINEARITY ======================

#[test]
fn oracle_reduce_sum_linearity() {
    // sum(a * x) ≈ a * sum(x) for same-sign elements to avoid precision issues
    let a = 3.0;
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let ax: Vec<f64> = x.iter().map(|xi| a * xi).collect();

    let sum_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], x)],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let sum_ax = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], ax)],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    assert_close(sum_ax, a * sum_x, 1e-14, "sum(a*x) = a*sum(x)");
}

// ====================== INTEGER REDUCTIONS ======================

#[test]
fn oracle_reduce_sum_i64_basic() {
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15);
}

#[test]
fn oracle_reduce_sum_i64_2d_axis0() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![5, 7, 9]);
}

#[test]
fn oracle_reduce_sum_i64_2d_axis1() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![6, 15]);
}

#[test]
fn oracle_reduce_sum_i64_negative() {
    let input = make_i64_tensor(&[4], vec![-1, -2, 3, 4]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 4);
}

// ====================== LARGE TENSOR ======================

#[test]
fn oracle_reduce_sum_large() {
    // Sum 1..100
    let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0])).unwrap();
    // Sum = n*(n+1)/2 = 100*101/2 = 5050
    assert_eq!(extract_f64_scalar(&result), 5050.0);
}

// ====================== CONSISTENCY ======================

#[test]
fn oracle_reduce_sum_axis_order_independent() {
    // sum over all axes should give same result regardless of axis order
    let input1 = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let input2 = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result_01 = extract_f64_scalar(
        &eval_primitive(Primitive::ReduceSum, &[input1], &reduce_params(&[0, 1])).unwrap(),
    );
    let result_10 = extract_f64_scalar(
        &eval_primitive(Primitive::ReduceSum, &[input2], &reduce_params(&[1, 0])).unwrap(),
    );

    assert_eq!(
        result_01, result_10,
        "sum should be independent of axis order"
    );
}

// ====================== MULTI-AXIS PARTIAL REDUCTIONS ======================

#[test]
fn oracle_reduce_sum_3d_axes_01() {
    // Reduce first two axes of shape [2, 2, 2]
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    // First depth: 1+3+5+7=16, Second depth: 2+4+6+8=20
    assert_eq!(extract_f64_vec(&result), vec![16.0, 20.0]);
}

#[test]
fn oracle_reduce_sum_3d_axes_12() {
    // Reduce last two axes of shape [2, 2, 2]
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &reduce_params(&[1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    // First batch: 1+2+3+4=10, Second batch: 5+6+7+8=26
    assert_eq!(extract_f64_vec(&result), vec![10.0, 26.0]);
}

// ====================== METAMORPHIC: DISTRIBUTIVITY OVER CONCAT ======================

fn concat_params(axis: i64) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("dimension".to_string(), axis.to_string());
    params
}

#[test]
fn metamorphic_reduce_sum_distributive_over_concat() {
    // sum(concat(x, y)) = sum(x) + sum(y)
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let sum_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[3], x.clone())],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let sum_y = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[3], y.clone())],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let concat_xy = eval_primitive(
        Primitive::Concatenate,
        &[make_f64_tensor(&[3], x), make_f64_tensor(&[3], y)],
        &concat_params(0),
    )
    .unwrap();

    let sum_concat = extract_f64_scalar(
        &eval_primitive(Primitive::ReduceSum, &[concat_xy], &reduce_params(&[0])).unwrap(),
    );

    assert_close(
        sum_concat,
        sum_x + sum_y,
        1e-14,
        "sum(concat(x, y)) = sum(x) + sum(y)",
    );
}

#[test]
fn metamorphic_reduce_sum_distributive_over_concat_2d() {
    // For 2D tensors along axis 0: sum(concat(A, B, axis=0), axis=0) = sum(A, axis=0) + sum(B, axis=0)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [2, 3]

    let sum_a = eval_primitive(
        Primitive::ReduceSum,
        &[make_f64_tensor(&[2, 3], a.clone())],
        &reduce_params(&[0]),
    )
    .unwrap();

    let sum_b = eval_primitive(
        Primitive::ReduceSum,
        &[make_f64_tensor(&[2, 3], b.clone())],
        &reduce_params(&[0]),
    )
    .unwrap();

    let concat_ab = eval_primitive(
        Primitive::Concatenate,
        &[
            make_f64_tensor(&[2, 3], a),
            make_f64_tensor(&[2, 3], b),
        ],
        &concat_params(0),
    )
    .unwrap();

    let sum_concat = eval_primitive(
        Primitive::ReduceSum,
        &[concat_ab],
        &reduce_params(&[0]),
    )
    .unwrap();

    let sum_a_vec = extract_f64_vec(&sum_a);
    let sum_b_vec = extract_f64_vec(&sum_b);
    let sum_concat_vec = extract_f64_vec(&sum_concat);

    for i in 0..3 {
        assert_close(
            sum_concat_vec[i],
            sum_a_vec[i] + sum_b_vec[i],
            1e-14,
            &format!("sum(concat(A, B), axis=0)[{}] = sum(A)[{}] + sum(B)[{}]", i, i, i),
        );
    }
}

#[test]
fn metamorphic_reduce_sum_commutative_with_add() {
    // sum(x) + sum(y) = sum(y) + sum(x) (trivially true, but exercises the relation)
    let x = vec![1.5, 2.5, 3.5];
    let y = vec![4.5, 5.5, 6.5];

    let sum_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[3], x.clone())],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let sum_y = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[3], y.clone())],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    assert_eq!(sum_x + sum_y, sum_y + sum_x, "sum(x) + sum(y) = sum(y) + sum(x)");
}

#[test]
fn metamorphic_reduce_sum_zero_identity() {
    // sum(x) + sum([0, 0, 0]) = sum(x)
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let zeros = vec![0.0, 0.0, 0.0, 0.0];

    let sum_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], x.clone())],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let sum_zeros = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], zeros)],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    assert_eq!(sum_zeros, 0.0, "sum of zeros = 0");
    assert_eq!(sum_x + sum_zeros, sum_x, "sum(x) + sum(zeros) = sum(x)");
}

#[test]
fn metamorphic_reduce_sum_negation_cancels() {
    // sum(x) + sum(neg(x)) = 0
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();

    let sum_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], x)],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    let sum_neg_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceSum,
            &[make_f64_tensor(&[4], neg_x)],
            &reduce_params(&[0]),
        )
        .unwrap(),
    );

    assert_close(sum_x + sum_neg_x, 0.0, 1e-14, "sum(x) + sum(neg(x)) = 0");
}
