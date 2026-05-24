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
        &[make_f64_tensor(&[2, 3], a), make_f64_tensor(&[2, 3], b)],
        &concat_params(0),
    )
    .unwrap();

    let sum_concat =
        eval_primitive(Primitive::ReduceSum, &[concat_ab], &reduce_params(&[0])).unwrap();

    let sum_a_vec = extract_f64_vec(&sum_a);
    let sum_b_vec = extract_f64_vec(&sum_b);
    let sum_concat_vec = extract_f64_vec(&sum_concat);

    for i in 0..3 {
        assert_close(
            sum_concat_vec[i],
            sum_a_vec[i] + sum_b_vec[i],
            1e-14,
            &format!(
                "sum(concat(A, B), axis=0)[{}] = sum(A)[{}] + sum(B)[{}]",
                i, i, i
            ),
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

    assert_eq!(
        sum_x + sum_y,
        sum_y + sum_x,
        "sum(x) + sum(y) = sum(y) + sum(x)"
    );
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

// Regression test for the dtype-preservation fix to fj-lax::reduction
// (eval_reduce / eval_reduce_axes used to force the float output to
// DType::F64 with F64Bits elements, widening F32/BF16/F16 inputs).
#[test]
fn oracle_reduce_sum_f32_full_preserves_dtype() {
    let data: Vec<Literal> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(Literal::from_f32)
        .collect();
    let input = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: vec![4] }, data).unwrap());

    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    match result {
        Value::Scalar(Literal::F32Bits(bits)) => {
            let value = f32::from_bits(bits);
            assert!((value - 10.0).abs() < 1e-6, "expected 10.0, got {value}");
        }
        other => panic!("expected F32Bits scalar from F32 reduce_sum, got {other:?}"),
    }
}

#[test]
fn oracle_reduce_sum_f32_axes_preserves_dtype() {
    let data: Vec<Literal> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(Literal::from_f32)
        .collect();
    let input =
        Value::Tensor(TensorValue::new(DType::F32, Shape { dims: vec![2, 3] }, data).unwrap());

    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "1".to_string());
    let result = eval_primitive(Primitive::ReduceSum, &[input], &params).unwrap();
    let Value::Tensor(t) = result else {
        panic!("expected tensor");
    };
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.shape.dims, vec![2]);
    t.validate_dtype_consistency()
        .expect("F32 reduce_sum axes output dtype/element invariant");
}

// Property sweep across BF16/F16/F32/F64. The reduction dtype-preservation
// fix in 9f9c800 affects all float dtypes, but the regression test above
// only covered F32. This sweep guards against future regressions specific
// to BF16, F16, or F64.
#[test]
fn property_reduce_sum_preserves_all_float_dtypes() {
    fn make_vec<F>(dtype: DType, values: &[f64], lit_for: F) -> Value
    where
        F: Fn(f64) -> Literal,
    {
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![values.len() as u32],
                },
                values.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let values = [1.0_f64, 2.0, 3.0, 4.0];
    let cases: Vec<(DType, Value)> = vec![
        (
            DType::BF16,
            make_vec(DType::BF16, &values, |v| Literal::from_bf16_f32(v as f32)),
        ),
        (
            DType::F16,
            make_vec(DType::F16, &values, |v| Literal::from_f16_f32(v as f32)),
        ),
        (
            DType::F32,
            make_vec(DType::F32, &values, |v| Literal::from_f32(v as f32)),
        ),
        (DType::F64, make_vec(DType::F64, &values, Literal::from_f64)),
    ];

    for (dtype, input) in cases {
        // Full reduction → scalar
        let result = eval_primitive(
            Primitive::ReduceSum,
            std::slice::from_ref(&input),
            &BTreeMap::new(),
        )
        .unwrap_or_else(|e| panic!("reduce_sum {dtype:?} failed: {e}"));
        match result {
            Value::Scalar(lit) => {
                assert!(
                    lit.matches_dtype(dtype),
                    "dtype={dtype:?}: scalar reduce_sum literal {lit:?} does not match",
                );
            }
            other => panic!("dtype={dtype:?}: expected scalar, got {other:?}"),
        }

        // Axis reduction → tensor
        let mut params = BTreeMap::new();
        params.insert("axes".to_string(), "0".to_string());
        let axis_result = eval_primitive(Primitive::ReduceSum, &[input], &params)
            .unwrap_or_else(|e| panic!("reduce_sum {dtype:?} axes failed: {e}"));
        // For a rank-1 input reducing axis 0, the result might be a scalar.
        match axis_result {
            Value::Scalar(lit) => assert!(
                lit.matches_dtype(dtype),
                "dtype={dtype:?}: axis-reduced scalar dtype mismatch"
            ),
            Value::Tensor(t) => {
                assert_eq!(t.dtype, dtype, "dtype={dtype:?}: axis tensor dtype");
                t.validate_dtype_consistency().unwrap_or_else(|e| {
                    panic!("dtype={dtype:?}: validate_dtype_consistency failed: {e}")
                });
            }
        }
    }
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================

fn make_complex64_tensor(shape: &[u32], pairs: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], pairs: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        Value::Scalar(l) => l.as_complex64().unwrap(),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        Value::Scalar(l) => l.as_complex128().unwrap(),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

fn assert_complex128_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

#[test]
fn oracle_reduce_sum_complex64_1d() {
    // Sum of [(1+2i), (3+4i), (5+6i)] = (9+12i)
    let input = make_complex64_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (9.0, 12.0), 1e-5, "sum of complex vector");
}

#[test]
fn oracle_reduce_sum_complex64_2d_full() {
    // Full reduction of 2x2 matrix
    // [[1+1i, 2+2i], [3+3i, 4+4i]] -> 10+10i
    let input = make_complex64_tensor(
        &[2, 2],
        vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
    );
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close(
        (re, im),
        (10.0, 10.0),
        1e-5,
        "full reduction of 2x2 complex",
    );
}

#[test]
fn oracle_reduce_sum_complex64_axis0() {
    // Reduce axis 0 of 2x3 matrix
    // [[1+0i, 2+0i, 3+0i], [4+0i, 5+0i, 6+0i]] axis=0 -> [5+0i, 7+0i, 9+0i]
    let input = make_complex64_tensor(
        &[2, 3],
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ],
    );
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "0".to_string());
    let result = eval_primitive(Primitive::ReduceSum, &[input], &params).unwrap();
    let vals = extract_complex64_vec(&result);

    assert_complex64_close(vals[0], (5.0, 0.0), 1e-5, "sum axis 0 [0]");
    assert_complex64_close(vals[1], (7.0, 0.0), 1e-5, "sum axis 0 [1]");
    assert_complex64_close(vals[2], (9.0, 0.0), 1e-5, "sum axis 0 [2]");
}

#[test]
fn oracle_reduce_sum_complex64_axis1() {
    // Reduce axis 1 of 2x3 matrix
    // [[1+1i, 2+2i, 3+3i], [4+4i, 5+5i, 6+6i]] axis=1 -> [6+6i, 15+15i]
    let input = make_complex64_tensor(
        &[2, 3],
        vec![
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
            (5.0, 5.0),
            (6.0, 6.0),
        ],
    );
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "1".to_string());
    let result = eval_primitive(Primitive::ReduceSum, &[input], &params).unwrap();
    let vals = extract_complex64_vec(&result);

    assert_complex64_close(vals[0], (6.0, 6.0), 1e-5, "sum axis 1 [0]");
    assert_complex64_close(vals[1], (15.0, 15.0), 1e-5, "sum axis 1 [1]");
}

#[test]
fn oracle_reduce_sum_complex64_single_element() {
    // Sum of single element returns that element
    let input = make_complex64_tensor(&[1], vec![(3.0, 4.0)]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (3.0, 4.0), 1e-6, "sum of single element");
}

#[test]
fn oracle_reduce_sum_complex128_1d() {
    // Sum with higher precision
    let input = make_complex128_tensor(&[3], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (9.0, 12.0), 1e-12, "sum Complex128");
}

#[test]
fn oracle_reduce_sum_complex64_empty() {
    // Sum of empty tensor is 0
    let input = make_complex64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.0, 0.0), 1e-6, "sum of empty = 0");
}

#[test]
fn oracle_reduce_sum_complex64_preserves_dtype() {
    let input = make_complex64_tensor(&[3], vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_reduce_sum_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
    let result = eval_primitive(Primitive::ReduceSum, &[input], &BTreeMap::new()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
