//! Oracle tests for ReduceProd primitive.
//!
//! Tests product reduction semantics:
//! - Full reduction: product of all elements
//! - Axis reduction: product along specified axes
//! - Identity: empty product is 1
//! - Zero absorption: any zero makes product zero

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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
        _ => unreachable!("expected tensor"),
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
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(l) => l.as_i64().unwrap(),
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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "axes".to_string(),
        axes.iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== Basic 1D Full Reduction ========================

#[test]
fn oracle_reduce_prod_1d_basic() {
    // 1 * 2 * 3 * 4 * 5 = 120
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 120);
}

#[test]
fn oracle_reduce_prod_1d_with_ones() {
    // 1 * 1 * 1 * 1 = 1
    let input = make_i64_tensor(&[4], vec![1, 1, 1, 1]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1);
}

#[test]
fn oracle_reduce_prod_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 42);
}

#[test]
fn oracle_reduce_prod_scalar() {
    let input = make_i64_tensor(&[], vec![7]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 7);
}

// ======================== Zero Absorption ========================

#[test]
fn oracle_reduce_prod_with_zero() {
    // Any zero makes the product zero
    let input = make_i64_tensor(&[5], vec![1, 2, 0, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_all_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_zero_at_start() {
    let input = make_i64_tensor(&[3], vec![0, 5, 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_prod_zero_at_end() {
    let input = make_i64_tensor(&[3], vec![5, 10, 0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Negative Numbers ========================

#[test]
fn oracle_reduce_prod_negative_even() {
    // (-1) * (-2) = 2 (even number of negatives -> positive)
    let input = make_i64_tensor(&[2], vec![-1, -2]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 2);
}

#[test]
fn oracle_reduce_prod_negative_odd() {
    // (-1) * (-2) * (-3) = -6 (odd number of negatives -> negative)
    let input = make_i64_tensor(&[3], vec![-1, -2, -3]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), -6);
}

#[test]
fn oracle_reduce_prod_mixed_signs() {
    // (-2) * 3 * (-4) = 24
    let input = make_i64_tensor(&[3], vec![-2, 3, -4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

// ======================== 2D Full Reduction ========================

#[test]
fn oracle_reduce_prod_2d_full() {
    // [[1, 2], [3, 4]] -> 1*2*3*4 = 24
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

#[test]
fn oracle_reduce_prod_2d_with_zero() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 0, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

// ======================== Axis Reduction ========================

#[test]
fn oracle_reduce_prod_2d_axis0() {
    // [[1, 2, 3], [4, 5, 6]] -> axis 0 -> [1*4, 2*5, 3*6] = [4, 10, 18]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![4, 10, 18]);
}

#[test]
fn oracle_reduce_prod_2d_axis1() {
    // [[1, 2, 3], [4, 5, 6]] -> axis 1 -> [1*2*3, 4*5*6] = [6, 120]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![6, 120]);
}

#[test]
fn oracle_reduce_prod_3d_axis0() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 0 -> [[1*5,2*6],[3*7,4*8]] = [[5,12],[21,32]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![5, 12, 21, 32]);
}

#[test]
fn oracle_reduce_prod_3d_axis1() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 1 -> [[1*3,2*4],[5*7,6*8]] = [[3,8],[35,48]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 8, 35, 48]);
}

#[test]
fn oracle_reduce_prod_3d_axis2() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] -> axis 2 -> [[1*2,3*4],[5*6,7*8]] = [[2,12],[30,56]]
    let input = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![2, 12, 30, 56]);
}

#[test]
fn oracle_reduce_prod_multiple_axes() {
    // [[1, 2], [3, 4]] -> axes [0, 1] -> full reduction -> 24
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0, 1])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 24);
}

// ======================== Float Tests ========================

#[test]
fn oracle_reduce_prod_f64_basic() {
    // 1.5 * 2.0 * 4.0 = 12.0
    let input = make_f64_tensor(&[3], vec![1.5, 2.0, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_with_fractions() {
    // 0.5 * 0.5 * 4.0 = 1.0
    let input = make_f64_tensor(&[3], vec![0.5, 0.5, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_negative() {
    // -1.0 * 2.0 * -3.0 = 6.0
    let input = make_f64_tensor(&[3], vec![-1.0, 2.0, -3.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_prod_f64_with_zero() {
    let input = make_f64_tensor(&[3], vec![5.0, 0.0, 10.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_reduce_prod_f64_axis() {
    // [[1.0, 2.0], [3.0, 4.0]] -> axis 0 -> [3.0, 8.0]
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 8.0).abs() < 1e-10);
}

// ======================== Special Float Values ========================

#[test]
fn oracle_reduce_prod_f64_infinity() {
    // Large values can overflow to infinity
    let input = make_f64_tensor(&[2], vec![1e200, 1e200]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn oracle_reduce_prod_f64_underflow() {
    // Small values can underflow to zero
    let input = make_f64_tensor(&[2], vec![1e-200, 1e-200]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0);
}

#[test]
fn oracle_reduce_prod_f64_nan_propagates() {
    let input = make_f64_tensor(&[3], vec![1.0, f64::NAN, 3.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan());
}

#[test]
fn oracle_reduce_prod_f64_inf_times_zero() {
    // inf * 0 = NaN
    let input = make_f64_tensor(&[2], vec![f64::INFINITY, 0.0]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan());
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reduce_prod_large_values() {
    // Test with moderately large values that don't overflow
    let input = make_i64_tensor(&[4], vec![10, 10, 10, 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 10000);
}

#[test]
fn oracle_reduce_prod_powers_of_two() {
    // 2^10 = 1024
    let input = make_i64_tensor(&[10], vec![2; 10]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 1024);
}

#[test]
fn oracle_reduce_prod_factorials() {
    // 5! = 120
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 120);

    // 6! = 720
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result), 720);
}

// ======================== Associativity Check ========================

#[test]
fn oracle_reduce_prod_associativity() {
    // Product is associative: (a*b)*c = a*(b*c)
    // Verify the order doesn't matter for integer product
    let input1 = make_i64_tensor(&[4], vec![2, 3, 5, 7]);
    let input2 = make_i64_tensor(&[4], vec![7, 5, 3, 2]);
    let result1 = eval_primitive(Primitive::ReduceProd, &[input1], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::ReduceProd, &[input2], &no_params()).unwrap();
    assert_eq!(extract_i64_scalar(&result1), extract_i64_scalar(&result2));
    assert_eq!(extract_i64_scalar(&result1), 210); // 2*3*5*7
}

// ======================== Metamorphic Tests ========================

fn concat_params(axis: i64) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("dimension".to_string(), axis.to_string());
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

#[test]
fn metamorphic_reduce_prod_distributive_over_concat() {
    // prod(concat(x, y)) = prod(x) * prod(y)
    let x = vec![2.0, 3.0];
    let y = vec![4.0, 5.0];

    let prod_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[2], x.clone())],
            &no_params(),
        )
        .unwrap(),
    );

    let prod_y = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[2], y.clone())],
            &no_params(),
        )
        .unwrap(),
    );

    let concat_xy = eval_primitive(
        Primitive::Concatenate,
        &[make_f64_tensor(&[2], x), make_f64_tensor(&[2], y)],
        &concat_params(0),
    )
    .unwrap();

    let prod_concat = extract_f64_scalar(
        &eval_primitive(Primitive::ReduceProd, &[concat_xy], &no_params()).unwrap(),
    );

    assert_close(
        prod_concat,
        prod_x * prod_y,
        1e-14,
        "prod(concat(x, y)) = prod(x) * prod(y)",
    );
}

#[test]
fn metamorphic_reduce_prod_one_identity() {
    // prod(x) * prod([1, 1, 1]) = prod(x)
    let x = vec![2.0, 3.0, 5.0];
    let ones = vec![1.0, 1.0, 1.0];

    let prod_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[3], x.clone())],
            &no_params(),
        )
        .unwrap(),
    );

    let prod_ones = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[3], ones)],
            &no_params(),
        )
        .unwrap(),
    );

    assert_eq!(prod_ones, 1.0, "prod of ones = 1");
    assert_eq!(prod_x * prod_ones, prod_x, "prod(x) * prod(ones) = prod(x)");
}

#[test]
fn metamorphic_reduce_prod_reciprocal_cancels() {
    // prod(x) * prod(1/x) = 1 for non-zero x
    let x = vec![2.0, 3.0, 5.0];
    let recip_x: Vec<f64> = x.iter().map(|v| 1.0 / v).collect();

    let prod_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[3], x)],
            &no_params(),
        )
        .unwrap(),
    );

    let prod_recip_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[3], recip_x)],
            &no_params(),
        )
        .unwrap(),
    );

    assert_close(prod_x * prod_recip_x, 1.0, 1e-14, "prod(x) * prod(1/x) = 1");
}

#[test]
fn metamorphic_reduce_prod_commutative() {
    // prod(x) * prod(y) = prod(y) * prod(x)
    let x = vec![2.0, 3.0];
    let y = vec![5.0, 7.0];

    let prod_x = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[2], x.clone())],
            &no_params(),
        )
        .unwrap(),
    );

    let prod_y = extract_f64_scalar(
        &eval_primitive(
            Primitive::ReduceProd,
            &[make_f64_tensor(&[2], y.clone())],
            &no_params(),
        )
        .unwrap(),
    );

    assert_eq!(prod_x * prod_y, prod_y * prod_x, "prod(x) * prod(y) = prod(y) * prod(x)");
}

#[test]
fn metamorphic_reduce_prod_zero_absorbs() {
    // prod(concat(x, [0])) = 0 regardless of x
    let x = vec![1000.0, 2000.0, 3000.0];
    let zero = vec![0.0];

    let concat_with_zero = eval_primitive(
        Primitive::Concatenate,
        &[make_f64_tensor(&[3], x), make_f64_tensor(&[1], zero)],
        &concat_params(0),
    )
    .unwrap();

    let prod_with_zero = extract_f64_scalar(
        &eval_primitive(Primitive::ReduceProd, &[concat_with_zero], &no_params()).unwrap(),
    );

    assert_eq!(prod_with_zero, 0.0, "prod with zero = 0");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_reduce_prod_preserves_all_float_dtypes() {
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
        Value::Tensor(
            TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap(),
        )
    }

    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        // ReduceProd returns scalar or tensor depending on impl
        assert_eq!(result.dtype(), dtype, "reduce_prod {dtype:?}: dtype mismatch");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: shape.to_vec() },
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
            Shape { dims: shape.to_vec() },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        Value::Scalar(l) => l.as_complex64().unwrap(),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        Value::Scalar(l) => l.as_complex128().unwrap(),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_complex64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_complex128().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]

fn oracle_reduce_prod_complex64_1d() {
    // (1+0i) * (2+0i) * (3+0i) = 6+0i
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!((re - 6.0).abs() < 1e-5);
    assert!(im.abs() < 1e-5);
}

#[test]

fn oracle_reduce_prod_complex64_with_imaginary() {
    // i * i = -1, so i * i * i = -i
    let input = make_complex64_tensor(&[3], vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!(re.abs() < 1e-5, "expected 0, got {re}");
    assert!((im - (-1.0)).abs() < 1e-5, "expected -1, got {im}");
}

#[test]

fn oracle_reduce_prod_complex128_1d() {
    // (1+1i) * (1-1i) = 1 - i^2 = 2
    let input = make_complex128_tensor(&[2], vec![(1.0, 1.0), (1.0, -1.0)]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert!((re - 2.0).abs() < 1e-10);
    assert!(im.abs() < 1e-10);
}

#[test]

fn oracle_reduce_prod_complex64_2d_axis0() {
    // [[1+0i, 2+0i], [3+0i, 4+0i]] along axis 0 => [3+0i, 8+0i]
    let input = make_complex64_tensor(&[2, 2], vec![
        (1.0, 0.0), (2.0, 0.0),
        (3.0, 0.0), (4.0, 0.0),
    ]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 3.0).abs() < 1e-5);
    assert!((vals[1].0 - 8.0).abs() < 1e-5);
}

#[test]

fn oracle_reduce_prod_complex64_with_zero() {
    // (2+0i) * (0+0i) = 0+0i
    let input = make_complex64_tensor(&[2], vec![(2.0, 0.0), (0.0, 0.0)]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!(re.abs() < 1e-5);
    assert!(im.abs() < 1e-5);
}

#[test]

fn oracle_reduce_prod_complex64_single_element() {
    let input = make_complex64_tensor(&[1], vec![(42.0, -17.0)]);
    let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!((re - 42.0).abs() < 1e-5);
    assert!((im - (-17.0)).abs() < 1e-5);
}

#[test]

fn property_reduce_prod_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[3], vec![
                (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
            ]),
            DType::Complex128 => make_complex128_tensor(&[3], vec![
                (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
            ]),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::ReduceProd, &[input], &no_params())
            .expect("reduce_prod should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "reduce_prod {dtype:?}: dtype mismatch");
    }
}
