//! Oracle tests for Heaviside primitive.
//!
//! heaviside(x, h0) returns:
//! - 0 if x < 0
//! - h0 if x == 0
//! - 1 if x > 0
//!
//! Tests:
//! - Basic: negative, zero, positive
//! - Different h0 values
//! - Special values: infinity, NaN
//! - Broadcast-compatible h0 inputs
//! - Tensor shapes

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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Basic Cases ========================

#[test]
fn oracle_heaviside_positive() {
    let x = make_f64_tensor(&[], vec![1.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(1, h0) = 1");
}

#[test]
fn oracle_heaviside_negative() {
    let x = make_f64_tensor(&[], vec![-1.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(-1, h0) = 0");
}

#[test]
fn oracle_heaviside_zero_half() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5, "heaviside(0, 0.5) = 0.5");
}

#[test]
fn oracle_heaviside_zero_one() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(0, 1) = 1");
}

#[test]
fn oracle_heaviside_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(0, 0) = 0");
}

// ======================== Small Values ========================

#[test]
fn oracle_heaviside_small_positive() {
    let x = make_f64_tensor(&[], vec![1e-10]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        1.0,
        "heaviside(small positive, h0) = 1"
    );
}

#[test]
fn oracle_heaviside_small_negative() {
    let x = make_f64_tensor(&[], vec![-1e-10]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "heaviside(small negative, h0) = 0"
    );
}

// ======================== Special Values ========================

#[test]
fn oracle_heaviside_positive_inf() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(inf, h0) = 1");
}

#[test]
fn oracle_heaviside_negative_inf() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(-inf, h0) = 0");
}

#[test]
fn oracle_heaviside_nan() {
    // NaN compares neither less-than nor greater-than zero, matching JAX's
    // fallback to h0.
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.5,
        "heaviside(NaN, 0.5) = 0.5"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_heaviside_vector() {
    let x = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let h0 = make_f64_tensor(&[5], vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn oracle_heaviside_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![-1.0, 0.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 2], vec![0.5, 0.5, 1.0, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 1.0, 1.0]);
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_heaviside_scalar_x_tensor_h0_broadcast() {
    // scalar x with tensor h0 (zero x, so h0 values are used)
    let x = scalar_f64(0.0);
    let h0 = make_f64_tensor(&[4], vec![0.1, 0.5, 0.9, 1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.1, 0.5, 0.9, 1.0]);
}

#[test]
fn oracle_heaviside_tensor_x_scalar_h0_broadcast() {
    // tensor x with scalar h0
    let x = make_f64_tensor(&[4], vec![-1.0, 0.0, 0.0, 1.0]);
    let h0 = scalar_f64(0.5);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 0.5, 1.0]);
}

#[test]
fn oracle_heaviside_singleton_x_vector_h0_broadcast() {
    // [1] x with [3] h0 -> [3] (x=0, so h0 values are used)
    let x = make_f64_tensor(&[1], vec![0.0]);
    let h0 = make_f64_tensor(&[3], vec![0.25, 0.5, 0.75]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.25, 0.5, 0.75]);
}

#[test]
fn oracle_heaviside_vector_x_singleton_h0_broadcast() {
    // [3] x with [1] h0 -> [3]
    let x = make_f64_tensor(&[3], vec![-1.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[1], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 1.0]);
}

#[test]
fn oracle_heaviside_column_x_matrix_h0_broadcast() {
    // [2, 1] x with [2, 3] h0 -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x=0, so use h0 values: 0.1, 0.2, 0.3
    assert_eq!(vals[0], 0.1);
    assert_eq!(vals[1], 0.2);
    assert_eq!(vals[2], 0.3);
    // Row 1: x=1 (positive), so result is 1.0 for all
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
    assert_eq!(vals[5], 1.0);
}

#[test]
fn oracle_heaviside_different_ranks_broadcast() {
    // [3] x with [2, 3] h0 -> [2, 3]
    let x = make_f64_tensor(&[3], vec![-1.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: heaviside(-1, 0.1)=0, heaviside(0, 0.2)=0.2, heaviside(1, 0.3)=1
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.2);
    assert_eq!(vals[2], 1.0);
    // Row 1: heaviside(-1, 0.4)=0, heaviside(0, 0.5)=0.5, heaviside(1, 0.6)=1
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 0.5);
    assert_eq!(vals[5], 1.0);
}

#[test]
fn oracle_heaviside_all_scalars_broadcast() {
    // scalar heaviside scalar -> scalar
    let x = scalar_f64(0.0);
    let h0 = scalar_f64(0.5);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5);
}

#[test]
fn oracle_heaviside_incompatible_shapes_error() {
    // [2] heaviside [3] should error
    let x = make_f64_tensor(&[2], vec![0.0, 1.0]);
    let h0 = make_f64_tensor(&[3], vec![0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_heaviside_matrix_scalar_h0_broadcast() {
    let x = make_f64_tensor(
        &[3, 3],
        vec![-2.0, 0.0, 3.0, 5.0, -1.0, 0.0, 0.0, 7.0, -3.0],
    );
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0]
    );
}

#[test]
fn oracle_heaviside_matrix_row_h0_broadcast() {
    let x = make_f64_tensor(
        &[3, 3],
        vec![-2.0, 0.0, 3.0, 5.0, -1.0, 0.0, 0.0, 7.0, -3.0],
    );
    let h0 = make_f64_tensor(&[3], vec![2.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]
    );
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_heaviside_3d() {
    let x = make_f64_tensor(&[2, 2, 2], vec![-1.0, 0.0, 1.0, -2.0, 3.0, 0.0, -0.5, 0.5]);
    let h0 = make_f64_tensor(&[2, 2, 2], vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0]
    );
}

#[test]
fn oracle_heaviside_empty() {
    let x = make_f64_tensor(&[0], vec![]);
    let h0 = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_heaviside_2d_empty() {
    let x =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let h0 =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_heaviside_preserves_dtype() {
    let x = make_f64_tensor(&[3], vec![-1.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[3], vec![0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_heaviside_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let x = make_f64_tensor(&[2], vec![subnormal, -subnormal]);
    let h0 = make_f64_tensor(&[2], vec![0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // Subnormal positive should return 1, subnormal negative should return 0
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 0.0);
}

#[test]
fn oracle_heaviside_neg_zero() {
    // -0.0 compares equal to 0.0, so should return h0
    let x = make_f64_tensor(&[], vec![-0.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_heaviside_preserves_all_float_dtypes() {
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

    let x_values = [-1.0_f64, 0.0, 1.0];
    let h0_values = [0.5_f64, 0.5, 0.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let x = make_vec(dtype, &x_values);
        let h0 = make_vec(dtype, &h0_values);
        let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "heaviside {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== METAMORPHIC: mathematical identities ========================

#[test]
fn metamorphic_heaviside_complement_sums_to_one() {
    // heaviside(x, 0.5) + heaviside(-x, 0.5) = 1 for all x
    let x = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let neg_x = make_f64_tensor(&[5], vec![2.0, 1.0, 0.0, -1.0, -2.0]);
    let h0 = make_f64_tensor(&[5], vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let result1 = eval_primitive(Primitive::Heaviside, &[x, h0.clone()], &no_params()).unwrap();
    let result2 = eval_primitive(Primitive::Heaviside, &[neg_x, h0], &no_params()).unwrap();
    let vals1 = extract_f64_vec(&result1);
    let vals2 = extract_f64_vec(&result2);
    for (i, (&v1, &v2)) in vals1.iter().zip(vals2.iter()).enumerate() {
        assert!(
            (v1 + v2 - 1.0).abs() < 1e-10,
            "heaviside(x, 0.5) + heaviside(-x, 0.5) should equal 1 at index {i}: got {}",
            v1 + v2
        );
    }
}

#[test]
fn metamorphic_heaviside_idempotent_on_positive() {
    // heaviside(heaviside(x, h), h) = heaviside(x, h) when h > 0
    // For x positive: heaviside(x, h) = 1, then heaviside(1, h) = 1
    // For x negative: heaviside(x, h) = 0, then heaviside(0, h) = h
    // For x = 0: heaviside(0, h) = h, then heaviside(h, h) depends on sign of h
    let x = make_f64_tensor(&[3], vec![-1.0, 1.0, 2.0]);
    let h0 = make_f64_tensor(&[3], vec![0.5, 0.5, 0.5]);
    let first = eval_primitive(Primitive::Heaviside, &[x, h0.clone()], &no_params()).unwrap();
    let second = eval_primitive(Primitive::Heaviside, &[first.clone(), h0], &no_params()).unwrap();
    let first_vals = extract_f64_vec(&first);
    let second_vals = extract_f64_vec(&second);
    // first = [0, 1, 1], second = heaviside([0, 1, 1], 0.5) = [0.5, 1, 1]
    for (i, &v) in second_vals.iter().enumerate() {
        let expected = if first_vals[i] > 0.0 { 1.0 } else { 0.5 };
        assert!(
            (v - expected).abs() < 1e-10,
            "heaviside idempotent at index {i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn metamorphic_heaviside_h0_independent_for_nonzero() {
    // For x != 0, the h0 value doesn't affect the result
    let x = make_f64_tensor(&[4], vec![-2.0, -1.0, 1.0, 2.0]);
    let h0_a = make_f64_tensor(&[4], vec![0.0, 0.25, 0.5, 0.75]);
    let h0_b = make_f64_tensor(&[4], vec![1.0, 0.0, 0.9, 0.1]);
    let result_a = eval_primitive(Primitive::Heaviside, &[x.clone(), h0_a], &no_params()).unwrap();
    let result_b = eval_primitive(Primitive::Heaviside, &[x, h0_b], &no_params()).unwrap();
    let vals_a = extract_f64_vec(&result_a);
    let vals_b = extract_f64_vec(&result_b);
    for (i, (&va, &vb)) in vals_a.iter().zip(vals_b.iter()).enumerate() {
        assert!(
            (va - vb).abs() < 1e-10,
            "for x != 0, heaviside should be independent of h0 at index {i}: got {} vs {}",
            va,
            vb
        );
    }
}

#[test]
fn metamorphic_heaviside_output_bounded() {
    // heaviside output is always in [0, 1] when h0 is in [0, 1]
    let x = make_f64_tensor(&[5], vec![-100.0, -0.001, 0.0, 0.001, 100.0]);
    let h0 = make_f64_tensor(&[5], vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "heaviside output should be in [0, 1] at index {i}: got {v}"
        );
    }
}
