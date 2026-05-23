//! Oracle tests for Log2 and Exp2 primitives.
//!
//! log2(x) = log base 2 of x
//! exp2(x) = 2^x
//!
//! Tests:
//! - Identity: log2(2^n) = n
//! - Inverse: exp2(log2(x)) = x
//! - Special values: log2(1) = 0, exp2(0) = 1
//! - Edge cases: infinity, NaN, negative inputs
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

// ======================== Log2: Basic Values ========================

#[test]
fn oracle_log2_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "log2(1) = 0");
}

#[test]
fn oracle_log2_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "log2(2) = 1");
}

#[test]
fn oracle_log2_four() {
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "log2(4) = 2");
}

#[test]
fn oracle_log2_power_of_two() {
    for n in 0..10 {
        let val = (1_u64 << n) as f64;
        let input = make_f64_tensor(&[], vec![val]);
        let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
        assert_eq!(extract_f64_scalar(&result), n as f64, "log2(2^{}) = {}", n, n);
    }
}

#[test]
fn oracle_log2_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "log2(0.5) = -1");
}

// ======================== Log2: Special Values ========================

#[test]
fn oracle_log2_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "log2(0) = -inf"
    );
}

#[test]
fn oracle_log2_negative() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log2(-1) = NaN");
}

#[test]
fn oracle_log2_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "log2(inf) = inf"
    );
}

#[test]
fn oracle_log2_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log2(NaN) = NaN");
}

// ======================== Exp2: Basic Values ========================

#[test]
fn oracle_exp2_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "exp2(0) = 1");
}

#[test]
fn oracle_exp2_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "exp2(1) = 2");
}

#[test]
fn oracle_exp2_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "exp2(2) = 4");
}

#[test]
fn oracle_exp2_ten() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1024.0, "exp2(10) = 1024");
}

#[test]
fn oracle_exp2_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5, "exp2(-1) = 0.5");
}

// ======================== Exp2: Special Values ========================

#[test]
fn oracle_exp2_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "exp2(inf) = inf"
    );
}

#[test]
fn oracle_exp2_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "exp2(-inf) = 0");
}

#[test]
fn oracle_exp2_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "exp2(NaN) = NaN");
}

// ======================== Inverse Relationship ========================

#[test]
fn oracle_exp2_log2_inverse() {
    for x in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let log2_result = eval_primitive(Primitive::Log2, &[input.clone()], &no_params()).unwrap();
        let roundtrip =
            eval_primitive(Primitive::Exp2, &[log2_result], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - x).abs() / x < 1e-10,
            "exp2(log2({})) = {} (expected {})",
            x,
            actual,
            x
        );
    }
}

#[test]
fn oracle_log2_exp2_inverse() {
    for x in [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let exp2_result = eval_primitive(Primitive::Exp2, &[input.clone()], &no_params()).unwrap();
        let roundtrip =
            eval_primitive(Primitive::Log2, &[exp2_result], &no_params()).unwrap();
        let actual = extract_f64_scalar(&roundtrip);
        assert!(
            (actual - x).abs() < 1e-10,
            "log2(exp2({})) = {} (expected {})",
            x,
            actual,
            x
        );
    }
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_log2_vector() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 4.0, 8.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn oracle_exp2_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 4.0, 8.0]);
}

#[test]
fn oracle_log2_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 4.0, 16.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 4.0]);
}

#[test]
fn oracle_exp2_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, 2.0, 4.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 4.0, 16.0]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_log2_3d_shape() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[7], 7.0);
}

#[test]
fn oracle_exp2_3d_shape() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[7], 128.0);
}

#[test]
fn oracle_log2_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_exp2_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert!(extract_f64_vec(&result).is_empty());
}

#[test]
fn oracle_log2_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_exp2_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_log2_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 4.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_exp2_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_log2_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::NEG_INFINITY, "log2(-0) = -inf");
}

#[test]
fn oracle_exp2_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "exp2(-0) = 1");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_log2_preserves_all_float_dtypes() {
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

    // log2 domain is x > 0
    let values = [0.5_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Log2, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "log2 {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_exp2_preserves_all_float_dtypes() {
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

    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Exp2, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "exp2 {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
