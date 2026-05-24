//! Oracle tests for Signbit primitive.
//!
//! signbit(x) = true if the sign bit is set (negative numbers and -0.0)
//!
//! Tests:
//! - Positive values: signbit = false
//! - Negative values: signbit = true
//! - Zero: signbit(+0.0) = false, signbit(-0.0) = true
//! - Infinity: signbit(+inf) = false, signbit(-inf) = true
//! - NaN: signbit preserves sign of NaN
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

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected Bool literal"),
            })
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            match &t.elements[0] {
                Literal::Bool(b) => *b,
                _ => panic!("expected Bool literal"),
            }
        }
        Value::Scalar(Literal::Bool(b)) => *b,
        _ => panic!("expected Bool scalar"),
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

// ======================== Positive Values ========================

#[test]
fn oracle_signbit_positive_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(+0.0) = false");
}

#[test]
fn oracle_signbit_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1.0) = false");
}

#[test]
fn oracle_signbit_large_positive() {
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1e100) = false");
}

#[test]
fn oracle_signbit_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(1e-100) = false");
}

// ======================== Negative Values ========================

#[test]
fn oracle_signbit_negative_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-0.0) = true");
}

#[test]
fn oracle_signbit_negative_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1.0) = true");
}

#[test]
fn oracle_signbit_large_negative() {
    let input = make_f64_tensor(&[], vec![-1e100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1e100) = true");
}

#[test]
fn oracle_signbit_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-1e-100) = true");
}

// ======================== Infinity ========================

#[test]
fn oracle_signbit_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "signbit(+inf) = false");
}

#[test]
fn oracle_signbit_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "signbit(-inf) = true");
}

// ======================== NaN ========================

#[test]
fn oracle_signbit_positive_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(
        !extract_bool_scalar(&result),
        "signbit(+NaN) = false (positive NaN)"
    );
}

#[test]
fn oracle_signbit_negative_nan() {
    let neg_nan = -f64::NAN;
    let input = make_f64_tensor(&[], vec![neg_nan]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert!(
        extract_bool_scalar(&result),
        "signbit(-NaN) = true (negative NaN)"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_signbit_vector() {
    let input = make_f64_tensor(&[5], vec![1.0, -1.0, 0.0, -0.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, true, false, true, false]
    );
}

#[test]
fn oracle_signbit_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, -1.0, -0.0, 0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, true, false]);
}

// ======================== Output DType ========================

#[test]
fn oracle_signbit_output_dtype() {
    let input = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "signbit output dtype should be Bool");
    }
}

#[test]
fn oracle_signbit_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_bool_vec(&result), vec![] as Vec<bool>);
}

#[test]
fn oracle_signbit_3d() {
    let input = make_f64_tensor(&[2, 1, 2], vec![1.0, -1.0, -0.0, 0.0]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, true, false]);
}

#[test]
fn oracle_signbit_subnormal() {
    // Subnormal positive and negative values
    let tiny_pos = f64::MIN_POSITIVE / 2.0;
    let tiny_neg = -f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![tiny_pos, tiny_neg]);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false, true]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_signbit_f32_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape { dims: vec![3] },
            vec![
                Literal::F32Bits(1.0_f32.to_bits()),
                Literal::F32Bits((-1.0_f32).to_bits()),
                Literal::F32Bits(0.0_f32.to_bits()),
            ],
        )
        .unwrap(),
    );
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false, true, false]);
}

#[test]
fn oracle_signbit_large_tensor() {
    let data: Vec<f64> = (0..100)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let input = make_f64_tensor(&[100], data);
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    let vals = extract_bool_vec(&result);
    assert_eq!(vals.len(), 100);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, i % 2 != 0, "signbit at index {i}");
    }
}

#[test]
fn oracle_signbit_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

// ======================== PROPERTY: signbit always returns Bool ========================

#[test]
fn property_signbit_always_returns_bool() {
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

    let values = [-1.0_f64, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Signbit, &[input], &no_params()).unwrap();
        assert_eq!(
            result.dtype(),
            DType::Bool,
            "signbit with {dtype:?} input should return Bool"
        );
    }
}
