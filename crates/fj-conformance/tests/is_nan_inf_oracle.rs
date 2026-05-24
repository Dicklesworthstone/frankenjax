//! Oracle tests for IsNan and IsInf predicates.
//!
//! isnan(x) = true if x is NaN
//! isinf(x) = true if x is +inf or -inf
//!
//! Tests:
//! - Normal values: isnan/isinf return false
//! - Zero: neither NaN nor Inf
//! - Infinity: isinf true, isnan false
//! - NaN: isnan true, isinf false
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

fn extract_bool_literal(literal: &Literal) -> bool {
    match literal {
        Literal::Bool(value) => *value,
        other => {
            assert!(matches!(other, Literal::Bool(_)), "expected Bool literal");
            false
        }
    }
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(extract_bool_literal).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            extract_bool_literal(&t.elements[0])
        }
        Value::Scalar(literal) => extract_bool_literal(literal),
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

// ======================== IsNan: Normal Values ========================

#[test]
fn oracle_isnan_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(0) = false");
}

#[test]
fn oracle_isnan_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-0) = false");
}

#[test]
fn oracle_isnan_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(1) = false");
}

#[test]
fn oracle_isnan_negative() {
    let input = make_f64_tensor(&[], vec![-42.5]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-42.5) = false");
}

// ======================== IsNan: Special Values ========================

#[test]
fn oracle_isnan_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(+inf) = false");
}

#[test]
fn oracle_isnan_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isnan(-inf) = false");
}

#[test]
fn oracle_isnan_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isnan(NaN) = true");
}

// ======================== IsInf: Normal Values ========================

#[test]
fn oracle_isinf_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(0) = false");
}

#[test]
fn oracle_isinf_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(-0) = false");
}

#[test]
fn oracle_isinf_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(1) = false");
}

#[test]
fn oracle_isinf_large() {
    let input = make_f64_tensor(&[], vec![1e308]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(1e308) = false");
}

// ======================== IsInf: Special Values ========================

#[test]
fn oracle_isinf_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isinf(+inf) = true");
}

#[test]
fn oracle_isinf_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(extract_bool_scalar(&result), "isinf(-inf) = true");
}

#[test]
fn oracle_isinf_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert!(!extract_bool_scalar(&result), "isinf(NaN) = false");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_isnan_vector() {
    let input = make_f64_tensor(
        &[5],
        vec![0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, false, true, false, false]
    );
}

#[test]
fn oracle_isinf_vector() {
    let input = make_f64_tensor(
        &[5],
        vec![0.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(
        extract_bool_vec(&result),
        vec![false, false, false, true, true]
    );
}

#[test]
fn oracle_isnan_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, f64::NAN, f64::INFINITY, -1.0]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true, false, false]);
}

#[test]
fn oracle_isinf_matrix() {
    let input = make_f64_tensor(
        &[2, 2],
        vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    );
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![false, false, true, true]);
}

// ======================== Output DType ========================

#[test]
fn oracle_isnan_output_dtype() {
    let input = make_f64_tensor(&[2], vec![0.0, f64::NAN]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "isnan output dtype should be Bool");
    }
}

#[test]
fn oracle_isinf_output_dtype() {
    let input = make_f64_tensor(&[2], vec![0.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    if let Value::Tensor(t) = result {
        assert_eq!(t.dtype, DType::Bool, "isinf output dtype should be Bool");
    }
}

#[test]
fn oracle_isnan_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_bool_vec(&result), vec![] as Vec<bool>);
}

#[test]
fn oracle_isinf_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_bool_vec(&result), vec![] as Vec<bool>);
}

#[test]
fn oracle_isnan_3d() {
    let input = make_f64_tensor(&[2, 1, 2], vec![f64::NAN, 1.0, 2.0, f64::NAN]);
    let result = eval_primitive(Primitive::IsNan, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, false, true]);
}

#[test]
fn oracle_isinf_subnormal() {
    // Subnormal values are not infinite
    let tiny = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![tiny, -tiny]);
    let result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false, false]);
}

// ======================== PROPERTY: isnan/isinf always return Bool ========================

#[test]
fn property_isnan_isinf_always_return_bool() {
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

    let values = [1.0_f64, f64::INFINITY, f64::NAN];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let isnan_result =
            eval_primitive(Primitive::IsNan, &[input.clone()], &no_params()).unwrap();
        let isinf_result = eval_primitive(Primitive::IsInf, &[input], &no_params()).unwrap();
        assert_eq!(
            isnan_result.dtype(),
            DType::Bool,
            "is_nan with {dtype:?} input should return Bool"
        );
        assert_eq!(
            isinf_result.dtype(),
            DType::Bool,
            "is_inf with {dtype:?} input should return Bool"
        );
    }
}
