//! Oracle tests for the ConvertElementType primitive.
//!
//! Upstream `jax.lax.convert_element_type` is an elementwise cast that preserves
//! operand shape while changing dtype.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn convert_params(new_dtype: &str) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("new_dtype".to_string(), new_dtype.to_string());
    params
}

fn f64_tensor(shape: &[u32], values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .expect("valid f64 tensor"),
    )
}

fn i64_tensor(shape: &[u32], values: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            values.iter().copied().map(Literal::I64).collect(),
        )
        .expect("valid i64 tensor"),
    )
}

fn bool_tensor(shape: &[u32], values: &[bool]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            values.iter().copied().map(Literal::Bool).collect(),
        )
        .expect("valid bool tensor"),
    )
}

fn u32_tensor(shape: &[u32], values: &[u32]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape {
                dims: shape.to_vec(),
            },
            values.iter().copied().map(Literal::U32).collect(),
        )
        .expect("valid u32 tensor"),
    )
}

fn convert(input: Value, new_dtype: &str) -> Result<Value, fj_lax::EvalError> {
    eval_primitive(
        Primitive::ConvertElementType,
        &[input],
        &convert_params(new_dtype),
    )
}

fn shape(value: &Value) -> Shape {
    value.as_tensor().expect("expected tensor").shape.clone()
}

fn f32_values(value: &Value) -> Vec<Option<f32>> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| match literal {
            Literal::F32Bits(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        })
        .collect()
}

fn i64_values(value: &Value) -> Vec<i64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_i64().expect("expected i64 literal"))
        .collect()
}

fn bool_values(value: &Value) -> Vec<Option<bool>> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| match literal {
            Literal::Bool(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn u64_values(value: &Value) -> Vec<u64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_u64().expect("expected unsigned literal"))
        .collect()
}

#[test]
fn convert_element_type_f64_tensor_to_f32_preserves_shape() {
    let input = f64_tensor(&[2, 2], &[1.5, -2.25, 0.0, 7.75]);
    let result = convert(input, "f32").expect("f64 to f32 conversion should succeed");

    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(shape(&result), Shape { dims: vec![2, 2] });
    assert_eq!(
        f32_values(&result),
        vec![Some(1.5_f32), Some(-2.25_f32), Some(0.0), Some(7.75_f32)]
    );
}

#[test]
fn convert_element_type_f64_tensor_to_i64_truncates_toward_zero() {
    let input = f64_tensor(&[4], &[1.9, -2.9, 0.0, 3.1]);
    let result = convert(input, "i64").expect("f64 to i64 conversion should succeed");

    assert_eq!(result.dtype(), DType::I64);
    assert_eq!(i64_values(&result), vec![1, -2, 0, 3]);
}

#[test]
fn convert_element_type_i64_tensor_to_f64_preserves_values() {
    let input = i64_tensor(&[3], &[-3, 0, 42]);
    let result = convert(input, "f64").expect("i64 to f64 conversion should succeed");

    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(
        result
            .as_tensor()
            .expect("expected tensor")
            .elements
            .iter()
            .map(|literal| literal.as_f64().expect("expected f64 literal"))
            .collect::<Vec<_>>(),
        vec![-3.0, 0.0, 42.0]
    );
}

#[test]
fn convert_element_type_bool_tensor_to_i64_uses_zero_and_one() {
    let input = bool_tensor(&[4], &[true, false, true, false]);
    let result = convert(input, "i64").expect("bool to i64 conversion should succeed");

    assert_eq!(result.dtype(), DType::I64);
    assert_eq!(i64_values(&result), vec![1, 0, 1, 0]);
}

#[test]
fn convert_element_type_i64_tensor_to_bool_uses_zero_truthiness() {
    let input = i64_tensor(&[4], &[0, -2, 3, 0]);
    let result = convert(input, "bool").expect("i64 to bool conversion should succeed");

    assert_eq!(result.dtype(), DType::Bool);
    assert_eq!(
        bool_values(&result),
        vec![Some(false), Some(true), Some(true), Some(false)]
    );
}

#[test]
fn convert_element_type_u32_tensor_to_u64_preserves_unsigned_values() {
    let input = u32_tensor(&[3], &[0, 17, u32::MAX]);
    let result = convert(input, "u64").expect("u32 to u64 conversion should succeed");

    assert_eq!(result.dtype(), DType::U64);
    assert_eq!(u64_values(&result), vec![0, 17, u64::from(u32::MAX)]);
}

#[test]
fn convert_element_type_scalar_f64_to_complex128_sets_zero_imaginary_part() {
    let result = convert(Value::Scalar(Literal::from_f64(-3.5)), "complex128")
        .expect("f64 to complex128 conversion should succeed");

    assert_eq!(result.dtype(), DType::Complex128);
    assert_eq!(result.as_complex128_scalar(), Some((-3.5, 0.0)));
}

#[test]
fn convert_element_type_scalar_complex128_to_f64_uses_real_component() {
    let result = convert(Value::scalar_complex128(4.25, -9.0), "f64")
        .expect("complex128 to f64 conversion should succeed");

    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(result.as_f64_scalar(), Some(4.25));
}

#[test]
fn convert_element_type_rejects_missing_new_dtype() {
    let err = eval_primitive(
        Primitive::ConvertElementType,
        &[Value::Scalar(Literal::from_f64(1.0))],
        &BTreeMap::new(),
    )
    .expect_err("missing new_dtype should fail");

    assert!(
        err.to_string()
            .contains("missing required param 'new_dtype'"),
        "unexpected missing dtype error: {err}"
    );
}

#[test]
fn convert_element_type_rejects_unknown_dtype_name() {
    let err = convert(Value::Scalar(Literal::from_f64(1.0)), "float80")
        .expect_err("unknown dtype should fail");

    assert!(
        err.to_string().contains("unsupported dtype"),
        "unexpected unknown dtype error: {err}"
    );
}

#[test]
fn convert_element_type_rejects_wrong_arity() {
    let err = eval_primitive(Primitive::ConvertElementType, &[], &convert_params("f64"))
        .expect_err("missing operand should fail");

    assert!(
        err.to_string().contains("expected 1"),
        "unexpected arity error: {err}"
    );
}

#[test]
fn convert_element_type_f64_to_i32() {
    let input = f64_tensor(&[3], &[1.5, -2.7, 100.9]);
    let result = convert(input, "i32").expect("f64 to i32 conversion should succeed");

    assert_eq!(result.dtype(), DType::I32);
    let values: Vec<i64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().expect("expected i32 stored as i64"))
        .collect();
    assert_eq!(values, vec![1, -2, 100]);
}

#[test]
fn convert_element_type_empty_tensor() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
    let result = convert(input, "i64").expect("empty tensor conversion should succeed");

    assert_eq!(result.dtype(), DType::I64);
    let tensor = result.as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![0]);
    assert!(tensor.elements.is_empty());
}

#[test]
fn convert_element_type_noop_same_dtype() {
    let input = f64_tensor(&[2], &[1.5, 2.5]);
    let result = convert(input, "f64").expect("noop conversion should succeed");

    assert_eq!(result.dtype(), DType::F64);
    let values: Vec<f64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(values, vec![1.5, 2.5]);
}

#[test]
fn convert_element_type_rank2_shape_preserved() {
    let input = i64_tensor(&[2, 3], &[1, 2, 3, 4, 5, 6]);
    let result = convert(input, "f32").expect("rank-2 conversion should succeed");

    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(shape(&result).dims, vec![2, 3]);
}

// ======================== Additional Coverage ========================

#[test]
fn convert_element_type_rank3_shape_preserved() {
    let input = i64_tensor(&[2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
    let result = convert(input, "f64").expect("rank-3 conversion should succeed");

    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(shape(&result).dims, vec![2, 2, 2]);
}

#[test]
fn convert_element_type_f64_to_complex128() {
    let input = f64_tensor(&[3], &[1.0, 2.0, 3.0]);
    let result = convert(input, "complex128").expect("f64 to complex128 should succeed");

    assert_eq!(result.dtype(), DType::Complex128);
    assert_eq!(shape(&result).dims, vec![3]);
}

#[test]
fn convert_element_type_special_values() {
    let input = f64_tensor(&[4], &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0]);
    let result = convert(input, "f32").expect("special values conversion should succeed");

    assert_eq!(result.dtype(), DType::F32);
    let vals = f32_values(&result);
    assert!(vals[0].unwrap().is_infinite() && vals[0].unwrap().is_sign_positive());
    assert!(vals[1].unwrap().is_infinite() && vals[1].unwrap().is_sign_negative());
    assert!(vals[2].unwrap().is_nan());
    assert_eq!(vals[3], Some(0.0));
}

#[test]
fn convert_element_type_u64_to_i64() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::U64,
            Shape { dims: vec![3] },
            vec![Literal::U64(0), Literal::U64(100), Literal::U64(1000)],
        )
        .expect("valid u64 tensor"),
    );
    let result = convert(input, "i64").expect("u64 to i64 conversion should succeed");

    assert_eq!(result.dtype(), DType::I64);
    assert_eq!(i64_values(&result), vec![0, 100, 1000]);
}

#[test]
fn convert_element_type_bool_to_f64() {
    let input = bool_tensor(&[3], &[true, false, true]);
    let result = convert(input, "f64").expect("bool to f64 conversion should succeed");

    assert_eq!(result.dtype(), DType::F64);
    let vals: Vec<f64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(vals, vec![1.0, 0.0, 1.0]);
}

#[test]
fn convert_element_type_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 5] }, vec![]).unwrap());
    let result = convert(input, "i32").expect("2D empty tensor conversion should succeed");

    assert_eq!(result.dtype(), DType::I32);
    assert_eq!(shape(&result).dims, vec![0, 5]);
}
