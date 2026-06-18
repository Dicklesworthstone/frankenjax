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
fn convert_element_type_bool_tensor_to_f64_uses_zero_and_one() {
    // bool -> float: true -> 1.0, false -> 0.0 (JAX/XLA) — the float sibling of the
    // bool -> i64 conversion above, which the oracle did not cover.
    let input = bool_tensor(&[4], &[true, false, true, false]);
    let result = convert(input, "f64").expect("bool to f64 conversion should succeed");
    assert_eq!(result.dtype(), DType::F64);
    let vals: Vec<f64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(vals, vec![1.0, 0.0, 1.0, 0.0]);
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
fn convert_element_type_i64_to_f16_rounds_at_2_pow_11_boundary() {
    // f16 has a 10-bit mantissa, so integers above 2^11 = 2048 are not all
    // representable (the step is 2 there). i64 -> f16 rounds to nearest-even (IEEE):
    // 2048 is exact, 2049 is halfway between 2048 and 2050 and rounds to the even
    // 2048. Verified by widening the f16 result back to f64 (exact for these).
    let input = i64_tensor(&[2], &[2048, 2049]);
    let half = convert(input, "f16").expect("i64 to f16 conversion should succeed");
    assert_eq!(half.dtype(), DType::F16);
    let back = convert(half, "f64").expect("f16 to f64 conversion should succeed");
    let vals: Vec<f64> = back
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(
        vals,
        vec![2048.0, 2048.0],
        "i64 -> f16 must round to nearest-even at the 2^11 mantissa boundary"
    );
}

#[test]
fn convert_element_type_bf16_to_f32_widens_exactly() {
    // bf16 is literally the top 16 bits of f32 (same exponent range, 7-bit mantissa),
    // so widening bf16 -> f32 is exact (mantissa zero-extends). Use values exactly
    // representable in bf16 (1.5, -0.25, 384.0) — they appear bit-identically in f32.
    let input = Value::Tensor(
        TensorValue::new(
            DType::BF16,
            Shape { dims: vec![3] },
            vec![
                Literal::from_bf16_f64(1.5),
                Literal::from_bf16_f64(-0.25),
                Literal::from_bf16_f64(384.0),
            ],
        )
        .unwrap(),
    );
    let result = convert(input, "f32").expect("bf16 to f32 conversion should succeed");
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(
        f32_values(&result),
        vec![Some(1.5), Some(-0.25), Some(384.0)],
        "bf16 -> f32 widening must be exact for representable values"
    );
}

#[test]
fn convert_element_type_bf16_to_i32_truncates_toward_zero() {
    // bf16 is a distinct half-float type from f16 (8-bit mantissa, f32 exponent
    // range) with its own storage/convert path. bf16 -> int truncates toward zero
    // like f16/f32/f64 -> int: bf16(2.7) -> 2, bf16(-2.7) -> -2, bf16(0.9) -> 0.
    let input = Value::Tensor(
        TensorValue::new(
            DType::BF16,
            Shape { dims: vec![3] },
            vec![
                Literal::from_bf16_f64(2.7),
                Literal::from_bf16_f64(-2.7),
                Literal::from_bf16_f64(0.9),
            ],
        )
        .unwrap(),
    );
    let result = convert(input, "i32").expect("bf16 to i32 conversion should succeed");
    assert_eq!(result.dtype(), DType::I32);
    let values: Vec<i64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().expect("expected i32 stored as i64"))
        .collect();
    assert_eq!(values, vec![2, -2, 0]);
}

#[test]
fn convert_element_type_f16_to_f32_widens_exactly() {
    // f16 is a strict subset of f32, so widening f16 -> f32 is exact (no rounding).
    // Use values exactly representable in f16 (1.5, -0.5, 256.0) — they must appear
    // bit-identically in f32.
    let input = Value::Tensor(
        TensorValue::new(
            DType::F16,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f16_f64(1.5),
                Literal::from_f16_f64(-0.5),
                Literal::from_f16_f64(256.0),
            ],
        )
        .unwrap(),
    );
    let result = convert(input, "f32").expect("f16 to f32 conversion should succeed");
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(
        f32_values(&result),
        vec![Some(1.5), Some(-0.5), Some(256.0)],
        "f16 -> f32 widening must be exact for representable values"
    );
}

#[test]
fn convert_element_type_f16_to_i32_truncates_toward_zero() {
    // Half-float source -> int truncates toward zero like f64/f32 -> int (negatives
    // toward zero too). The oracle had no f16 convert coverage. f16(2.7) rounds to a
    // value in (2,3) and truncates to 2; f16(-2.7) -> -2; f16(0.9) -> 0.
    let input = Value::Tensor(
        TensorValue::new(
            DType::F16,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f16_f64(2.7),
                Literal::from_f16_f64(-2.7),
                Literal::from_f16_f64(0.9),
            ],
        )
        .unwrap(),
    );
    let result = convert(input, "i32").expect("f16 to i32 conversion should succeed");
    assert_eq!(result.dtype(), DType::I32);
    let values: Vec<i64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().expect("expected i32 stored as i64"))
        .collect();
    assert_eq!(values, vec![2, -2, 0]);
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
fn convert_element_type_f32_to_i32_truncates_toward_zero() {
    // f32 is JAX's DEFAULT float dtype; f32 -> i32 truncates toward zero exactly like
    // f64 -> i32 (negative truncates toward zero too: -2.9 -> -2, not -3). Guards the
    // f32 source boundary the f64-only conversion tests miss.
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape { dims: vec![4] },
            [1.9_f32, -2.9, 0.0, 100.7]
                .into_iter()
                .map(|x| Literal::F32Bits(x.to_bits()))
                .collect(),
        )
        .unwrap(),
    );
    let result = convert(input, "i32").expect("f32 to i32 conversion should succeed");
    assert_eq!(result.dtype(), DType::I32);
    let values: Vec<i64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().expect("expected i32 stored as i64"))
        .collect();
    assert_eq!(values, vec![1, -2, 0, 100]);
}

#[test]
fn convert_element_type_f64_to_f16_overflows_to_signed_infinity() {
    // f16's max finite is 65504; a larger magnitude overflows to signed infinity
    // (IEEE round-to-nearest), matching JAX/XLA — not a saturation to f16::MAX.
    // Checked by widening the f16 result back to f64 (exact for 1.0 and ±inf).
    let input = f64_tensor(&[3], &[1.0, 70000.0, -70000.0]);
    let half = convert(input, "f16").expect("f64 to f16 conversion should succeed");
    assert_eq!(half.dtype(), DType::F16);
    let back = convert(half, "f64").expect("f16 to f64 conversion should succeed");
    let vals: Vec<f64> = back
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(vals, vec![1.0, f64::INFINITY, f64::NEG_INFINITY]);
}

#[test]
fn convert_element_type_f64_to_f32_overflows_to_signed_infinity() {
    // A finite f64 magnitude beyond f32::MAX (~3.4e38) narrows to signed infinity
    // (IEEE round-to-nearest overflow), matching JAX/XLA convert_element_type —
    // it must NOT saturate to f32::MAX or produce NaN. In-range values are exact.
    let input = f64_tensor(&[3], &[1e300, -1e300, 1.0]);
    let result = convert(input, "f32").expect("f64 to f32 conversion should succeed");
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(
        f32_values(&result),
        vec![Some(f32::INFINITY), Some(f32::NEG_INFINITY), Some(1.0)]
    );
}

#[test]
fn convert_element_type_i64_to_f32_rounds_to_nearest_even() {
    // f32 has a 24-bit mantissa, so not every integer above 2^24 is representable
    // (the step is 2 there). i64 -> f32 rounds to nearest-even (IEEE), matching
    // JAX/XLA: 2^24 = 16777216 is exact, and 2^24+1 = 16777217 is exactly halfway
    // between 16777216 and 16777218, so it rounds to the even 16777216.
    let input = i64_tensor(&[2], &[16_777_216, 16_777_217]);
    let result = convert(input, "f32").expect("i64 to f32 conversion should succeed");
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(
        f32_values(&result),
        vec![Some(16_777_216.0), Some(16_777_216.0)],
        "i64 -> f32 must round to nearest-even, not truncate or round up"
    );
}

#[test]
fn convert_element_type_i64_to_f64_rounds_at_2_pow_53_boundary() {
    // f64 has a 53-bit mantissa; integers above 2^53 are not all representable
    // (the step is 2 there). i64 -> f64 rounds to nearest-even (IEEE), matching
    // JAX/XLA: 2^53 = 9007199254740992 is exact, and 2^53+1 = 9007199254740993 is
    // exactly halfway between 2^53 and 2^53+2, so it rounds to the even 2^53.
    let input = i64_tensor(&[2], &[9_007_199_254_740_992, 9_007_199_254_740_993]);
    let result = convert(input, "f64").expect("i64 to f64 conversion should succeed");
    assert_eq!(result.dtype(), DType::F64);
    let vals: Vec<f64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(
        vals,
        vec![9_007_199_254_740_992.0, 9_007_199_254_740_992.0],
        "i64 -> f64 must round to nearest-even at the 2^53 mantissa boundary"
    );
}

#[test]
fn convert_element_type_u64_to_f64_rounds_at_2_pow_53_boundary() {
    // u64 source path (distinct from i64) — and u64 commonly exceeds 2^53. Same
    // nearest-even rounding at the f64 mantissa boundary: 2^53 is exact, 2^53+1
    // rounds to the even 2^53. 2^63 is an exact power of two.
    let input = Value::Tensor(
        TensorValue::new(
            DType::U64,
            Shape { dims: vec![3] },
            vec![
                Literal::U64(9_007_199_254_740_992),
                Literal::U64(9_007_199_254_740_993),
                Literal::U64(9_223_372_036_854_775_808),
            ],
        )
        .unwrap(),
    );
    let result = convert(input, "f64").expect("u64 to f64 conversion should succeed");
    assert_eq!(result.dtype(), DType::F64);
    let vals: Vec<f64> = result
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().expect("expected f64"))
        .collect();
    assert_eq!(
        vals,
        vec![
            9_007_199_254_740_992.0,
            9_007_199_254_740_992.0,
            9_223_372_036_854_775_808.0,
        ],
        "u64 -> f64 must round to nearest-even at the 2^53 boundary; 2^63 is exact"
    );
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

#[test]
fn convert_lossless_round_trips_preserve_values() {
    let extract_f64 = |v: &Value| -> Vec<f64> {
        v.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    };
    let extract_i64 = |v: &Value| -> Vec<i64> {
        v.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect()
    };

    // f64 -> f32 -> f64 is the identity for f32-exact values.
    let f32_exact = [0.0, 1.0, -1.0, 0.5, -3.25, 256.0, 1.0e6, -0.125];
    let rt = convert(convert(f64_tensor(&[8], &f32_exact), "f32").unwrap(), "f64").unwrap();
    assert_eq!(
        extract_f64(&rt),
        f32_exact.to_vec(),
        "f64->f32->f64 must be identity for f32-exact values"
    );

    // i64 -> f64 -> i64 is the identity within +/-2^53 (exactly representable in f64).
    let small = [0_i64, 1, -1, 42, -1000, 1 << 40, -(1 << 40)];
    let rt2 = convert(convert(i64_tensor(&[7], &small), "f64").unwrap(), "i64").unwrap();
    assert_eq!(
        extract_i64(&rt2),
        small.to_vec(),
        "i64->f64->i64 must be identity within 2^53"
    );

    // i64 -> i32 -> i64 is the identity for values in i32 range.
    let i32range = [0_i64, 1, -1, i32::MAX as i64, i32::MIN as i64, 123456];
    let rt3 = convert(convert(i64_tensor(&[6], &i32range), "i32").unwrap(), "i64").unwrap();
    assert_eq!(
        extract_i64(&rt3),
        i32range.to_vec(),
        "i64->i32->i64 must be identity in i32 range"
    );

    // u32 -> i64 -> u32 is the identity (u32 fits losslessly in i64).
    let us = [0_u32, 1, 42, u32::MAX, 1 << 31];
    let rt4 = convert(convert(u32_tensor(&[5], &us), "i64").unwrap(), "u32").unwrap();
    let u_out: Vec<u32> = rt4
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Literal::U32(x) => *x,
            other => panic!("expected U32, got {other:?}"),
        })
        .collect();
    assert_eq!(u_out, us.to_vec(), "u32->i64->u32 must be identity");
}
