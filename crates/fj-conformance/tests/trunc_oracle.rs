//! Oracle tests for Trunc primitive.
//!
//! trunc(x) = round towards zero (truncate fractional part)
//!
//! Tests:
//! - Integers: trunc(n) = n
//! - Positive fractional: trunc(1.9) = 1
//! - Negative fractional: trunc(-1.9) = -1 (different from floor!)
//! - Zero: trunc(0) = 0, trunc(-0) = -0
//! - Infinity: trunc(±inf) = ±inf
//! - NaN propagation
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

fn make_f32_bits_tensor(shape: &[u32], bits: Vec<u32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: shape.to_vec(),
            },
            bits.into_iter().map(Literal::F32Bits).collect(),
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

fn extract_f32_bits_vec(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::F32Bits(bits) => *bits,
                other => panic!("expected F32Bits, got {other:?}"),
            })
            .collect(),
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

fn assert_same_f64_bits(actual: f64, expected: f64, msg: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{msg}: expected bits {:#018x}, got {:#018x}",
        expected.to_bits(),
        actual.to_bits()
    );
}

// ======================== Integers ========================

#[test]
fn oracle_trunc_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "trunc(0) = +0");
}

#[test]
fn oracle_trunc_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "trunc(-0.0) = -0");
}

#[test]
fn oracle_trunc_f32_signed_zero_and_nan_bits() {
    let input = make_f32_bits_tensor(
        &[9],
        vec![
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            (-0.5_f32).to_bits(),
            1.9_f32.to_bits(),
            (-1.9_f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            f32::NAN.to_bits(),
            0xffc0_0000,
        ],
    );
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let bits = extract_f32_bits_vec(&result);

    assert_eq!(bits[0], 0.0_f32.to_bits(), "trunc(+0.0_f32) = +0");
    assert_eq!(
        bits[1],
        (-0.0_f32).to_bits(),
        "trunc(-0.0_f32) = -0"
    );
    assert_eq!(
        bits[2],
        (-0.0_f32).to_bits(),
        "trunc(-0.5_f32) = -0"
    );
    assert_eq!(bits[3], 1.0_f32.to_bits(), "trunc(1.9_f32) = 1");
    assert_eq!(bits[4], (-1.0_f32).to_bits(), "trunc(-1.9_f32) = -1");
    assert_eq!(bits[5], f32::INFINITY.to_bits(), "trunc(+inf_f32)");
    assert_eq!(
        bits[6],
        f32::NEG_INFINITY.to_bits(),
        "trunc(-inf_f32)"
    );
    assert!(f32::from_bits(bits[7]).is_nan(), "trunc(+nan_f32) = NaN");
    assert!(f32::from_bits(bits[8]).is_nan(), "trunc(-nan_f32) = NaN");
}

#[test]
fn oracle_trunc_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1) = 1");
}

#[test]
fn oracle_trunc_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1) = -1");
}

#[test]
fn oracle_trunc_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "trunc(1000) = 1000");
}

// ======================== Positive Fractional ========================

#[test]
fn oracle_trunc_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.1) = 1");
}

#[test]
fn oracle_trunc_one_point_five() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.5) = 1");
}

#[test]
fn oracle_trunc_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.9) = 1");
}

// ======================== Negative Fractional ========================

#[test]
fn oracle_trunc_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.1) = -1");
}

#[test]
fn oracle_trunc_neg_one_point_five() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.5) = -1");
}

#[test]
fn oracle_trunc_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.9) = -1");
}

// ======================== Special Values ========================

#[test]
fn oracle_trunc_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "trunc(+inf) = +inf"
    );
}

#[test]
fn oracle_trunc_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "trunc(-inf) = -inf"
    );
}

#[test]
fn oracle_trunc_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "trunc(NaN) = NaN");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_trunc_vector() {
    let input = make_f64_tensor(&[4], vec![1.1, -1.1, 2.9, -2.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, -1.0, 2.0, -2.0]);
}

#[test]
fn oracle_trunc_matrix() {
    let input = make_f64_tensor(&[2, 3], vec![0.1, 0.5, 0.9, -0.1, -0.5, -0.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_same_f64_bits(vals[3], -0.0, "trunc(-0.1)");
    assert_same_f64_bits(vals[4], -0.0, "trunc(-0.5)");
    assert_same_f64_bits(vals[5], -0.0, "trunc(-0.9)");
}

#[test]
fn oracle_trunc_3d_tensor() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 4.5, -4.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    );
}

// ======================== Comparison with Floor ========================

#[test]
fn oracle_trunc_vs_floor_negative() {
    let input = make_f64_tensor(&[3], vec![-0.1, -0.5, -0.9]);
    let trunc_result = eval_primitive(Primitive::Trunc, &[input.clone()], &no_params()).unwrap();
    let floor_result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let trunc_vals = extract_f64_vec(&trunc_result);
    assert_eq!(trunc_vals, vec![0.0, 0.0, 0.0]);
    assert_same_f64_bits(trunc_vals[0], -0.0, "trunc(-0.1)");
    assert_same_f64_bits(trunc_vals[1], -0.0, "trunc(-0.5)");
    assert_same_f64_bits(trunc_vals[2], -0.0, "trunc(-0.9)");
    assert_eq!(extract_f64_vec(&floor_result), vec![-1.0, -1.0, -1.0]);
}

// ======================== Specific Edge Cases ========================

#[test]
fn oracle_trunc_very_small_positive() {
    let input = make_f64_tensor(&[], vec![0.0000001]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "trunc(tiny) = 0");
}

#[test]
fn oracle_trunc_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-0.0000001]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "trunc(-tiny) = 0");
}

#[test]
fn oracle_trunc_large_magnitude() {
    let input = make_f64_tensor(&[], vec![1e15 + 0.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1e15, "trunc(1e15 + 0.5)");
}

#[test]
fn oracle_trunc_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}

#[test]
fn oracle_trunc_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![1.5, 2.5, 3.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_trunc_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_trunc_4d_shape() {
    let input = make_f64_tensor(
        &[2, 2, 2, 2],
        vec![
            1.1, -1.1, 2.2, -2.2, 3.3, -3.3, 4.4, -4.4, 5.5, -5.5, 6.6, -6.6, 7.7, -7.7, 8.8, -8.8,
        ],
    );
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[15], -8.0);
}

#[test]
fn oracle_trunc_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[], vec![subnormal]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "trunc(subnormal) = 0");
}

#[test]
fn oracle_trunc_subnormal_negative() {
    let subnormal = -(f64::MIN_POSITIVE / 2.0);
    let input = make_f64_tensor(&[], vec![subnormal]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0, "trunc(-subnormal) = 0");
    assert_eq!(
        val.to_bits(),
        (-0.0_f64).to_bits(),
        "trunc(-subnormal) = -0.0"
    );
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_trunc_preserves_all_float_dtypes() {
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

    let values = [1.5_f64, 2.7, -3.9];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "trunc {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
