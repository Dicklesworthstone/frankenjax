//! Oracle tests for Ceil primitive.
//!
//! ceil(x) = smallest integer >= x
//!
//! Tests:
//! - Integers: ceil(n) = n
//! - Positive fractional: ceil(1.1) = 2
//! - Negative fractional: ceil(-1.1) = -1
//! - Zero: ceil(0) = 0, ceil(-0) = -0
//! - Infinity: ceil(±inf) = ±inf
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

// ======================== Integers ========================

#[test]
fn oracle_ceil_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "ceil(0) = +0");
}

#[test]
fn oracle_ceil_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "ceil(-0.0) = -0");
}

#[test]
fn oracle_ceil_f32_signed_zero_and_nan_bits() {
    let input = make_f32_bits_tensor(
        &[9],
        vec![
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            (-0.5_f32).to_bits(),
            1.1_f32.to_bits(),
            (-1.1_f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            f32::NAN.to_bits(),
            0xffc0_0000,
        ],
    );
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let bits = extract_f32_bits_vec(&result);

    assert_eq!(bits[0], 0.0_f32.to_bits(), "ceil(+0.0_f32) = +0");
    assert_eq!(
        bits[1],
        (-0.0_f32).to_bits(),
        "ceil(-0.0_f32) = -0"
    );
    assert_eq!(
        bits[2],
        (-0.0_f32).to_bits(),
        "ceil(-0.5_f32) = -0"
    );
    assert_eq!(bits[3], 2.0_f32.to_bits(), "ceil(1.1_f32) = 2");
    assert_eq!(bits[4], (-1.0_f32).to_bits(), "ceil(-1.1_f32) = -1");
    assert_eq!(bits[5], f32::INFINITY.to_bits(), "ceil(+inf_f32)");
    assert_eq!(
        bits[6],
        f32::NEG_INFINITY.to_bits(),
        "ceil(-inf_f32)"
    );
    assert!(f32::from_bits(bits[7]).is_nan(), "ceil(+nan_f32) = NaN");
    assert!(f32::from_bits(bits[8]).is_nan(), "ceil(-nan_f32) = NaN");
}

#[test]
fn oracle_ceil_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(1) = 1");
}

#[test]
fn oracle_ceil_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1) = -1");
}

#[test]
fn oracle_ceil_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "ceil(1000) = 1000");
}

// ======================== Positive Fractional ========================

#[test]
fn oracle_ceil_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.1) = 2");
}

#[test]
fn oracle_ceil_one_point_five() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.5) = 2");
}

#[test]
fn oracle_ceil_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "ceil(1.9) = 2");
}

#[test]
fn oracle_ceil_point_one() {
    let input = make_f64_tensor(&[], vec![0.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.1) = 1");
}

#[test]
fn oracle_ceil_point_five() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.5) = 1");
}

#[test]
fn oracle_ceil_point_nine() {
    let input = make_f64_tensor(&[], vec![0.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(0.9) = 1");
}

// ======================== Negative Fractional ========================

#[test]
fn oracle_ceil_neg_point_one() {
    // ceil(-0.1) = 0 (round toward positive infinity)
    let input = make_f64_tensor(&[], vec![-0.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.1) = 0");
}

#[test]
fn oracle_ceil_neg_point_five() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.5) = 0");
}

#[test]
fn oracle_ceil_neg_point_nine() {
    let input = make_f64_tensor(&[], vec![-0.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-0.9) = 0");
}

#[test]
fn oracle_ceil_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.1) = -1");
}

#[test]
fn oracle_ceil_neg_one_point_five() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.5) = -1");
}

#[test]
fn oracle_ceil_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "ceil(-1.9) = -1");
}

// ======================== Infinity ========================

#[test]
fn oracle_ceil_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "ceil(+inf) = +inf");
}

#[test]
fn oracle_ceil_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "ceil(-inf) = -inf");
}

// ======================== NaN ========================

#[test]
fn oracle_ceil_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "ceil(NaN) = NaN");
}

// ======================== Very Small Values ========================

#[test]
fn oracle_ceil_very_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "ceil(1e-100) = 1");
}

#[test]
fn oracle_ceil_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "ceil(-1e-100) = 0");
}

// ======================== ceil vs floor relationship ========================

#[test]
fn oracle_ceil_floor_relationship() {
    // ceil(x) = -floor(-x) for all x
    for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let ceil_result = eval_primitive(Primitive::Ceil, &[input_pos], &no_params()).unwrap();
        let floor_result = eval_primitive(Primitive::Floor, &[input_neg], &no_params()).unwrap();

        let ceil_val = extract_f64_scalar(&ceil_result);
        let floor_neg_val = -extract_f64_scalar(&floor_result);

        assert_eq!(ceil_val, floor_neg_val, "ceil({}) = -floor(-{})", x, x);
    }
}

// ======================== Bounds: x <= ceil(x) < x + 1 ========================

#[test]
fn oracle_ceil_bounds() {
    for x in [-2.7, -1.3, -0.1, 0.1, 1.3, 2.7] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);

        assert!(val >= x, "ceil({}) >= {}", x, x);
        assert!(val < x + 1.0, "ceil({}) < {} + 1", x, x);
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_ceil_1d() {
    let input = make_f64_tensor(&[5], vec![-1.5, -0.5, 0.0, 0.5, 1.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], -1.0, "ceil(-1.5)");
    assert_eq!(vals[1], 0.0, "ceil(-0.5)");
    assert_eq!(
        vals[1].to_bits(),
        (-0.0_f64).to_bits(),
        "ceil(-0.5) should produce exact -0.0 bits"
    );
    assert_eq!(vals[2], 0.0, "ceil(0)");
    assert_eq!(vals[3], 1.0, "ceil(0.5)");
    assert_eq!(vals[4], 2.0, "ceil(1.5)");
}

#[test]
fn oracle_ceil_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "ceil(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "ceil(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "ceil(-inf)");
    assert!(vals[3].is_nan(), "ceil(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_ceil_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 1.5, 1.9, -1.1, -1.5, -1.9]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 2.0, "ceil(1.1)");
    assert_eq!(vals[1], 2.0, "ceil(1.5)");
    assert_eq!(vals[2], 2.0, "ceil(1.9)");
    assert_eq!(vals[3], -1.0, "ceil(-1.1)");
    assert_eq!(vals[4], -1.0, "ceil(-1.5)");
    assert_eq!(vals[5], -1.0, "ceil(-1.9)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_ceil_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.1, 0.9, 1.5, 2.5, -0.1, -0.9, -1.5, -2.5]);
    let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 1.0, "ceil(0.1)");
    assert_eq!(vals[1], 1.0, "ceil(0.9)");
    assert_eq!(vals[2], 2.0, "ceil(1.5)");
    assert_eq!(vals[3], 3.0, "ceil(2.5)");
    assert_eq!(vals[4], 0.0, "ceil(-0.1)");
    assert_eq!(
        vals[4].to_bits(),
        (-0.0_f64).to_bits(),
        "ceil(-0.1) should produce exact -0.0 bits"
    );
    assert_eq!(vals[5], 0.0, "ceil(-0.9)");
    assert_eq!(
        vals[5].to_bits(),
        (-0.0_f64).to_bits(),
        "ceil(-0.9) should produce exact -0.0 bits"
    );
    assert_eq!(vals[6], -1.0, "ceil(-1.5)");
    assert_eq!(vals[7], -2.0, "ceil(-2.5)");
}

// ======================== Idempotency: ceil(ceil(x)) = ceil(x) ========================

#[test]
fn oracle_ceil_idempotent() {
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Ceil, &[input1], &no_params()).unwrap();
        let ceiled = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![ceiled]);
        let result2 = eval_primitive(Primitive::Ceil, &[input2], &no_params()).unwrap();
        let double_ceiled = extract_f64_scalar(&result2);

        assert_eq!(ceiled, double_ceiled, "ceil(ceil({})) = ceil({})", x, x);
    }
}

// ======================== METAMORPHIC: ceil(x) >= x ========================

#[test]
fn metamorphic_ceil_geq_x() {
    // ceil(x) >= x for all finite x
    for x in [-2.7, -1.5, -0.1, 0.0, 0.1, 1.5, 2.7, 100.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let ceiled = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let ceil_val = extract_f64_scalar(&ceiled);

        assert!(
            ceil_val >= x,
            "ceil({}) = {} should be >= {}",
            x,
            ceil_val,
            x
        );
    }
}

// ======================== METAMORPHIC: ceil(Neg(x)) = Neg(floor(x)) ========================

#[test]
fn metamorphic_ceil_neg_floor() {
    // ceil(-x) = -floor(x) for all finite x
    for x in [-2.7, -1.5, -0.1, 0.0, 0.1, 1.5, 2.7] {
        let input = make_f64_tensor(&[], vec![x]);

        // ceil(Neg(x))
        let neg_x =
            eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();
        let ceil_neg = eval_primitive(Primitive::Ceil, &[neg_x], &no_params()).unwrap();

        // Neg(floor(x))
        let floor_x = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        let neg_floor = eval_primitive(Primitive::Neg, &[floor_x], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&ceil_neg),
            extract_f64_scalar(&neg_floor),
            "ceil(Neg({})) = Neg(floor({}))",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: ceil idempotent via primitive ========================

#[test]
fn metamorphic_ceil_idempotent() {
    // ceil(ceil(x)) = ceil(x) using Ceil primitive twice
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let ceil1 = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let ceil2 =
            eval_primitive(Primitive::Ceil, std::slice::from_ref(&ceil1), &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&ceil1),
            extract_f64_scalar(&ceil2),
            "ceil(ceil({})) = ceil({})",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: tensor ceil idempotent ========================

#[test]
fn metamorphic_ceil_tensor_idempotent() {
    // For tensor: ceil(ceil(x)) = ceil(x)
    let data = vec![-2.7, -1.5, 0.0, 1.5, 2.7];
    let input = make_f64_tensor(&[5], data);

    let ceil1 = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
    let ceil2 =
        eval_primitive(Primitive::Ceil, std::slice::from_ref(&ceil1), &no_params()).unwrap();

    assert_eq!(extract_shape(&ceil1), vec![5]);
    let vec1 = extract_f64_vec(&ceil1);
    let vec2 = extract_f64_vec(&ceil2);
    for (i, (&c1, &c2)) in vec1.iter().zip(vec2.iter()).enumerate() {
        assert_eq!(c1, c2, "element {}: ceil idempotent", i);
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_ceil_preserves_all_float_dtypes() {
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

    let values = [-1.5_f64, 0.0, 1.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "ceil {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
