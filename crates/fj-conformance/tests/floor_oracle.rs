//! Oracle tests for Floor primitive.
//!
//! floor(x) = largest integer <= x
//!
//! Tests:
//! - Integers: floor(n) = n
//! - Positive fractional: floor(1.9) = 1
//! - Negative fractional: floor(-1.1) = -2
//! - Zero: floor(0) = 0, floor(-0) = -0
//! - Infinity: floor(±inf) = ±inf
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

// ======================== Integers ========================

#[test]
fn oracle_floor_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "floor(0) = 0");
}

#[test]
fn oracle_floor_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(val, 0.0, "floor(-0.0) = 0");
}

#[test]
fn oracle_floor_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "floor(1) = 1");
}

#[test]
fn oracle_floor_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "floor(-1) = -1");
}

#[test]
fn oracle_floor_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "floor(1000) = 1000");
}

// ======================== Positive Fractional ========================

#[test]
fn oracle_floor_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "floor(1.1) = 1");
}

#[test]
fn oracle_floor_one_point_five() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "floor(1.5) = 1");
}

#[test]
fn oracle_floor_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "floor(1.9) = 1");
}

#[test]
fn oracle_floor_point_one() {
    let input = make_f64_tensor(&[], vec![0.1]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "floor(0.1) = 0");
}

#[test]
fn oracle_floor_point_five() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "floor(0.5) = 0");
}

#[test]
fn oracle_floor_point_nine() {
    let input = make_f64_tensor(&[], vec![0.9]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "floor(0.9) = 0");
}

// ======================== Negative Fractional ========================

#[test]
fn oracle_floor_neg_point_one() {
    // floor(-0.1) = -1 (round toward negative infinity)
    let input = make_f64_tensor(&[], vec![-0.1]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "floor(-0.1) = -1");
}

#[test]
fn oracle_floor_neg_point_five() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "floor(-0.5) = -1");
}

#[test]
fn oracle_floor_neg_point_nine() {
    let input = make_f64_tensor(&[], vec![-0.9]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "floor(-0.9) = -1");
}

#[test]
fn oracle_floor_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "floor(-1.1) = -2");
}

#[test]
fn oracle_floor_neg_one_point_five() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "floor(-1.5) = -2");
}

#[test]
fn oracle_floor_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -2.0, "floor(-1.9) = -2");
}

// ======================== Infinity ========================

#[test]
fn oracle_floor_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "floor(+inf) = +inf");
}

#[test]
fn oracle_floor_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "floor(-inf) = -inf");
}

// ======================== NaN ========================

#[test]
fn oracle_floor_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "floor(NaN) = NaN");
}

// ======================== Very Small Values ========================

#[test]
fn oracle_floor_very_small_positive() {
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "floor(1e-100) = 0");
}

#[test]
fn oracle_floor_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-1e-100]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "floor(-1e-100) = -1");
}

// ======================== floor vs ceil relationship ========================

#[test]
fn oracle_floor_ceil_relationship() {
    // floor(x) = -ceil(-x) for all x
    for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let floor_result = eval_primitive(Primitive::Floor, &[input_pos], &no_params()).unwrap();
        let ceil_result = eval_primitive(Primitive::Ceil, &[input_neg], &no_params()).unwrap();

        let floor_val = extract_f64_scalar(&floor_result);
        let ceil_neg_val = -extract_f64_scalar(&ceil_result);

        assert_eq!(floor_val, ceil_neg_val, "floor({}) = -ceil(-{})", x, x);
    }
}

// ======================== Bounds: x - 1 < floor(x) <= x ========================

#[test]
fn oracle_floor_bounds() {
    for x in [-2.7, -1.3, -0.1, 0.1, 1.3, 2.7] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);

        assert!(val <= x, "floor({}) <= {}", x, x);
        assert!(val > x - 1.0, "floor({}) > {} - 1", x, x);
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_floor_1d() {
    let input = make_f64_tensor(&[5], vec![-1.5, -0.5, 0.0, 0.5, 1.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], -2.0, "floor(-1.5)");
    assert_eq!(vals[1], -1.0, "floor(-0.5)");
    assert_eq!(vals[2], 0.0, "floor(0)");
    assert_eq!(vals[3], 0.0, "floor(0.5)");
    assert_eq!(vals[4], 1.0, "floor(1.5)");
}

#[test]
fn oracle_floor_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "floor(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "floor(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "floor(-inf)");
    assert!(vals[3].is_nan(), "floor(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_floor_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.1, 1.5, 1.9, -1.1, -1.5, -1.9]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 1.0, "floor(1.1)");
    assert_eq!(vals[1], 1.0, "floor(1.5)");
    assert_eq!(vals[2], 1.0, "floor(1.9)");
    assert_eq!(vals[3], -2.0, "floor(-1.1)");
    assert_eq!(vals[4], -2.0, "floor(-1.5)");
    assert_eq!(vals[5], -2.0, "floor(-1.9)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_floor_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.1, 0.9, 1.5, 2.5, -0.1, -0.9, -1.5, -2.5]);
    let result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "floor(0.1)");
    assert_eq!(vals[1], 0.0, "floor(0.9)");
    assert_eq!(vals[2], 1.0, "floor(1.5)");
    assert_eq!(vals[3], 2.0, "floor(2.5)");
    assert_eq!(vals[4], -1.0, "floor(-0.1)");
    assert_eq!(vals[5], -1.0, "floor(-0.9)");
    assert_eq!(vals[6], -2.0, "floor(-1.5)");
    assert_eq!(vals[7], -3.0, "floor(-2.5)");
}

// ======================== Idempotency: floor(floor(x)) = floor(x) ========================

#[test]
fn oracle_floor_idempotent() {
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let result1 = eval_primitive(Primitive::Floor, &[input1], &no_params()).unwrap();
        let floored = extract_f64_scalar(&result1);

        let input2 = make_f64_tensor(&[], vec![floored]);
        let result2 = eval_primitive(Primitive::Floor, &[input2], &no_params()).unwrap();
        let double_floored = extract_f64_scalar(&result2);

        assert_eq!(
            floored, double_floored,
            "floor(floor({})) = floor({})",
            x, x
        );
    }
}

// ======================== METAMORPHIC: floor(x) <= x ========================

#[test]
fn metamorphic_floor_leq_x() {
    // floor(x) <= x for all finite x
    for x in [-2.7, -1.5, -0.1, 0.0, 0.1, 1.5, 2.7, 100.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let floored = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        let floor_val = extract_f64_scalar(&floored);

        assert!(
            floor_val <= x,
            "floor({}) = {} should be <= {}",
            x,
            floor_val,
            x
        );
    }
}

// ======================== METAMORPHIC: floor(Neg(x)) = Neg(ceil(x)) ========================

#[test]
fn metamorphic_floor_neg_ceil() {
    // floor(-x) = -ceil(x) for all finite x
    for x in [-2.7, -1.5, -0.1, 0.0, 0.1, 1.5, 2.7] {
        let input = make_f64_tensor(&[], vec![x]);

        // floor(Neg(x))
        let neg_x = eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &no_params()).unwrap();
        let floor_neg = eval_primitive(Primitive::Floor, &[neg_x], &no_params()).unwrap();

        // Neg(ceil(x))
        let ceil_x = eval_primitive(Primitive::Ceil, &[input], &no_params()).unwrap();
        let neg_ceil = eval_primitive(Primitive::Neg, &[ceil_x], &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&floor_neg),
            extract_f64_scalar(&neg_ceil),
            "floor(Neg({})) = Neg(ceil({}))",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: floor idempotent via primitive ========================

#[test]
fn metamorphic_floor_idempotent() {
    // floor(floor(x)) = floor(x) using Floor primitive twice
    for x in [-2.7, -1.5, 0.0, 1.5, 2.7, 100.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let floor1 = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        let floor2 = eval_primitive(Primitive::Floor, std::slice::from_ref(&floor1), &no_params()).unwrap();

        assert_eq!(
            extract_f64_scalar(&floor1),
            extract_f64_scalar(&floor2),
            "floor(floor({})) = floor({})",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: tensor floor idempotent ========================

#[test]
fn metamorphic_floor_tensor_idempotent() {
    // For tensor: floor(floor(x)) = floor(x)
    let data = vec![-2.7, -1.5, 0.0, 1.5, 2.7];
    let input = make_f64_tensor(&[5], data);

    let floor1 = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    let floor2 = eval_primitive(Primitive::Floor, std::slice::from_ref(&floor1), &no_params()).unwrap();

    assert_eq!(extract_shape(&floor1), vec![5]);
    let vec1 = extract_f64_vec(&floor1);
    let vec2 = extract_f64_vec(&floor2);
    for (i, (&f1, &f2)) in vec1.iter().zip(vec2.iter()).enumerate() {
        assert_eq!(f1, f2, "element {}: floor idempotent", i);
    }
}
