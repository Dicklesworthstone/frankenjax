//! Oracle tests for Clamp primitive.
//!
//! Tests against expected behavior matching JAX/lax.clamp:
//! - clamp(min, x, max) returns x clamped to [min, max]
//! - If x < min, returns min; if x > max, returns max; else returns x

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Scalar(lit) => lit.as_f64().unwrap(),
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar tensor");
            t.elements[0].as_f64().unwrap()
        }
    }
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_clamp_scalar_in_range() {
    // JAX: lax.clamp(0, 5, 10) => 5
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(5);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 5),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_below_min() {
    // JAX: lax.clamp(0, -5, 10) => 0
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(-5);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 0),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_above_max() {
    // JAX: lax.clamp(0, 15, 10) => 10
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(15);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 10),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_at_min() {
    // JAX: lax.clamp(0, 0, 10) => 0
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(0);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 0),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_at_max() {
    // JAX: lax.clamp(0, 10, 10) => 10
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(10);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 10),
        _ => unreachable!("expected scalar"),
    }
}

// ======================== 1D Tensor Tests ========================

#[test]
fn oracle_clamp_1d_i64() {
    // JAX: lax.clamp(jnp.array([0,0,0,0,0]), jnp.array([-2, 3, 7, 15, -5]), jnp.array([10,10,10,10,10]))
    // => [0, 3, 7, 10, 0]
    let lo = make_i64_tensor(&[5], vec![0, 0, 0, 0, 0]);
    let x = make_i64_tensor(&[5], vec![-2, 3, 7, 15, -5]);
    let hi = make_i64_tensor(&[5], vec![10, 10, 10, 10, 10]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 3, 7, 10, 0]);
}

#[test]
fn oracle_clamp_1d_f64() {
    // JAX: lax.clamp(0.0, jnp.array([-1.5, 0.5, 1.5]), 1.0)
    // => [0.0, 0.5, 1.0]
    let lo = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let x = make_f64_tensor(&[3], vec![-1.5, 0.5, 1.5]);
    let hi = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[1] - 0.5).abs() < 1e-10);
    assert!((vals[2] - 1.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_clamp_negative_range() {
    // Clamp to negative range
    let lo = Value::scalar_i64(-10);
    let x = Value::scalar_i64(5);
    let hi = Value::scalar_i64(-5);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), -5),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_single_point() {
    // min == max case
    let lo = Value::scalar_i64(5);
    let x = Value::scalar_i64(10);
    let hi = Value::scalar_i64(5);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 5),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_f64_scalar() {
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(0.7));
    let hi = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!((lit.as_f64().unwrap() - 0.7).abs() < 1e-10),
        _ => unreachable!("expected scalar"),
    }
}

// ======================== Special Values ========================

#[test]
fn oracle_clamp_positive_infinity_clamped() {
    // clamp(0, +Inf, 10) => 10 (clamped to max)
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let hi = Value::Scalar(Literal::from_f64(10.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!((lit.as_f64().unwrap() - 10.0).abs() < 1e-10),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_negative_infinity_clamped() {
    // clamp(0, -Inf, 10) => 0 (clamped to min)
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let hi = Value::Scalar(Literal::from_f64(10.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!((lit.as_f64().unwrap() - 0.0).abs() < 1e-10),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_nan_propagates() {
    // clamp(0, NaN, 10) => NaN (NaN propagates)
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(f64::NAN));
    let hi = Value::Scalar(Literal::from_f64(10.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!(lit.as_f64().unwrap().is_nan(), "clamp(0, NaN, 10) = NaN"),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_preserves_negative_zero_operand_at_lower_bound() {
    // JAX: lax.clamp(+0.0, -0.0, 1.0) returns the in-range operand, not the bound.
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(-0.0));
    let hi = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(
        val.to_bits(),
        (-0.0_f64).to_bits(),
        "clamp should preserve the operand's -0.0 sign at an equal lower bound"
    );
}

#[test]
fn oracle_clamp_preserves_positive_zero_operand_at_upper_bound() {
    // JAX: lax.clamp(-1.0, +0.0, -0.0) returns the in-range operand, not the bound.
    let lo = Value::Scalar(Literal::from_f64(-1.0));
    let x = Value::Scalar(Literal::from_f64(0.0));
    let hi = Value::Scalar(Literal::from_f64(-0.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(
        val.to_bits(),
        0.0_f64.to_bits(),
        "clamp should preserve the operand's +0.0 sign at an equal upper bound"
    );
}

#[test]
fn oracle_clamp_infinite_bounds() {
    // clamp(-Inf, 5, +Inf) => 5 (finite value in infinite range)
    let lo = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let x = Value::Scalar(Literal::from_f64(5.0));
    let hi = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!((lit.as_f64().unwrap() - 5.0).abs() < 1e-10),
        _ => unreachable!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_tensor_special_values() {
    // Test special values in tensor form
    let lo = make_f64_tensor(&[4], vec![0.0, 0.0, 0.0, f64::NEG_INFINITY]);
    let x = make_f64_tensor(&[4], vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 5.0]);
    let hi = make_f64_tensor(&[4], vec![10.0, 10.0, 10.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert!((vals[0] - 10.0).abs() < 1e-10, "clamp(0, +Inf, 10) = 10");
    assert!((vals[1] - 0.0).abs() < 1e-10, "clamp(0, -Inf, 10) = 0");
    assert!(vals[2].is_nan(), "clamp(0, NaN, 10) = NaN");
    assert!((vals[3] - 5.0).abs() < 1e-10, "clamp(-Inf, 5, +Inf) = 5");
}

// ======================== METAMORPHIC: idempotence ========================

#[test]
fn metamorphic_clamp_idempotent() {
    // clamp(lo, clamp(lo, x, hi), hi) = clamp(lo, x, hi)
    // Once clamped, clamping again with the same bounds produces the same result
    for x in [-10.0, -1.0, 0.5, 5.0, 15.0] {
        let lo = make_f64_tensor(&[], vec![0.0]);
        let hi = make_f64_tensor(&[], vec![10.0]);
        let input = make_f64_tensor(&[], vec![x]);

        let clamp1 = eval_primitive(
            Primitive::Clamp,
            &[lo.clone(), input, hi.clone()],
            &no_params(),
        )
        .unwrap();
        let clamp2 = eval_primitive(
            Primitive::Clamp,
            &[lo.clone(), clamp1.clone(), hi.clone()],
            &no_params(),
        )
        .unwrap();

        assert_eq!(
            extract_f64_scalar(&clamp1),
            extract_f64_scalar(&clamp2),
            "clamp(0, clamp(0, {}, 10), 10) = clamp(0, {}, 10)",
            x,
            x
        );
    }
}

// ======================== METAMORPHIC: output always in range ========================

#[test]
fn metamorphic_clamp_output_in_range() {
    // For any x, lo <= clamp(lo, x, hi) <= hi (when lo <= hi)
    let lo = 0.0;
    let hi = 10.0;

    for x in [
        -100.0,
        -1.0,
        0.0,
        5.0,
        10.0,
        100.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let lo_val = make_f64_tensor(&[], vec![lo]);
        let hi_val = make_f64_tensor(&[], vec![hi]);
        let input = make_f64_tensor(&[], vec![x]);

        let result =
            eval_primitive(Primitive::Clamp, &[lo_val, input, hi_val], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);

        if !x.is_nan() {
            assert!(
                val >= lo && val <= hi,
                "clamp(0, {}, 10) = {} should be in [0, 10]",
                x,
                val
            );
        }
    }
}

// ======================== METAMORPHIC: tensor idempotence ========================

#[test]
fn metamorphic_clamp_tensor_idempotent() {
    let lo = make_f64_tensor(&[5], vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let x = make_f64_tensor(&[5], vec![-5.0, 0.0, 5.0, 10.0, 15.0]);
    let hi = make_f64_tensor(&[5], vec![10.0, 10.0, 10.0, 10.0, 10.0]);

    let clamp1 =
        eval_primitive(Primitive::Clamp, &[lo.clone(), x, hi.clone()], &no_params()).unwrap();
    let clamp2 = eval_primitive(
        Primitive::Clamp,
        &[lo.clone(), clamp1.clone(), hi.clone()],
        &no_params(),
    )
    .unwrap();

    let vals1 = extract_f64_vec(&clamp1);
    let vals2 = extract_f64_vec(&clamp2);

    for (v1, v2) in vals1.iter().zip(vals2.iter()) {
        assert_eq!(*v1, *v2, "clamp is idempotent for tensors");
    }
}

// `eval_clamp`'s mixed-dtype tensor arm previously emitted Literal::from_f64
// while declaring the tensor's original dtype (BF16/F16/F32), violating the
// dtype/element invariant. The fix in 1x85 threaded an Option<DType> target
// through clamp_literal. This property sweep guards against per-dtype
// regressions across all four float variants for the same-dtype path.
#[test]
fn property_clamp_preserves_all_float_dtypes() {
    fn make_scalar(dtype: DType, v: f64) -> Value {
        Value::Scalar(match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        })
    }

    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: vec![values.len() as u32],
                },
                values.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let values = [-5.0_f64, 2.0, 7.0, 12.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let x = make_vec(dtype, &values);
        let lo = make_scalar(dtype, 0.0);
        let hi = make_scalar(dtype, 10.0);

        let result = eval_primitive(Primitive::Clamp, &[x, lo, hi], &no_params())
            .unwrap_or_else(|e| unreachable!("clamp {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            unreachable!("clamp {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "clamp {dtype:?}: tensor dtype mismatch");
        t.validate_dtype_consistency().unwrap_or_else(|e| {
            unreachable!("clamp {dtype:?}: validate_dtype_consistency failed: {e}")
        });
    }
}

// ======================== JAX-order broadcast tests ========================

fn extract_shape(v: &Value) -> Vec<u32> {
    v.as_tensor().expect("expected tensor").shape.dims.clone()
}

#[test]
fn oracle_clamp_jax_order_scalar_bounds() {
    // JAX order: clamp(min, x, max) with scalar bounds
    let lo = Value::scalar_f64(0.0);
    let x = make_f64_tensor(&[4], vec![-5.0, 2.0, 7.0, 12.0]);
    let hi = Value::scalar_f64(10.0);

    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 2.0, 7.0, 10.0]);
}

#[test]
fn oracle_clamp_jax_order_2d_scalar_bounds() {
    // JAX order with 2D tensor and scalar bounds
    let lo = Value::scalar_f64(1.0);
    let x = make_f64_tensor(&[2, 3], vec![0.0, 1.5, 5.0, -1.0, 3.0, 10.0]);
    let hi = Value::scalar_f64(4.0);

    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 1.5, 4.0, 1.0, 3.0, 4.0]);
}

#[test]
fn oracle_clamp_tensor_bounds_broadcast_row_vector() {
    // Tensor bounds with row vector broadcast: [2,3] vs [3]
    let lo = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let x = make_f64_tensor(&[2, 3], vec![-1.0, 0.0, 1.0, 5.0, 6.0, 7.0]);
    let hi = make_f64_tensor(&[3], vec![3.0, 4.0, 5.0]);

    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    // Row 0: clamp([-1,0,1], [0,1,2], [3,4,5]) = [0,1,2]
    // Row 1: clamp([5,6,7], [0,1,2], [3,4,5]) = [3,4,5]
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn oracle_clamp_scalar_lo_tensor_x_tensor_hi() {
    let lo = Value::scalar_f64(0.0);
    let x = make_f64_tensor(&[3], vec![-1.0, 5.0, 15.0]);
    let hi = make_f64_tensor(&[3], vec![2.0, 10.0, 8.0]);

    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    // clamp([-1,5,15], 0, [2,10,8]) = [0,5,8]
    assert_eq!(extract_f64_vec(&result), vec![0.0, 5.0, 8.0]);
}

#[test]
fn oracle_clamp_tensor_lo_tensor_x_scalar_hi() {
    let lo = make_f64_tensor(&[3], vec![-2.0, 0.0, 3.0]);
    let x = make_f64_tensor(&[3], vec![-5.0, 5.0, 1.0]);
    let hi = Value::scalar_f64(10.0);

    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    // clamp([-5,5,1], [-2,0,3], 10) = [-2,5,3]
    assert_eq!(extract_f64_vec(&result), vec![-2.0, 5.0, 3.0]);
}

// ======================== Comprehensive Broadcast Tests ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_clamp_all_scalars_broadcast() {
    let lo = scalar_f64(0.0);
    let x = scalar_f64(5.0);
    let hi = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert!((extract_f64_scalar(&result) - 5.0).abs() < 1e-14);
}

#[test]
fn oracle_clamp_scalar_lo_scalar_hi_tensor_x_broadcast() {
    let lo = scalar_f64(0.0);
    let x = make_f64_tensor(&[4], vec![-5.0, 2.0, 7.0, 12.0]);
    let hi = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 2.0, 7.0, 10.0]);
}

#[test]
fn oracle_clamp_singleton_lo_vector_x_singleton_hi_broadcast() {
    let lo = make_f64_tensor(&[1], vec![0.0]);
    let x = make_f64_tensor(&[4], vec![-5.0, 2.0, 7.0, 12.0]);
    let hi = make_f64_tensor(&[1], vec![10.0]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 2.0, 7.0, 10.0]);
}

#[test]
fn oracle_clamp_column_lo_matrix_x_scalar_hi_broadcast() {
    // [2,1] lo, [2,3] x, scalar hi -> [2,3]
    let lo = make_f64_tensor(&[2, 1], vec![0.0, 5.0]);
    let x = make_f64_tensor(&[2, 3], vec![-1.0, 2.0, 8.0, 3.0, 6.0, 12.0]);
    let hi = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    // Row 0: clamp([-1,2,8], 0, 10) = [0,2,8]
    // Row 1: clamp([3,6,12], 5, 10) = [5,6,10]
    assert_eq!(extract_f64_vec(&result), vec![0.0, 2.0, 8.0, 5.0, 6.0, 10.0]);
}

#[test]
fn oracle_clamp_different_ranks_broadcast() {
    // [3] lo, [2,3] x, [] hi -> [2,3]
    let lo = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let x = make_f64_tensor(&[2, 3], vec![-1.0, 0.5, 1.5, 5.0, 6.0, 7.0]);
    let hi = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    // Row 0: clamp([-1,0.5,1.5], [0,1,2], 10) = [0,1,2]
    // Row 1: clamp([5,6,7], [0,1,2], 10) = [5,6,7]
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-14);
    assert!((vals[1] - 1.0).abs() < 1e-14);
    assert!((vals[2] - 2.0).abs() < 1e-14);
    assert!((vals[3] - 5.0).abs() < 1e-14);
    assert!((vals[4] - 6.0).abs() < 1e-14);
    assert!((vals[5] - 7.0).abs() < 1e-14);
}

#[test]
fn oracle_clamp_incompatible_shapes_error() {
    let lo = make_f64_tensor(&[2], vec![0.0, 1.0]);
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let hi = scalar_f64(10.0);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}
