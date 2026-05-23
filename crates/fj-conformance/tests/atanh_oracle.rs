//! Oracle tests for Atanh (inverse hyperbolic tangent) primitive.
//!
//! atanh(x) = 0.5 * ln((1 + x) / (1 - x))
//!
//! Properties:
//! - atanh(0) = 0
//! - Domain: |x| < 1 (real-valued), ±inf at ±1, NaN outside
//! - atanh is odd: atanh(-x) = -atanh(x)
//! - Metamorphic: tanh(atanh(x)) = x for |x| < 1
//! - Metamorphic: atanh(tanh(x)) = x

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff={}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ======================== Zero ========================

#[test]
fn oracle_atanh_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "atanh(0) = +0");
}

#[test]
fn oracle_atanh_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "atanh(-0.0) = -0");
}

// ======================== Basic Values (|x| < 1) ========================

#[test]
fn oracle_atanh_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5_f64.atanh(),
        1e-14,
        "atanh(0.5)",
    );
}

#[test]
fn oracle_atanh_neg_half() {
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-0.5_f64).atanh(),
        1e-14,
        "atanh(-0.5)",
    );
}

// ======================== Odd Function: atanh(-x) = -atanh(x) ========================

#[test]
fn oracle_atanh_odd_function() {
    for x in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Atanh, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Atanh, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("atanh(-{}) = -atanh({})", x, x),
        );
    }
}

// ======================== Domain boundary: atanh(±1) = ±inf ========================

#[test]
fn oracle_atanh_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "atanh(1) = +inf");
}

#[test]
fn oracle_atanh_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "atanh(-1) = -inf");
}

// ======================== Outside domain: |x| > 1 returns NaN ========================

#[test]
fn oracle_atanh_above_domain() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "atanh(1.5) = NaN (outside domain)"
    );
}

#[test]
fn oracle_atanh_below_domain() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "atanh(-1.5) = NaN (outside domain)"
    );
}

// ======================== METAMORPHIC: tanh(atanh(x)) = x for |x| < 1 ========================

#[test]
fn metamorphic_tanh_atanh_identity() {
    for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let atanh_result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let tanh_atanh = eval_primitive(Primitive::Tanh, &[atanh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&tanh_atanh),
            x,
            1e-12,
            &format!("tanh(atanh({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: atanh(tanh(x)) = x ========================

#[test]
fn metamorphic_atanh_tanh_identity() {
    for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let tanh_result = eval_primitive(Primitive::Tanh, &[input], &no_params()).unwrap();
        let atanh_tanh = eval_primitive(Primitive::Atanh, &[tanh_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&atanh_tanh),
            x,
            1e-12,
            &format!("atanh(tanh({})) = {}", x, x),
        );
    }
}

// ======================== Near boundary ========================

#[test]
fn oracle_atanh_near_one() {
    let input = make_f64_tensor(&[], vec![0.999]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.999_f64.atanh(),
        1e-12,
        "atanh(0.999)",
    );
}

#[test]
fn oracle_atanh_near_neg_one() {
    let input = make_f64_tensor(&[], vec![-0.999]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-0.999_f64).atanh(),
        1e-12,
        "atanh(-0.999)",
    );
}

// ======================== NaN ========================

#[test]
fn oracle_atanh_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "atanh(NaN) = NaN");
}

// ======================== Stdlib comparison ========================

#[test]
fn oracle_atanh_stdlib() {
    for x in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            x.atanh(),
            1e-14,
            &format!("atanh({}) vs stdlib", x),
        );
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 0.5, -0.5]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 0.5, -0.5, 0.9]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![4]);
            let vals = extract_f64_vec(&result);
            assert_close(vals[0], 0.0, 1e-14, "atanh(0)");
            assert_close(vals[1], 0.5_f64.atanh(), 1e-14, "atanh(0.5)");
            assert_close(vals[2], (-0.5_f64).atanh(), 1e-14, "atanh(-0.5)");
            assert_close(vals[3], 0.9_f64.atanh(), 1e-14, "atanh(0.9)");
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![0]);
            assert!(t.elements.is_empty());
        }
        _ => panic!("expected tensor"),
    }
}

// ======================== Additional Coverage ========================

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_atanh_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 0.5, -0.5, 0.9, -0.9, 0.1, -0.1, 0.0]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

#[test]
fn oracle_atanh_2d_empty() {
    let input = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap(),
    );
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_atanh_very_small() {
    let input = make_f64_tensor(&[], vec![1e-15]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-15,
        1e-20,
        "atanh(tiny) ≈ tiny for |x| << 1",
    );
}

#[test]
fn oracle_atanh_subnormal() {
    let tiny = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![tiny, -tiny]);
    let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // For very small x, atanh(x) ≈ x
    assert_close(vals[0], tiny.atanh(), 1e-30, "atanh(subnormal)");
    assert_close(vals[1], (-tiny).atanh(), 1e-30, "atanh(-subnormal)");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_atanh_preserves_all_float_dtypes() {
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

    // atanh domain is |x| < 1
    let values = [-0.5_f64, 0.0, 0.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Atanh, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "atanh {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
