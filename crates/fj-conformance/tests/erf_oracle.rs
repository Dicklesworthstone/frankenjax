//! Oracle tests for Erf primitive.
//!
//! erf(x) = Gauss error function = (2/sqrt(π)) * ∫[0,x] exp(-t²) dt
//!
//! Domain: (-inf, inf)
//! Range: (-1, 1)
//!
//! Properties:
//! - erf(0) = 0
//! - erf(inf) = 1
//! - erf(-inf) = -1
//! - Odd function: erf(-x) = -erf(x)
//!
//! Tests:
//! - Special values
//! - Odd function property
//! - Range verification
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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
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

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_erf_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "erf(0) = 0");
}

#[test]
fn oracle_erf_one() {
    // erf(1) ≈ 0.8427007929497149
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.8427007929497149,
        1e-12,
        "erf(1)",
    );
}

#[test]
fn oracle_erf_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -0.8427007929497149,
        1e-12,
        "erf(-1)",
    );
}

#[test]
fn oracle_erf_two() {
    // erf(2) ≈ 0.9953222650189527
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.9953222650189527,
        1e-12,
        "erf(2)",
    );
}

#[test]
fn oracle_erf_half() {
    // erf(0.5) ≈ 0.5204998778130465
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.5204998778130465,
        1e-12,
        "erf(0.5)",
    );
}

// ====================== INFINITY ======================

#[test]
fn oracle_erf_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "erf(inf) = 1");
}

#[test]
fn oracle_erf_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "erf(-inf) = -1");
}

// ====================== NaN ======================

#[test]
fn oracle_erf_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "erf(NaN) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_erf_range() {
    // erf(x) should always be in (-1, 1)
    let test_values = [-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0];
    for x in test_values {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (-1.0..=1.0).contains(&val),
            "erf({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

// ====================== ODD FUNCTION ======================

#[test]
fn oracle_erf_odd_function() {
    // erf(-x) = -erf(x)
    for x in [0.1, 0.5, 1.0, 2.0, 3.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Erf, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Erf, &[neg_input], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&neg_result),
            -extract_f64_scalar(&pos_result),
            1e-12,
            &format!("erf(-{}) = -erf({})", x, x),
        );
    }
}

// ====================== ASYMPTOTIC BEHAVIOR ======================

#[test]
fn oracle_erf_large_positive() {
    // For large x, erf(x) → 1
    for x in [5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val - 1.0).abs() < 1e-4,
            "erf({}) should be very close to 1, got {}",
            x,
            val
        );
    }
}

#[test]
fn oracle_erf_large_negative() {
    // For large negative x, erf(x) → -1
    for x in [-5.0, -10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(
            (val + 1.0).abs() < 1e-4,
            "erf({}) should be very close to -1, got {}",
            x,
            val
        );
    }
}

#[test]
fn oracle_erf_small_values() {
    // For small |x|, erf(x) ≈ (2/sqrt(π)) * x
    let coeff = 2.0 / std::f64::consts::PI.sqrt();
    for x in [0.001, 0.0001] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = coeff * x;
        assert_close(
            val,
            expected,
            expected.abs() * 0.01,
            &format!("erf({}) ≈ 2x/sqrt(π)", x),
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_erf_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, 0.0, 0.5, 1.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -0.8427007929497149, 1e-12, "erf(-1)");
    assert_eq!(vals[1], 0.0, "erf(0)");
    assert_close(vals[2], 0.5204998778130465, 1e-12, "erf(0.5)");
    assert_close(vals[3], 0.8427007929497149, 1e-12, "erf(1)");
    assert_eq!(vals[4], 1.0, "erf(inf)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_erf_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0);
    assert_close(vals[1], 0.8427007929497149, 1e-12, "erf(1)");
    assert_close(vals[2], -0.8427007929497149, 1e-12, "erf(-1)");
    assert_close(vals[3], 0.9953222650189527, 1e-12, "erf(2)");
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_erf_monotonic() {
    // erf is strictly increasing
    let values: Vec<f64> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "erf should be strictly increasing: erf[{}] = {} > erf[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ====================== ERFC RELATIONSHIP ======================

#[test]
fn oracle_erf_erfc_relationship() {
    // erf(x) + erfc(x) = 1
    for x in [0.0, 0.5, 1.0, 2.0, -0.5, -1.0] {
        let erf_input = make_f64_tensor(&[], vec![x]);
        let erfc_input = make_f64_tensor(&[], vec![x]);
        let erf_result = eval_primitive(Primitive::Erf, &[erf_input], &no_params()).unwrap();
        let erfc_result = eval_primitive(Primitive::Erfc, &[erfc_input], &no_params()).unwrap();
        let erf_val = extract_f64_scalar(&erf_result);
        let erfc_val = extract_f64_scalar(&erfc_result);
        assert_close(
            erf_val + erfc_val,
            1.0,
            1e-12,
            &format!("erf({}) + erfc({}) = 1", x, x),
        );
    }
}

// ======================== METAMORPHIC: erfinv(erf(x)) = x ========================

#[test]
fn metamorphic_erfinv_erf_identity() {
    // erfinv(erf(x)) = x for "moderate" x (erf approaches ±1 asymptotically)
    // Use values where erf(x) is not too close to ±1 for numerical stability
    for x in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5] {
        let input = make_f64_tensor(&[], vec![x]);
        let erf_result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let erfinv_erf = eval_primitive(Primitive::ErfInv, &[erf_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&erfinv_erf),
            x,
            1e-10,
            &format!("erfinv(erf({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: erf(erfinv(y)) = y ========================

#[test]
fn metamorphic_erf_erfinv_identity() {
    // erf(erfinv(y)) = y for y in (-1, 1)
    for y in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![y]);
        let erfinv_result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
        let erf_erfinv = eval_primitive(Primitive::Erf, &[erfinv_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&erf_erfinv),
            y,
            1e-10,
            &format!("erf(erfinv({})) = {}", y, y),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_erf_erfinv_tensor_roundtrip() {
    // For a tensor with values in (-1, 1): erf(erfinv(y)) = y
    let input = make_f64_tensor(&[5], vec![-0.8, -0.4, 0.0, 0.4, 0.8]);
    let erfinv_result = eval_primitive(
        Primitive::ErfInv,
        std::slice::from_ref(&input),
        &no_params(),
    )
    .unwrap();
    let roundtrip = eval_primitive(Primitive::Erf, &[erfinv_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let recovered = extract_f64_vec(&roundtrip);

    for (orig, rec) in original.iter().zip(recovered.iter()) {
        assert_close(
            *rec,
            *orig,
            1e-10,
            &format!("erf(erfinv({})) = {}", orig, orig),
        );
    }
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_erf_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[2], 0.0, 1e-10, "erf(0) = 0");
}

#[test]
fn oracle_erf_empty() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
}

#[test]
fn oracle_erf_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0, 3]);
}

#[test]
fn oracle_erf_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn oracle_erf_subnormal() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_tensor(&[2], vec![subnormal, -subnormal]);
    let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    // For very small x, erf(x) should be very small and have correct sign
    assert!(
        vals[0] >= 0.0 && vals[0] < 1e-6,
        "erf(subnormal) should be small positive"
    );
    assert!(
        vals[1] <= 0.0 && vals[1] > -1e-6,
        "erf(-subnormal) should be small negative"
    );
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_erf_preserves_all_float_dtypes() {
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
        let result = eval_primitive(Primitive::Erf, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "erf {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

// erf no longer computes complex (float-only per w8u0a); this helper is retained for
// potential future complex-erf coverage but is currently unused by the rejection tests.
#[allow(dead_code)]
fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => unreachable!("expected tensor"),
    }
}

#[test]

fn oracle_erf_complex64_real_axis_rejected() {
    // erf is JAX standard_unop(_float): it REJECTS complex operands at the eval level
    // (w8u0a float-only decision; complex_unary_elementwise returns None for Erf), even
    // for real-axis complex inputs. (This test previously asserted complex-erf VALUES,
    // which is stale — erf no longer computes complex.)
    let input = make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let err = eval_primitive(Primitive::Erf, &[input], &no_params())
        .expect_err("erf is float-only and must reject complex64");
    assert!(
        err.to_string().contains("complex"),
        "unexpected complex erf error: {err}"
    );
}

#[test]

fn oracle_erf_complex64_purely_imaginary_rejected() {
    // erf rejects complex even on the imaginary axis (float-only per w8u0a; previously
    // asserted erf(i) ≈ 1.6505i, now stale).
    let input = make_complex64_tensor(&[1], vec![(0.0, 1.0)]);
    let err = eval_primitive(Primitive::Erf, &[input], &no_params())
        .expect_err("erf is float-only and must reject complex64");
    assert!(
        err.to_string().contains("complex"),
        "unexpected complex erf error: {err}"
    );
}

#[test]
fn oracle_erf_rejects_complex_operands_even_large() {
    // CORRECTION of a previously-wrong test: JAX erf is standard_unop(_float), so erf
    // REJECTS complex operands at the eval level (complex_unary_elementwise returns None
    // -> "complex operands" error), consistent with complex_ops_oracle's float-vs-complex
    // boundary guard. (The complex_erf function exists and is exercised by its own fj-lax
    // lib test, but it is NOT wired into the eval dispatch — erf(complex) fails closed.)
    // Confirm rejection holds for large real-axis complex inputs too, not just small ones.
    let input = make_complex64_tensor(&[2], vec![(4.0, 0.0), (5.0, 0.0)]);
    let err = eval_primitive(Primitive::Erf, &[input], &no_params())
        .expect_err("JAX erf is float-only and must reject complex operands");
    assert!(
        err.to_string().contains("complex"),
        "unexpected complex erf error: {err}"
    );
}

#[test]

fn oracle_erf_complex128_rejected() {
    // erf rejects complex128 too (float-only per w8u0a; previously asserted dtype
    // preservation, which is stale — there is no complex erf result).
    let input = make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let err = eval_primitive(Primitive::Erf, &[input], &no_params())
        .expect_err("erf is float-only and must reject complex128");
    assert!(
        err.to_string().contains("complex"),
        "unexpected complex erf error: {err}"
    );
}

#[test]

fn property_erf_rejects_all_complex_dtypes() {
    // erf is float-only (w8u0a): every complex dtype is rejected (previously asserted
    // dtype preservation, now stale).
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            DType::Complex128 => make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            _ => unreachable!(),
        };
        let err = eval_primitive(Primitive::Erf, &[input], &no_params())
            .expect_err("erf is float-only and must reject all complex dtypes");
        assert!(
            err.to_string().contains("complex"),
            "erf {dtype:?}: unexpected error {err}"
        );
    }
}
