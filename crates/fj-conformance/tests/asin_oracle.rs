//! Oracle tests for Asin (arc sine) primitive.
//!
//! asin(x) = inverse sine of x
//!
//! Domain: [-1, 1]
//! Range: [-π/2, π/2]
//!
//! Tests:
//! - Boundary: asin(-1) = -π/2, asin(1) = π/2
//! - Special: asin(0) = 0
//! - Out of domain: asin(x) = NaN for |x| > 1
//! - Infinity: asin(±inf) = NaN
//! - NaN propagation
//! - Odd function property
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

// ======================== Boundary Values ========================

#[test]
fn oracle_asin_one() {
    // asin(1) = π/2
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_2,
        1e-14,
        "asin(1)",
    );
}

#[test]
fn oracle_asin_neg_one() {
    // asin(-1) = -π/2
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_2,
        1e-14,
        "asin(-1)",
    );
}

#[test]
fn oracle_asin_zero() {
    // asin(0) = 0
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "asin(0) = +0");
}

#[test]
fn oracle_asin_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "asin(-0.0) = -0");
}

// ======================== Common Values ========================

#[test]
fn oracle_asin_half() {
    // asin(0.5) = π/6
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_6,
        1e-14,
        "asin(0.5)",
    );
}

#[test]
fn oracle_asin_neg_half() {
    // asin(-0.5) = -π/6
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_6,
        1e-14,
        "asin(-0.5)",
    );
}

#[test]
fn oracle_asin_sqrt2_over_2() {
    // asin(√2/2) = π/4
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_1_SQRT_2]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_4,
        1e-14,
        "asin(√2/2)",
    );
}

#[test]
fn oracle_asin_neg_sqrt2_over_2() {
    // asin(-√2/2) = -π/4
    let input = make_f64_tensor(&[], vec![-std::f64::consts::FRAC_1_SQRT_2]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::FRAC_PI_4,
        1e-14,
        "asin(-√2/2)",
    );
}

#[test]
fn oracle_asin_sqrt3_over_2() {
    // asin(√3/2) = π/3
    let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
    let input = make_f64_tensor(&[], vec![sqrt3_over_2]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_PI_3,
        1e-14,
        "asin(√3/2)",
    );
}

// ======================== Out of Domain ========================

#[test]
fn oracle_asin_greater_than_one() {
    for x in [1.1, 2.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result).is_nan(),
            "asin({}) should be NaN",
            x
        );
    }
}

#[test]
fn oracle_asin_less_than_neg_one() {
    for x in [-1.1, -2.0, -10.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result).is_nan(),
            "asin({}) should be NaN",
            x
        );
    }
}

// ======================== Infinity ========================

#[test]
fn oracle_asin_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "asin(+inf) = NaN");
}

#[test]
fn oracle_asin_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "asin(-inf) = NaN");
}

// ======================== NaN ========================

#[test]
fn oracle_asin_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "asin(NaN) = NaN");
}

// ======================== Range: output in [-π/2, π/2] ========================

#[test]
fn oracle_asin_range() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val >= -std::f64::consts::FRAC_PI_2, "asin({}) >= -π/2", x);
        assert!(val <= std::f64::consts::FRAC_PI_2, "asin({}) <= π/2", x);
    }
}

// ======================== Odd Function: asin(-x) = -asin(x) ========================

#[test]
fn oracle_asin_odd_function() {
    for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Asin, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Asin, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_neg,
            -val_pos,
            1e-14,
            &format!("asin(-{}) = -asin({})", x, x),
        );
    }
}

// ======================== Monotonicity: asin is increasing ========================

#[test]
fn oracle_asin_monotonic_increasing() {
    let inputs = vec![-0.9, -0.5, 0.0, 0.5, 0.9];
    let input = make_f64_tensor(&[5], inputs);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "asin should be monotonically increasing"
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_asin_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -std::f64::consts::FRAC_PI_2, 1e-14, "asin(-1)");
    assert_close(vals[1], -std::f64::consts::FRAC_PI_6, 1e-14, "asin(-0.5)");
    assert_eq!(vals[2], 0.0, "asin(0)");
    assert_close(vals[3], std::f64::consts::FRAC_PI_6, 1e-14, "asin(0.5)");
    assert_close(vals[4], std::f64::consts::FRAC_PI_2, 1e-14, "asin(1)");
}

#[test]
fn oracle_asin_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, 2.0, f64::INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "asin(0)");
    assert!(vals[1].is_nan(), "asin(2) = NaN");
    assert!(vals[2].is_nan(), "asin(inf) = NaN");
    assert!(vals[3].is_nan(), "asin(NaN) = NaN");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_asin_2d() {
    let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
    let input = make_f64_tensor(&[2, 3], vec![-1.0, -sqrt2_2, 0.0, sqrt2_2, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -std::f64::consts::FRAC_PI_2, 1e-14, "asin(-1)");
    assert_eq!(vals[2], 0.0, "asin(0)");
    assert_close(vals[5], std::f64::consts::FRAC_PI_2, 1e-14, "asin(1)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_asin_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-1.0, -0.5, 0.0, 0.5, -0.9, -0.1, 0.1, 0.9]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], -std::f64::consts::FRAC_PI_2, 1e-14, "asin(-1)");
    assert_eq!(vals[2], 0.0, "asin(0)");
}

// ======================== Identity: sin(asin(x)) = x ========================

#[test]
fn oracle_asin_sin_identity() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
        let asin_val = extract_f64_scalar(&result);

        let input2 = make_f64_tensor(&[], vec![asin_val]);
        let result2 = eval_primitive(Primitive::Sin, &[input2], &no_params()).unwrap();
        let roundtrip = extract_f64_scalar(&result2);

        assert_close(roundtrip, x, 1e-14, &format!("sin(asin({})) = {}", x, x));
    }
}

// ======================== METAMORPHIC: asin(sin(x)) = x for x in [-π/2, π/2] ========================

#[test]
fn metamorphic_asin_sin_identity() {
    // asin(sin(x)) = x for x in [-π/2, π/2]
    for x in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let sin_result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
        let asin_sin = eval_primitive(Primitive::Asin, &[sin_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&asin_sin),
            x,
            1e-12,
            &format!("asin(sin({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_asin_tensor_roundtrip() {
    // Test both directions on a tensor
    let input = make_f64_tensor(&[5], vec![-0.9, -0.5, 0.0, 0.5, 0.9]);
    let asin_result =
        eval_primitive(Primitive::Asin, std::slice::from_ref(&input), &no_params()).unwrap();
    let sin_asin = eval_primitive(Primitive::Sin, &[asin_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&sin_asin);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(
            *rt,
            *orig,
            1e-12,
            &format!("sin(asin({})) = {}", orig, orig),
        );
    }
}

// ======================== Relationship: asin(x) + acos(x) = π/2 ========================

#[test]
fn oracle_asin_acos_relationship() {
    for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let asin_result =
            eval_primitive(Primitive::Asin, std::slice::from_ref(&input), &no_params()).unwrap();
        let acos_result = eval_primitive(Primitive::Acos, &[input], &no_params()).unwrap();

        let asin_val = extract_f64_scalar(&asin_result);
        let acos_val = extract_f64_scalar(&acos_result);

        assert_close(
            asin_val + acos_val,
            std::f64::consts::FRAC_PI_2,
            1e-14,
            &format!("asin({}) + acos({}) = π/2", x, x),
        );
    }
}

// ======================== Property: dtype preservation across all float types ========================

#[test]
fn property_asin_preserves_all_float_dtypes() {
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

    // Use values in asin domain [-1, 1]
    let values = [-0.5_f64, 0.0, 0.5];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Asin, &[input], &no_params())
            .unwrap_or_else(|e| panic!("asin {dtype:?} failed: {e}"));
        let Value::Tensor(t) = result else {
            panic!("asin {dtype:?}: expected tensor");
        };
        assert_eq!(t.dtype, dtype, "asin {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("asin {dtype:?}: validate_dtype_consistency failed: {e}"));
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex asin: asin(z) = -i * asinh(i*z) = -i * log(i*z + sqrt(1 - z²))

fn make_complex64_tensor(shape: &[u32], data: &[(f32, f32)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: &[(f64, f64)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex64Bits(re, im) => (f32::from_bits(*re), f32::from_bits(*im)),
                _ => panic!("expected Complex64"),
            })
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let (ar, ai) = actual;
    let (er, ei) = expected;
    let re_diff = (ar - er).abs();
    let im_diff = (ai - ei).abs();
    assert!(
        re_diff < tol && im_diff < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        er,
        ei,
        ar,
        ai,
        re_diff,
        im_diff
    );
}

#[test]
fn oracle_asin_complex64_zero() {
    let input = make_complex64_tensor(&[], &[(0.0, 0.0)]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-6,
        "asin(0+0i)",
    );
}

#[test]
fn oracle_asin_complex128_real_in_domain() {
    let x = 0.5_f64;
    let input = make_complex128_tensor(&[], &[(x, 0.0)]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (x.asin(), 0.0), 1e-10, "asin(0.5+0i) = asin(0.5)");
}

#[test]
fn oracle_asin_complex128_pure_imaginary() {
    // asin(i) = i * asinh(1) ≈ i * 0.8814
    let input = make_complex128_tensor(&[], &[(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(
        vec[0],
        (0.0, 1.0_f64.asinh()),
        1e-10,
        "asin(i) = i*asinh(1)",
    );
}

#[test]
fn oracle_asin_complex64_vector() {
    let data: &[(f32, f32)] = &[(0.0, 0.0), (0.5, 0.0), (0.0, 1.0)];
    let input = make_complex64_tensor(&[3], data);
    let result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 3);

    // asin(0) = 0
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.0, 0.0),
        1e-5,
        "asin(0)",
    );

    // asin(0.5) = asin(0.5)
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (0.5_f64.asin(), 0.0),
        1e-4,
        "asin(0.5)",
    );

    // asin(i) = i*asinh(1)
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.0, 1.0_f64.asinh()),
        1e-4,
        "asin(i)",
    );
}

#[test]
fn oracle_asin_complex_sin_inverse_identity() {
    // sin(asin(z)) = z for |Re(z)| <= pi/2
    let values: &[(f64, f64)] = &[(0.5, 0.0), (0.0, 0.5), (0.3, 0.2)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);
        let asin_result = eval_primitive(Primitive::Asin, &[input], &no_params()).unwrap();
        let sin_asin = eval_primitive(Primitive::Sin, &[asin_result], &no_params()).unwrap();

        let result = extract_complex128_vec(&sin_asin)[0];
        assert_complex_close(
            result,
            (a, b),
            1e-9,
            &format!("sin(asin({a}+{b}i)) = {a}+{b}i"),
        );
    }
}

#[test]
fn oracle_asin_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(0.5, 0.0), (0.0, 0.5)]);
    let c64_result = eval_primitive(Primitive::Asin, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "asin should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(0.5, 0.0), (0.0, 0.5)]);
    let c128_result = eval_primitive(Primitive::Asin, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "asin should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
