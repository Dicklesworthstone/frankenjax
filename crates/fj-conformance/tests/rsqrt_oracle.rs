//! Oracle tests for Rsqrt (reciprocal square root) primitive.
//!
//! rsqrt(x) = 1/sqrt(x)
//!
//! Tests:
//! - Perfect squares: rsqrt(4) = 0.5, rsqrt(9) = 1/3
//! - Non-perfect squares
//! - Zero: rsqrt(+0.0) = +infinity, rsqrt(-0.0) = -infinity
//! - Negative: rsqrt(-x) = NaN
//! - Infinity: rsqrt(inf) = 0
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

// ======================== Perfect Squares ========================

#[test]
fn oracle_rsqrt_one() {
    // rsqrt(1) = 1/sqrt(1) = 1
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "rsqrt(1)");
}

#[test]
fn oracle_rsqrt_four() {
    // rsqrt(4) = 1/sqrt(4) = 0.5
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "rsqrt(4)");
}

#[test]
fn oracle_rsqrt_nine() {
    // rsqrt(9) = 1/sqrt(9) = 1/3
    let input = make_f64_tensor(&[], vec![9.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0 / 3.0, 1e-14, "rsqrt(9)");
}

#[test]
fn oracle_rsqrt_sixteen() {
    // rsqrt(16) = 1/sqrt(16) = 0.25
    let input = make_f64_tensor(&[], vec![16.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.25, 1e-14, "rsqrt(16)");
}

#[test]
fn oracle_rsqrt_hundred() {
    // rsqrt(100) = 1/sqrt(100) = 0.1
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.1, 1e-14, "rsqrt(100)");
}

// ======================== Non-Perfect Squares ========================

#[test]
fn oracle_rsqrt_two() {
    // rsqrt(2) = 1/sqrt(2) ≈ 0.7071067811865476
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 2.0_f64.sqrt(),
        1e-14,
        "rsqrt(2)",
    );
}

#[test]
fn oracle_rsqrt_three() {
    // rsqrt(3) = 1/sqrt(3) ≈ 0.5773502691896257
    let input = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 3.0_f64.sqrt(),
        1e-14,
        "rsqrt(3)",
    );
}

#[test]
fn oracle_rsqrt_five() {
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / 5.0_f64.sqrt(),
        1e-14,
        "rsqrt(5)",
    );
}

// ======================== Fractional Values ========================

#[test]
fn oracle_rsqrt_quarter() {
    // rsqrt(0.25) = 1/sqrt(0.25) = 1/0.5 = 2
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "rsqrt(0.25)");
}

#[test]
fn oracle_rsqrt_half() {
    // rsqrt(0.5) = 1/sqrt(0.5) = sqrt(2)
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.sqrt(),
        1e-14,
        "rsqrt(0.5)",
    );
}

#[test]
fn oracle_rsqrt_hundredth() {
    // rsqrt(0.01) = 1/sqrt(0.01) = 10
    let input = make_f64_tensor(&[], vec![0.01]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 10.0, 1e-12, "rsqrt(0.01)");
}

// ======================== Zero and Infinity ========================

#[test]
fn oracle_rsqrt_zero() {
    // rsqrt(0) = 1/0 = +infinity
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_eq!(
        val.to_bits(),
        f64::INFINITY.to_bits(),
        "rsqrt(+0.0) should be exact +inf"
    );
}

#[test]
fn oracle_rsqrt_negative_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(
        val.is_infinite() && val.is_sign_negative(),
        "rsqrt(-0.0) should be -inf"
    );
}

#[test]
fn oracle_rsqrt_positive_infinity() {
    // rsqrt(+inf) = 0
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result).to_bits(),
        0.0_f64.to_bits(),
        "rsqrt(+inf) should be exact +0.0"
    );
}

#[test]
fn oracle_rsqrt_f32_signed_zero_and_nan_bits() {
    let input = make_f32_bits_tensor(
        &[9],
        vec![
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            4.0_f32.to_bits(),
            0.25_f32.to_bits(),
            (-1.0_f32).to_bits(),
            f32::NAN.to_bits(),
            0xffc0_0000,
        ],
    );
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let bits = extract_f32_bits_vec(&result);

    assert_eq!(bits[0], f32::INFINITY.to_bits(), "rsqrt(+0.0_f32) = +inf");
    assert_eq!(
        bits[1],
        f32::NEG_INFINITY.to_bits(),
        "rsqrt(-0.0_f32) = -inf"
    );
    assert_eq!(bits[2], 0.0_f32.to_bits(), "rsqrt(+inf_f32) = +0");
    assert!(f32::from_bits(bits[3]).is_nan(), "rsqrt(-inf_f32) = NaN");
    assert_eq!(bits[4], 0.5_f32.to_bits(), "rsqrt(4.0_f32)");
    assert_eq!(bits[5], 2.0_f32.to_bits(), "rsqrt(0.25_f32)");
    assert!(f32::from_bits(bits[6]).is_nan(), "rsqrt(-1.0_f32) = NaN");
    assert!(f32::from_bits(bits[7]).is_nan(), "rsqrt(+nan_f32) = NaN");
    assert!(f32::from_bits(bits[8]).is_nan(), "rsqrt(-nan_f32) = NaN");
}

// ======================== Negative Values (NaN) ========================

#[test]
fn oracle_rsqrt_negative_one() {
    // rsqrt(-1) = NaN (complex result)
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-1) should be NaN"
    );
}

#[test]
fn oracle_rsqrt_negative_four() {
    let input = make_f64_tensor(&[], vec![-4.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-4) should be NaN"
    );
}

#[test]
fn oracle_rsqrt_negative_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(-inf) should be NaN"
    );
}

// ======================== NaN Propagation ========================

#[test]
fn oracle_rsqrt_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "rsqrt(NaN) should be NaN"
    );
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_rsqrt_very_small() {
    // rsqrt(1e-100) = 1e50
    let input = make_f64_tensor(&[], vec![1e-100]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1e50, 1e35, "rsqrt(1e-100)");
}

#[test]
fn oracle_rsqrt_very_large() {
    // rsqrt(1e100) = 1e-50
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1e-50, 1e-64, "rsqrt(1e100)");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_rsqrt_1d() {
    let input = make_f64_tensor(&[4], vec![1.0, 4.0, 9.0, 16.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "rsqrt(1)");
    assert_close(vals[1], 0.5, 1e-14, "rsqrt(4)");
    assert_close(vals[2], 1.0 / 3.0, 1e-14, "rsqrt(9)");
    assert_close(vals[3], 0.25, 1e-14, "rsqrt(16)");
}

#[test]
fn oracle_rsqrt_1d_mixed() {
    // Mix of positive values and special cases
    let input = make_f64_tensor(&[5], vec![4.0, 0.0, 1.0, -1.0, f64::INFINITY]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.5, 1e-14, "rsqrt(4)");
    assert_eq!(
        vals[1].to_bits(),
        f64::INFINITY.to_bits(),
        "rsqrt(+0.0) = exact +inf"
    );
    assert_close(vals[2], 1.0, 1e-14, "rsqrt(1)");
    assert!(vals[3].is_nan(), "rsqrt(-1) = NaN");
    assert_eq!(
        vals[4].to_bits(),
        0.0_f64.to_bits(),
        "rsqrt(+inf) = exact +0.0"
    );
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_rsqrt_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "rsqrt(1)");
    assert_close(vals[1], 0.5, 1e-14, "rsqrt(4)");
    assert_close(vals[2], 1.0 / 3.0, 1e-14, "rsqrt(9)");
    assert_close(vals[3], 0.25, 1e-14, "rsqrt(16)");
    assert_close(vals[4], 0.2, 1e-14, "rsqrt(25)");
    assert_close(vals[5], 1.0 / 6.0, 1e-14, "rsqrt(36)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_rsqrt_3d() {
    let input = make_f64_tensor(
        &[2, 2, 2],
        vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0],
    );
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[6], 1.0 / 7.0, 1e-14, "rsqrt(49)");
    assert_close(vals[7], 0.125, 1e-14, "rsqrt(64)");
}

// ======================== Identity: rsqrt(x) * sqrt(x) = 1 ========================

#[test]
fn oracle_rsqrt_identity() {
    // Verify: rsqrt(x) = 1/sqrt(x) by checking rsqrt(x) * sqrt(x) ≈ 1
    for x in [2.0, 3.0, 5.0, 7.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        let rsqrt_val = extract_f64_scalar(&result);
        let product = rsqrt_val * x.sqrt();
        assert_close(
            product,
            1.0,
            1e-14,
            &format!("rsqrt({}) * sqrt({}) = 1", x, x),
        );
    }
}

// ======================== Identity: rsqrt(x)^2 * x = 1 ========================

#[test]
fn oracle_rsqrt_squared_identity() {
    // rsqrt(x)^2 = 1/x
    for x in [2.0, 4.0, 9.0, 16.0, 25.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        let rsqrt_val = extract_f64_scalar(&result);
        let squared = rsqrt_val * rsqrt_val;
        assert_close(
            squared,
            1.0 / x,
            1e-14,
            &format!("rsqrt({})^2 = 1/{}", x, x),
        );
    }
}

// ======================== METAMORPHIC: rsqrt(x) = Reciprocal(Sqrt(x)) ========================

#[test]
fn metamorphic_rsqrt_equals_reciprocal_sqrt() {
    // rsqrt(x) = Reciprocal(Sqrt(x)) using primitives
    for x in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);

        // rsqrt(x)
        let rsqrt_result =
            eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&input), &no_params()).unwrap();

        // Reciprocal(Sqrt(x))
        let sqrt_x = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
        let recip_sqrt = eval_primitive(Primitive::Reciprocal, &[sqrt_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&rsqrt_result),
            extract_f64_scalar(&recip_sqrt),
            1e-14,
            &format!("rsqrt({}) = Reciprocal(Sqrt({}))", x, x),
        );
    }
}

// ======================== METAMORPHIC: Square(rsqrt(x)) = Reciprocal(x) ========================

#[test]
fn metamorphic_rsqrt_squared_reciprocal() {
    // Square(rsqrt(x)) = Reciprocal(x) using primitives
    for x in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);

        // Square(rsqrt(x))
        let rsqrt_x =
            eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&input), &no_params()).unwrap();
        let squared = eval_primitive(Primitive::Square, &[rsqrt_x], &no_params()).unwrap();

        // Reciprocal(x)
        let recip_x = eval_primitive(Primitive::Reciprocal, &[input], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&squared),
            extract_f64_scalar(&recip_x),
            1e-14,
            &format!("Square(rsqrt({})) = Reciprocal({})", x, x),
        );
    }
}

// ======================== METAMORPHIC: Mul(rsqrt(x), Sqrt(x)) = 1 ========================

#[test]
fn metamorphic_rsqrt_mul_sqrt_one() {
    // Mul(rsqrt(x), Sqrt(x)) = 1 using primitives
    for x in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);

        let rsqrt_x =
            eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&input), &no_params()).unwrap();
        let sqrt_x = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
        let product = eval_primitive(Primitive::Mul, &[rsqrt_x, sqrt_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&product),
            1.0,
            1e-14,
            &format!("Mul(rsqrt({}), Sqrt({})) = 1", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor rsqrt = reciprocal(sqrt) ========================

#[test]
fn metamorphic_rsqrt_tensor_reciprocal_sqrt() {
    // For tensor: rsqrt(x) = Reciprocal(Sqrt(x))
    let data = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    let input = make_f64_tensor(&[5], data);

    let rsqrt_result =
        eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&input), &no_params()).unwrap();
    let sqrt_x = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    let recip_sqrt = eval_primitive(Primitive::Reciprocal, &[sqrt_x], &no_params()).unwrap();

    assert_eq!(extract_shape(&rsqrt_result), vec![5]);
    let rsqrt_vec = extract_f64_vec(&rsqrt_result);
    let recip_sqrt_vec = extract_f64_vec(&recip_sqrt);
    for (i, (&r1, &r2)) in rsqrt_vec.iter().zip(recip_sqrt_vec.iter()).enumerate() {
        assert_close(r1, r2, 1e-14, &format!("element {}", i));
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_rsqrt_preserves_all_float_dtypes() {
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

    // rsqrt domain is x > 0
    let values = [1.0_f64, 4.0, 9.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "rsqrt {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 coverage ========================
//
// Complex rsqrt: rsqrt(z) = 1/sqrt(z)

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
fn oracle_rsqrt_complex128_real_positive() {
    // rsqrt(4+0i) = 0.5+0i
    let input = make_complex128_tensor(&[], &[(4.0, 0.0)]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.5, 0.0), 1e-10, "rsqrt(4+0i) = 0.5");
}

#[test]
fn oracle_rsqrt_complex128_unit_imaginary() {
    // rsqrt(i) = (1-i)/sqrt(2) = 0.7071... - 0.7071...i
    // since sqrt(i) = (1+i)/sqrt(2) and rsqrt = 1/sqrt
    let input = make_complex128_tensor(&[], &[(0.0, 1.0)]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    assert_complex_close(vec[0], (sqrt2_inv, -sqrt2_inv), 1e-10, "rsqrt(i)");
}

#[test]
fn oracle_rsqrt_complex128_three_four() {
    // sqrt(3+4i) = 2+i, so rsqrt(3+4i) = 1/(2+i) = (2-i)/5 = 0.4-0.2i
    let input = make_complex128_tensor(&[], &[(3.0, 4.0)]);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vec = extract_complex128_vec(&result);
    assert_eq!(vec.len(), 1);
    assert_complex_close(vec[0], (0.4, -0.2), 1e-10, "rsqrt(3+4i) = 0.4-0.2i");
}

#[test]
fn oracle_rsqrt_complex64_vector() {
    let data: &[(f32, f32)] = &[(4.0, 0.0), (1.0, 0.0), (3.0, 4.0)];
    let input = make_complex64_tensor(&[3], data);
    let result = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
    let vec = extract_complex64_vec(&result);
    assert_eq!(vec.len(), 3);

    // rsqrt(4) = 0.5
    assert_complex_close(
        (vec[0].0 as f64, vec[0].1 as f64),
        (0.5, 0.0),
        1e-4,
        "rsqrt(4)",
    );

    // rsqrt(1) = 1
    assert_complex_close(
        (vec[1].0 as f64, vec[1].1 as f64),
        (1.0, 0.0),
        1e-4,
        "rsqrt(1)",
    );

    // rsqrt(3+4i) = 0.4-0.2i
    assert_complex_close(
        (vec[2].0 as f64, vec[2].1 as f64),
        (0.4, -0.2),
        1e-4,
        "rsqrt(3+4i)",
    );
}

#[test]
fn oracle_rsqrt_complex_sqrt_reciprocal_identity() {
    // rsqrt(z) = reciprocal(sqrt(z))
    let values: &[(f64, f64)] = &[(4.0, 0.0), (3.0, 4.0), (1.0, 1.0)];

    for &(a, b) in values {
        let input = make_complex128_tensor(&[], &[(a, b)]);

        let rsqrt_result =
            eval_primitive(Primitive::Rsqrt, &[input.clone()], &no_params()).unwrap();
        let sqrt_result = eval_primitive(Primitive::Sqrt, &[input.clone()], &no_params()).unwrap();
        let recip_sqrt =
            eval_primitive(Primitive::Reciprocal, &[sqrt_result], &no_params()).unwrap();

        let rsqrt_val = extract_complex128_vec(&rsqrt_result)[0];
        let recip_sqrt_val = extract_complex128_vec(&recip_sqrt)[0];

        assert_complex_close(
            rsqrt_val,
            recip_sqrt_val,
            1e-10,
            &format!("rsqrt({a}+{b}i) = 1/sqrt({a}+{b}i)"),
        );
    }
}

#[test]
fn oracle_rsqrt_complex_dtype_preservation() {
    // Complex64 -> Complex64
    let c64_input = make_complex64_tensor(&[2], &[(4.0, 0.0), (3.0, 4.0)]);
    let c64_result = eval_primitive(Primitive::Rsqrt, &[c64_input], &no_params()).unwrap();
    match &c64_result {
        Value::Tensor(t) => {
            assert_eq!(t.dtype, DType::Complex64, "rsqrt should preserve Complex64");
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }

    // Complex128 -> Complex128
    let c128_input = make_complex128_tensor(&[2], &[(4.0, 0.0), (3.0, 4.0)]);
    let c128_result = eval_primitive(Primitive::Rsqrt, &[c128_input], &no_params()).unwrap();
    match &c128_result {
        Value::Tensor(t) => {
            assert_eq!(
                t.dtype,
                DType::Complex128,
                "rsqrt should preserve Complex128"
            );
            t.validate_dtype_consistency().unwrap();
        }
        _ => panic!("expected tensor"),
    }
}
