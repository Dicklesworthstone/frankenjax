//! Oracle tests for Logistic (sigmoid) primitive.
//!
//! logistic(x) = 1 / (1 + e^(-x))
//!
//! Tests:
//! - Zero: logistic(0) = 0.5
//! - Positive values → approaches 1
//! - Negative values → approaches 0
//! - Infinity: logistic(+inf) = 1, logistic(-inf) = 0
//! - NaN propagation
//! - Symmetry: logistic(-x) = 1 - logistic(x)
//! - Bounds: output always in (0, 1) for finite x

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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ======================== Zero ========================

#[test]
fn oracle_logistic_zero() {
    // logistic(0) = 1/(1+e^0) = 1/2 = 0.5
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "logistic(0)");
}

#[test]
fn oracle_logistic_neg_zero() {
    // logistic(-0.0) = 0.5
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "logistic(-0.0)");
}

// ======================== Positive Values ========================

#[test]
fn oracle_logistic_one() {
    // logistic(1) = 1/(1+e^-1) ≈ 0.7310586
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        sigmoid(1.0),
        1e-14,
        "logistic(1)",
    );
}

#[test]
fn oracle_logistic_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        sigmoid(2.0),
        1e-14,
        "logistic(2)",
    );
}

#[test]
fn oracle_logistic_five() {
    let input = make_f64_tensor(&[], vec![5.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, sigmoid(5.0), 1e-14, "logistic(5)");
    assert!(val > 0.99, "logistic(5) should be > 0.99");
}

#[test]
fn oracle_logistic_ten() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, sigmoid(10.0), 1e-14, "logistic(10)");
    assert!(val > 0.9999, "logistic(10) should be > 0.9999");
}

#[test]
fn oracle_logistic_large() {
    // logistic(50) should be very close to 1
    let input = make_f64_tensor(&[], vec![50.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1.0, 1e-14, "logistic(50)");
}

// ======================== Negative Values ========================

#[test]
fn oracle_logistic_neg_one() {
    // logistic(-1) = 1/(1+e^1) ≈ 0.2689414
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        sigmoid(-1.0),
        1e-14,
        "logistic(-1)",
    );
}

#[test]
fn oracle_logistic_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        sigmoid(-2.0),
        1e-14,
        "logistic(-2)",
    );
}

#[test]
fn oracle_logistic_neg_five() {
    let input = make_f64_tensor(&[], vec![-5.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, sigmoid(-5.0), 1e-14, "logistic(-5)");
    assert!(val < 0.01, "logistic(-5) should be < 0.01");
}

#[test]
fn oracle_logistic_neg_ten() {
    let input = make_f64_tensor(&[], vec![-10.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, sigmoid(-10.0), 1e-14, "logistic(-10)");
    assert!(val < 0.0001, "logistic(-10) should be < 0.0001");
}

#[test]
fn oracle_logistic_neg_large() {
    // logistic(-50) should be very close to 0
    let input = make_f64_tensor(&[], vec![-50.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 0.0, 1e-14, "logistic(-50)");
}

// ======================== Infinity ========================

#[test]
fn oracle_logistic_pos_infinity() {
    // logistic(+inf) = 1
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "logistic(+inf) = 1");
}

#[test]
fn oracle_logistic_neg_infinity() {
    // logistic(-inf) = 0
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "logistic(-inf) = 0");
}

// ======================== NaN ========================

#[test]
fn oracle_logistic_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "logistic(NaN) = NaN");
}

// ======================== Bounds: output in (0, 1) ========================

#[test]
fn oracle_logistic_bounds_positive() {
    for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > 0.5, "logistic({}) > 0.5", x);
        assert!(val <= 1.0, "logistic({}) <= 1", x);
    }
}

#[test]
fn oracle_logistic_bounds_negative() {
    for x in [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val >= 0.0, "logistic({}) >= 0", x);
        assert!(val < 0.5, "logistic({}) < 0.5", x);
    }
}

// ======================== Symmetry: logistic(-x) = 1 - logistic(x) ========================

#[test]
fn oracle_logistic_symmetry() {
    for x in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Logistic, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Logistic, &[input_neg], &no_params()).unwrap();

        let val_pos = extract_f64_scalar(&result_pos);
        let val_neg = extract_f64_scalar(&result_neg);

        assert_close(
            val_pos + val_neg,
            1.0,
            1e-14,
            &format!("logistic({}) + logistic(-{}) = 1", x, x),
        );
    }
}

// ======================== Derivative property: logistic'(x) = logistic(x) * (1 - logistic(x)) ========================

#[test]
fn oracle_logistic_derivative_at_zero() {
    // At x=0, logistic(0) = 0.5, so derivative = 0.5 * 0.5 = 0.25
    // This is the maximum derivative
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    let derivative = val * (1.0 - val);
    assert_close(derivative, 0.25, 1e-14, "logistic'(0) = 0.25");
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_logistic_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], sigmoid(-2.0), 1e-14, "logistic(-2)");
    assert_close(vals[1], sigmoid(-1.0), 1e-14, "logistic(-1)");
    assert_close(vals[2], 0.5, 1e-14, "logistic(0)");
    assert_close(vals[3], sigmoid(1.0), 1e-14, "logistic(1)");
    assert_close(vals[4], sigmoid(2.0), 1e-14, "logistic(2)");
}

#[test]
fn oracle_logistic_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 0.5, 1e-14, "logistic(0)");
    assert_eq!(vals[1], 1.0, "logistic(+inf) = 1");
    assert_eq!(vals[2], 0.0, "logistic(-inf) = 0");
    assert!(vals[3].is_nan(), "logistic(NaN) = NaN");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_logistic_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], sigmoid(-3.0), 1e-14, "logistic(-3)");
    assert_close(vals[1], sigmoid(-1.0), 1e-14, "logistic(-1)");
    assert_close(vals[2], 0.5, 1e-14, "logistic(0)");
    assert_close(vals[3], sigmoid(1.0), 1e-14, "logistic(1)");
    assert_close(vals[4], sigmoid(3.0), 1e-14, "logistic(3)");
    assert_close(vals[5], sigmoid(5.0), 1e-14, "logistic(5)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_logistic_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], sigmoid(-4.0), 1e-14, "logistic(-4)");
    assert_close(vals[3], 0.5, 1e-14, "logistic(0)");
    assert_close(vals[7], sigmoid(8.0), 1e-14, "logistic(8)");
}

// ======================== Identity: logistic(x) computed vs. formula ========================

#[test]
fn oracle_logistic_identity() {
    for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = 1.0 / (1.0 + (-x).exp());
        assert_close(
            val,
            expected,
            1e-14,
            &format!("logistic({}) = 1/(1+e^-{})", x, x),
        );
    }
}

// ======================== Monotonicity ========================

#[test]
fn oracle_logistic_monotonic_increasing() {
    let inputs = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "logistic should be monotonically increasing: {} should be > {}",
            vals[i],
            vals[i - 1]
        );
    }
}

// ======================== Very Small/Large Values ========================

#[test]
fn oracle_logistic_very_small() {
    // logistic(1e-10) ≈ 0.5 + small (detectable difference from 0.5)
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(
        (0.5..0.6).contains(&val),
        "logistic(1e-10) should be >= 0.5"
    );
}

#[test]
fn oracle_logistic_very_large_positive() {
    // logistic(1000) = 1 (saturates)
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "logistic(1000) = 1");
}

#[test]
fn oracle_logistic_very_large_negative() {
    // logistic(-1000) = 0 (saturates)
    let input = make_f64_tensor(&[], vec![-1000.0]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "logistic(-1000) = 0");
}

// ======================== METAMORPHIC: logistic(-x) = 1 - logistic(x) ========================

#[test]
fn metamorphic_logistic_symmetry() {
    // logistic(Neg(x)) + logistic(x) = 1 using primitives
    for x in [0.5, 1.0, 2.0, 3.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);

        // logistic(x)
        let logistic_x = eval_primitive(
            Primitive::Logistic,
            std::slice::from_ref(&input),
            &no_params(),
        )
        .unwrap();

        // logistic(Neg(x))
        let neg_x = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let logistic_neg_x = eval_primitive(Primitive::Logistic, &[neg_x], &no_params()).unwrap();

        // Add(logistic(x), logistic(-x)) should equal 1
        let sum =
            eval_primitive(Primitive::Add, &[logistic_x, logistic_neg_x], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sum),
            1.0,
            1e-14,
            &format!("logistic({}) + logistic(-{}) = 1", x, x),
        );
    }
}

// ======================== METAMORPHIC: logistic(x) = Reciprocal(1 + exp(-x)) ========================

#[test]
fn metamorphic_logistic_definition() {
    // logistic(x) = Reciprocal(Add(1, Exp(Neg(x))))
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let one = make_f64_tensor(&[], vec![1.0]);

        // logistic(x) directly
        let logistic_x = eval_primitive(
            Primitive::Logistic,
            std::slice::from_ref(&input),
            &no_params(),
        )
        .unwrap();

        // Reciprocal(Add(1, Exp(Neg(x))))
        let neg_x = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let exp_neg_x = eval_primitive(Primitive::Exp, &[neg_x], &no_params()).unwrap();
        let one_plus_exp = eval_primitive(Primitive::Add, &[one, exp_neg_x], &no_params()).unwrap();
        let recip = eval_primitive(Primitive::Reciprocal, &[one_plus_exp], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&logistic_x),
            extract_f64_scalar(&recip),
            1e-14,
            &format!("logistic({}) = 1/(1+exp(-{}))", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor logistic symmetry ========================

#[test]
fn metamorphic_logistic_tensor_symmetry() {
    // For tensor: logistic(x) + logistic(-x) = 1
    let data = vec![0.5, 1.0, 2.0, 3.0, 5.0];
    let input = make_f64_tensor(&[5], data);

    let logistic_x = eval_primitive(
        Primitive::Logistic,
        std::slice::from_ref(&input),
        &no_params(),
    )
    .unwrap();
    let neg_x = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
    let logistic_neg_x = eval_primitive(Primitive::Logistic, &[neg_x], &no_params()).unwrap();
    let sum = eval_primitive(Primitive::Add, &[logistic_x, logistic_neg_x], &no_params()).unwrap();

    assert_eq!(extract_shape(&sum), vec![5]);
    let sum_vec = extract_f64_vec(&sum);
    for (i, &s) in sum_vec.iter().enumerate() {
        assert_close(s, 1.0, 1e-14, &format!("element {}", i));
    }
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_logistic_preserves_all_float_dtypes() {
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
        let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "logistic {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== COMPLEX64/COMPLEX128 TESTS ========================

fn make_complex64_scalar(re: f32, im: f32) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![] },
            vec![Literal::from_complex64(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex128_scalar(re: f64, im: f64) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![] },
            vec![Literal::from_complex128(re, im)],
        )
        .unwrap(),
    )
}

fn make_complex64_tensor(shape: &[u32], pairs: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            pairs
                .into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex64().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_complex128().unwrap()
        }
        _ => unreachable!("expected tensor"),
    }
}

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

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

fn assert_complex128_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff_re = (actual.0 - expected.0).abs();
    let diff_im = (actual.1 - expected.1).abs();
    assert!(
        diff_re < tol && diff_im < tol,
        "{}: expected ({}, {}), got ({}, {}), diff=({}, {})",
        msg,
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff_re,
        diff_im
    );
}

#[test]
fn oracle_logistic_complex64_zero() {
    // logistic(0) = 1/(1+e^0) = 1/2 = 0.5
    let input = make_complex64_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert_complex64_close((re, im), (0.5, 0.0), 1e-6, "logistic(0) = 0.5");
}

#[test]
fn oracle_logistic_complex64_real_positive() {
    // For large positive x, logistic(x) -> 1
    let input = make_complex64_scalar(5.0, 0.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    // logistic(5) ≈ 0.9933
    assert!(re > 0.99, "logistic(5) should be close to 1");
    assert!(im.abs() < 1e-5, "imaginary part should be near zero");
}

#[test]
fn oracle_logistic_complex64_real_negative() {
    // For large negative x, logistic(x) -> 0
    let input = make_complex64_scalar(-5.0, 0.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    // logistic(-5) ≈ 0.0067
    assert!(re < 0.01, "logistic(-5) should be close to 0");
    assert!(im.abs() < 1e-5, "imaginary part should be near zero");
}

#[test]
fn oracle_logistic_complex64_pure_imaginary() {
    // logistic(i*pi/2) = 1/(1+e^(-i*pi/2)) = 1/(1-i)
    // 1/(1-i) = (1+i)/2 = 0.5 + 0.5i
    let pi_half = std::f32::consts::FRAC_PI_2;
    let input = make_complex64_scalar(0.0, pi_half);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    // Expected: 1/(1+e^(-i*pi/2)) = 1/(1+cos(-pi/2)+i*sin(-pi/2)) = 1/(1-i) = (1+i)/2
    assert_complex64_close((re, im), (0.5, 0.5), 1e-4, "logistic(i*pi/2)");
}

#[test]
fn oracle_logistic_complex64_symmetry() {
    // logistic(z) + logistic(-z) = 1
    let z = make_complex64_scalar(0.5, 0.3);
    let neg_z = make_complex64_scalar(-0.5, -0.3);
    let log_z = eval_primitive(Primitive::Logistic, &[z], &no_params()).unwrap();
    let log_neg_z = eval_primitive(Primitive::Logistic, &[neg_z], &no_params()).unwrap();

    let (re1, im1) = extract_complex64_scalar(&log_z);
    let (re2, im2) = extract_complex64_scalar(&log_neg_z);

    // Sum should be 1 + 0i
    assert_complex64_close(
        (re1 + re2, im1 + im2),
        (1.0, 0.0),
        1e-4,
        "logistic(z) + logistic(-z) = 1",
    );
}

#[test]
fn oracle_logistic_complex64_vector() {
    let input = make_complex64_tensor(&[3], vec![(0.0, 0.0), (5.0, 0.0), (-5.0, 0.0)]);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);

    // logistic(0) = 0.5
    assert_complex64_close(vals[0], (0.5, 0.0), 1e-5, "logistic(0)");
    // logistic(5) ≈ 0.9933
    assert!(vals[1].0 > 0.99 && vals[1].1.abs() < 1e-5, "logistic(5)");
    // logistic(-5) ≈ 0.0067
    assert!(vals[2].0 < 0.01 && vals[2].1.abs() < 1e-5, "logistic(-5)");
}

#[test]
fn oracle_logistic_complex128_zero() {
    // logistic(0) = 0.5 with higher precision
    let input = make_complex128_scalar(0.0, 0.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert_complex128_close((re, im), (0.5, 0.0), 1e-12, "logistic(0) Complex128");
}

#[test]
fn oracle_logistic_complex64_preserves_dtype() {
    let input = make_complex64_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_logistic_complex128_preserves_dtype() {
    let input = make_complex128_scalar(1.0, 1.0);
    let result = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}
