//! Oracle tests for Nextafter primitive.
//!
//! Tests against expected behavior matching C's nextafter:
//! - Returns the next representable floating-point value after x towards y
//! - If x == y, returns y
//! - If x or y is NaN, returns NaN

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

fn make_f32_tensor(shape: &[u32], data: Vec<f32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f32).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_f32_vec(v: &Value) -> Result<Vec<f32>, String> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => Ok(f32::from_bits(*bits)),
                other => Err(format!("expected F32Bits, got {other:?}")),
            })
            .collect(),
        Value::Scalar(Literal::F32Bits(bits)) => Ok(vec![f32::from_bits(*bits)]),
        other => Err(format!("expected F32 tensor or scalar, got {other:?}")),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn nextafter_f64_scalar(x: f64, y: f64) -> Result<f64, String> {
    let result = eval_primitive(
        Primitive::Nextafter,
        &[
            Value::Scalar(Literal::from_f64(x)),
            Value::Scalar(Literal::from_f64(y)),
        ],
        &no_params(),
    )
    .map_err(|err| format!("{err:?}"))?;
    match &result {
        Value::Scalar(lit) => lit
            .as_f64()
            .ok_or_else(|| format!("expected f64 scalar, got {result:?}")),
        Value::Tensor(t) if t.elements.len() == 1 => t
            .elements
            .first()
            .and_then(|lit| lit.as_f64())
            .ok_or_else(|| format!("expected single f64 tensor element, got {result:?}")),
        other => Err(format!("expected scalar nextafter result, got {other:?}")),
    }
}

fn assert_f32_tensor_dtype(value: &Value) {
    assert!(
        matches!(value, Value::Tensor(t) if t.dtype == DType::F32),
        "expected F32 tensor, got {value:?}"
    );
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_nextafter_same() {
    // nextafter(x, x) = x
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
}

#[test]
fn oracle_nextafter_towards_positive() {
    // nextafter(1.0, 2.0) > 1.0
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0);
    assert!(vals[0] < 1.0 + 1e-10);
}

#[test]
fn oracle_nextafter_towards_negative() {
    // nextafter(1.0, 0.0) < 1.0
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < 1.0);
    assert!(vals[0] > 1.0 - 1e-10);
}

#[test]
fn oracle_nextafter_zero_to_positive() {
    // nextafter(0.0, 1.0) > 0.0 (smallest positive subnormal)
    let a = Value::Scalar(Literal::from_f64(0.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0);
    assert!(vals[0] < 1e-300);
}

#[test]
fn oracle_nextafter_zero_to_negative() {
    // nextafter(0.0, -1.0) < 0.0 (smallest negative subnormal)
    let a = Value::Scalar(Literal::from_f64(0.0));
    let b = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < 0.0);
    assert!(vals[0] > -1e-300);
}

#[test]
fn oracle_nextafter_zero_f64_subnormal_bits() -> Result<(), String> {
    let positive = nextafter_f64_scalar(0.0, 1.0)?;
    assert_eq!(positive.to_bits(), 1, "nextafter(0.0, 1.0)");

    let negative = nextafter_f64_scalar(0.0, -1.0)?;
    assert_eq!(
        negative.to_bits(),
        1 | (1_u64 << 63),
        "nextafter(0.0, -1.0)"
    );

    Ok(())
}

#[test]
fn oracle_nextafter_nan_x() {
    // nextafter(NaN, y) = NaN
    let a = Value::Scalar(Literal::from_f64(f64::NAN));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
}

#[test]
fn oracle_nextafter_nan_y() {
    // nextafter(x, NaN) = NaN
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
}

#[test]
fn oracle_nextafter_f32_scalar_preserves_dtype() -> Result<(), String> {
    let a = Value::Scalar(Literal::from_f32(1.0));
    let b = Value::Scalar(Literal::from_f32(2.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params())
        .map_err(|err| format!("{err:?}"))?;
    let vals = extract_f32_vec(&result)?;
    assert!(vals[0] > 1.0);
    assert!(vals[0] < 1.0 + 1e-5);
    Ok(())
}

#[test]
fn oracle_nextafter_zero_f32_subnormal_bits() -> Result<(), String> {
    let positive = eval_primitive(
        Primitive::Nextafter,
        &[
            Value::Scalar(Literal::from_f32(0.0)),
            Value::Scalar(Literal::from_f32(1.0)),
        ],
        &no_params(),
    )
    .map_err(|err| format!("{err:?}"))?;
    let positive_vals = extract_f32_vec(&positive)?;
    assert_eq!(positive_vals[0].to_bits(), 1, "nextafter(0.0f32, 1.0)");

    let negative = eval_primitive(
        Primitive::Nextafter,
        &[
            Value::Scalar(Literal::from_f32(0.0)),
            Value::Scalar(Literal::from_f32(-1.0)),
        ],
        &no_params(),
    )
    .map_err(|err| format!("{err:?}"))?;
    let negative_vals = extract_f32_vec(&negative)?;
    assert_eq!(
        negative_vals[0].to_bits(),
        1 | (1_u32 << 31),
        "nextafter(0.0f32, -1.0)"
    );

    Ok(())
}

// ======================== 1D Tests ========================

#[test]
fn oracle_nextafter_1d() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0); // towards 2
    assert!(vals[1] < 2.0); // towards 1
    assert!((vals[2] - 3.0).abs() < 1e-15); // same
}

#[test]
fn oracle_nextafter_1d_zeros() {
    let a = make_f64_tensor(&[2], vec![0.0, 0.0]);
    let b = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0); // towards positive
    assert!(vals[1] < 0.0); // towards negative
}

#[test]
fn oracle_nextafter_f32_tensor_preserves_dtype() -> Result<(), String> {
    let a = make_f32_tensor(&[3], vec![1.0, 2.0, 0.0]);
    let b = make_f32_tensor(&[3], vec![2.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params())
        .map_err(|err| format!("{err:?}"))?;
    assert_eq!(extract_shape(&result), vec![3]);
    assert_f32_tensor_dtype(&result);
    let vals = extract_f32_vec(&result)?;
    assert!(vals[0] > 1.0);
    assert!(vals[1] < 2.0);
    assert!(vals[2] < 0.0);
    Ok(())
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_nextafter_f64_one_step_roundtrip_bits() -> Result<(), String> {
    let samples = [
        1.0,
        -1.0,
        42.25,
        -1024.5,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(0x0010_0000_0000_0001),
        -f64::from_bits(0x0010_0000_0000_0001),
    ];

    for x in samples {
        let up = nextafter_f64_scalar(x, f64::INFINITY)?;
        assert!(up > x, "nextafter({x}, +inf) did not move upward");
        let back_down = nextafter_f64_scalar(up, f64::NEG_INFINITY)?;
        assert_eq!(
            back_down.to_bits(),
            x.to_bits(),
            "upward one-step roundtrip failed for {x}"
        );

        let down = nextafter_f64_scalar(x, f64::NEG_INFINITY)?;
        assert!(down < x, "nextafter({x}, -inf) did not move downward");
        let back_up = nextafter_f64_scalar(down, f64::INFINITY)?;
        assert_eq!(
            back_up.to_bits(),
            x.to_bits(),
            "downward one-step roundtrip failed for {x}"
        );
    }
    Ok(())
}

#[test]
fn metamorphic_nextafter_f32_tensor_roundtrip_preserves_bits_shape_and_dtype() -> Result<(), String>
{
    let samples = vec![
        1.0_f32,
        -1.0,
        0.5,
        -0.5,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    let input = make_f32_tensor(&[2, 3], samples.clone());
    let upward_targets = make_f32_tensor(&[2, 3], vec![f32::INFINITY; samples.len()]);
    let upward = eval_primitive(Primitive::Nextafter, &[input, upward_targets], &no_params())
        .map_err(|err| format!("{err:?}"))?;
    assert_eq!(extract_shape(&upward), vec![2, 3]);
    assert_f32_tensor_dtype(&upward);

    let downward_targets = make_f32_tensor(&[2, 3], vec![f32::NEG_INFINITY; samples.len()]);
    let recovered = eval_primitive(
        Primitive::Nextafter,
        &[upward, downward_targets],
        &no_params(),
    )
    .map_err(|err| format!("{err:?}"))?;
    assert_eq!(extract_shape(&recovered), vec![2, 3]);
    assert_f32_tensor_dtype(&recovered);

    for (original, recovered) in samples.iter().zip(extract_f32_vec(&recovered)?) {
        assert_eq!(
            recovered.to_bits(),
            original.to_bits(),
            "tensor one-step roundtrip failed for {original}"
        );
    }
    Ok(())
}

// ======================== 2D Tests ========================

#[test]
fn oracle_nextafter_2d() {
    let a = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let b = make_f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 3.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0); // 0 towards 1
    assert!(vals[1] < 1.0); // 1 towards 0
    assert!(vals[2] > -1.0); // -1 towards 0
    assert!(vals[3] > 2.0); // 2 towards 3
}

// ======================== Edge Cases ========================

#[test]
fn oracle_nextafter_inf_towards_finite() {
    // nextafter(inf, 0) < inf
    let a = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let b = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < f64::INFINITY);
    assert!(vals[0] > 1e308);
}

#[test]
fn oracle_nextafter_finite_towards_inf() {
    // nextafter(MAX, inf) = inf
    let a = Value::Scalar(Literal::from_f64(f64::MAX));
    let b = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
}

#[test]
fn oracle_nextafter_negative_zero() {
    // nextafter(-0.0, 1.0) = smallest positive
    let a = Value::Scalar(Literal::from_f64(-0.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0);
}

// ======================== Broadcast Tests ========================

#[test]
fn oracle_nextafter_scalar_tensor_broadcast() {
    // Scalar x, tensor direction -> broadcasts scalar over tensor shape
    let x = Value::Scalar(Literal::from_f64(1.0));
    let y = make_f64_tensor(&[3], vec![2.0, 0.0, 1.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0, "towards 2.0");
    assert!(vals[1] < 1.0, "towards 0.0");
    assert!((vals[2] - 1.0).abs() < 1e-15, "same value");
}

#[test]
fn oracle_nextafter_tensor_scalar_broadcast() {
    // Tensor x, scalar direction -> broadcasts direction over tensor shape
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 0.0]);
    let y = Value::Scalar(Literal::from_f64(10.0));
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0, "1.0 towards 10.0");
    assert!(vals[1] > 2.0, "2.0 towards 10.0");
    assert!(vals[2] > 0.0, "0.0 towards 10.0");
}

#[test]
fn oracle_nextafter_scalar_tensor_broadcast_2d() {
    let x = Value::Scalar(Literal::from_f64(0.0));
    let y = make_f64_tensor(&[2, 2], vec![1.0, -1.0, 2.0, -2.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0, "0 towards 1");
    assert!(vals[1] < 0.0, "0 towards -1");
    assert!(vals[2] > 0.0, "0 towards 2");
    assert!(vals[3] < 0.0, "0 towards -2");
}

#[test]
fn oracle_nextafter_all_scalars_broadcast() {
    // scalar nextafter scalar -> scalar
    let x = Value::Scalar(Literal::from_f64(1.0));
    let y = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0, "1 towards 2");
}

#[test]
fn oracle_nextafter_singleton_x_vector_y_broadcast() {
    // [1] x with [3] y -> [3]
    let x = make_f64_tensor(&[1], vec![0.0]);
    let y = make_f64_tensor(&[3], vec![1.0, -1.0, 100.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0, "0 towards 1");
    assert!(vals[1] < 0.0, "0 towards -1");
    assert!(vals[2] > 0.0, "0 towards 100");
}

#[test]
fn oracle_nextafter_vector_x_singleton_y_broadcast() {
    // [3] x with [1] y -> [3]
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[1], vec![10.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0, "1 towards 10");
    assert!(vals[1] > 2.0, "2 towards 10");
    assert!(vals[2] > 3.0, "3 towards 10");
}

#[test]
fn oracle_nextafter_column_x_matrix_y_broadcast() {
    // [2, 1] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![0.0, 5.0]);
    let y = make_f64_tensor(&[2, 3], vec![1.0, -1.0, 0.0, 10.0, 0.0, 5.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: 0 towards 1 (positive), 0 towards -1 (negative), 0 towards 0 (same)
    assert!(vals[0] > 0.0, "0 towards 1");
    assert!(vals[1] < 0.0, "0 towards -1");
    assert!((vals[2] - 0.0).abs() < 1e-15, "0 towards 0");
    // Row 1: 5 towards 10 (bigger), 5 towards 0 (smaller), 5 towards 5 (same)
    assert!(vals[3] > 5.0, "5 towards 10");
    assert!(vals[4] < 5.0, "5 towards 0");
    assert!((vals[5] - 5.0).abs() < 1e-15, "5 towards 5");
}

#[test]
fn oracle_nextafter_different_ranks_broadcast() {
    // [3] x with [2, 3] y -> [2, 3]
    let x = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let y = make_f64_tensor(&[2, 3], vec![10.0, 10.0, 10.0, -10.0, -10.0, -10.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: towards 10 (all positive direction)
    assert!(vals[0] > 0.0, "0 towards 10");
    assert!(vals[1] > 1.0, "1 towards 10");
    assert!(vals[2] > -1.0, "-1 towards 10");
    // Row 1: towards -10 (all negative direction)
    assert!(vals[3] < 0.0, "0 towards -10");
    assert!(vals[4] < 1.0, "1 towards -10");
    assert!(vals[5] < -1.0, "-1 towards -10");
}

#[test]
fn oracle_nextafter_incompatible_shapes_error() {
    // [2] nextafter [3] should error
    let x = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![10.0, 10.0, 10.0]);
    let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_nextafter_preserves_f32_f64_dtypes() {
    // Note: BF16/F16 inputs are promoted to F64 for numerical precision.
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not F32 or F64"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let x_values = [0.0_f64, 1.0, -1.0];
    let y_values = [1.0_f64, 2.0, 0.0];
    for dtype in [DType::F32, DType::F64] {
        let x = make_vec(dtype, &x_values);
        let y = make_vec(dtype, &y_values);
        let result = eval_primitive(Primitive::Nextafter, &[x, y], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "nextafter {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
