//! Oracle tests for Cumsum and Cumprod primitives.
//!
//! Tests against expected behavior matching JAX/NumPy:
//! - jnp.cumsum: cumulative sum along axis
//! - jnp.cumprod: cumulative product along axis

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

fn reverse_axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = axis_params(axis);
    p.insert("reverse".to_owned(), "true".to_owned());
    p
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
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect()
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_i64_scalar(v: &Value) -> Option<i64> {
    match v {
        Value::Scalar(l) => l.as_i64(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements.first()?.as_i64(),
        _ => None,
    }
}

// ======================== Cumsum Oracle Tests ========================

#[test]
fn oracle_cumsum_1d_i64() {
    // JAX: jnp.cumsum(jnp.array([1, 2, 3, 4, 5])) => [1, 3, 6, 10, 15]
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 6, 10, 15]);
}

#[test]
fn cumsum_last_element_equals_reduce_sum() {
    // Cross-validate Cumsum against ReduceSum: the final prefix sum must equal the
    // total. i64 so summation order is irrelevant (exact, no FP reassociation).
    // Oracle-free. (Cumsum reads the "axis" param; ReduceSum reads "axes".)
    let data = vec![1_i64, 2, 3, 4, 5, 6];
    let input = make_i64_tensor(&[data.len() as u32], data.clone());
    let cum = extract_i64_vec(
        &eval_primitive(Primitive::Cumsum, &[input.clone()], &axis_params(0)).unwrap(),
    );
    let reduce_axes = BTreeMap::from([("axes".to_string(), "0".to_string())]);
    let total =
        extract_i64_scalar(&eval_primitive(Primitive::ReduceSum, &[input], &reduce_axes).unwrap())
            .expect("reduce_sum should produce an i64 scalar");
    assert_eq!(
        *cum.last().unwrap(),
        total,
        "cumsum's last element must equal reduce_sum"
    );
    assert_eq!(
        total,
        data.iter().sum::<i64>(),
        "and both equal the host sum"
    );
}

#[test]
fn oracle_cumsum_1d_f64() {
    // JAX: jnp.cumsum(jnp.array([1.0, 2.0, 3.0])) => [1.0, 3.0, 6.0]
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_cumsum_2d_last_axis() {
    // JAX: jnp.cumsum(jnp.array([[1,2,3],[4,5,6]]), axis=-1)
    // => [[1, 3, 6], [4, 9, 15]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn oracle_cumsum_2d_first_axis() {
    // JAX: jnp.cumsum(jnp.array([[1,2,3],[4,5,6]]), axis=0)
    // => [[1, 2, 3], [5, 7, 9]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 5, 7, 9]);
}

#[test]
fn oracle_cumsum_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_cumsum_with_negatives() {
    // JAX: jnp.cumsum(jnp.array([1, -2, 3, -4])) => [1, -1, 2, -2]
    let input = make_i64_tensor(&[4], vec![1, -2, 3, -4]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, -1, 2, -2]);
}

#[test]
fn oracle_cumsum_reverse_1d_i64() {
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10, 9, 7, 4]);
}

#[test]
fn oracle_cumsum_zeros() {
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

// ======================== Cumprod Oracle Tests ========================

#[test]
fn oracle_cumprod_1d_i64() {
    // JAX: jnp.cumprod(jnp.array([1, 2, 3, 4])) => [1, 2, 6, 24]
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 6, 24]);
}

#[test]
fn cumprod_last_element_equals_reduce_prod() {
    // Cross-validate Cumprod against ReduceProd: cumprod(x)[-1] == reduce_prod(x).
    // Small i64 values so the product is exact and does not overflow. Oracle-free.
    // (Cumprod reads the "axis" param; ReduceProd reads "axes".)
    let data = vec![1_i64, 2, 3, 4, 5];
    let input = make_i64_tensor(&[data.len() as u32], data.clone());
    let cum = extract_i64_vec(
        &eval_primitive(Primitive::Cumprod, &[input.clone()], &axis_params(0)).unwrap(),
    );
    let reduce_axes = BTreeMap::from([("axes".to_string(), "0".to_string())]);
    let total =
        extract_i64_scalar(&eval_primitive(Primitive::ReduceProd, &[input], &reduce_axes).unwrap())
            .expect("reduce_prod should produce an i64 scalar");
    assert_eq!(
        *cum.last().unwrap(),
        total,
        "cumprod's last element must equal reduce_prod"
    );
    assert_eq!(
        total,
        data.iter().product::<i64>(),
        "and both equal the host product"
    );
}

#[test]
fn oracle_cumprod_1d_f64() {
    // JAX: jnp.cumprod(jnp.array([1.0, 2.0, 3.0])) => [1.0, 2.0, 6.0]
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 6.0).abs() < 1e-10);
}

#[test]
fn oracle_cumprod_2d_last_axis() {
    // JAX: jnp.cumprod(jnp.array([[1,2,3],[4,5,6]]), axis=-1)
    // => [[1, 2, 6], [4, 20, 120]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 6, 4, 20, 120]);
}

#[test]
fn oracle_cumprod_2d_first_axis() {
    // JAX: jnp.cumprod(jnp.array([[1,2,3],[4,5,6]]), axis=0)
    // => [[1, 2, 3], [4, 10, 18]]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 10, 18]);
}

#[test]
fn oracle_cumprod_single_element() {
    let input = make_i64_tensor(&[1], vec![7]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![7]);
}

#[test]
fn oracle_cumprod_with_zero() {
    // JAX: jnp.cumprod(jnp.array([2, 3, 0, 4])) => [2, 6, 0, 0]
    let input = make_i64_tensor(&[4], vec![2, 3, 0, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![2, 6, 0, 0]);
}

#[test]
fn oracle_cumprod_with_negatives() {
    // JAX: jnp.cumprod(jnp.array([1, -2, 3, -4])) => [1, -2, -6, 24]
    let input = make_i64_tensor(&[4], vec![1, -2, 3, -4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, -2, -6, 24]);
}

#[test]
fn oracle_cumprod_all_ones() {
    let input = make_i64_tensor(&[5], vec![1, 1, 1, 1, 1]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1, 1]);
}

#[test]
fn oracle_cumprod_reverse_2d_last_axis() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &reverse_axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![6, 6, 3, 120, 30, 6]);
}

// ======================== Cummax Oracle Tests ========================

#[test]
fn oracle_cummax_1d_i64_with_duplicates() {
    let input = make_i64_tensor(&[6], vec![1, 3, 2, 4, 4, 0]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 3, 4, 4, 4]);
}

#[test]
fn oracle_cummax_2d_first_axis() {
    let input = make_i64_tensor(&[2, 3], vec![1, 5, 2, 4, 3, 6]);
    let result = eval_primitive(Primitive::Cummax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 5, 2, 4, 5, 6]);
}

#[test]
fn oracle_cummax_reverse_1d_i64() {
    let input = make_i64_tensor(&[4], vec![1, 4, 2, 3]);
    let result = eval_primitive(Primitive::Cummax, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4, 4, 3, 3]);
}

// ======================== Cummin Oracle Tests ========================

#[test]
fn oracle_cummin_1d_i64_with_duplicates() {
    let input = make_i64_tensor(&[5], vec![4, 2, 3, 2, 1]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![4, 2, 2, 2, 1]);
}

#[test]
fn oracle_cummin_2d_last_axis() {
    let input = make_i64_tensor(&[2, 3], vec![3, 1, 2, 4, 6, 5]);
    let result = eval_primitive(Primitive::Cummin, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 1, 4, 4, 4]);
}

#[test]
fn oracle_cummin_reverse_2d_first_axis() {
    let input = make_i64_tensor(&[3, 2], vec![3, 5, 2, 7, 4, 1]);
    let result = eval_primitive(Primitive::Cummin, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![2, 1, 2, 1, 4, 1]);
}

#[test]
fn oracle_cumulative_rejects_invalid_reverse_param() {
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let mut params = axis_params(0);
    params.insert("reverse".to_owned(), "maybe".to_owned());
    let err = eval_primitive(Primitive::Cumsum, &[input], &params)
        .expect_err("invalid reverse parameter should fail");
    assert!(
        err.to_string().contains("reverse"),
        "unexpected error: {err}"
    );
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_cumsum_last_equals_sum() {
    // last(cumsum(x)) = reduce_sum(x)
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let cumsum_result = eval_primitive(
        Primitive::Cumsum,
        std::slice::from_ref(&input),
        &no_params(),
    )
    .unwrap();
    let cumsum_vals = extract_i64_vec(&cumsum_result);

    let sum_result = eval_primitive(Primitive::ReduceSum, &[input], &axis_params(0)).unwrap();

    assert_eq!(
        cumsum_vals.last().copied(),
        extract_i64_scalar(&sum_result),
        "last(cumsum(x)) should equal reduce_sum(x)"
    );
}

#[test]
fn metamorphic_cumsum_first_element_identity() {
    // cumsum(x)[0] = x[0]
    let input = make_i64_tensor(&[4], vec![7, 2, 9, 3]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 7, "cumsum(x)[0] should equal x[0]");
}

#[test]
fn metamorphic_cumprod_last_equals_product() {
    // last(cumprod(x)) = reduce_prod(x)
    let input = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let cumprod_result = eval_primitive(
        Primitive::Cumprod,
        std::slice::from_ref(&input),
        &no_params(),
    )
    .unwrap();
    let cumprod_vals = extract_i64_vec(&cumprod_result);

    let prod_result = eval_primitive(Primitive::ReduceProd, &[input], &axis_params(0)).unwrap();

    assert_eq!(
        cumprod_vals.last().copied(),
        extract_i64_scalar(&prod_result),
        "last(cumprod(x)) should equal reduce_prod(x)"
    );
}

#[test]
fn metamorphic_cumprod_first_element_identity() {
    // cumprod(x)[0] = x[0]
    let input = make_i64_tensor(&[4], vec![5, 3, 2, 4]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 5, "cumprod(x)[0] should equal x[0]");
}

// Regression tests for ff512bc — eval_cumulative previously emitted
// Literal::from_f64 for every accumulator step regardless of input
// dtype, leaving F32 cumulative outputs declaring DType::F32 while
// storing F64Bits elements.
#[test]
fn oracle_cumsum_f32_preserves_dtype() {
    let data: Vec<Literal> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(Literal::from_f32)
        .collect();
    let input = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: vec![4] }, data).unwrap());
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let t = result.as_tensor().expect("expected tensor");
    assert_eq!(t.dtype, DType::F32);
    t.validate_dtype_consistency()
        .expect("F32 cumsum output dtype/element invariant");
    let last_value = t.elements.last().and_then(|literal| match literal {
        Literal::F32Bits(bits) => Some(f32::from_bits(*bits)),
        _ => None,
    });
    assert!(
        matches!(last_value, Some(v) if (v - 10.0).abs() < 1e-5),
        "expected final F32 cumsum value near 10.0, got {last_value:?}",
    );
}

#[test]
fn oracle_cumprod_f32_preserves_dtype() {
    let data: Vec<Literal> = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(Literal::from_f32)
        .collect();
    let input = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: vec![4] }, data).unwrap());
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let t = result.as_tensor().expect("expected tensor");
    assert_eq!(t.dtype, DType::F32);
    t.validate_dtype_consistency()
        .expect("F32 cumprod output dtype/element invariant");
}

// Property sweep across BF16/F16/F32/F64. The fix in ff512bc routes
// through `reduce_real_literal`/`reduce_real_output_dtype`, which apply
// per-dtype. Single-dtype point tests can mask regressions in any one
// arm.
#[test]
fn property_cumulative_preserves_all_float_dtypes() {
    fn make_vec<F>(dtype: DType, values: &[f64], lit_for: F) -> Value
    where
        F: Fn(f64) -> Literal,
    {
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

    let values = [1.0_f64, 2.0, 3.0, 4.0];
    let cases: Vec<(DType, Value)> = vec![
        (
            DType::BF16,
            make_vec(DType::BF16, &values, |v| Literal::from_bf16_f32(v as f32)),
        ),
        (
            DType::F16,
            make_vec(DType::F16, &values, |v| Literal::from_f16_f32(v as f32)),
        ),
        (
            DType::F32,
            make_vec(DType::F32, &values, |v| Literal::from_f32(v as f32)),
        ),
        (DType::F64, make_vec(DType::F64, &values, Literal::from_f64)),
    ];

    for (dtype, input) in cases {
        for primitive in [
            Primitive::Cumsum,
            Primitive::Cumprod,
            Primitive::Cummax,
            Primitive::Cummin,
        ] {
            let result = eval_primitive(primitive, std::slice::from_ref(&input), &no_params())
                .expect("cumulative primitive should succeed for float dtype");
            let t = result
                .as_tensor()
                .expect("cumulative primitive should return tensor");
            assert_eq!(
                t.dtype, dtype,
                "{primitive:?} {dtype:?}: tensor dtype mismatch"
            );
            t.validate_dtype_consistency()
                .expect("cumulative output should preserve dtype consistency");
        }
    }
}

// ======================== Complex64/Complex128 Tests ========================

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

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect()
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_complex128().unwrap())
        .collect()
}

fn extract_shape(v: &Value) -> Vec<u32> {
    v.as_tensor().expect("expected tensor").shape.dims.clone()
}

#[test]

fn oracle_cumsum_complex64_1d() {
    // cumsum([1+i, 2+2i, 3+3i]) = [1+i, 3+3i, 6+6i]
    let input = make_complex64_tensor(&[3], vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 1.0), (3.0, 3.0), (6.0, 6.0)]);
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]

fn oracle_cumsum_complex64_2d_axis0() {
    // cumsum along axis 0
    let input = make_complex64_tensor(
        &[2, 3],
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ],
    );
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_complex64_vec(&result);
    // Row 0 stays same, row 1 = row 0 + row 1
    assert_eq!(
        vals,
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (5.0, 0.0),
            (7.0, 0.0),
            (9.0, 0.0),
        ]
    );
}

#[test]

fn oracle_cumsum_complex64_2d_axis1() {
    // cumsum along axis 1 (within each row)
    let input = make_complex64_tensor(
        &[2, 3],
        vec![
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
            (5.0, 5.0),
            (6.0, 6.0),
        ],
    );
    let result = eval_primitive(Primitive::Cumsum, &[input], &axis_params(1)).unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 1.0),
            (3.0, 3.0),
            (6.0, 6.0),
            (4.0, 4.0),
            (9.0, 9.0),
            (15.0, 15.0),
        ]
    );
}

#[test]

fn oracle_cumprod_complex64_1d() {
    // cumprod([1+0i, 2+0i, 3+0i]) = [1, 2, 6] for real parts
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(1.0, 0.0), (2.0, 0.0), (6.0, 0.0)]);
}

#[test]

fn oracle_cumprod_complex64_with_imaginary() {
    // cumprod([i, i]) = [i, i*i] = [i, -1]
    let input = make_complex64_tensor(&[2], vec![(0.0, 1.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 0.0).abs() < 1e-6); // 0 + i
    assert!((vals[0].1 - 1.0).abs() < 1e-6);
    assert!((vals[1].0 - (-1.0)).abs() < 1e-6); // -1 + 0i
    assert!((vals[1].1 - 0.0).abs() < 1e-6);
}

#[test]

fn oracle_cumsum_complex128_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_complex128_vec(&result);
    assert_eq!(vals, vec![(1.0, -1.0), (3.0, -3.0), (6.0, -6.0)]);
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]

fn oracle_cumprod_complex128_1d() {
    let input = make_complex128_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    let vals = extract_complex128_vec(&result);
    assert_eq!(vals, vec![(1.0, 0.0), (2.0, 0.0), (6.0, 0.0)]);
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]

fn oracle_cumsum_complex64_single_element() {
    let input = make_complex64_tensor(&[1], vec![(42.0, -42.0)]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(42.0, -42.0)]);
}

#[test]

fn oracle_cumsum_complex64_preserves_dtype() {
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]

fn oracle_cumprod_complex64_preserves_dtype() {
    let input = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let result = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]

fn property_cumulative_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => {
                make_complex64_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            }
            DType::Complex128 => {
                make_complex128_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            }
            _ => unreachable!(),
        };

        for primitive in [Primitive::Cumsum, Primitive::Cumprod] {
            let result = eval_primitive(primitive, std::slice::from_ref(&input), &no_params())
                .expect("cumulative primitive should succeed for complex dtype");
            let t = result.as_tensor().expect("should return tensor");
            assert_eq!(t.dtype, dtype, "{primitive:?} {dtype:?}: dtype mismatch");
            t.validate_dtype_consistency()
                .expect("cumulative output should preserve dtype consistency");
        }
    }
}

#[test]
fn oracle_associative_scan_body_ops() {
    // associative_scan (jax.lax.associative_scan) produces the cumulative result along
    // axis 0 via the body_op reducer; it had no conformance value oracle. Covers the
    // arithmetic (add/mul/max/min) and bitwise (and/or/xor) reducers.
    let scan = |data: Vec<i64>, op: &str| -> Vec<i64> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_string(), op.to_string());
        let r = eval_primitive(
            Primitive::AssociativeScan,
            &[make_i64_tensor(&[data.len() as u32], data)],
            &p,
        )
        .unwrap_or_else(|e| panic!("associative_scan body_op={op} failed: {e:?}"));
        extract_i64_vec(&r)
    };
    assert_eq!(
        scan(vec![1, 2, 3, 4], "add"),
        vec![1, 3, 6, 10],
        "add = cumulative sum"
    );
    assert_eq!(
        scan(vec![1, 2, 3, 4], "mul"),
        vec![1, 2, 6, 24],
        "mul = cumulative product"
    );
    assert_eq!(
        scan(vec![1, 3, 2, 4], "max"),
        vec![1, 3, 3, 4],
        "max = cumulative max"
    );
    assert_eq!(
        scan(vec![4, 2, 3, 1], "min"),
        vec![4, 2, 2, 1],
        "min = cumulative min"
    );
    // bitwise reducers (i64): accumulate left-to-right.
    assert_eq!(
        scan(vec![7, 3, 6], "and"),
        vec![7, 3, 2],
        "and: 7, 7&3=3, 3&6=2"
    );
    assert_eq!(
        scan(vec![1, 2, 4], "or"),
        vec![1, 3, 7],
        "or: 1, 1|2=3, 3|4=7"
    );
    assert_eq!(
        scan(vec![1, 3, 5], "xor"),
        vec![1, 2, 7],
        "xor: 1, 1^3=2, 2^5=7"
    );
}
