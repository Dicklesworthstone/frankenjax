//! Oracle tests for the DotGeneral primitive.
//!
//! jax.lax.dot_general is a generalized dot product supporting:
//! - Contracted dimensions (summed over)
//! - Batch dimensions (preserved in output)
//! - Remaining dimensions form outer product

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn vector_f64(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn matrix_f64(rows: u32, cols: u32, values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![rows, cols],
            },
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn tensor_f64(shape: Vec<u32>, values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: shape },
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn params(
    lhs_contracting: &str,
    rhs_contracting: &str,
    lhs_batch: &str,
    rhs_batch: &str,
) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("lhs_contracting_dims".to_string(), lhs_contracting.to_string());
    p.insert("rhs_contracting_dims".to_string(), rhs_contracting.to_string());
    p.insert("lhs_batch_dims".to_string(), lhs_batch.to_string());
    p.insert("rhs_batch_dims".to_string(), rhs_batch.to_string());
    p
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Scalar(l) => l.as_f64().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_f64().unwrap(),
        _ => panic!("expected scalar or 0-d tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_shape(v: &Value) -> Vec<u32> {
    v.as_tensor().expect("expected tensor").shape.dims.clone()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < tol,
            "mismatch: {a} vs {e}, diff {}",
            (a - e).abs()
        );
    }
}

// ======================== Vector-Vector Contraction ========================

#[test]
fn dot_general_vector_dot_product() {
    // jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())))
    // = sum(a * b) = 1*4 + 2*5 + 3*6 = 32
    let a = vector_f64(&[1.0, 2.0, 3.0]);
    let b = vector_f64(&[4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
    let val = extract_f64_scalar(&result);
    assert!((val - 32.0).abs() < 1e-12, "dot product should be 32.0");
}

// ======================== Matrix Multiplication ========================

#[test]
fn dot_general_matmul() {
    // A (2x3) @ B (3x2) = C (2x2)
    // Contract lhs dim 1 with rhs dim 0
    let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = matrix_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    // C[0,0] = 1*1 + 2*3 + 3*5 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 64
    assert_close(&extract_f64_vec(&result), &[22.0, 28.0, 49.0, 64.0], 1e-12);
}

#[test]
fn dot_general_identity_matmul() {
    // A @ I = A
    let a = matrix_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let identity = matrix_f64(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, identity], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_close(&extract_f64_vec(&result), &[1.0, 2.0, 3.0, 4.0], 1e-12);
}

// ======================== Outer Product ========================

#[test]
fn dot_general_outer_product() {
    // No contracting dims = outer product
    // [1, 2] outer [3, 4, 5] = [[3, 4, 5], [6, 8, 10]]
    let a = vector_f64(&[1.0, 2.0]);
    let b = vector_f64(&[3.0, 4.0, 5.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("", "", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_close(
        &extract_f64_vec(&result),
        &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        1e-12,
    );
}

// ======================== Batched Matrix Multiplication ========================

#[test]
fn dot_general_batched_matmul() {
    // Batch of 2 matrix multiplications
    // A: [2, 2, 3], B: [2, 3, 1]
    // Contract lhs dim 2 with rhs dim 1, batch over dim 0
    let a = tensor_f64(
        vec![2, 2, 3],
        &[
            // Batch 0: [[1,2,3], [4,5,6]]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            // Batch 1: [[7,8,9], [10,11,12]]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let b = tensor_f64(
        vec![2, 3, 1],
        &[
            // Batch 0: [[1], [0], [0]]
            1.0, 0.0, 0.0,
            // Batch 1: [[0], [1], [0]]
            0.0, 1.0, 0.0,
        ],
    );
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("2", "1", "0", "0")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2, 1]);
    // Batch 0: A[0] @ [[1],[0],[0]] = [[1], [4]]
    // Batch 1: A[1] @ [[0],[1],[0]] = [[8], [11]]
    assert_close(&extract_f64_vec(&result), &[1.0, 4.0, 8.0, 11.0], 1e-12);
}

// ======================== Error Cases ========================

#[test]
fn dot_general_rejects_mismatched_contracting_counts() {
    let a = vector_f64(&[1.0, 2.0, 3.0]);
    let b = vector_f64(&[4.0, 5.0, 6.0]);
    let err = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "", "", ""))
        .expect_err("mismatched contracting dims should fail");

    assert!(
        err.to_string().contains("contracting")
            || err.to_string().contains("same number"),
        "unexpected error: {err}"
    );
}

#[test]
fn dot_general_rejects_mismatched_batch_counts() {
    let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = matrix_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let err = eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "0", ""))
        .expect_err("mismatched batch dims should fail");

    assert!(
        err.to_string().contains("batch") || err.to_string().contains("same number"),
        "unexpected error: {err}"
    );
}

#[test]
fn dot_general_rejects_out_of_range_dim() {
    let a = vector_f64(&[1.0, 2.0, 3.0]);
    let b = vector_f64(&[4.0, 5.0, 6.0]);
    let err = eval_primitive(Primitive::DotGeneral, &[a, b], &params("5", "0", "", ""))
        .expect_err("out-of-range dim should fail");

    assert!(
        err.to_string().contains("out of range") || err.to_string().contains("rank"),
        "unexpected error: {err}"
    );
}

#[test]
fn dot_general_rejects_scalar_inputs() {
    let err = eval_primitive(
        Primitive::DotGeneral,
        &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
        &params("", "", "", ""),
    )
    .expect_err("scalar inputs should fail");

    assert!(
        err.to_string().contains("tensor") || err.to_string().contains("requires"),
        "unexpected error: {err}"
    );
}

// ======================== Shape Preservation ========================

#[test]
fn dot_general_preserves_remaining_dims() {
    // A: [2, 3, 4], B: [4, 5]
    // Contract A dim 2 with B dim 0 => [2, 3, 5]
    let a = tensor_f64(vec![2, 3, 4], &(0..24).map(|x| x as f64).collect::<Vec<_>>());
    let b = tensor_f64(vec![4, 5], &(0..20).map(|x| x as f64).collect::<Vec<_>>());
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("2", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3, 5]);
}

#[test]
fn dot_general_multiple_contracting_dims() {
    // Contract over two dimensions simultaneously
    // A: [2, 3], B: [2, 3] => scalar (contract dims 0,1)
    let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("0,1", "0,1", "", "")).unwrap();

    // Sum of squares: 1+4+9+16+25+36 = 91
    let val = extract_f64_scalar(&result);
    assert!((val - 91.0).abs() < 1e-12, "expected 91.0, got {val}");
}

#[test]
fn dot_general_preserves_dtype() {
    let a = vector_f64(&[1.0, 2.0]);
    let b = vector_f64(&[3.0, 4.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
    assert_eq!(result.dtype(), DType::F64);
}

#[test]
fn dot_general_transpose_matmul() {
    // A @ B^T via contracting different dims
    // A: [2, 3], B: [2, 3] => contract A dim 1 with B dim 1 => [2, 2]
    let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = matrix_f64(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "1", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    // A @ B^T where B^T = [[1, 0], [0, 1], [0, 0]]
    // Row 0 of A [1, 2, 3] dot cols of B^T: [1, 2]
    // Row 1 of A [4, 5, 6] dot cols of B^T: [4, 5]
    assert_close(&extract_f64_vec(&result), &[1.0, 2.0, 4.0, 5.0], 1e-12);
}

#[test]
fn dot_general_single_element_tensors() {
    let a = tensor_f64(vec![1, 1], &[3.0]);
    let b = tensor_f64(vec![1, 1], &[4.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_close(&extract_f64_vec(&result), &[12.0], 1e-12);
}

// ======================== Additional Coverage ========================

#[test]
fn dot_general_vector_matrix_product() {
    // Vector [1, 3] @ Matrix [3, 2] => [1, 2]
    let a = vector_f64(&[1.0, 2.0, 3.0]);
    let b = matrix_f64(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    // [1,2,3] . col0 = 1*1 + 2*3 + 3*5 = 22
    // [1,2,3] . col1 = 1*2 + 2*4 + 3*6 = 28
    assert_close(&extract_f64_vec(&result), &[22.0, 28.0], 1e-12);
}

#[test]
fn dot_general_matrix_vector_product() {
    // Matrix [2, 3] @ Vector [3] => [2]
    let a = matrix_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = vector_f64(&[1.0, 0.0, 1.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    // Row 0: 1*1 + 2*0 + 3*1 = 4
    // Row 1: 4*1 + 5*0 + 6*1 = 10
    assert_close(&extract_f64_vec(&result), &[4.0, 10.0], 1e-12);
}

#[test]
fn dot_general_empty_contracting_dim() {
    // Contract over empty dimension - should give zeros
    let a = tensor_f64(vec![2, 0], &[]);
    let b = tensor_f64(vec![0, 3], &[]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    // All zeros since sum over empty dim is 0
    assert_close(&extract_f64_vec(&result), &[0.0; 6], 1e-12);
}

#[test]
fn dot_general_negative_values() {
    let a = matrix_f64(2, 2, &[-1.0, 2.0, -3.0, 4.0]);
    let b = matrix_f64(2, 2, &[1.0, -1.0, -2.0, 2.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    // C[0,0] = -1*1 + 2*(-2) = -5
    // C[0,1] = -1*(-1) + 2*2 = 5
    // C[1,0] = -3*1 + 4*(-2) = -11
    // C[1,1] = -3*(-1) + 4*2 = 11
    assert_close(&extract_f64_vec(&result), &[-5.0, 5.0, -11.0, 11.0], 1e-12);
}

#[test]
fn dot_general_4d_batched() {
    // 4D batched matmul: [2, 2, 2, 3] @ [2, 2, 3, 1] => [2, 2, 2, 1]
    let a = tensor_f64(vec![2, 2, 2, 3], &(0..24).map(|x| x as f64).collect::<Vec<_>>());
    let b = tensor_f64(vec![2, 2, 3, 1], &(0..12).map(|x| x as f64).collect::<Vec<_>>());
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("3", "2", "0,1", "0,1")).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 1]);
}

#[test]
fn dot_general_large_matrices() {
    let a = tensor_f64(vec![10, 20], &(0..200).map(|x| x as f64).collect::<Vec<_>>());
    let b = tensor_f64(vec![20, 15], &(0..300).map(|x| x as f64).collect::<Vec<_>>());
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();
    assert_eq!(extract_shape(&result), vec![10, 15]);
}

fn tensor_i64(shape: Vec<u32>, values: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: shape },
            values.iter().copied().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

#[test]
fn dot_general_i64_dtype() {
    let a = tensor_i64(vec![2, 3], &[1, 2, 3, 4, 5, 6]);
    let b = tensor_i64(vec![3, 2], &[1, 2, 3, 4, 5, 6]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();
    assert_eq!(result.dtype(), DType::I64);
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn dot_general_row_vector_col_vector() {
    // [1, 3] @ [3, 1] => [1, 1] scalar in matrix form
    let a = tensor_f64(vec![1, 3], &[1.0, 2.0, 3.0]);
    let b = tensor_f64(vec![3, 1], &[4.0, 5.0, 6.0]);
    let result =
        eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    // 1*4 + 2*5 + 3*6 = 32
    assert_close(&extract_f64_vec(&result), &[32.0], 1e-12);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_dot_general_preserves_all_float_dtypes() {
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
    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_vec(dtype, &values);
        let b = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
        assert_eq!(result.dtype(), dtype, "dot_general {dtype:?}: dtype mismatch");
        match &result {
            Value::Tensor(t) => t.validate_dtype_consistency().expect("literal/dtype consistency"),
            Value::Scalar(_) => {}
        }
    }
}

// ======================== Complex Type Tests ========================

fn vector_complex64(data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape::vector(data.len() as u32),
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn vector_complex128(data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape::vector(data.len() as u32),
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn matrix_complex64(rows: u32, cols: u32, data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![rows, cols] },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_scalar(v: &Value) -> (f32, f32) {
    match v {
        Value::Scalar(l) => l.as_complex64().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_complex64().unwrap(),
        _ => panic!("expected scalar or 0-d tensor"),
    }
}

fn extract_complex128_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Scalar(l) => l.as_complex128().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_complex128().unwrap(),
        _ => panic!("expected scalar or 0-d tensor"),
    }
}

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect()
}

#[test]
fn oracle_dot_general_complex64_vec_dot() {
    // [1+i, 2+2i] . [1-i, 1-i] = (1+i)(1-i) + (2+2i)(1-i)
    // = (1 - i^2) + (2 - 2i + 2i - 2i^2) = 2 + (2 + 2) = 2 + 4 = 6
    let a = vector_complex64(vec![(1.0, 1.0), (2.0, 2.0)]);
    let b = vector_complex64(vec![(1.0, -1.0), (1.0, -1.0)]);
    let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!((re - 6.0).abs() < 1e-5, "expected 6, got {re}");
    assert!(im.abs() < 1e-5, "expected 0, got {im}");
}

#[test]
fn oracle_dot_general_complex64_purely_imaginary() {
    // [i] . [i] = i * i = -1
    let a = vector_complex64(vec![(0.0, 1.0)]);
    let b = vector_complex64(vec![(0.0, 1.0)]);
    let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
    let (re, im) = extract_complex64_scalar(&result);
    assert!((re - (-1.0)).abs() < 1e-5, "expected -1, got {re}");
    assert!(im.abs() < 1e-5, "expected 0, got {im}");
}

#[test]
fn oracle_dot_general_complex128_vec_dot() {
    // [1+0i, 2+0i] . [3+0i, 4+0i] = 3 + 8 = 11
    let a = vector_complex128(vec![(1.0, 0.0), (2.0, 0.0)]);
    let b = vector_complex128(vec![(3.0, 0.0), (4.0, 0.0)]);
    let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", "")).unwrap();
    let (re, im) = extract_complex128_scalar(&result);
    assert!((re - 11.0).abs() < 1e-10, "expected 11, got {re}");
    assert!(im.abs() < 1e-10, "expected 0, got {im}");
}

#[test]
fn oracle_dot_general_complex64_matmul() {
    // [[1+i]] * [[2+0i]] = [[2+2i]]
    let a = matrix_complex64(1, 1, vec![(1.0, 1.0)]);
    let b = matrix_complex64(1, 1, vec![(2.0, 0.0)]);
    let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    let vals = extract_complex64_vec(&result);
    assert!((vals[0].0 - 2.0).abs() < 1e-5);
    assert!((vals[0].1 - 2.0).abs() < 1e-5);
}

#[test]
fn oracle_dot_general_complex64_2x2_matmul() {
    // [[1, i], [0, 1]] * [[1, 0], [i, 1]]
    // Row 0: (1*1 + i*i, 1*0 + i*1) = (1-1, i) = (0, i)
    // Row 1: (0*1 + 1*i, 0*0 + 1*1) = (i, 1)
    let a = matrix_complex64(2, 2, vec![
        (1.0, 0.0), (0.0, 1.0),
        (0.0, 0.0), (1.0, 0.0),
    ]);
    let b = matrix_complex64(2, 2, vec![
        (1.0, 0.0), (0.0, 0.0),
        (0.0, 1.0), (1.0, 0.0),
    ]);
    let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("1", "0", "", "")).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_complex64_vec(&result);
    // Result should be [[0, i], [i, 1]]
    assert!(vals[0].0.abs() < 1e-5, "expected 0, got {}", vals[0].0);
    assert!(vals[0].1.abs() < 1e-5, "expected 0, got {}", vals[0].1);
    assert!(vals[1].0.abs() < 1e-5, "expected 0, got {}", vals[1].0);
    assert!((vals[1].1 - 1.0).abs() < 1e-5, "expected 1, got {}", vals[1].1);
    assert!(vals[2].0.abs() < 1e-5, "expected 0, got {}", vals[2].0);
    assert!((vals[2].1 - 1.0).abs() < 1e-5, "expected 1, got {}", vals[2].1);
    assert!((vals[3].0 - 1.0).abs() < 1e-5, "expected 1, got {}", vals[3].0);
    assert!(vals[3].1.abs() < 1e-5, "expected 0, got {}", vals[3].1);
}

#[test]
fn property_dot_general_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (a, b) = match dtype {
            DType::Complex64 => (
                vector_complex64(vec![(1.0, 0.0), (2.0, 0.0)]),
                vector_complex64(vec![(3.0, 0.0), (4.0, 0.0)]),
            ),
            DType::Complex128 => (
                vector_complex128(vec![(1.0, 0.0), (2.0, 0.0)]),
                vector_complex128(vec![(3.0, 0.0), (4.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(Primitive::DotGeneral, &[a, b], &params("0", "0", "", ""))
            .expect("dot_general should succeed for complex dtype");
        assert_eq!(result.dtype(), dtype, "dot_general {dtype:?}: dtype mismatch");
    }
}
