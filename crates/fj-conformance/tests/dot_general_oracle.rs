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
