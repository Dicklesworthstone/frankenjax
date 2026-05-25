//! Oracle conformance tests for jnp.einsum.
//!
//! Reference values computed with JAX:
//! ```python
//! import jax.numpy as jnp
//! jnp.einsum('ij,jk->ik', a, b)  # matrix multiply
//! jnp.einsum('i,i->', a, b)      # dot product
//! jnp.einsum('ij->ji', a)        # transpose
//! ```

use fj_lax::einsum::{EinsumError, einsum1, einsum2};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

// JAX reference: einsum('ij,jk->ik', A, B) for matrix multiply
// A (2x3) @ B (3x2) = C (2x2)
// A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
// C = [[22,28],[49,64]]
#[test]
fn test_einsum_matmul() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum2("ij,jk->ik", &a, &[2, 3], &b, &[3, 2]).unwrap();
    assert_eq!(shape, vec![2, 2]);
    let expected = [22.0, 28.0, 49.0, 64.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "matmul: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('i,i->', a, b) for dot product
// a = [1,2,3], b = [4,5,6]
// result = 1*4 + 2*5 + 3*6 = 32
#[test]
fn test_einsum_dot_product() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let (result, shape) = einsum2("i,i->", &a, &[3], &b, &[3]).unwrap();
    assert!(shape.is_empty(), "scalar result should have empty shape");
    assert!(
        approx_eq(result[0], 32.0, 1e-10),
        "dot product: got {}, expected 32.0",
        result[0]
    );
}

// JAX reference: einsum('i,j->ij', a, b) for outer product
// a = [1,2], b = [3,4,5]
// result = [[3,4,5],[6,8,10]]
#[test]
fn test_einsum_outer_product() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0, 5.0];
    let (result, shape) = einsum2("i,j->ij", &a, &[2], &b, &[3]).unwrap();
    assert_eq!(shape, vec![2, 3]);
    let expected = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "outer product: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('ij->ji', A) for transpose
// A = [[1,2,3],[4,5,6]] (2x3)
// result = [[1,4],[2,5],[3,6]] (3x2)
#[test]
fn test_einsum_transpose() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum1("ij->ji", &a, &[2, 3]).unwrap();
    assert_eq!(shape, vec![3, 2]);
    let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "transpose: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('ii->', A) for trace
// A = [[1,2],[3,4]] (2x2)
// trace = 1 + 4 = 5
#[test]
fn test_einsum_trace() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let (result, shape) = einsum1("ii->", &a, &[2, 2]).unwrap();
    assert!(shape.is_empty(), "trace should be scalar");
    assert!(
        approx_eq(result[0], 5.0, 1e-10),
        "trace: got {}, expected 5.0",
        result[0]
    );
}

// JAX reference: einsum('ij->', A) for sum
// A = [[1,2],[3,4]] (2x2)
// sum = 1 + 2 + 3 + 4 = 10
#[test]
fn test_einsum_sum_all() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let (result, shape) = einsum1("ij->", &a, &[2, 2]).unwrap();
    assert!(shape.is_empty(), "sum should be scalar");
    assert!(
        approx_eq(result[0], 10.0, 1e-10),
        "sum: got {}, expected 10.0",
        result[0]
    );
}

// JAX reference: einsum('ij->i', A) for row sum
// A = [[1,2,3],[4,5,6]] (2x3)
// result = [6, 15]
#[test]
fn test_einsum_row_sum() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum1("ij->i", &a, &[2, 3]).unwrap();
    assert_eq!(shape, vec![2]);
    let expected = [6.0, 15.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "row sum: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('ij->j', A) for column sum
// A = [[1,2,3],[4,5,6]] (2x3)
// result = [5, 7, 9]
#[test]
fn test_einsum_col_sum() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum1("ij->j", &a, &[2, 3]).unwrap();
    assert_eq!(shape, vec![3]);
    let expected = [5.0, 7.0, 9.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "col sum: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('ii->i', A) for diagonal
// A = [[1,2,3],[4,5,6],[7,8,9]] (3x3)
// result = [1, 5, 9]
#[test]
fn test_einsum_diagonal() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (result, shape) = einsum1("ii->i", &a, &[3, 3]).unwrap();
    assert_eq!(shape, vec![3]);
    let expected = [1.0, 5.0, 9.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "diagonal: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: einsum('ik,kj->ij', A, B) for batched matmul
// Same as matmul but using different index labels
#[test]
fn test_einsum_matmul_different_indices() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum2("ik,kj->ij", &a, &[2, 3], &b, &[3, 2]).unwrap();
    assert_eq!(shape, vec![2, 2]);
    let expected = [22.0, 28.0, 49.0, 64.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "matmul variant: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Test error handling for mismatched dimensions
#[test]
fn test_einsum_shape_mismatch() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0, 4.0];
    let result = einsum2("i,i->", &a, &[3], &b, &[4]);
    assert!(matches!(result, Err(EinsumError::ShapeMismatch { .. })));
}

// Test error handling for invalid subscript
#[test]
fn test_einsum_invalid_subscript() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let result = einsum1("12->", &a, &[4]);
    assert!(matches!(result, Err(EinsumError::InvalidSubscript(_))));
}

// Test implicit output (no ->)
// einsum('ij,jk') should produce 'ik' (indices that appear once)
#[test]
fn test_einsum_implicit_output() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = einsum2("ij,jk", &a, &[2, 3], &b, &[3, 2]).unwrap();
    // Implicit output is 'ik' - same as matmul
    assert_eq!(shape, vec![2, 2]);
    let expected = [22.0, 28.0, 49.0, 64.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "implicit output: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Test element-wise product with contraction
// einsum('i,i->i', a, b) = a * b (element-wise)
#[test]
fn test_einsum_elementwise_product() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let (result, shape) = einsum2("i,i->i", &a, &[3], &b, &[3]).unwrap();
    assert_eq!(shape, vec![3]);
    let expected = [4.0, 10.0, 18.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "element-wise: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Test identity (copy)
// einsum('i->i', a) = a
#[test]
fn test_einsum_identity() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let (result, shape) = einsum1("i->i", &a, &[5]).unwrap();
    assert_eq!(shape, vec![5]);
    assert!(
        vec_approx_eq(&result, &a, 1e-10),
        "identity: got {:?}, expected {:?}",
        result,
        a
    );
}
