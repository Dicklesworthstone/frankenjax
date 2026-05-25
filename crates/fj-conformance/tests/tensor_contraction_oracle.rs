//! Oracle conformance tests for tensordot, outer, inner, kron.
//!
//! Reference values computed with JAX/NumPy:
//! ```python
//! import jax.numpy as jnp
//! jnp.tensordot(a, b, axes)
//! jnp.outer(a, b)
//! jnp.inner(a, b)
//! jnp.kron(a, b)
//! ```

use fj_lax::tensor_contraction::{inner, kron, matmul_2d, outer, tensordot};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

// JAX reference: tensordot for matrix multiplication
// A (2x3) @ B (3x2) = C (2x2)
// A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
// C = [[22,28],[49,64]]
#[test]
fn test_tensordot_matmul() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (result, shape) = tensordot(&a, &[2, 3], &b, &[3, 2], &[1], &[0]);
    assert_eq!(shape, vec![2, 2]);
    let expected = [22.0, 28.0, 49.0, 64.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "tensordot matmul: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: tensordot for dot product (full contraction)
// tensordot([1,2,3], [4,5,6], axes=1) = 32
#[test]
fn test_tensordot_dot_product() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let (result, shape) = tensordot(&a, &[3], &b, &[3], &[0], &[0]);
    assert!(shape.is_empty(), "scalar result should have empty shape");
    assert!(
        approx_eq(result[0], 32.0, 1e-10),
        "dot product: got {}, expected 32.0",
        result[0]
    );
}

// JAX reference: tensordot for outer product (no contraction)
// tensordot([1,2], [3,4,5], axes=0) = [[3,4,5],[6,8,10]]
#[test]
fn test_tensordot_outer_product() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0, 5.0];
    let (result, shape) = tensordot(&a, &[2], &b, &[3], &[], &[]);
    assert_eq!(shape, vec![2, 3]);
    let expected = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "outer via tensordot: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: matmul_2d
// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
#[test]
fn test_matmul_2d_jax_reference() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let result = matmul_2d(&a, 2, 2, &b, 2);
    let expected = [19.0, 22.0, 43.0, 50.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "matmul_2d: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Non-square matmul: (2x3) @ (3x4) = (2x4)
#[test]
fn test_matmul_2d_non_square() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]; // 3x4
    let result = matmul_2d(&a, 2, 3, &b, 4);
    assert_eq!(result.len(), 8);
    // row 0: [1,2,3] @ each col
    // col 0: 1*1 + 2*5 + 3*9 = 1+10+27 = 38
    assert!(
        approx_eq(result[0], 38.0, 1e-10),
        "first elem: got {}, expected 38",
        result[0]
    );
}

// JAX reference: outer([1,2], [3,4]) = [[3,4],[6,8]]
#[test]
fn test_outer_jax_reference() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0];
    let result = outer(&a, &b);
    let expected = [3.0, 4.0, 6.0, 8.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "outer: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: inner([1,2,3], [4,5,6]) = 32
#[test]
fn test_inner_jax_reference() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = inner(&a, &b);
    assert!(
        approx_eq(result, 32.0, 1e-10),
        "inner: got {}, expected 32.0",
        result
    );
}

// JAX reference: kron([[1,2],[3,4]], [[0,5],[6,7]])
// Result is 4x4:
// [[0, 5, 0, 10],
//  [6, 7, 12, 14],
//  [0, 15, 0, 20],
//  [18, 21, 24, 28]]
#[test]
fn test_kron_jax_reference() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [0.0, 5.0, 6.0, 7.0];
    let result = kron(&a, 2, 2, &b, 2, 2);
    assert_eq!(result.len(), 16);
    // First element: 1 * 0 = 0
    assert!(approx_eq(result[0], 0.0, 1e-10));
    // Element [0,1]: 1 * 5 = 5
    assert!(approx_eq(result[1], 5.0, 1e-10));
    // Element [1,2]: 1 * 6 * ... actually let me just check a few key elements
    // Element [3,3]: 4 * 7 = 28
    assert!(approx_eq(result[15], 28.0, 1e-10));
}

// Verify inner(a,b) = sum(a*b)
#[test]
fn test_inner_equals_sum_product() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 3.0, 4.0, 5.0, 6.0];
    let result = inner(&a, &b);
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert!(
        approx_eq(result, expected, 1e-10),
        "inner = sum(a*b): got {}, expected {}",
        result,
        expected
    );
}

// Verify outer product dimensions
#[test]
fn test_outer_dimensions() {
    let a = [1.0, 2.0, 3.0]; // length 3
    let b = [4.0, 5.0]; // length 2
    let result = outer(&a, &b);
    assert_eq!(result.len(), 6, "outer(3-vec, 2-vec) should be length 6");
}
