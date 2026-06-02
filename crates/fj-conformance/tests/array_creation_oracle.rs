//! Oracle conformance tests for jnp array creation functions.
//!
//! Reference values computed with JAX:
//! ```python
//! import jax.numpy as jnp
//! jnp.zeros((2, 3))
//! jnp.ones((2, 3))
//! jnp.eye(3)
//! jnp.linspace(0, 1, 5)
//! jnp.arange(0, 5, 1)
//! ```

use fj_core::{DType, Value};
use fj_lax::array_creation::{
    arange, diag, diagonal, eye, flip_2d, full, hstack_1d, linspace, logspace, ones, repeat_1d,
    roll_1d, stack_1d, tile_1d, trace, tri, tril, triu, vstack_1d, zeros,
};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

fn extract_f64_values(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().filter_map(|l| l.as_f64()).collect(),
        _ => vec![],
    }
}

// JAX reference: jnp.zeros((2, 3))
#[test]
fn test_zeros_2x3() {
    let result = zeros(&[2, 3], DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 6);
    assert!(values.iter().all(|&v| approx_eq(v, 0.0, 1e-10)));
}

// JAX reference: jnp.ones((2, 3))
#[test]
fn test_ones_2x3() {
    let result = ones(&[2, 3], DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 6);
    assert!(values.iter().all(|&v| approx_eq(v, 1.0, 1e-10)));
}

// JAX reference: jnp.full((2, 2), 3.14)
#[test]
fn test_full_pi() {
    let fill_value = f64::from(314_u16) / f64::from(100_u16);
    let result = full(&[2, 2], fill_value, DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|&v| approx_eq(v, fill_value, 1e-10)));
}

// JAX reference: jnp.eye(3)
// [[1,0,0],[0,1,0],[0,0,1]]
#[test]
fn test_eye_3x3() {
    let result = eye(3, None, 0, DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 9);
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert!(vec_approx_eq(&values, &expected, 1e-10));
}

// JAX reference: jnp.eye(3, 4)
// [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
#[test]
fn test_eye_3x4() {
    let result = eye(3, Some(4), 0, DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 12);
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert!(vec_approx_eq(&values, &expected, 1e-10));
}

// JAX reference: jnp.eye(3, k=1)
// [[0,1,0],[0,0,1],[0,0,0]]
#[test]
fn test_eye_diagonal_offset() {
    let result = eye(3, None, 1, DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 9);
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    assert!(vec_approx_eq(&values, &expected, 1e-10));
}

// JAX reference: jnp.linspace(0, 1, 5)
// [0.0, 0.25, 0.5, 0.75, 1.0]
#[test]
fn test_linspace_endpoint() {
    let result = linspace(0.0, 1.0, 5, true).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 5);
    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "linspace: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.linspace(0, 1, 5, endpoint=False)
// [0.0, 0.2, 0.4, 0.6, 0.8]
#[test]
fn test_linspace_no_endpoint() {
    let result = linspace(0.0, 1.0, 5, false).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 5);
    let expected = [0.0, 0.2, 0.4, 0.6, 0.8];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "linspace no endpoint: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.arange(0, 5, 1)
// [0, 1, 2, 3, 4]
#[test]
fn test_arange_integers() {
    let result = arange(0.0, 5.0, 1.0).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 5);
    let expected = [0.0, 1.0, 2.0, 3.0, 4.0];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "arange: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.arange(0, 1, 0.3)
// [0.0, 0.3, 0.6, 0.9]
#[test]
fn test_arange_fractional() {
    let result = arange(0.0, 1.0, 0.3).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 4);
    let expected = [0.0, 0.3, 0.6, 0.9];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "arange fractional: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.logspace(0, 2, 3)
// [1.0, 10.0, 100.0]
#[test]
fn test_logspace_base10() {
    let result = logspace(0.0, 2.0, 3, true, 10.0).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 3);
    let expected = [1.0, 10.0, 100.0];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "logspace: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.diag([1, 2, 3])
// [[1,0,0],[0,2,0],[0,0,3]]
#[test]
fn test_diag_vector() {
    let result = diag(&[1.0, 2.0, 3.0], 0).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 9);
    let expected = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    assert!(
        vec_approx_eq(&values, &expected, 1e-10),
        "diag: got {:?}, expected {:?}",
        values,
        expected
    );
}

// JAX reference: jnp.triu([[1,2,3],[4,5,6],[7,8,9]])
// [[1,2,3],[0,5,6],[0,0,9]]
#[test]
fn test_triu_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = triu(&a, 3, 3, 0);
    let expected = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "triu: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.tril([[1,2,3],[4,5,6],[7,8,9]])
// [[1,0,0],[4,5,0],[7,8,9]]
#[test]
fn test_tril_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = tril(&a, 3, 3, 0);
    let expected = [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "tril: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.trace([[1,2,3],[4,5,6],[7,8,9]])
// 1 + 5 + 9 = 15
#[test]
fn test_trace_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = trace(&a, 3, 3, 0);
    assert!(
        approx_eq(result, 15.0, 1e-10),
        "trace: got {}, expected 15.0",
        result
    );
}

// JAX reference: jnp.diagonal([[1,2,3],[4,5,6],[7,8,9]])
// [1, 5, 9]
#[test]
fn test_diagonal_3x3() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = diagonal(&a, 3, 3, 0);
    let expected = [1.0, 5.0, 9.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "diagonal: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.flip([[1,2],[3,4]], axis=0)
// [[3,4],[1,2]]
#[test]
fn test_flip_axis0() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let result = flip_2d(&a, 2, 2, 0);
    let expected = [3.0, 4.0, 1.0, 2.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "flip axis=0: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.flip([[1,2],[3,4]], axis=1)
// [[2,1],[4,3]]
#[test]
fn test_flip_axis1() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let result = flip_2d(&a, 2, 2, 1);
    let expected = [2.0, 1.0, 4.0, 3.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "flip axis=1: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.roll([1,2,3,4,5], 2)
// [4, 5, 1, 2, 3]
#[test]
fn test_roll_positive() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = roll_1d(&a, 2);
    let expected = [4.0, 5.0, 1.0, 2.0, 3.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "roll +2: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.roll([1,2,3,4,5], -2)
// [3, 4, 5, 1, 2]
#[test]
fn test_roll_negative() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = roll_1d(&a, -2);
    let expected = [3.0, 4.0, 5.0, 1.0, 2.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "roll -2: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.tri(3)
// [[1,0,0],[1,1,0],[1,1,1]]
#[test]
fn test_tri_3x3() {
    let result = tri(3, 3, 0);
    let expected = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "tri: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.stack([[1,2],[3,4]])
#[test]
fn test_stack_1d() {
    let a1 = [1.0, 2.0];
    let a2 = [3.0, 4.0];
    let (result, shape) = stack_1d(&[&a1[..], &a2[..]]);
    assert_eq!(shape, vec![2, 2]);
    let expected = [1.0, 2.0, 3.0, 4.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "stack: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.vstack([[1,2],[3,4]])
// [1, 2, 3, 4]
#[test]
fn test_vstack_1d() {
    let a1 = [1.0, 2.0];
    let a2 = [3.0, 4.0];
    let result = vstack_1d(&[&a1[..], &a2[..]]);
    let expected = [1.0, 2.0, 3.0, 4.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "vstack: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.hstack([[1,2],[3,4]])
// [1, 2, 3, 4]
#[test]
fn test_hstack_1d() {
    let a1 = [1.0, 2.0];
    let a2 = [3.0, 4.0];
    let result = hstack_1d(&[&a1[..], &a2[..]]);
    let expected = [1.0, 2.0, 3.0, 4.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "hstack: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.repeat([1, 2, 3], 2)
// [1, 1, 2, 2, 3, 3]
#[test]
fn test_repeat_1d() {
    let a = [1.0, 2.0, 3.0];
    let result = repeat_1d(&a, 2);
    let expected = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "repeat: got {:?}, expected {:?}",
        result,
        expected
    );
}

// JAX reference: jnp.tile([1, 2, 3], 3)
// [1, 2, 3, 1, 2, 3, 1, 2, 3]
#[test]
fn test_tile_1d() {
    let a = [1.0, 2.0, 3.0];
    let result = tile_1d(&a, 3);
    let expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    assert!(
        vec_approx_eq(&result, &expected, 1e-10),
        "tile: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Test empty array
#[test]
fn test_zeros_empty() {
    let result = zeros(&[0], DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert!(values.is_empty());
}

// Test scalar (0-d array)
#[test]
fn test_ones_scalar() {
    let result = ones(&[], DType::F64).unwrap();
    let values = extract_f64_values(&result);
    assert_eq!(values.len(), 1);
    assert!(approx_eq(values[0], 1.0, 1e-10));
}
