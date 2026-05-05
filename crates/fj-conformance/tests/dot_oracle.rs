//! Oracle tests for Dot primitive.
//!
//! dot(a, b) = matrix multiplication / tensor contraction
//!
//! For vectors: dot product (sum of element-wise products)
//! For matrices: standard matrix multiplication
//!
//! Tests:
//! - Vector dot product: sum of a[i]*b[i]
//! - Matrix-vector multiplication
//! - Matrix-matrix multiplication
//! - Identity matrix
//! - Zero matrix
//! - Transpose relationship

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

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
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

// ====================== VECTOR DOT PRODUCT ======================

#[test]
fn oracle_dot_vector_basic() {
    // [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_f64_scalar(&result), 32.0, "dot([1,2,3], [4,5,6])");
}

#[test]
fn oracle_dot_vector_zeros() {
    // [1, 2, 3] · [0, 0, 0] = 0
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "dot with zero vector");
}

#[test]
fn oracle_dot_vector_orthogonal() {
    // Orthogonal vectors: [1, 0] · [0, 1] = 0
    let a = make_f64_tensor(&[2], vec![1.0, 0.0]);
    let b = make_f64_tensor(&[2], vec![0.0, 1.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "orthogonal vectors");
}

#[test]
fn oracle_dot_vector_parallel() {
    // Parallel vectors: [2, 4] · [1, 2] = 2 + 8 = 10
    let a = make_f64_tensor(&[2], vec![2.0, 4.0]);
    let b = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 10.0, "parallel vectors");
}

#[test]
fn oracle_dot_vector_unit() {
    // Unit vector dot with itself = 1
    let sqrt_half = (0.5_f64).sqrt();
    let a = make_f64_tensor(&[2], vec![sqrt_half, sqrt_half]);
    let b = make_f64_tensor(&[2], vec![sqrt_half, sqrt_half]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "unit vector");
}

// ====================== MATRIX-VECTOR ======================

#[test]
fn oracle_dot_matrix_vector() {
    // [[1, 2], [3, 4]] @ [1, 2] = [1*1+2*2, 3*1+4*2] = [5, 11]
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, 11.0]);
}

#[test]
fn oracle_dot_identity_vector() {
    // Identity matrix times vector = vector
    let a = make_f64_tensor(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let b = make_f64_tensor(&[3], vec![5.0, 7.0, 9.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![5.0, 7.0, 9.0]);
}

// ====================== MATRIX-MATRIX ======================

#[test]
fn oracle_dot_matrix_basic() {
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn oracle_dot_identity_matrix() {
    // Identity @ any = any
    let identity = make_f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let a = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Dot, &[identity, a], &no_params()).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn oracle_dot_matrix_identity_right() {
    // any @ Identity = any
    let a = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let identity = make_f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = eval_primitive(Primitive::Dot, &[a, identity], &no_params()).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn oracle_dot_zero_matrix() {
    // Zero @ any = Zero
    let zero = make_f64_tensor(&[2, 2], vec![0.0, 0.0, 0.0, 0.0]);
    let a = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Dot, &[zero, a], &no_params()).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.0, 0.0]);
}

// ====================== RECTANGULAR MATRICES ======================

#[test]
fn oracle_dot_rectangular() {
    // [2, 3] @ [3, 2] = [2, 2]
    // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
    // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    let a = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = make_f64_tensor(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn oracle_dot_row_column() {
    // Row vector @ column vector = scalar (via matrix multiplication)
    // [1, 3] @ [2, 4] (as matrices) should work as [1,2] @ [2,1] = [1,1]
    let a = make_f64_tensor(&[1, 2], vec![1.0, 3.0]);
    let b = make_f64_tensor(&[2, 1], vec![2.0, 4.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_f64_vec(&result), vec![14.0]); // 1*2 + 3*4 = 14
}

// ====================== MATHEMATICAL PROPERTIES ======================

#[test]
fn oracle_dot_commutative_vectors() {
    // Vector dot product is commutative: a·b = b·a
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![5.0, 6.0, 7.0, 8.0]);
    let result_ab = eval_primitive(Primitive::Dot, &[a.clone(), b.clone()], &no_params()).unwrap();
    let result_ba = eval_primitive(Primitive::Dot, &[b, a], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result_ab),
        extract_f64_scalar(&result_ba),
        "vector dot is commutative"
    );
}

#[test]
fn oracle_dot_distributive() {
    // Vector: a·(b + c) = a·b + a·c
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let c = make_f64_tensor(&[3], vec![7.0, 8.0, 9.0]);

    let ab = eval_primitive(Primitive::Dot, &[a.clone(), b.clone()], &no_params()).unwrap();
    let ac = eval_primitive(Primitive::Dot, &[a.clone(), c.clone()], &no_params()).unwrap();
    let ab_plus_ac = extract_f64_scalar(&ab) + extract_f64_scalar(&ac);

    let b_plus_c = make_f64_tensor(&[3], vec![11.0, 13.0, 15.0]); // b + c
    let a_bc = eval_primitive(Primitive::Dot, &[a, b_plus_c], &no_params()).unwrap();

    assert_close(
        extract_f64_scalar(&a_bc),
        ab_plus_ac,
        1e-12,
        "distributive property",
    );
}

#[test]
fn oracle_dot_scalar_mult_factor() {
    // (ka)·b = k(a·b)
    let k = 3.0;
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let ka = make_f64_tensor(&[3], vec![k * 1.0, k * 2.0, k * 3.0]);

    let ab = eval_primitive(Primitive::Dot, &[a, b.clone()], &no_params()).unwrap();
    let ka_b = eval_primitive(Primitive::Dot, &[ka, b], &no_params()).unwrap();

    assert_close(
        extract_f64_scalar(&ka_b),
        k * extract_f64_scalar(&ab),
        1e-12,
        "scalar multiplication factor",
    );
}

// ====================== LARGER MATRICES ======================

#[test]
fn oracle_dot_3x3() {
    // 3x3 @ 3x3
    let a = make_f64_tensor(&[3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = make_f64_tensor(&[3, 3], vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    // Row 0: [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
    // Row 1: [4*9+5*6+6*3, 4*8+5*5+6*2, 4*7+5*4+6*1] = [84, 69, 54]
    // Row 2: [7*9+8*6+9*3, 7*8+8*5+9*2, 7*7+8*4+9*1] = [138, 114, 90]
    assert_eq!(
        extract_f64_vec(&result),
        vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0]
    );
}

// ====================== SPECIAL FLOATING-POINT VALUES ======================
// JAX parity: NaN and Inf propagation in dot products

#[test]
fn oracle_dot_nan_propagates() {
    // JAX: jnp.dot([1.0, nan, 3.0], [4.0, 5.0, 6.0]) = nan
    let a = make_f64_tensor(&[3], vec![1.0, f64::NAN, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan(), "dot with NaN element should produce NaN");
}

#[test]
fn oracle_dot_inf_propagates() {
    // JAX: jnp.dot([1.0, inf, 3.0], [4.0, 5.0, 6.0]) = inf
    let a = make_f64_tensor(&[3], vec![1.0, f64::INFINITY, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(
        val.is_infinite() && val > 0.0,
        "dot with +Inf should produce +Inf"
    );
}

#[test]
fn oracle_dot_inf_times_zero_is_nan() {
    // JAX: jnp.dot([inf], [0.0]) = nan (because inf * 0 = nan)
    let a = make_f64_tensor(&[1], vec![f64::INFINITY]);
    let b = make_f64_tensor(&[1], vec![0.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan(), "inf * 0 in dot should produce NaN");
}

#[test]
fn oracle_dot_inf_minus_inf_is_nan() {
    // JAX: jnp.dot([inf, -inf], [1.0, 1.0]) = nan (because inf + (-inf) = nan)
    let a = make_f64_tensor(&[2], vec![f64::INFINITY, f64::NEG_INFINITY]);
    let b = make_f64_tensor(&[2], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_nan(), "inf + (-inf) in dot should produce NaN");
}

#[test]
fn oracle_dot_matrix_nan_row() {
    // JAX: matrix with NaN in one row, only that row of output is NaN
    let a = make_f64_tensor(&[2, 2], vec![1.0, 2.0, f64::NAN, 4.0]);
    let b = make_f64_tensor(&[2], vec![1.0, 1.0]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 2);
    assert!((vals[0] - 3.0).abs() < 1e-10, "row 0 should be 1+2=3");
    assert!(vals[1].is_nan(), "row 1 with NaN should produce NaN");
}

// ====================== COMPLEX NUMBER TESTS ======================
// JAX parity: Complex64 and Complex128 matrix multiplication

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

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        Value::Scalar(l) => vec![l.as_complex128().unwrap()],
    }
}

fn assert_complex_close(actual: (f64, f64), expected: (f64, f64), tol: f64, msg: &str) {
    let diff = ((actual.0 - expected.0).powi(2) + (actual.1 - expected.1).powi(2)).sqrt();
    assert!(
        diff < tol,
        "{msg}: expected ({}, {}i), got ({}, {}i), diff = {}",
        expected.0,
        expected.1,
        actual.0,
        actual.1,
        diff
    );
}

#[test]
fn oracle_dot_complex128_vector_dot_product() {
    // JAX: jnp.dot(jnp.array([1+2j, 3+4j]), jnp.array([5+6j, 7+8j]))
    // = (1+2j)*(5+6j) + (3+4j)*(7+8j)
    // = (1*5 - 2*6) + (1*6 + 2*5)j + (3*7 - 4*8) + (3*8 + 4*7)j
    // = (5-12) + (6+10)j + (21-32) + (24+28)j
    // = -7 + 16j + -11 + 52j
    // = -18 + 68j
    let a = make_complex128_tensor(&[2], vec![(1.0, 2.0), (3.0, 4.0)]);
    let b = make_complex128_tensor(&[2], vec![(5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let vals = extract_complex128_vec(&result);
    assert_eq!(vals.len(), 1);
    assert_complex_close(vals[0], (-18.0, 68.0), 1e-10, "complex vector dot");
}

#[test]
fn oracle_dot_complex128_matmul_2x2() {
    // JAX: [[1+1j, 2+0j], [0+1j, 1-1j]] @ [[1+0j, 0+1j], [1+1j, 0+0j]]
    // Row 0: (1+1j)*(1+0j) + (2+0j)*(1+1j) = (1+1j) + (2+2j) = 3+3j
    //        (1+1j)*(0+1j) + (2+0j)*(0+0j) = (-1+1j) + 0 = -1+1j
    // Row 1: (0+1j)*(1+0j) + (1-1j)*(1+1j) = (0+1j) + (1+1-1+1j) = (0+1j) + (2+0j) = 2+1j
    //        (0+1j)*(0+1j) + (1-1j)*(0+0j) = -1 + 0 = -1+0j
    let a = make_complex128_tensor(&[2, 2], vec![(1.0, 1.0), (2.0, 0.0), (0.0, 1.0), (1.0, -1.0)]);
    let b = make_complex128_tensor(&[2, 2], vec![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_complex128_vec(&result);
    assert_complex_close(vals[0], (3.0, 3.0), 1e-10, "C[0,0]");
    assert_complex_close(vals[1], (-1.0, 1.0), 1e-10, "C[0,1]");
    assert_complex_close(vals[2], (2.0, 1.0), 1e-10, "C[1,0]");
    assert_complex_close(vals[3], (-1.0, 0.0), 1e-10, "C[1,1]");
}

#[test]
fn oracle_dot_complex64_vector_dot_product() {
    // Same as complex128 but with f32 precision
    let a = make_complex64_tensor(&[2], vec![(1.0, 2.0), (3.0, 4.0)]);
    let b = make_complex64_tensor(&[2], vec![(5.0, 6.0), (7.0, 8.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    // Complex64 result - extract as f32 then convert
    let val = match &result {
        Value::Tensor(t) => t.elements[0].as_complex64().unwrap(),
        Value::Scalar(l) => l.as_complex64().unwrap(),
    };
    let val_f64 = (val.0 as f64, val.1 as f64);
    assert_complex_close(val_f64, (-18.0, 68.0), 1e-4, "complex64 vector dot");
}

#[test]
fn oracle_dot_complex128_matrix_vector() {
    // JAX: [[1+0j, 0+1j], [1+1j, 1-1j]] @ [1+0j, 0+1j]
    // Row 0: (1+0j)*(1+0j) + (0+1j)*(0+1j) = 1 + (-1) = 0+0j
    // Row 1: (1+1j)*(1+0j) + (1-1j)*(0+1j) = (1+1j) + (1+1j) = 2+2j
    let a = make_complex128_tensor(&[2, 2], vec![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)]);
    let b = make_complex128_tensor(&[2], vec![(1.0, 0.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_complex128_vec(&result);
    assert_complex_close(vals[0], (0.0, 0.0), 1e-10, "row 0");
    assert_complex_close(vals[1], (2.0, 2.0), 1e-10, "row 1");
}

#[test]
fn oracle_dot_complex128_identity_matrix() {
    // A @ I = A for complex matrices
    let a = make_complex128_tensor(&[2, 2], vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]);
    let identity =
        make_complex128_tensor(&[2, 2], vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, identity], &no_params()).unwrap();
    let vals = extract_complex128_vec(&result);
    assert_complex_close(vals[0], (1.0, 2.0), 1e-10, "A[0,0]");
    assert_complex_close(vals[1], (3.0, 4.0), 1e-10, "A[0,1]");
    assert_complex_close(vals[2], (5.0, 6.0), 1e-10, "A[1,0]");
    assert_complex_close(vals[3], (7.0, 8.0), 1e-10, "A[1,1]");
}

#[test]
fn oracle_dot_complex128_preserves_dtype() {
    // Result dtype should be Complex128 when inputs are Complex128
    let a = make_complex128_tensor(&[2], vec![(1.0, 0.0), (0.0, 1.0)]);
    let b = make_complex128_tensor(&[2], vec![(1.0, 0.0), (0.0, 1.0)]);
    let result = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap();
    match result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::Complex128),
        Value::Scalar(l) => assert!(l.is_complex()),
    }
}

fn reduce_sum_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axes".to_string(), "0".to_string());
    p
}

// ======================== METAMORPHIC: Dot(a, b) = ReduceSum(Mul(a, b)) for vectors ========================

#[test]
fn metamorphic_dot_equals_reduce_sum_mul() {
    // For vectors: Dot(a, b) = ReduceSum(Mul(a, b))
    let a = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_tensor(&[4], vec![5.0, 6.0, 7.0, 8.0]);

    // Direct dot product
    let dot_result = eval_primitive(Primitive::Dot, &[a.clone(), b.clone()], &no_params()).unwrap();

    // Element-wise multiply then sum
    let mul_result = eval_primitive(Primitive::Mul, &[a, b], &no_params()).unwrap();
    let sum_result = eval_primitive(Primitive::ReduceSum, &[mul_result], &reduce_sum_params()).unwrap();

    assert_close(
        extract_f64_scalar(&dot_result),
        extract_f64_scalar(&sum_result),
        1e-14,
        "Dot(a, b) = ReduceSum(Mul(a, b))",
    );
}

// ======================== METAMORPHIC: Dot(k*a, b) = k * Dot(a, b) ========================

#[test]
fn metamorphic_dot_scalar_factor_out() {
    // Dot(k*a, b) = k * Dot(a, b) (scalar multiplication factors out)
    let k = 3.0;
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let k_val = make_f64_tensor(&[3], vec![k, k, k]);

    // Dot(a, b)
    let dot_ab = eval_primitive(Primitive::Dot, &[a.clone(), b.clone()], &no_params()).unwrap();

    // Dot(k*a, b)
    let ka = eval_primitive(Primitive::Mul, &[k_val, a], &no_params()).unwrap();
    let dot_ka_b = eval_primitive(Primitive::Dot, &[ka, b], &no_params()).unwrap();

    assert_close(
        extract_f64_scalar(&dot_ka_b),
        k * extract_f64_scalar(&dot_ab),
        1e-12,
        "Dot(k*a, b) = k * Dot(a, b)",
    );
}

// ======================== METAMORPHIC: Dot(a, a) >= 0 (non-negative for real vectors) ========================

#[test]
fn metamorphic_dot_self_nonnegative() {
    // Dot(a, a) >= 0 for any real vector (squared magnitude)
    for vals in [
        vec![1.0, 2.0, 3.0],
        vec![-1.0, -2.0, -3.0],
        vec![0.0, 0.0, 0.0],
        vec![0.5, -0.5, 0.5, -0.5],
    ] {
        let a = make_f64_tensor(&[vals.len() as u32], vals.clone());
        let result = eval_primitive(Primitive::Dot, &[a.clone(), a], &no_params()).unwrap();
        let dot_val = extract_f64_scalar(&result);
        assert!(
            dot_val >= 0.0,
            "Dot(a, a) >= 0 for {:?}, got {}",
            vals,
            dot_val
        );
    }
}

// ======================== METAMORPHIC: Dot(a, 0) = 0 ========================

#[test]
fn metamorphic_dot_with_zero_vector() {
    // Dot(a, 0) = 0 for any vector a
    for vals in [vec![1.0, 2.0, 3.0], vec![-5.0, 10.0], vec![0.0, 0.0, 0.0, 0.0]] {
        let len = vals.len() as u32;
        let a = make_f64_tensor(&[len], vals.clone());
        let zero = make_f64_tensor(&[len], vec![0.0; vals.len()]);
        let result = eval_primitive(Primitive::Dot, &[a, zero], &no_params()).unwrap();
        assert_close(
            extract_f64_scalar(&result),
            0.0,
            1e-14,
            &format!("Dot({:?}, 0) = 0", vals),
        );
    }
}
