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
