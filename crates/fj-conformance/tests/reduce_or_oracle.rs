//! Oracle tests for ReduceOr primitive.
//!
//! ReduceOr: OR reduction along specified axes
//! - For booleans: any(values) - true if any value is true
//! - For integers: bitwise OR of all values along axis
//!
//! Tests:
//! - Boolean OR reduction
//! - Integer bitwise OR reduction
//! - Single axis reduction
//! - Multi-axis reduction
//! - Full reduction

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_bool_tensor(shape: &[u32], data: Vec<bool>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
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

fn extract_bool_scalar(v: &Value) -> bool {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            match &t.elements[0] {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool"),
            }
        }
        Value::Scalar(Literal::Bool(b)) => *b,
        _ => panic!("expected bool"),
    }
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => panic!("expected bool"),
    }
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_i64().unwrap()
        }
        Value::Scalar(lit) => lit.as_i64().unwrap(),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn reduce_params(axes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "axes".to_string(),
        axes.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ====================== BOOLEAN OR REDUCTION ======================

#[test]
fn oracle_reduce_or_bool_all_false() {
    // OR of all false = false
    let input = make_bool_tensor(&[4], vec![false, false, false, false]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_bool_scalar(&result), false);
}

#[test]
fn oracle_reduce_or_bool_all_true() {
    // OR of all true = true
    let input = make_bool_tensor(&[4], vec![true, true, true, true]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_bool_scalar(&result), true);
}

#[test]
fn oracle_reduce_or_bool_mixed() {
    // OR with at least one true = true
    let input = make_bool_tensor(&[4], vec![false, true, false, false]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_bool_scalar(&result), true);
}

#[test]
fn oracle_reduce_or_bool_single_true() {
    // Single true among many false = true
    let input = make_bool_tensor(&[5], vec![false, false, false, false, true]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_bool_scalar(&result), true);
}

// ====================== BOOLEAN 2D ======================

#[test]
fn oracle_reduce_or_bool_2d_axis0() {
    // [[F, T], [F, F]] -> [F|F, T|F] = [F, T]
    let input = make_bool_tensor(&[2, 2], vec![false, true, false, false]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_bool_vec(&result), vec![false, true]);
}

#[test]
fn oracle_reduce_or_bool_2d_axis1() {
    // [[F, T], [F, F]] -> [F|T, F|F] = [T, F]
    let input = make_bool_tensor(&[2, 2], vec![false, true, false, false]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false]);
}

#[test]
fn oracle_reduce_or_bool_2d_full() {
    // Full reduction: any true in entire tensor
    let input = make_bool_tensor(&[2, 2], vec![false, false, false, true]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_bool_scalar(&result), true);
}

#[test]
fn oracle_reduce_or_bool_2d_full_all_false() {
    let input = make_bool_tensor(&[2, 2], vec![false, false, false, false]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0, 1])).unwrap();
    assert_eq!(extract_bool_scalar(&result), false);
}

// ====================== INTEGER BITWISE OR ======================

#[test]
fn oracle_reduce_or_i64_basic() {
    // 0b0001 | 0b0010 | 0b0100 | 0b1000 = 0b1111 = 15
    let input = make_i64_tensor(&[4], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 15);
}

#[test]
fn oracle_reduce_or_i64_with_zero() {
    // x | 0 = x
    let input = make_i64_tensor(&[3], vec![0xFF, 0, 0]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0xFF);
}

#[test]
fn oracle_reduce_or_i64_all_same() {
    // x | x | x = x
    let input = make_i64_tensor(&[3], vec![0xABCD, 0xABCD, 0xABCD]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0xABCD);
}

#[test]
fn oracle_reduce_or_i64_all_zeros() {
    // 0 | 0 | 0 = 0
    let input = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0);
}

#[test]
fn oracle_reduce_or_i64_complementary() {
    // 0xF0 | 0x0F = 0xFF
    let input = make_i64_tensor(&[2], vec![0xF0, 0x0F]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0xFF);
}

// ====================== INTEGER 2D ======================

#[test]
fn oracle_reduce_or_i64_2d_axis0() {
    // [[1, 2], [4, 8]] -> [1|4, 2|8] = [5, 10]
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![5, 10]);
}

#[test]
fn oracle_reduce_or_i64_2d_axis1() {
    // [[1, 2], [4, 8]] -> [1|2, 4|8] = [3, 12]
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 12]);
}

#[test]
fn oracle_reduce_or_i64_2d_full() {
    // Full reduction: OR all elements
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 4, 8]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0, 1])).unwrap();
    assert_eq!(extract_shape(&result), Vec::<u32>::new());
    assert_eq!(extract_i64_scalar(&result), 15); // 1|2|4|8 = 15
}

// ====================== 3D TENSOR ======================

#[test]
fn oracle_reduce_or_bool_3d_axis0() {
    // Shape [2, 2, 2]
    // [[[F, T], [F, F]], [[T, F], [F, T]]]
    // axis 0: [[F|T, T|F], [F|F, F|T]] = [[T, T], [F, T]]
    let input = make_bool_tensor(
        &[2, 2, 2],
        vec![false, true, false, false, true, false, false, true],
    );
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![true, true, false, true]);
}

// ====================== MATHEMATICAL PROPERTIES ======================

#[test]
fn oracle_reduce_or_idempotent() {
    // x | x = x
    let input = make_i64_tensor(&[3], vec![0x55, 0x55, 0x55]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0x55);
}

#[test]
fn oracle_reduce_or_absorbing_element() {
    // For integers: -1 (all bits set) is absorbing: x | (-1) = -1
    let input = make_i64_tensor(&[3], vec![0x12345, -1, 0xABCDE]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), -1);
}

#[test]
fn oracle_reduce_or_identity_element() {
    // 0 is identity for OR: x | 0 = x
    let input = make_i64_tensor(&[3], vec![0, 0x5678, 0]);
    let result = eval_primitive(Primitive::ReduceOr, &[input], &reduce_params(&[0])).unwrap();
    assert_eq!(extract_i64_scalar(&result), 0x5678);
}
