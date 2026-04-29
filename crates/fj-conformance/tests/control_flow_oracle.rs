//! Oracle tests for control flow primitives: Cond, Switch, Scan, While.
//!
//! These primitives implement control flow semantics:
//! - Cond: conditional select between two values based on boolean predicate
//! - Switch: select among multiple values based on integer index
//! - Scan: iterate over tensor slices accumulating a carry value
//! - While: loop while condition holds, applying body operation

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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
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

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn scan_params(body_op: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("body_op".to_string(), body_op.to_string());
    p
}

fn scan_params_reverse(body_op: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("body_op".to_string(), body_op.to_string());
    p.insert("reverse".to_string(), "true".to_string());
    p
}

fn while_params(body_op: &str, cond_op: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("body_op".to_string(), body_op.to_string());
    p.insert("cond_op".to_string(), cond_op.to_string());
    p
}

fn switch_params(num_branches: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("num_branches".to_string(), num_branches.to_string());
    p
}

// ======================== Cond Tests ========================

#[test]
fn oracle_cond_true_scalar() {
    // pred=true -> return true_val
    let pred = Value::scalar_bool(true);
    let true_val = Value::Scalar(Literal::from_f64(1.0));
    let false_val = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_cond_false_scalar() {
    // pred=false -> return false_val
    let pred = Value::scalar_bool(false);
    let true_val = Value::Scalar(Literal::from_f64(1.0));
    let false_val = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0).abs() < 1e-10);
}

#[test]
fn oracle_cond_true_tensor() {
    let pred = Value::scalar_bool(true);
    let true_val = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let false_val = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn oracle_cond_false_tensor() {
    let pred = Value::scalar_bool(false);
    let true_val = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let false_val = make_f64_tensor(&[3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![4.0, 5.0, 6.0]);
}

#[test]
fn oracle_cond_2d_tensor() {
    let pred = Value::scalar_bool(true);
    let true_val = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let false_val = make_f64_tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
}

#[test]
fn oracle_cond_negative_values() {
    let pred = Value::scalar_bool(false);
    let true_val = Value::Scalar(Literal::from_f64(-1.0));
    let false_val = Value::Scalar(Literal::from_f64(-2.0));
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-2.0)).abs() < 1e-10);
}

#[test]
fn oracle_cond_integer_values() {
    let pred = Value::scalar_bool(true);
    let true_val = Value::scalar_i64(42);
    let false_val = Value::scalar_i64(99);
    let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 42);
}

// ======================== Switch Tests ========================

#[test]
fn oracle_switch_index_0() {
    // index=0 -> return first branch
    let index = Value::scalar_i64(0);
    let branch0 = Value::Scalar(Literal::from_f64(10.0));
    let branch1 = Value::Scalar(Literal::from_f64(20.0));
    let branch2 = Value::Scalar(Literal::from_f64(30.0));
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1, branch2],
        &switch_params(3),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 10.0).abs() < 1e-10);
}

#[test]
fn oracle_switch_index_1() {
    let index = Value::scalar_i64(1);
    let branch0 = Value::Scalar(Literal::from_f64(10.0));
    let branch1 = Value::Scalar(Literal::from_f64(20.0));
    let branch2 = Value::Scalar(Literal::from_f64(30.0));
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1, branch2],
        &switch_params(3),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 20.0).abs() < 1e-10);
}

#[test]
fn oracle_switch_index_2() {
    let index = Value::scalar_i64(2);
    let branch0 = Value::Scalar(Literal::from_f64(10.0));
    let branch1 = Value::Scalar(Literal::from_f64(20.0));
    let branch2 = Value::Scalar(Literal::from_f64(30.0));
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1, branch2],
        &switch_params(3),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 30.0).abs() < 1e-10);
}

#[test]
fn oracle_switch_tensor_branches() {
    let index = Value::scalar_i64(1);
    let branch0 = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let branch1 = make_f64_tensor(&[2], vec![3.0, 4.0]);
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1],
        &switch_params(2),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![3.0, 4.0]);
}

#[test]
fn oracle_switch_two_branches() {
    // Binary switch (like cond but with index)
    let index = Value::scalar_i64(0);
    let branch0 = Value::Scalar(Literal::from_f64(100.0));
    let branch1 = Value::Scalar(Literal::from_f64(200.0));
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1],
        &switch_params(2),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 100.0).abs() < 1e-10);
}

#[test]
fn oracle_switch_integer_branches() {
    let index = Value::scalar_i64(1);
    let branch0 = Value::scalar_i64(111);
    let branch1 = Value::scalar_i64(222);
    let result = eval_primitive(
        Primitive::Switch,
        &[index, branch0, branch1],
        &switch_params(2),
    )
    .unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 222);
}

// ======================== Scan Tests ========================

#[test]
fn oracle_scan_sum_basic() {
    // init=0, xs=[1,2,3,4], body_op=add -> 0+1+2+3+4 = 10
    let init = Value::Scalar(Literal::from_f64(0.0));
    let xs = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 10.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_sum_with_init() {
    // init=5, xs=[1,2,3], body_op=add -> 5+1+2+3 = 11
    let init = Value::Scalar(Literal::from_f64(5.0));
    let xs = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 11.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_product() {
    // init=1, xs=[2,3,4], body_op=mul -> 1*2*3*4 = 24
    let init = Value::Scalar(Literal::from_f64(1.0));
    let xs = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("mul")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 24.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_sub() {
    // init=10, xs=[1,2,3], body_op=sub -> 10-1-2-3 = 4
    let init = Value::Scalar(Literal::from_f64(10.0));
    let xs = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("sub")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_max() {
    // init=-inf, xs=[3,1,4,1,5], body_op=max -> 5
    let init = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let xs = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("max")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_min() {
    // init=inf, xs=[3,1,4,1,5], body_op=min -> 1
    let init = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let xs = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("min")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_scan_reverse() {
    // init=0, xs=[1,2,3], body_op=sub, reverse -> 0-3-2-1 = -6
    let init = Value::Scalar(Literal::from_f64(0.0));
    let xs = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params_reverse("sub")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-6.0)).abs() < 1e-10);
}

#[test]
fn oracle_scan_single_element() {
    // init=5, xs=[3], body_op=add -> 5+3 = 8
    let init = Value::Scalar(Literal::from_f64(5.0));
    let xs = make_f64_tensor(&[1], vec![3.0]);
    let result = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 8.0).abs() < 1e-10);
}

// ======================== While Tests ========================

#[test]
fn oracle_while_count_up() {
    // init=0, step=1, threshold=5, body_op=add, cond_op=lt
    // Loop: 0<5? add 1 -> 1<5? add 1 -> ... -> 5<5? false -> return 5
    let init = Value::Scalar(Literal::from_f64(0.0));
    let step = Value::Scalar(Literal::from_f64(1.0));
    let threshold = Value::Scalar(Literal::from_f64(5.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("add", "lt"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 5.0).abs() < 1e-10);
}

#[test]
fn oracle_while_count_down() {
    // init=10, step=2, threshold=3, body_op=sub, cond_op=gt
    // Loop: 10>3? sub 2 -> 8>3? sub 2 -> 6>3? sub 2 -> 4>3? sub 2 -> 2>3? false -> return 2
    let init = Value::Scalar(Literal::from_f64(10.0));
    let step = Value::Scalar(Literal::from_f64(2.0));
    let threshold = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("sub", "gt"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.0).abs() < 1e-10);
}

#[test]
fn oracle_while_immediate_exit() {
    // init=10, threshold=5, cond_op=lt -> 10<5 is false, return immediately
    let init = Value::Scalar(Literal::from_f64(10.0));
    let step = Value::Scalar(Literal::from_f64(1.0));
    let threshold = Value::Scalar(Literal::from_f64(5.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("add", "lt"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 10.0).abs() < 1e-10);
}

#[test]
fn oracle_while_le_condition() {
    // init=0, step=1, threshold=3, cond_op=le
    // Loop: 0<=3? add 1 -> 1<=3? add 1 -> 2<=3? add 1 -> 3<=3? add 1 -> 4<=3? false -> return 4
    let init = Value::Scalar(Literal::from_f64(0.0));
    let step = Value::Scalar(Literal::from_f64(1.0));
    let threshold = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("add", "le"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 4.0).abs() < 1e-10);
}

#[test]
fn oracle_while_ge_condition() {
    // init=5, step=1, threshold=2, body_op=sub, cond_op=ge
    // Loop: 5>=2? sub 1 -> 4>=2? sub 1 -> 3>=2? sub 1 -> 2>=2? sub 1 -> 1>=2? false -> return 1
    let init = Value::Scalar(Literal::from_f64(5.0));
    let step = Value::Scalar(Literal::from_f64(1.0));
    let threshold = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("sub", "ge"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_while_ne_condition() {
    // init=0, step=1, threshold=3, cond_op=ne
    // Loop until carry == threshold
    let init = Value::Scalar(Literal::from_f64(0.0));
    let step = Value::Scalar(Literal::from_f64(1.0));
    let threshold = Value::Scalar(Literal::from_f64(3.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("add", "ne"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_while_multiply() {
    // init=1, step=2, threshold=16, body_op=mul, cond_op=lt
    // Loop: 1<16? mul 2 -> 2<16? mul 2 -> 4<16? mul 2 -> 8<16? mul 2 -> 16<16? false -> return 16
    let init = Value::Scalar(Literal::from_f64(1.0));
    let step = Value::Scalar(Literal::from_f64(2.0));
    let threshold = Value::Scalar(Literal::from_f64(16.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("mul", "lt"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 16.0).abs() < 1e-10);
}

#[test]
fn oracle_while_negative_step() {
    // init=10, step=-2, threshold=0, body_op=add, cond_op=gt
    // Counting down by adding negative step
    let init = Value::Scalar(Literal::from_f64(10.0));
    let step = Value::Scalar(Literal::from_f64(-2.0));
    let threshold = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(
        Primitive::While,
        &[init, step, threshold],
        &while_params("add", "gt"),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
}
