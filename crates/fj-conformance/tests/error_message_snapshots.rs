//! Snapshot tests for error messages.
//!
//! These tests freeze the format of user-facing error messages to catch
//! unintentional regressions in error reporting quality.

use fj_core::{DType, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

#[test]
fn snapshot_arity_mismatch_error() {
    let err = eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0)], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"arity mismatch for add: expected 2, got 1");
}

#[test]
fn snapshot_type_mismatch_scalar_error() {
    let err = eval_primitive(
        Primitive::Neg,
        &[Value::Scalar(fj_core::Literal::Bool(true))],
        &no_params(),
    )
    .unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"type mismatch for neg: expected numeric scalar, got bool");
}

#[test]
fn snapshot_shape_mismatch_error() {
    let a = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            vec![fj_core::Literal::from_f64(0.0); 6],
        )
        .unwrap(),
    );
    let b = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 2] },
            vec![fj_core::Literal::from_f64(0.0); 6],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"shape mismatch for add: left=[2, 3] right=[3, 2]");
}

#[test]
fn snapshot_unsupported_primitive_error() {
    let err = eval_primitive(Primitive::AllGather, &[], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported all_gather behavior: collective operation requires pmap context with multi-device backend");
}

#[test]
fn snapshot_cholesky_non_square_error() {
    let matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            vec![fj_core::Literal::from_f64(1.0); 6],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::Cholesky, &[matrix], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported cholesky behavior: Cholesky requires a square matrix, got 2x3");
}
