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

// ====================== Additional Error Variants ======================

#[test]
fn snapshot_invalid_tensor_empty_with_nonzero_shape() {
    let err = TensorValue::new(
        DType::F64,
        Shape { dims: vec![2, 3] },
        vec![], // empty elements but shape says 6 elements
    )
    .unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"tensor element count mismatch for shape [2, 3]: expected 6, got 0");
}

#[test]
fn snapshot_invalid_tensor_count_mismatch() {
    let err = TensorValue::new(
        DType::F64,
        Shape { dims: vec![2, 2] },
        vec![fj_core::Literal::from_f64(1.0); 5], // 5 elements but shape is 2x2=4
    )
    .unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"tensor element count mismatch for shape [2, 2]: expected 4, got 5");
}

#[test]
fn snapshot_qr_rank_too_low_error() {
    // QR requires at least rank 2
    let vector = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![4] },
            vec![fj_core::Literal::from_f64(1.0); 4],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::Qr, &[vector], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported qr behavior: expected rank-2 tensor (matrix), got rank-1");
}

#[test]
fn snapshot_svd_rank_too_low_error() {
    // SVD requires at least rank 2
    let vector = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![4] },
            vec![fj_core::Literal::from_f64(1.0); 4],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::Svd, &[vector], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported svd behavior: expected rank-2 tensor (matrix), got rank-1");
}

#[test]
fn snapshot_eigh_non_square_error() {
    // Eigh requires square matrix
    let matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            vec![fj_core::Literal::from_f64(1.0); 6],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::Eigh, &[matrix], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported eigh behavior: Eigh requires a square matrix, got 2x3");
}

#[test]
fn snapshot_triangular_solve_shape_error() {
    // TriangularSolve needs compatible shapes
    let a = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![fj_core::Literal::from_f64(1.0); 4],
        )
        .unwrap(),
    );
    let b = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 1] }, // incompatible with 2x2
            vec![fj_core::Literal::from_f64(1.0); 3],
        )
        .unwrap(),
    );
    let err = eval_primitive(Primitive::TriangularSolve, &[a, b], &no_params()).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"shape mismatch for triangular_solve: left=[2, 2] right=[3, 1]");
}

#[test]
fn snapshot_reduce_window_invalid_padding() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![4] },
            vec![fj_core::Literal::from_f64(1.0); 4],
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("window_dimensions".to_string(), "2".to_string());
    params.insert("padding".to_string(), "INVALID_PADDING".to_string());
    let err = eval_primitive(Primitive::ReduceWindow, &[input], &params).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @r###"unsupported reduce_window behavior: unsupported reduce_window padding mode "INVALID_PADDING""###);
}

#[test]
fn snapshot_cumsum_axis_out_of_bounds() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![4] },
            vec![fj_core::Literal::I64(1); 4],
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), "5".to_string()); // rank is 1, axis 5 is invalid
    let err = eval_primitive(Primitive::Cumsum, &[input], &params).unwrap_err();
    insta::assert_snapshot!(err.to_string(), @"unsupported cumsum behavior: axis 5 out of bounds for rank 1");
}
