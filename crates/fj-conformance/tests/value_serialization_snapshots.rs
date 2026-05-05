//! Golden artifact tests for Value JSON serialization format.
//!
//! These tests freeze the JSON serialization format for Value types.
//! Unintentional changes to serialization format would break:
//! - Cached computation results
//! - Inter-process communication
//! - Persistent storage

use fj_core::{DType, Literal, Shape, TensorValue, Value};

// ======================== Scalar Snapshots ========================

#[test]
fn snapshot_scalar_f64() {
    let val = Value::scalar_f64(3.14159);
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn snapshot_scalar_i64() {
    let val = Value::Scalar(Literal::I64(42));
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Scalar": {
        "I64": 42
      }
    }
    "###);
}

#[test]
fn snapshot_scalar_bool() {
    let val = Value::Scalar(Literal::Bool(true));
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Scalar": {
        "Bool": true
      }
    }
    "###);
}

#[test]
fn snapshot_scalar_complex() {
    let val = Value::Scalar(Literal::from_complex128(1.0, -2.0));
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json);
}

// ======================== Tensor Snapshots ========================

#[test]
fn snapshot_tensor_1d_f64() {
    let val = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(2.0),
                Literal::from_f64(3.0),
            ],
        )
        .unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn snapshot_tensor_2d_i64() {
    let val = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::I64(1),
                Literal::I64(2),
                Literal::I64(3),
                Literal::I64(4),
            ],
        )
        .unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Tensor": {
        "dtype": "I64",
        "shape": {
          "dims": [
            2,
            2
          ]
        },
        "elements": [
          {
            "I64": 1
          },
          {
            "I64": 2
          },
          {
            "I64": 3
          },
          {
            "I64": 4
          }
        ]
      }
    }
    "###);
}

#[test]
fn snapshot_tensor_scalar_shape() {
    let val = Value::Tensor(
        TensorValue::new(DType::F32, Shape { dims: vec![] }, vec![Literal::from_f32(1.5)]).unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json);
}

// ======================== Special Values ========================

#[test]
fn snapshot_tensor_special_f64() {
    let val = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f64(f64::INFINITY),
                Literal::from_f64(f64::NEG_INFINITY),
                Literal::from_f64(f64::NAN),
            ],
        )
        .unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json);
}

// ======================== Empty Tensor ========================

#[test]
fn snapshot_tensor_empty() {
    let val = Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Tensor": {
        "dtype": "F64",
        "shape": {
          "dims": [
            0
          ]
        },
        "elements": []
      }
    }
    "###);
}

// ======================== High-Rank Tensor ========================

#[test]
fn snapshot_tensor_3d() {
    let val = Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape {
                dims: vec![2, 1, 2],
            },
            vec![
                Literal::U32(1),
                Literal::U32(2),
                Literal::U32(3),
                Literal::U32(4),
            ],
        )
        .unwrap(),
    );
    let json = serde_json::to_string_pretty(&val).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Tensor": {
        "dtype": "U32",
        "shape": {
          "dims": [
            2,
            1,
            2
          ]
        },
        "elements": [
          {
            "U32": 1
          },
          {
            "U32": 2
          },
          {
            "U32": 3
          },
          {
            "U32": 4
          }
        ]
      }
    }
    "###);
}
