//! JAX oracle parity for `lax.dynamic_slice` start-index clamping.
//!
//! Reference outputs were captured from JAX dynamic_slice semantics:
//! negative start indices are interpreted relative to the operand dimension and
//! then clamped to the valid `[0, dim - slice_size]` window on each axis.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

struct DynamicSliceOracleCase {
    case_id: &'static str,
    operand_shape: &'static [u32],
    operand_values: &'static [i64],
    slice_sizes: &'static [usize],
    starts: &'static [i64],
    expected_shape: &'static [u32],
    expected_values: &'static [i64],
}

fn tensor_i64(shape: &[u32], values: &[i64]) -> Result<Value, String> {
    TensorValue::new(
        DType::I64,
        Shape {
            dims: shape.to_vec(),
        },
        values.iter().copied().map(Literal::I64).collect(),
    )
    .map(Value::Tensor)
    .map_err(|err| format!("failed to build i64 tensor {shape:?}: {err}"))
}

fn tensor_i64_parts(value: &Value) -> Result<(Vec<u32>, Vec<i64>), String> {
    let Value::Tensor(tensor) = value else {
        return Err(format!("expected tensor, got {value:?}"));
    };

    let values = tensor
        .elements
        .iter()
        .map(|literal| match literal {
            Literal::I64(value) => Ok(*value),
            other => Err(format!("expected i64 literal, got {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tensor.shape.dims.clone(), values))
}

fn slice_sizes_param(slice_sizes: &[usize]) -> String {
    slice_sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn dynamic_slice_cases() -> Vec<DynamicSliceOracleCase> {
    vec![
        DynamicSliceOracleCase {
            case_id: "jax_rank1_positive_start_clamps_to_last_valid_window",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            slice_sizes: &[3],
            starts: &[10],
            expected_shape: &[3],
            expected_values: &[3, 4, 5],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank1_negative_start_is_relative_then_clamped",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            slice_sizes: &[3],
            starts: &[-1],
            expected_shape: &[3],
            expected_values: &[3, 4, 5],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank1_negative_start_inside_valid_window",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            slice_sizes: &[3],
            starts: &[-5],
            expected_shape: &[3],
            expected_values: &[1, 2, 3],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank2_positive_starts_clamp_per_axis",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            slice_sizes: &[2, 2],
            starts: &[2, 3],
            expected_shape: &[2, 2],
            expected_values: &[6, 7, 10, 11],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank2_negative_starts_clamp_per_axis",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            slice_sizes: &[2, 2],
            starts: &[-1, -1],
            expected_shape: &[2, 2],
            expected_values: &[6, 7, 10, 11],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank2_mixed_positive_and_too_negative_starts",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            slice_sizes: &[2, 2],
            starts: &[1, -5],
            expected_shape: &[2, 2],
            expected_values: &[4, 5, 8, 9],
        },
        DynamicSliceOracleCase {
            case_id: "jax_rank2_too_negative_start_clamps_low",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            slice_sizes: &[2, 2],
            starts: &[-4, 2],
            expected_shape: &[2, 2],
            expected_values: &[2, 3, 6, 7],
        },
    ]
}

#[test]
fn dynamic_slice_start_clamping_matches_jax_reference() -> Result<(), String> {
    for case in dynamic_slice_cases() {
        let mut inputs = vec![tensor_i64(case.operand_shape, case.operand_values)?];
        inputs.extend(case.starts.iter().copied().map(Value::scalar_i64));

        let mut params = BTreeMap::new();
        params.insert(
            "slice_sizes".to_owned(),
            slice_sizes_param(case.slice_sizes),
        );

        let actual = eval_primitive(Primitive::DynamicSlice, &inputs, &params)
            .map_err(|err| format!("{}: evaluation failed: {err}", case.case_id))?;
        let (actual_shape, actual_values) = tensor_i64_parts(&actual)?;

        assert_eq!(
            actual_shape, case.expected_shape,
            "{}: output shape must match JAX reference",
            case.case_id
        );
        assert_eq!(
            actual_values, case.expected_values,
            "{}: FrankenJAX diverged from JAX dynamic_slice",
            case.case_id
        );
    }

    Ok(())
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_dynamic_slice_full_is_identity() {
    // DynamicSlice with start=0 and slice_sizes=shape returns original
    let operand = tensor_i64(&[4], &[10, 20, 30, 40]).unwrap();
    let start = Value::scalar_i64(0);
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "4".to_owned());

    let result = eval_primitive(Primitive::DynamicSlice, &[operand, start], &params).unwrap();
    let (_, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(vals, vec![10, 20, 30, 40], "full slice should be identity");
}

#[test]
fn metamorphic_dynamic_slice_single_element() {
    // DynamicSlice of size 1 at index i extracts element i
    let operand = tensor_i64(&[5], &[100, 200, 300, 400, 500]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "1".to_owned());

    for i in 0..5i64 {
        let start = Value::scalar_i64(i);
        let result =
            eval_primitive(Primitive::DynamicSlice, &[operand.clone(), start], &params).unwrap();
        let (_, vals) = tensor_i64_parts(&result).unwrap();
        assert_eq!(
            vals,
            vec![(i + 1) * 100],
            "single element slice at {i} should be element {i}"
        );
    }
}

#[test]
fn metamorphic_dynamic_slice_adjacent_cover() {
    // Two adjacent slices of size 2 should cover all 4 elements
    let operand = tensor_i64(&[4], &[1, 2, 3, 4]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2".to_owned());

    let slice0 =
        eval_primitive(Primitive::DynamicSlice, &[operand.clone(), Value::scalar_i64(0)], &params)
            .unwrap();
    let slice1 =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(2)], &params).unwrap();

    let (_, v0) = tensor_i64_parts(&slice0).unwrap();
    let (_, v1) = tensor_i64_parts(&slice1).unwrap();

    let mut combined: Vec<i64> = v0;
    combined.extend(v1);
    assert_eq!(combined, vec![1, 2, 3, 4], "adjacent slices should cover all");
}

#[test]
fn metamorphic_dynamic_slice_preserves_dtype() {
    let operand = tensor_i64(&[3], &[7, 8, 9]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(0)], &params).unwrap();
    match result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64, "dtype should be preserved"),
        _ => panic!("expected tensor"),
    }
}

// ======================== Additional Coverage ========================

fn tensor_f64(shape: &[u32], values: &[f64]) -> Result<Value, String> {
    TensorValue::new(
        DType::F64,
        Shape {
            dims: shape.to_vec(),
        },
        values.iter().copied().map(Literal::from_f64).collect(),
    )
    .map(Value::Tensor)
    .map_err(|err| format!("failed to build f64 tensor {shape:?}: {err}"))
}

fn tensor_f64_values(value: &Value) -> Result<Vec<f64>, String> {
    let Value::Tensor(tensor) = value else {
        return Err(format!("expected tensor, got {value:?}"));
    };
    tensor
        .elements
        .iter()
        .map(|literal| match literal {
            Literal::F64Bits(bits) => Ok(f64::from_bits(*bits)),
            other => Err(format!("expected f64 literal, got {other:?}")),
        })
        .collect()
}

#[test]
fn dynamic_slice_f64_dtype() {
    let operand = tensor_f64(&[4], &[1.5, 2.5, 3.5, 4.5]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(1)], &params).unwrap();
    let vals = tensor_f64_values(&result).unwrap();
    assert!((vals[0] - 2.5).abs() < 1e-10);
    assert!((vals[1] - 3.5).abs() < 1e-10);
}

#[test]
fn dynamic_slice_2d_both_axes() {
    // Slice [2,2] from [3,4] starting at [1,1]
    let operand = tensor_i64(&[3, 4], &(0..12).collect::<Vec<_>>()).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2,2".to_owned());

    let result = eval_primitive(
        Primitive::DynamicSlice,
        &[operand, Value::scalar_i64(1), Value::scalar_i64(1)],
        &params,
    )
    .unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![2, 2]);
    // Row 1 cols 1,2: [5, 6], Row 2 cols 1,2: [9, 10]
    assert_eq!(vals, vec![5, 6, 9, 10]);
}

#[test]
fn dynamic_slice_empty_output() {
    // Slice with size 0
    let operand = tensor_i64(&[5], &[1, 2, 3, 4, 5]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "0".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(0)], &params).unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![0]);
    assert!(vals.is_empty());
}

#[test]
fn dynamic_slice_full_slice() {
    let operand = tensor_i64(&[4], &[10, 20, 30, 40]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "4".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(0)], &params).unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![4]);
    assert_eq!(vals, vec![10, 20, 30, 40]);
}

#[test]
fn dynamic_slice_preserves_dtype() {
    let operand = tensor_i64(&[3], &[1, 2, 3]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(0)], &params).unwrap();
    match result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn dynamic_slice_single_element() {
    let operand = tensor_i64(&[5], &[100, 200, 300, 400, 500]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "1".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(2)], &params).unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![1]);
    assert_eq!(vals, vec![300]);
}

#[test]
fn dynamic_slice_rank3() {
    let operand = tensor_i64(&[2, 3, 4], &(0..24).collect::<Vec<_>>()).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "1,2,2".to_owned());

    let result = eval_primitive(
        Primitive::DynamicSlice,
        &[operand, Value::scalar_i64(1), Value::scalar_i64(0), Value::scalar_i64(1)],
        &params,
    )
    .unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![1, 2, 2]);
    assert_eq!(vals, vec![13, 14, 17, 18]);
}

#[test]
fn dynamic_slice_rank4() {
    let operand = tensor_i64(&[2, 2, 2, 2], &(0..16).collect::<Vec<_>>()).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "1,1,2,2".to_owned());

    let result = eval_primitive(
        Primitive::DynamicSlice,
        &[
            operand,
            Value::scalar_i64(1),
            Value::scalar_i64(0),
            Value::scalar_i64(0),
            Value::scalar_i64(0),
        ],
        &params,
    )
    .unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![1, 1, 2, 2]);
    assert_eq!(vals, vec![8, 9, 10, 11]);
}

#[test]
fn dynamic_slice_bool_dtype() {
    fn tensor_bool(shape: &[u32], values: &[bool]) -> Result<Value, String> {
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            values.iter().copied().map(Literal::Bool).collect(),
        )
        .map(Value::Tensor)
        .map_err(|err| format!("failed to build bool tensor: {err}"))
    }

    let operand = tensor_bool(&[5], &[true, false, true, false, true]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "3".to_owned());

    let result =
        eval_primitive(Primitive::DynamicSlice, &[operand, Value::scalar_i64(1)], &params).unwrap();
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.dtype, DType::Bool);
    let values: Vec<bool> = tensor
        .elements
        .iter()
        .map(|l| match l {
            Literal::Bool(b) => *b,
            _ => panic!("expected bool"),
        })
        .collect();
    assert_eq!(values, vec![false, true, false]);
}

#[test]
fn dynamic_slice_2d_empty_output() {
    // Slice with size 0 on one dimension
    let operand = tensor_i64(&[3, 4], &(0..12).collect::<Vec<_>>()).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2,0".to_owned());

    let result = eval_primitive(
        Primitive::DynamicSlice,
        &[operand, Value::scalar_i64(0), Value::scalar_i64(0)],
        &params,
    )
    .unwrap();
    let (shape, vals) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![2, 0]);
    assert!(vals.is_empty());
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_dynamic_slice_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![4] }, lits).unwrap())
    }

    let values = [1.0_f64, 2.0, 3.0, 4.0];
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), "2".to_owned());

    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(
            Primitive::DynamicSlice,
            &[input, Value::scalar_i64(1)],
            &params,
        )
        .unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "dynamic_slice {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
