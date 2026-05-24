//! JAX oracle parity for `lax.dynamic_update_slice` start-index clamping and
//! general semantics.
//!
//! Reference outputs were captured with:
//! `uv run --with 'jax[cpu]' python ...`
//! using JAX 0.10.0 with `jax_enable_x64 = True`.

#![allow(clippy::useless_vec)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

struct DynamicUpdateSliceOracleCase {
    case_id: &'static str,
    operand_shape: &'static [u32],
    operand_values: &'static [i64],
    update_shape: &'static [u32],
    update_values: &'static [i64],
    starts: &'static [i64],
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

fn dynamic_update_slice_cases() -> Vec<DynamicUpdateSliceOracleCase> {
    vec![
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank1_positive_start_clamps_to_last_valid_window",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            update_shape: &[3],
            update_values: &[70, 71, 72],
            starts: &[10],
            expected_values: &[0, 1, 2, 70, 71, 72],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank1_negative_start_is_relative_then_clamped",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            update_shape: &[3],
            update_values: &[70, 71, 72],
            starts: &[-1],
            expected_values: &[0, 1, 2, 70, 71, 72],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank1_negative_start_inside_valid_window",
            operand_shape: &[6],
            operand_values: &[0, 1, 2, 3, 4, 5],
            update_shape: &[3],
            update_values: &[70, 71, 72],
            starts: &[-5],
            expected_values: &[0, 70, 71, 72, 4, 5],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank2_positive_starts_clamp_per_axis",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            update_shape: &[2, 2],
            update_values: &[90, 91, 92, 93],
            starts: &[2, 3],
            expected_values: &[0, 1, 2, 3, 4, 5, 90, 91, 8, 9, 92, 93],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank2_negative_starts_clamp_per_axis",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            update_shape: &[2, 2],
            update_values: &[90, 91, 92, 93],
            starts: &[-1, -1],
            expected_values: &[0, 1, 2, 3, 4, 5, 90, 91, 8, 9, 92, 93],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank2_mixed_positive_and_negative_starts",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            update_shape: &[2, 2],
            update_values: &[90, 91, 92, 93],
            starts: &[1, -5],
            expected_values: &[0, 1, 2, 3, 90, 91, 6, 7, 92, 93, 10, 11],
        },
        DynamicUpdateSliceOracleCase {
            case_id: "jax_rank2_too_negative_start_clamps_low",
            operand_shape: &[3, 4],
            operand_values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            update_shape: &[2, 2],
            update_values: &[90, 91, 92, 93],
            starts: &[-4, 2],
            expected_values: &[0, 1, 90, 91, 4, 5, 92, 93, 8, 9, 10, 11],
        },
    ]
}

#[test]
fn dynamic_update_slice_start_clamping_matches_jax_reference() -> Result<(), String> {
    for case in dynamic_update_slice_cases() {
        let mut inputs = vec![
            tensor_i64(case.operand_shape, case.operand_values)?,
            tensor_i64(case.update_shape, case.update_values)?,
        ];
        inputs.extend(case.starts.iter().copied().map(Value::scalar_i64));

        let actual = eval_primitive(Primitive::DynamicUpdateSlice, &inputs, &BTreeMap::new())
            .map_err(|err| format!("{}: evaluation failed: {err}", case.case_id))?;
        let (actual_shape, actual_values) = tensor_i64_parts(&actual)?;

        assert_eq!(
            actual_shape, case.operand_shape,
            "{}: output shape must match operand shape",
            case.case_id
        );
        assert_eq!(
            actual_values, case.expected_values,
            "{}: FrankenJAX diverged from JAX dynamic_update_slice",
            case.case_id
        );
    }

    Ok(())
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
fn dynamic_update_slice_start_at_zero() {
    let operand = tensor_i64(&[5], &[0, 1, 2, 3, 4]).unwrap();
    let update = tensor_i64(&[2], &[90, 91]).unwrap();
    let starts = vec![Value::scalar_i64(0)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![90, 91, 2, 3, 4]);
}

#[test]
fn dynamic_update_slice_full_replacement() {
    let operand = tensor_i64(&[3], &[1, 2, 3]).unwrap();
    let update = tensor_i64(&[3], &[7, 8, 9]).unwrap();
    let starts = vec![Value::scalar_i64(0)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![7, 8, 9]);
}

#[test]
fn dynamic_update_slice_single_element() {
    let operand = tensor_i64(&[5], &[0, 1, 2, 3, 4]).unwrap();
    let update = tensor_i64(&[1], &[99]).unwrap();
    let starts = vec![Value::scalar_i64(2)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![0, 1, 99, 3, 4]);
}

#[test]
fn dynamic_update_slice_f64_dtype() {
    let operand = tensor_f64(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let update = tensor_f64(&[2], &[10.5, 11.5]).unwrap();
    let starts = vec![Value::scalar_i64(1)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let values = tensor_f64_values(&result).unwrap();
    assert!((values[0] - 1.0).abs() < 1e-10);
    assert!((values[1] - 10.5).abs() < 1e-10);
    assert!((values[2] - 11.5).abs() < 1e-10);
    assert!((values[3] - 4.0).abs() < 1e-10);
}

#[test]
fn dynamic_update_slice_rank3() {
    // [2, 2, 2] operand, update [1, 1, 2] at position [1, 0, 0]
    let operand = tensor_i64(&[2, 2, 2], &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    let update = tensor_i64(&[1, 1, 2], &[90, 91]).unwrap();
    let starts = vec![
        Value::scalar_i64(1),
        Value::scalar_i64(0),
        Value::scalar_i64(0),
    ];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[
            operand,
            update,
            starts[0].clone(),
            starts[1].clone(),
            starts[2].clone(),
        ],
        &BTreeMap::new(),
    )
    .unwrap();

    let (shape, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![2, 2, 2]);
    assert_eq!(values, vec![0, 1, 2, 3, 90, 91, 6, 7]);
}

#[test]
fn dynamic_update_slice_preserves_untouched_elements() {
    let operand = tensor_i64(&[2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let update = tensor_i64(&[1, 2], &[90, 91]).unwrap();
    let starts = vec![Value::scalar_i64(0), Value::scalar_i64(1)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone(), starts[1].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    // Row 0: [1, 90, 91, 4], Row 1 unchanged: [5, 6, 7, 8]
    assert_eq!(values, vec![1, 90, 91, 4, 5, 6, 7, 8]);
}

#[test]
fn dynamic_update_slice_at_end() {
    let operand = tensor_i64(&[5], &[0, 1, 2, 3, 4]).unwrap();
    let update = tensor_i64(&[2], &[88, 99]).unwrap();
    let starts = vec![Value::scalar_i64(3)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![0, 1, 2, 88, 99]);
}

#[test]
fn dynamic_update_slice_preserves_dtype() {
    let operand = tensor_i64(&[4], &[1, 2, 3, 4]).unwrap();
    let update = tensor_i64(&[2], &[10, 20]).unwrap();
    let starts = vec![Value::scalar_i64(1)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    match result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64, "dtype should be preserved"),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn dynamic_update_slice_metamorphic_sequential_updates() {
    // Two sequential updates: first at index 0, then at index 2
    let operand = tensor_i64(&[5], &[0, 1, 2, 3, 4]).unwrap();
    let update1 = tensor_i64(&[2], &[10, 11]).unwrap();
    let update2 = tensor_i64(&[2], &[20, 21]).unwrap();

    let result1 = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update1, Value::scalar_i64(0)],
        &BTreeMap::new(),
    )
    .unwrap();

    let result2 = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[result1, update2, Value::scalar_i64(2)],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result2).unwrap();
    assert_eq!(values, vec![10, 11, 20, 21, 4]);
}

#[test]
fn dynamic_update_slice_middle_update_2d() {
    // Update center of a 3x3 matrix
    let operand = tensor_i64(&[3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    let update = tensor_i64(&[1, 1], &[99]).unwrap();
    let starts = vec![Value::scalar_i64(1), Value::scalar_i64(1)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone(), starts[1].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![1, 2, 3, 4, 99, 6, 7, 8, 9]);
}

#[test]
fn dynamic_update_slice_rank4() {
    // [2, 2, 2, 2] operand, update [1, 1, 1, 2] at various positions
    let operand_values: Vec<i64> = (0..16).collect();
    let operand = tensor_i64(&[2, 2, 2, 2], &operand_values).unwrap();
    let update = tensor_i64(&[1, 1, 1, 2], &[90, 91]).unwrap();
    let starts = vec![
        Value::scalar_i64(0),
        Value::scalar_i64(1),
        Value::scalar_i64(1),
        Value::scalar_i64(0),
    ];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[
            operand,
            update,
            starts[0].clone(),
            starts[1].clone(),
            starts[2].clone(),
            starts[3].clone(),
        ],
        &BTreeMap::new(),
    )
    .unwrap();

    let (shape, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(shape, vec![2, 2, 2, 2]);
    // Position [0, 1, 1, 0..2] = indices 6 and 7
    let mut expected: Vec<i64> = (0..16).collect();
    expected[6] = 90;
    expected[7] = 91;
    assert_eq!(values, expected);
}

#[test]
fn dynamic_update_slice_bool_dtype() {
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

    let operand = tensor_bool(&[5], &[false, false, false, false, false]).unwrap();
    let update = tensor_bool(&[2], &[true, true]).unwrap();
    let starts = vec![Value::scalar_i64(1)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

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
    assert_eq!(values, vec![false, true, true, false, false]);
}

#[test]
fn dynamic_update_slice_noop_zero_size_update() {
    // Update with zero-size tensor should be a no-op
    let operand = tensor_i64(&[5], &[1, 2, 3, 4, 5]).unwrap();
    let update = tensor_i64(&[0], &[]).unwrap();
    let starts = vec![Value::scalar_i64(2)];

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, starts[0].clone()],
        &BTreeMap::new(),
    )
    .unwrap();

    let (_, values) = tensor_i64_parts(&result).unwrap();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn dynamic_update_slice_scalar_operand() {
    // Scalar update on scalar operand
    let operand = tensor_i64(&[], &[42]).unwrap();
    let update = tensor_i64(&[], &[99]).unwrap();
    // No starts needed for scalar

    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update],
        &BTreeMap::new(),
    )
    .unwrap();

    let (shape, values) = tensor_i64_parts(&result).unwrap();
    assert!(shape.is_empty(), "scalar should have empty shape");
    assert_eq!(values, vec![99]);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_dynamic_update_slice_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, shape: &[u32], values: &[f64]) -> Value {
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
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape {
                    dims: shape.to_vec(),
                },
                lits,
            )
            .unwrap(),
        )
    }

    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let operand = make_vec(dtype, &[4], &[1.0, 2.0, 3.0, 4.0]);
        let update = make_vec(dtype, &[2], &[10.0, 20.0]);
        let result = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(1)],
            &BTreeMap::new(),
        )
        .unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(
            t.dtype, dtype,
            "dynamic_update_slice {dtype:?}: dtype mismatch"
        );
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex64/Complex128 Tests ========================

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

fn extract_complex64_vec(v: &Value) -> Vec<(f32, f32)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex64().unwrap())
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_complex128_vec(v: &Value) -> Vec<(f64, f64)> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn dynamic_update_slice_complex64_1d_middle() {
    let operand = make_complex64_tensor(
        &[5],
        vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)],
    );
    let update = make_complex64_tensor(&[2], vec![(10.0, 10.0), (20.0, 20.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(1)],
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 0.0),
            (10.0, 10.0),
            (20.0, 20.0),
            (4.0, 0.0),
            (5.0, 0.0),
        ]
    );
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn dynamic_update_slice_complex64_1d_start() {
    let operand = make_complex64_tensor(&[4], vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]);
    let update = make_complex64_tensor(&[2], vec![(10.0, -10.0), (20.0, -20.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(0)],
        &BTreeMap::new(),
    )
    .unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![(10.0, -10.0), (20.0, -20.0), (3.0, 3.0), (4.0, 4.0),]
    );
}

#[test]
fn dynamic_update_slice_complex64_1d_end() {
    let operand = make_complex64_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
    let update = make_complex64_tensor(&[2], vec![(30.0, 30.0), (40.0, 40.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(2)],
        &BTreeMap::new(),
    )
    .unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![(1.0, 0.0), (2.0, 0.0), (30.0, 30.0), (40.0, 40.0),]
    );
}

#[test]
fn dynamic_update_slice_complex64_2d() {
    let operand = make_complex64_tensor(
        &[3, 3],
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0),
            (8.0, 0.0),
            (9.0, 0.0),
        ],
    );
    let update = make_complex64_tensor(
        &[2, 2],
        vec![(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)],
    );
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(1), Value::scalar_i64(1)],
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    let vals = extract_complex64_vec(&result);
    assert_eq!(
        vals,
        vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (10.0, 10.0),
            (20.0, 20.0),
            (7.0, 0.0),
            (30.0, 30.0),
            (40.0, 40.0),
        ]
    );
}

#[test]
fn dynamic_update_slice_complex128_1d() {
    let operand =
        make_complex128_tensor(&[4], vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]);
    let update = make_complex128_tensor(&[2], vec![(10.0, -10.0), (20.0, -20.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(1)],
        &BTreeMap::new(),
    )
    .unwrap();
    let vals = extract_complex128_vec(&result);
    assert_eq!(
        vals,
        vec![(1.0, 1.0), (10.0, -10.0), (20.0, -20.0), (4.0, 4.0),]
    );
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn dynamic_update_slice_complex64_full_replace() {
    let operand = make_complex64_tensor(&[3], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
    let update = make_complex64_tensor(&[3], vec![(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(0)],
        &BTreeMap::new(),
    )
    .unwrap();
    let vals = extract_complex64_vec(&result);
    assert_eq!(vals, vec![(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]);
}

#[test]
fn dynamic_update_slice_complex64_preserves_dtype() {
    let operand = make_complex64_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
    let update = make_complex64_tensor(&[2], vec![(10.0, 0.0), (20.0, 0.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(1)],
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn dynamic_update_slice_complex128_preserves_dtype() {
    let operand =
        make_complex128_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
    let update = make_complex128_tensor(&[2], vec![(10.0, 0.0), (20.0, 0.0)]);
    let result = eval_primitive(
        Primitive::DynamicUpdateSlice,
        &[operand, update, Value::scalar_i64(1)],
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn property_dynamic_update_slice_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (operand, update) = match dtype {
            DType::Complex64 => (
                make_complex64_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]),
                make_complex64_tensor(&[2], vec![(10.0, 0.0), (20.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_tensor(&[4], vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]),
                make_complex128_tensor(&[2], vec![(10.0, 0.0), (20.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(1)],
            &BTreeMap::new(),
        )
        .unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(
            t.dtype, dtype,
            "dynamic_update_slice {dtype:?}: dtype mismatch"
        );
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
