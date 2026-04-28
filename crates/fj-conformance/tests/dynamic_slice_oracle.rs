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
