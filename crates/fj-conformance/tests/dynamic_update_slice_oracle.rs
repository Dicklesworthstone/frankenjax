//! JAX oracle parity for `lax.dynamic_update_slice` start-index clamping.
//!
//! Reference outputs were captured with:
//! `uv run --with 'jax[cpu]' python ...`
//! using JAX 0.10.0 with `jax_enable_x64 = True`.

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
