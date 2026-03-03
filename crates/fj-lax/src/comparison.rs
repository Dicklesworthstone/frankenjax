#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::compare_literals;

/// Comparison operators: return Bool scalars/tensors.
#[inline]
pub(crate) fn eval_comparison(
    primitive: Primitive,
    inputs: &[Value],
    int_cmp: impl Fn(i128, i128) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            let result = compare_literals(*lhs, *rhs, primitive, &int_cmp, &float_cmp)?;
            Ok(Value::scalar_bool(result))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape != rhs.shape {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }
            let elements = lhs
                .elements
                .iter()
                .copied()
                .zip(rhs.elements.iter().copied())
                .map(|(l, r)| {
                    compare_literals(l, r, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            let elements = rhs
                .elements
                .iter()
                .copied()
                .map(|r| {
                    compare_literals(*lhs, r, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            let elements = lhs
                .elements
                .iter()
                .copied()
                .map(|l| {
                    compare_literals(l, *rhs, primitive, &int_cmp, &float_cmp).map(Literal::Bool)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}
