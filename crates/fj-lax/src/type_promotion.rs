#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive};

use crate::EvalError;

/// Infer the DType from a slice of Literal elements.
/// Returns I64 if all are I64, Bool if all are Bool, otherwise F64.
#[inline]
pub(crate) fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::{Bool, F32, F64, I32, I64};
    match (lhs, rhs) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (Bool, Bool) => Bool,
    }
}

/// Infer the DType from a slice of Literal elements.
/// Returns I64 if all are I64, Bool if all are Bool, otherwise F64.
#[inline]
pub(crate) fn infer_dtype(elements: &[Literal]) -> DType {
    if elements.is_empty() {
        return DType::F64;
    }
    if elements
        .iter()
        .all(|literal| matches!(literal, Literal::I64(_)))
    {
        DType::I64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Bool(_)))
    {
        DType::Bool
    } else {
        DType::F64
    }
}

/// Apply a binary operation to two literals, dispatching on int vs float.
#[inline]
pub(crate) fn binary_literal_op(
    lhs: Literal,
    rhs: Literal,
    primitive: Primitive,
    int_op: &impl Fn(i64, i64) -> i64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Result<Literal, EvalError> {
    match (lhs, rhs) {
        (Literal::I64(left), Literal::I64(right)) => Ok(Literal::I64(int_op(left, right))),
        (left, right) => {
            let lhs_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs",
            })?;
            let rhs_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs",
            })?;
            Ok(Literal::from_f64(float_op(lhs_f, rhs_f)))
        }
    }
}

/// Compare two literals, dispatching on int vs float.
#[inline]
pub(crate) fn compare_literals(
    lhs: Literal,
    rhs: Literal,
    primitive: Primitive,
    int_cmp: &impl Fn(i64, i64) -> bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<bool, EvalError> {
    match (lhs, rhs) {
        (Literal::I64(a), Literal::I64(b)) => Ok(int_cmp(a, b)),
        (Literal::Bool(a), Literal::Bool(b)) => Ok(int_cmp(a as i64, b as i64)),
        (left, right) => {
            let lhs_f = left.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs for comparison",
            })?;
            let rhs_f = right.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs for comparison",
            })?;
            Ok(float_cmp(lhs_f, rhs_f))
        }
    }
}
