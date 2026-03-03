#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive};

use crate::EvalError;

#[inline]
fn literal_dtype(literal: Literal) -> DType {
    match literal {
        Literal::I64(_) => DType::I64,
        Literal::U32(_) => DType::U32,
        Literal::U64(_) => DType::U64,
        Literal::Bool(_) => DType::Bool,
        Literal::F64Bits(_) => DType::F64,
        Literal::Complex64Bits(..) => DType::Complex64,
        Literal::Complex128Bits(..) => DType::Complex128,
    }
}

#[inline]
fn literal_to_u64(literal: Literal) -> Option<u64> {
    match literal {
        Literal::U32(value) => Some(u64::from(value)),
        Literal::U64(value) => Some(value),
        Literal::I64(value) => u64::try_from(value).ok(),
        Literal::Bool(value) => Some(u64::from(value)),
        _ => None,
    }
}

#[inline]
fn literal_to_i128(literal: Literal) -> Option<i128> {
    match literal {
        Literal::I64(value) => Some(i128::from(value)),
        Literal::U32(value) => Some(i128::from(value)),
        Literal::U64(value) => Some(i128::from(value)),
        Literal::Bool(value) => Some(i128::from(value)),
        _ => None,
    }
}

/// Infer the DType from a slice of Literal elements.
/// Returns I64 if all are I64, Bool if all are Bool, otherwise F64.
#[inline]
pub(crate) fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::{Bool, Complex64, Complex128, F32, F64, I32, I64, U32, U64};
    match (lhs, rhs) {
        (Complex128, _) | (_, Complex128) => Complex128,
        (Complex64, _) | (_, Complex64) => Complex64,
        (U64, I64) | (I64, U64) => F64,
        (U32, F32) | (F32, U32) => F64,
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (I32, U32) | (U32, I32) => I64,
        (I64, U32) | (U32, I64) => I64,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (U64, _) | (_, U64) => U64,
        (U32, _) | (_, U32) => U32,
        (Bool, Bool) => Bool,
    }
}

/// Infer the DType from a slice of Literal elements.
/// Returns I64 if all are I64, Bool if all are Bool, otherwise F64.
#[inline]
#[allow(dead_code)]
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
    let out_dtype = promote_dtype(literal_dtype(lhs), literal_dtype(rhs));

    match out_dtype {
        DType::U32 | DType::U64 => {
            let left = literal_to_u64(lhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected unsigned/integral lhs",
            })?;
            let right = literal_to_u64(rhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected unsigned/integral rhs",
            })?;

            let out = match primitive {
                Primitive::Add => left.wrapping_add(right),
                Primitive::Sub => left.wrapping_sub(right),
                Primitive::Mul => left.wrapping_mul(right),
                Primitive::Div => left.checked_div(right).unwrap_or(0),
                Primitive::Rem => left.checked_rem(right).unwrap_or(0),
                Primitive::Max => left.max(right),
                Primitive::Min => left.min(right),
                Primitive::Pow => left.wrapping_pow(u32::try_from(right).unwrap_or(u32::MAX)),
                _ => {
                    let lhs_f = lhs.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric lhs",
                    })?;
                    let rhs_f = rhs.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric rhs",
                    })?;
                    return Ok(Literal::from_f64(float_op(lhs_f, rhs_f)));
                }
            };

            if out_dtype == DType::U32 {
                Ok(Literal::U32(out as u32))
            } else {
                Ok(Literal::U64(out))
            }
        }
        DType::I64 | DType::I32 => {
            if let (Some(left), Some(right)) = (literal_to_i128(lhs), literal_to_i128(rhs)) {
                let left_i64 = i64::try_from(left).map_err(|_| EvalError::TypeMismatch {
                    primitive,
                    detail: "integral lhs does not fit i64",
                })?;
                let right_i64 = i64::try_from(right).map_err(|_| EvalError::TypeMismatch {
                    primitive,
                    detail: "integral rhs does not fit i64",
                })?;
                Ok(Literal::I64(int_op(left_i64, right_i64)))
            } else {
                let lhs_f = lhs.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric lhs",
                })?;
                let rhs_f = rhs.as_f64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric rhs",
                })?;
                Ok(Literal::from_f64(float_op(lhs_f, rhs_f)))
            }
        }
        _ => {
            let lhs_f = lhs.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs",
            })?;
            let rhs_f = rhs.as_f64().ok_or(EvalError::TypeMismatch {
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
    int_cmp: &impl Fn(i128, i128) -> bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<bool, EvalError> {
    if let (Some(left), Some(right)) = (literal_to_i128(lhs), literal_to_i128(rhs)) {
        return Ok(int_cmp(left, right));
    }

    let lhs_f = lhs.as_f64().ok_or(EvalError::TypeMismatch {
        primitive,
        detail: "expected numeric lhs for comparison",
    })?;
    let rhs_f = rhs.as_f64().ok_or(EvalError::TypeMismatch {
        primitive,
        detail: "expected numeric rhs for comparison",
    })?;
    Ok(float_cmp(lhs_f, rhs_f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_promotion_i32_u32() {
        assert_eq!(promote_dtype(DType::I32, DType::U32), DType::I64);
        assert_eq!(promote_dtype(DType::U32, DType::I32), DType::I64);
    }

    #[test]
    fn test_type_promotion_u32_f32() {
        assert_eq!(promote_dtype(DType::U32, DType::F32), DType::F64);
        assert_eq!(promote_dtype(DType::F32, DType::U32), DType::F64);
    }

    #[test]
    fn test_type_promotion_u64_f64() {
        assert_eq!(promote_dtype(DType::U64, DType::F64), DType::F64);
        assert_eq!(promote_dtype(DType::F64, DType::U64), DType::F64);
    }

    #[test]
    fn test_type_promotion_matrix_unsigned() {
        assert_eq!(promote_dtype(DType::Bool, DType::U32), DType::U32);
        assert_eq!(promote_dtype(DType::U32, DType::U64), DType::U64);
        assert_eq!(promote_dtype(DType::U64, DType::I64), DType::F64);
        assert_eq!(promote_dtype(DType::U32, DType::I64), DType::I64);
    }
}
