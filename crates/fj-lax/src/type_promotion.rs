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
        Literal::BF16Bits(_) => DType::BF16,
        Literal::F16Bits(_) => DType::F16,
        Literal::F32Bits(_) => DType::F32,
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

#[inline]
fn literal_to_numeric_f64(literal: Literal) -> Option<f64> {
    match literal {
        Literal::Bool(value) => Some(f64::from(u8::from(value))),
        _ => literal.as_f64(),
    }
}

#[inline]
fn literal_from_numeric_f64(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f32(value as f32),
        DType::F16 => Literal::from_f16_f32(value as f32),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

/// Infer the DType from a slice of Literal elements.
/// Returns I64 if all are I64, Bool if all are Bool, otherwise F64.
#[inline]
pub fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::{BF16, Bool, Complex64, Complex128, F16, F32, F64, I32, I64, U32, U64};
    // JAX type promotion lattice (jax.numpy.promote_types):
    // - Half-precision types (BF16, F16) absorb integer and boolean types
    // - BF16 + F16 promotes to F32 (cross-half promotion)
    // - Same half type stays: BF16+BF16→BF16, F16+F16→F16
    // - F32 absorbs half types; F64 absorbs everything
    // - U64+I64 → F64 (no common integer type)
    // - U32+F32 → F32 (JAX lattice; F32 absorbs U32)
    match (lhs, rhs) {
        // Complex128 absorbs everything
        (Complex128, _) | (_, Complex128) => Complex128,
        // Complex64 + types that promote to f64 → Complex128
        // JAX: complex64 + float64 → complex128, complex64 + int64 → complex128,
        //       complex64 + uint64 → complex128
        (Complex64, F64 | I64 | U64) | (F64 | I64 | U64, Complex64) => Complex128,
        // Complex64 + types that stay within f32 → Complex64
        (Complex64, _) | (_, Complex64) => Complex64,
        // F64 absorbs everything
        (F64, _) | (_, F64) => F64,
        // F32 absorbs everything below it (JAX lattice: U32+F32→F32)
        (F32, _) | (_, F32) => F32,
        // Cross-half promotion
        (BF16, F16) | (F16, BF16) => F32,
        // Same half type stays
        (BF16, BF16) => BF16,
        (F16, F16) => F16,
        // Half types absorb integers and booleans
        (BF16, Bool | I32 | I64 | U32 | U64) | (Bool | I32 | I64 | U32 | U64, BF16) => BF16,
        (F16, Bool | I32 | I64 | U32 | U64) | (Bool | I32 | I64 | U32 | U64, F16) => F16,
        // Integer promotion
        (U64, I64) | (I64, U64) => F64,
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
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::F32Bits(_)))
    {
        DType::F32
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
        DType::Bool => {
            let left = literal_to_i128(lhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected boolean lhs",
            })? != 0;
            let right = literal_to_i128(rhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected boolean rhs",
            })? != 0;

            let out = match primitive {
                Primitive::Add | Primitive::Max => left || right,
                Primitive::Mul | Primitive::Min => left && right,
                _ => {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "unsupported bool/bool binary operation",
                    });
                }
            };
            Ok(Literal::Bool(out))
        }
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
                    let lhs_f = literal_to_numeric_f64(lhs).ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric lhs",
                    })?;
                    let rhs_f = literal_to_numeric_f64(rhs).ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric rhs",
                    })?;
                    return Ok(literal_from_numeric_f64(out_dtype, float_op(lhs_f, rhs_f)));
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
                let lhs_f = literal_to_numeric_f64(lhs).ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric lhs",
                })?;
                let rhs_f = literal_to_numeric_f64(rhs).ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric rhs",
                })?;
                Ok(literal_from_numeric_f64(out_dtype, float_op(lhs_f, rhs_f)))
            }
        }
        _ => {
            let lhs_f = literal_to_numeric_f64(lhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric lhs",
            })?;
            let rhs_f = literal_to_numeric_f64(rhs).ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric rhs",
            })?;
            Ok(literal_from_numeric_f64(out_dtype, float_op(lhs_f, rhs_f)))
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
        // JAX lattice: U32+F32 → F32 (not F64)
        assert_eq!(promote_dtype(DType::U32, DType::F32), DType::F32);
        assert_eq!(promote_dtype(DType::F32, DType::U32), DType::F32);
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

    #[test]
    fn bool_bool_add_and_mul_match_oracle_dtype() {
        let add = binary_literal_op(
            Literal::Bool(true),
            Literal::Bool(false),
            Primitive::Add,
            &|a, b| a + b,
            &|a, b| a + b,
        )
        .expect("bool + bool should evaluate");
        assert_eq!(add, Literal::Bool(true));

        let mul = binary_literal_op(
            Literal::Bool(true),
            Literal::Bool(false),
            Primitive::Mul,
            &|a, b| a * b,
            &|a, b| a * b,
        )
        .expect("bool * bool should evaluate");
        assert_eq!(mul, Literal::Bool(false));
    }

    #[test]
    fn bool_float_uses_numeric_bool_value() {
        let out = binary_literal_op(
            Literal::Bool(true),
            Literal::from_f64(2.5),
            Primitive::Add,
            &|a, b| a + b,
            &|a, b| a + b,
        )
        .expect("bool + f64 should evaluate");

        assert_eq!(out.as_f64(), Some(3.5));
    }

    #[test]
    fn half_precision_outputs_preserve_literal_width() {
        let bf16 = binary_literal_op(
            Literal::Bool(true),
            Literal::from_bf16_f32(2.5),
            Primitive::Add,
            &|a, b| a + b,
            &|a, b| a + b,
        )
        .expect("bool + bf16 should evaluate");
        assert!(matches!(bf16, Literal::BF16Bits(_)));
        assert_eq!(bf16.as_f64(), Some(3.5));

        let f16 = binary_literal_op(
            Literal::Bool(true),
            Literal::from_f16_f32(2.5),
            Primitive::Add,
            &|a, b| a + b,
            &|a, b| a + b,
        )
        .expect("bool + f16 should evaluate");
        assert!(matches!(f16, Literal::F16Bits(_)));
        assert_eq!(f16.as_f64(), Some(3.5));
    }

    #[test]
    fn f32_outputs_preserve_literal_width() -> Result<(), EvalError> {
        let out = binary_literal_op(
            Literal::Bool(true),
            Literal::from_f32(2.5),
            Primitive::Add,
            &|a, b| a + b,
            &|a, b| a + b,
        )?;

        assert!(matches!(out, Literal::F32Bits(_)));
        assert_eq!(out.as_f64(), Some(3.5));
        Ok(())
    }
}
