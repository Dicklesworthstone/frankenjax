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

#[cfg(test)]
mod tests {
    use super::*;

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn s_i64(v: i64) -> Value {
        Value::Scalar(Literal::I64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                fj_core::Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn extract_bool(val: &Value) -> bool {
        match val {
            Value::Scalar(Literal::Bool(b)) => *b,
            _ => panic!("expected bool scalar, got {val:?}"),
        }
    }
    fn extract_bools(val: &Value) -> Vec<bool> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool element"),
            })
            .collect()
    }

    #[test]
    fn eq_scalar_true() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn eq_scalar_false() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(3.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ne_scalar() {
        let result = eval_comparison(
            Primitive::Ne,
            &[s_i64(1), s_i64(2)],
            |a, b| a != b,
            |a, b| a != b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn lt_scalar() {
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(1.0), s_f64(2.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(2.0), s_f64(1.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ge_scalar() {
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(4.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn eq_tensor_elementwise() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[1.0, 9.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, true]);
    }

    #[test]
    fn lt_tensor_elementwise() {
        let a = v_f64(&[1.0, 5.0, 3.0]);
        let b = v_f64(&[2.0, 4.0, 3.0]);
        let result = eval_comparison(Primitive::Lt, &[a, b], |a, b| a < b, |a, b| a < b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, false]);
    }

    #[test]
    fn comparison_shape_mismatch() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }

    #[test]
    fn comparison_scalar_tensor_broadcast() {
        let a = s_f64(2.0);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![false, true, false]);
    }

    #[test]
    fn comparison_arity_error() {
        let result = eval_comparison(Primitive::Eq, &[s_f64(1.0)], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }
}
