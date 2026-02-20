#![forbid(unsafe_code)]

mod arithmetic;
mod comparison;
mod reduction;
mod tensor_ops;
mod type_promotion;

use fj_core::{Primitive, Shape, Value, ValueError};
use std::collections::BTreeMap;

use arithmetic::{
    eval_binary_elementwise, eval_dot, eval_select, eval_unary_elementwise,
    eval_unary_int_or_float, erf_approx,
};
use comparison::eval_comparison;
use reduction::{eval_reduce, eval_reduce_axes};
use tensor_ops::{eval_broadcast_in_dim, eval_concatenate, eval_reshape, eval_slice, eval_transpose};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    ArityMismatch {
        primitive: Primitive,
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        primitive: Primitive,
        detail: &'static str,
    },
    ShapeMismatch {
        primitive: Primitive,
        left: Shape,
        right: Shape,
    },
    Unsupported {
        primitive: Primitive,
        detail: String,
    },
    InvalidTensor(ValueError),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch {
                primitive,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "arity mismatch for {}: expected {}, got {}",
                    primitive.as_str(),
                    expected,
                    actual
                )
            }
            Self::TypeMismatch { primitive, detail } => {
                write!(f, "type mismatch for {}: {}", primitive.as_str(), detail)
            }
            Self::ShapeMismatch {
                primitive,
                left,
                right,
            } => {
                write!(
                    f,
                    "shape mismatch for {}: left={:?} right={:?}",
                    primitive.as_str(),
                    left.dims,
                    right.dims
                )
            }
            Self::Unsupported { primitive, detail } => {
                write!(f, "unsupported {} behavior: {}", primitive.as_str(), detail)
            }
            Self::InvalidTensor(err) => write!(f, "invalid tensor: {err}"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<ValueError> for EvalError {
    fn from(value: ValueError) -> Self {
        Self::InvalidTensor(value)
    }
}

#[inline]
pub fn eval_primitive(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    match primitive {
        // Binary arithmetic
        Primitive::Add => eval_binary_elementwise(primitive, inputs, |a, b| a + b, |a, b| a + b),
        Primitive::Sub => eval_binary_elementwise(primitive, inputs, |a, b| a - b, |a, b| a - b),
        Primitive::Mul => eval_binary_elementwise(primitive, inputs, |a, b| a * b, |a, b| a * b),
        Primitive::Max => eval_binary_elementwise(primitive, inputs, |a, b| a.max(b), f64::max),
        Primitive::Min => eval_binary_elementwise(primitive, inputs, |a, b| a.min(b), f64::min),
        Primitive::Pow => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| (a as f64).powf(b as f64) as i64,
            f64::powf,
        ),
        // Unary arithmetic
        Primitive::Neg => eval_unary_int_or_float(primitive, inputs, |x| -x, |x| -x),
        Primitive::Abs => eval_unary_int_or_float(primitive, inputs, i64::abs, f64::abs),
        Primitive::Exp => eval_unary_elementwise(primitive, inputs, f64::exp),
        Primitive::Log => eval_unary_elementwise(primitive, inputs, f64::ln),
        Primitive::Sqrt => eval_unary_elementwise(primitive, inputs, f64::sqrt),
        Primitive::Rsqrt => eval_unary_elementwise(primitive, inputs, |x| 1.0 / x.sqrt()),
        Primitive::Floor => eval_unary_elementwise(primitive, inputs, f64::floor),
        Primitive::Ceil => eval_unary_elementwise(primitive, inputs, f64::ceil),
        Primitive::Round => eval_unary_elementwise(primitive, inputs, f64::round),
        // Trigonometric
        Primitive::Sin => eval_unary_elementwise(primitive, inputs, f64::sin),
        Primitive::Cos => eval_unary_elementwise(primitive, inputs, f64::cos),
        Primitive::Tan => eval_unary_elementwise(primitive, inputs, f64::tan),
        Primitive::Asin => eval_unary_elementwise(primitive, inputs, f64::asin),
        Primitive::Acos => eval_unary_elementwise(primitive, inputs, f64::acos),
        Primitive::Atan => eval_unary_elementwise(primitive, inputs, f64::atan),
        // Hyperbolic
        Primitive::Sinh => eval_unary_elementwise(primitive, inputs, f64::sinh),
        Primitive::Cosh => eval_unary_elementwise(primitive, inputs, f64::cosh),
        Primitive::Tanh => eval_unary_elementwise(primitive, inputs, f64::tanh),
        // Additional math
        Primitive::Expm1 => eval_unary_elementwise(primitive, inputs, f64::exp_m1),
        Primitive::Log1p => eval_unary_elementwise(primitive, inputs, f64::ln_1p),
        Primitive::Sign => eval_unary_int_or_float(
            primitive,
            inputs,
            |x| x.signum(),
            |x| {
                if x.is_nan() {
                    f64::NAN
                } else {
                    x.signum()
                }
            },
        ),
        Primitive::Square => eval_unary_int_or_float(primitive, inputs, |x| x * x, |x| x * x),
        Primitive::Reciprocal => eval_unary_elementwise(primitive, inputs, |x| 1.0 / x),
        Primitive::Logistic => {
            eval_unary_elementwise(primitive, inputs, |x| 1.0 / (1.0 + (-x).exp()))
        }
        Primitive::Erf => eval_unary_elementwise(primitive, inputs, erf_approx),
        Primitive::Erfc => eval_unary_elementwise(primitive, inputs, |x| 1.0 - erf_approx(x)),
        // Binary math
        Primitive::Div => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| if b != 0 { a / b } else { 0 },
            |a, b| a / b,
        ),
        Primitive::Rem => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| if b != 0 { a % b } else { 0 },
            |a, b| a % b,
        ),
        Primitive::Atan2 => {
            eval_binary_elementwise(primitive, inputs, |a, b| (a as f64).atan2(b as f64) as i64, f64::atan2)
        }
        // Selection
        Primitive::Select => eval_select(primitive, inputs),
        // Dot product
        Primitive::Dot => eval_dot(inputs),
        // Comparison
        Primitive::Eq => eval_comparison(primitive, inputs, |a, b| a == b, |a, b| a == b),
        Primitive::Ne => eval_comparison(primitive, inputs, |a, b| a != b, |a, b| a != b),
        Primitive::Lt => eval_comparison(primitive, inputs, |a, b| a < b, |a, b| a < b),
        Primitive::Le => eval_comparison(primitive, inputs, |a, b| a <= b, |a, b| a <= b),
        Primitive::Gt => eval_comparison(primitive, inputs, |a, b| a > b, |a, b| a > b),
        Primitive::Ge => eval_comparison(primitive, inputs, |a, b| a >= b, |a, b| a >= b),
        // Reductions (axis-aware)
        Primitive::ReduceSum => {
            eval_reduce_axes(primitive, inputs, params, 0_i64, 0.0, |a, b| a + b, |a, b| a + b)
        }
        Primitive::ReduceMax => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            f64::max,
        ),
        Primitive::ReduceMin => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            f64::min,
        ),
        Primitive::ReduceProd => {
            eval_reduce_axes(primitive, inputs, params, 1_i64, 1.0, |a, b| a * b, |a, b| a * b)
        }
        // Shape manipulation
        Primitive::Reshape => eval_reshape(inputs, params),
        Primitive::Transpose => eval_transpose(inputs, params),
        Primitive::BroadcastInDim => eval_broadcast_in_dim(inputs, params),
        Primitive::Concatenate => eval_concatenate(inputs, params),
        Primitive::Slice => eval_slice(inputs, params),
        // Not yet implemented
        Primitive::Gather | Primitive::Scatter => Err(EvalError::Unsupported {
            primitive,
            detail: "runtime kernel not implemented yet for this primitive".to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalError, eval_primitive};
    use fj_core::{DType, Primitive, Value};
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    #[test]
    fn add_i64_scalars() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_i64(5)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn add_vector_and_scalar_broadcasts() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let out = eval_primitive(Primitive::Add, &[input, Value::scalar_i64(2)], &no_params())
            .expect("broadcasted add should succeed");

        let expected = Value::vector_i64(&[3, 4, 5]).expect("vector value should build");
        assert_eq!(out, expected);
    }

    #[test]
    fn sub_i64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn sub_f64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_f64(5.5), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.5).abs() < 1e-10);
    }

    #[test]
    fn neg_i64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_i64(7)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(-7)));
    }

    #[test]
    fn neg_f64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - (-3.5)).abs() < 1e-10);
    }

    #[test]
    fn abs_negative_i64() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_i64(-42)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(42)));
    }

    #[test]
    fn abs_negative_f64() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(-2.78)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 2.78).abs() < 1e-10);
    }

    #[test]
    fn max_i64_scalars() {
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn min_i64_scalars() {
        let out = eval_primitive(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(3)));
    }

    #[test]
    fn exp_scalar() {
        let out = eval_primitive(Primitive::Exp, &[Value::scalar_f64(1.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar() {
        let out = eval_primitive(
            Primitive::Log,
            &[Value::scalar_f64(std::f64::consts::E)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_scalar() {
        let out = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(9.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn rsqrt_scalar() {
        let out =
            eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(4.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn floor_scalar() {
        let out =
            eval_primitive(Primitive::Floor, &[Value::scalar_f64(3.7)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn ceil_scalar() {
        let out = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(3.2)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn round_scalar() {
        let out =
            eval_primitive(Primitive::Round, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn pow_f64_scalars() {
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eq_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_i64_scalars() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(3), Value::scalar_i64(5)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(5), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn le_ge_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Le,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Ge,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn gt_f64_scalars() {
        let out = eval_primitive(
            Primitive::Gt,
            &[Value::scalar_f64(3.5), Value::scalar_f64(2.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn comparison_on_vectors() {
        let lhs = Value::vector_i64(&[1, 2, 3]).unwrap();
        let rhs = Value::vector_i64(&[2, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::Lt, &[lhs, rhs], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 3);
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true));
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false));
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false));
        } else {
            panic!("expected tensor output for vector comparison");
        }
    }

    #[test]
    fn reduce_max_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(9));
    }

    #[test]
    fn reduce_min_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(1));
    }

    #[test]
    fn reduce_prod_vector() {
        let input = Value::vector_i64(&[2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(24));
    }

    #[test]
    fn neg_vector() {
        let input = Value::vector_i64(&[1, -2, 3]).unwrap();
        let out = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let expected = Value::vector_i64(&[-1, 2, -3]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_vector_works() {
        let lhs = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let rhs = Value::vector_i64(&[4, 5, 6]).expect("vector value should build");
        let out =
            eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).expect("dot should succeed");
        assert_eq!(out, Value::scalar_i64(32));
    }

    #[test]
    fn reduce_sum_requires_single_argument() {
        let err = eval_primitive(Primitive::ReduceSum, &[], &no_params()).expect_err("should fail");
        assert_eq!(
            err,
            EvalError::ArityMismatch {
                primitive: Primitive::ReduceSum,
                expected: 1,
                actual: 0,
            }
        );
    }

    // --- Shape manipulation tests ---

    #[test]
    fn reshape_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,3".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            assert_eq!(t.elements.len(), 6);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn reshape_with_inferred_dim() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,-1".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn transpose_2d() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape { dims: vec![2, 3] },
            vec![
                fj_core::Literal::I64(1),
                fj_core::Literal::I64(2),
                fj_core::Literal::I64(3),
                fj_core::Literal::I64(4),
                fj_core::Literal::I64(5),
                fj_core::Literal::I64(6),
            ],
        )
        .unwrap();
        let out =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2]);
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(4));
            assert_eq!(t.elements[2], fj_core::Literal::I64(2));
            assert_eq!(t.elements[3], fj_core::Literal::I64(5));
            assert_eq!(t.elements[4], fj_core::Literal::I64(3));
            assert_eq!(t.elements[5], fj_core::Literal::I64(6));
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn broadcast_in_dim_scalar_to_vector() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());
        let out =
            eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(5)], &params).unwrap();
        let expected = Value::vector_i64(&[5, 5, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn broadcast_in_dim_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("broadcast_dimensions".into(), "1".into());
        let out = eval_primitive(Primitive::BroadcastInDim, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            // Row 0: [1,2,3], Row 1: [1,2,3]
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(2));
            assert_eq!(t.elements[2], fj_core::Literal::I64(3));
            assert_eq!(t.elements[3], fj_core::Literal::I64(1));
            assert_eq!(t.elements[4], fj_core::Literal::I64(2));
            assert_eq!(t.elements[5], fj_core::Literal::I64(3));
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn concatenate_vectors() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[3, 4, 5]).unwrap();
        let out = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
        let expected = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn slice_vector() {
        let input = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "4".into());
        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();
        let expected = Value::vector_i64(&[20, 30, 40]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_lax_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("lax", "add")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_lax_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ===================================================================
    // NaN / Inf edge cases (IEEE 754 compliance)
    // ===================================================================

    #[test]
    fn add_nan_propagates() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sub_inf_inf_is_nan() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn mul_nan_propagates() {
        let out = eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(0.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn log_zero_is_neg_inf() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.is_infinite() && v < 0.0);
    }

    #[test]
    fn log_negative_is_nan() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let out =
            eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn exp_neg_inf_is_zero() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn exp_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_infinite());
    }

    #[test]
    fn neg_zero_is_neg_zero() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_negative());
    }

    #[test]
    fn abs_neg_zero_is_pos_zero() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_f64(-0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_positive());
    }

    #[test]
    fn abs_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn abs_neg_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Abs,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    #[test]
    fn reduce_sum_with_nan() {
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceSum, &[input], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn reduce_max_with_nan() {
        // f64::max returns the non-NaN value (Rust/JAX behavior)
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn max_nan_returns_other() {
        // f64::max(NaN, x) returns x (Rust/JAX behavior, not IEEE 754-2019 maximum)
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(5.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eq_nan_nan_is_false() {
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_nan_nan_is_true() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_nan_always_false() {
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    // ===================================================================
    // Type promotion tests
    // ===================================================================

    #[test]
    fn add_i64_f64_promotes_to_f64() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_f64(3.5)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 5.5).abs() < 1e-10);
    }

    #[test]
    fn sub_vector_broadcasts_scalar() {
        let vec = Value::vector_i64(&[10, 20, 30]).unwrap();
        let out = eval_primitive(Primitive::Sub, &[vec, Value::scalar_i64(5)], &no_params()).unwrap();
        let expected = Value::vector_i64(&[5, 15, 25]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn mul_scalar_broadcasts_to_vector() {
        let vec = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Mul, &[Value::scalar_i64(10), vec], &no_params()).unwrap();
        let expected = Value::vector_i64(&[10, 20, 30]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pow_int_int_cast_through_float() {
        // pow(2, 10) should give 1024 (cast through f64)
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_i64(2), Value::scalar_i64(10)],
            &no_params(),
        )
        .unwrap();
        // i64 pow goes through f64 path: (2 as f64).powf(10 as f64) as i64 = 1024
        if let Value::Scalar(fj_core::Literal::I64(v)) = out {
            assert_eq!(v, 1024);
        } else {
            panic!("expected i64 scalar from int pow");
        }
    }

    // ===================================================================
    // Broadcasting and shape mismatch error tests
    // ===================================================================

    #[test]
    fn add_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn binary_wrong_arity_error() {
        let err = eval_primitive(Primitive::Add, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn unary_wrong_arity_error() {
        let err = eval_primitive(
            Primitive::Neg,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn dot_wrong_arity_error() {
        let err =
            eval_primitive(Primitive::Dot, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn comparison_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // Reduction edge cases
    // ===================================================================

    #[test]
    fn reduce_sum_scalar_identity() {
        let out = eval_primitive(Primitive::ReduceSum, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn reduce_prod_f64_vector() {
        let input = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 24.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_f64_with_neg_inf() {
        let input = Value::vector_f64(&[1.0, f64::NEG_INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn reduce_max_f64_with_inf() {
        let input = Value::vector_f64(&[1.0, f64::INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    // ===================================================================
    // Tensor manipulation edge cases
    // ===================================================================

    #[test]
    fn reshape_incompatible_size_error() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,2".into());
        let err = eval_primitive(Primitive::Reshape, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn transpose_invalid_permutation_error() {
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape { dims: vec![2, 3] },
            vec![
                fj_core::Literal::I64(1),
                fj_core::Literal::I64(2),
                fj_core::Literal::I64(3),
                fj_core::Literal::I64(4),
                fj_core::Literal::I64(5),
                fj_core::Literal::I64(6),
            ],
        )
        .unwrap();
        let mut params = BTreeMap::new();
        params.insert("permutation".into(), "0".into()); // wrong length
        let err =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn transpose_scalar_is_identity() {
        let out =
            eval_primitive(Primitive::Transpose, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn slice_out_of_bounds_error() {
        let input = Value::vector_i64(&[10, 20, 30]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "5".into()); // exceeds dim
        let err = eval_primitive(Primitive::Slice, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn concatenate_scalar_error() {
        let err = eval_primitive(
            Primitive::Concatenate,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn gather_unsupported() {
        let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn scatter_unsupported() {
        let err =
            eval_primitive(Primitive::Scatter, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    // ===================================================================
    // Trigonometric tests
    // ===================================================================

    #[test]
    fn sin_zero_is_zero() {
        let out = eval_primitive(Primitive::Sin, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cos_zero_is_one() {
        let out = eval_primitive(Primitive::Cos, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sin_pi_is_zero() {
        let out = eval_primitive(
            Primitive::Sin,
            &[Value::scalar_f64(std::f64::consts::PI)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn sin_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Sin, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    // ===================================================================
    // Comparison with broadcast (scalar + tensor)
    // ===================================================================

    #[test]
    fn gt_scalar_tensor_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Gt, &[Value::scalar_i64(3), vec], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 3 > 1
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 3 > 5
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false)); // 3 > 3
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn le_tensor_scalar_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Le, &[vec, Value::scalar_i64(3)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 1 <= 3
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 5 <= 3
            assert_eq!(t.elements[2], fj_core::Literal::Bool(true)); // 3 <= 3
        } else {
            panic!("expected tensor");
        }
    }

    // ===================================================================
    // Dot product edge cases
    // ===================================================================

    #[test]
    fn dot_f64_vectors() {
        let lhs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let rhs = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let out = eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 32.0).abs() < 1e-10);
    }

    #[test]
    fn dot_scalar_multiply() {
        let out = eval_primitive(
            Primitive::Dot,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(21));
    }

    #[test]
    fn dot_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // EvalError display formatting
    // ===================================================================

    #[test]
    fn eval_error_display_formatting() {
        let err = EvalError::ArityMismatch {
            primitive: Primitive::Add,
            expected: 2,
            actual: 1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("add"));
        assert!(msg.contains("expected 2"));
        assert!(msg.contains("got 1"));
    }

    // ===================================================================
    // Vector-level elementwise tests for transcendentals
    // ===================================================================

    #[test]
    fn exp_vector() {
        let input = Value::vector_f64(&[0.0, 1.0]).unwrap();
        let out = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let e0 = t.elements[0].as_f64().unwrap();
            let e1 = t.elements[1].as_f64().unwrap();
            assert!((e0 - 1.0).abs() < 1e-10);
            assert!((e1 - std::f64::consts::E).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn floor_vector() {
        let input = Value::vector_f64(&[1.9, 2.1, -0.5]).unwrap();
        let out = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 2.0).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - (-1.0)).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn rsqrt_vector() {
        let input = Value::vector_f64(&[1.0, 4.0, 16.0]).unwrap();
        let out = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 0.5).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - 0.25).abs() < 1e-10);
        } else {
            panic!("expected tensor");
        }
    }
}

#[cfg(test)]
mod prop_tests {
    use super::eval_primitive;
    use fj_core::{Primitive, Value};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    proptest! {
        #[test]
        fn prop_add_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_mul_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_max_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_min_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_neg_involution(x in -1000i64..1000) {
            let neg1 = eval_primitive(
                Primitive::Neg,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            let neg2 = eval_primitive(
                Primitive::Neg,
                &[neg1],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(neg2, Value::scalar_i64(x));
        }

        #[test]
        fn prop_abs_non_negative(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Abs,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            if let Value::Scalar(fj_core::Literal::I64(v)) = out {
                prop_assert!(v >= 0, "abs({x}) = {v} should be non-negative");
            }
        }

        #[test]
        fn prop_reshape_roundtrip(a in -100i64..100, b in -100i64..100, c in -100i64..100) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();

            let mut to_3x1 = BTreeMap::new();
            to_3x1.insert("new_shape".into(), "3,1".into());
            let reshaped = eval_primitive(Primitive::Reshape, &[input.clone()], &to_3x1).unwrap();

            let mut to_3 = BTreeMap::new();
            to_3.insert("new_shape".into(), "3".into());
            let restored = eval_primitive(Primitive::Reshape, &[reshaped], &to_3).unwrap();

            prop_assert_eq!(restored, input);
        }

        #[test]
        fn prop_reduce_sum_matches_manual(
            a in -100i64..100,
            b in -100i64..100,
            c in -100i64..100
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceSum,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a + b + c));
        }

        #[test]
        fn prop_reduce_prod_matches_manual(
            a in -10i64..10,
            b in -10i64..10,
            c in -10i64..10
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceProd,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a * b * c));
        }

        #[test]
        fn prop_eq_reflexive(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Eq,
                &[Value::scalar_i64(x), Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_bool(true));
        }

        #[test]
        fn prop_add_sub_inverse(a in -1000i64..1000, b in -1000i64..1000) {
            let sum = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let result = eval_primitive(
                Primitive::Sub,
                &[sum, Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(result, Value::scalar_i64(a));
        }
    }
}
