#![forbid(unsafe_code)]

pub mod errors;
pub mod transforms;

pub use errors::ApiError;
pub use fj_ad::{
    AdError, JvpResult, clear_custom_derivative_rules, grad_jaxpr_with_cotangent,
    register_custom_jaxpr_jvp, register_custom_jaxpr_vjp, register_custom_jvp, register_custom_vjp,
};
pub use fj_core::{DType, Shape, Value};
pub use transforms::{
    CheckpointWrapped, ComposedTransform, CustomJvpWrapped, CustomVjpWrapped, GradWrapped,
    HessianWrapped, JacobianWrapped, JitWrapped, PmapWrapped, ValueAndGradWrapped, VmapWrapped,
};
pub use transforms::{
    checkpoint, compose, custom_jvp, custom_vjp, grad, hessian, jacobian, jit, pmap,
    value_and_grad, vmap,
};

// Re-export make_jaxpr tracing API from fj-trace
pub use fj_trace::{ShapedArray, TracerRef, make_jaxpr, make_jaxpr_fallible};

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{
        Atom, Equation, Jaxpr, Primitive, ProgramSpec, Transform, Value, VarId, build_program,
    };
    use std::sync::{Mutex, OnceLock};

    fn custom_rule_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        match GUARD.get_or_init(|| Mutex::new(())).lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn custom_square_jaxpr(input: u32, output: u32) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(input)],
            vec![],
            vec![VarId(output)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: vec![Atom::Var(VarId(input)), Atom::Var(VarId(input))].into(),
                outputs: vec![VarId(output)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    // --- Basic transform tests ---

    #[test]
    fn jit_add_scalar() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
            .expect("jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn jit_is_identity() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_i64(5), Value::scalar_i64(7)])
            .expect("jit should succeed");
        let direct_result =
            fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(7)])
                .expect("direct eval should succeed");
        assert_eq!(jit_result, direct_result);
    }

    #[test]
    fn grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("grad should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn grad_at_zero() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(0.0)])
            .expect("grad at zero should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!(derivative.abs() < 1e-3);
    }

    #[test]
    fn grad_negative_input() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(-5.0)])
            .expect("grad with negative input should succeed");
        let derivative = result[0]
            .as_f64_scalar()
            .expect("grad output should be scalar");
        assert!((derivative - (-10.0)).abs() < 1e-3);
    }

    #[test]
    fn vmap_add_one() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = vmap(jaxpr)
            .call(vec![
                Value::vector_i64(&[10, 20, 30]).expect("vector should build"),
            ])
            .expect("vmap should succeed");
        let output = result[0].as_tensor().expect("vmap output should be tensor");
        let values: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64 element"))
            .collect();
        assert_eq!(values, vec![11, 21, 31]);
    }

    #[test]
    fn value_and_grad_square() {
        let jaxpr = build_program(ProgramSpec::Square);
        let (value, gradient) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad should succeed");
        assert_eq!(value.len(), 1);
        assert_eq!(gradient.len(), 1);

        let val = value[0].as_f64_scalar().expect("value should be scalar");
        assert!((val - 16.0).abs() < 1e-6);

        let grad_val = gradient[0]
            .as_f64_scalar()
            .expect("gradient should be scalar");
        assert!((grad_val - 8.0).abs() < 1e-3);
    }

    #[test]
    fn value_and_grad_returns_gradient_for_each_input() {
        let jaxpr = build_program(ProgramSpec::LaxMul);
        let (value, gradients) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("value_and_grad should succeed");

        assert_eq!(value.len(), 1);
        assert_eq!(gradients.len(), 2);
        assert!((value[0].as_f64_scalar().expect("scalar output") - 6.0).abs() < 1e-6);
        assert!((gradients[0].as_f64_scalar().expect("scalar gradient") - 3.0).abs() < 1e-6);
        assert!((gradients[1].as_f64_scalar().expect("scalar gradient") - 2.0).abs() < 1e-6);
    }

    // --- Transform composition tests ---

    #[test]
    fn jit_grad_composition() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jit(jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(5.0)])
            .expect("jit(grad(f)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 10.0).abs() < 1e-3);
    }

    #[test]
    fn jit_vmap_composition() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = jit(jaxpr)
            .compose_vmap()
            .call(vec![
                Value::vector_i64(&[1, 2, 3]).expect("vector should build"),
            ])
            .expect("jit(vmap(f)) should succeed");
        let output = result[0].as_tensor().expect("should be tensor");
        let values: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn vmap_grad_composition() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = vmap(jaxpr)
            .compose_grad()
            .call(vec![
                Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
            ])
            .expect("vmap(grad(f)) should succeed");
        let output = result[0].as_tensor().expect("should be tensor");
        let values = output.to_f64_vec().expect("f64 elements");
        assert_eq!(values.len(), 3);
        assert!((values[0] - 2.0).abs() < 1e-3);
        assert!((values[1] - 4.0).abs() < 1e-3);
        assert!((values[2] - 6.0).abs() < 1e-3);
    }

    #[test]
    fn compose_helper() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = compose(jaxpr, vec![Transform::Jit, Transform::Grad])
            .call(vec![Value::scalar_f64(7.0)])
            .expect("compose(jit, grad) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 14.0).abs() < 1e-3);
    }

    // --- Error path tests ---

    #[test]
    fn grad_non_scalar_input_fails() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = grad(jaxpr)
            .call(vec![
                Value::vector_f64(&[1.0, 2.0]).expect("vector should build"),
            ])
            .expect_err("grad with vector input should fail");
        assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    }

    #[test]
    fn grad_empty_args_fails() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = grad(jaxpr)
            .call(vec![])
            .expect_err("grad with no args should fail");
        assert!(matches!(err, ApiError::EvalError { .. }));
    }

    #[test]
    fn vmap_scalar_input_fails() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let err = vmap(jaxpr)
            .call(vec![Value::scalar_i64(42)])
            .expect_err("vmap with scalar input should fail");
        assert!(matches!(err, ApiError::EvalError { .. }));
    }

    #[test]
    fn grad_vmap_fails_non_scalar() {
        let jaxpr = build_program(ProgramSpec::Square);
        let err = compose(jaxpr, vec![Transform::Grad, Transform::Vmap])
            .call(vec![
                Value::vector_f64(&[1.0, 2.0]).expect("vector should build"),
            ])
            .expect_err("grad(vmap(f)) should fail with non-scalar input to grad");
        assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    }

    #[test]
    fn vmap_dimension_mismatch_preserves_info() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = vmap(jaxpr)
            .call(vec![
                Value::vector_i64(&[1, 2, 3]).expect("vec3"),
                Value::vector_i64(&[1, 2]).expect("vec2"),
            ])
            .expect_err("vmap with mismatched leading dims should fail");
        match err {
            ApiError::VmapDimensionMismatch { expected, actual } => {
                assert_eq!(expected, 3, "expected leading dim should be 3");
                assert_eq!(actual, 2, "actual leading dim should be 2");
            }
            other => {
                std::panic::panic_any(format!("expected VmapDimensionMismatch, got: {other:?}"))
            }
        }
    }

    // --- Mode configuration tests ---

    #[test]
    fn with_mode_hardened() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = jit(jaxpr)
            .with_mode(fj_core::CompatibilityMode::Hardened)
            .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
            .expect("hardened jit should succeed");
        assert_eq!(result, vec![Value::scalar_i64(3)]);
    }

    #[test]
    fn composed_with_mode() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jit(jaxpr)
            .compose_grad()
            .with_mode(fj_core::CompatibilityMode::Hardened)
            .call(vec![Value::scalar_f64(2.0)])
            .expect("hardened jit(grad(f)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be scalar");
        assert!((derivative - 4.0).abs() < 1e-3);
    }

    #[test]
    fn api_exposes_custom_vjp_registration() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();
        register_custom_vjp(Primitive::CountLeadingZeros, |_inputs, _g, _params| {
            Ok(vec![Value::scalar_f64(7.0)])
        });

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::CountLeadingZeros,
                inputs: vec![Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let result = grad(jaxpr)
            .call(vec![Value::scalar_i64(8)])
            .expect("grad should use custom VJP");
        let derivative = result[0].as_f64_scalar().expect("scalar derivative");
        assert!((derivative - 7.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn api_exposes_custom_jvp_registration() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();
        register_custom_jvp(
            Primitive::CountLeadingZeros,
            |_primals, _tangents, _params| Ok(Value::scalar_f64(5.0)),
        );

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::CountLeadingZeros,
                inputs: vec![Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: std::collections::BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let result = fj_ad::jvp(&jaxpr, &[Value::scalar_i64(8)], &[Value::scalar_f64(1.0)])
            .expect("jvp should use custom rule");
        let tangent = result.tangents[0].as_f64_scalar().expect("scalar tangent");
        assert!((tangent - 5.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_vjp_composes_through_jit_grad_and_value_and_grad() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let wrapped = custom_vjp(
            custom_square_jaxpr(901, 902),
            |primals| {
                let x = primals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP forward expected scalar x".to_owned())
                })?;
                Ok((vec![Value::scalar_f64(x * x)], vec![Value::scalar_f64(x)]))
            },
            |residuals, cotangent| {
                let x = residuals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP backward expected residual x".to_owned())
                })?;
                let g = cotangent.as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom VJP backward expected scalar cotangent".to_owned())
                })?;
                Ok(vec![Value::scalar_f64(10.0 * x * g)])
            },
        );

        let primal = wrapped
            .call(vec![Value::scalar_f64(3.0)])
            .expect("custom_vjp primal call should preserve original function");
        assert!((primal[0].as_f64_scalar().expect("scalar primal") - 9.0).abs() < 1e-10);

        let composed_grad = wrapped
            .compose_jit_grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("jit(grad(custom_vjp(f))) should use the custom VJP");
        assert!((composed_grad[0].as_f64_scalar().expect("scalar gradient") - 30.0).abs() < 1e-10);

        let (values, gradients) = wrapped
            .value_and_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad(custom_vjp(f)) should use the custom VJP");
        assert!((values[0].as_f64_scalar().expect("scalar value") - 16.0).abs() < 1e-10);
        assert!((gradients[0].as_f64_scalar().expect("scalar gradient") - 40.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_vjp_wrappers_keep_equal_jaxpr_rules_isolated() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let make_wrapped = |scale: f64| {
            custom_vjp(
                custom_square_jaxpr(905, 906),
                |primals| {
                    let x = primals[0].as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed("custom VJP forward expected scalar x".to_owned())
                    })?;
                    Ok((vec![Value::scalar_f64(x * x)], vec![Value::scalar_f64(x)]))
                },
                move |residuals, cotangent| {
                    let x = residuals[0].as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed("custom VJP backward expected residual x".to_owned())
                    })?;
                    let g = cotangent.as_f64_scalar().ok_or_else(|| {
                        AdError::EvalFailed(
                            "custom VJP backward expected scalar cotangent".to_owned(),
                        )
                    })?;
                    Ok(vec![Value::scalar_f64(scale * x * g)])
                },
            )
        };

        let first = make_wrapped(10.0);
        let second = make_wrapped(20.0);

        let first_grad = first
            .grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("first custom_vjp grad should use first rule");
        let second_grad = second
            .grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("second custom_vjp grad should use second rule");
        assert!(
            (first_grad[0]
                .as_f64_scalar()
                .expect("first scalar gradient")
                - 30.0)
                .abs()
                < 1e-10
        );
        assert!(
            (second_grad[0]
                .as_f64_scalar()
                .expect("second scalar gradient")
                - 60.0)
                .abs()
                < 1e-10
        );

        let (_, first_value_grad) = first
            .value_and_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first value_and_grad should use first rule");
        let first_composed_grad = first
            .compose_jit_grad()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first jit(grad(custom_vjp)) should use first rule");
        assert!(
            (first_value_grad[0]
                .as_f64_scalar()
                .expect("first value_and_grad scalar gradient")
                - 40.0)
                .abs()
                < 1e-10
        );
        assert!(
            (first_composed_grad[0]
                .as_f64_scalar()
                .expect("first composed scalar gradient")
                - 40.0)
                .abs()
                < 1e-10
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_jvp_drives_jvp_and_jacobian() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let wrapped = custom_jvp(custom_square_jaxpr(903, 904), |primals, tangents| {
            let x = primals[0]
                .as_f64_scalar()
                .ok_or_else(|| AdError::EvalFailed("custom JVP expected scalar x".to_owned()))?;
            let dx = tangents[0].as_f64_scalar().ok_or_else(|| {
                AdError::EvalFailed("custom JVP expected scalar tangent".to_owned())
            })?;
            Ok((
                vec![Value::scalar_f64(x * x)],
                vec![Value::scalar_f64(40.0 + dx)],
            ))
        });

        let jvp = wrapped
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("custom_jvp function rule should drive direct JVP");
        assert!((jvp.primals[0].as_f64_scalar().expect("scalar primal") - 16.0).abs() < 1e-10);
        assert!((jvp.tangents[0].as_f64_scalar().expect("scalar tangent") - 42.0).abs() < 1e-10);

        let jacobian = wrapped
            .jacobian()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("jacobian(custom_jvp(f)) should use the custom JVP");
        if let Some(value) = jacobian.as_f64_scalar() {
            assert!((value - 41.0).abs() < 1e-10);
        } else {
            let tensor = jacobian.as_tensor().expect("jacobian tensor output");
            let values = tensor.to_f64_vec().expect("f64 jacobian tensor");
            assert_eq!(values.len(), 1);
            assert!((values[0] - 41.0).abs() < 1e-10);
        }

        clear_custom_derivative_rules();
    }

    #[test]
    fn function_custom_jvp_wrappers_keep_equal_jaxpr_rules_isolated() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let make_wrapped = |offset: f64| {
            custom_jvp(custom_square_jaxpr(907, 908), move |primals, tangents| {
                let x = primals[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom JVP expected scalar x".to_owned())
                })?;
                let dx = tangents[0].as_f64_scalar().ok_or_else(|| {
                    AdError::EvalFailed("custom JVP expected scalar tangent".to_owned())
                })?;
                Ok((
                    vec![Value::scalar_f64(x * x)],
                    vec![Value::scalar_f64(offset + dx)],
                ))
            })
        };

        let first = make_wrapped(100.0);
        let second = make_wrapped(200.0);

        let first_jvp = first
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("first custom_jvp call should use first rule");
        let second_jvp = second
            .jvp_call(vec![Value::scalar_f64(4.0)], vec![Value::scalar_f64(2.0)])
            .expect("second custom_jvp call should use second rule");
        assert!(
            (first_jvp.tangents[0]
                .as_f64_scalar()
                .expect("first scalar tangent")
                - 102.0)
                .abs()
                < 1e-10
        );
        assert!(
            (second_jvp.tangents[0]
                .as_f64_scalar()
                .expect("second scalar tangent")
                - 202.0)
                .abs()
                < 1e-10
        );

        let first_jacobian = first
            .jacobian()
            .call(vec![Value::scalar_f64(4.0)])
            .expect("first jacobian should use first rule");
        let first_jacobian_value = first_jacobian
            .as_f64_scalar()
            .or_else(|| {
                let tensor = first_jacobian.as_tensor()?;
                tensor.to_f64_vec()?.first().copied()
            })
            .expect("first jacobian should be scalar-like");
        assert!((first_jacobian_value - 101.0).abs() < 1e-10);

        clear_custom_derivative_rules();
    }

    #[test]
    fn jacobian_two_outputs_two_inputs() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let jac = jacobian(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("jacobian should succeed");
        let tensor = jac.as_tensor().expect("jacobian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let values = tensor.to_f64_vec().expect("f64 tensor");
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
        assert!((values[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn hessian_matches_quadratic_cross_term() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: vec![Atom::Var(VarId(3)), Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: std::collections::BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let hes = hessian(jaxpr)
            .call(vec![Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("hessian should succeed");
        let tensor = hes.as_tensor().expect("hessian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let values = tensor.to_f64_vec().expect("f64 tensor");
        assert!((values[0] - 6.0).abs() < 1e-3);
        assert!((values[1] - 4.0).abs() < 1e-3);
        assert!((values[2] - 4.0).abs() < 1e-3);
        assert!(values[3].abs() < 1e-3);
    }

    // ── End-to-end trace → transform integration tests ───────

    #[test]
    fn trace_jit_square_e2e() {
        // Trace x*x via make_jaxpr, then jit it
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(7.0)])
            .expect("jit(traced x*x) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 49.0).abs() < 1e-10, "7^2 should be 49, got {val}");
    }

    #[test]
    fn trace_grad_square_e2e() {
        // Trace x*x, then grad: d(x^2)/dx = 2x
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(5.0)])
            .expect("grad(traced x*x) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 10.0).abs() < 1e-3,
            "d(x^2)/dx at x=5 should be 10, got {derivative}"
        );
    }

    #[test]
    fn trace_jit_grad_square_e2e() {
        // Trace x*x, then jit(grad): d(x^2)/dx = 2x via composition
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(3.0)])
            .expect("jit(grad(traced x*x)) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 6.0).abs() < 1e-3,
            "d(x^2)/dx at x=3 should be 6, got {derivative}"
        );
    }

    #[test]
    fn trace_chain_e2e() {
        // Trace x*x + x + x + x = x^2 + 3x, then grad: d(x^2 + 3x)/dx = 2x + 3
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let x = &inputs[0];
                let x_sq = x.binary_op(Primitive::Mul, x).unwrap();
                let sum1 = x_sq.binary_op(Primitive::Add, x).unwrap();
                let sum2 = sum1.binary_op(Primitive::Add, x).unwrap();
                let sum3 = sum2.binary_op(Primitive::Add, x).unwrap();
                vec![sum3]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        // f(x) = x^2 + 3x, f'(x) = 2x + 3
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("grad(traced x^2 + 3x) should succeed");
        let derivative = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (derivative - 11.0).abs() < 1e-3,
            "d(x^2 + 3x)/dx at x=4 should be 11, got {derivative}"
        );
    }

    #[test]
    fn trace_multi_input_e2e() {
        // Trace f(x, y) = x + y, then jit
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let aval = ShapedArray {
            dtype: DType::F64,
            shape: fj_core::Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let sum = inputs[0].binary_op(Primitive::Add, &inputs[1]).unwrap();
                vec![sum]
            },
            vec![aval.clone(), aval],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
            .expect("jit(traced x+y) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 7.0).abs() < 1e-10, "3+4 should be 7, got {val}");
    }

    #[test]
    fn trace_unary_chain_e2e() {
        // Trace neg(exp(x)), then jit
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp = inputs[0].unary_op(Primitive::Exp).unwrap();
                let neg = exp.unary_op(Primitive::Neg).unwrap();
                vec![neg]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(1.0)])
            .expect("jit(traced neg(exp(x))) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        let expected = -(1.0_f64.exp());
        assert!(
            (val - expected).abs() < 1e-10,
            "-exp(1) should be {expected}, got {val}"
        );
    }

    // ================================================================
    // E2E trace→transform tests for complex programs (frankenjax-kdi)
    // ================================================================

    #[test]
    fn trace_sin_cos_chain_grad_e2e() {
        // f(x) = sin(cos(x)), grad: f'(x) = cos(cos(x)) * (-sin(x))
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let cos_x = inputs[0].unary_op(Primitive::Cos).unwrap();
                let sin_cos = cos_x.unary_op(Primitive::Sin).unwrap();
                vec![sin_cos]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 1.0_f64;
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(sin(cos(x))) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = x.cos().cos() * (-x.sin());
        assert!(
            (d - expected).abs() < 1e-10,
            "grad(sin(cos(1.0))) should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_exp_mul_grad_e2e() {
        // f(x) = x * exp(x), grad: f'(x) = exp(x) + x * exp(x) = (1+x) * exp(x)
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp_x = inputs[0].unary_op(Primitive::Exp).unwrap();
                let x_exp = inputs[0].binary_op(Primitive::Mul, &exp_x).unwrap();
                vec![x_exp]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 2.0_f64;
        let result = grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(x*exp(x)) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = (1.0 + x) * x.exp();
        assert!(
            (d - expected).abs() < 1e-6,
            "grad(x*exp(x)) at x=2 should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_polynomial_jit_grad_e2e() {
        // f(x) = x^3 + 2*x^2 + x via tracing: t1=x*x, t2=t1*x, t3=t1+t1, t4=t2+t3, y=t4+x
        // f'(x) = 3x^2 + 4x + 1
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let x = &inputs[0];
                let x2 = x.binary_op(Primitive::Mul, x).unwrap();
                let x3 = x2.binary_op(Primitive::Mul, x).unwrap();
                let two_x2 = x2.binary_op(Primitive::Add, &x2).unwrap();
                let sum = x3.binary_op(Primitive::Add, &two_x2).unwrap();
                let y = sum.binary_op(Primitive::Add, x).unwrap();
                vec![y]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 3.0_f64;
        let result = jit(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad(x^3+2x^2+x)) should succeed");
        let d = result[0].as_f64_scalar().expect("should be f64");
        let expected = 3.0 * x * x + 4.0 * x + 1.0;
        assert!(
            (d - expected).abs() < 1e-3,
            "jit(grad(x^3+2x^2+x)) at x=3 should be {expected}, got {d}"
        );
    }

    #[test]
    fn trace_sin_vmap_grad_e2e() {
        // Trace sin(x), then vmap(grad) over a batch
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sin_x = inputs[0].unary_op(Primitive::Sin).unwrap();
                vec![sin_x]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let inputs = [0.0, 1.0, 2.0, 3.0];
        let result = vmap(closed.jaxpr)
            .compose_grad()
            .call(vec![Value::vector_f64(&inputs).expect("vector")])
            .expect("vmap(grad(sin)) should succeed");
        let t = result[0].as_tensor().expect("tensor");
        let vals = t.to_f64_vec().expect("f64 vec");
        for (i, (&actual, &x)) in vals.iter().zip(inputs.iter()).enumerate() {
            let expected = x.cos();
            assert!(
                (actual - expected).abs() < 1e-6,
                "vmap(grad(sin))[{i}]: expected cos({x})={expected}, got {actual}"
            );
        }
    }

    #[test]
    fn trace_exp_neg_jit_e2e() {
        // f(x) = -exp(x), jit execution
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let exp_x = inputs[0].unary_op(Primitive::Exp).unwrap();
                let neg_exp = exp_x.unary_op(Primitive::Neg).unwrap();
                vec![neg_exp]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let result = jit(closed.jaxpr)
            .call(vec![Value::scalar_f64(0.0)])
            .expect("jit(-exp(x)) should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!(
            (val - (-1.0)).abs() < 1e-10,
            "-exp(0) should be -1, got {val}"
        );
    }

    #[test]
    fn trace_tanh_grad_grad_e2e() {
        // f(x) = tanh(x), f''(x) = -2*tanh(x)*sech^2(x) = -2*tanh(x)*(1-tanh^2(x))
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let tanh_x = inputs[0].unary_op(Primitive::Tanh).unwrap();
                vec![tanh_x]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let x = 0.5_f64;
        let result = compose(closed.jaxpr, vec![Transform::Grad, Transform::Grad])
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(grad(tanh)) should succeed");
        let d2 = result[0].as_f64_scalar().expect("should be f64");
        let t = x.tanh();
        let expected = -2.0 * t * (1.0 - t * t);
        assert!(
            (d2 - expected).abs() < 1e-4,
            "tanh''(0.5) should be {expected}, got {d2}"
        );
    }

    #[test]
    fn trace_fallible_error_propagation() {
        // make_jaxpr_fallible should propagate errors cleanly
        use fj_core::DType;
        use fj_trace::{ShapedArray, TraceError};

        let result = make_jaxpr_fallible(
            |_inputs| Err(TraceError::InvalidAbstractValue),
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        );
        assert!(result.is_err());
    }

    // ── value_and_grad e2e test (frankenjax-rjj) ──

    #[test]
    fn trace_value_and_grad_e2e() {
        // Trace x*x, then value_and_grad: returns (x^2, 2x)
        use fj_core::DType;
        use fj_trace::ShapedArray;

        let closed = make_jaxpr(
            |inputs| {
                let sq = inputs[0].binary_op(Primitive::Mul, &inputs[0]).unwrap();
                vec![sq]
            },
            vec![ShapedArray {
                dtype: DType::F64,
                shape: fj_core::Shape::scalar(),
            }],
        )
        .unwrap();

        let (values, gradients) = value_and_grad(closed.jaxpr)
            .call(vec![Value::scalar_f64(4.0)])
            .expect("value_and_grad(traced x*x) should succeed");
        let val = values[0].as_f64_scalar().expect("value should be f64");
        assert!((val - 16.0).abs() < 1e-3, "f(4) = 4^2 = 16, got {val}");
        assert!(!gradients.is_empty(), "should produce gradients");
    }

    #[test]
    fn jacobian_single_output_single_input() {
        // f(x) = x*x, Jacobian = [2x]
        let jaxpr = build_program(ProgramSpec::Square);
        let jac = jacobian(jaxpr)
            .call(vec![Value::scalar_f64(5.0)])
            .expect("jacobian(x^2) should succeed");
        // Result should be a scalar (1x1 Jacobian) or a 1-element tensor
        if let Some(v) = jac.as_f64_scalar() {
            assert!(
                (v - 10.0).abs() < 1e-3,
                "J(x^2) at x=5 should be 10, got {v}"
            );
        } else if let Some(t) = jac.as_tensor() {
            let vals = t.to_f64_vec().expect("f64 tensor");
            assert!(!vals.is_empty(), "jacobian tensor should have values");
        } else {
            std::panic::panic_any("jacobian should return a scalar or tensor");
        }
    }

    #[test]
    fn hessian_quadratic() {
        // f(x) = x^2, H = [2]
        let jaxpr = build_program(ProgramSpec::Square);
        let hes = hessian(jaxpr)
            .call(vec![Value::scalar_f64(3.0)])
            .expect("hessian(x^2) should succeed");
        if let Some(h) = hes.as_f64_scalar() {
            assert!((h - 2.0).abs() < 1e-2, "d²/dx²(x²) should be 2, got {h}");
        } else if let Some(t) = hes.as_tensor() {
            let vals = t.to_f64_vec().expect("f64 tensor");
            assert!(!vals.is_empty(), "hessian tensor should have values");
        }
    }
}
