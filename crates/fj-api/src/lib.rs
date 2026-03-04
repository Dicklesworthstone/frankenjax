#![forbid(unsafe_code)]

pub mod errors;
pub mod transforms;

pub use errors::ApiError;
pub use fj_ad::{clear_custom_derivative_rules, register_custom_jvp, register_custom_vjp};
pub use transforms::{
    ComposedTransform, GradWrapped, HessianWrapped, JacobianWrapped, JitWrapped,
    ValueAndGradWrapped, VmapWrapped,
};
pub use transforms::{compose, grad, hessian, jacobian, jit, value_and_grad, vmap};

// Re-export make_jaxpr tracing API from fj-trace
pub use fj_trace::{TracerRef, make_jaxpr, make_jaxpr_fallible};

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{
        Atom, Equation, Jaxpr, Primitive, ProgramSpec, Transform, Value, VarId, build_program,
    };
    use std::sync::{Mutex, OnceLock};

    fn custom_rule_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("custom rule test guard lock should succeed")
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
            other => panic!("expected VmapDimensionMismatch, got: {other:?}"),
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
}
