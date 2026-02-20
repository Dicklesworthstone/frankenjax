#![forbid(unsafe_code)]

pub mod errors;
pub mod transforms;

pub use errors::ApiError;
pub use transforms::{
    ComposedTransform, GradWrapped, JitWrapped, ValueAndGradWrapped, VmapWrapped,
};
pub use transforms::{compose, grad, jit, value_and_grad, vmap};

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, Transform, Value, build_program};

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
}
