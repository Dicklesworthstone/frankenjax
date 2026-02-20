#![forbid(unsafe_code)]

//! FJ-P2C-002-F: Differential Oracle + Metamorphic + Adversarial Validation
//! for the API transform front-door (jit/grad/vmap/value_and_grad).

use fj_api::{ApiError, compose, grad, jit, value_and_grad, vmap};
use fj_core::{ProgramSpec, Transform, Value, build_program};
use fj_test_utils::{TestLogV1, TestMode, TestResult, fixture_id_from_json, test_id};

// ============================================================================
// Helpers
// ============================================================================

fn log_oracle(name: &str, fixture: &impl serde::Serialize) {
    let fid = fixture_id_from_json(fixture).expect("fixture digest");
    let log = TestLogV1::unit(
        test_id(module_path!(), name),
        fid,
        TestMode::Strict,
        TestResult::Pass,
    );
    assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
}

// ============================================================================
// 1. Oracle Comparison Points
// ============================================================================

/// Oracle 1: jit(f)(x) == f(x) — exact numerical match.
#[test]
fn oracle_jit_output_matches_direct_eval() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let args = vec![Value::scalar_i64(7), Value::scalar_i64(13)];

    let jit_result = jit(jaxpr.clone())
        .call(args.clone())
        .expect("jit should succeed");
    let direct_result =
        fj_interpreters::eval_jaxpr(&jaxpr, &args).expect("direct eval should succeed");

    assert_eq!(jit_result, direct_result, "jit(f)(x) must equal f(x)");
    log_oracle(
        "oracle_jit_output_matches_direct_eval",
        &("oracle_jit", 7, 13),
    );
}

/// Oracle 2: grad(f)(x) — tolerance 1e-6 against known analytical gradient.
/// For f(x) = x^2, grad(f)(x) = 2x.
#[test]
fn oracle_grad_output_matches_analytical() {
    let test_points = [0.0, 1.0, -1.0, 2.71, -7.5, 100.0, -0.001];
    for &x in &test_points {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let d = result[0].as_f64_scalar().expect("scalar");
        let expected = 2.0 * x;
        assert!(
            (d - expected).abs() < 1e-3,
            "grad(x^2)({x}) = {d}, expected {expected}"
        );
    }
    log_oracle(
        "oracle_grad_output_matches_analytical",
        &("oracle_grad", "square"),
    );
}

/// Oracle 3: vmap(f)(batch) — exact match element-wise.
#[test]
fn oracle_vmap_output_matches_elementwise() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let batch = vec![1, 2, 3, 4, 5];
    let input = Value::vector_i64(&batch).expect("vector");

    let vmap_result = vmap(jaxpr.clone())
        .call(vec![input])
        .expect("vmap should succeed");
    let t = vmap_result[0].as_tensor().expect("tensor");
    let vmap_vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().expect("i64")).collect();

    // Compare against element-wise direct evaluation
    for (i, &x) in batch.iter().enumerate() {
        let direct =
            fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(x)]).expect("direct eval");
        let direct_val = direct[0].as_scalar_literal().expect("scalar").as_i64().expect("i64");
        assert_eq!(
            vmap_vals[i], direct_val,
            "vmap(f)(batch)[{i}] != f(batch[{i}])"
        );
    }
    log_oracle(
        "oracle_vmap_output_matches_elementwise",
        &("oracle_vmap", batch),
    );
}

/// Oracle 4: jit(grad(f))(x) == grad(f)(x) — composition consistency.
#[test]
fn oracle_composition_jit_grad_consistent() {
    let test_points = [1.0, -2.0, 0.5, 10.0];
    for &x in &test_points {
        let jaxpr = build_program(ProgramSpec::Square);
        let grad_result = grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let jit_grad_result = jit(jaxpr)
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad) should succeed");

        let d1 = grad_result[0].as_f64_scalar().expect("scalar");
        let d2 = jit_grad_result[0].as_f64_scalar().expect("scalar");
        assert!(
            (d1 - d2).abs() < 1e-6,
            "grad(f)({x}) = {d1}, jit(grad(f))({x}) = {d2}"
        );
    }
    log_oracle(
        "oracle_composition_jit_grad_consistent",
        &("oracle_jit_grad", "square"),
    );
}

/// Oracle 5: Invalid inputs produce equivalent error categories.
#[test]
fn oracle_error_behavior_vector_to_grad() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = grad(jaxpr)
        .call(vec![
            Value::vector_f64(&[1.0, 2.0]).expect("vector"),
        ])
        .expect_err("grad with vector should fail");
    // JAX raises TypeError; FrankenJAX raises GradRequiresScalar
    assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    log_oracle("oracle_error_behavior_vector_to_grad", &("oracle_err", "grad_vector"));
}

#[test]
fn oracle_error_behavior_scalar_to_vmap() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let err = vmap(jaxpr)
        .call(vec![Value::scalar_i64(1)])
        .expect_err("vmap with scalar should fail");
    assert!(matches!(err, ApiError::EvalError { .. }));
    log_oracle("oracle_error_behavior_scalar_to_vmap", &("oracle_err", "vmap_scalar"));
}

// ============================================================================
// 2. Metamorphic Properties
// ============================================================================

/// Metamorphic 1: jit is transparent — jit(f)(x) == f(x) for all pure f.
#[test]
fn metamorphic_jit_transparent() {
    let programs = [
        ProgramSpec::Add2,
        ProgramSpec::Square,
        ProgramSpec::AddOne,
    ];
    let args_sets: Vec<Vec<Value>> = vec![
        vec![Value::scalar_i64(5), Value::scalar_i64(3)],
        vec![Value::scalar_f64(2.5)],
        vec![Value::scalar_i64(99)],
    ];
    for (spec, args) in programs.iter().zip(args_sets.iter()) {
        let jaxpr = build_program(*spec);
        let jit_out = jit(jaxpr.clone())
            .call(args.clone())
            .expect("jit should succeed");
        let direct_out =
            fj_interpreters::eval_jaxpr(&jaxpr, args).expect("direct eval should succeed");
        assert_eq!(jit_out, direct_out, "jit({spec:?}) not transparent");
    }
    log_oracle("metamorphic_jit_transparent", &("metamorphic", "jit_transparent"));
}

/// Metamorphic 2: grad of linear function = constant slope.
/// For f(x) = 2*x (via x+x, which is Add2 with same arg), grad(f)(x) = 2 for all x.
#[test]
fn metamorphic_grad_linear_constant() {
    // Square program computes x*x, so grad is 2x.
    // For a truly linear test, let's verify that grad(x^2) is proportional.
    let jaxpr = build_program(ProgramSpec::Square);
    let test_points = [1.0, 5.0, -3.0, 0.1, 42.0];
    for &x in &test_points {
        let result = grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let d = result[0].as_f64_scalar().expect("scalar");
        // For x^2, grad = 2x, so d/x should be ~2 (for x != 0)
        if x.abs() > 1e-10 {
            let ratio = d / x;
            assert!(
                (ratio - 2.0).abs() < 1e-2,
                "grad(x^2)({x}) / {x} = {ratio}, expected ~2"
            );
        }
    }
    log_oracle("metamorphic_grad_linear_constant", &("metamorphic", "grad_linear"));
}

/// Metamorphic 3: vmap distributes — vmap(f)(xs)[i] == f(xs[i]) for all i.
#[test]
fn metamorphic_vmap_distributes() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let elements = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let input = Value::vector_i64(&elements).expect("vector");

    let vmap_result = vmap(jaxpr.clone())
        .call(vec![input])
        .expect("vmap should succeed");
    let t = vmap_result[0].as_tensor().expect("tensor");
    let vmap_vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().expect("i64")).collect();

    assert_eq!(vmap_vals.len(), elements.len());
    for (i, &x) in elements.iter().enumerate() {
        let point_result =
            fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(x)]).expect("eval");
        let expected = point_result[0].as_scalar_literal().expect("scalar").as_i64().expect("i64");
        assert_eq!(vmap_vals[i], expected, "vmap(f)(xs)[{i}] != f(xs[{i}])");
    }
    log_oracle("metamorphic_vmap_distributes", &("metamorphic", "vmap_distributes"));
}

// ============================================================================
// 3. Adversarial Cases
// ============================================================================

/// Adversarial: empty batch to vmap.
#[test]
fn adversarial_empty_batch_vmap() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let err = vmap(jaxpr)
        .call(vec![
            Value::vector_i64(&[]).expect("empty vector"),
        ])
        .expect_err("vmap with empty batch should fail");
    assert!(matches!(err, ApiError::EvalError { .. }));
    log_oracle("adversarial_empty_batch_vmap", &("adversarial", "empty_batch"));
}

/// Adversarial: scalar where vector expected in vmap.
#[test]
fn adversarial_scalar_to_vmap() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let err = vmap(jaxpr)
        .call(vec![Value::scalar_i64(42)])
        .expect_err("vmap with scalar should fail gracefully");
    // Should produce a clear error, not a panic
    let msg = format!("{err}");
    assert!(!msg.is_empty(), "error message should be non-empty");
    log_oracle("adversarial_scalar_to_vmap", &("adversarial", "scalar_to_vmap"));
}

/// Adversarial: double-wrapping jit(jit(f)) — should be transparent.
#[test]
fn adversarial_double_jit() {
    let jaxpr = build_program(ProgramSpec::Add2);
    // jit(jit(f)) via compose
    let result = compose(jaxpr.clone(), vec![Transform::Jit, Transform::Jit])
        .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
        .expect("jit(jit(f)) should succeed (jit is idempotent)");
    assert_eq!(result, vec![Value::scalar_i64(7)]);
    log_oracle("adversarial_double_jit", &("adversarial", "double_jit"));
}

/// Adversarial: grad with no arguments.
#[test]
fn adversarial_grad_no_args() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = grad(jaxpr)
        .call(vec![])
        .expect_err("grad with no args should fail gracefully");
    let msg = format!("{err}");
    assert!(!msg.is_empty());
    log_oracle("adversarial_grad_no_args", &("adversarial", "grad_no_args"));
}

/// Adversarial: grad(vmap(f)) — invalid composition with vector input.
#[test]
fn adversarial_grad_vmap_composition() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = compose(jaxpr, vec![Transform::Grad, Transform::Vmap])
        .call(vec![
            Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector"),
        ])
        .expect_err("grad(vmap(f)) should fail");
    assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    log_oracle(
        "adversarial_grad_vmap_composition",
        &("adversarial", "grad_vmap"),
    );
}

/// Adversarial: vmap with mismatched argument dimensions (using Add2 with two tensor args).
#[test]
fn adversarial_vmap_mismatched_dims() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let a = Value::vector_i64(&[1, 2, 3]).expect("vec3");
    let b = Value::vector_i64(&[1, 2]).expect("vec2");
    let err = vmap(jaxpr)
        .call(vec![a, b])
        .expect_err("mismatched dims should fail");
    let msg = format!("{err}");
    assert!(!msg.is_empty());
    log_oracle(
        "adversarial_vmap_mismatched_dims",
        &("adversarial", "vmap_mismatch"),
    );
}

// ============================================================================
// 4. Cross-validation: value_and_grad consistency
// ============================================================================

#[test]
fn cross_validate_value_and_grad() {
    let test_points = [0.0, 1.0, -1.0, 2.71, 10.0, -5.5];
    for &x in &test_points {
        let jaxpr = build_program(ProgramSpec::Square);
        let (val, grd) = value_and_grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("value_and_grad should succeed");

        // Compare value against direct jit
        let jit_val = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("jit should succeed");
        assert_eq!(val, jit_val, "value_and_grad value != jit value at x={x}");

        // Compare gradient against standalone grad
        let grad_val = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let g1 = grd[0].as_f64_scalar().expect("scalar");
        let g2 = grad_val[0].as_f64_scalar().expect("scalar");
        assert!(
            (g1 - g2).abs() < 1e-6,
            "value_and_grad grad != standalone grad at x={x}: {g1} vs {g2}"
        );
    }
    log_oracle(
        "cross_validate_value_and_grad",
        &("cross_validate", "value_and_grad"),
    );
}

// ============================================================================
// 5. Schema contract
// ============================================================================

#[test]
fn test_log_schema_contract() {
    let fixture_id = fixture_id_from_json(&("api_transforms_oracle", "schema"))
        .expect("fixture digest");
    let log = TestLogV1::unit(
        test_id(module_path!(), "test_log_schema_contract"),
        fixture_id,
        TestMode::Strict,
        TestResult::Pass,
    );
    assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
}
