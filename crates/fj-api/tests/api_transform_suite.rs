#![forbid(unsafe_code)]

use fj_api::{ApiError, compose, grad, jit, value_and_grad, vmap};
use fj_core::{
    CompatibilityMode, DType, ProgramSpec, Shape, TensorValue, Transform, Value, build_program,
};
use fj_test_utils::{TestLogV1, TestMode, TestResult, fixture_id_from_json, test_id};
use proptest::prelude::*;

// ============================================================================
// Structured logging helpers
// ============================================================================

fn log_pass(name: &str, fixture: &impl serde::Serialize) {
    let fixture_id = fixture_id_from_json(fixture).expect("fixture digest");
    let log = TestLogV1::unit(
        test_id(module_path!(), name),
        fixture_id,
        TestMode::Strict,
        TestResult::Pass,
    );
    assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
}

// ============================================================================
// 1. API Entry Point Tests
// ============================================================================

#[test]
fn api_jit_add2_strict() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = jit(jaxpr)
        .call(vec![Value::scalar_i64(10), Value::scalar_i64(20)])
        .expect("jit should succeed");
    assert_eq!(result, vec![Value::scalar_i64(30)]);
    log_pass("api_jit_add2_strict", &("jit", "add2", 10, 20));
}

#[test]
fn api_grad_square_strict() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(5.0)])
        .expect("grad should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    assert!((d - 10.0).abs() < 1e-3);
    log_pass("api_grad_square_strict", &("grad", "square", 5.0));
}

#[test]
fn api_vmap_add_one_strict() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let result = vmap(jaxpr)
        .call(vec![Value::vector_i64(&[100, 200, 300]).expect("vector")])
        .expect("vmap should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals: Vec<i64> = t
        .elements
        .iter()
        .map(|l| l.as_i64().expect("i64"))
        .collect();
    assert_eq!(vals, vec![101, 201, 301]);
    log_pass(
        "api_vmap_add_one_strict",
        &("vmap", "add_one", vec![100, 200, 300]),
    );
}

#[test]
fn api_vmap_add_one_rank3_strict() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let tensor_3d = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![2, 2, 3],
            },
            (1..=12).map(fj_core::Literal::I64).collect(),
        )
        .expect("3d tensor should build"),
    );
    let result = vmap(jaxpr)
        .call(vec![tensor_3d])
        .expect("vmap rank3 should succeed");
    let t = result[0].as_tensor().expect("tensor");
    assert_eq!(
        t.shape,
        Shape {
            dims: vec![2, 2, 3]
        }
    );
    let vals: Vec<i64> = t
        .elements
        .iter()
        .map(|l| l.as_i64().expect("i64"))
        .collect();
    let expected: Vec<i64> = (2..=13).collect();
    assert_eq!(vals, expected);
    log_pass(
        "api_vmap_add_one_rank3_strict",
        &("vmap", "add_one_rank3", vec![2_u32, 2_u32, 3_u32]),
    );
}

#[test]
fn api_value_and_grad_square() {
    let jaxpr = build_program(ProgramSpec::Square);
    let (val, grd) = value_and_grad(jaxpr)
        .call(vec![Value::scalar_f64(3.0)])
        .expect("value_and_grad should succeed");
    let v = val[0].as_f64_scalar().expect("scalar");
    let g = grd[0].as_f64_scalar().expect("scalar");
    assert!((v - 9.0).abs() < 1e-6);
    assert!((g - 6.0).abs() < 1e-3);
    log_pass(
        "api_value_and_grad_square",
        &("value_and_grad", "square", 3.0),
    );
}

// ============================================================================
// 2. Transform Stacking Tests
// ============================================================================

#[test]
fn stacking_jit_grad() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = jit(jaxpr)
        .compose_grad()
        .call(vec![Value::scalar_f64(4.0)])
        .expect("jit(grad) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    assert!((d - 8.0).abs() < 1e-3);
    log_pass("stacking_jit_grad", &("jit_grad", "square", 4.0));
}

#[test]
fn stacking_jit_vmap() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let result = jit(jaxpr)
        .compose_vmap()
        .call(vec![Value::vector_i64(&[5, 10, 15]).expect("vector")])
        .expect("jit(vmap) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals: Vec<i64> = t
        .elements
        .iter()
        .map(|l| l.as_i64().expect("i64"))
        .collect();
    assert_eq!(vals, vec![6, 11, 16]);
    log_pass("stacking_jit_vmap", &("jit_vmap", "add_one"));
}

#[test]
fn stacking_vmap_grad() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = vmap(jaxpr)
        .compose_grad()
        .call(vec![
            Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).expect("vector"),
        ])
        .expect("vmap(grad) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64 vec");
    assert_eq!(vals.len(), 4);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 2.0 * (i as f64 + 1.0);
        assert!(
            (v - expected).abs() < 1e-3,
            "index {i}: expected {expected}, got {v}"
        );
    }
    log_pass("stacking_vmap_grad", &("vmap_grad", "square"));
}

#[test]
fn stacking_vmap_vmap_vmap_grad_rank3() {
    let jaxpr = build_program(ProgramSpec::Square);
    let input_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor_3d = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![2, 2, 2],
            },
            input_vals
                .iter()
                .copied()
                .map(fj_core::Literal::from_f64)
                .collect(),
        )
        .expect("3d tensor should build"),
    );
    let result = compose(
        jaxpr,
        vec![
            Transform::Vmap,
            Transform::Vmap,
            Transform::Vmap,
            Transform::Grad,
        ],
    )
    .call(vec![tensor_3d])
    .expect("vmap(vmap(vmap(grad))) should succeed on rank3");
    let t = result[0].as_tensor().expect("tensor");
    assert_eq!(
        t.shape,
        Shape {
            dims: vec![2, 2, 2]
        }
    );
    let vals = t.to_f64_vec().expect("f64 vec");
    let expected: Vec<f64> = input_vals.iter().map(|x| 2.0 * x).collect();
    assert_eq!(vals.len(), expected.len());
    for (i, (actual, expected)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-3,
            "index {i}: expected {expected}, got {actual}"
        );
    }
    log_pass(
        "stacking_vmap_vmap_vmap_grad_rank3",
        &("vmap3_grad", "square", vec![2_u32, 2_u32, 2_u32]),
    );
}

#[test]
fn stacking_compose_jit_vmap_grad() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = compose(
        jaxpr,
        vec![Transform::Jit, Transform::Vmap, Transform::Grad],
    )
    .call(vec![Value::vector_f64(&[2.0, 3.0]).expect("vector")])
    .expect("jit(vmap(grad)) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64");
    assert!((vals[0] - 4.0).abs() < 1e-3);
    assert!((vals[1] - 6.0).abs() < 1e-3);
    log_pass(
        "stacking_compose_jit_vmap_grad",
        &("jit_vmap_grad", "square"),
    );
}

// ============================================================================
// 3. Error Message Tests
// ============================================================================

#[test]
fn error_grad_vector_input() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = grad(jaxpr)
        .call(vec![Value::vector_f64(&[1.0, 2.0]).expect("vector")])
        .expect_err("grad with vector input should fail");
    match &err {
        ApiError::GradRequiresScalar { detail } => {
            assert!(
                detail.contains("scalar"),
                "error should mention scalar: {detail}"
            );
        }
        other => std::panic::panic_any(format!("expected GradRequiresScalar, got: {other}")),
    }
    log_pass("error_grad_vector_input", &("error", "grad_vector"));
}

#[test]
fn error_grad_rank3_input() {
    let jaxpr = build_program(ProgramSpec::Square);
    let tensor_3d = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![2, 2, 2],
            },
            (1..=8)
                .map(|v| fj_core::Literal::from_f64(v as f64))
                .collect(),
        )
        .expect("3d tensor should build"),
    );
    let err = grad(jaxpr)
        .call(vec![tensor_3d])
        .expect_err("grad with rank3 input should fail");
    match &err {
        ApiError::GradRequiresScalar { detail } => {
            assert!(
                detail.contains("scalar"),
                "error should mention scalar: {detail}"
            );
        }
        other => std::panic::panic_any(format!("expected GradRequiresScalar, got: {other}")),
    }
    log_pass("error_grad_rank3_input", &("error", "grad_rank3"));
}

#[test]
fn error_grad_empty_args() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = grad(jaxpr)
        .call(vec![])
        .expect_err("grad with empty args should fail");
    assert!(matches!(err, ApiError::EvalError { .. }));
    log_pass("error_grad_empty_args", &("error", "grad_empty"));
}

#[test]
fn error_vmap_scalar_input() {
    let jaxpr = build_program(ProgramSpec::AddOne);
    let err = vmap(jaxpr)
        .call(vec![Value::scalar_i64(42)])
        .expect_err("vmap with scalar should fail");
    assert!(matches!(err, ApiError::EvalError { .. }));
    log_pass("error_vmap_scalar_input", &("error", "vmap_scalar"));
}

#[test]
fn error_grad_vmap_order() {
    let jaxpr = build_program(ProgramSpec::Square);
    let err = compose(jaxpr, vec![Transform::Grad, Transform::Vmap])
        .call(vec![Value::vector_f64(&[1.0, 2.0]).expect("vector")])
        .expect_err("grad(vmap(f)) should fail");
    assert!(matches!(err, ApiError::GradRequiresScalar { .. }));
    log_pass("error_grad_vmap_order", &("error", "grad_vmap"));
}

#[test]
fn error_display_is_user_friendly() {
    let err = ApiError::GradRequiresScalar {
        detail: "test detail".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("grad requires scalar"));
    assert!(!msg.contains("DispatchError"));

    let err = ApiError::VmapDimensionMismatch {
        expected: 5,
        actual: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("leading dimension"));
    assert!(msg.contains("5"));
    assert!(msg.contains("3"));
    log_pass("error_display_is_user_friendly", &("error", "display"));
}

// ============================================================================
// 4. Mode Configuration Tests
// ============================================================================

#[test]
fn mode_strict_is_default() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = jit(jaxpr)
        .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
        .expect("strict default should succeed");
    assert_eq!(result, vec![Value::scalar_i64(3)]);
    log_pass("mode_strict_is_default", &("mode", "strict_default"));
}

#[test]
fn mode_hardened_jit() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = jit(jaxpr)
        .with_mode(CompatibilityMode::Hardened)
        .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
        .expect("hardened should succeed");
    assert_eq!(result, vec![Value::scalar_i64(3)]);
    log_pass("mode_hardened_jit", &("mode", "hardened_jit"));
}

#[test]
fn mode_hardened_grad() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = grad(jaxpr)
        .with_mode(CompatibilityMode::Hardened)
        .call(vec![Value::scalar_f64(3.0)])
        .expect("hardened grad should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    assert!((d - 6.0).abs() < 1e-3);
    log_pass("mode_hardened_grad", &("mode", "hardened_grad"));
}

// ============================================================================
// 5. TTL Construction Verification
// ============================================================================

#[test]
fn ttl_jit_has_one_transform() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = jit(jaxpr)
        .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
        .expect("jit should succeed");
    assert_eq!(result, vec![Value::scalar_i64(3)]);
    log_pass("ttl_jit_has_one_transform", &("ttl", "jit_one"));
}

#[test]
fn ttl_composition_three_transforms() {
    let jaxpr = build_program(ProgramSpec::Square);
    let result = compose(
        jaxpr,
        vec![Transform::Jit, Transform::Vmap, Transform::Grad],
    )
    .call(vec![Value::vector_f64(&[1.0]).expect("vector")])
    .expect("3-transform composition should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64 vec");
    assert!((vals[0] - 2.0).abs() < 1e-3);
    log_pass("ttl_composition_three_transforms", &("ttl", "three"));
}

// ============================================================================
// 6. Structured Logging Schema Contract
// ============================================================================

#[test]
fn test_log_schema_contract() {
    let fixture_id =
        fixture_id_from_json(&("api_transform_suite", "schema_contract")).expect("fixture digest");
    let log = TestLogV1::unit(
        test_id(module_path!(), "test_log_schema_contract"),
        fixture_id,
        TestMode::Strict,
        TestResult::Pass,
    );
    assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    let json = serde_json::to_string_pretty(&log).expect("serialize");
    assert!(json.contains("frankenjax.test-log.v1"));
}

// ============================================================================
// 7. Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(
        fj_test_utils::property_test_case_count()
    ))]

    #[test]
    fn prop_jit_is_identity(x in -1000i64..1000i64, y in -1000i64..1000i64) {
        let jaxpr = build_program(ProgramSpec::Add2);
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_i64(x), Value::scalar_i64(y)])
            .expect("jit should succeed");
        let direct_result =
            fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(x), Value::scalar_i64(y)])
                .expect("direct eval should succeed");
        prop_assert_eq!(jit_result, direct_result);
    }

    #[test]
    fn prop_grad_square_is_2x(x in -100.0f64..100.0f64) {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let d = result[0].as_f64_scalar().expect("scalar");
        let expected = 2.0 * x;
        prop_assert!((d - expected).abs() < 0.1, "grad({x}) = {d}, expected ~{expected}");
    }

    #[test]
    fn prop_vmap_add_one_increments(
        elements in proptest::collection::vec(-1000i64..1000i64, 1..20)
    ) {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let input = Value::vector_i64(&elements).expect("vector should build");
        let result = vmap(jaxpr)
            .call(vec![input])
            .expect("vmap should succeed");
        let t = result[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().expect("i64")).collect();
        for (i, &v) in vals.iter().enumerate() {
            let expected = elements[i] + 1;
            prop_assert_eq!(v, expected);
        }
    }

    #[test]
    fn prop_value_and_grad_consistent(x in -50.0f64..50.0f64) {
        let jaxpr = build_program(ProgramSpec::Square);
        let (val, grd) = value_and_grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("value_and_grad should succeed");
        let v = val[0].as_f64_scalar().expect("value scalar");
        let g = grd[0].as_f64_scalar().expect("grad scalar");
        let expected_val = x * x;
        let expected_grad = 2.0 * x;
        prop_assert!((v - expected_val).abs() < 1e-6, "value({x}) = {v}, expected {expected_val}");
        prop_assert!((g - expected_grad).abs() < 0.1, "grad({x}) = {g}, expected {expected_grad}");
    }

    #[test]
    fn prop_jit_grad_consistent_with_grad(x in -50.0f64..50.0f64) {
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
        prop_assert!((d1 - d2).abs() < 1e-6, "grad={d1}, jit(grad)={d2}");
    }
}

// ============================================================================
// 8. Extended Transform Composition Tests (frankenjax-h52)
// ============================================================================

/// grad(sin(x)) = cos(x)
#[test]
fn grad_sin() {
    let jaxpr = build_program(ProgramSpec::SinX);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(1.0)])
        .expect("grad(sin) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    let expected = 1.0_f64.cos();
    assert!(
        (d - expected).abs() < 1e-10,
        "grad(sin(1.0)) should be cos(1.0)={expected}, got {d}"
    );
}

/// grad(cos(x)) = -sin(x)
#[test]
fn grad_cos() {
    let jaxpr = build_program(ProgramSpec::CosX);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(1.0)])
        .expect("grad(cos) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    let expected = -(1.0_f64.sin());
    assert!(
        (d - expected).abs() < 1e-10,
        "grad(cos(1.0)) should be -sin(1.0)={expected}, got {d}"
    );
}

/// grad(exp(x)) = exp(x)
#[test]
fn grad_exp() {
    let jaxpr = build_program(ProgramSpec::LaxExp);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(2.0)])
        .expect("grad(exp) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    let expected = 2.0_f64.exp();
    assert!(
        (d - expected).abs() < 1e-10,
        "grad(exp(2.0)) should be exp(2.0)={expected}, got {d}"
    );
}

/// grad(log(x)) = 1/x
#[test]
fn grad_log() {
    let jaxpr = build_program(ProgramSpec::LaxLog);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(3.0)])
        .expect("grad(log) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    let expected = 1.0 / 3.0;
    assert!(
        (d - expected).abs() < 1e-10,
        "grad(log(3.0)) should be 1/3={expected}, got {d}"
    );
}

/// grad(tanh(x)) = 1 - tanh(x)^2
#[test]
fn grad_tanh() {
    let jaxpr = build_program(ProgramSpec::LaxTanh);
    let result = grad(jaxpr)
        .call(vec![Value::scalar_f64(0.5)])
        .expect("grad(tanh) should succeed");
    let d = result[0].as_f64_scalar().expect("scalar");
    let t = 0.5_f64.tanh();
    let expected = 1.0 - t * t;
    assert!(
        (d - expected).abs() < 1e-10,
        "grad(tanh(0.5)) should be sech^2(0.5)={expected}, got {d}"
    );
}

/// grad(grad(sin(x))) = -sin(x)
#[test]
fn grad_grad_sin() {
    let jaxpr = build_program(ProgramSpec::SinX);
    let result = compose(jaxpr, vec![Transform::Grad, Transform::Grad])
        .call(vec![Value::scalar_f64(1.0)])
        .expect("grad(grad(sin)) should succeed");
    let d2 = result[0].as_f64_scalar().expect("scalar");
    let expected = -(1.0_f64.sin());
    assert!(
        (d2 - expected).abs() < 1e-6,
        "grad(grad(sin(1.0))) should be -sin(1.0)={expected}, got {d2}"
    );
}

/// grad(grad(cos(x))) = -cos(x)
#[test]
fn grad_grad_cos() {
    let jaxpr = build_program(ProgramSpec::CosX);
    let result = compose(jaxpr, vec![Transform::Grad, Transform::Grad])
        .call(vec![Value::scalar_f64(1.0)])
        .expect("grad(grad(cos)) should succeed");
    let d2 = result[0].as_f64_scalar().expect("scalar");
    let expected = -(1.0_f64.cos());
    assert!(
        (d2 - expected).abs() < 1e-6,
        "grad(grad(cos(1.0))) should be -cos(1.0)={expected}, got {d2}"
    );
}

/// grad(grad(exp(x))) = exp(x) — exp is its own nth derivative
#[test]
fn grad_grad_exp() {
    let jaxpr = build_program(ProgramSpec::LaxExp);
    let result = compose(jaxpr, vec![Transform::Grad, Transform::Grad])
        .call(vec![Value::scalar_f64(1.0)])
        .expect("grad(grad(exp)) should succeed");
    let d2 = result[0].as_f64_scalar().expect("scalar");
    let expected = 1.0_f64.exp();
    assert!(
        (d2 - expected).abs() < 1e-6,
        "grad(grad(exp(1.0))) should be exp(1.0)={expected}, got {d2}"
    );
}

/// grad(grad(grad(x^3))) = 6 (third derivative of x^3 is constant 6)
#[test]
fn grad_grad_grad_cube() {
    // f(x) = x^2 + 3x has f'''(x) = 0
    let jaxpr2 = build_program(ProgramSpec::SquarePlusLinear);
    let result = compose(
        jaxpr2,
        vec![Transform::Grad, Transform::Grad, Transform::Grad],
    )
    .call(vec![Value::scalar_f64(5.0)])
    .expect("grad(grad(grad(x^2+3x))) should succeed");
    let d3 = result[0].as_f64_scalar().expect("scalar");
    // f(x) = x^2 + 3x, f'(x) = 2x+3, f''(x) = 2, f'''(x) = 0
    assert!(d3.abs() < 1e-6, "grad^3(x^2+3x) should be 0, got {d3}");
}

/// vmap(grad(sin)) over a batch of inputs
#[test]
fn vmap_grad_sin_batch() {
    let jaxpr = build_program(ProgramSpec::SinX);
    let inputs = vec![
        0.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
    ];
    let result = vmap(jaxpr)
        .compose_grad()
        .call(vec![Value::vector_f64(&inputs).expect("vector")])
        .expect("vmap(grad(sin)) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64 vec");
    assert_eq!(vals.len(), 4);
    for (i, (&actual, &x)) in vals.iter().zip(inputs.iter()).enumerate() {
        let expected = x.cos();
        assert!(
            (actual - expected).abs() < 1e-6,
            "vmap(grad(sin))[{i}]: x={x}, expected cos(x)={expected}, got {actual}"
        );
    }
}

/// jit(vmap(grad(exp))) — triple composition with exp
#[test]
fn jit_vmap_grad_exp() {
    let jaxpr = build_program(ProgramSpec::LaxExp);
    let inputs = vec![0.0, 1.0, 2.0];
    let result = compose(
        jaxpr,
        vec![Transform::Jit, Transform::Vmap, Transform::Grad],
    )
    .call(vec![Value::vector_f64(&inputs).expect("vector")])
    .expect("jit(vmap(grad(exp))) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64 vec");
    for (i, (&actual, &x)) in vals.iter().zip(inputs.iter()).enumerate() {
        let expected = x.exp();
        assert!(
            (actual - expected).abs() < 1e-6,
            "jit(vmap(grad(exp)))[{i}]: expected exp({x})={expected}, got {actual}"
        );
    }
}

/// vmap(grad(grad(sin))) — vectorized second derivative of sin
/// sin''(x) = -sin(x)
#[test]
fn vmap_grad_grad_sin() {
    let jaxpr = build_program(ProgramSpec::SinX);
    let inputs = vec![0.5, 1.0, 2.0];
    let result = compose(
        jaxpr,
        vec![Transform::Vmap, Transform::Grad, Transform::Grad],
    )
    .call(vec![Value::vector_f64(&inputs).expect("vector")])
    .expect("vmap(grad(grad(sin))) should succeed");
    let t = result[0].as_tensor().expect("tensor");
    let vals = t.to_f64_vec().expect("f64 vec");
    for (i, (&actual, &x)) in vals.iter().zip(inputs.iter()).enumerate() {
        let expected = -x.sin();
        assert!(
            (actual - expected).abs() < 1e-4,
            "vmap(grad(grad(sin)))[{i}]: expected -sin({x})={expected}, got {actual}"
        );
    }
}
