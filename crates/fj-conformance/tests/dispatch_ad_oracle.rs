#![forbid(unsafe_code)]

//! FJ-P2C-004-F: Differential Oracle + Metamorphic + Adversarial Validation
//! for dispatch/AD/effects runtime.
//!
//! Covers:
//! - Oracle: reverse-mode grad vs finite-diff vs analytical (three-way)
//! - Oracle: forward-mode JVP vs reverse-mode VJP agreement
//! - Oracle: dispatch routing correctness
//! - Metamorphic: linearity of differentiation, grad of constant, AD/finite-diff cross-check
//! - Adversarial: non-scalar grad, empty args, mismatched vmap dims, composition violations

use fj_ad::{grad_first, jvp_grad_first};
use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, Primitive, ProgramSpec, TraceTransformLedger,
    Transform, Value, VarId, build_program,
};
use fj_dispatch::{dispatch, DispatchError, DispatchRequest, EffectContext, TransformExecutionError};
use fj_interpreters::eval_jaxpr;
use fj_test_utils::{TestLogV1, TestMode, TestResult, fixture_id_from_json, test_id};
use std::collections::BTreeMap;

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

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(*transform, format!("evidence-{}-{}", transform.as_str(), idx));
    }
    ledger
}

fn make_sin_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Sin,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
        }],
    )
}

fn make_exp_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Exp,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
        }],
    )
}

// ============================================================================
// 1. Oracle Comparison Points (three-way: analytical, reverse-AD, forward-JVP)
// ============================================================================

/// Oracle 1: grad(x²) = 2x — three-way comparison at multiple points.
#[test]
fn oracle_grad_square_three_way() {
    let jaxpr = build_program(ProgramSpec::Square);
    let test_points = [0.0, 1.0, -1.0, 2.5, -7.3, 100.0];

    for x in test_points {
        let analytical = 2.0 * x;
        let reverse = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let forward = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();

        assert!(
            (reverse - analytical).abs() < 1e-6,
            "reverse-mode: x={x}, got {reverse}, expected {analytical}"
        );
        assert!(
            (forward - analytical).abs() < 1e-6,
            "forward-mode: x={x}, got {forward}, expected {analytical}"
        );
        assert!(
            (forward - reverse).abs() < 1e-10,
            "forward/reverse mismatch: x={x}, fwd={forward}, rev={reverse}"
        );
    }

    log_oracle(
        "oracle_grad_square_three_way",
        &("grad_square", "three_way", test_points.len()),
    );
}

/// Oracle 2: grad(sin(x)) = cos(x) — three-way at multiple points.
#[test]
fn oracle_grad_sin_three_way() {
    let jaxpr = make_sin_jaxpr();
    let test_points: [f64; 4] = [0.0, 1.0, -1.0, 2.71];

    for x in test_points {
        let analytical = x.cos();
        let reverse = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let forward = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();

        assert!(
            (reverse - analytical).abs() < 1e-6,
            "reverse-mode: x={x}, got {reverse}, expected {analytical}"
        );
        assert!(
            (forward - analytical).abs() < 1e-6,
            "forward-mode: x={x}, got {forward}, expected {analytical}"
        );
    }

    log_oracle("oracle_grad_sin_three_way", &("grad_sin", "three_way"));
}

/// Oracle 3: grad(exp(x)) = exp(x) — three-way.
#[test]
fn oracle_grad_exp_three_way() {
    let jaxpr = make_exp_jaxpr();
    let test_points: [f64; 4] = [0.0, 1.0, -1.0, 2.0];

    for x in test_points {
        let analytical = x.exp();
        let reverse = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let forward = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();

        assert!(
            (reverse - analytical).abs() < 1e-6,
            "reverse-mode: x={x}, got {reverse}, expected {analytical}"
        );
        assert!(
            (forward - analytical).abs() < 1e-6,
            "forward-mode: x={x}, got {forward}, expected {analytical}"
        );
    }

    log_oracle("oracle_grad_exp_three_way", &("grad_exp", "three_way"));
}

/// Oracle 4: grad(x² + 2x) = 2x + 2 — multi-equation jaxpr.
#[test]
fn oracle_grad_polynomial_three_way() {
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    let test_points = [0.0, 1.0, -3.0, 5.5];

    for x in test_points {
        let analytical = 2.0 * x + 2.0;
        let reverse = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let forward = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();

        assert!(
            (reverse - analytical).abs() < 1e-6,
            "reverse-mode: x={x}, got {reverse}, expected {analytical}"
        );
        assert!(
            (forward - analytical).abs() < 1e-6,
            "forward-mode: x={x}, got {forward}, expected {analytical}"
        );
    }

    log_oracle(
        "oracle_grad_polynomial_three_way",
        &("grad_polynomial", "three_way"),
    );
}

/// Oracle 5: Dispatch routing — jit(f)(x) == f(x) (JIT is identity).
#[test]
fn oracle_dispatch_jit_identity() {
    let programs = [
        (ProgramSpec::Add2, vec![Value::scalar_i64(3), Value::scalar_i64(7)]),
        (ProgramSpec::Square, vec![Value::scalar_f64(5.0)]),
        (ProgramSpec::AddOne, vec![Value::scalar_i64(42)]),
    ];

    for (program, args) in &programs {
        let direct = eval_jaxpr(&build_program(*program), args).unwrap();

        let jit_response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(*program, &[Transform::Jit]),
            args: args.clone(),
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .unwrap();

        assert_eq!(
            jit_response.outputs, direct,
            "jit(f)(x) != f(x) for {:?}",
            program
        );
    }

    log_oracle("oracle_dispatch_jit_identity", &("jit_identity", 3));
}

// ============================================================================
// 2. Metamorphic Properties
// ============================================================================

/// Metamorphic: grad(f + g) == grad(f) + grad(g) — linearity of differentiation.
/// Since we can't directly add jaxprs, we verify that the derivative of a sum
/// equals the sum of derivatives using known programs.
#[test]
fn metamorphic_grad_linearity() {
    // f(x) = x², g(x) = 2x => f+g = x² + 2x
    // grad(f)(3) = 6, grad(g)(3) = 2, grad(f+g)(3) = 8
    let x = 3.0;
    let grad_f = grad_first(&build_program(ProgramSpec::Square), &[Value::scalar_f64(x)]).unwrap();
    let grad_fg = grad_first(
        &build_program(ProgramSpec::SquarePlusLinear),
        &[Value::scalar_f64(x)],
    )
    .unwrap();
    // grad(g)(x) = 2 for g(x) = 2x
    let grad_g = 2.0;

    assert!(
        (grad_fg - (grad_f + grad_g)).abs() < 1e-6,
        "linearity: grad(f+g) = {grad_fg}, grad(f)+grad(g) = {}",
        grad_f + grad_g
    );

    log_oracle("metamorphic_grad_linearity", &("linearity", x));
}

/// Metamorphic: grad of constant function is zero.
#[test]
fn metamorphic_grad_constant_is_zero() {
    // Build constant function: f(x) = 5 (uses Mul by 0 + Add with literal)
    // Simpler: just check that grad of AddOne at any x has derivative = 0
    // since AddOne(x) = x + 1, grad = 1. Instead use a known-zero case.
    // grad(x * 0) = 0 at any x — but we don't have a "zero" program.
    // Alternative: verify grad(floor(x)) is zero since floor has zero derivative.
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Floor,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
        }],
    );
    let grad = grad_first(&jaxpr, &[Value::scalar_f64(2.5)]).unwrap();
    assert!(
        grad.abs() < 1e-10,
        "grad(floor(x)) should be 0, got {grad}"
    );

    log_oracle("metamorphic_grad_constant_is_zero", &("constant_zero",));
}

/// Metamorphic: AD matches finite-diff within tolerance.
#[test]
fn metamorphic_ad_matches_finite_diff() {
    let jaxpr = build_program(ProgramSpec::Square);
    let test_points = [1.0, 2.0, 5.0, -3.0, 0.5];
    let eps = 1e-6;

    for x in test_points {
        let ad_grad = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();

        let plus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x + eps)])
            .unwrap()[0]
            .as_f64_scalar()
            .unwrap();
        let minus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x - eps)])
            .unwrap()[0]
            .as_f64_scalar()
            .unwrap();
        let fd_grad = (plus - minus) / (2.0 * eps);

        assert!(
            (ad_grad - fd_grad).abs() < 1e-4,
            "x={x}: AD={ad_grad}, finite-diff={fd_grad}"
        );
    }

    log_oracle(
        "metamorphic_ad_matches_finite_diff",
        &("ad_vs_fd", test_points.len()),
    );
}

/// Metamorphic: Effect tokens consumed in declaration order.
#[test]
fn metamorphic_effect_token_ordering() {
    let mut ctx = EffectContext::new();
    let effects = ["ordered_print", "ordered_random", "unordered_log"];
    for name in &effects {
        ctx.thread_token(name);
    }

    let tokens = ctx.finalize();
    assert_eq!(tokens.len(), 3);
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(token.effect_name, effects[i]);
        assert_eq!(token.sequence_number, i as u64);
    }

    log_oracle(
        "metamorphic_effect_token_ordering",
        &("token_order", effects.len()),
    );
}

// ============================================================================
// 3. Adversarial Cases
// ============================================================================

/// Adversarial: grad of non-scalar input rejected with clear error.
#[test]
fn adversarial_grad_nonsalar_input() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
        args: vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::NonScalarGradientInput)
    ));
    log_oracle("adversarial_grad_nonscalar_input", &("nonscalar_grad",));
}

/// Adversarial: empty argument list for grad.
#[test]
fn adversarial_grad_empty_args() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
        args: vec![],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::EmptyArgumentList { .. })
    ));
    log_oracle("adversarial_grad_empty_args", &("empty_grad_args",));
}

/// Adversarial: vmap with mismatched leading dimensions.
#[test]
fn adversarial_vmap_mismatched_dims() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
        args: vec![
            Value::vector_i64(&[1, 2, 3]).unwrap(),
            Value::vector_i64(&[1, 2]).unwrap(), // mismatched: 3 vs 2
        ],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(
            TransformExecutionError::VmapMismatchedLeadingDimension { .. }
        )
    ));
    log_oracle(
        "adversarial_vmap_mismatched_dims",
        &("mismatched_dims", 3, 2),
    );
}

/// Adversarial: vmap with scalar input (rank 0).
#[test]
fn adversarial_vmap_scalar_input() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
        args: vec![Value::scalar_i64(42)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(
            TransformExecutionError::VmapRequiresRankOneLeadingArgument
        )
    ));
    log_oracle("adversarial_vmap_scalar_input", &("scalar_vmap",));
}

/// Adversarial: double grad rejected by composition proof.
#[test]
fn adversarial_double_grad_rejected() {
    let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    ttl.push_transform(Transform::Grad, "grad-1".to_owned());
    ttl.push_transform(Transform::Grad, "grad-2".to_owned());

    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![Value::scalar_f64(3.0)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(
        matches!(err, DispatchError::TransformInvariant(_)),
        "double grad should fail composition proof, got: {err:?}"
    );
    log_oracle("adversarial_double_grad_rejected", &("double_grad",));
}

/// Adversarial: double vmap rejected by composition proof.
#[test]
fn adversarial_double_vmap_rejected() {
    let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::AddOne));
    ttl.push_transform(Transform::Vmap, "vmap-1".to_owned());
    ttl.push_transform(Transform::Vmap, "vmap-2".to_owned());

    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(
        matches!(err, DispatchError::TransformInvariant(_)),
        "double vmap should fail composition proof, got: {err:?}"
    );
    log_oracle("adversarial_double_vmap_rejected", &("double_vmap",));
}

/// Adversarial: evidence count mismatch rejected.
#[test]
fn adversarial_evidence_mismatch() {
    let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Add2));
    // Push transform without matching evidence
    ttl.transform_stack.push(Transform::Jit);
    // Don't push evidence — creates mismatch

    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap_err();
    assert!(
        matches!(err, DispatchError::TransformInvariant(_)),
        "evidence mismatch should fail composition proof, got: {err:?}"
    );
    log_oracle("adversarial_evidence_mismatch", &("evidence_mismatch",));
}
