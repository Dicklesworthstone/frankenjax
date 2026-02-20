#![forbid(unsafe_code)]

//! Comprehensive unit + property tests for dispatch/effects runtime (FJ-P2C-004-E).
//! Covers transform dispatch, gradient accuracy, effect tokens, cache integration,
//! evidence ledger, and property-based dispatch determinism.

use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, Primitive, ProgramSpec, TraceTransformLedger,
    Transform, Value, VarId, build_program,
};
use fj_dispatch::{
    DispatchError, DispatchRequest, EffectContext, TransformExecutionError, dispatch,
};
use std::collections::BTreeMap;

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("evidence-{}-{}", transform.as_str(), idx),
        );
    }
    ledger
}

fn make_request(
    program: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
) -> DispatchRequest {
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(program, transforms),
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

// ── 1. Transform Dispatch Tests ────────────────────────────────────

#[test]
fn jit_scalar_add() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(10), Value::scalar_i64(20)],
    ))
    .unwrap();
    assert_eq!(r.outputs, vec![Value::scalar_i64(30)]);
}

#[test]
fn jit_vector_add_one() {
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Jit],
        vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![2, 3, 4]);
}

#[test]
fn grad_polynomial_square_at_3() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (derivative - 6.0).abs() < 1e-3,
        "d/dx(x²) at 3 = 6, got {derivative}"
    );
}

#[test]
fn grad_square_plus_linear_at_2() {
    let r = dispatch(make_request(
        ProgramSpec::SquarePlusLinear,
        &[Transform::Grad],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    // d/dx(x² + 2x) at x=2 = 2*2 + 2 = 6
    assert!((derivative - 6.0).abs() < 1e-3, "got {derivative}");
}

#[test]
fn vmap_varying_batch_sizes() {
    for size in [1, 5, 10] {
        let data: Vec<i64> = (1..=size).collect();
        let r = dispatch(make_request(
            ProgramSpec::AddOne,
            &[Transform::Vmap],
            vec![Value::vector_i64(&data).unwrap()],
        ))
        .unwrap();
        let out = r.outputs[0].as_tensor().unwrap();
        let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        let expected: Vec<i64> = data.iter().map(|x| x + 1).collect();
        assert_eq!(elems, expected, "batch_size={size}");
    }
}

#[test]
fn jit_grad_composition() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(5.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (derivative - 10.0).abs() < 1e-3,
        "jit(grad(x²)) at 5 = 10, got {derivative}"
    );
}

#[test]
fn vmap_grad_batch_gradient() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap().to_f64_vec().unwrap();
    assert_eq!(out.len(), 3);
    assert!((out[0] - 2.0).abs() < 1e-3);
    assert!((out[1] - 4.0).abs() < 1e-3);
    assert!((out[2] - 6.0).abs() < 1e-3);
}

#[test]
fn jit_vmap_batch_execution() {
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Jit, Transform::Vmap],
        vec![Value::vector_i64(&[10, 20, 30]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![11, 21, 31]);
}

#[test]
fn invalid_grad_vmap_rejected() {
    // grad(vmap(f)) should fail: grad requires scalar input, gets vector
    let err = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Grad, Transform::Vmap],
        vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::NonScalarGradientInput)
    ));
}

#[test]
fn invalid_empty_args_grad() {
    let err = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::EmptyArgumentList { .. })
    ));
}

#[test]
fn invalid_empty_args_vmap() {
    let err = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::EmptyArgumentList { .. })
    ));
}

// ── 2. Gradient Accuracy Tests ─────────────────────────────────────

#[test]
fn grad_sin_at_zero_is_one() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Sin,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
        }],
    );
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Grad, "grad-sin".to_owned());
    let r = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![Value::scalar_f64(0.0)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();
    let deriv = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (deriv - 1.0).abs() < 1e-3,
        "d/dx(sin(x)) at 0 = cos(0) = 1, got {deriv}"
    );
}

#[test]
fn grad_numerical_vs_analytical_comparison() {
    // grad(x²) at x=7 should be 14 (analytical)
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(7.0)],
    ))
    .unwrap();
    let analytical = r.outputs[0].as_f64_scalar().unwrap();

    // Numerical: (f(x+eps) - f(x-eps)) / (2*eps)
    let eps = 1e-6;
    let x = 7.0;
    let plus: f64 = (x + eps) * (x + eps);
    let minus: f64 = (x - eps) * (x - eps);
    let numerical = (plus - minus) / (2.0 * eps);

    assert!(
        (analytical - numerical).abs() < 1e-4,
        "analytical={analytical}, numerical={numerical}"
    );
}

// ── 3. Effect Token Tests ──────────────────────────────────────────

#[test]
fn effect_context_empty() {
    let ctx = EffectContext::new();
    assert_eq!(ctx.effect_count(), 0);
    let tokens = ctx.finalize();
    assert!(tokens.is_empty());
}

#[test]
fn effect_context_duplicate_overwrites() {
    let mut ctx = EffectContext::new();
    let t1 = ctx.thread_token("jit");
    let t2 = ctx.thread_token("jit"); // same name, new sequence
    assert_eq!(t1.sequence_number, 0);
    assert_eq!(t2.sequence_number, 1);
    // BTreeMap overwrites: only one "jit" entry remains
    assert_eq!(ctx.effect_count(), 1);
    let tokens = ctx.finalize();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].sequence_number, 1); // latest wins
}

#[test]
fn effect_tokens_in_dispatch_single_transform() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "effect_token_count")
        .unwrap();
    assert_eq!(signal.detail, "effect_tokens=1");
}

#[test]
fn effect_tokens_in_dispatch_triple_transform() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "effect_token_count")
        .unwrap();
    assert_eq!(signal.detail, "effect_tokens=3");
}

// ── 4. Cache Integration Tests ─────────────────────────────────────

#[test]
fn same_input_produces_same_cache_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    assert_eq!(r1.cache_key, r2.cache_key);
}

#[test]
fn different_program_produces_different_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit],
        vec![Value::scalar_f64(1.0)],
    ))
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key);
}

#[test]
fn different_transforms_produce_different_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key);
}

#[test]
fn strict_rejects_unknown_features() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["unknown.feature.v3".to_owned()],
    })
    .unwrap_err();
    assert!(matches!(err, DispatchError::Cache(_)));
}

#[test]
fn hardened_includes_features_in_key() {
    let r1 = dispatch(DispatchRequest {
        mode: CompatibilityMode::Hardened,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();
    let r2 = dispatch(DispatchRequest {
        mode: CompatibilityMode::Hardened,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["feature.x".to_owned()],
    })
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key, "features should affect key");
}

// ── 5. Evidence Ledger Tests ───────────────────────────────────────

#[test]
fn every_dispatch_produces_ledger_entry() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    assert_eq!(r.evidence_ledger.len(), 1);
}

#[test]
fn ledger_entry_contains_all_signals() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    assert_eq!(entry.signals.len(), 4);

    let signal_names: Vec<&str> = entry
        .signals
        .iter()
        .map(|s| s.signal_name.as_str())
        .collect();
    assert!(signal_names.contains(&"eqn_count"));
    assert!(signal_names.contains(&"transform_depth"));
    assert!(signal_names.contains(&"transform_stack_hash"));
    assert!(signal_names.contains(&"effect_token_count"));
}

#[test]
fn ledger_decision_id_matches_cache_key() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    assert_eq!(entry.decision_id, r.cache_key);
}

#[test]
fn ledger_transform_depth_signal_correct() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let depth_signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "transform_depth")
        .unwrap();
    assert_eq!(depth_signal.detail, "transform_depth=3");
}

// ── 6. Property Tests ──────────────────────────────────────────────

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]

        #[test]
        fn prop_dispatch_deterministic(x in prop::num::f64::NORMAL.prop_filter(
            "finite",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r1 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 1");
            let r2 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 2");
            prop_assert_eq!(r1.cache_key, r2.cache_key);
            prop_assert_eq!(r1.outputs.len(), r2.outputs.len());
            let v1 = r1.outputs[0].as_f64_scalar().unwrap();
            let v2 = r2.outputs[0].as_f64_scalar().unwrap();
            prop_assert!((v1 - v2).abs() < 1e-12, "non-deterministic: {v1} vs {v2}");
        }

        #[test]
        fn prop_cache_key_stability(x in prop::num::f64::NORMAL.prop_filter(
            "finite",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            // Cache key should be independent of argument values
            // (it depends on program structure, not runtime values)
            let r1 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 1");
            let r2 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x + 1.0)],
            )).expect("dispatch 2");
            // Same program + transforms = same cache key regardless of args
            prop_assert_eq!(r1.cache_key, r2.cache_key);
        }

        #[test]
        fn prop_ledger_always_populated(x in prop::num::i64::ANY.prop_filter(
            "not extreme",
            |x| x.abs() < i64::MAX / 2
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r = dispatch(make_request(
                ProgramSpec::Add2,
                &[Transform::Jit],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("dispatch");
            prop_assert_eq!(r.evidence_ledger.len(), 1);
            prop_assert_eq!(r.evidence_ledger.entries()[0].signals.len(), 4);
        }

        #[test]
        fn prop_grad_is_2x_for_square(x in prop::num::f64::NORMAL.prop_filter(
            "finite and moderate",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch");
            let deriv = r.outputs[0].as_f64_scalar().unwrap();
            prop_assert!((deriv - 2.0 * x).abs() < 1e-3, "d/dx(x²) at {x}: got {deriv}");
        }

        #[test]
        fn prop_jit_is_identity(x in prop::num::i64::ANY.prop_filter(
            "not extreme",
            |x| x.abs() < i64::MAX / 2
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            // jit(add2)(x, 1) should equal add2(x, 1) without jit
            let with_jit = dispatch(make_request(
                ProgramSpec::Add2,
                &[Transform::Jit],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("with jit");
            let without_jit = dispatch(make_request(
                ProgramSpec::Add2,
                &[],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("without jit");
            prop_assert_eq!(with_jit.outputs, without_jit.outputs);
        }
    }
}
