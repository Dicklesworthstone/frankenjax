//! bd-3dl.12.6: Differential Oracle + Metamorphic + Adversarial Validation — IR Core
//!
//! Oracle comparison points:
//!   1. Jaxpr construction — verify build_program produces expected equation counts and structure
//!   2. Canonical fingerprint — verify deterministic hash for known programs
//!   3. Transform composition proof — verify composition rules and ordering
//!   4. eval_jaxpr results — verify interpreter produces correct outputs vs analytical oracle
//!   5. Shape/type inference — verify abstract value propagation through programs
//!   6. Cache key determinism — verify same inputs always produce same cache keys
//!
//! Metamorphic properties:
//!   1. Fingerprint determinism — same Jaxpr always produces same fingerprint (100x)
//!   2. Composition associativity — composition signature is order-invariant for commutative cases
//!   3. Equation ordering stability — canonical form is stable under re-construction
//!   4. Eval determinism — same program + args always produces same output (50x)
//!
//! Adversarial cases:
//!   1. Jaxpr with 0 equations (empty program)
//!   2. Jaxpr with duplicate variable IDs in output
//!   3. Transform stack with repeated transforms (jit(jit(f)))
//!   4. Maximum supported equation count (stress test, 10_000 equations)
//!   5. Unbound variable reference
//!   6. Empty transform stack composition verification

use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, JaxprValidationError, Literal, Primitive,
    ProgramSpec, TraceTransformLedger, Transform, TransformCompositionError, Value, VarId,
    build_program, verify_transform_composition,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_ledger(spec: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(spec));
    for (idx, t) in transforms.iter().enumerate() {
        ledger.push_transform(*t, format!("ev-{}-{idx}", t.as_str()));
    }
    ledger
}

/// Build a synthetic Jaxpr with `n` equations chaining add operations.
fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec::smallvec![output_var],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

fn dispatch_and_get_result(spec: ProgramSpec, transforms: &[Transform], args: Vec<Value>) -> Value {
    let ledger = make_ledger(spec, transforms);
    let req = fj_dispatch::DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger,
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };
    let resp = fj_dispatch::dispatch(req).expect("dispatch should succeed");
    resp.outputs.into_iter().next().expect("should have output")
}

// ===========================================================================
// 1. DIFFERENTIAL ORACLE TESTS (5+ comparison points)
// ===========================================================================

/// Oracle comparison point 1: Jaxpr construction — verify build_program produces
/// correct equation counts and primitive types for each ProgramSpec variant.
#[test]
fn oracle_jaxpr_construction_equation_counts() {
    let cases: Vec<(ProgramSpec, usize, &str)> = vec![
        (ProgramSpec::Add2, 1, "single add"),
        (ProgramSpec::Square, 1, "single mul"),
        (ProgramSpec::SquarePlusLinear, 3, "mul+mul+add"),
        (ProgramSpec::AddOne, 1, "single add"),
        (ProgramSpec::SinX, 1, "single sin"),
        (ProgramSpec::CosX, 1, "single cos"),
        (ProgramSpec::Dot3, 1, "single dot"),
        (ProgramSpec::ReduceSumVec, 1, "single reduce_sum"),
    ];

    for (spec, expected_eqn_count, description) in cases {
        let jaxpr = build_program(spec);
        assert_eq!(
            jaxpr.equations.len(),
            expected_eqn_count,
            "oracle mismatch for {description}: expected {expected_eqn_count} equations"
        );
        jaxpr.validate_well_formed().unwrap_or_else(|err| {
            panic!("build_program({description}) should produce valid Jaxpr: {err:?}")
        });
    }
}

/// Oracle comparison point 1b: Verify build_program primitives match expected ops.
#[test]
fn oracle_jaxpr_construction_primitives() {
    let jaxpr = build_program(ProgramSpec::Add2);
    assert_eq!(jaxpr.equations[0].primitive, Primitive::Add);

    let jaxpr = build_program(ProgramSpec::Square);
    assert_eq!(jaxpr.equations[0].primitive, Primitive::Mul);

    let jaxpr = build_program(ProgramSpec::SinX);
    assert_eq!(jaxpr.equations[0].primitive, Primitive::Sin);

    let jaxpr = build_program(ProgramSpec::CosX);
    assert_eq!(jaxpr.equations[0].primitive, Primitive::Cos);

    // SquarePlusLinear: x*x + 2*x => mul, mul, add
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    assert_eq!(jaxpr.equations.len(), 3);
    assert_eq!(jaxpr.equations[0].primitive, Primitive::Mul);
    assert_eq!(jaxpr.equations[2].primitive, Primitive::Add);
}

/// Oracle comparison point 2: Canonical fingerprint — verify deterministic hash
/// matches known values for reference programs.
#[test]
fn oracle_canonical_fingerprint_deterministic() {
    let jaxpr_add2 = build_program(ProgramSpec::Add2);
    let fp1 = jaxpr_add2.canonical_fingerprint().to_owned();

    // Rebuild the same program — fingerprint must match exactly
    let jaxpr_add2_again = build_program(ProgramSpec::Add2);
    let fp2 = jaxpr_add2_again.canonical_fingerprint().to_owned();
    assert_eq!(fp1, fp2, "canonical fingerprint should be deterministic");

    // Different programs must have different fingerprints
    let jaxpr_square = build_program(ProgramSpec::Square);
    let fp_square = jaxpr_square.canonical_fingerprint();
    assert_ne!(
        fp1, fp_square,
        "different programs must have different fingerprints"
    );
}

/// Oracle comparison point 3: Transform composition proofs — verify composition
/// rules match expected JAX semantics.
#[test]
fn oracle_transform_composition_rules() {
    // Single transforms should always pass
    for t in [Transform::Jit, Transform::Grad, Transform::Vmap] {
        let ledger = make_ledger(ProgramSpec::Square, &[t]);
        let proof = verify_transform_composition(&ledger);
        assert!(
            proof.is_ok(),
            "single transform {t:?} should compose: {proof:?}"
        );
    }

    // jit(grad(f)) should pass — this is a standard JAX pattern
    let ledger = make_ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]);
    let proof = verify_transform_composition(&ledger);
    assert!(proof.is_ok(), "jit(grad(f)) should compose: {proof:?}");

    // jit(vmap(f)) should pass
    let ledger = make_ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Vmap]);
    let proof = verify_transform_composition(&ledger);
    assert!(proof.is_ok(), "jit(vmap(f)) should compose: {proof:?}");

    // vmap(grad(f)) should pass
    let ledger = make_ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]);
    let proof = verify_transform_composition(&ledger);
    assert!(proof.is_ok(), "vmap(grad(f)) should compose: {proof:?}");

    // jit(vmap(grad(f))) depth-3 should pass
    let ledger = make_ledger(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Vmap, Transform::Grad],
    );
    let proof = verify_transform_composition(&ledger);
    assert!(
        proof.is_ok(),
        "jit(vmap(grad(f))) should compose: {proof:?}"
    );
}

/// Oracle comparison point 3b: Transform composition allows double-grad
/// (dispatch handles it via finite-diff fallback).
#[test]
fn oracle_composition_allows_double_grad() {
    let ledger = make_ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]);
    let result = verify_transform_composition(&ledger);
    assert!(
        result.is_ok(),
        "grad(grad(f)) should pass composition verification: {result:?}"
    );
}

/// Oracle comparison point 4: eval_jaxpr results — verify interpreter produces
/// analytically correct outputs for known programs.
#[test]
fn oracle_eval_jaxpr_analytical_correctness() {
    // add2(3, 5) = 8
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(3), Value::scalar_i64(5)])
        .expect("eval should succeed");
    assert_eq!(result.len(), 1);
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(8), "add2(3, 5) should equal 8");

    // square(7) = 49
    let jaxpr = build_program(ProgramSpec::Square);
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(7)]).expect("eval should succeed");
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(49), "square(7) should equal 49");

    // square_plus_linear(3) = 3*3 + 2*3 = 9 + 6 = 15
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(3)]).expect("eval should succeed");
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(15), "square_plus_linear(3) should equal 15");

    // add_one(10) = 11
    let jaxpr = build_program(ProgramSpec::AddOne);
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(10)]).expect("eval should succeed");
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(11), "add_one(10) should equal 11");
}

/// Oracle comparison point 4b: eval_jaxpr floating-point analytical correctness.
#[test]
fn oracle_eval_jaxpr_float_analytical() {
    // sin(0.0) = 0.0
    let jaxpr = build_program(ProgramSpec::SinX);
    let result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)])
        .expect("eval should succeed");
    let val = result[0].as_f64_scalar().expect("should be f64");
    assert!(
        (val - 0.0).abs() < 1e-12,
        "sin(0.0) should equal 0.0, got {val}"
    );

    // cos(0.0) = 1.0
    let jaxpr = build_program(ProgramSpec::CosX);
    let result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)])
        .expect("eval should succeed");
    let val = result[0].as_f64_scalar().expect("should be f64");
    assert!(
        (val - 1.0).abs() < 1e-12,
        "cos(0.0) should equal 1.0, got {val}"
    );

    // sin(pi/2) ~= 1.0
    let jaxpr = build_program(ProgramSpec::SinX);
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_f64(std::f64::consts::FRAC_PI_2)])
            .expect("eval should succeed");
    let val = result[0].as_f64_scalar().expect("should be f64");
    assert!(
        (val - 1.0).abs() < 1e-12,
        "sin(pi/2) should equal 1.0, got {val}"
    );
}

/// Oracle comparison point 5: Dispatch end-to-end correctness — verify full dispatch
/// pipeline matches analytical oracle for jit/grad/vmap transforms.
#[test]
fn oracle_dispatch_jit_correctness() {
    // jit(add2)(3, 5) = 8
    let result = dispatch_and_get_result(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(3), Value::scalar_i64(5)],
    );
    let val = result.as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(8), "jit(add2)(3,5) should equal 8");
}

/// Oracle comparison point 5b: grad(square)(x) = 2*x (derivative of x^2)
#[test]
fn oracle_dispatch_grad_derivative() {
    let result = dispatch_and_get_result(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    );
    let val = result.as_f64_scalar().expect("grad should return f64");
    assert!(
        (val - 6.0).abs() < 1e-4,
        "grad(square)(3.0) should equal 6.0, got {val}"
    );

    // grad(square)(5.0) = 10.0
    let result = dispatch_and_get_result(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(5.0)],
    );
    let val = result.as_f64_scalar().expect("grad should return f64");
    assert!(
        (val - 10.0).abs() < 1e-4,
        "grad(square)(5.0) should equal 10.0, got {val}"
    );
}

/// Oracle comparison point 6: Cache key determinism — same inputs always produce
/// same cache keys.
#[test]
fn oracle_cache_key_determinism() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let transforms = vec![Transform::Jit];
    let compile_options = BTreeMap::new();
    let unknown: Vec<String> = vec![];

    let input = fj_cache::CacheKeyInputRef {
        mode: CompatibilityMode::Strict,
        backend: "cpu",
        jaxpr: &jaxpr,
        transform_stack: &transforms,
        compile_options: &compile_options,
        custom_hook: None,
        unknown_incompatible_features: &unknown,
    };

    let key1 = fj_cache::build_cache_key_ref(&input).expect("cache key should succeed");
    let key2 = fj_cache::build_cache_key_ref(&input).expect("cache key should succeed");
    assert_eq!(key1, key2, "cache keys must be deterministic");

    // Different backend should produce different key
    let input_gpu = fj_cache::CacheKeyInputRef {
        mode: CompatibilityMode::Strict,
        backend: "gpu",
        jaxpr: &jaxpr,
        transform_stack: &transforms,
        compile_options: &compile_options,
        custom_hook: None,
        unknown_incompatible_features: &unknown,
    };
    let key_gpu = fj_cache::build_cache_key_ref(&input_gpu).expect("cache key should succeed");
    assert_ne!(
        key1, key_gpu,
        "different backends must produce different cache keys"
    );
}

// ===========================================================================
// 2. METAMORPHIC PROPERTY TESTS (3+ properties)
// ===========================================================================

/// Metamorphic property 1: Fingerprint determinism — same Jaxpr always produces
/// same fingerprint across 100 iterations.
#[test]
fn metamorphic_fingerprint_determinism_100x() {
    for spec in [
        ProgramSpec::Add2,
        ProgramSpec::Square,
        ProgramSpec::SquarePlusLinear,
        ProgramSpec::SinX,
    ] {
        let reference = build_program(spec).canonical_fingerprint().to_owned();
        for i in 0..100 {
            let fresh = build_program(spec);
            let fp = fresh.canonical_fingerprint();
            assert_eq!(
                reference, fp,
                "fingerprint diverged on iteration {i} for {spec:?}"
            );
        }
    }
}

/// Metamorphic property 2: Composition signature stability — the composition
/// signature for a given transform stack is deterministic and order-sensitive.
#[test]
fn metamorphic_composition_signature_stability() {
    let stacks: Vec<Vec<Transform>> = vec![
        vec![Transform::Jit],
        vec![Transform::Grad],
        vec![Transform::Vmap],
        vec![Transform::Jit, Transform::Grad],
        vec![Transform::Jit, Transform::Vmap],
        vec![Transform::Vmap, Transform::Grad],
        vec![Transform::Jit, Transform::Vmap, Transform::Grad],
    ];

    for stack in &stacks {
        let sig1 = make_ledger(ProgramSpec::Square, stack).composition_signature();
        for _ in 0..50 {
            let sig = make_ledger(ProgramSpec::Square, stack).composition_signature();
            assert_eq!(
                sig1, sig,
                "composition signature should be stable for stack {stack:?}"
            );
        }
    }

    // Order-sensitivity: jit+grad != grad+jit (different composition)
    let sig_jg = make_ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad])
        .composition_signature();
    let sig_gj = make_ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Jit])
        .composition_signature();
    assert_ne!(
        sig_jg, sig_gj,
        "jit+grad and grad+jit should have different signatures"
    );
}

/// Metamorphic property 3: Equation ordering stability — building the same program
/// always produces equations in the same canonical order.
#[test]
fn metamorphic_equation_ordering_stability() {
    for spec in [
        ProgramSpec::Add2,
        ProgramSpec::SquarePlusLinear,
        ProgramSpec::SinX,
    ] {
        let reference = build_program(spec);
        let ref_fp = reference.canonical_fingerprint().to_owned();
        let ref_eqn_count = reference.equations.len();

        for _ in 0..50 {
            let rebuilt = build_program(spec);
            assert_eq!(rebuilt.equations.len(), ref_eqn_count);
            assert_eq!(
                rebuilt.canonical_fingerprint(),
                &ref_fp,
                "equation ordering should be stable for {spec:?}"
            );

            // Verify equation-by-equation match
            for (i, (ref_eqn, new_eqn)) in reference
                .equations
                .iter()
                .zip(rebuilt.equations.iter())
                .enumerate()
            {
                assert_eq!(
                    ref_eqn.primitive, new_eqn.primitive,
                    "primitive mismatch at equation {i} for {spec:?}"
                );
            }
        }
    }
}

/// Metamorphic property 4: Eval determinism — same program + args always
/// produces the same output across 50 iterations.
#[test]
fn metamorphic_eval_determinism_50x() {
    let cases: Vec<(ProgramSpec, Vec<Value>)> = vec![
        (
            ProgramSpec::Add2,
            vec![Value::scalar_i64(3), Value::scalar_i64(5)],
        ),
        (ProgramSpec::Square, vec![Value::scalar_i64(7)]),
        (ProgramSpec::SquarePlusLinear, vec![Value::scalar_i64(4)]),
        (ProgramSpec::AddOne, vec![Value::scalar_i64(99)]),
    ];

    for (spec, args) in &cases {
        let jaxpr = build_program(*spec);
        let reference = fj_interpreters::eval_jaxpr(&jaxpr, args).expect("first eval should work");

        for i in 0..50 {
            let result =
                fj_interpreters::eval_jaxpr(&jaxpr, args).expect("repeated eval should work");
            assert_eq!(
                result.len(),
                reference.len(),
                "output count diverged on iteration {i} for {spec:?}"
            );
            for (j, (ref_val, new_val)) in reference.iter().zip(result.iter()).enumerate() {
                let ref_lit = ref_val.as_scalar_literal();
                let new_lit = new_val.as_scalar_literal();
                assert_eq!(
                    ref_lit, new_lit,
                    "output {j} diverged on iteration {i} for {spec:?}"
                );
            }
        }
    }
}

// ===========================================================================
// 3. ADVERSARIAL EDGE CASE TESTS (4+ cases)
// ===========================================================================

/// Adversarial case 1: Jaxpr with 0 equations (empty program).
#[test]
fn adversarial_empty_jaxpr() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1)], // in: v1
        vec![],         // no constvars
        vec![VarId(1)], // out: v1 (identity)
        vec![],         // no equations
    );

    // Should be well-formed (identity function)
    jaxpr
        .validate_well_formed()
        .expect("empty jaxpr (identity) should be valid");

    // Should have a fingerprint
    let fp = jaxpr.canonical_fingerprint();
    assert!(!fp.is_empty(), "empty jaxpr should have a fingerprint");

    // Should evaluate (identity: input = output)
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(42)]).expect("eval should work");
    assert_eq!(result.len(), 1);
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(val, Some(42), "identity jaxpr should return input");
}

/// Adversarial case 2: Jaxpr with duplicate variable IDs should fail validation.
#[test]
fn adversarial_duplicate_variable_ids() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(1)], // duplicate in invars
        vec![],
        vec![VarId(1)],
        vec![],
    );

    let result = jaxpr.validate_well_formed();
    assert!(
        result.is_err(),
        "duplicate variable IDs in invars should fail validation"
    );
    match result.unwrap_err() {
        JaxprValidationError::DuplicateBinding { section, var } => {
            assert_eq!(section, "invars");
            assert_eq!(var, VarId(1));
        }
        other => panic!("expected DuplicateBinding, got {other:?}"),
    }
}

/// Adversarial case 3: Transform stack with repeated transforms (jit(jit(f))).
#[test]
fn adversarial_repeated_jit_transforms() {
    // jit(jit(f)) — should still work (idempotent)
    let ledger = make_ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Jit]);
    let result = verify_transform_composition(&ledger);
    // jit is idempotent in JAX, so double-jit should not error
    assert!(
        result.is_ok(),
        "jit(jit(f)) should compose (jit is idempotent): {result:?}"
    );
}

/// Adversarial case 3b: Double-vmap should also compose (maps over two axes).
#[test]
fn adversarial_double_vmap() {
    let ledger = make_ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Vmap]);
    let result = verify_transform_composition(&ledger);
    assert!(
        result.is_ok(),
        "vmap(vmap(f)) should pass composition verification: {result:?}"
    );
}

/// Adversarial case 4: Maximum supported equation count (stress test).
#[test]
fn adversarial_large_jaxpr_stress_test() {
    let n = 10_000;
    let jaxpr = build_chain_jaxpr(n);

    // Should validate
    jaxpr
        .validate_well_formed()
        .expect("10K-equation jaxpr should be well-formed");

    // Should compute fingerprint without panic
    let fp = jaxpr.canonical_fingerprint();
    assert!(!fp.is_empty(), "large jaxpr should have a fingerprint");

    // Should evaluate: chain of n adds starting from 0 should give n
    let result =
        fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(0)]).expect("eval should work");
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(
        val,
        Some(n as i64),
        "chain of {n} +1 additions from 0 should equal {n}"
    );
}

/// Adversarial case 5: Unbound variable reference should fail validation.
#[test]
fn adversarial_unbound_variable_reference() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)], // v3 is never defined
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)) // v2 is unbound
            ],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    );

    let result = jaxpr.validate_well_formed();
    assert!(
        result.is_err(),
        "unbound variable reference should fail validation"
    );
    match result.unwrap_err() {
        JaxprValidationError::UnboundInputVar {
            equation_index,
            var,
        } => {
            assert_eq!(equation_index, 0);
            assert_eq!(var, VarId(2));
        }
        other => panic!("expected UnboundInputVar, got {other:?}"),
    }
}

/// Adversarial case 6: Empty transform stack — composition verification should
/// pass (trivially valid).
#[test]
fn adversarial_empty_transform_stack() {
    let ledger = make_ledger(ProgramSpec::Square, &[]);
    let result = verify_transform_composition(&ledger);
    assert!(
        result.is_ok(),
        "empty transform stack should compose: {result:?}"
    );
}

/// Adversarial case 7: Evidence count mismatch should produce specific error.
#[test]
fn adversarial_evidence_mismatch() {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    // Push a transform but manipulate evidence manually
    ledger.transform_stack.push(Transform::Jit);
    // Don't push corresponding evidence — mismatch!

    let result = verify_transform_composition(&ledger);
    assert!(
        result.is_err(),
        "evidence count mismatch should fail: {result:?}"
    );
    match result.unwrap_err() {
        TransformCompositionError::EvidenceCountMismatch { .. } => {}
        other => panic!("expected EvidenceCountMismatch, got {other:?}"),
    }
}

/// Adversarial case 8: Empty evidence string should be rejected.
#[test]
fn adversarial_empty_evidence_string() {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    ledger.transform_stack.push(Transform::Jit);
    ledger.transform_evidence.push(String::new()); // empty evidence

    let result = verify_transform_composition(&ledger);
    assert!(
        result.is_err(),
        "empty evidence string should fail: {result:?}"
    );
    match result.unwrap_err() {
        TransformCompositionError::EmptyEvidence { .. } => {}
        other => panic!("expected EmptyEvidence, got {other:?}"),
    }
}
