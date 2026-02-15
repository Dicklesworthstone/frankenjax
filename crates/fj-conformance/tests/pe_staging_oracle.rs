//! bd-3dl.14.6: Differential Oracle + Metamorphic + Adversarial Validation — Partial Eval
//!
//! Oracle comparison points:
//!   1. PE split correctness — all-known yields empty residual, all-unknown yields full residual
//!   2. Constant folding — expressions with all-constant inputs produce correct values
//!   3. Residual shape — residual Jaxpr has correct equation count and unknowns mask
//!   4. DCE correctness — unreachable equations removed, preserving semantics
//!   5. Staging pipeline — stage_jaxpr + execute_staged == eval_jaxpr for all mask combos
//!
//! Metamorphic properties:
//!   1. PE with all knowns == full eval (no residual)
//!   2. PE with no knowns == identity (original Jaxpr preserved in residual)
//!   3. PE is deterministic — same inputs produce same split (100x)
//!
//! Adversarial cases:
//!   1. Partial eval with contradictory mask length
//!   2. Residual that requires all inputs (nothing foldable)
//!   3. Large chain PE (1000 equations)
//!   4. PE of identity Jaxpr (no equations)

use fj_core::{
    Atom, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Value, VarId, build_program,
};
use fj_interpreters::partial_eval::{dce_jaxpr, partial_eval_jaxpr};
use fj_interpreters::staging::{execute_staged, make_jaxpr, stage_jaxpr};
use smallvec::smallvec;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// { a, b -> c = neg(a); d = mul(c, b) -> d }
/// Used for mixed known/unknown tests: a known splits neg to known jaxpr.
fn make_neg_mul_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
            },
        ],
    )
}

/// { a, b -> c = add(a, b); d = neg(a) -> c, d }
fn make_multi_output_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3), VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
            },
        ],
    )
}

/// Build a chain of `n` additions: x + 1 + 1 + ... + 1
fn build_add_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec![output_var],
            params: BTreeMap::new(),
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

// ===========================================================================
// 1. DIFFERENTIAL ORACLE TESTS (5 comparison points)
// ===========================================================================

/// Oracle comparison point 1: PE split correctness — all-known inputs produce
/// a jaxpr_known with all equations and an empty jaxpr_unknown.
#[test]
fn oracle_pe_all_known_yields_full_known_jaxpr() {
    let specs_and_inputs: Vec<(ProgramSpec, usize)> = vec![
        (ProgramSpec::Add2, 2),
        (ProgramSpec::Square, 1),
        (ProgramSpec::SquarePlusLinear, 1),
        (ProgramSpec::AddOne, 1),
        (ProgramSpec::SinX, 1),
        (ProgramSpec::CosX, 1),
    ];

    for (spec, n_inputs) in specs_and_inputs {
        let jaxpr = build_program(spec);
        let unknowns = vec![false; n_inputs];
        let result = partial_eval_jaxpr(&jaxpr, &unknowns)
            .unwrap_or_else(|e| panic!("PE should succeed for {spec:?}: {e}"));

        assert_eq!(
            result.jaxpr_known.equations.len(),
            jaxpr.equations.len(),
            "all-known PE for {spec:?} should put all equations in known jaxpr"
        );
        assert_eq!(
            result.jaxpr_unknown.equations.len(),
            0,
            "all-known PE for {spec:?} should have empty unknown jaxpr"
        );
        assert!(
            result.out_unknowns.iter().all(|u| !u),
            "all-known PE for {spec:?} should have no unknown outputs"
        );
    }
}

/// Oracle comparison point 1b: PE split correctness — all-unknown inputs produce
/// a jaxpr_unknown with all equations and an empty jaxpr_known.
#[test]
fn oracle_pe_all_unknown_yields_full_unknown_jaxpr() {
    let specs_and_inputs: Vec<(ProgramSpec, usize)> = vec![
        (ProgramSpec::Add2, 2),
        (ProgramSpec::Square, 1),
        (ProgramSpec::SquarePlusLinear, 1),
        (ProgramSpec::AddOne, 1),
        (ProgramSpec::SinX, 1),
        (ProgramSpec::CosX, 1),
    ];

    for (spec, n_inputs) in specs_and_inputs {
        let jaxpr = build_program(spec);
        let unknowns = vec![true; n_inputs];
        let result = partial_eval_jaxpr(&jaxpr, &unknowns)
            .unwrap_or_else(|e| panic!("PE should succeed for {spec:?}: {e}"));

        assert_eq!(
            result.jaxpr_known.equations.len(),
            0,
            "all-unknown PE for {spec:?} should have empty known jaxpr"
        );
        assert_eq!(
            result.jaxpr_unknown.equations.len(),
            jaxpr.equations.len(),
            "all-unknown PE for {spec:?} should put all equations in unknown jaxpr"
        );
        assert!(
            result.out_unknowns.iter().all(|u| *u),
            "all-unknown PE for {spec:?} should have all unknown outputs"
        );
    }
}

/// Oracle comparison point 2: Constant folding — PE with known inputs allows
/// evaluating the known jaxpr, producing correct intermediate values.
#[test]
fn oracle_pe_constant_folding_correctness() {
    // neg(-5) = 5; 5 * b where b is unknown
    let jaxpr = make_neg_mul_jaxpr();
    let pe = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();

    // Known jaxpr: neg(a)
    assert_eq!(pe.jaxpr_known.equations.len(), 1);
    assert_eq!(pe.jaxpr_known.equations[0].primitive, Primitive::Neg);

    // Evaluate known jaxpr: neg(-5) = 5
    let known_out = fj_interpreters::eval_jaxpr_with_consts(
        &pe.jaxpr_known,
        &pe.known_consts,
        &[Value::scalar_i64(-5)],
    )
    .unwrap();
    // The output includes residuals; the residual for VarId(3) = neg(-5) = 5
    assert!(!known_out.is_empty());

    // Now verify the residual feeds correctly into the unknown jaxpr
    let mut unk_inputs = known_out;
    unk_inputs.push(Value::scalar_i64(3)); // b = 3
    let staged_result = fj_interpreters::eval_jaxpr(&pe.jaxpr_unknown, &unk_inputs).unwrap();

    // Full eval: neg(-5) * 3 = 5 * 3 = 15
    let full = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(-5), Value::scalar_i64(3)])
        .unwrap();
    assert_eq!(
        staged_result, full,
        "PE constant folding should preserve semantics"
    );
}

/// Oracle comparison point 3: Residual shape — verify equation counts in each
/// sub-jaxpr match expected splits for various mask configurations.
#[test]
fn oracle_pe_residual_shape_correctness() {
    let jaxpr = make_neg_mul_jaxpr();

    // [false, true]: neg(a) known, mul(neg(a), b) unknown
    let pe = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 1);
    assert_eq!(pe.jaxpr_unknown.equations.len(), 1);
    assert!(!pe.residual_avals.is_empty(), "should have residuals");
    assert_eq!(pe.out_unknowns, vec![true]);

    // [true, false]: both equations are unknown (add chain depends on a)
    let pe = partial_eval_jaxpr(&jaxpr, &[true, false]).unwrap();
    assert_eq!(
        pe.jaxpr_unknown.equations.len(),
        2,
        "when first input unknown, both equations depend on it"
    );
    assert_eq!(pe.out_unknowns, vec![true]);

    // Multi-output: { a, b -> c=add(a,b); d=neg(a) -> c, d }
    let jaxpr_multi = make_multi_output_jaxpr();

    // [false, false]: all known
    let pe = partial_eval_jaxpr(&jaxpr_multi, &[false, false]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 2);
    assert_eq!(pe.out_unknowns, vec![false, false]);

    // [false, true]: add(a,b) unknown (depends on b), neg(a) known
    let pe = partial_eval_jaxpr(&jaxpr_multi, &[false, true]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 1); // neg(a) known
    assert_eq!(pe.jaxpr_unknown.equations.len(), 1); // add(a,b) unknown
    assert_eq!(pe.out_unknowns, vec![true, false]);
}

/// Oracle comparison point 4: DCE correctness — dead equations are removed,
/// living equations preserved, semantics unchanged.
#[test]
fn oracle_dce_correctness() {
    let jaxpr_multi = make_multi_output_jaxpr();

    // Only use first output: add(a,b); neg(a) should be removed
    let (pruned, used_inputs) = dce_jaxpr(&jaxpr_multi, &[true, false]);
    assert_eq!(pruned.equations.len(), 1);
    assert_eq!(pruned.equations[0].primitive, Primitive::Add);
    assert_eq!(used_inputs, vec![true, true]); // both inputs needed for add

    // Only use second output: neg(a); add(a,b) should be removed
    let (pruned, used_inputs) = dce_jaxpr(&jaxpr_multi, &[false, true]);
    assert_eq!(pruned.equations.len(), 1);
    assert_eq!(pruned.equations[0].primitive, Primitive::Neg);
    assert_eq!(used_inputs, vec![true, false]); // only a needed for neg

    // Neither output used: both removed
    let (pruned, used_inputs) = dce_jaxpr(&jaxpr_multi, &[false, false]);
    assert_eq!(pruned.equations.len(), 0);
    assert_eq!(used_inputs, vec![false, false]);

    // Both outputs used: nothing removed
    let (pruned, used_inputs) = dce_jaxpr(&jaxpr_multi, &[true, true]);
    assert_eq!(pruned.equations.len(), 2);
    assert_eq!(used_inputs, vec![true, true]);
}

/// Oracle comparison point 5: Staging pipeline — stage_jaxpr + execute_staged
/// produces the same result as eval_jaxpr for various programs and masks.
#[test]
fn oracle_staging_pipeline_equivalence() {
    // Test 1: neg_mul with first known
    {
        let jaxpr = make_neg_mul_jaxpr();
        let a = Value::scalar_i64(7);
        let b = Value::scalar_i64(3);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).unwrap();
        let staged = stage_jaxpr(&jaxpr, &[false, true], &[a]).unwrap();
        let result = execute_staged(&staged, &[b]).unwrap();
        assert_eq!(result, full, "staging neg_mul(7, 3) should match full eval");
    }

    // Test 2: add2 with both known
    {
        let jaxpr = build_program(ProgramSpec::Add2);
        let a = Value::scalar_i64(10);
        let b = Value::scalar_i64(20);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).unwrap();
        let staged = stage_jaxpr(&jaxpr, &[false, false], &[a, b]).unwrap();
        let result = execute_staged(&staged, &[]).unwrap();
        assert_eq!(result, full, "staging add2(10, 20) should match full eval");
    }

    // Test 3: square with known input
    {
        let jaxpr = build_program(ProgramSpec::Square);
        let x = Value::scalar_i64(5);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &[x.clone()]).unwrap();
        let staged = stage_jaxpr(&jaxpr, &[false], &[x]).unwrap();
        let result = execute_staged(&staged, &[]).unwrap();
        assert_eq!(result, full, "staging square(5) should match full eval");
    }

    // Test 4: add_one with unknown input
    {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let x = Value::scalar_i64(99);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &[x.clone()]).unwrap();
        let staged = stage_jaxpr(&jaxpr, &[true], &[]).unwrap();
        let result = execute_staged(&staged, &[x]).unwrap();
        assert_eq!(result, full, "staging add_one(99) should match full eval");
    }
}

/// Oracle comparison point 5b: Staging with negative, zero, and boundary values.
#[test]
fn oracle_staging_boundary_values() {
    let jaxpr = make_neg_mul_jaxpr();
    let cases: Vec<(i64, i64)> = vec![(0, 0), (1, -1), (-1, 1), (100, -100), (-999, 999)];

    for (a, b) in cases {
        let va = Value::scalar_i64(a);
        let vb = Value::scalar_i64(b);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &[va.clone(), vb.clone()]).unwrap();
        let staged = stage_jaxpr(&jaxpr, &[false, true], &[va]).unwrap();
        let result = execute_staged(&staged, &[vb]).unwrap();
        assert_eq!(result, full, "staging mismatch for neg_mul({a}, {b})");
    }
}

// ===========================================================================
// 2. METAMORPHIC PROPERTY TESTS (3 properties)
// ===========================================================================

/// Metamorphic property 1: PE with all knowns == full eval (no residual needed).
/// The known jaxpr alone should produce the final answer.
#[test]
fn metamorphic_pe_all_known_equals_full_eval() {
    let specs_and_args: Vec<(ProgramSpec, Vec<Value>)> = vec![
        (
            ProgramSpec::Add2,
            vec![Value::scalar_i64(3), Value::scalar_i64(5)],
        ),
        (ProgramSpec::Square, vec![Value::scalar_i64(7)]),
        (ProgramSpec::AddOne, vec![Value::scalar_i64(42)]),
        (ProgramSpec::SquarePlusLinear, vec![Value::scalar_i64(4)]),
    ];

    for (spec, args) in specs_and_args {
        let jaxpr = build_program(spec);
        let full = fj_interpreters::eval_jaxpr(&jaxpr, &args).unwrap();

        let unknowns = vec![false; args.len()];
        let pe = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();

        // All equations should be in known jaxpr
        assert_eq!(
            pe.jaxpr_known.equations.len(),
            jaxpr.equations.len(),
            "all-known PE for {spec:?} should fold everything"
        );

        // Evaluating the known jaxpr should produce the output
        let known_result =
            fj_interpreters::eval_jaxpr_with_consts(&pe.jaxpr_known, &pe.known_consts, &args)
                .unwrap();

        // The known result should contain the original outputs (may also have residuals)
        assert_eq!(
            known_result[0], full[0],
            "all-known PE eval for {spec:?} should match full eval"
        );
    }
}

/// Metamorphic property 2: PE with no knowns == identity (original Jaxpr preserved
/// in residual). The unknown jaxpr should contain all original equations.
#[test]
fn metamorphic_pe_no_knowns_is_identity() {
    let specs_and_inputs: Vec<(ProgramSpec, usize)> = vec![
        (ProgramSpec::Add2, 2),
        (ProgramSpec::Square, 1),
        (ProgramSpec::SquarePlusLinear, 1),
        (ProgramSpec::AddOne, 1),
    ];

    for (spec, n_inputs) in specs_and_inputs {
        let jaxpr = build_program(spec);
        let unknowns = vec![true; n_inputs];
        let pe = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();

        // Unknown jaxpr should have exactly the same equations
        assert_eq!(
            pe.jaxpr_unknown.equations.len(),
            jaxpr.equations.len(),
            "no-knowns PE for {spec:?} should preserve all equations in residual"
        );

        // Primitives should match in order
        for (i, (orig, residual)) in jaxpr
            .equations
            .iter()
            .zip(pe.jaxpr_unknown.equations.iter())
            .enumerate()
        {
            assert_eq!(
                orig.primitive, residual.primitive,
                "primitive mismatch at equation {i} for {spec:?}"
            );
        }

        // No residuals needed (no known→unknown flow)
        assert!(
            pe.residual_avals.is_empty(),
            "no-knowns PE for {spec:?} should have no residuals"
        );
    }
}

/// Metamorphic property 3: PE is deterministic — same inputs produce same split
/// across 100 iterations.
#[test]
fn metamorphic_pe_determinism_100x() {
    let jaxpr = make_neg_mul_jaxpr();
    let unknowns = [false, true];

    let reference = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();
    let ref_known_count = reference.jaxpr_known.equations.len();
    let ref_unknown_count = reference.jaxpr_unknown.equations.len();
    let ref_residual_count = reference.residual_avals.len();
    let ref_out_unknowns = reference.out_unknowns.clone();

    for i in 0..100 {
        let result = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();
        assert_eq!(
            result.jaxpr_known.equations.len(),
            ref_known_count,
            "known equation count diverged on iteration {i}"
        );
        assert_eq!(
            result.jaxpr_unknown.equations.len(),
            ref_unknown_count,
            "unknown equation count diverged on iteration {i}"
        );
        assert_eq!(
            result.residual_avals.len(),
            ref_residual_count,
            "residual count diverged on iteration {i}"
        );
        assert_eq!(
            result.out_unknowns, ref_out_unknowns,
            "out_unknowns diverged on iteration {i}"
        );
    }
}

/// Metamorphic property 3b: DCE is deterministic across 100 iterations.
#[test]
fn metamorphic_dce_determinism_100x() {
    let jaxpr = make_multi_output_jaxpr();
    let used = [true, false];

    let (ref_pruned, ref_used_inputs) = dce_jaxpr(&jaxpr, &used);
    let ref_eqn_count = ref_pruned.equations.len();

    for i in 0..100 {
        let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &used);
        assert_eq!(
            pruned.equations.len(),
            ref_eqn_count,
            "DCE equation count diverged on iteration {i}"
        );
        assert_eq!(
            used_inputs, ref_used_inputs,
            "DCE used_inputs diverged on iteration {i}"
        );
    }
}

/// Metamorphic property: staging is deterministic across 50 iterations.
#[test]
fn metamorphic_staging_determinism_50x() {
    let jaxpr = make_neg_mul_jaxpr();
    let a = Value::scalar_i64(5);
    let b = Value::scalar_i64(3);

    let reference_staged = stage_jaxpr(&jaxpr, &[false, true], &[a.clone()]).unwrap();
    let reference_result = execute_staged(&reference_staged, &[b.clone()]).unwrap();

    for i in 0..50 {
        let staged = stage_jaxpr(&jaxpr, &[false, true], &[a.clone()]).unwrap();
        let result = execute_staged(&staged, &[b.clone()]).unwrap();
        assert_eq!(
            result, reference_result,
            "staging result diverged on iteration {i}"
        );
    }
}

// ===========================================================================
// 3. ADVERSARIAL EDGE CASE TESTS (4+ cases)
// ===========================================================================

/// Adversarial case 1: Partial eval with wrong mask length should fail.
#[test]
fn adversarial_pe_mask_length_mismatch() {
    let jaxpr = build_program(ProgramSpec::Add2); // 2 inputs

    // Too short
    let err = partial_eval_jaxpr(&jaxpr, &[false]).unwrap_err();
    assert!(
        matches!(
            err,
            fj_interpreters::partial_eval::PartialEvalError::InputMaskMismatch { .. }
        ),
        "mask too short should produce InputMaskMismatch"
    );

    // Too long
    let err = partial_eval_jaxpr(&jaxpr, &[false, true, false]).unwrap_err();
    assert!(
        matches!(
            err,
            fj_interpreters::partial_eval::PartialEvalError::InputMaskMismatch { .. }
        ),
        "mask too long should produce InputMaskMismatch"
    );

    // Empty mask for non-empty jaxpr
    let err = partial_eval_jaxpr(&jaxpr, &[]).unwrap_err();
    assert!(
        matches!(
            err,
            fj_interpreters::partial_eval::PartialEvalError::InputMaskMismatch { .. }
        ),
        "empty mask for 2-input jaxpr should produce InputMaskMismatch"
    );
}

/// Adversarial case 2: PE where nothing can be folded (all inputs unknown,
/// every equation has at least one unknown dependency).
#[test]
fn adversarial_pe_nothing_foldable() {
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear); // x*x + 2*x
    let pe = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();

    // Everything should be in the unknown jaxpr
    assert_eq!(pe.jaxpr_known.equations.len(), 0);
    assert_eq!(pe.jaxpr_unknown.equations.len(), jaxpr.equations.len());
    assert!(pe.residual_avals.is_empty());

    // The unknown jaxpr should evaluate correctly
    let x = Value::scalar_i64(3);
    let full = fj_interpreters::eval_jaxpr(&jaxpr, &[x.clone()]).unwrap();
    let residual_result = fj_interpreters::eval_jaxpr(&pe.jaxpr_unknown, &[x]).unwrap();
    assert_eq!(residual_result, full);
}

/// Adversarial case 3: Large chain PE (1000 equations) — stress test.
#[test]
fn adversarial_pe_large_chain() {
    let n = 1000;
    let jaxpr = build_add_chain_jaxpr(n);

    // All known — should fold everything
    let pe_known = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
    assert_eq!(pe_known.jaxpr_known.equations.len(), n);
    assert_eq!(pe_known.jaxpr_unknown.equations.len(), 0);

    // All unknown — should residualize everything
    let pe_unknown = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
    assert_eq!(pe_unknown.jaxpr_known.equations.len(), 0);
    assert_eq!(pe_unknown.jaxpr_unknown.equations.len(), n);

    // Verify staging correctness for the known case
    let staged = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(0)]).unwrap();
    let result = execute_staged(&staged, &[]).unwrap();
    let val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assert_eq!(
        val,
        Some(n as i64),
        "staging chain of {n} +1 should equal {n}"
    );
}

/// Adversarial case 4: PE of identity Jaxpr (no equations).
#[test]
fn adversarial_pe_identity_jaxpr() {
    let jaxpr = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);

    // With known input
    let pe = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 0);
    assert_eq!(pe.jaxpr_unknown.equations.len(), 0);
    assert_eq!(pe.out_unknowns, vec![false]);

    // With unknown input
    let pe = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 0);
    assert_eq!(pe.jaxpr_unknown.equations.len(), 0);
    assert_eq!(pe.out_unknowns, vec![true]);
}

/// Adversarial case 5: PE with empty Jaxpr (no inputs, no equations).
#[test]
fn adversarial_pe_empty_jaxpr() {
    let jaxpr = Jaxpr::new(vec![], vec![], vec![], vec![]);
    let pe = partial_eval_jaxpr(&jaxpr, &[]).unwrap();
    assert_eq!(pe.jaxpr_known.equations.len(), 0);
    assert_eq!(pe.jaxpr_unknown.equations.len(), 0);
    assert!(pe.out_unknowns.is_empty());
}

/// Adversarial case 6: DCE on a large chain with only the final output used.
#[test]
fn adversarial_dce_large_chain() {
    let n = 1000;
    let jaxpr = build_add_chain_jaxpr(n);

    // All equations contribute to the single output — nothing should be removed
    let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);
    assert_eq!(
        pruned.equations.len(),
        n,
        "DCE should keep all equations in a chain"
    );
    assert_eq!(used_inputs, vec![true]);
}

/// Adversarial case 7: Staging error propagation — mask mismatch flows through.
#[test]
fn adversarial_staging_error_propagation() {
    let jaxpr = build_program(ProgramSpec::Add2);
    let err = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(1)]);
    assert!(err.is_err(), "staging with wrong mask should fail");
}

/// Adversarial case 8: make_jaxpr produces well-formed Jaxprs for all specs.
#[test]
fn adversarial_make_jaxpr_all_specs_well_formed() {
    let specs = [
        ProgramSpec::Add2,
        ProgramSpec::Square,
        ProgramSpec::SquarePlusLinear,
        ProgramSpec::AddOne,
        ProgramSpec::SinX,
        ProgramSpec::CosX,
        ProgramSpec::Dot3,
        ProgramSpec::ReduceSumVec,
    ];
    for spec in specs {
        let jaxpr = make_jaxpr(spec);
        jaxpr
            .validate_well_formed()
            .unwrap_or_else(|e| panic!("make_jaxpr({spec:?}) should be well-formed: {e:?}"));
    }
}
