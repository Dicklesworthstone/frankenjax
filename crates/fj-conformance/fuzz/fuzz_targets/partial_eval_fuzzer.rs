#![no_main]

mod common;

use common::{ByteCursor, sample_program};
use fj_core::build_program;
use fj_interpreters::partial_eval::{partial_eval_jaxpr, PartialEvalResult};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);

    let jaxpr = build_program(sample_program(&mut cursor));
    let input_count = jaxpr.invars.len();

    if input_count == 0 {
        return;
    }

    let unknowns: Vec<bool> = (0..input_count).map(|_| cursor.take_bool()).collect();

    let result = match partial_eval_jaxpr(&jaxpr, &unknowns) {
        Ok(res) => res,
        Err(_) => return,
    };

    verify_partial_eval_invariants(&jaxpr, &unknowns, &result);
});

fn verify_partial_eval_invariants(
    original: &fj_core::Jaxpr,
    unknowns: &[bool],
    result: &PartialEvalResult,
) {
    let PartialEvalResult {
        jaxpr_known,
        known_consts: _,
        jaxpr_unknown,
        out_unknowns,
        residual_avals,
    } = result;

    assert_eq!(
        out_unknowns.len(),
        original.outvars.len(),
        "out_unknowns length must match original output count"
    );

    let known_input_count = unknowns.iter().filter(|u| !**u).count();
    let unknown_input_count = unknowns.iter().filter(|u| **u).count();

    assert!(
        jaxpr_known.invars.len() <= known_input_count,
        "known jaxpr inputs cannot exceed original known input count"
    );

    let expected_unknown_inputs = residual_avals.len() + unknown_input_count;
    assert_eq!(
        jaxpr_unknown.invars.len(),
        expected_unknown_inputs,
        "unknown jaxpr inputs = residuals + original unknown inputs"
    );

    let known_out_count = out_unknowns.iter().filter(|u| !**u).count();
    let unknown_out_count = out_unknowns.iter().filter(|u| **u).count();

    assert!(
        jaxpr_known.outvars.len() >= known_out_count,
        "known jaxpr outputs >= known outputs (may include residuals)"
    );

    assert_eq!(
        jaxpr_unknown.outvars.len(),
        unknown_out_count,
        "unknown jaxpr outputs = original unknown outputs"
    );

    if unknowns.iter().all(|u| !*u) {
        assert!(
            out_unknowns.iter().all(|u| !*u),
            "all-known inputs must produce all-known outputs"
        );
        assert!(
            residual_avals.is_empty(),
            "all-known path should have no residuals"
        );
    }

    if unknowns.iter().all(|u| *u) {
        assert!(
            jaxpr_known.equations.is_empty(),
            "all-unknown inputs means known jaxpr should be empty"
        );
    }
}
