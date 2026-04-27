#![forbid(unsafe_code)]

//! Staging pipeline: trace → partial_eval → eval.
//!
//! Provides `make_jaxpr` for standalone Jaxpr construction from program specs,
//! and `StagedProgram` for the staging pipeline used by JIT.

use fj_core::{Jaxpr, ProgramSpec, Value, build_program};

use crate::partial_eval::{PartialEvalError, PartialEvalResult, partial_eval_jaxpr};
use crate::{InterpreterError, eval_jaxpr, eval_jaxpr_with_consts};

/// A staged program ready for execution with partial evaluation applied.
#[derive(Debug, Clone)]
pub struct StagedProgram {
    /// The known sub-Jaxpr (evaluated once with known inputs).
    pub jaxpr_known: Jaxpr,

    /// Constants for the known Jaxpr's constvars.
    pub known_consts: Vec<Value>,

    /// The unknown (residual) sub-Jaxpr (evaluated per call with dynamic inputs).
    pub jaxpr_unknown: Jaxpr,

    /// Which original outputs are produced by the unknown jaxpr.
    pub out_unknowns: Vec<bool>,

    /// Original outputs that are already known after staging.
    pub known_outputs: Vec<Value>,

    /// Pre-computed residual values consumed by the unknown sub-jaxpr.
    pub residuals: Vec<Value>,
}

/// Errors during staging.
#[derive(Debug, Clone)]
pub enum StagingError {
    /// Partial evaluation failed.
    PartialEval(PartialEvalError),
    /// Known-jaxpr evaluation failed.
    KnownEval(InterpreterError),
    /// Unknown-jaxpr evaluation failed.
    UnknownEval(InterpreterError),
    /// Partitioning known outputs from residual outputs failed.
    OutputPartition {
        expected_known: usize,
        actual_outputs: usize,
    },
    /// Reconstructing final outputs from known/unknown streams failed.
    OutputReconstruction {
        expected_known: usize,
        actual_known: usize,
        expected_unknown: usize,
        actual_unknown: usize,
    },
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PartialEval(e) => write!(f, "staging: partial eval failed: {e}"),
            Self::KnownEval(e) => write!(f, "staging: known eval failed: {e}"),
            Self::UnknownEval(e) => write!(f, "staging: unknown eval failed: {e}"),
            Self::OutputPartition {
                expected_known,
                actual_outputs,
            } => write!(
                f,
                "staging: output partition mismatch (expected at least {} known outputs, got {})",
                expected_known, actual_outputs
            ),
            Self::OutputReconstruction {
                expected_known,
                actual_known,
                expected_unknown,
                actual_unknown,
            } => write!(
                f,
                "staging: output reconstruction mismatch (known expected {}, got {}; unknown expected {}, got {})",
                expected_known, actual_known, expected_unknown, actual_unknown
            ),
        }
    }
}

impl std::error::Error for StagingError {}

/// Construct a Jaxpr from a program spec (FrankenJAX equivalent of `jax.make_jaxpr`).
///
/// Returns a valid Jaxpr with correct input/output types.
#[must_use]
pub fn make_jaxpr(spec: ProgramSpec) -> Jaxpr {
    build_program(spec)
}

/// Stage a Jaxpr for execution with known/unknown input classification.
///
/// This implements the staging pipeline:
/// 1. Partial evaluate the Jaxpr to split known/unknown
/// 2. Evaluate the known sub-Jaxpr to get residuals
/// 3. Return a StagedProgram ready for dynamic execution
///
/// # Arguments
/// * `jaxpr` - The Jaxpr to stage.
/// * `unknowns` - Boolean mask of which inputs are unknown.
/// * `known_values` - Concrete values for the known inputs.
pub fn stage_jaxpr(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
    known_values: &[Value],
) -> Result<StagedProgram, StagingError> {
    let pe_result: PartialEvalResult =
        partial_eval_jaxpr(jaxpr, unknowns).map_err(StagingError::PartialEval)?;

    // Evaluate known sub-jaxpr and split known outputs vs residuals.
    let known_output_count = pe_result
        .out_unknowns
        .iter()
        .filter(|is_unknown| !**is_unknown)
        .count();

    let (known_outputs, residuals) = if !pe_result.jaxpr_known.outvars.is_empty() {
        let known_eval_outputs = eval_jaxpr_with_consts(
            &pe_result.jaxpr_known,
            &pe_result.known_consts,
            known_values,
        )
        .map_err(StagingError::KnownEval)?;
        if known_eval_outputs.len() < known_output_count {
            return Err(StagingError::OutputPartition {
                expected_known: known_output_count,
                actual_outputs: known_eval_outputs.len(),
            });
        }

        let (known_slice, residual_slice) = known_eval_outputs.split_at(known_output_count);
        (known_slice.to_vec(), residual_slice.to_vec())
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(StagedProgram {
        jaxpr_known: pe_result.jaxpr_known,
        known_consts: pe_result.known_consts,
        jaxpr_unknown: pe_result.jaxpr_unknown,
        out_unknowns: pe_result.out_unknowns,
        known_outputs,
        residuals,
    })
}

/// Execute a staged program with dynamic (unknown) inputs.
///
/// # Arguments
/// * `staged` - The staged program from `stage_jaxpr`.
/// * `dynamic_args` - Concrete values for the originally-unknown inputs.
pub fn execute_staged(
    staged: &StagedProgram,
    dynamic_args: &[Value],
) -> Result<Vec<Value>, StagingError> {
    let expected_unknown = staged.out_unknowns.iter().filter(|u| **u).count();
    let expected_known = staged.out_unknowns.len().saturating_sub(expected_unknown);

    let unknown_outputs = if staged.jaxpr_unknown.outvars.is_empty() {
        Vec::new()
    } else {
        // Build inputs for the unknown jaxpr: residuals ++ dynamic_args.
        let mut unknown_inputs: Vec<Value> = staged.residuals.clone();
        unknown_inputs.extend(dynamic_args.iter().cloned());
        eval_jaxpr(&staged.jaxpr_unknown, &unknown_inputs).map_err(StagingError::UnknownEval)?
    };

    if staged.known_outputs.len() != expected_known || unknown_outputs.len() != expected_unknown {
        return Err(StagingError::OutputReconstruction {
            expected_known,
            actual_known: staged.known_outputs.len(),
            expected_unknown,
            actual_unknown: unknown_outputs.len(),
        });
    }

    let mut combined: Vec<Value> = Vec::with_capacity(staged.out_unknowns.len());
    let mut known_iter = staged.known_outputs.iter();
    let mut unknown_iter = unknown_outputs.iter();
    for is_unknown in &staged.out_unknowns {
        if *is_unknown {
            if let Some(v) = unknown_iter.next() {
                combined.push(v.clone());
            }
        } else if let Some(v) = known_iter.next() {
            combined.push(v.clone());
        }
    }

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{Atom, Equation, Jaxpr, Primitive, ProgramSpec, VarId};
    use serde::Serialize;
    use smallvec::smallvec;
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const PACKET_ID: &str = "FJ-P2C-003";
    const SUITE_ID: &str = "fj-interpreters";

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn test_log_path(test_id: &str) -> PathBuf {
        let file_name = test_id.replace("::", "__");
        repo_root()
            .join("artifacts")
            .join("testing")
            .join("logs")
            .join(SUITE_ID)
            .join(format!("{file_name}.json"))
    }

    fn replay_command(test_id: &str) -> String {
        format!("cargo test -p fj-interpreters --lib {test_id} -- --exact --nocapture")
    }

    fn duration_ms(start: Instant) -> u64 {
        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn write_log(path: &Path, log: &fj_test_utils::TestLogV1) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| format!("log dir create failed: {err}"))?;
        }
        let payload = serde_json::to_string_pretty(log)
            .map_err(|err| format!("log serialize failed: {err}"))?;
        fs::write(path, payload).map_err(|err| format!("log write failed: {err}"))
    }

    fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
        if let Some(msg) = payload.downcast_ref::<String>() {
            return msg.clone();
        }
        if let Some(msg) = payload.downcast_ref::<&str>() {
            return (*msg).to_owned();
        }
        "non-string panic payload".to_owned()
    }

    fn run_logged_test<Fixture, F>(
        test_name: &str,
        fixture: &Fixture,
        mode: fj_test_utils::TestMode,
        body: F,
    ) where
        Fixture: Serialize,
        F: FnOnce() -> Result<Vec<String>, String> + std::panic::UnwindSafe,
    {
        let overall_start = Instant::now();
        let setup_start = Instant::now();
        let fixture_id = fj_test_utils::fixture_id_from_json(fixture).expect("fixture digest");
        let test_id = fj_test_utils::test_id(module_path!(), test_name);
        let mut log = fj_test_utils::TestLogV1::unit(
            test_id.clone(),
            fixture_id,
            mode,
            fj_test_utils::TestResult::Fail,
        );
        log.phase_timings.setup_ms = duration_ms(setup_start);

        let execute_start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(body));
        log.phase_timings.execute_ms = duration_ms(execute_start);

        let verify_start = Instant::now();
        let mut panic_payload: Option<Box<dyn Any + Send>> = None;
        let mut failure_detail: Option<String> = None;

        match outcome {
            Ok(Ok(mut artifact_refs)) => {
                log.result = fj_test_utils::TestResult::Pass;
                artifact_refs.push(format!("packet:{PACKET_ID}"));
                artifact_refs.push(format!("replay: {}", replay_command(&test_id)));
                log.artifact_refs = artifact_refs;
                log.details = Some(format!(
                    "packet_id={PACKET_ID};suite_id={SUITE_ID};result=pass"
                ));
            }
            Ok(Err(detail)) => {
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
            }
            Err(payload) => {
                let detail = panic_payload_to_string(payload.as_ref());
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
                panic_payload = Some(payload);
            }
        }
        log.phase_timings.verify_ms = duration_ms(verify_start);

        let log_path = test_log_path(&test_id);
        log.artifact_refs.push(log_path.display().to_string());
        log.duration_ms = duration_ms(overall_start);

        let teardown_start = Instant::now();
        write_log(&log_path, &log).expect("test log write should succeed");
        log.phase_timings.teardown_ms = duration_ms(teardown_start);
        log.duration_ms = duration_ms(overall_start);
        write_log(&log_path, &log).expect("test log rewrite should succeed");

        if let Some(payload) = panic_payload {
            std::panic::resume_unwind(payload);
        }
        if let Some(detail) = failure_detail {
            std::panic::panic_any(detail);
        }
    }

    /// { a, b -> c = neg(a); d = mul(c, b) -> d }
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
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    /// { a, b -> c = neg(a); d = mul(c, b) -> a, d }
    fn make_mixed_outputs_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(1), VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    // ── make_jaxpr tests ────────────────────────────────────────────

    #[test]
    fn test_staging_make_jaxpr_all_specs() {
        run_logged_test(
            "test_staging_make_jaxpr_all_specs",
            &("staging", "make_jaxpr", "all_specs"),
            fj_test_utils::TestMode::Strict,
            || {
                let specs = [
                    ProgramSpec::Add2,
                    ProgramSpec::Square,
                    ProgramSpec::AddOne,
                    ProgramSpec::SinX,
                    ProgramSpec::CosX,
                    ProgramSpec::SquarePlusLinear,
                    ProgramSpec::Dot3,
                    ProgramSpec::ReduceSumVec,
                ];
                for spec in specs {
                    let jaxpr = make_jaxpr(spec);
                    assert!(
                        !jaxpr.invars.is_empty() || !jaxpr.equations.is_empty(),
                        "{spec:?} produced empty jaxpr"
                    );
                }
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_make_jaxpr_well_formed() {
        run_logged_test(
            "test_staging_make_jaxpr_well_formed",
            &("staging", "make_jaxpr", "well_formed"),
            fj_test_utils::TestMode::Strict,
            || {
                let specs = [ProgramSpec::Add2, ProgramSpec::Square, ProgramSpec::AddOne];
                for spec in specs {
                    let jaxpr = make_jaxpr(spec);
                    jaxpr
                        .validate_well_formed()
                        .map_err(|e| format!("{spec:?} failed well-formedness: {e:?}"))?;
                }
                Ok(vec![])
            },
        );
    }

    // ── stage_jaxpr tests ───────────────────────────────────────────

    #[test]
    fn test_staging_all_known_captures_known_outputs() {
        run_logged_test(
            "test_staging_all_known_captures_known_outputs",
            &("staging", "all_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let staged = stage_jaxpr(
                    &jaxpr,
                    &[false, false],
                    &[Value::scalar_i64(3), Value::scalar_i64(4)],
                )
                .unwrap();

                assert!(staged.jaxpr_unknown.equations.is_empty());
                assert_eq!(staged.known_outputs, vec![Value::scalar_i64(7)]);
                assert!(staged.residuals.is_empty());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_all_unknown_no_residuals() {
        run_logged_test(
            "test_staging_all_unknown_no_residuals",
            &("staging", "all_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let staged = stage_jaxpr(&jaxpr, &[true, true], &[]).unwrap();

                assert!(!staged.jaxpr_unknown.equations.is_empty());
                assert!(staged.residuals.is_empty());
                assert!(staged.known_outputs.is_empty());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_mixed_neg_mul() {
        run_logged_test(
            "test_staging_mixed_neg_mul",
            &("staging", "mixed", "neg_mul"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_neg_mul_jaxpr();
                let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)]).unwrap();
                assert_eq!(staged.jaxpr_known.equations.len(), 1);
                assert_eq!(staged.jaxpr_unknown.equations.len(), 1);
                assert!(staged.known_outputs.is_empty());
                assert!(!staged.residuals.is_empty());
                Ok(vec![])
            },
        );
    }

    // ── execute_staged tests ────────────────────────────────────────

    #[test]
    fn test_staging_execute_roundtrip() {
        run_logged_test(
            "test_staging_execute_roundtrip",
            &("staging", "execute", "roundtrip"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_neg_mul_jaxpr();
                let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)]).unwrap();
                let result = execute_staged(&staged, &[Value::scalar_i64(3)]).unwrap();

                let full =
                    eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(3)]).unwrap();
                assert_eq!(result, full);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_execute_all_known_returns_residuals() {
        run_logged_test(
            "test_staging_execute_all_known_returns_residuals",
            &("staging", "execute", "all_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let staged = stage_jaxpr(
                    &jaxpr,
                    &[false, false],
                    &[Value::scalar_i64(10), Value::scalar_i64(20)],
                )
                .unwrap();
                let result = execute_staged(&staged, &[]).unwrap();
                assert_eq!(result, vec![Value::scalar_i64(30)]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_execute_add2_various_values() {
        run_logged_test(
            "test_staging_execute_add2_various_values",
            &("staging", "execute", "add2_values"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let cases: Vec<(i64, i64, i64)> =
                    vec![(0, 0, 0), (1, 2, 3), (-5, 5, 0), (100, -100, 0)];
                for (a, b, expected) in cases {
                    let staged = stage_jaxpr(
                        &jaxpr,
                        &[false, false],
                        &[Value::scalar_i64(a), Value::scalar_i64(b)],
                    )
                    .unwrap();
                    let result = execute_staged(&staged, &[]).unwrap();
                    assert_eq!(
                        result,
                        vec![Value::scalar_i64(expected)],
                        "add2({a}, {b}) failed"
                    );
                }
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_execute_reconstructs_mixed_known_and_unknown_outputs() {
        run_logged_test(
            "test_staging_execute_reconstructs_mixed_known_and_unknown_outputs",
            &("staging", "execute", "mixed_known_unknown_outputs"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_mixed_outputs_jaxpr();
                let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)]).unwrap();
                assert_eq!(staged.known_outputs, vec![Value::scalar_i64(5)]);
                assert_eq!(staged.residuals.len(), 1);

                let result = execute_staged(&staged, &[Value::scalar_i64(3)]).unwrap();
                let full =
                    eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(3)]).unwrap();
                assert_eq!(result, full);
                Ok(vec![])
            },
        );
    }

    // ── StagingError tests ──────────────────────────────────────────

    #[test]
    fn test_staging_error_display_variants() {
        run_logged_test(
            "test_staging_error_display_variants",
            &("staging", "errors", "display"),
            fj_test_utils::TestMode::Strict,
            || {
                let e1 = StagingError::PartialEval(
                    crate::partial_eval::PartialEvalError::InputMaskMismatch {
                        expected: 2,
                        actual: 1,
                    },
                );
                let msg = format!("{e1}");
                assert!(msg.contains("partial eval"));

                let e2 = StagingError::KnownEval(InterpreterError::InputArity {
                    expected: 1,
                    actual: 0,
                });
                let msg2 = format!("{e2}");
                assert!(msg2.contains("known eval"));

                let e3 = StagingError::UnknownEval(InterpreterError::InputArity {
                    expected: 1,
                    actual: 0,
                });
                let msg3 = format!("{e3}");
                assert!(msg3.contains("unknown eval"));

                let e4 = StagingError::OutputPartition {
                    expected_known: 2,
                    actual_outputs: 1,
                };
                let msg4 = format!("{e4}");
                assert!(msg4.contains("partition"));

                let e5 = StagingError::OutputReconstruction {
                    expected_known: 1,
                    actual_known: 0,
                    expected_unknown: 1,
                    actual_unknown: 2,
                };
                let msg5 = format!("{e5}");
                assert!(msg5.contains("reconstruction"));

                let _: &dyn std::error::Error = &e1;
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_error_mask_mismatch_propagated() {
        run_logged_test(
            "test_staging_error_mask_mismatch_propagated",
            &("staging", "errors", "mask_mismatch"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let err = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(1)]).unwrap_err();
                assert!(matches!(err, StagingError::PartialEval(_)));
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_staging_rejects_constvar_jaxpr() {
        run_logged_test(
            "test_staging_rejects_constvar_jaxpr",
            &("staging", "constvars", "unsupported"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(vec![VarId(1)], vec![VarId(2)], vec![VarId(1)], vec![]);
                let err = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(1)]).unwrap_err();
                assert!(matches!(
                    err,
                    StagingError::PartialEval(PartialEvalError::UnsupportedConstvars { count: 1 })
                ));
                Ok(vec![])
            },
        );
    }

    // ── Schema contract ─────────────────────────────────────────────

    #[test]
    fn test_staging_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("staging", "schema")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_staging_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Property tests ──────────────────────────────────────────────

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]

            /// stage + execute == full eval for neg*mul with arbitrary i64 inputs.
            #[test]
            fn prop_staging_roundtrip_neg_mul(
                a in -100_i64..100,
                b in -100_i64..100,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_neg_mul_jaxpr();
                let va = Value::scalar_i64(a);
                let vb = Value::scalar_i64(b);

                let full = eval_jaxpr(&jaxpr, &[va.clone(), vb.clone()]).unwrap();

                let staged = stage_jaxpr(&jaxpr, &[false, true], &[va]).unwrap();
                let result = execute_staged(&staged, &[vb]).unwrap();

                prop_assert_eq!(full, result);
            }

            /// stage with all known + execute == full eval for Add2.
            #[test]
            fn prop_staging_all_known_add2(
                a in -1000_i64..1000,
                b in -1000_i64..1000,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_jaxpr(ProgramSpec::Add2);
                let va = Value::scalar_i64(a);
                let vb = Value::scalar_i64(b);

                let full = eval_jaxpr(&jaxpr, &[va.clone(), vb.clone()]).unwrap();

                let staged = stage_jaxpr(&jaxpr, &[false, false], &[va, vb]).unwrap();
                let result = execute_staged(&staged, &[]).unwrap();

                prop_assert_eq!(full, result);
            }

            /// make_jaxpr always produces well-formed Jaxprs.
            #[test]
            fn prop_make_jaxpr_well_formed(
                spec_idx in 0_usize..5,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let spec = match spec_idx {
                    0 => ProgramSpec::Add2,
                    1 => ProgramSpec::Square,
                    2 => ProgramSpec::AddOne,
                    3 => ProgramSpec::SinX,
                    _ => ProgramSpec::CosX,
                };
                let jaxpr = make_jaxpr(spec);
                prop_assert!(jaxpr.validate_well_formed().is_ok());
            }
        }
    }
}
