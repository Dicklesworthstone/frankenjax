#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use fj_conformance::error_taxonomy::{
    ERROR_TAXONOMY_BEAD_ID, ERROR_TAXONOMY_REPORT_SCHEMA_VERSION, ErrorTaxonomyReport,
    build_error_taxonomy_report, error_taxonomy_markdown, error_taxonomy_summary_json,
    validate_error_taxonomy_report, write_error_taxonomy_outputs,
};
use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn error_taxonomy_matrix_has_required_rows() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_error_taxonomy_report(temp.path());
    assert_eq!(report.schema_version, ERROR_TAXONOMY_REPORT_SCHEMA_VERSION);
    assert_eq!(report.bead_id, ERROR_TAXONOMY_BEAD_ID);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);

    let case_ids = report
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "ir_validation_unknown_outvar",
        "transform_proof_duplicate_evidence",
        "transform_proof_missing_evidence",
        "primitive_arity_add",
        "primitive_shape_add_broadcast",
        "primitive_type_sin_bool",
        "interpreter_missing_variable",
        "cache_strict_unknown_feature",
        "cache_hardened_unknown_feature",
        "vmap_axis_mismatch",
        "durability_missing_artifact",
        "unsupported_transform_tail_without_fallback",
        "unsupported_control_flow_grad_vmap_vector",
    ] {
        assert!(
            case_ids.contains(required),
            "missing required case {required}"
        );
    }
}

#[test]
fn error_taxonomy_rows_are_typed_replayable_and_panic_free() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_error_taxonomy_report(temp.path());
    assert_eq!(report.coverage.observed_case_count, report.cases.len());
    assert_eq!(report.coverage.pass_count, report.cases.len());
    assert_eq!(report.coverage.panic_free_count, report.cases.len());
    assert!(
        report.coverage.typed_error_count >= 12,
        "expected all malformed rows except hardened cache success to be typed: {report:#?}"
    );

    for case in &report.cases {
        assert_eq!(case.status, "pass", "case: {case:#?}");
        assert_eq!(case.expected_error_class, case.actual_error_class);
        assert_eq!(case.panic_status, "no_panic");
        assert!(!case.boundary.trim().is_empty());
        assert!(!case.mode.trim().is_empty());
        assert!(!case.input_class.trim().is_empty());
        assert!(!case.error_variant.trim().is_empty());
        assert!(!case.message_shape.trim().is_empty());
        assert!(!case.replay_hint.trim().is_empty());
        assert!(!case.replay_command.trim().is_empty());
        assert!(!case.evidence_refs.is_empty());
    }
}

#[test]
fn error_taxonomy_strict_hardened_divergence_is_allowlisted() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_error_taxonomy_report(temp.path());
    let divergent = report
        .cases
        .iter()
        .filter(|case| case.strict_hardened_divergence)
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        divergent,
        BTreeSet::from([
            "cache_hardened_unknown_feature",
            "cache_strict_unknown_feature"
        ])
    );
    for case in report
        .cases
        .iter()
        .filter(|case| case.strict_hardened_divergence)
    {
        assert!(case.divergence_allowed, "case: {case:#?}");
        assert!(case.strict_behavior.contains("Strict") || case.strict_behavior.contains("strict"));
        assert!(
            case.hardened_behavior.contains("Hardened")
                || case.hardened_behavior.contains("hardened")
        );
    }
}

#[test]
fn error_taxonomy_outputs_round_trip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report_path = temp.path().join("error_taxonomy_matrix.v1.json");
    let markdown_path = temp.path().join("error_taxonomy_matrix.v1.md");
    let report = write_error_taxonomy_outputs(temp.path(), &report_path, &markdown_path)
        .expect("outputs should write");

    let raw = std::fs::read_to_string(&report_path).expect("report should be readable");
    let parsed: ErrorTaxonomyReport = serde_json::from_str(&raw).expect("report JSON should parse");
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.cases.len(), report.cases.len());
    assert!(validate_error_taxonomy_report(&parsed).is_empty());

    let markdown = std::fs::read_to_string(&markdown_path).expect("markdown should be readable");
    assert!(markdown.contains("Error Taxonomy Matrix Gate"));
    assert!(markdown.contains("No error taxonomy matrix issues found."));
}

#[test]
fn error_taxonomy_summary_is_dashboard_ready() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_error_taxonomy_report(temp.path());
    let summary = error_taxonomy_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["case_count"], report.cases.len());
    assert_eq!(summary["issue_count"], 0);
    assert_eq!(
        summary["coverage"]["typed_error_count"],
        report.coverage.typed_error_count
    );

    let markdown = error_taxonomy_markdown(&report);
    assert!(markdown.contains("cache_strict_unknown_feature"));
    assert!(markdown.contains("unsupported_control_flow_grad_vmap_vector"));
}

#[test]
fn error_taxonomy_e2e_log_schema_accepts_gate_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let log_path = temp.path().join("error_taxonomy.e2e.json");
    let mut log = E2EForensicLogV1::new(
        ERROR_TAXONOMY_BEAD_ID,
        "e2e_error_taxonomy_gate",
        "error_taxonomy_gate_test",
        vec!["error_taxonomy_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["primitive_arity_add".to_owned()];
    log.oracle_ids = vec!["local:error-taxonomy-matrix.v1".to_owned()];
    log.transform_stack = vec!["jit".to_owned(), "grad".to_owned(), "vmap".to_owned()];
    log.replay_command = "./scripts/run_error_taxonomy_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log).expect("write e2e log");
    validate_e2e_log_path(&log_path, Path::new(".")).expect("e2e log should validate");
}
