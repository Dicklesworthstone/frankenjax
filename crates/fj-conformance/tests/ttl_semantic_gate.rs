#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use fj_conformance::ttl_semantic::{
    TTL_SEMANTIC_BEAD_ID, TTL_SEMANTIC_REPORT_SCHEMA_VERSION, TtlSemanticReport,
    build_ttl_semantic_report, ttl_semantic_markdown, ttl_semantic_summary_json,
    validate_ttl_semantic_report, write_ttl_semantic_outputs,
};
use fj_core::{ProgramSpec, build_program};
use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn ttl_semantic_matrix_has_required_rows() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_ttl_semantic_report(temp.path());
    assert_eq!(report.schema_version, TTL_SEMANTIC_REPORT_SCHEMA_VERSION);
    assert_eq!(report.bead_id, TTL_SEMANTIC_BEAD_ID);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);

    let case_ids = report
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "valid_single_jit_square",
        "valid_single_grad_square",
        "valid_single_vmap_square",
        "valid_jit_grad_fixture",
        "valid_grad_jit_order_sensitive",
        "valid_vmap_grad_fixture",
        "fail_closed_grad_vmap_vector_output",
        "invalid_duplicate_evidence",
        "invalid_missing_evidence",
        "invalid_stale_input_fingerprint",
        "invalid_wrong_transform_binding",
        "invalid_missing_fixture_link",
    ] {
        assert!(
            case_ids.contains(required),
            "missing required row {required}"
        );
    }
}

#[test]
fn ttl_semantic_accepts_valid_structural_replays() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_ttl_semantic_report(temp.path());
    for case_id in [
        "valid_single_jit_square",
        "valid_single_grad_square",
        "valid_single_vmap_square",
        "valid_jit_grad_fixture",
        "valid_grad_jit_order_sensitive",
        "valid_vmap_grad_fixture",
    ] {
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == case_id)
            .expect("case should exist");
        assert_eq!(case.verifier_decision, "accept", "case: {case:#?}");
        assert_eq!(case.expected_decision, "accept", "case: {case:#?}");
        assert!(case.rejection_reason.is_none(), "case: {case:#?}");
        assert!(
            case.structural_checks
                .iter()
                .any(|check| check == "shape_dtype_matched"),
            "case: {case:#?}"
        );
        assert_eq!(
            case.output_fingerprint, case.expected_output_fingerprint,
            "case: {case:#?}"
        );
    }
}

#[test]
fn ttl_semantic_rejects_invalid_and_fail_closed_rows() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_ttl_semantic_report(temp.path());
    for (case_id, expected_reason) in [
        (
            "fail_closed_grad_vmap_vector_output",
            "transform_execution.non_scalar_gradient_input",
        ),
        (
            "invalid_duplicate_evidence",
            "transform_invariant.duplicate_evidence",
        ),
        (
            "invalid_missing_evidence",
            "transform_invariant.missing_evidence",
        ),
        (
            "invalid_stale_input_fingerprint",
            "semantic.stale_input_fingerprint",
        ),
        (
            "invalid_wrong_transform_binding",
            "transform_invariant.wrong_transform_binding",
        ),
        (
            "invalid_missing_fixture_link",
            "semantic.missing_oracle_fixture_link",
        ),
    ] {
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == case_id)
            .expect("case should exist");
        assert_eq!(case.status, "pass", "case: {case:#?}");
        assert_eq!(case.verifier_decision, "reject", "case: {case:#?}");
        assert_eq!(
            case.rejection_reason.as_deref(),
            Some(expected_reason),
            "case: {case:#?}"
        );
    }
}

#[test]
fn ttl_semantic_hash_is_transform_order_sensitive() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_ttl_semantic_report(temp.path());
    let jit_grad = report
        .cases
        .iter()
        .find(|case| case.case_id == "valid_jit_grad_fixture")
        .expect("jit grad case");
    let grad_jit = report
        .cases
        .iter()
        .find(|case| case.case_id == "valid_grad_jit_order_sensitive")
        .expect("grad jit case");
    assert_ne!(jit_grad.transform_stack, grad_jit.transform_stack);
    assert_ne!(jit_grad.stack_signature, grad_jit.stack_signature);
    assert_ne!(jit_grad.stack_hash_hex, grad_jit.stack_hash_hex);
    assert_eq!(jit_grad.output_fingerprint, grad_jit.output_fingerprint);
}

#[test]
fn ttl_semantic_validation_catches_accepted_stale_fingerprint() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut report = build_ttl_semantic_report(temp.path());
    let case = report
        .cases
        .iter_mut()
        .find(|case| case.case_id == "valid_single_jit_square")
        .expect("case should exist");
    case.expected_input_fingerprint = build_program(ProgramSpec::Add2)
        .canonical_fingerprint()
        .to_owned();
    let issues = validate_ttl_semantic_report(&report);
    assert!(
        issues
            .iter()
            .any(|issue| issue.code == "accepted_stale_input_fingerprint"),
        "issues: {issues:#?}"
    );
}

#[test]
fn ttl_semantic_validation_requires_fixture_links_when_declared() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut report = build_ttl_semantic_report(temp.path());
    let case = report
        .cases
        .iter_mut()
        .find(|case| case.case_id == "valid_jit_grad_fixture")
        .expect("case should exist");
    case.oracle_fixture_id = None;
    let issues = validate_ttl_semantic_report(&report);
    assert!(
        issues
            .iter()
            .any(|issue| issue.code == "accepted_missing_fixture_link"),
        "issues: {issues:#?}"
    );
}

#[test]
fn ttl_semantic_outputs_round_trip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report_path = temp.path().join("ttl_semantic_proof_matrix.v1.json");
    let markdown_path = temp.path().join("ttl_semantic_proof_matrix.v1.md");
    let report = write_ttl_semantic_outputs(temp.path(), &report_path, &markdown_path)
        .expect("outputs should write");

    let raw = std::fs::read_to_string(&report_path).expect("report should be readable");
    let parsed: TtlSemanticReport = serde_json::from_str(&raw).expect("report JSON should parse");
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.cases.len(), report.cases.len());
    assert!(validate_ttl_semantic_report(&parsed).is_empty());

    let markdown = std::fs::read_to_string(&markdown_path).expect("markdown should be readable");
    assert!(markdown.contains("TTL Semantic Proof Matrix Gate"));
    assert!(markdown.contains("No TTL semantic proof matrix issues found."));
}

#[test]
fn ttl_semantic_summary_is_dashboard_ready() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_ttl_semantic_report(temp.path());
    let summary = ttl_semantic_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["case_count"], report.cases.len());
    assert_eq!(summary["issue_count"], 0);
    assert_eq!(
        summary["coverage"]["accepted_count"],
        report.coverage.accepted_count
    );

    let markdown = ttl_semantic_markdown(&report);
    assert!(markdown.contains("valid_jit_grad_fixture"));
    assert!(markdown.contains("invalid_missing_fixture_link"));
}

#[test]
fn ttl_semantic_e2e_log_schema_accepts_gate_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let log_path = temp.path().join("ttl_semantic.e2e.json");
    let mut log = E2EForensicLogV1::new(
        TTL_SEMANTIC_BEAD_ID,
        "e2e_ttl_semantic_gate",
        "ttl_semantic_gate_test",
        vec!["ttl_semantic_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["valid_jit_grad_fixture".to_owned()];
    log.oracle_ids = vec!["jit_grad_poly_x5.0".to_owned()];
    log.transform_stack = vec!["jit".to_owned(), "grad".to_owned()];
    log.replay_command = "./scripts/run_ttl_semantic_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log).expect("write e2e log");
    validate_e2e_log_path(&log_path, Path::new(".")).expect("e2e log should validate");
}
