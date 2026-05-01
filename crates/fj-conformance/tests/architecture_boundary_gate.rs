#![forbid(unsafe_code)]

use fj_conformance::architecture_decision::{
    ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION, ArchitectureBoundaryIssue, BoundaryDecisionKind,
    WorkspaceEdge, architecture_boundary_markdown, architecture_boundary_summary_json,
    build_architecture_boundary_report, capture_workspace_snapshot,
    validate_architecture_boundary_report, write_architecture_boundary_outputs,
};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn current_report() -> fj_conformance::architecture_decision::ArchitectureBoundaryReport {
    let snapshot = capture_workspace_snapshot(&repo_root()).expect("cargo metadata should parse");
    build_architecture_boundary_report(snapshot)
}

fn issue_codes(issues: &[ArchitectureBoundaryIssue]) -> Vec<&str> {
    issues.iter().map(|issue| issue.code.as_str()).collect()
}

#[test]
fn current_workspace_architecture_boundary_report_passes() {
    let report = current_report();
    assert_eq!(
        report.schema_version,
        ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report.bead_id, "frankenjax-cstq.12");
    assert_eq!(report.status, "pass");
    assert_eq!(report.crate_count, 15);
    assert!(report.issues.is_empty());
    assert!(
        report
            .workspace_crates
            .iter()
            .any(|krate| krate.name == "fj-api")
    );
    assert!(
        report
            .workspace_crates
            .iter()
            .any(|krate| krate.name == "fj-backend-cpu")
    );
    assert!(
        report
            .workspace_crates
            .iter()
            .any(|krate| krate.name == "fj-ffi")
    );
    assert!(
        report
            .decisions
            .iter()
            .any(|decision| decision.boundary_id == "user_api_facade"
                && decision.decision == BoundaryDecisionKind::KeepCurrentBoundary)
    );
    assert!(report.decisions.iter().any(|decision| {
        decision.boundary_id == "transform_stack"
            && decision.decision == BoundaryDecisionKind::DeferExtraction
            && decision
                .follow_up_beads
                .contains(&"frankenjax-cstq.2".to_owned())
    }));
}

#[test]
fn missing_required_facade_crate_is_rejected() {
    let mut report = current_report();
    report
        .workspace_crates
        .retain(|krate| krate.name != "fj-api");
    report.crate_count = report.workspace_crates.len();
    let issues = validate_architecture_boundary_report(&report);
    let codes = issue_codes(&issues);
    assert!(codes.contains(&"missing_required_crate"));
    assert!(codes.contains(&"owner_crate_missing"));
}

#[test]
fn production_dependency_on_conformance_is_rejected() {
    let mut report = current_report();
    report.normal_edges.push(WorkspaceEdge {
        from: "fj-api".to_owned(),
        to: "fj-conformance".to_owned(),
    });
    let issues = validate_architecture_boundary_report(&report);
    assert!(issue_codes(&issues).contains(&"production_depends_on_conformance"));
}

#[test]
fn malformed_decision_records_are_rejected() {
    let mut report = current_report();
    let mut duplicate = report.decisions[0].clone();
    duplicate.rejected_options.clear();
    duplicate.guardrails.clear();
    duplicate.decision = BoundaryDecisionKind::ExtractNow;
    duplicate.follow_up_beads.clear();
    report.decisions.push(duplicate);

    let issues = validate_architecture_boundary_report(&report);
    let codes = issue_codes(&issues);
    assert!(codes.contains(&"duplicate_boundary_id"));
    assert!(codes.contains(&"missing_rejected_options"));
    assert!(codes.contains(&"missing_guardrails"));
    assert!(codes.contains(&"extract_now_without_beads"));
}

#[test]
fn generated_markdown_and_summary_are_dashboard_ready() {
    let report = current_report();
    let markdown = architecture_boundary_markdown(&report);
    assert!(markdown.contains("Architecture Boundary Decision"));
    assert!(markdown.contains("user_api_facade"));
    assert!(markdown.contains("lowering_execution"));

    let summary = architecture_boundary_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["crate_count"], 15);
    assert_eq!(summary["issue_count"], 0);
    assert!(
        summary["decisions"]
            .as_array()
            .expect("summary decisions should be array")
            .iter()
            .any(|row| row["boundary_id"] == "conformance_harness")
    );
}

#[test]
fn output_files_round_trip_as_json_and_markdown() {
    let tmp = tempfile::tempdir().expect("tempdir should create");
    let report_path = tmp.path().join("architecture_boundary_decision.v1.json");
    let markdown_path = tmp.path().join("architecture_boundary_decision.v1.md");
    let report = write_architecture_boundary_outputs(&repo_root(), &report_path, &markdown_path)
        .expect("outputs should write");

    let raw = std::fs::read_to_string(&report_path).expect("report should be readable");
    let parsed: fj_conformance::architecture_decision::ArchitectureBoundaryReport =
        serde_json::from_str(&raw).expect("report JSON should parse");
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.decisions.len(), report.decisions.len());

    let markdown = std::fs::read_to_string(&markdown_path).expect("markdown should be readable");
    assert!(markdown.contains("No architecture boundary issues found."));
}
