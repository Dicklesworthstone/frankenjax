#![forbid(unsafe_code)]

use fj_conformance::decision_ledger::{
    DECISION_LEDGER_BEAD_ID, DECISION_LEDGER_REPORT_SCHEMA_VERSION, build_decision_ledger_report,
    decision_ledger_markdown, validate_decision_ledger_report, write_decision_ledger_outputs,
};
use fj_conformance::e2e_log::{E2ELogStatus, validate_e2e_log_path};
use std::collections::BTreeSet;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn decision_ledger_report_covers_required_classes() {
    let root = repo_root();
    let report = build_decision_ledger_report(&root);
    assert_eq!(report.schema_version, DECISION_LEDGER_REPORT_SCHEMA_VERSION);
    assert_eq!(report.bead_id, DECISION_LEDGER_BEAD_ID);
    assert_eq!(report.status, "pass", "issues: {:#?}", report.issues);
    assert!(report.issues.is_empty());
    assert_eq!(report.summary.required_decision_class_count, 9);
    assert_eq!(report.summary.observed_decision_class_count, 9);
    assert_eq!(report.summary.stale_calibration_count, 0);

    let classes = report
        .rows
        .iter()
        .map(|row| row.decision_class.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "cache_hit_recompute",
        "strict_rejection",
        "hardened_recovery",
        "fallback_denial",
        "optimization_selection",
        "durability_recovery",
        "transform_admission",
        "unsupported_scope",
        "runtime_budget_deadline",
    ] {
        assert!(classes.contains(required), "missing {required}");
    }
}

#[test]
fn decision_ledger_validation_rejects_bad_rows() {
    let root = repo_root();
    let mut report = build_decision_ledger_report(&root);
    report.rows[0].decision_id = report.rows[1].decision_id.clone();
    report.rows[0].alternatives_considered = vec!["kill".to_owned()];
    report.rows[0].confidence = 1.5;
    report.rows[0].evidence_signals.clear();
    report.rows[0].calibration_bucket = "missing_bucket".to_owned();

    let issue_codes = validate_decision_ledger_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("duplicate_decision_id"));
    assert!(issue_codes.contains("missing_alternatives"));
    assert!(issue_codes.contains("bad_probability"));
    assert!(issue_codes.contains("missing_evidence_signals"));
    assert!(issue_codes.contains("missing_calibration_bucket"));
}

#[test]
fn decision_ledger_validation_rejects_bad_calibration_and_redaction() {
    let root = repo_root();
    let mut report = build_decision_ledger_report(&root);
    report.calibration_buckets[1].lower_bound = 0.10;
    report.calibration_buckets[1].count = 1;
    report.calibration_buckets[1].stale = false;
    report.calibration_buckets[1].observed_probability = 0.0;
    report.rows[0].evidence_signals[0].detail = "contains SECRET material".to_owned();

    let issue_codes = validate_decision_ledger_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("non_monotonic_calibration_bucket"));
    assert!(issue_codes.contains("stale_calibration_not_marked"));
    assert!(issue_codes.contains("bad_calibration_drift_status"));
    assert!(issue_codes.contains("redaction_policy_violation"));
}

#[test]
fn decision_ledger_validation_rejects_loss_matrix_drift() {
    let root = repo_root();
    let mut report = build_decision_ledger_report(&root);
    report.rows[0].expected_loss_keep += 1.0;
    report.rows[1].selected_action = "keep".to_owned();

    let issue_codes = validate_decision_ledger_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("loss_matrix_mismatch"));
    assert!(issue_codes.contains("selected_action_mismatch"));
}

#[test]
fn decision_ledger_outputs_write_and_validate_e2e_log() -> Result<(), String> {
    let root = repo_root();
    let temp = tempfile::tempdir().map_err(|err| err.to_string())?;
    let report_path = temp.path().join("decision_ledger_calibration.v1.json");
    let markdown_path = temp.path().join("decision_ledger_calibration.v1.md");
    let e2e_path = temp.path().join("e2e_decision_ledger_gate.e2e.json");
    let report = write_decision_ledger_outputs(&root, &report_path, &markdown_path)
        .map_err(|err| err.to_string())?;
    assert_eq!(report.status, "pass");
    assert!(report_path.exists());
    assert!(markdown_path.exists());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fj_decision_ledger_gate"))
        .arg("--root")
        .arg(&root)
        .arg("--report")
        .arg(&report_path)
        .arg("--markdown")
        .arg(&markdown_path)
        .arg("--e2e")
        .arg(&e2e_path)
        .arg("--enforce")
        .output()
        .map_err(|err| err.to_string())?;
    assert!(
        output.status.success(),
        "gate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed =
        validate_e2e_log_path(&e2e_path, temp.path()).map_err(|issues| format!("{issues:#?}"))?;
    assert_eq!(parsed.status, E2ELogStatus::Pass);
    assert_eq!(parsed.bead_id, DECISION_LEDGER_BEAD_ID);
    Ok(())
}

#[test]
fn decision_ledger_markdown_is_dashboard_ready() {
    let root = repo_root();
    let report = build_decision_ledger_report(&root);
    let markdown = decision_ledger_markdown(&report);
    assert!(markdown.contains("Decision Ledger Calibration"));
    assert!(markdown.contains("cache_hit_recompute"));
    assert!(markdown.contains("No decision-ledger calibration issues found."));
}
